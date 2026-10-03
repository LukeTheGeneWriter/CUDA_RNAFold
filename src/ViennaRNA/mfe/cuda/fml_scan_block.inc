/* fml_scan_block.inc -- the fML row scan, as a block-level device function.
 *
 * Not a per-cell lift. This is a block-cooperative Hillis-Steele scan along j
 * (affine min-plus maps composed left to right), launched one block per record
 * and carrying state across tiles in shared memory. The fused megakernel runs
 * it on one designated block per record, which is the same shape the standalone
 * kernel has today.
 *
 * H and the two shared tiles are parameters: the megakernel picks the record
 * and owns its shared memory. __syncthreads() inside a function that only some
 * blocks call is still correct -- it synchronises the CALLING block, and every
 * thread of that block enters together.
 */

template<int TW>
__device__ inline void
fml_scan_block(const int nfiles, const int i_row, const int turn,
                const int*  __restrict__ new_e,      //in  d_new_e      == host new_C
                const int*  __restrict__ e3p00,      //in  d_energy_3p00_row
                const int*  __restrict__ gq_row,     //in  d_gq_row, or NULL when -g is off
                const int*  __restrict__ fml_prev,   //in  d_fml_prev   (previous row)
                const char* __restrict__ up_ml_ok,   //in  d_up_ml_ok
                const cuda_param2_t* __restrict__ P, //in  d_param2 (MLbase)
                      int*  __restrict__ energy_min, //out d_energy_min
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ seq_off_H,
                const size_t* __restrict__ size_off_H,
                const int* __restrict__ i_H,
               const int H, int* sa, int* sc) {
  // PHASE A (continuous flow): the row index is PER-RECORD now. It equals
  // i_row today, and the assert proves that at RUNTIME rather than by
  // argument -- .cu files never see -DNDEBUG, so a table that ever
  // disagrees traps instead of folding silently wrong.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const long long width = (long long)size_off_H[H+1] - (long long)size_off_H[H];
  if(width <= 0) return;                  // has not joined the sweep -- whole block returns
  const size_t o  = row_off_H[H];
  const size_t so = seq_off_H[H];
  const int j0     = i + turn + 1 + (int)size_off_H[nfiles + 1];   /* + the slot's column shift (3a.2) */
  const int MLbase = P->MLbase;
  const int en_i   = up_ml_ok[so + (size_t)i] ? MLbase : INF;

  // sa/sc are supplied by the caller: the megakernel owns its shared memory.

  /* E[j0-1]. A whole row starts at the diagonal, where it is INF (the host's `j == i+turn+1`
   * special case). A column block (blocked md 3a.2: a nonzero shift) continues the row
   * from the previous block, whose last E is already in energy_min -- the scan is an
   * affine (min,+) composition, so a carry-in is exact. */
  int carry = (size_off_H[nfiles + 1] > 0) ? energy_min[o + j0 - 1] : INF;
  const int t = threadIdx.x;

  for(long long base = 0; base < width; base += TW) {
    const long long k = base + t;
    int a = INF, c = 0;                   // identity, for lanes past the end
    if(k < width) {
      const int j = j0 + (int)k;
      // BOTH OPERANDS, BOTH TERMS. These two lines each guarded exactly one
      // side of a sum whose other side can be INF, and INF is a SENTINEL, not a
      // number: INF plus a real negative energy lands just BELOW INF, so it is
      // no longer recognisable as "no such decomposition" while still being far
      // too large to win any min against a real energy. The value therefore
      // survives, invisibly, in fML.
      //
      // Upstream guards the operand this code did not: extend_fm_3p()
      // (mfe/mfe_multibranch.c:949-950) reads `en = c[ij]` and tests
      // `if (en != INF)` BEFORE adding the stem energy. The e3p00 test here
      // stands in for its enclosing evaluate(..., VRNA_DECOMP_ML_STEM) instead,
      // so the c[ij] side was never tested at all. The second line's own
      // comment already said "hazard 1: en_i NOT guarded" -- it was a known
      // hazard that had not bitten.
      //
      // HOW IT SURFACED: --noClosingGU + RNA_FML_INT16, 2026-09-11. GU/UG pairs
      // keep MB_LOOP_ENC but lose HP_LOOP and MB_LOOP, so c[i][j] is INF for
      // many more (i,j) that still reach this sum -- and fML then carried
      // 9999890 (INF minus an E_MLstem of 110). pack_fml_kernel tests `v == INF`
      // to choose the FML_INF16 sentinel, 9999890 is not INF, so it became a
      // block BASELINE and the next real value in that block was 9999710 away
      // from it. __trap(), correctly.
      //
      // On the int32 path this was invisible and harmless -- a ~1e7 value never
      // wins a min against a real energy -- which is why it survived every
      // byte-identical run this project has. It is fixed rather than clamped
      // because the same fML feeds DMLi, and DMLi carrying near-INF sentinels is
      // PORT_INVESTIGATIONS.md item 3.
      //
      // fml_tadd() is the INF-safe add already defined above for exactly this.
      const int c_term = fml_tadd(new_e[o+j], e3p00[o+j]);
      const int e3     = fml_tadd(fml_prev[o+j], en_i);
      a = fml_tmin(c_term, e3);
      // G-quadruplex (G1). extend_fm_3p()'s gquad case is a SIBLING of its
      // c[ij] stem case at the same MIN2, not a modifier of it: fM[i][j] can be
      // one quadruplex spanning i..j instead of one base pair spanning i..j.
      //
      //     en = c_gq(i,j) + vrna_E_multibranch_stem(0,-1,-1,P) * n_seq
      //
      // and that stem energy is exactly P->MLintern[0]: E_MLstem() takes NONE
      // of its dangle branches when si1==sj1==-1 (eval/multibranch.h:171-176),
      // and 0 > 2 is false so no TerminalAU. n_seq == 1 for a single sequence.
      // Do NOT reach for E_MLstem_device() here -- it indexes
      // mismatchM[type][si1][sj1] unconditionally, so (0,-1,-1) reads out of
      // bounds. The host-side identity is what makes the constant safe.
      //
      // gq_row is NULL unless a c_gq was uploaded, so with -g off this is one
      // predicted null test per cell against a kernel-uniform pointer.
      if (gq_row) {
        const int g = gq_row[o+j];
        if (g != INF)
          a = fml_tmin(a, g + P->MLintern[0]);
      }
      c = up_ml_ok[so + (size_t)j] ? MLbase : INF;
    }
    sa[t] = a; sc[t] = c;
    __syncthreads();

    // Hillis-Steele inclusive scan over the composition above. Left operand is
    // the earlier segment (applied first), right the later one.
    for(int d = 1; d < TW; d <<= 1) {
      int na = sa[t], nc = sc[t];
      if(t >= d) {
        na = fml_tmin(sa[t], fml_tadd(sa[t-d], sc[t]));
        nc = fml_tadd(sc[t-d], sc[t]);
      }
      __syncthreads();
      sa[t] = na; sc[t] = nc;
      __syncthreads();
    }

    if(k < width) {
      const int j = j0 + (int)k;
      energy_min[o+j] = fml_tmin(sa[t], fml_tadd(carry, sc[t]));
    }
    // Carry E across the tile boundary. Every thread reads the same lane, so
    // the new carry is block-uniform without a broadcast.
    const long long rem  = width - base;
    const int       last = (int)((rem < TW) ? (rem - 1) : (TW - 1));
    const int nextcarry  = fml_tmin(sa[last], fml_tadd(carry, sc[last]));
    __syncthreads();          // everyone has read sa/sc before the next tile overwrites them
    carry = nextcarry;
  }
}

/* ======================= RNA_FML_SCAN=2: K elements per thread =======================
 *
 * Why (measured 2026-09-30, RTX 3050, ncu): fml_scan_block above costs ~12.5 us per
 * 256-wide tile + ~10 us fixed -- 22.9 us at width 100, 148 us at width 2800 -- because
 * each tile is 8 Hillis-Steele steps x 2 __syncthreads() on ONE block per record. That is
 * barrier latency, not bandwidth, and it sits on the critical path between c(i) and md(i).
 *
 * This version gives each thread K CONSECUTIVE elements, so a tile covers TW*K (2048 at
 * 256x8) and a 5601-wide row is 3 tiles instead of 22, each with ~3 barriers:
 *
 *   1. stage (a,c) for the tile in shared, coalesced -- the element arithmetic is copied
 *      VERBATIM from fml_scan_block (same hazards, same guards);
 *   2. each thread composes its own K maps in registers, starting from its FIRST element
 *      rather than from the (INF,0) identity -- composing with the identity would apply
 *      min(a, INF) to a first element that can legitimately exceed INF (hazard 1);
 *   3. a warp-shuffle inclusive scan of those aggregates, then of the warp totals;
 *   4. each thread turns its exclusive prefix into the INCOMING VALUE E_in by APPLYING
 *      the maps to the tile's carry, and then RE-WALKS its K elements with the host's own
 *      recurrence  E = min(A[j], E (+) b[j]).  So every stored value is produced by the
 *      literal recurrence; the only regrouped quantity is E_in, which is the same class of
 *      regrouping every TW of the original already performs (and is verified for).
 *
 * The (a,c) composition is (a_L,c_L) then (a_R,c_R) = (min(a_R, a_L (+) c_R), c_L (+) c_R);
 * see the derivation above fml_scan_kernel in hp_mb_loop.cu.
 */
__device__ __forceinline__ void
fml_compose(int& a, int& c, const int aL, const int cL)   // (aL,cL) then (a,c), in place
{
  a = fml_tmin(a, fml_tadd(aL, c));
  c = fml_tadd(cL, c);
}

#define FML_PAD(x) ((x) + ((x) >> 5))   /* one int per 32: K=8 thread-serial reads hit 32 banks */

template<int TW, int K>
__device__ inline void
fml_scan_block2(const int nfiles, const int i_row, const int turn,
                const int*  __restrict__ new_e,
                const int*  __restrict__ e3p00,
                const int*  __restrict__ gq_row,
                const int*  __restrict__ fml_prev,
                const char* __restrict__ up_ml_ok,
                const cuda_param2_t* __restrict__ P,
                      int*  __restrict__ energy_min,
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ seq_off_H,
                const size_t* __restrict__ size_off_H,
                const int* __restrict__ i_H,
                const int H, int* sa, int* sc, int* wa, int* wc, int* sE)
{
  static_assert(TW % 32 == 0 && TW <= 1024, "TW must be a multiple of 32");
  constexpr int NW   = TW / 32;
  constexpr int TILE = TW * K;
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const long long width = (long long)size_off_H[H+1] - (long long)size_off_H[H];
  if(width <= 0) return;                  // whole block returns together
  const size_t o  = row_off_H[H];
  const size_t so = seq_off_H[H];
  const int j0     = i + turn + 1 + (int)size_off_H[nfiles + 1];   /* + the slot's column shift (3a.2) */
  const int MLbase = P->MLbase;
  const int en_i   = up_ml_ok[so + (size_t)i] ? MLbase : INF;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  int carry = (size_off_H[nfiles + 1] > 0) ? energy_min[o + j0 - 1] : INF;   /* E[j0-1]; see fml_scan_block() */

  for(long long base = 0; base < width; base += TILE) {
    // 1. stage (a,c), coalesced. Past the end: (INF,0), which the re-walk passes through.
#pragma unroll
    for(int r = 0; r < K; r++) {
      const int idx = r*TW + t;
      const long long k = base + idx;
      int a = INF, c = 0;
      if(k < width) {
        const int j = j0 + (int)k;
        const int c_term = fml_tadd(new_e[o+j], e3p00[o+j]);
        const int e3     = fml_tadd(fml_prev[o+j], en_i);
        a = fml_tmin(c_term, e3);
        if (gq_row) {
          const int g = gq_row[o+j];
          if (g != INF)
            a = fml_tmin(a, g + P->MLintern[0]);
        }
        c = up_ml_ok[so + (size_t)j] ? MLbase : INF;
      }
      sa[FML_PAD(idx)] = a; sc[FML_PAD(idx)] = c;
    }
    __syncthreads();

    // 2. this thread's K maps, composed from its first element (no identity).
    int ta = sa[FML_PAD(t*K)], tc = sc[FML_PAD(t*K)];
#pragma unroll
    for(int q = 1; q < K; q++) {
      int a = sa[FML_PAD(t*K+q)], c = sc[FML_PAD(t*K+q)];
      fml_compose(a, c, ta, tc); ta = a; tc = c;
    }

    // 3a. inclusive warp scan of the thread aggregates.
    int ia = ta, ic = tc;
#pragma unroll
    for(int d = 1; d < 32; d <<= 1) {
      const int la = __shfl_up_sync(0xffffffffu, ia, d);
      const int lc = __shfl_up_sync(0xffffffffu, ic, d);
      if(lane >= d) fml_compose(ia, ic, la, lc);
    }
    // exclusive-in-warp = the inclusive value of lane-1 (none for lane 0)
    const int xa = __shfl_up_sync(0xffffffffu, ia, 1);
    const int xc = __shfl_up_sync(0xffffffffu, ic, 1);
    if(lane == 31) { wa[warp] = ia; wc[warp] = ic; }
    __syncthreads();
    // 3b. inclusive scan of the warp totals, by warp 0.
    if(warp == 0) {
      int va = (lane < NW) ? wa[lane] : INF, vc = (lane < NW) ? wc[lane] : 0;
#pragma unroll
      for(int d = 1; d < NW; d <<= 1) {
        const int la = __shfl_up_sync(0xffffffffu, va, d);
        const int lc = __shfl_up_sync(0xffffffffu, vc, d);
        if(lane >= d && lane < NW) fml_compose(va, vc, la, lc);
      }
      if(lane < NW) { wa[lane] = va; wc[lane] = vc; }
    }
    __syncthreads();

    // 4. E_in: APPLY (never compose with) the carry, then earlier warps, then earlier
    //    lanes -- f(x) = min(a, x (+) c) -- and re-walk with the host recurrence.
    int E = carry;
    if(warp > 0) E = fml_tmin(wa[warp-1], fml_tadd(E, wc[warp-1]));
    if(lane > 0) E = fml_tmin(xa, fml_tadd(E, xc));
#pragma unroll
    for(int q = 0; q < K; q++) {
      const int idx = t*K + q;
      E = fml_tmin(sa[FML_PAD(idx)], fml_tadd(E, sc[FML_PAD(idx)]));
      sa[FML_PAD(idx)] = E;               // this thread owns idx: safe in place
    }
    if(t == TW-1) *sE = E;                // the tile's last element -> next carry
    __syncthreads();
    // write back coalesced
#pragma unroll
    for(int r = 0; r < K; r++) {
      const int idx = r*TW + t;
      const long long k = base + idx;
      if(k < width) energy_min[o + j0 + (int)k] = sa[FML_PAD(idx)];
    }
    carry = *sE;
    __syncthreads();                      // sa/sE fully read before the next tile
  }
}
#undef FML_PAD

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
  const int j0     = i + turn + 1;
  const int MLbase = P->MLbase;
  const int en_i   = up_ml_ok[so + (size_t)i] ? MLbase : INF;

  // sa/sc are supplied by the caller: the megakernel owns its shared memory.

  int carry = INF;                        // E[j0-1], the host's `j == i+turn+1` special case
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

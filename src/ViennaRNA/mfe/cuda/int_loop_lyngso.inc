/* int_loop_lyngso.inc -- Lyngsø interior loops (PORT_LYNGSO_INTLOOP.md).
 *
 * S2: RNA_INT_LOOP_LYNGSO=1 replaces the dense enumeration. Per row, ly_carry_kernel (below) builds
 * G(i,·,u) for every column, then the warp kernel's LY variant (int_loop_cell.inc) enumerates only
 * the direct loops and adds the 25 G uses. Default on since S5 (RNA_INT_LOOP_LYNGSO=0 turns it off).
 *
 * S1, the selftest that proved it, follows:
 *
 * The Lyngsø form of the interior-loop term is computed BESIDE the dense int_loop kernel and compared
 * with it, cell for cell, every row. Nothing reads the Lyngsø result -- the dense kernel stays
 * authoritative -- so this stage cannot change any output. It proves, on the device and on the
 * device's own c (triangle or ring), what tools/lyngso_equiv.c proved on upstream's matrices.
 *
 *   RNA_INT_LOOP_LYNGSO_SELFTEST=1   run it (counters and the first mismatches printed at exit)
 *   RNA_INT_LOOP_LYNGSO_NEGCTL=1     drop the carry from (i+1,j-1) -- must go red
 *
 * THE FORM. A generic interior loop -- short side >= 2, not 2x2 / 2x3 / 3x2 -- costs
 *     internal_loop[u] + mismatchI[type][S(i+1)][S(j-1)]                       (outer, per cell)
 *   + mismatchI[type_2][S(q+1)][S(p-1)] + MIN2(max_ninio, |u1-u2| ninio) + c(p,q)   (inner, "entry")
 *   + SaltLoop[u+2]
 * (interior_loopx.h's generic branch exactly). The entry does not depend on (i,j), so
 *     G(i,j,u) = min( G(i+1,j-1,u-2) if up_int[i+1] >= 1 && up_int[j-1] >= 1 ,
 *                     fresh entries at (i,j): short side exactly 2, and 3x3 / 3x4 / 4x3 )
 * and the cell is min( DIRECT loops through Energy() -- stacks, bulges, 1xn, 2x2, 2x3, 3x2 --
 *                       min_u G(i,j,u) + internal_loop[u] + mismatchI(outer) + SaltLoop[u+2] ).
 * Generic u runs 6..30 (3x3 and 2x4 are the smallest). Entries obey the same hard-constraint, pair
 * type (0 -> 7, then rtype) and --noClosingGU rules as Energy(), which is what S0 established.
 *
 * G is maintained for EVERY column of the row, not only for the cells the dense kernel computes:
 * a cell's carry source (i+1,j-1) must hold a value for that very row, never one left two rows ago
 * in the same ping-pong buffer. */

#ifndef LY_U0                                     /* also defined by int_loop_cell.inc */
#define LY_U0   6
#define LY_NU   (MAXLOOP - LY_U0 + 1)            /* 25 planes: u = 6..30 */
#endif
#define LY_NSTAT 6   /* 0 cells compared, 1 mismatches, 2 live cells, 3 fresh entries, 4 carried entries, 5 direct evals */
#define LY_NBAD  4

static int*                d_ly_G[2] = { NULL, NULL };   /* [LY_NU][g_row_total], rows of parity 0/1 */
static int*                d_ly_E    = NULL;             /* S6b: [32][g_row_total], e(k,l) at slot k & 31 */
static unsigned long long* d_ly_stat = NULL;
static int*                d_ly_bad  = NULL;
static int                 g_ly_on   = 0;
static unsigned long long  g_ly_tot[LY_NSTAT];
static int                 g_ly_badh[LY_NBAD * 5];
static int                 g_ly_nbadh = 0, g_ly_chunks = 0;

static int
ly_selftest(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("RNA_INT_LOOP_LYNGSO_SELFTEST"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
  return v;
}

/* S2: the warp kernel's LY variant REPLACES the dense enumeration. DEFAULT ON since S5 (2026-10-07,
 * A100: exact on the 20-case matrix, 14 adversarial and 10,082 natural records; wall -4.6 % at
 * 400x5601, -4.3 % at 3000x1200; PORT_LYNGSO_INTLOOP.md). RNA_INT_LOOP_LYNGSO=0 restores the dense
 * int_loop. Unset, the selftest knob alone keeps it off, so the selftest still has a dense kernel to
 * compare with. */
static int g_ly_eval = 0;   /* this chunk */
static int
ly_eval_explicit(void)      /* RNA_INT_LOOP_LYNGSO set by the user: refusals are said aloud */
{
  const char *e = getenv("RNA_INT_LOOP_LYNGSO");
  return (e && e[0]) ? 1 : 0;
}

static int
ly_eval_knob(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_INT_LOOP_LYNGSO");
    if (e && e[0]) v = (e[0] != '0') ? 1 : 0;
    else           v = ly_selftest() ? 0 : 1;
  }
  return v;
}

static int
ly_negctl(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("RNA_INT_LOOP_LYNGSO_NEGCTL"); v = (e && e[0]) ? atoi(e) : 0; if (v < 0 || v > 2) v = 1; }
  return v;   /* 1: the carry dropped; 2 (S6b): the e-ring writes dropped */
}

/* S6b (PORT_LYNGSO_INTLOOP.md 10.3): fresh entries read e(k,l) from a 32-row ring that carry(i)
 * fills for row i+1, instead of recomputing c + Hc + type + mismatch up to ~50 times per (k,l).
 * DEFAULT ON since S8 (2026-10-07, A100: exact, wall -9.4 % vs v1 at 400x5601; 10.10).
 * RNA_INT_LOOP_LYNGSO_ERING=0 restores the recomputed entries. */
static int
ly_knob_default_on(const char *name)
{
  const char *e = getenv(name);
  return (e && e[0]) ? (e[0] != '0') : 1;
}

static int
ly_knob_explicit(const char *name)   /* set by the user: its banner is said aloud */
{
  const char *e = getenv(name);
  return (e && e[0]) ? 1 : 0;
}

static int
ly_ering(void)
{
  static int v = -1;
  if (v < 0) v = ly_knob_default_on("RNA_INT_LOOP_LYNGSO_ERING");
  return v;
}

/* S6a' (PORT_LYNGSO_INTLOOP.md 10.7): RNA_INT_LOOP_LYNGSO_V2=1 runs the carry on a 3-D grid --
 * z the record, x the columns j > i, y a group of PT planes -- so there is no per-thread record
 * search and no thread below the diagonal. PT (planes per thread, 25 | 5 | 1) is chosen PER ROW
 * from the row's active slots against the device's thread capacity: a full row keeps one thread
 * per slot (25 planes each), an under-filled one splits the planes to fill the GPU. S6a measured
 * the fixed split (PT = 1 always): exact, but 1.6-1.9x the instructions where the GPU was already
 * full. RNA_INT_LOOP_LYNGSO_PT=25|5|1 forces one PT (the bars and the sweeps); a forced PT implies
 * V2. DEFAULT ON since S8 (2026-10-07, with ERING; 10.10); RNA_INT_LOOP_LYNGSO_V2=0 runs the v1
 * carry. ly_carry_u() is shared with v1, so the arithmetic is the same by construction. */
static int
ly_v2(void)
{
  static int v = -1;
  if (v < 0) {
    const char *p = getenv("RNA_INT_LOOP_LYNGSO_PT");
    v = (p && p[0]) ? 1 : ly_knob_default_on("RNA_INT_LOOP_LYNGSO_V2");
  }
  return v;
}

/* 10.11: RNA_INT_LOOP_LYNGSO_LB=1 launches the ring carry at __launch_bounds__(128, 16) (32 registers).
 * Off until the A100 says which wins; the laptop (1536 threads/SM) is full at either count. */
static int
ly_lb(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("RNA_INT_LOOP_LYNGSO_LB"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
  return v;
}

/* 10.11 measurement arm: RNA_INT_LOOP_LYNGSO_FRAME=1 runs the de804f63 carry (with its local frame). */
static int
ly_frame(void)
{
  static int v = -1;
  if (v < 0) { const char *e = getenv("RNA_INT_LOOP_LYNGSO_FRAME"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
  return v;
}

static int
ly_pt_forced(void)          /* 0 = adaptive */
{
  static int v = -1;
  if (v < 0) {
    const char *p = getenv("RNA_INT_LOOP_LYNGSO_PT");
    v = (p && p[0]) ? atoi(p) : 0;
    if (v != 0 && v != 1 && v != 5 && v != 25) v = 0;
  }
  return v;
}

/* Host side of the per-row choice: each record's length (from row_off_H), and the device's
 * resident-thread capacity, queried once. */
static int*  g_ly_len   = NULL;
static int   g_ly_n     = 0, g_ly_lmax = 0;
static long  g_ly_cap   = 0;
static long  g_ly_rows_pt[3] = { 0, 0, 0 };   /* rows launched at PT 25 / 5 / 1, this chunk */

static long
ly_capacity(void)
{
  if (g_ly_cap == 0) {
    int dev = 0, sms = 0, tps = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&tps, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    g_ly_cap = (long)sms * (long)tps;
    if (g_ly_cap <= 0) g_ly_cap = 1;
  }
  return g_ly_cap;
}

static void
ly_report(void)
{
  const unsigned long long *t = g_ly_tot;
  fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_SELFTEST%s: %d chunk(s), %llu cells compared (%llu live), %llu MISMATCHING; "
                  "entries %llu fresh + %llu carried, %llu direct evaluations\n",
          __FILE__, ly_negctl() == 1 ? " (NEGCTL: carry dropped)" : ly_negctl() == 2 ? " (NEGCTL: e ring not written)" : "", g_ly_chunks, t[0], t[2], t[1], t[3], t[4], t[5]);
  for (int b = 0; b < g_ly_nbadh; b++)
    fprintf(stderr, "%-24s   mismatch H=%d i=%d j=%d dense %d lyngso %d\n", __FILE__,
            g_ly_badh[5*b], g_ly_badh[5*b+1], g_ly_badh[5*b+2], g_ly_badh[5*b+3], g_ly_badh[5*b+4]);
}

__global__ void
ly_fill_kernel(int* __restrict__ a, const size_t n)
{
  const size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (m < n) a[m] = INF;
}

/* init_gpu2(): one pair of G row buffers for the chunk. Refused where records are not on one row. */
static void
ly_alloc(const int nfiles, const size_t* row_off_H)
{
  g_ly_on = 0;
  g_ly_eval = 0;
  if (!ly_selftest() && !ly_eval_knob()) return;
  const char *knob = ly_eval_knob() ? "RNA_INT_LOOP_LYNGSO" : "RNA_INT_LOOP_LYNGSO_SELFTEST";
  const char *why = NULL;
  if (rnafold_continuous_flow())      why = "continuous flow (records on different rows)";
  else if (rnafold_slot_flow() >= 1)  why = "slot flow (slots are handed over mid-sweep)";
  else if (rnafold_megakernel())      why = "the megakernel (its own fused int_loop)";
  else if (ly_eval_knob() && rnafold_int_loop_unroll() > 1)
                                      why = "RNA_INT_LOOP_UNROLL > 1 (the LY variant is instantiated at U = 1)";
  else if (ly_eval_knob() && !rnafold_int_loop_warp())
                                      why = "the block-per-cell int_loop kernel (LY lives in the warp kernel)";
  if (why) {
    static int said = 0;
    /* asked for (or the selftest) -> always said; the default -> said unless quiet, like the other banners */
    if (!said && (ly_selftest() || ly_eval_explicit() || !vrna_cuda_quiet())) {
      said = 1; fprintf(stderr, "%-24s %s REFUSED: %s -- the dense int_loop runs\n", __FILE__, knob, why);
    }
    return;
  }
  if (ly_eval_knob() && ly_selftest()) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_SELFTEST IGNORED: with RNA_INT_LOOP_LYNGSO=1 "
                                           "there is no dense int_loop to compare with\n", __FILE__); }
  }
  g_ly_eval = ly_eval_knob();
  const size_t n = (size_t)LY_NU * g_row_total;
  for (int b = 0; b < 2; b++) {
    gpuErrchk( cudaMalloc((void **)&d_ly_G[b], sizeof(int) * (n ? n : 1)) );
    if (n) { ly_fill_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d_ly_G[b], n); gpuErrchk( cudaPeekAtLastError() ); }
  }
  if (ly_ering()) {
    const size_t ne = (size_t)32 * g_row_total;
    gpuErrchk( cudaMalloc((void **)&d_ly_E, sizeof(int) * (ne ? ne : 1)) );
    if (ne) { ly_fill_kernel<<<(unsigned)((ne + 255) / 256), 256>>>(d_ly_E, ne); gpuErrchk( cudaPeekAtLastError() ); }
  }
  gpuErrchk( cudaMalloc((void **)&d_ly_stat, sizeof(unsigned long long) * LY_NSTAT) );
  gpuErrchk( cudaMalloc((void **)&d_ly_bad,  sizeof(int) * LY_NBAD * 5) );
  gpuErrchk( cudaMemset(d_ly_stat, 0, sizeof(unsigned long long) * LY_NSTAT) );
  g_ly_on = 1;
  g_ly_chunks++;
  /* S6a': the record lengths, for the per-row active-slot count */
  free(g_ly_len); g_ly_len = (int *)malloc(sizeof(int) * (nfiles > 0 ? nfiles : 1));
  g_ly_n = nfiles; g_ly_lmax = 0;
  for (int H = 0; H < nfiles; H++) {
    g_ly_len[H] = (int)(row_off_H[H + 1] - row_off_H[H]) - 1;
    if (g_ly_len[H] > g_ly_lmax) g_ly_lmax = g_ly_len[H];
  }
  g_ly_rows_pt[0] = g_ly_rows_pt[1] = g_ly_rows_pt[2] = 0;
  static int said = 0;
  if (!said) {
    said = 1;
    const int loud = !vrna_cuda_quiet();
    if (ly_ering() && (loud || ly_knob_explicit("RNA_INT_LOOP_LYNGSO_ERING")))   /* asked for: always said */
      fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_ERING ACTIVE: fresh entries read a 32-row e ring (%.1f MB)\n",
              __FILE__, 32.0 * (double)g_row_total * sizeof(int) / 1048576.0);
    if (ly_v2() && (loud || ly_knob_explicit("RNA_INT_LOOP_LYNGSO_V2") || ly_knob_explicit("RNA_INT_LOOP_LYNGSO_PT")))
                   /* asked for: always said (positive evidence for the S6a' bars) */
      fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_V2 ACTIVE: 3-D carry grid, planes per thread %s (capacity %ld threads)\n",
              __FILE__, ly_pt_forced() ? (ly_pt_forced() == 25 ? "forced 25" : ly_pt_forced() == 5 ? "forced 5" : "forced 1")
                                       : "per row (5, or 1 when 5 cannot fill)", ly_capacity());
    if (ly_frame() && ly_v2())   /* asked for: always said */
      fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_FRAME ACTIVE: the de804f63 carry (f1/f2 local frame), measurement arm\n", __FILE__);
    if (ly_lb() && ly_v2() && ly_ering() && !ly_frame())   /* asked for: always said */
      fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_LB ACTIVE: ring carry at __launch_bounds__(128, 16)\n", __FILE__);
    if (g_ly_eval) {
      if (!vrna_cuda_quiet())
        fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO ACTIVE: int_loop enumerates the direct loops only; generic "
                        "interior loops come from the carried table (G %.1f MB%s)\n", __FILE__,
                2.0 * (double)n * sizeof(int) / 1048576.0,
                ly_negctl() == 1 ? ", NEGCTL: carry dropped -- WRONG ANSWERS EXPECTED"
                                : ly_negctl() == 2 ? ", NEGCTL: e ring not written -- WRONG ANSWERS EXPECTED" : "");
    } else {
      atexit(ly_report);
      fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_SELFTEST ACTIVE: the Lyngsø carry form computed beside the dense "
                      "int_loop and compared cell for cell (G %.1f MB%s)\n", __FILE__,
              2.0 * (double)n * sizeof(int) / 1048576.0,
              ly_negctl() == 1 ? ", NEGCTL: carry dropped" : ly_negctl() == 2 ? ", NEGCTL: e ring not written" : "");
    }
  }
}

/* teardown_gpu2(): counters first, then free by the pointer. */
static void
ly_free(void)
{
  if (g_ly_on && ly_v2() && (!vrna_cuda_quiet() || ly_knob_explicit("RNA_INT_LOOP_LYNGSO_V2")
                             || ly_knob_explicit("RNA_INT_LOOP_LYNGSO_PT")))   /* which split each row ran */
    fprintf(stderr, "%-24s RNA_INT_LOOP_LYNGSO_V2: rows by planes per thread: 25:%ld 5:%ld 1:%ld\n",
            __FILE__, g_ly_rows_pt[0], g_ly_rows_pt[1], g_ly_rows_pt[2]);
  if (d_ly_stat) {
    unsigned long long s[LY_NSTAT]; int b[LY_NBAD * 5];
    gpuErrchk( cudaMemcpy(s, d_ly_stat, sizeof(s), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(b, d_ly_bad,  sizeof(b), cudaMemcpyDeviceToHost) );
    for (int t = 0; t < LY_NSTAT; t++) g_ly_tot[t] += s[t];
    for (unsigned long long m = 0; m < s[1] && m < LY_NBAD && g_ly_nbadh < LY_NBAD; m++, g_ly_nbadh++)
      memcpy(&g_ly_badh[5 * g_ly_nbadh], &b[5 * m], 5 * sizeof(int));
  }
  for (int k = 0; k < 2; k++) if (d_ly_G[k]) { gpuErrchk( cudaFree(d_ly_G[k]) ); d_ly_G[k] = NULL; }
  if (d_ly_E) { gpuErrchk( cudaFree(d_ly_E) ); d_ly_E = NULL; }
  if (d_ly_stat) { gpuErrchk( cudaFree(d_ly_stat) ); d_ly_stat = NULL; }
  if (d_ly_bad)  { gpuErrchk( cudaFree(d_ly_bad) );  d_ly_bad  = NULL; }
  g_ly_on = 0;
}

/* The pair type Energy() uses: raw ptype 0 promoted to 7 (cell_invariants()), and for an inner
 * pair rtype[] applied AFTER the promotion. */
__device__ __forceinline__ int
ly_type(const unsigned int* __restrict__ S, const char* __restrict__ pair_, const int H, const int s_stride,
        const int a, const int b)
{
  const unsigned char t = Ptype(S, pair_, H, s_stride, a, b);
  return (t == 0) ? 7 : (int)t;
}

/* The c reader the dense kernel would use for this record: the triangle, or the 32-row ring. */
__device__ __forceinline__ c_tri_reader
ly_reader(c_tri_reader*, const int* my_c, const int*, const size_t, const size_t tri, const size_t)
{
  c_tri_reader r; r.base = my_c + tri; return r;
}

__device__ __forceinline__ c_ring_reader
ly_reader(c_ring_reader*, const int*, const int* ring, const size_t stride, const size_t, const size_t o)
{
  c_ring_reader r; r.base = ring + o; r.stride = stride; return r;
}

/* S6b: the expensive part of a fresh entry, independent of the outer pair (PORT_LYNGSO_INTLOOP.md 10.3):
 *     e(k,l) = c(k,l) + mismatchI[rtype(type(k,l))][S(l+1)][S(k-1)],
 * INF where Hc forbids (k,l), c is INF, or --noClosingGU rejects the inner type. The ninio term
 * (u alone) and the up_int gates (i, u1 / l, u2) stay with the use. */
template<class CREAD>
__device__ __forceinline__ int
ly_entry(const int k, const int l, const int H, const int s_stride, const CREAD& cr,
         const unsigned int* __restrict__ hc_H, const cuda_param_t* __restrict__ P,
         const char* __restrict__ pair_, const unsigned int* __restrict__ S)
{
  if (!Hc(Indx(k, l), hc_H)) return INF;
  const int c = cr(k, l);
  if (c >= INF) return INF;
  const int t2 = P->rtype[ly_type(S, pair_, H, s_stride, k, l)];
  if (P->noGUclosure && (t2 == 3 || t2 == 4)) return INF;
  return c + P->mismatchI[t2][unpack(S, H, s_stride, l + 1)][unpack(S, H, s_stride, k - 1)];
}

/* One fresh entry (u1, u2) of G(i, j): the interior loop (i,j) -> (k,l), k = i+u1+1, l = j-u2-1.
 * A call per entry, not a loop over an f1[]/f2[] array: the runtime-indexed array was a 32-byte
 * local-memory frame that the sm_80 carry read 52 MB per launch (S8 10.10, 10.11). RING: 1 = e from
 * the ring only, 0 = computed only, 2 = Er decides at run time (v1). A compile-time choice keeps the
 * unused path, and its registers, out of the v2 kernels. */
template<class CREAD, int RING>
__device__ __forceinline__ void
ly_fresh(const int u1, const int u2, const int i, const int j, const int H, const size_t o,
         const int s_stride, const CREAD& cr, const unsigned int* __restrict__ hc_H,
         const unsigned char* __restrict__ up, const cuda_param_t* __restrict__ P,
         const char* __restrict__ pair_, const unsigned int* __restrict__ S,
         const size_t row_total, const int* Er, int& g, unsigned long long& nfresh)
{
  const int k = i + u1 + 1, l = j - u2 - 1;
  if (l - k < 1) return;
  if (up && ((int)up[i + 1] < u1 || (int)up[l + 1] < u2)) return;
  const int e = (RING == 1 || (RING == 2 && Er)) ? Er[(size_t)(k & 31) * row_total + o + (size_t)l]
                                                  : ly_entry(k, l, H, s_stride, cr, hc_H, P, pair_, S);
  if (e >= INF) return;
  const int d = u1 > u2 ? u1 - u2 : u2 - u1;
  const int v = e + MIN2(P->max_ninio, d * P->ninio2);
  nfresh++;
  if (v < g) g = v;
}

/* One plane u of G at (i, j): the carry from (i+1, j-1) and this cell's fresh entries. Shared by the
 * u-loop kernel (v1) and the 3-D kernel (S6a'), so both run the same arithmetic. Er (S6b): the e
 * ring, 32 rows of e(k,l) at (k & 31) * row_total + o + l, or NULL to compute e in place. */
template<class CREAD, int RING>
__device__ __forceinline__ int
ly_carry_u(const int u, const int i, const int j, const int H, const size_t o, const int s_stride,
           const int carry_ok, const CREAD& cr, const unsigned int* __restrict__ hc_H,
           const unsigned char* __restrict__ up, const cuda_param_t* __restrict__ P,
           const char* __restrict__ pair_, const unsigned int* __restrict__ S,
           const size_t row_total, const int* __restrict__ Gp, const int* Er,
           unsigned long long& nfresh, unsigned long long& ncarry)
{
  int g = INF;
  if (carry_ok && u - 2 >= LY_U0) {
    g = Gp[(size_t)(u - 2 - LY_U0) * row_total + (o + (size_t)(j - 1))];
    if (g < INF) ncarry++;
  }
  /* fresh at (i,j): (2,u-2) and (u-2,2) for u >= 6, plus (3,3) at 6 and (3,4),(4,3) at 7 */
  ly_fresh<CREAD, RING>(2, u - 2, i, j, H, o, s_stride, cr, hc_H, up, P, pair_, S, row_total, Er, g, nfresh);
  if (u - 2 != 2)                                     /* (2,2) is never here; guard the duplicate anyway */
    ly_fresh<CREAD, RING>(u - 2, 2, i, j, H, o, s_stride, cr, hc_H, up, P, pair_, S, row_total, Er, g, nfresh);
  if (u == 6)
    ly_fresh<CREAD, RING>(3, 3, i, j, H, o, s_stride, cr, hc_H, up, P, pair_, S, row_total, Er, g, nfresh);
  if (u == 7) {
    ly_fresh<CREAD, RING>(3, 4, i, j, H, o, s_stride, cr, hc_H, up, P, pair_, S, row_total, Er, g, nfresh);
    ly_fresh<CREAD, RING>(4, 3, i, j, H, o, s_stride, cr, hc_H, up, P, pair_, S, row_total, Er, g, nfresh);
  }
  return g;
}

/* S6b: row i+1's e, for column j, into the ring. Row i+1's c is final: it was written on this
 * stream before int_loop(i), whose stacks read it. carry(i) reads e rows i+3 .. i+29 only, so a
 * kernel never reads the row it writes. negctl 2 skips the write: stale e, must go red. */
template<class CREAD>
__device__ __forceinline__ void
ly_entry_store(int* Er, const int negctl, const int i, const int j, const int H, const size_t o,
               const int s_stride, const CREAD& cr, const unsigned int* __restrict__ hc_H,
               const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
               const unsigned int* __restrict__ S, const size_t row_total)
{
  const int k = i + 1;
  if (!Er || negctl == 2 || j <= k) return;
  Er[(size_t)(k & 31) * row_total + o + (size_t)j] = ly_entry(k, j, H, s_stride, cr, hc_H, P, pair_, S);
}

/* Row i: G for every column j in (i, len_H], one thread per row-layout slot, all 25 planes (v1). */
template<class CREAD, int RING>
__global__ void
ly_carry_kernel(const int nfiles, const int i, const int length, const int negctl,
                const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
                const unsigned int* __restrict__ S, const unsigned int* __restrict__ hccc,
                const unsigned char* __restrict__ up_int,
                const int* __restrict__ my_c, const int* __restrict__ c_ring, const size_t c_ring_stride,
                const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ hc_off_H, const size_t row_total,
                const int* __restrict__ Gp, int* __restrict__ Gc, int* Er,
                unsigned long long* __restrict__ stat)
{
  const size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= row_total) return;
  const int    H = flatten_index_to_H(m, row_off_H, nfiles);
  const size_t o = row_off_H[H];
  const int    j = (int)(m - o);
  const int  len = (int)(row_off_H[H + 1] - o) - 1;
  /* slots j <= i (and past the record) are never read: a reader at row r takes (r+1, j-1) only for
   * j-1 > r+1, and a cell of row i only its own j > i. So nothing is written there. */
  if (j <= i || j > len) return;

  const CREAD cr = ly_reader((CREAD *)0, my_c, c_ring, c_ring_stride, tri_off_H[H], o);

  const unsigned int*  hc_H = &hccc[hc_off_H[H]];
  const unsigned char* up   = up_int ? &up_int[o] : NULL;
  const int s_stride = S_STRIDE(length);
  const int carry_ok = negctl != 1 && j - 1 > i + 1 && (!up || (up[i + 1] >= 1 && up[j - 1] >= 1));
  unsigned long long nfresh = 0, ncarry = 0;

#pragma unroll 1
  for (int u = LY_U0; u <= MAXLOOP; u++)
    Gc[(size_t)(u - LY_U0) * row_total + m] =
      ly_carry_u<CREAD, RING>(u, i, j, H, o, s_stride, carry_ok, cr, hc_H, up, P, pair_, S, row_total, Gp, Er, nfresh, ncarry);
  ly_entry_store(Er, negctl, i, j, H, o, s_stride, cr, hc_H, P, pair_, S, row_total);
  if (stat && nfresh) atomicAdd(&stat[3], nfresh);   /* stat NULL in production (S2) */
  if (stat && ncarry) atomicAdd(&stat[4], ncarry);
}

/* S6a': the same row on a 3-D grid. z = the record H (no search), x = j - (i + 1) (no thread
 * below the diagonal), y = a group of PT planes starting at LY_U0 + y * PT. The y = 0 group
 * writes the e row (S6b). RING: 1 when the e ring is on (Er non-NULL), else 0. The body is shared
 * by two kernels (10.11): ly_carry3_kernel, unbounded, and ly_carry3_lb_kernel, the ring carry at
 * __launch_bounds__(128, 16), i.e. 32 registers (full A100 occupancy, no spill at sm_80) where ptxas
 * picks 38-40 unbounded. A (128, 1) bound is NOT neutral (40 -> 55), hence two kernels.
 * RNA_INT_LOOP_LYNGSO_LB=1 picks the bounded one. */
template<class CREAD, int PT, int RING>
__device__ __forceinline__ void
ly_carry3_body(const int i, const int length, const int negctl,
                 const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
                 const unsigned int* __restrict__ S, const unsigned int* __restrict__ hccc,
                 const unsigned char* __restrict__ up_int,
                 const int* __restrict__ my_c, const int* __restrict__ c_ring, const size_t c_ring_stride,
                 const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                 const size_t* __restrict__ hc_off_H, const size_t row_total,
                 const int* __restrict__ Gp, int* __restrict__ Gc, int* Er,
                 unsigned long long* __restrict__ stat)
{
  const int    H   = (int)blockIdx.z;
  const size_t o   = row_off_H[H];
  const int    len = (int)(row_off_H[H + 1] - o) - 1;
  const int    j   = i + 1 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (j > len) return;
  const size_t m = o + (size_t)j;

  const CREAD cr = ly_reader((CREAD *)0, my_c, c_ring, c_ring_stride, tri_off_H[H], o);
  const unsigned int*  hc_H = &hccc[hc_off_H[H]];
  const unsigned char* up   = up_int ? &up_int[o] : NULL;
  const int s_stride = S_STRIDE(length);
  const int carry_ok = negctl != 1 && j - 1 > i + 1 && (!up || (up[i + 1] >= 1 && up[j - 1] >= 1));
  unsigned long long nfresh = 0, ncarry = 0;

  const int u0 = LY_U0 + (int)blockIdx.y * PT;
#pragma unroll 1
  for (int t = 0; t < PT; t++) {
    const int u = u0 + t;
    Gc[(size_t)(u - LY_U0) * row_total + m] =
      ly_carry_u<CREAD, RING>(u, i, j, H, o, s_stride, carry_ok, cr, hc_H, up, P, pair_, S, row_total, Gp, Er, nfresh, ncarry);
  }
  if (blockIdx.y == 0) ly_entry_store(Er, negctl, i, j, H, o, s_stride, cr, hc_H, P, pair_, S, row_total);
  if (stat && nfresh) atomicAdd(&stat[3], nfresh);
  if (stat && ncarry) atomicAdd(&stat[4], ncarry);
}

template<class CREAD, int PT, int RING>
__global__ void
ly_carry3_kernel(const int i, const int length, const int negctl,
                 const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
                 const unsigned int* __restrict__ S, const unsigned int* __restrict__ hccc,
                 const unsigned char* __restrict__ up_int,
                 const int* __restrict__ my_c, const int* __restrict__ c_ring, const size_t c_ring_stride,
                 const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                 const size_t* __restrict__ hc_off_H, const size_t row_total,
                 const int* __restrict__ Gp, int* __restrict__ Gc, int* Er,
                 unsigned long long* __restrict__ stat)
{
  ly_carry3_body<CREAD, PT, RING>(i, length, negctl, P, pair_, S, hccc, up_int, my_c, c_ring, c_ring_stride, tri_off_H, row_off_H, hc_off_H, row_total, Gp, Gc, Er, stat);
}

template<class CREAD, int PT>
__global__ void
__launch_bounds__(128, 16)
ly_carry3_lb_kernel(const int i, const int length, const int negctl,
                 const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
                 const unsigned int* __restrict__ S, const unsigned int* __restrict__ hccc,
                 const unsigned char* __restrict__ up_int,
                 const int* __restrict__ my_c, const int* __restrict__ c_ring, const size_t c_ring_stride,
                 const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                 const size_t* __restrict__ hc_off_H, const size_t row_total,
                 const int* __restrict__ Gp, int* __restrict__ Gc, int* Er,
                 unsigned long long* __restrict__ stat)
{
  ly_carry3_body<CREAD, PT, 1>(i, length, negctl, P, pair_, S, hccc, up_int, my_c, c_ring, c_ring_stride, tri_off_H, row_off_H, hc_off_H, row_total, Gp, Gc, Er, stat);
}

/* 10.11 MEASUREMENT ARM, to retire after S9: the carry exactly as de804f63 shipped it -- the f1/f2
 * array (a 32-byte local frame at sm_80), unroll 5, the ring chosen at run time. RNA_INT_LOOP_LYNGSO_FRAME=1
 * runs it in place of ly_carry3_kernel, so the A100 compares the frame against its removal in one binary. */
template<class CREAD>
__device__ __forceinline__ int
ly_carry_u_frame(const int u, const int i, const int j, const int H, const size_t o, const int s_stride,
           const int carry_ok, const CREAD& cr, const unsigned int* __restrict__ hc_H,
           const unsigned char* __restrict__ up, const cuda_param_t* __restrict__ P,
           const char* __restrict__ pair_, const unsigned int* __restrict__ S,
           const size_t row_total, const int* __restrict__ Gp, const int* Er,
           unsigned long long& nfresh, unsigned long long& ncarry)
{
  int g = INF;
  if (carry_ok && u - 2 >= LY_U0) {
    g = Gp[(size_t)(u - 2 - LY_U0) * row_total + (o + (size_t)(j - 1))];
    if (g < INF) ncarry++;
  }
  /* fresh at (i,j): (2,u-2) and (u-2,2) for u >= 6, plus (3,3) at 6 and (3,4),(4,3) at 7 */
  int f1[4], f2[4], nf = 0;
  f1[nf] = 2;     f2[nf] = u - 2; nf++;
  f1[nf] = u - 2; f2[nf] = 2;     nf++;
  if (u == 6) { f1[nf] = 3; f2[nf] = 3; nf++; }
  if (u == 7) { f1[nf] = 3; f2[nf] = 4; nf++; f1[nf] = 4; f2[nf] = 3; nf++; }
  for (int t = 0; t < nf; t++) {
    const int u1 = f1[t], u2 = f2[t];
    if (t == 1 && u1 == u2) continue;                 /* (2,2) is never here; guard the duplicate anyway */
    const int k = i + u1 + 1, l = j - u2 - 1;
    if (l - k < 1) continue;
    if (up && ((int)up[i + 1] < u1 || (int)up[l + 1] < u2)) continue;
    const int e = Er ? Er[(size_t)(k & 31) * row_total + o + (size_t)l]
                     : ly_entry(k, l, H, s_stride, cr, hc_H, P, pair_, S);
    if (e >= INF) continue;
    const int d = u1 > u2 ? u1 - u2 : u2 - u1;
    const int v = e + MIN2(P->max_ninio, d * P->ninio2);
    nfresh++;
    if (v < g) g = v;
  }
  return g;
}

template<class CREAD, int PT>
__global__ void
ly_carry3_frame_kernel(const int i, const int length, const int negctl,
                 const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
                 const unsigned int* __restrict__ S, const unsigned int* __restrict__ hccc,
                 const unsigned char* __restrict__ up_int,
                 const int* __restrict__ my_c, const int* __restrict__ c_ring, const size_t c_ring_stride,
                 const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                 const size_t* __restrict__ hc_off_H, const size_t row_total,
                 const int* __restrict__ Gp, int* __restrict__ Gc, int* Er,
                 unsigned long long* __restrict__ stat)
{
  const int    H   = (int)blockIdx.z;
  const size_t o   = row_off_H[H];
  const int    len = (int)(row_off_H[H + 1] - o) - 1;
  const int    j   = i + 1 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (j > len) return;
  const size_t m = o + (size_t)j;

  const CREAD cr = ly_reader((CREAD *)0, my_c, c_ring, c_ring_stride, tri_off_H[H], o);
  const unsigned int*  hc_H = &hccc[hc_off_H[H]];
  const unsigned char* up   = up_int ? &up_int[o] : NULL;
  const int s_stride = S_STRIDE(length);
  const int carry_ok = negctl != 1 && j - 1 > i + 1 && (!up || (up[i + 1] >= 1 && up[j - 1] >= 1));
  unsigned long long nfresh = 0, ncarry = 0;

  const int u0 = LY_U0 + (int)blockIdx.y * PT;
#pragma unroll 5
  for (int t = 0; t < PT; t++) {
    const int u = u0 + t;
    Gc[(size_t)(u - LY_U0) * row_total + m] =
      ly_carry_u_frame(u, i, j, H, o, s_stride, carry_ok, cr, hc_H, up, P, pair_, S, row_total, Gp, Er, nfresh, ncarry);
  }
  if (blockIdx.y == 0) ly_entry_store(Er, negctl, i, j, H, o, s_stride, cr, hc_H, P, pair_, S, row_total);
  if (stat && nfresh) atomicAdd(&stat[3], nfresh);
  if (stat && ncarry) atomicAdd(&stat[4], ncarry);
}

/* Row i: the cells the dense kernel computes (its own enumeration), one thread each. */
template<class CREAD>
__global__ void
ly_check_kernel(const int nfiles, const int i, const int length,
                const cuda_param_t* __restrict__ P, const char* __restrict__ pair_,
                const unsigned int* __restrict__ S, const unsigned int* __restrict__ hccc,
                const unsigned char* __restrict__ up_int,
                const int* __restrict__ my_c, const int* __restrict__ c_ring, const size_t c_ring_stride,
                const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ hc_off_H, const size_t* __restrict__ size_off_H,
                const size_t row_total, const int* __restrict__ Gc,
                const int* __restrict__ energy_min,
                unsigned long long* __restrict__ stat, int* __restrict__ bad)
{
  const size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= size_off_H[nfiles]) return;
  const int    H = flatten_index_to_H(m, size_off_H, nfiles);
  const size_t local = m - size_off_H[H];
  const int    j = (int)local + i + turn + 1 + (int)size_off_H[nfiles + 1];   /* int_loop_cell.inc's j */
  const size_t o = row_off_H[H];
  const unsigned int* hc_H = &hccc[hc_off_H[H]];
  const int dense = energy_min[o + j];
  int ly = INF;
  unsigned long long ndirect = 0;

  if (Hc(Indx(i, j), hc_H)) {
    const CREAD cr = ly_reader((CREAD *)0, my_c, c_ring, c_ring_stride, tri_off_H[H], o);
    const int s_stride = S_STRIDE(length);
    const cell_inv_t ci = cell_invariants(S, pair_, H, s_stride, i, j);
    const unsigned char* up = up_int ? &up_int[o] : NULL;
    /* DIRECT: short side <= 1 (stacks, bulges, 1xn) and 2x2 / 2x3 / 3x2, through Energy() itself */
    for (int u1 = 0; u1 <= MAXLOOP; u1++)
      for (int u2 = 0; u1 + u2 <= MAXLOOP; u2++) {
        const int s = u1 < u2 ? u1 : u2, l_ = u1 > u2 ? u1 : u2;
        const int generic = (s >= 2) && !(s == 2 && l_ <= 3);
        if (generic) continue;
        const int p = i + u1 + 1, q = j - u2 - 1;
        if (q - p < 1) continue;
        if (!Hc(Indx(p, q), hc_H)) continue;
        const int e = Energy(H, s_stride, i, j, q, p, ci, cr, up, S, pair_, P,
                             P->TerminalAU, P->ninio2, P->bulge, P->internal_loop, P->lxc,
                             P->mismatchI, P->mismatch1nI, P->mismatch23I, P->stack,
                             P->int11, P->int21, P->int22);
        ndirect++;
        if (e < ly) ly = e;
      }
    /* the carried generic loops, used once per u; the outer pair's --noClosingGU gate */
    if (!(P->noGUclosure && (ci.type == 3 || ci.type == 4))) {
      const int outer = P->mismatchI[ci.type][ci.si1][ci.sj1];
      for (int u = LY_U0; u <= MAXLOOP; u++) {
        const int g = Gc[(size_t)(u - LY_U0) * row_total + (o + (size_t)j)];
        if (g >= INF) continue;
        const int e = g + P->internal_loop[u] + outer + P->SaltLoop[u + 2];
        if (e < ly) ly = e;
      }
    }
    atomicAdd(&stat[2], 1ULL);
  }
  atomicAdd(&stat[0], 1ULL);
  if (ndirect) atomicAdd(&stat[5], ndirect);
  const int same = (ly >= INF && dense >= INF) || ly == dense;
  if (!same) {
    const unsigned long long b = atomicAdd(&stat[1], 1ULL);
    if (b < LY_NBAD) { bad[5*b] = H; bad[5*b+1] = i; bad[5*b+2] = j; bad[5*b+3] = dense; bad[5*b+4] = ly; }
  }
}

/* One row of G, on the cell stream. stat: the selftest counters, or NULL. */
static void
ly_carry_row(const int nfiles, const int i, const int length, unsigned long long* stat)
{
  int* Gc = d_ly_G[i & 1];
  const int* Gp = d_ly_G[(i + 1) & 1];
  int* Er = d_ly_E;                                   /* S6b, or NULL */
  const size_t rt = g_row_total;
  const int nbr = (int)((rt + 127) / 128);
  if (ly_v2() && nfiles <= 65535 && nfiles == g_ly_n) {
    /* S6a': this row's active slots decide the split */
    long act = 0;
    for (int H = 0; H < g_ly_n; H++) if (g_ly_len[H] > i) act += g_ly_len[H] - i;
    if (act == 0) return;                      /* nothing to carry: no record reaches past i */
    const long cap = ly_capacity();
    int pt = ly_pt_forced();
    /* laptop sweep (10-07): PT 5 is best or tied at every width, 25 loses even on full rows
     * (-24 % vs v1 at 48x5601); 1 only where 5 cannot fill the device */
    if (!pt) pt = (act * 5 >= cap) ? 5 : 1;
    g_ly_rows_pt[pt == 25 ? 0 : pt == 5 ? 1 : 2]++;
    const int  w   = g_ly_lmax - i;                              /* widest record's columns past i */
    const int  bx  = w >= 128 ? 128 : ((w + 31) / 32) * 32;     /* narrow rows: no idle 128-blocks */
    const dim3 grid((unsigned)((w + bx - 1) / bx), (unsigned)(LY_NU / pt), (unsigned)nfiles);
#define LY_CARRY3K(...) \
    __VA_ARGS__<<<grid, bx, 0, rnafold_stream_cell()>>>(i, length, ly_negctl(), d_param, d_pair, \
        d_S, d_hccc, d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, rt, Gp, Gc, Er, stat)
#define LY_CARRY3(CR, PT_) do { if (ly_frame())          LY_CARRY3K(ly_carry3_frame_kernel<CR, PT_>);   \
                                     else if (Er && ly_lb()) LY_CARRY3K(ly_carry3_lb_kernel<CR, PT_>); \
                                     else if (Er)        LY_CARRY3K(ly_carry3_kernel<CR, PT_, 1>);     \
                                     else                LY_CARRY3K(ly_carry3_kernel<CR, PT_, 0>); } while (0)
    if (g_c_ring_on) { if (pt == 25) LY_CARRY3(c_ring_reader, 25); else if (pt == 5) LY_CARRY3(c_ring_reader, 5); else LY_CARRY3(c_ring_reader, 1); }
    else             { if (pt == 25) LY_CARRY3(c_tri_reader,  25); else if (pt == 5) LY_CARRY3(c_tri_reader,  5); else LY_CARRY3(c_tri_reader,  1); }
#undef LY_CARRY3
#undef LY_CARRY3K
  } else if (g_c_ring_on) {
    if (Er) ly_carry_kernel<c_ring_reader, 1><<<nbr, 128, 0, rnafold_stream_cell()>>>(nfiles, i, length, ly_negctl(), d_param, d_pair,
        d_S, d_hccc, d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, rt, Gp, Gc, Er, stat);
    else    ly_carry_kernel<c_ring_reader, 0><<<nbr, 128, 0, rnafold_stream_cell()>>>(nfiles, i, length, ly_negctl(), d_param, d_pair,
        d_S, d_hccc, d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, rt, Gp, Gc, Er, stat);
  } else {
    if (Er) ly_carry_kernel<c_tri_reader, 1><<<nbr, 128, 0, rnafold_stream_cell()>>>(nfiles, i, length, ly_negctl(), d_param, d_pair,
        d_S, d_hccc, d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, rt, Gp, Gc, Er, stat);
    else    ly_carry_kernel<c_tri_reader, 0><<<nbr, 128, 0, rnafold_stream_cell()>>>(nfiles, i, length, ly_negctl(), d_param, d_pair,
        d_S, d_hccc, d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, rt, Gp, Gc, Er, stat);
  }
  gpuErrchk( cudaPeekAtLastError() );
}

/* S2: int_loop_cuda() calls this BEFORE the warp kernel, so the LY cells read this row's G.
 * Same stream, so stream order is the only synchronisation needed. */
static void
ly_eval_row(const int nfiles, const int i, const int length)
{
  if (g_ly_on && g_ly_eval) ly_carry_row(nfiles, i, length, NULL);
}

static void
ly_selftest_row(const int nfiles, const int i, const int length)
{
  if (!g_ly_on || g_ly_eval) return;
  int* Gc = d_ly_G[i & 1];
  const size_t rt = g_row_total;
  ly_carry_row(nfiles, i, length, d_ly_stat);
  size_t cells = 0;
  gpuErrchk( cudaMemcpy(&cells, &d_size_off_H[nfiles], sizeof(size_t), cudaMemcpyDeviceToHost) );   /* selftest: a sync is fine */
  if (cells == 0) return;
  const int nbc = (int)((cells + 127) / 128);
  if (g_c_ring_on)
    ly_check_kernel<c_ring_reader><<<nbc, 128, 0, rnafold_stream_cell()>>>(nfiles, i, length, d_param, d_pair, d_S, d_hccc,
        d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, d_size_off_H, rt, Gc,
        d_energy_min2, d_ly_stat, d_ly_bad);
  else
    ly_check_kernel<c_tri_reader><<<nbc, 128, 0, rnafold_stream_cell()>>>(nfiles, i, length, d_param, d_pair, d_S, d_hccc,
        d_up_int, d_my_c, d_c_ring, g_row_total, d_tri_off_H, d_row_off_H, d_hc_off_H, d_size_off_H, rt, Gc,
        d_energy_min2, d_ly_stat, d_ly_bad);
  gpuErrchk( cudaPeekAtLastError() );
}

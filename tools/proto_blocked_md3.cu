/* proto_blocked_md3.cu -- the LEGAL blocked decomposition, and what it costs.
 *
 * WHY A THIRD PROTOTYPE. proto_blocked_md2.cu measures 7.66x, and that number is an
 * UPPER BOUND that no schedule can reach. Its tile (I,J) blocks the whole k range
 * [imax_I + TURN+1, jmax_J - TURN-2], which includes k blocks whose B operand
 * fML[k+1][j] has k+1 INSIDE the row block -- values that do not exist yet when the
 * tile runs. It gets away with it because it is handed a complete synthetic fML. The
 * real sweep is not.
 *
 * So this program measures the decomposition that IS legal, and the question it
 * answers is the one that decides whether the in-tree driver rewrite is worth doing:
 * after the corners are paid, what is left of the 7.66x?
 *
 * THE DECOMPOSITION. Row block I holds rows [imin, imax] (production sweeps i
 * DESCENDING, so imax is processed first); column block J holds columns
 * [j0, j0+CB-1], and the schedule is:
 *
 *     for I = row blocks, descending            # as today's i-loop, RB rows wide
 *       for J = I .. last, ASCENDING            # column blocks
 *         BULK(I,J)                             # once, all RB x CB cells at once
 *         for i in block I, descending          # the existing per-row kernels,
 *           physics(i, columns of J)            #   restricted to this column block
 *           CORNERS(i, columns of J)
 *
 * Splitting md's k range [i+TURN+1, j-TURN-2] at imax and j0 gives three pieces:
 *
 *   bulk  k in [imax,     j0-1 ]   A: row i, columns < j0   -> earlier COLUMN blocks
 *                                  B: rows  k+1 > imax      -> earlier BLOCK-ROWS
 *   cor1  k in [i+TURN+1, imax-1]  A: row i, near-diagonal  -> earlier column blocks
 *                                  B: rows  k+1 in [i+TURN+2, imax] -> INSIDE the
 *                                     block, so only after the rows above i
 *   cor2  k in [j0,       j-TURN-2] A: row i, THIS column block -> after i's fML scan
 *                                  B: rows  k+1 > j0 > imax -> earlier block-rows
 *
 * The three are contiguous and cover the range exactly, so min of the three is md.
 * Only the BULK is available for every row of the block up front, which is precisely
 * why only the bulk can be blocked -- and it is (d-1)/d of the work at block distance
 * d, so the corners are a few percent at CB = RB = 32..64 and a quarter of it at 512.
 * That trade-off against the launch count is what this program prices.
 *
 * WHAT IS FAITHFUL HERE AND WHAT IS NOT.
 *   faithful: the k ranges, the INF sentinel, int32 staging of decoded values, the
 *             MASKVAL trick, warp-per-cell corners with lanes striding k (md_cell's
 *             own shape, so the corner's per-element cost is comparable to the
 *             reference), and the triangle layout for B.
 *   NOT:      the per-row launch overhead. The corners are launched once per tile
 *             here, not once per (row, column block). That overhead is real but it is
 *             SHARED with the physics kernels the same restructure moves, so it
 *             belongs in the driver's accounting, not the kernel's. It is priced
 *             separately from the measured per-launch cost (~3 us, RNA_LAUNCH_STATS).
 *
 *   build: nvcc -O3 -arch=native -o proto3 proto_blocked_md3.cu
 *   run:   ./proto3 [n] [pct_inf]
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define TURN 3
#define INFV 10000000            /* md's INF, exactly */
#define INF16 ((short)32767)     /* this cell is INF (production's FML_INF16) */
#define MASKVAL (INFV + 1000000) /* out of range: must lose against INF. See md_block.inc */

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {                       \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__host__ __device__ __forceinline__ long long Indx(int i, int j)
{
  return (long long)j * (j - 1) / 2 + i;
}

__host__ __device__ __forceinline__ int decode(short o)
{
  return (o == INF16) ? INFV : (int)o;
}

/* ------------------------------------------------------ the streaming reference ---
 * md_cell's shape: one cell per TILE lanes, lanes stride y (= k), so the B reads walk
 * a contiguous run DOWN column j. This is what production does today.
 */
template <int TILE>
__global__ void md_stream_all(const int n, const long long ncell,
                              const int *__restrict__ ci, const int *__restrict__ cj,
                              const short *__restrict__ R, const short *__restrict__ T,
                              int *__restrict__ out)
{
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long m = g / TILE;
  const int lane = (int)(g & (TILE - 1));
  const bool active = (m < ncell);

  int v = INFV, i = 0, j = 0;
  if (active) {
    i = ci[m]; j = cj[m];
    const int x = j - i - 2 * TURN - 3;
    const long long ij0 = Indx(i, j) + TURN + 2;
    for (int y = lane; y <= x; y += TILE)
      v = min(v, decode(R[(long long)i * n + (i + TURN + 1 + y)]) + decode(T[ij0 + y]));
  }
#pragma unroll
  for (int off = TILE / 2; off > 0; off >>= 1)
    v = min(v, __shfl_down_sync(0xffffffffu, v, off, TILE));
  if (active && lane == 0) out[Indx(i, j)] = v;
}

/* ----------------------------------------------------------------- the BULK tile ---
 * k in [imax, j0-1] only. Everything it reads is final before the tile runs.
 *
 * RB is the row block and CB the column block, and they are independent because the
 * schedule wants them to be: RB sets the REUSE (each staged B element is used by RB
 * rows) and CB sets how often the physics has to be relaunched.
 *
 * KB is the k-depth of the staged patch and is DELIBERATELY independent of CB.
 *
 * Tying them (which the first version did) couples two knobs that answer different
 * questions: CB is how many columns a tile owns, and therefore how often the physics
 * kernels have to be relaunched, while KB is only the shared-memory footprint of one
 * staging round, 4*(KB*(RB+1) + KB*(CB+1)) bytes. Tied, CB=128 needs 99 KB and cannot
 * launch at all; decoupled, CB=128 with KB=32 needs 25 KB and the CB sweep -- the one
 * that prices the launch multiple -- becomes possible.
 */
template <int RB, int CB, int KB, int RM, int RN>
__global__ void md_bulk(const int n, const short *__restrict__ R,
                        const short *__restrict__ T, int *__restrict__ out)
{
  __shared__ int Xs[KB][RB + 1];      /* A: [t][row] */
  __shared__ int Ys[KB][CB + 1];      /* B: [t][col] */

  const int I = blockIdx.y, J = blockIdx.x;
  const int imin = I * RB + 1, imax = imin + RB - 1;
  const int j0   = J * CB + 1;
  if (j0 <= imax) return;             /* diagonal / overlapping: no bulk exists */

  const int TPB = (RB / RM) * (CB / RN);
  const int tid = threadIdx.x;
  const int cbase = (tid % (CB / RN)) * RN;
  const int rbase = (tid / (CB / RN)) * RM;

  int acc[RM][RN];
#pragma unroll
  for (int u = 0; u < RM; u++)
#pragma unroll
    for (int v = 0; v < RN; v++) acc[u][v] = INFV;

  /* The bulk range, walked in KB-deep staging rounds. */
  for (int k0 = imax; k0 <= j0 - 1; k0 += KB) {
    for (int e = tid; e < KB * RB; e += TPB) {
      const int t = e / RB, rr = e % RB;
      const int i = imin + rr, k = k0 + t;
      /* Out of the ARRAY, out of the RECURRENCE, or out of the BULK range: all three
       * mean "does not participate", and all three must LOSE against INF, not equal
       * it. Genuine INF is decoded as INF. See md_block.inc's header. */
      const bool live = (i <= n) && (k <= n) && (k >= i + TURN + 1) && (k <= j0 - 1);
      Xs[t][rr] = live ? decode(R[(long long)i * n + k]) : MASKVAL;
    }
    for (int e = tid; e < KB * CB; e += TPB) {
      const int t = e / CB, cc = e % CB;
      const int j = j0 + cc, kp1 = k0 + t + 1;
      const bool live = (j <= n) && (kp1 >= 1) && (kp1 <= j) && (kp1 <= j - TURN - 1);
      Ys[t][cc] = live ? decode(T[Indx(kp1, j)]) : MASKVAL;
    }
    __syncthreads();

#pragma unroll 4
    for (int t = 0; t < KB; t++) {
      int xa[RM], ya[RN];
#pragma unroll
      for (int u = 0; u < RM; u++) xa[u] = Xs[t][rbase + u];
#pragma unroll
      for (int v = 0; v < RN; v++) ya[v] = Ys[t][cbase + v];
#pragma unroll
      for (int u = 0; u < RM; u++)
#pragma unroll
        for (int v = 0; v < RN; v++) acc[u][v] = min(acc[u][v], xa[u] + ya[v]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int u = 0; u < RM; u++)
#pragma unroll
    for (int v = 0; v < RN; v++) {
      const int i = imin + rbase + u, j = j0 + cbase + v;
      if (i <= n && j <= n && j >= i + 2 * TURN + 3) out[Indx(i, j)] = acc[u][v];
    }
}

/* -------------------------------------------------------------- the two CORNERS ---
 * md_cell's shape (warp per cell, lanes striding k) over the two short ranges. No
 * staging: the data a corner touches is a CB x CB patch shared by the whole tile, so
 * it is L1/L2 resident by construction -- which is the difference between a short
 * stream and md's full-column stream, and the reason the corners are not simply
 * "the streaming kernel again".
 *
 * `combine` = true does min against what the bulk already wrote; false initialises.
 */
/* ONE THREAD PER CELL: no cross-lane reduction at all.
 *
 * The warp-per-cell shape below is md_cell's, and it is the WRONG shape for a short
 * range: with TILE=32 over ~64 elements each lane handles two, and then five shuffles
 * run to reduce them. The reduction, not the loads, is the cost. This variant gives
 * one thread the whole range -- worse coalescing (consecutive threads walk different
 * columns) but zero reduction, and the corner's working set is a CB x CB patch shared
 * by the tile, so it is L1-resident and the coalescing matters less than the issue
 * slots. Which effect wins is a measurement, not an argument.
 */
template <int RB, int CB>
__global__ void md_corners_thread(const int n, const short *__restrict__ R,
                                  const short *__restrict__ T, int *__restrict__ out)
{
  const int I = blockIdx.y, J = blockIdx.z;
  const int imin = I * RB + 1, imax = imin + RB - 1;
  const int j0   = J * CB + 1;

  const long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int rr = (int)(m / CB), cc = (int)(m % CB);
  if (rr >= RB) return;
  const int i = imin + rr, j = j0 + cc;
  if (!(i <= n && j <= n && j >= i + 2 * TURN + 3)) return;

  const int klo = i + TURN + 1, khi = j - TURN - 2;
  int v = INFV;
  for (int k = klo; k <= min(imax - 1, khi); k++)
    v = min(v, decode(R[(long long)i * n + k]) + decode(T[Indx(k + 1, j)]));
  for (int k = max(j0, klo); k <= khi; k++)
    v = min(v, decode(R[(long long)i * n + k]) + decode(T[Indx(k + 1, j)]));
  const long long o = Indx(i, j);
  out[o] = min(out[o], v);
}

template <int TILE, int RB, int CB>
__global__ void md_corners(const int n, const short *__restrict__ R,
                           const short *__restrict__ T, int *__restrict__ out,
                           const int combine)
{
  const int I = blockIdx.y, J = blockIdx.z;
  const int imin = I * RB + 1, imax = imin + RB - 1;
  const int j0   = J * CB + 1;

  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long m = g / TILE;
  const int lane = (int)(g & (TILE - 1));

  const int rr = (int)(m / CB), cc = (int)(m % CB);
  if (rr >= RB) return;
  const int i = imin + rr, j = j0 + cc;
  const bool active = (i <= n) && (j <= n) && (j >= i + 2 * TURN + 3);

  int v = INFV;
  if (active) {
    const int klo = i + TURN + 1, khi = j - TURN - 2;
    /* cor1: k in [klo, imax-1]   cor2: k in [j0, khi]
     * Written as two strided loops rather than one bounded loop so each keeps the
     * coalesced B walk down column j. */
    for (int k = klo + lane; k <= min(imax - 1, khi); k += TILE)
      v = min(v, decode(R[(long long)i * n + k]) + decode(T[Indx(k + 1, j)]));
    for (int k = max(j0, klo) + lane; k <= khi; k += TILE)
      v = min(v, decode(R[(long long)i * n + k]) + decode(T[Indx(k + 1, j)]));
  }
#pragma unroll
  for (int off = TILE / 2; off > 0; off >>= 1)
    v = min(v, __shfl_down_sync(0xffffffffu, v, off, TILE));
  if (active && lane == 0) {
    const long long o = Indx(i, j);
    out[o] = combine ? min(out[o], v) : v;
  }
}

static double ms_of(cudaEvent_t a, cudaEvent_t b)
{ float t = 0.f; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; }

/* ------------------------------------------------------------------------ driver ---*/
static int    g_n;
static short *g_R, *g_T;
static int   *g_out;
static std::vector<int> g_ref;
/* the per-row reference, kept live so every arm can be timed against a reference
 * measured at the SAME clock -- see time_ref() */
static int   *g_rci, *g_rcj;
static std::vector<long long> g_roff;
static size_t g_ntri;

/* One full per-row reference pass. THIS LAPTOP IS POWER-CAPPED and its SM clock moves
 * by 2x between runs; the streaming reference is DRAM-bound and the blocked kernels are
 * SM-bound, so their RATIO moves with it. Measured across three runs of this program
 * the same arm reported 3.10x, 5.37x and 19.3x. A reference timed once at the top is
 * therefore worthless: each arm gets its own, immediately before and after. */
static double time_ref(void)
{
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaMemset(g_out, 0x7f, sizeof(int) * g_ntri));
  CK(cudaEventRecord(a));
  for (int i = 1; i <= g_n; i++) {
    const long long cnt = g_roff[i + 1] - g_roff[i];
    if (cnt <= 0) continue;
    md_stream_all<32><<<(int)((cnt * 32 + 255) / 256), 256>>>(
        g_n, cnt, g_rci + g_roff[i], g_rcj + g_roff[i], g_R, g_T, g_out);
  }
  CK(cudaEventRecord(b));
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  const double ms = ms_of(a, b);
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms;
}

/* TILE = 0 selects the one-thread-per-cell corner. */
template <int RB, int CB, int KB, int RM, int RN, int TILE>
static void run_decomp(const char *tag, double ref_ms)
{
  const int nblk_r = (g_n + RB - 1) / RB, nblk_c = (g_n + CB - 1) / CB;
  const int TPB    = (RB / RM) * (CB / RN);
  const int cthreads = 128;
  const long long cells = (long long)RB * CB;
  const int cblocks = (int)((cells * (TILE ? TILE : 1) + cthreads - 1) / cthreads);

  cudaEvent_t e0, e1, e2;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));

  const double ref_a = time_ref();          /* ABA around the arm: the clock moves */

  double bulk_ms = 0, corn_ms = 0;
  for (int rep = 0; rep < 3; rep++) {
    CK(cudaMemset(g_out, 0x7f, sizeof(int) * (size_t)Indx(0, g_n + 1)));
    CK(cudaEventRecord(e0));
    md_bulk<RB, CB, KB, RM, RN><<<dim3(nblk_c, nblk_r), TPB>>>(g_n, g_R, g_T, g_out);
    /* A LAUNCH THAT FAILS COSTS 0.00 ms AND LOOKS LIKE A WIN. RB256 CB256 8x8 needs
     * 1024 threads x 64+ registers, over the 65536-per-block limit, and the first
     * version of this harness reported it as "0.00 ms, 33.58x, MISMATCH" instead of
     * as a failed launch. Check every launch. */
    { cudaError_t le = cudaGetLastError();
      if (le != cudaSuccess) {
        printf("  %-24s  LAUNCH FAILED: %s (shared %.1f KB, %d threads)\n", tag,
               cudaGetErrorString(le),
               4.0 * (KB * (RB + 1) + KB * (CB + 1)) / 1024.0, TPB);
        return;
      } }
    CK(cudaEventRecord(e1));
    /* combine=0 on the tiles the bulk skipped, 1 where it wrote: handled by giving
     * the corner kernel the same grid and letting it min against a 0x7f7f7f7f
     * initialiser, which is larger than any sum we produce. */
    if (TILE)
      md_corners<TILE ? TILE : 1, RB, CB><<<dim3(cblocks, nblk_r, nblk_c), cthreads>>>(g_n, g_R, g_T, g_out, 1);
    else
      md_corners_thread<RB, CB><<<dim3(cblocks, nblk_r, nblk_c), cthreads>>>(g_n, g_R, g_T, g_out);
    CK(cudaEventRecord(e2));
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    if (rep) { bulk_ms += ms_of(e0, e1); corn_ms += ms_of(e1, e2); }
  }
  bulk_ms /= 2; corn_ms /= 2;
  const double ref_b = time_ref();
  const double ref_ms_local = 0.5 * (ref_a + ref_b);

  std::vector<int> got(g_ref.size());
  CK(cudaMemcpy(got.data(), g_out, sizeof(int) * got.size(), cudaMemcpyDeviceToHost));
  long long bad = 0; int fi = -1, fj = -1;
  for (int j = 1; j <= g_n; j++)
    for (int i = 1; i <= j - 2 * TURN - 3; i++) {
      const long long o = Indx(i, j);
      if (got[o] != g_ref[o]) { if (!bad) { fi = i; fj = j; } bad++; }
    }

  const double tot = bulk_ms + corn_ms;
  (void)ref_ms;
  printf("  %-24s %8.2f %8.2f %8.2f %7.0f %8.2fx  %s", tag, bulk_ms, corn_ms, tot,
         ref_ms_local, ref_ms_local / tot, bad ? "MISMATCH" : "exact");
  if (bad) printf(" (%lld cells, first (%d,%d) got %d want %d)", bad, fi, fj,
                  got[Indx(fi, fj)], g_ref[Indx(fi, fj)]);
  printf("\n");
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaEventDestroy(e2));
}

int main(int argc, char **argv)
{
  const int n       = (argc > 1) ? atoi(argv[1]) : 4096;
  const int pct_inf = (argc > 2) ? atoi(argv[2]) : 15;
  g_n = n;

  const size_t ntri = (size_t)Indx(0, n + 1);
  g_ntri = ntri;
  printf("proto_blocked_md3: the LEGAL decomposition, n=%d, %d%% INF, triangle %.1f MB (int16)\n",
         n, pct_inf, ntri * 2 / 1048576.0);

  /* Synthetic fML: a triangle T (production's layout, a column is contiguous) and a
   * row-major mirror R of the same values, which is what the A operand needs. */
  std::vector<short> T(ntri), R((size_t)n * (n + 1));
  srand(1234);
  for (int j = 1; j <= n; j++)
    for (int i = 1; i <= j; i++) {
      const bool inf = (rand() % 100) < pct_inf;
      const short v = inf ? INF16 : (short)(-2000 + (rand() % 2200));
      T[Indx(i, j)] = v;
      R[(size_t)i * n + j] = v;
    }

  CK(cudaMalloc(&g_T, sizeof(short) * ntri));
  CK(cudaMalloc(&g_R, sizeof(short) * R.size()));
  CK(cudaMalloc(&g_out, sizeof(int) * ntri));
  CK(cudaMemcpy(g_T, T.data(), sizeof(short) * ntri, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g_R, R.data(), sizeof(short) * R.size(), cudaMemcpyHostToDevice));

  /* THE REFERENCE, THREE WAYS, because the ordering of the cell list turns out to
   * matter enormously and proto_blocked_md2.cu's 7.66x was measured against exactly
   * one of them.
   *
   *   i-major   one launch, cells ordered (i, then j). What md2 used.
   *   j-major   one launch, cells ordered (j, then i).
   *   per-row   ONE LAUNCH PER ROW i, which is what production actually does, and
   *             therefore the only honest baseline for an end-to-end claim.
   *
   * A blocked kernel compared against the wrong one of these is not comparable to
   * anything that ships. */
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  double ref_ms = 0;

  for (int mode = 0; mode < 3; mode++) {
    std::vector<int> ci, cj;
    if (mode == 0)
      for (int i = 1; i <= n; i++)
        for (int j = i + 2 * TURN + 3; j <= n; j++) { ci.push_back(i); cj.push_back(j); }
    else
      for (int j = 1; j <= n; j++)
        for (int i = 1; i <= j - 2 * TURN - 3; i++) { ci.push_back(i); cj.push_back(j); }
    const long long ncell = (long long)ci.size();
    int *d_ci, *d_cj;
    CK(cudaMalloc(&d_ci, sizeof(int) * ncell)); CK(cudaMalloc(&d_cj, sizeof(int) * ncell));
    CK(cudaMemcpy(d_ci, ci.data(), sizeof(int) * ncell, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cj, cj.data(), sizeof(int) * ncell, cudaMemcpyHostToDevice));

    /* row offsets into the i-major list, so mode 2 can launch one row at a time */
    std::vector<long long> roff(n + 2, 0);
    if (mode == 2) {
      long long at = 0;
      for (int i = 1; i <= n; i++) {
        roff[i] = at;
        const long long cnt = (n >= i + 2 * TURN + 3) ? (n - (i + 2 * TURN + 3) + 1) : 0;
        at += cnt;
      }
      roff[n + 1] = at;
    }

    double ms = 0;
    for (int rep = 0; rep < 3; rep++) {
      CK(cudaMemset(g_out, 0x7f, sizeof(int) * ntri));
      CK(cudaEventRecord(a));
      if (mode != 2) {
        md_stream_all<32><<<(int)((ncell * 32 + 255) / 256), 256>>>(n, ncell, d_ci, d_cj, g_R, g_T, g_out);
      } else {
        for (int i = 1; i <= n; i++) {
          const long long cnt = roff[i + 1] - roff[i];
          if (cnt <= 0) continue;
          md_stream_all<32><<<(int)((cnt * 32 + 255) / 256), 256>>>(
              n, cnt, d_ci + roff[i], d_cj + roff[i], g_R, g_T, g_out);
        }
      }
      CK(cudaEventRecord(b));
      CK(cudaDeviceSynchronize());
      if (rep) ms += ms_of(a, b);
    }
    ms /= 2;
    CK(cudaGetLastError());

    if (mode == 0) {                       /* md2's ordering defines the reference data */
      g_ref.resize(ntri);
      CK(cudaMemcpy(g_ref.data(), g_out, sizeof(int) * ntri, cudaMemcpyDeviceToHost));
    } else {                               /* and the others must agree with it */
      std::vector<int> got(ntri);
      CK(cudaMemcpy(got.data(), g_out, sizeof(int) * ntri, cudaMemcpyDeviceToHost));
      long long bad = 0;
      for (int j = 1; j <= n; j++)
        for (int i = 1; i <= j - 2 * TURN - 3; i++)
          if (got[Indx(i, j)] != g_ref[Indx(i, j)]) bad++;
      if (bad) printf("  !! ordering %d disagrees with i-major in %lld cells\n", mode, bad);
    }
    printf("  %-22s %8s %8s %8.2f %8s   %s\n", mode == 0 ? "streaming i-major"
           : mode == 1 ? "streaming j-major" : "streaming PER ROW",
           "", "", ms, mode == 2 ? "<= prod" : "",
           mode == 0 ? "md2's baseline" : mode == 1 ? "(md3's first baseline)"
                                                    : "what production launches");
    if (mode == 2) {
      ref_ms = ms;                         /* compare against production's shape */
      g_rci = d_ci; g_rcj = d_cj; g_roff = roff;   /* kept: time_ref() re-runs it */
    } else {
      CK(cudaFree(d_ci)); CK(cudaFree(d_cj));
    }
  }
  printf("\n  Each arm below is timed between TWO per-row reference passes and the\n"
         "  speedup uses their mean, because this laptop's clock moves 2x between\n"
         "  runs and the two kernels have different limiters. `ref` is that mean.\n\n");

  printf("  %-24s %8s %8s %8s %9s\n", "arm", "bulk ms", "corner", "total", "vs ref");
  /* The corner's SHAPE first, at one (RB,CB), because it is 53% of the time in the
   * first run and md_cell's warp-per-cell shape is the wrong one for a short range. */
  run_decomp<64, 64, 32, 4, 4, 32>("RB64 CB64 4x4 cor32", ref_ms);
  run_decomp<64, 64, 32, 4, 4, 16>("RB64 CB64 4x4 cor16", ref_ms);
  run_decomp<64, 64, 32, 4, 4, 8>("RB64 CB64 4x4 cor8 ", ref_ms);
  run_decomp<64, 64, 32, 4, 4, 4>("RB64 CB64 4x4 cor4 ", ref_ms);
  run_decomp<64, 64, 32, 4, 4, 2>("RB64 CB64 4x4 cor2 ", ref_ms);
  run_decomp<64, 64, 32, 4, 4, 0>("RB64 CB64 4x4 cor1T", ref_ms);
  printf("\n");
  /* then RB (the bulk's REUSE, want large), CB (how often the physics relaunches,
   * want large) against the corner, which grows as RB+CB (want both small). The
   * optimum of that tension is the whole question. */
  run_decomp<32,  32, 32, 4, 4, 4>("RB32  CB32  KB32 4x4", ref_ms);
  run_decomp<64,  32, 32, 4, 4, 4>("RB64  CB32  KB32 4x4", ref_ms);
  run_decomp<64,  64, 64, 4, 4, 4>("RB64  CB64  KB64 4x4", ref_ms);
  run_decomp<64, 128, 32, 4, 4, 4>("RB64  CB128 KB32 4x4", ref_ms);
  run_decomp<128, 64, 32, 4, 4, 4>("RB128 CB64  KB32 4x4", ref_ms);
  run_decomp<128,128, 32, 4, 4, 4>("RB128 CB128 KB32 4x4", ref_ms);
  run_decomp<128,128, 32, 4, 4, 8>("RB128 CB128 KB32 cor8", ref_ms);
  run_decomp<128,256, 16, 8, 8, 8>("RB128 CB256 KB16 8x8", ref_ms);
  run_decomp<256,256, 16, 8, 8, 8>("RB256 CB256 KB16 8x8", ref_ms);
  run_decomp<64,  64, 32, 8, 8, 4>("RB64  CB64  KB32 8x8", ref_ms);
  return 0;
}

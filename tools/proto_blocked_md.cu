/* proto_blocked_md.cu -- does blocking the (min,+) product actually pay?
 *
 * WHAT THIS ISOLATES. modular_decomposition computes
 *
 *     fM2[i][j] = min over k in [i+turn+1, j-turn-2] of ( fML[i][k] + fML[k+1][j] )
 *
 * which is a (min,+) INNER PRODUCT, and over a tile of the output it is a (min,+)
 * MATRIX PRODUCT. The production kernel evaluates it one output cell per warp,
 * streaming a contiguous run down column j. That means:
 *
 *   - fML[i][k] comes from a contiguous row buffer, re-read by every j, so cached;
 *   - fML[k+1][j] is pure streaming with NO intra-row reuse -- per row, the union of
 *     those reads is the whole sub-triangle, each element read exactly once.
 *
 * So the loop is 2 ops per 2 bytes at int16 = 1.0 ops/byte, against a machine
 * balance of ~13 on an A100 and ~20 on this laptop. Being at the DRAM roofline is
 * what this loop IS, not a tuning failure -- and no amount of caching a stream with
 * no near-term reuse can change it. Blocking CREATES the reuse instead: two b x b
 * blocks in shared memory give b^3 work for 2b^2 loads, i.e. b/2 times the
 * intensity, so traffic falls by b/2 (16x at b=32).
 *
 * This program measures that claim on real hardware, on identical data, with a
 * bit-exactness check between the two. It does NOT integrate with the fold: fML is
 * a given matrix here, which is the point -- it isolates the n^3/6 term from the
 * recurrence that produces it.
 *
 * ON LEGALITY, because it is the first thing anyone will ask. In the real sweep
 * md(i) -> c(i-1) -> fML(i-1) -> md(i-1), so md's ROWS are strictly ordered and the
 * production schedule (whole row at a time) is forced by that. Tiles are not: for
 * output tile (I,J), every k-block strictly between I and J refers to fML blocks at
 * block-distance < J-I, so ordering tiles by anti-diagonal d = J-I makes them all
 * complete. Only the k values inside blocks I and J need the sequential treatment,
 * and for d >> 1 that is a vanishing share of the work. Blocked Zuker is legal; the
 * current row-at-a-time order is just the most traffic-heavy legal schedule.
 *
 *   build: nvcc -O3 -arch=native -o proto_blocked_md proto_blocked_md.cu
 *   run:   ./proto_blocked_md [n]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#define TURN 3
#define INF16 ((short)0x3f00)      /* a large-but-safe int16 sentinel */
#define BIG   (1 << 28)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {                  \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

/* Production's triangle layout: a COLUMN is contiguous, which is what makes the
 * streaming read coalesced. Kept identical here so the comparison is fair. */
__host__ __device__ __forceinline__ long long Indx(int i, int j)
{
  return (long long)j * (j - 1) / 2 + i;
}

/* ---------------------------------------------------------------- reference ---
 * One output cell per TILE lanes, lanes striding the reduction, shuffle-reduced --
 * the shape of md_cell(). Launched once per row i, as production does.
 */
template <int TILE>
__global__ void md_stream_row(const int n, const int i,
                              const short *__restrict__ R,   /* row-major fML  */
                              const short *__restrict__ T,   /* triangle  fML  */
                              int *__restrict__ out)
{
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long m = g / TILE;
  const int lane = (int)(g & (TILE - 1));

  const int jlo = i + 2 * (TURN + 1) + 1;
  const int ncell = (n - jlo + 1) > 0 ? (n - jlo + 1) : 0;
  const bool active = (m < ncell);

  int v = BIG;
  int j = 0;
  if (active) {
    j = (int)m + jlo;
    const int x = j - i - 2 * TURN - 3;          /* y in [0,x] */
    const long long ij0 = Indx(i, j) + TURN + 2;
    for (int y = lane; y <= x; y += TILE) {
      const int a = R[(long long)i * n + (i + TURN + 1 + y)];
      const int b = T[ij0 + y];
      v = min(v, a + b);
    }
  }
#pragma unroll
  for (int off = TILE / 2; off > 0; off >>= 1)
    v = min(v, __shfl_down_sync(0xffffffffu, v, off, TILE));

  if (active && lane == 0) out[Indx(i, j)] = v;
}

/* The same loop, but every (i,j) in ONE launch. Not a legal schedule for the real
 * sweep -- it is here to separate the cost of the LOOP from the cost of 4096
 * launches, so the blocked kernel is not credited with removing launch overhead it
 * did not remove. */
template <int TILE>
__global__ void md_stream_all(const int n, const long long ncell_tot,
                              const int *__restrict__ cell_i,
                              const int *__restrict__ cell_j,
                              const short *__restrict__ R,
                              const short *__restrict__ T,
                              int *__restrict__ out)
{
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long m = g / TILE;
  const int lane = (int)(g & (TILE - 1));
  const bool active = (m < ncell_tot);

  int v = BIG, i = 0, j = 0;
  if (active) {
    i = cell_i[m]; j = cell_j[m];
    const int x = j - i - 2 * TURN - 3;
    const long long ij0 = Indx(i, j) + TURN + 2;
    for (int y = lane; y <= x; y += TILE) {
      const int a = R[(long long)i * n + (i + TURN + 1 + y)];
      const int b = T[ij0 + y];
      v = min(v, a + b);
    }
  }
#pragma unroll
  for (int off = TILE / 2; off > 0; off >>= 1)
    v = min(v, __shfl_down_sync(0xffffffffu, v, off, TILE));

  if (active && lane == 0) out[Indx(i, j)] = v;
}

/* ------------------------------------------------------------------ blocked ---
 * One B x B output tile per block. For each k-block: stage fML[i][k] and
 * fML[k+1][j] in shared memory, then do the (min,+) product out of shared.
 *
 * Shared layout is chosen so BOTH stages load coalesced and BOTH are read with the
 * reduction index innermost:
 *   Xs[t][r] = fML[i0+r][k0+t]     (row-major source, so r is contiguous per t)
 *   Ys[t][c] = fML[k0+t+1][j0+c]   (triangle source, so t is contiguous per c)
 * The product then reads Xs[t][r] and Ys[t][c] with t as the loop variable, which
 * is a broadcast in r and a broadcast in c -- no bank conflicts either way.
 *
 * RPT results per thread, so a B x B tile needs B*B/RPT threads.
 */
template <int B, int RPT>
__global__ void md_blocked(const int n, const int ntb,
                           const short *__restrict__ R,
                           const short *__restrict__ T,
                           int *__restrict__ out)
{
  __shared__ short Xs[B][B + 1];
  __shared__ short Ys[B][B + 1];

  /* which output tile: a flat index over the upper-triangular tile set */
  const int tile = blockIdx.x;
  int I = blockIdx.y, J = blockIdx.x;
  (void)tile; (void)ntb;
  if (J < I) return;

  const int i0 = I * B + 1;          /* 1-based, like the triangle */
  const int j0 = J * B + 1;

  int acc[RPT];
#pragma unroll
  for (int u = 0; u < RPT; u++) acc[u] = BIG;

  /* this thread's RPT output cells: a column-strip of the tile */
  const int tid = threadIdx.x;
  const int c = tid % B;
  const int rbase = (tid / B) * RPT;

  /* k ranges over [i+TURN+1, j-TURN-2] per cell; the union over the tile is what
   * decides which k-blocks are worth loading at all. */
  const int kmin_tile = i0 + TURN + 1;
  const int kmax_tile = (j0 + B - 1) - TURN - 2;

  for (int k0 = (kmin_tile / B) * B; k0 <= kmax_tile; k0 += B) {
    /* ---- stage both blocks. B*B/RPT threads, B*B elements each. */
    for (int e = tid; e < B * B; e += (B * B / RPT)) {
      const int t = e / B, rr = e % B;
      const int i = i0 + rr, k = k0 + t;
      Xs[t][rr] = (i <= n && k >= 1 && k <= n && i <= k)
                ? R[(long long)i * n + k] : INF16;
    }
    for (int e = tid; e < B * B; e += (B * B / RPT)) {
      const int t = e / B, cc = e % B;
      const int j = j0 + cc, kp1 = k0 + t + 1;
      Ys[t][cc] = (j <= n && kp1 >= 1 && kp1 <= j) ? T[Indx(kp1, j)] : INF16;
    }
    __syncthreads();

    /* ---- the (min,+) product, entirely out of shared memory */
    const int j = j0 + c;
#pragma unroll
    for (int u = 0; u < RPT; u++) {
      const int r = rbase + u;
      const int i = i0 + r;
      const int klo = i + TURN + 1, khi = j - TURN - 2;
      int a = acc[u];
#pragma unroll 8
      for (int t = 0; t < B; t++) {
        const int k = k0 + t;
        if (k >= klo && k <= khi) {
          const int s = (int)Xs[t][r] + (int)Ys[t][c];
          a = min(a, s);
        }
      }
      acc[u] = a;
    }
    __syncthreads();
  }

  const int j = j0 + c;
#pragma unroll
  for (int u = 0; u < RPT; u++) {
    const int i = i0 + rbase + u;
    if (i <= n && j <= n && j >= i + 2 * TURN + 3)
      out[Indx(i, j)] = acc[u];
  }
}


/* ------------------------------------------------ blocked + register tiling ---
 * The first blocked kernel converted DRAM traffic into SHARED traffic almost
 * exactly 1:1 -- measured on this device at n=4096: DRAM 22.77 -> 1.49 GB (15.3x
 * less) but shared wavefronts 5.0e7 -> 7.8e8 (15.5x more), with the instruction
 * count UNCHANGED. So it stopped being DRAM-bound and became issue/shared-bound,
 * which is progress but not the win.
 *
 * The cause is arithmetic: one (min,+) step costs two shared loads, an add and a
 * min -- 2 useful ops for 4 instructions. Register tiling fixes exactly that. A
 * thread that owns an RM x RN sub-tile loads RM + RN values from shared and does
 * RM * RN steps, so shared accesses per step fall from 2 to (RM+RN)/(RM*RN):
 *
 *      RM=RN=1  ->  2.00 accesses/step   (the kernel above)
 *      RM=RN=2  ->  1.00
 *      RM=RN=4  ->  0.50
 *      RM=RN=8  ->  0.25
 *
 * SECOND TRICK, and it is free: the k range [i+TURN+1, j-TURN-2] is masked into
 * the STAGING rather than tested in the inner loop. X's half of the condition
 * depends only on (r,t) and Y's only on (c,t), so both can be applied while
 * writing shared memory. The inner loop then has no branch at all and unrolls
 * fully. It stays exact because a masked entry contributes INF16 plus a real
 * value, which is >= 12128 while every legitimate sum is <= 4000.
 */
template <int B, int RM, int RN>
__global__ void md_blocked_reg(const int n,
                               const short *__restrict__ R,
                               const short *__restrict__ T,
                               int *__restrict__ out)
{
  __shared__ short Xs[B][B + 1];
  __shared__ short Ys[B][B + 1];

  const int I = blockIdx.y, J = blockIdx.x;
  if (J < I) return;

  const int i0 = I * B + 1;
  const int j0 = J * B + 1;

  const int TPB = (B / RM) * (B / RN);
  const int tid = threadIdx.x;
  const int cbase = (tid % (B / RN)) * RN;
  const int rbase = (tid / (B / RN)) * RM;

  int acc[RM][RN];
#pragma unroll
  for (int u = 0; u < RM; u++)
#pragma unroll
    for (int v = 0; v < RN; v++) acc[u][v] = BIG;

  const int kmin_tile = i0 + TURN + 1;
  const int kmax_tile = (j0 + B - 1) - TURN - 2;

  for (int k0 = (kmin_tile / B) * B; k0 <= kmax_tile; k0 += B) {
    /* ---- stage, with the recurrence range baked in */
    for (int e = tid; e < B * B; e += TPB) {
      const int t = e / B, rr = e % B;
      const int i = i0 + rr, k = k0 + t;
      const bool ok = (i <= n) && (k >= 1) && (k <= n) && (i <= k)
                   && (k >= i + TURN + 1);
      Xs[t][rr] = ok ? R[(long long)i * n + k] : INF16;
    }
    for (int e = tid; e < B * B; e += TPB) {
      const int t = e / B, cc = e % B;
      const int j = j0 + cc, kp1 = k0 + t + 1;
      const bool ok = (j <= n) && (kp1 >= 1) && (kp1 <= j)
                   && (kp1 <= j - TURN - 1);
      Ys[t][cc] = ok ? T[Indx(kp1, j)] : INF16;
    }
    __syncthreads();

    /* ---- RM+RN shared loads, RM*RN (min,+) steps, no branches */
#pragma unroll 4
    for (int t = 0; t < B; t++) {
      int xa[RM], ya[RN];
#pragma unroll
      for (int u = 0; u < RM; u++) xa[u] = (int)Xs[t][rbase + u];
#pragma unroll
      for (int v = 0; v < RN; v++) ya[v] = (int)Ys[t][cbase + v];
#pragma unroll
      for (int u = 0; u < RM; u++)
#pragma unroll
        for (int v = 0; v < RN; v++)
          acc[u][v] = min(acc[u][v], xa[u] + ya[v]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int u = 0; u < RM; u++)
#pragma unroll
    for (int v = 0; v < RN; v++) {
      const int i = i0 + rbase + u, j = j0 + cbase + v;
      if (i <= n && j <= n && j >= i + 2 * TURN + 3)
        out[Indx(i, j)] = acc[u][v];
    }
}

/* ------------------------------------------------------------------- driver ---*/
static double ms_of(cudaEvent_t a, cudaEvent_t b)
{
  float t = 0.f; CK(cudaEventElapsedTime(&t, a, b)); return (double)t;
}

int main(int argc, char **argv)
{
  const int n = (argc > 1) ? atoi(argv[1]) : 2048;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));

  const long long tri = (long long)n * (n + 1) / 2 + 1;
  const long long rowm = (long long)(n + 1) * n;

  printf("%s: L2 %.2f MB, %d SMs, %d KB shared/SM, %d threads/SM\n",
         p.name, p.l2CacheSize / 1048576.0, p.multiProcessorCount,
         (int)(p.sharedMemPerMultiprocessor / 1024), p.maxThreadsPerMultiProcessor);
  printf("n = %d   triangle %lld cells (%.1f MB int16)   row-major %.1f MB\n",
         n, tri, tri * 2 / 1048576.0, rowm * 2 / 1048576.0);

  /* total (min,+) steps, which is the work both kernels must do */
  double steps = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i + 2 * TURN + 3; j <= n; j++)
      steps += (j - i - 2 * TURN - 3) + 1;
  printf("(min,+) steps = %.4e   streaming traffic at 2 B each = %.2f GB\n",
         steps, steps * 2 / 1e9);

  /* ---- synthetic fML. Values in a plausible energy range so nothing overflows. */
  std::vector<short> hT(tri, INF16), hR(rowm, INF16);
  srand(20260927);
  for (int j = 1; j <= n; j++)
    for (int i = 1; i <= j; i++) {
      const short v = (short)((rand() % 4000) - 2000);
      hT[Indx(i, j)] = v;
      hR[(long long)i * n + j] = v;
    }

  short *dT, *dR; int *dOutA, *dOutB;
  CK(cudaMalloc(&dT, tri * sizeof(short)));
  CK(cudaMalloc(&dR, rowm * sizeof(short)));
  CK(cudaMalloc(&dOutA, tri * sizeof(int)));
  CK(cudaMalloc(&dOutB, tri * sizeof(int)));
  CK(cudaMemcpy(dT, hT.data(), tri * sizeof(short), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dR, hR.data(), rowm * sizeof(short), cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const int TILE = 32, TPB = 256;

  /* ---- A: streaming, one launch per row (production's schedule) */
  CK(cudaMemset(dOutA, 0x3f, tri * sizeof(int)));
  CK(cudaEventRecord(e0));
  for (int i = 1; i <= n; i++) {
    const int jlo = i + 2 * (TURN + 1) + 1;
    const int ncell = (n - jlo + 1);
    if (ncell <= 0) continue;
    const long long thr = (long long)ncell * TILE;
    md_stream_row<32><<<(int)((thr + TPB - 1) / TPB), TPB>>>(n, i, dR, dT, dOutA);
  }
  CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
  const double tA = ms_of(e0, e1);
  CK(cudaGetLastError());

  /* ---- B: streaming, ONE launch (isolates the loop from launch overhead) */
  std::vector<int> ci, cj;
  for (int i = 1; i <= n; i++)
    for (int j = i + 2 * TURN + 3; j <= n; j++) { ci.push_back(i); cj.push_back(j); }
  const long long ncell_tot = (long long)ci.size();
  int *dci, *dcj;
  CK(cudaMalloc(&dci, ncell_tot * sizeof(int)));
  CK(cudaMalloc(&dcj, ncell_tot * sizeof(int)));
  CK(cudaMemcpy(dci, ci.data(), ncell_tot * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcj, cj.data(), ncell_tot * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaEventRecord(e0));
  {
    const long long thr = ncell_tot * TILE;
    md_stream_all<32><<<(int)((thr + TPB - 1) / TPB), TPB>>>(n, ncell_tot, dci, dcj,
                                                             dR, dT, dOutA);
  }
  CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
  const double tB = ms_of(e0, e1);
  CK(cudaGetLastError());

  printf("\n  %-34s %12s %12s %14s\n", "kernel", "ms", "GB/s eff", "Gops/s");
  printf("  %-34s %12.2f %12.1f %14.2f\n", "streaming, one launch per row",
         tA, steps * 2 / (tA * 1e6), steps * 2 / (tA * 1e6));
  printf("  %-34s %12.2f %12.1f %14.2f\n", "streaming, one launch total",
         tB, steps * 2 / (tB * 1e6), steps * 2 / (tB * 1e6));

  /* ---- C: blocked, several tile sizes */
  struct Res { int b, rpt; double ms; bool ok; };
  std::vector<Res> rs;
  std::vector<int> hA(tri), hB(tri);
  CK(cudaMemcpy(hA.data(), dOutA, tri * sizeof(int), cudaMemcpyDeviceToHost));

#define RUN_BLOCKED(BB, RR)                                                        \
  do {                                                                             \
    const int ntb = (n + (BB) - 1) / (BB);                                         \
    dim3 grid(ntb, ntb);                                                           \
    CK(cudaMemset(dOutB, 0x3f, tri * sizeof(int)));                                \
    CK(cudaEventRecord(e0));                                                       \
    md_blocked<BB, RR><<<grid, (BB) * (BB) / (RR)>>>(n, ntb, dR, dT, dOutB);       \
    CK(cudaEventRecord(e1));                                                       \
    cudaError_t ee = cudaDeviceSynchronize();                                      \
    double ms = -1; bool ok = false;                                               \
    if (ee == cudaSuccess) {                                                       \
      ms = ms_of(e0, e1);                                                          \
      CK(cudaMemcpy(hB.data(), dOutB, tri * sizeof(int), cudaMemcpyDeviceToHost)); \
      long long bad = 0; long long first = -1;                                     \
      for (int j = 1; j <= n; j++)                                                 \
        for (int i = 1; i <= j; i++) {                                             \
          if (j < i + 2 * TURN + 3) continue;                                      \
          const long long q = Indx(i, j);                                          \
          if (hA[q] != hB[q]) { if (first < 0) first = q; bad++; }                 \
        }                                                                          \
      ok = (bad == 0);                                                             \
      if (!ok) printf("    b=%-3d rpt=%-2d MISMATCH in %lld cells (first at %lld)\n",\
                      BB, RR, bad, first);                                          \
    } else {                                                                        \
      printf("    b=%-3d rpt=%-2d launch failed: %s\n", BB, RR,                     \
             cudaGetErrorString(ee));                                               \
    }                                                                               \
    rs.push_back({BB, RR, ms, ok});                                                 \
  } while (0)

  RUN_BLOCKED(16, 1);
  RUN_BLOCKED(32, 1);
  RUN_BLOCKED(32, 4);
  RUN_BLOCKED(64, 4);

  struct ResR { int b, rm, rn; double ms; bool ok; };
  std::vector<ResR> rsr;

#define RUN_REG(BB, RM, RN)                                                       \
  do {                                                                            \
    const int ntb = (n + (BB) - 1) / (BB);                                        \
    dim3 grid(ntb, ntb);                                                          \
    CK(cudaMemset(dOutB, 0x3f, tri * sizeof(int)));                               \
    CK(cudaEventRecord(e0));                                                      \
    md_blocked_reg<BB, RM, RN><<<grid, ((BB)/(RM))*((BB)/(RN))>>>(n, dR, dT, dOutB);\
    CK(cudaEventRecord(e1));                                                      \
    cudaError_t ee = cudaDeviceSynchronize();                                     \
    double ms = -1; bool ok = false;                                              \
    if (ee == cudaSuccess) {                                                      \
      ms = ms_of(e0, e1);                                                         \
      CK(cudaMemcpy(hB.data(), dOutB, tri * sizeof(int), cudaMemcpyDeviceToHost));\
      long long bad = 0;                                                          \
      for (int j = 1; j <= n; j++)                                                \
        for (int i = 1; i <= j; i++) {                                            \
          if (j < i + 2 * TURN + 3) continue;                                     \
          const long long q = Indx(i, j);                                         \
          if (hA[q] != hB[q]) bad++;                                              \
        }                                                                         \
      ok = (bad == 0);                                                            \
      if (!ok) printf("    reg b=%-3d %dx%d MISMATCH in %lld cells\n",           \
                      BB, RM, RN, bad);                                           \
    } else {                                                                      \
      printf("    reg b=%-3d %dx%d launch failed: %s\n", BB, RM, RN,             \
             cudaGetErrorString(ee));                                             \
    }                                                                             \
    rsr.push_back({BB, RM, RN, ms, ok});                                          \
  } while (0)

  RUN_REG(32, 2, 2);
  RUN_REG(64, 2, 2);
  RUN_REG(64, 4, 4);
  RUN_REG(96, 4, 4);
  RUN_REG(128, 8, 8);

  printf("\n  %-34s %12s %12s %10s %12s %s\n",
         "blocked", "ms", "vs per-row", "vs 1-launch", "traffic cut", "exact");
  for (size_t q = 0; q < rs.size(); q++) {
    const Res &r = rs[q];
    if (r.ms < 0) continue;
    char nm[64]; snprintf(nm, sizeof nm, "b=%d, %d result%s/thread",
                          r.b, r.rpt, r.rpt > 1 ? "s" : "");
    printf("  %-34s %12.2f %11.2fx %10.2fx %11.1fx %s\n", nm, r.ms,
           tA / r.ms, tB / r.ms, r.b / 2.0, r.ok ? "yes" : "*** NO ***");
  }
  printf("\n  %-34s %12s %12s %10s %12s %s\n",
         "blocked + register tiling", "ms", "vs per-row", "vs 1-launch",
         "shared/step", "exact");
  for (size_t q = 0; q < rsr.size(); q++) {
    const ResR &r = rsr[q];
    if (r.ms < 0) continue;
    char nm[80]; snprintf(nm, sizeof nm, "b=%d, %dx%d per thread", r.b, r.rm, r.rn);
    printf("  %-34s %12.2f %11.2fx %10.2fx %12.2f %s\n", nm, r.ms,
           tA / r.ms, tB / r.ms, (double)(r.rm + r.rn) / (r.rm * r.rn),
           r.ok ? "yes" : "*** NO ***");
  }

  printf("\n  `traffic cut` is the predicted b/2; `vs 1-launch` is the achieved\n"
         "  speedup of the loop alone. If the second is well short of the first the\n"
         "  blocked kernel has become compute- or shared-bound, which is the point:\n"
         "  it is no longer at the DRAM roofline.\n");

  CK(cudaFree(dT)); CK(cudaFree(dR)); CK(cudaFree(dOutA)); CK(cudaFree(dOutB));
  CK(cudaFree(dci)); CK(cudaFree(dcj));
  return 0;
}

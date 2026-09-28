/* proto_blocked_md2.cu -- what does md's INF sentinel cost the blocked form?
 *
 * The first prototype measured 6.30x, and it was UNFAITHFUL in two ways that the
 * in-tree stage-1 selftest then exposed:
 *
 *   1. It had NO INF SENTINEL. Every synthetic value was a real energy, so the
 *      branchless "mask the recurrence range to INF and let the product run over
 *      the whole patch" trick could not be wrong. In production it IS wrong: INF
 *      plus a real negative energy lands just BELOW INF and WINS the min, which
 *      the selftest caught immediately as blocked=9999950 against md=10000000.
 *   2. It staged int16 into shared memory. Production decodes to int on the way
 *      in (md_block.inc stages `int *Ys`), so the real shared footprint and the
 *      real shared bandwidth are DOUBLE what that prototype measured.
 *
 * So the 6.30x is not a number to keep quoting. This program re-measures with the
 * sentinel present and int staging, and compares three ways of handling the range:
 *
 *   A  streaming        -- md_cell's shape, the reference. Iterates only y in [0,x],
 *                          so it never touches an out-of-range entry.
 *   B  blocked, BOUNDED -- per-thread tlo/thi over t. What stage 1 landed. Exact,
 *                          but the bounds differ per cell, so a register tile has to
 *                          use a per-cell test and the loop stops being branchless.
 *   C  blocked, MASKVAL -- mask out-of-range entries to a sentinel LARGER than INF
 *                          rather than to INF, and initialise the accumulator to INF.
 *
 * C is the interesting one, and it is exact for a reason worth writing down:
 *
 *     md's answer  = min( INF, { A[y]+B[y] : y in range } )        (acc starts at INF)
 *     C's answer   = min( INF, { in-range sums }, { masked sums } )
 *
 * and every masked sum is >= INF by construction, so the third set never changes the
 * min. C therefore equals md EXACTLY while staying branchless -- it keeps the whole
 * point of blocking. The condition on MASKVAL is just
 *
 *     MASKVAL + (the most negative value that can appear)  >=  INF
 *
 * which is trivially satisfied with room to spare, and 2*MASKVAL must not overflow.
 * Genuine INF cells are staged as INF, not MASKVAL, so md's own INF-plus-real
 * near-INF values are reproduced rather than clamped away.
 *
 *   build: nvcc -O3 -arch=native -o proto2 proto_blocked_md2.cu
 *   run:   ./proto2 [n] [pct_inf]
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define TURN 3
#define INFV 10000000            /* md's INF, exactly */
#define INF16 ((short)32767)     /* reserved: this cell is INF (production's FML_INF16) */
/* Out-of-range marker for variant C. Must satisfy MASKVAL + most-negative >= INFV. */
#define MASKVAL (INFV + 1000000)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {                       \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__host__ __device__ __forceinline__ long long Indx(int i, int j)
{
  return (long long)j * (j - 1) / 2 + i;
}

/* production decodes an int16 offset against a per-64 baseline; the baseline part is
 * already validated in-tree, so here one global baseline of 0 keeps the focus on the
 * SENTINEL and the staging width. */
__host__ __device__ __forceinline__ int decode(short o)
{
  return (o == INF16) ? INFV : (int)o;
}

/* ------------------------------------------------------- A: streaming reference ---*/
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
    for (int y = lane; y <= x; y += TILE) {
      /* md_cell: no INF guard on the sum. Reproduced exactly. */
      v = min(v, decode(R[(long long)i * n + (i + TURN + 1 + y)]) + decode(T[ij0 + y]));
    }
  }
#pragma unroll
  for (int off = TILE / 2; off > 0; off >>= 1)
    v = min(v, __shfl_down_sync(0xffffffffu, v, off, TILE));
  if (active && lane == 0) out[Indx(i, j)] = v;
}

/* --------------------------------------- B and C: blocked, int staging, reg tile ---
 * MODE 0 = bounded (per-cell t range), MODE 1 = MASKVAL (branchless).
 */
template <int B, int RM, int RN, int MODE>
__global__ void md_blocked2(const int n, const short *__restrict__ R,
                            const short *__restrict__ T, int *__restrict__ out)
{
  /* int, not short: production stages DECODED values. This is the footprint the
   * real thing pays, and it is double what the first prototype measured. */
  __shared__ int Xs[B][B + 1];
  __shared__ int Ys[B][B + 1];

  const int I = blockIdx.y, J = blockIdx.x;
  if (J < I) return;

  const int i0 = I * B + 1, j0 = J * B + 1;
  const int TPB = (B / RM) * (B / RN);
  const int tid = threadIdx.x;
  const int cbase = (tid % (B / RN)) * RN;
  const int rbase = (tid / (B / RN)) * RM;

  int acc[RM][RN];
#pragma unroll
  for (int u = 0; u < RM; u++)
#pragma unroll
    for (int v = 0; v < RN; v++) acc[u][v] = INFV;   /* md's initial value */

  const int kmin_tile = i0 + TURN + 1;
  const int kmax_tile = (j0 + B - 1) - TURN - 2;

  for (int k0 = (kmin_tile / B) * B; k0 <= kmax_tile; k0 += B) {
    for (int e = tid; e < B * B; e += TPB) {
      const int t = e / B, rr = e % B;
      const int i = i0 + rr, k = k0 + t;
      const bool inarr = (i <= n) && (k >= 1) && (k <= n) && (i <= k);
      const bool inrng = (k >= i + TURN + 1);
      /* MODE 1 marks out-of-range with MASKVAL; MODE 0 leaves it INF and relies on
       * the loop bounds below. Genuine INF is INF in both. */
      Xs[t][rr] = !inarr ? (MODE ? MASKVAL : INFV)
                : (inrng ? decode(R[(long long)i * n + k])
                         : (MODE ? MASKVAL : INFV));
    }
    for (int e = tid; e < B * B; e += TPB) {
      const int t = e / B, cc = e % B;
      const int j = j0 + cc, kp1 = k0 + t + 1;
      const bool inarr = (j <= n) && (kp1 >= 1) && (kp1 <= j);
      const bool inrng = (kp1 <= j - TURN - 1);
      Ys[t][cc] = !inarr ? (MODE ? MASKVAL : INFV)
                : (inrng ? decode(T[Indx(kp1, j)])
                         : (MODE ? MASKVAL : INFV));
    }
    __syncthreads();

    if (MODE) {
      /* branchless over the whole patch: masked sums are >= INFV so they cannot
       * change a min that already contains INFV */
#pragma unroll 4
      for (int t = 0; t < B; t++) {
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
    } else {
      /* bounded: each cell has its own t range, so the test moves inside */
#pragma unroll
      for (int u = 0; u < RM; u++) {
        const int i = i0 + rbase + u;
        const int tlo = max(0, (i + TURN + 1) - k0);
#pragma unroll
        for (int v = 0; v < RN; v++) {
          const int j = j0 + cbase + v;
          const int thi = min(B - 1, (j - TURN - 2) - k0);
          int a = acc[u][v];
          for (int t = tlo; t <= thi; t++) a = min(a, Xs[t][rbase + u] + Ys[t][cbase + v]);
          acc[u][v] = a;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int u = 0; u < RM; u++)
#pragma unroll
    for (int v = 0; v < RN; v++) {
      const int i = i0 + rbase + u, j = j0 + cbase + v;
      if (i <= n && j <= n && j >= i + 2 * TURN + 3) out[Indx(i, j)] = acc[u][v];
    }
}

static double ms_of(cudaEvent_t a, cudaEvent_t b)
{ float t = 0.f; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; }

int main(int argc, char **argv)
{
  const int n       = (argc > 1) ? atoi(argv[1]) : 4096;
  const int pct_inf = (argc > 2) ? atoi(argv[2]) : 15;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));

  const long long tri = (long long)n * (n + 1) / 2 + 1;
  const long long rowm = (long long)(n + 1) * n;
  printf("%s: L2 %.2f MB, %d SMs, %d KB shared/SM\n", p.name, p.l2CacheSize / 1048576.0,
         p.multiProcessorCount, (int)(p.sharedMemPerMultiprocessor / 1024));
  printf("n = %d, %d%% of cells are INF (the sentinel the first prototype lacked)\n",
         n, pct_inf);

  double steps = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i + 2 * TURN + 3; j <= n; j++) steps += (j - i - 2 * TURN - 3) + 1;
  printf("(min,+) steps = %.4e\n", steps);

  std::vector<short> hT(tri, INF16), hR(rowm, INF16);
  srand(20260928);
  for (int j = 1; j <= n; j++)
    for (int i = 1; i <= j; i++) {
      const bool isinf = (rand() % 100) < pct_inf;
      const short v = isinf ? INF16 : (short)((rand() % 4000) - 2000);
      hT[Indx(i, j)] = v;
      hR[(long long)i * n + j] = v;
    }

  short *dT, *dR; int *dA, *dB;
  CK(cudaMalloc(&dT, tri * sizeof(short)));   CK(cudaMalloc(&dR, rowm * sizeof(short)));
  CK(cudaMalloc(&dA, tri * sizeof(int)));     CK(cudaMalloc(&dB, tri * sizeof(int)));
  CK(cudaMemcpy(dT, hT.data(), tri * sizeof(short), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dR, hR.data(), rowm * sizeof(short), cudaMemcpyHostToDevice));

  std::vector<int> ci, cj;
  for (int i = 1; i <= n; i++)
    for (int j = i + 2 * TURN + 3; j <= n; j++) { ci.push_back(i); cj.push_back(j); }
  const long long ncell = (long long)ci.size();
  int *dci, *dcj;
  CK(cudaMalloc(&dci, ncell * sizeof(int))); CK(cudaMalloc(&dcj, ncell * sizeof(int)));
  CK(cudaMemcpy(dci, ci.data(), ncell * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcj, cj.data(), ncell * sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  CK(cudaEventRecord(e0));
  md_stream_all<32><<<(int)((ncell * 32 + 255) / 256), 256>>>(n, ncell, dci, dcj, dR, dT, dA);
  CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  const double tref = ms_of(e0, e1);
  std::vector<int> hA(tri), hB(tri);
  CK(cudaMemcpy(hA.data(), dA, tri * sizeof(int), cudaMemcpyDeviceToHost));
  printf("\n  %-40s %10.2f ms   (reference)\n", "A  streaming, md_cell's shape", tref);

  struct R2 { const char *nm; int b, rm, rn, mode; double ms; bool ok; };
  std::vector<R2> rs;

#define RUN(BB, RM, RN, MODE, NAME)                                                   \
  do {                                                                                \
    const int ntb = (n + (BB) - 1) / (BB);                                            \
    dim3 grid(ntb, ntb);                                                              \
    CK(cudaMemset(dB, 0x3f, tri * sizeof(int)));                                      \
    CK(cudaEventRecord(e0));                                                          \
    md_blocked2<BB, RM, RN, MODE><<<grid, ((BB)/(RM))*((BB)/(RN))>>>(n, dR, dT, dB);   \
    CK(cudaEventRecord(e1));                                                          \
    cudaError_t ee = cudaDeviceSynchronize();                                         \
    double ms = -1; bool ok = false;                                                  \
    if (ee == cudaSuccess) {                                                          \
      ms = ms_of(e0, e1);                                                             \
      CK(cudaMemcpy(hB.data(), dB, tri * sizeof(int), cudaMemcpyDeviceToHost));        \
      long long bad = 0; long long fi = -1, fj = -1;                                   \
      for (int j = 1; j <= n; j++)                                                     \
        for (int i = 1; i <= j; i++) {                                                 \
          if (j < i + 2 * TURN + 3) continue;                                          \
          const long long q = Indx(i, j);                                              \
          if (hA[q] != hB[q]) { if (fi < 0) { fi = i; fj = j; } bad++; }                \
        }                                                                              \
      ok = (bad == 0);                                                                 \
      if (!ok) printf("    %s: %lld mismatches, first (%lld,%lld) ref=%d got=%d\n",     \
                      NAME, bad, fi, fj, hA[Indx((int)fi,(int)fj)],                     \
                      hB[Indx((int)fi,(int)fj)]);                                       \
    } else printf("    %s: launch failed: %s\n", NAME, cudaGetErrorString(ee));         \
    rs.push_back({NAME, BB, RM, RN, MODE, ms, ok});                                     \
  } while (0)

  RUN(32, 4, 4, 0, "B  blocked, BOUNDED  b=32 4x4");
  RUN(32, 4, 4, 1, "C  blocked, MASKVAL  b=32 4x4");
  RUN(64, 4, 4, 0, "B  blocked, BOUNDED  b=64 4x4");
  RUN(64, 4, 4, 1, "C  blocked, MASKVAL  b=64 4x4");
  RUN(64, 8, 8, 1, "C  blocked, MASKVAL  b=64 8x8");

  printf("\n  %-40s %10s %10s %s\n", "kernel", "ms", "vs A", "exact");
  printf("  %-40s %10.2f %10s %s\n", "A  streaming, md_cell's shape", tref, "1.00x", "-");
  for (size_t q = 0; q < rs.size(); q++) {
    if (rs[q].ms < 0) continue;
    printf("  %-40s %10.2f %9.2fx %s\n", rs[q].nm, rs[q].ms, tref / rs[q].ms,
           rs[q].ok ? "yes" : "*** NO ***");
  }
  printf("\n  An exact MASKVAL row means blocking can stay BRANCHLESS under md's INF\n"
         "  semantics after all, which is a better answer than the bounded loop that\n"
         "  stage 1 landed -- and the gap between the two rows is what the bounds cost.\n");

  CK(cudaFree(dT)); CK(cudaFree(dR)); CK(cudaFree(dA)); CK(cudaFree(dB));
  CK(cudaFree(dci)); CK(cudaFree(dcj));
  return 0;
}

/* bt_device.cu -- N5 device backtrack (PORT_DEVICE_BACKTRACK.md, P2): f5 and the MFE trace of a retired
 * chunk, on the device. bt_device.h has the interface; this file has no caller until P2c.
 *
 * The kernel is P1's (tools/proto_device_bt.cu, exact against upstream on every fixture, ties included)
 * with three changes the tree needs:
 *   - f5 runs on a BLOCK per record (D1), width RNA_DEVICE_BT_BLOCK; the trace stays on warp 0;
 *   - c and fML are read where the sweep left them, in the retired set: fML int32, or int16 decoded in
 *     place (fml_rd below);
 *   - every parameter struct is filled by the sweep's own rnafold_param_fill() / rnafold_param2_fill()
 *     (P2a), so the trace and the sweep cannot disagree about a loop's energy.
 *
 * Model: the default one -- dangles 2, linear, no --noLP, no G-quads, no constraints, default salt and
 * temperature (P2c refuses the rest). Under default hard constraints every `evaluate(...)` in upstream's
 * trace reduces to "the inner pair has c != INF", which is what the device tests.
 *
 * Search orders (backtrack/ in this tree, 2.7.2), each reproduced as the EARLIEST match in upstream's loop:
 *   F5 (bt_exterior_f5.c, case 2): nibble unpaired 3' ends while f5[j] == f5[j-1]; then u = j-1 DOWN to 1:
 *       f5[j] == E_ext(type(u,j), u>1 ? S[u-1] : -1, j<n ? S[j+1] : -1) + c[u,j] + f5[u-1]
 *   C (mfe.c backtrack(), repeat1): hairpin; then the internal loop, p = i+1 UP, q = j-1 DOWN to minq
 *       (bt_internal.c); then the multibranch closing, r = i+2 UP to j-3 (bt_multibranch.c bt_mb_loop)
 *   M (bt_mb_loop_split, case 2): nibble 3', nibble 5'; single stem; split u = i+1 UP to j-2
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/params/default.h"

#undef MIN2
#define MIN2(x,y) min(x,y)
#undef MAX2
#define MAX2(x,y) max(x,y)

/* The sweep's device headers, in int_loop.cu's order. `turn` is a macro there, defined only after
 * stub2.h, whose declarations use it as a parameter name. */
#include "interior_loopx.h"
#include "stub2.h"
#define turn 3
#include "int_loop_dev.h"
#include "hp_mb_dev.h"

#include "bt_device.h"

extern "C" int vrna_cuda_quiet(void);

#define DBT_CK(x) dbt_check((x), #x, __LINE__)

static void
dbt_check(const cudaError_t e, const char *what, const int line)
{
  if (e != cudaSuccess) {
    fprintf(stderr, "bt_device.cu:%d %s: %s\n", line, what, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

#define FULL 0xffffffffu
enum { BT_F5 = 1, BT_C = 2, BT_M = 3 };

/* What the exterior stem reads that the sweep's structs do not carry. */
struct ext_param_t {
  int mismatchExt[NBPAIRS + 1][5][5];
  int dangle5[NBPAIRS + 1][5];
  int dangle3[NBPAIRS + 1][5];
  int TerminalAU;
};

/* vrna_E_exterior_stem(), eval/eval_exterior.c:52, line for line. */
__device__ __forceinline__ int
e_ext(const int type, const int n5d, const int n3d, const ext_param_t *X)
{
  int e = 0;
  if (n5d >= 0 && n3d >= 0)
    e += X->mismatchExt[type][n5d][n3d];
  else if (n5d >= 0)
    e += X->dangle5[type][n5d];
  else if (n3d >= 0)
    e += X->dangle3[type][n3d];
  if (type > 2)
    e += X->TerminalAU;
  return e;
}

/* vrna_get_ptype(): the pair type, 0 promoted to 7. */
__device__ __forceinline__ int
ptype_of(const short *S, const char *pair_, const int i, const int j)
{
  const int t = pair_[S[i] * 8 + S[j]];
  return (t == 0) ? 7 : t;
}

/* One record's fML, read as the host's decoded triangle reads (fml_fetch_decode(),
 * modular_decomposition.cu). int32: the cell. int16: the cell's offset on its baseline, where the baseline
 * slot is boff + colb(j) + (i-1)/FML_BLK and colb(j) = FML_BLK*q(q+1)/2 + r(q+1), q = (j-1)/FML_BLK,
 * r = (j-1)%FML_BLK -- the closed form of the decode's running sum over ceil(k/FML_BLK), k < j.
 *
 * i > j: the M nibble reads fML(i, i-1), whose flat index Indx(i, i-1) is ANOTHER cell's (Indx(1, i)).
 * The host reads whatever its decoded flat array holds there, which is that cell decoded on its OWN
 * baseline, so the read is first mapped back to its own (i', j'). */
struct fml_rd {
  const int   *j32;    /* record's triangle, or NULL */
  const short *j16;    /* record's triangle, or NULL */
  const int   *b;      /* record's baselines (d_fml_b + base_off_H[H]) */

  __device__ __forceinline__ int
  operator()(int i, int j) const
  {
    const long long t = Indx(i, j);
    if (j32)
      return j32[t];
    if (i > j) {   /* column j' holds flat indices j'(j'-1)/2 + 1 .. j'(j'+1)/2 */
      while (t > (long long)j * (j + 1) / 2)
        j++;
      i = (int)(t - (long long)j * (j - 1) / 2);
    }
    const short o = j16[t];
    if (o == FML_INF16)
      return INF;
    const int q = (j - 1) / FML_BLK, r = (j - 1) % FML_BLK;
    return b[(long long)FML_BLK * q * (q + 1) / 2 + (long long)r * (q + 1) + (i - 1) / FML_BLK] + (int)o;
  }
};

/* The kernel's view of one set. Everything but c / fML is the set's own (bt_device.h). */
struct dbt_batch_t {
  int                  nrec;
  const int           *n_H;
  const size_t        *tri_off;   /* nrec+1: cells of record H start here in c and fML  */
  const size_t        *seq_off;   /* nrec+1: n+2 slots of record H (S, seq, f5, pt, str) */
  const size_t        *sp_off;    /* nrec+1: packed-sequence words of record H            */
  const short         *S;         /* upstream's sequence_encoding, 0..n+1                 */
  const char          *seq;       /* fc->sequence (the special hairpins)                  */
  const unsigned int  *Sp;        /* 3-bit packed, the layout unpack() reads (H = 0)      */
  const char          *pair_;
  const cuda_param_t  *P;
  const cuda_param2_t *P2;
  const ext_param_t   *X;
  const int           *salt_loop;
  int                 *f5, *stk, *pt, *status;
  char                *str;
  /* the retired set's */
  const int           *c;
  const int           *fml32;
  const short         *fml16;
  const int           *fml_b;
  const size_t        *base_off_H;
};

/* The chosen candidate among K, in upstream's order (candidate 0 first): the earliest match, or with
 * ORDER_REVERSED (the negative control) the last. */
template<class PRED>
__device__ __forceinline__ int
pick(const int K, const int order, PRED pred)
{
  const int lane  = threadIdx.x & 31;
  int       found = -1;

  for (int base = 0; base < K; base += 32) {
    const int      k = base + lane;
    const unsigned b = __ballot_sync(FULL, (k < K) && pred(k));
    if (!b)
      continue;
    if (order == RNAFOLD_DBT_ORDER_UPSTREAM)
      return base + __ffs(b) - 1;
    found = base + 31 - __clz(b);
  }
  return found;
}

/* One block per record. f5 on the whole block, then the trace on warp 0. */
__global__ void
dbt_kernel(const dbt_batch_t B, const int order)
{
  __shared__ int wmin[32];
  const int      r     = blockIdx.x;
  const int      tid   = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int      nwarp = (blockDim.x + 31) >> 5;

  const int           n     = B.n_H[r];
  const int          *c     = B.c + B.tri_off[r];
  const short        *S     = B.S + B.seq_off[r];
  const char         *seq   = B.seq + B.seq_off[r];
  const unsigned int *Sp    = B.Sp + B.sp_off[r];
  volatile int       *f5    = B.f5 + B.seq_off[r];
  volatile int       *stk   = B.stk + 3 * B.seq_off[r];
  int                *pt    = B.pt + B.seq_off[r];
  char               *str   = B.str + B.seq_off[r];
  const char         *pair_ = B.pair_;
  const cuda_param_t *P     = B.P;
  const cuda_param2_t *P2   = B.P2;
  const ext_param_t  *X     = B.X;
  const c_tri_reader  cread = { c };
  const fml_rd        fml   = { B.fml32 ? B.fml32 + B.tri_off[r] : NULL,
                                B.fml16 ? B.fml16 + B.tri_off[r] : NULL,
                                B.fml16 ? B.fml_b + B.base_off_H[r] : NULL };
  const int           cap   = n + 2;   /* stacked intervals are disjoint: at most n, plus one empty */

  /* ---- f5 (vrna_mfe_exterior_f5, dangles 2): threads over i, one j at a time ---- */
  if (tid == 0) {
    f5[0] = 0;
    f5[1] = 0;
  }
  __syncthreads();
  for (int j = 2; j <= n; j++) {
    const int sj1  = (j < n) ? S[j + 1] : -1;
    int       best = INF;
    for (int i = 2 + tid; i <= j - 1; i += blockDim.x) {
      const int cij = c[Indx(i, j)];
      if (cij != INF)
        best = min(best, f5[i - 1] + cij + e_ext(ptype_of(S, pair_, i, j), S[i - 1], sj1, X));
    }
    for (int o = 16; o > 0; o >>= 1)
      best = min(best, __shfl_xor_sync(FULL, best, o));
    if (lane == 0)
      wmin[warp] = best;
    __syncthreads();
    if (tid == 0) {
      for (int w = 1; w < nwarp; w++)
        best = min(best, wmin[w]);
      const int c1 = c[Indx(1, j)];
      if (c1 != INF)
        best = min(best, c1 + e_ext(ptype_of(S, pair_, 1, j), -1, sj1, X));
      f5[j] = min(best, (int)f5[j - 1]);
    }
    __syncthreads();   /* f5[j] is read by every thread from j+1 on, and wmin is reused */
  }

  if (warp != 0)
    return;   /* no block-wide barrier from here on */

  /* ---- the trace, on warp 0 ---- */
  for (int k = lane; k <= n + 1; k += 32)
    pt[k] = 0;
  __syncwarp();

  int sp = 0, fail = 0;
#define PUSH(a, b, m) do { if (sp >= cap) { fail = 9; break; } \
    if (lane == 0) { stk[3 * sp] = (a); stk[3 * sp + 1] = (b); stk[3 * sp + 2] = (m); } sp++; } while (0)

  PUSH(1, n, BT_F5);
  __syncwarp();

  while (sp > 0 && !fail) {
    sp--;
    int       i  = stk[3 * sp], j = stk[3 * sp + 1];
    const int ml = stk[3 * sp + 2];
    __syncwarp();   /* every lane has read the slot before lane 0 can overwrite it */

    if (ml == BT_F5) {
      if (j == 0)
        continue;
      int fij, fi;
      do {
        fij = f5[j];
        fi  = f5[j - 1];
        if (--j == 0)
          break;
      } while (fij == fi);
      j++;
      if (j < 2)
        continue;
      const int mm3 = (j < n) ? S[j + 1] : -1;
      const int jj  = j, f = fij;
      const int k   = pick(j - 1, order, [&](int kk) {
        const int u   = jj - 1 - kk;
        const int cuj = c[Indx(u, jj)];
        return (cuj != INF) &&
               (f == e_ext(ptype_of(S, pair_, u, jj), (u > 1) ? S[u - 1] : -1, mm3, X) + cuj + f5[u - 1]);
      });
      if (k < 0) {
        fail = 1;
        break;
      }
      const int u = j - 1 - k;
      PUSH(1, u - 1, BT_F5);
      PUSH(u, j, BT_C);
    } else if (ml == BT_M) {
      /* bt_mb_loop_split, dangles 2 */
      int fij, fi;
      do {
        fij = fml(i, j);
        fi  = P2->MLbase + fml(i, j - 1);
        if (--j == 0)
          break;
      } while (fij == fi);
      j++;
      do {
        fij = fml(i, j);
        fi  = P2->MLbase + fml(i + 1, j);
        if (++i >= j)
          break;
      } while (fij == fi);
      i--;
      if (j <= i) {
        fail = 2;
        break;
      }
      const int  ii = i, jj = j;
      const int  cij  = c[Indx(i, j)];
      const bool stem = (cij != INF) &&
                        (fij == cij + E_MLstem_device(ptype_of(S, pair_, i, j), S[i - 1], S[j + 1], P2));
      int        k    = -1;
      if (order != RNAFOLD_DBT_ORDER_UPSTREAM || !stem)
        k = pick(j - 2 - i, order, [&](int kk) {
          const int u = ii + 1 + kk;
          return fij == fml(ii, u) + fml(u + 1, jj);
        });
      if (order == RNAFOLD_DBT_ORDER_REVERSED ? (k < 0 && stem) : stem) {
        PUSH(i, j, BT_C);
      } else if (k >= 0) {
        const int u = i + 1 + k;
        PUSH(i, u, BT_M);
        PUSH(u + 1, j, BT_M);
      } else {
        fail = 3;
        break;
      }
    } else {
      /* BT_C: the pair, then backtrack()'s repeat1 -- hairpin, internal loop, multibranch */
      if (lane == 0) {
        pt[i] = j;
        pt[j] = i;
      }
      const int  cij  = c[Indx(i, j)];
      const int  type = ptype_of(S, pair_, i, j);
      const bool hp   = (E_Hairpin_device(j - i - 1, type, S[i + 1], S[j - 1], seq + i - 1, P2,
                                          B.salt_loop) == cij);

      /* internal loop: p UP, q DOWN to minq; one ballot per p (at most MAXLOOP+1 candidates) */
      int ip = -1, iq = -1;
      if (order != RNAFOLD_DBT_ORDER_UPSTREAM || !hp) {
        const cell_inv_t ci   = cell_invariants(Sp, pair_, 0, 0, i, j);
        const int        pmax = min(j - 2, i + MAXLOOP + 1);
        for (int s = 0; s < pmax - i; s++) {
          const int p    = (order == RNAFOLD_DBT_ORDER_REVERSED) ? pmax - s : i + 1 + s;
          const int minq = (p + 1 + MAXLOOP + 2 + i < j + p) ? p + j - i - MAXLOOP - 2 : p + 1;
          const int K    = j - minq;   /* q = j-1 .. minq */
          const int q    = j - 1 - lane;
          const unsigned b = __ballot_sync(FULL,
                                           (lane < K) &&
                                           (Energy(0, 0, i, j, q, p, ci, cread, NULL, Sp, pair_, P,
                                                   P->TerminalAU, P->ninio2, P->bulge, P->internal_loop,
                                                   P->lxc, P->mismatchI, P->mismatch1nI, P->mismatch23I,
                                                   P->stack, P->int11, P->int21, P->int22) == cij));
          if (!b)
            continue;
          ip = p;
          iq = j - 1 - ((order == RNAFOLD_DBT_ORDER_REVERSED) ? 31 - __clz(b) : __ffs(b) - 1);
          break;
        }
      }

      /* multibranch closing (bt_mb_loop, dangles 2): r = i+2 .. j-3 */
      int kr = -1;
      if ((order != RNAFOLD_DBT_ORDER_UPSTREAM || (!hp && ip < 0)) && (i + 2 * (turn + 2) < j)) {
        const int tm = P2->rtype[type];
        const int e  = cij - P2->MLclosing - E_MLstem_device(tm, S[j - 1], S[i + 1], P2);
        const int ii = i, jj = j;
        kr = pick(j - i - 4, order, [&](int kk) {
          const int rr = ii + 2 + kk;
          return e == fml(ii + 1, rr) + fml(rr + 1, jj - 1);
        });
      }

      if (order == RNAFOLD_DBT_ORDER_REVERSED) {
        if (kr >= 0) {
          const int rr = i + 2 + kr;
          PUSH(i + 1, rr, BT_M);
          PUSH(rr + 1, j - 1, BT_M);
        } else if (ip >= 0) {
          PUSH(ip, iq, BT_C);
        } else if (!hp) {
          fail = 4;
          break;
        }
      } else if (hp) {
        /* a hairpin: nothing more */
      } else if (ip >= 0) {
        PUSH(ip, iq, BT_C);
      } else if (kr >= 0) {
        const int rr = i + 2 + kr;
        PUSH(i + 1, rr, BT_M);
        PUSH(rr + 1, j - 1, BT_M);
      } else {
        fail = 5;
        break;
      }
    }
    __syncwarp();
  }
#undef PUSH

  /* the dot-bracket string, as vrna_db_from_bps() writes it without G-quads */
  __syncwarp();
  for (int k = 1 + lane; k <= n; k += 32) {
    const int p = pt[k];
    str[k - 1] = (p == 0) ? '.' : ((p > k) ? '(' : ')');
  }
  if (lane == 0) {
    str[n]      = 0;
    B.status[r] = fail;
  }
}

/* ======================================== host ======================================== */

/* RNA_DEVICE_BT_BLOCK: the f5 block width (D1), 32/64/128/256; P2d's measurement picks the default. */
static int
dbt_block(void)
{
  static int v = 0;

  if (!v) {
    const char *e = getenv("RNA_DEVICE_BT_BLOCK");
    const int   x = (e && e[0]) ? atoi(e) : 0;

    v = 128;
    if ((x == 32) || (x == 64) || (x == 128) || (x == 256)) {
      v = x;
      if (!vrna_cuda_quiet())
        fprintf(stderr, "bt_device.cu             RNA_DEVICE_BT_BLOCK=%d: f5 on %d threads a record\n",
                v, v);
    } else if (e && e[0]) {
      fprintf(stderr, "bt_device.cu             RNA_DEVICE_BT_BLOCK=%s IGNORED: 32, 64, 128 or 256; "
                      "using %d\n", e, v);
    }
  }

  return v;
}

/* The trace's own stream, at the device's LOWEST priority: beside the next chunk's sweep, the sweep's
 * launches go first. Priority orders new block launches only; it does not preempt resident blocks. */
static cudaStream_t   g_dbt_stream;
static pthread_once_t g_dbt_once = PTHREAD_ONCE_INIT;

static void
dbt_stream_make(void)
{
  int lo = 0, hi = 0;

  DBT_CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  DBT_CK(cudaStreamCreateWithPriority(&g_dbt_stream, cudaStreamNonBlocking, lo));
}

struct rnafold_dbt_s {
  int          nrec;
  size_t       slots;      /* sum of n+2 */
  size_t       bytes;      /* the device blob */
  char        *d_blob;     /* ONE allocation: rnafold_dbt_free() frees it by the pointer */
  dbt_batch_t  B;          /* device pointers into d_blob; c / fML filled by rnafold_dbt_run() */
  size_t       off_f5, off_str, off_status;   /* where the results sit in d_blob */
  size_t      *seq_off;    /* host copy */
  int         *n_H;        /* host copy */
  int         *h_f5;       /* results, slots ints */
  char        *h_str;      /* results, slots chars */
  int         *h_status;   /* results, nrec ints */
};

static size_t
dbt_carve(size_t *at, const size_t bytes)
{
  const size_t o = (*at + 255) & ~(size_t)255;

  *at = o + bytes;
  return o;
}

extern "C" rnafold_dbt_t *
rnafold_dbt_prepare(struct vrna_fc_s *const *VC, const int nslots, const size_t *tri_off_H)
{
  rnafold_dbt_t      *d  = (rnafold_dbt_t *)calloc(1, sizeof(rnafold_dbt_t));
  const vrna_param_t *Pp = VC[0]->params;
  int                 maxn = 0;

  if (!d) {
    fprintf(stderr, "bt_device.cu             prepare: out of host memory\n");
    exit(EXIT_FAILURE);
  }

  d->nrec    = nslots;
  d->n_H     = (int *)malloc((size_t)nslots * sizeof(int));
  d->seq_off = (size_t *)malloc((size_t)(nslots + 1) * sizeof(size_t));
  size_t *sp_off = (size_t *)malloc((size_t)(nslots + 1) * sizeof(size_t));

  d->seq_off[0] = sp_off[0] = 0;
  for (int H = 0; H < nslots; H++) {
    const int n = (int)VC[H]->length;
    assert(tri_off_H[H + 1] - tri_off_H[H] == (size_t)(n + 1) * (n + 2) / 2);
    d->n_H[H]         = n;
    d->seq_off[H + 1] = d->seq_off[H] + (size_t)n + 2;
    sp_off[H + 1]     = sp_off[H] + (size_t)S_STRIDE(n);
    if (n > maxn)
      maxn = n;
  }
  d->slots = d->seq_off[nslots];
  const size_t slots = d->slots, words = sp_off[nslots];

  /* ---- the blob: what is uploaded first, then the scratch ---- */
  size_t       at = 0;
  const size_t o_n   = dbt_carve(&at, (size_t)nslots * sizeof(int));
  const size_t o_tri = dbt_carve(&at, (size_t)(nslots + 1) * sizeof(size_t));
  const size_t o_seq = dbt_carve(&at, (size_t)(nslots + 1) * sizeof(size_t));
  const size_t o_spo = dbt_carve(&at, (size_t)(nslots + 1) * sizeof(size_t));
  const size_t o_P   = dbt_carve(&at, sizeof(cuda_param_t));
  const size_t o_P2  = dbt_carve(&at, sizeof(cuda_param2_t));
  const size_t o_X   = dbt_carve(&at, sizeof(ext_param_t));
  const size_t o_pr  = dbt_carve(&at, 64);
  const size_t o_sl  = dbt_carve(&at, (size_t)(maxn + 2) * sizeof(int));
  const size_t o_hp  = dbt_carve(&at, (size_t)(maxn + 2) * sizeof(int));
  const size_t o_S   = dbt_carve(&at, slots * sizeof(short));
  const size_t o_sq  = dbt_carve(&at, slots);
  const size_t o_Sp  = dbt_carve(&at, words * sizeof(unsigned int));
  const size_t up    = at;   /* everything before here is uploaded */
  const size_t o_f5  = dbt_carve(&at, slots * sizeof(int));
  const size_t o_stk = dbt_carve(&at, 3 * slots * sizeof(int));
  const size_t o_pt  = dbt_carve(&at, slots * sizeof(int));
  const size_t o_str = dbt_carve(&at, slots);
  const size_t o_st  = dbt_carve(&at, (size_t)nslots * sizeof(int));
  d->bytes = at;

  DBT_CK(cudaMalloc((void **)&d->d_blob, d->bytes));
  char *h = (char *)calloc(1, up);
  if (!h) {
    fprintf(stderr, "bt_device.cu             prepare: out of host memory\n");
    exit(EXIT_FAILURE);
  }

  memcpy(h + o_n, d->n_H, (size_t)nslots * sizeof(int));
  memcpy(h + o_tri, tri_off_H, (size_t)(nslots + 1) * sizeof(size_t));
  memcpy(h + o_seq, d->seq_off, (size_t)(nslots + 1) * sizeof(size_t));
  memcpy(h + o_spo, sp_off, (size_t)(nslots + 1) * sizeof(size_t));

  /* the parameters: the sweep's own fills (P2a), and the two tables init_gpu3() builds, from the
   * same expressions -- hp_len always, since the trace must match upstream's double arithmetic */
  rnafold_param_fill((cuda_param_t *)(h + o_P), Pp);
  cuda_param2_t *P2 = (cuda_param2_t *)(h + o_P2);
  rnafold_param2_fill(P2, Pp);
  P2->hp_len = (const int *)(d->d_blob + o_hp);

  ext_param_t *X = (ext_param_t *)(h + o_X);
  memcpy(X->mismatchExt, Pp->mismatchExt, sizeof(X->mismatchExt));
  memcpy(X->dangle5, Pp->dangle5, sizeof(X->dangle5));
  memcpy(X->dangle3, Pp->dangle3, sizeof(X->dangle3));
  X->TerminalAU = Pp->TerminalAU;

  for (int x = 0; x < 8; x++)
    for (int y = 0; y < 8; y++)
      h[o_pr + x * 8 + y] = (char)Pp->model_details.pair[x][y];

  rnafold_build_salt_table(Pp, maxn, (int *)(h + o_sl));
  {
    int *hb = (int *)(h + o_hp);
    for (int s = 0; s < maxn + 2; s++)
      hb[s] = (s <= 30) ? Pp->hairpin[s] : Pp->hairpin[30] + (int)(Pp->lxc * log((double)s / 30.));
  }

  /* the sequences: S1 shorts, the characters, and the 3-bit packing unpack() reads, per record */
  for (int H = 0; H < nslots; H++) {
    const int n = d->n_H[H];
    memcpy(h + o_S + d->seq_off[H] * sizeof(short), VC[H]->sequence_encoding, (size_t)(n + 2) * sizeof(short));
    memcpy(h + o_sq + d->seq_off[H], VC[H]->sequence, (size_t)n);
    unsigned int *w = (unsigned int *)(h + o_Sp) + sp_off[H];
    for (int i = 0; i <= n + 1; i++)
      w[i / 10] |= ((unsigned int)VC[H]->sequence_encoding[i] & 7u) << ((i % 10) * 3);
  }

  /* On the trace's OWN stream, and synchronised. A plain cudaMemcpy from pageable memory may return
   * before the DMA lands, and the trace kernel runs on a non-blocking stream that does not wait for the
   * legacy one: launched straight after (pipeline off), it read a part-uploaded blob -- status 5 on
   * every record of every chunk after the first, and on one A/U chunk a WRONG structure that traced
   * clean (P2c smoke, 2026-10-16). Same stream orders the launch after the upload. */
  pthread_once(&g_dbt_once, dbt_stream_make);
  DBT_CK(cudaMemcpyAsync(d->d_blob, h, up, cudaMemcpyHostToDevice, g_dbt_stream));
  DBT_CK(cudaStreamSynchronize(g_dbt_stream));
  free(h);
  free(sp_off);

  dbt_batch_t *B = &d->B;
  B->nrec      = nslots;
  B->n_H       = (const int *)(d->d_blob + o_n);
  B->tri_off   = (const size_t *)(d->d_blob + o_tri);
  B->seq_off   = (const size_t *)(d->d_blob + o_seq);
  B->sp_off    = (const size_t *)(d->d_blob + o_spo);
  B->P         = (const cuda_param_t *)(d->d_blob + o_P);
  B->P2        = (const cuda_param2_t *)(d->d_blob + o_P2);
  B->X         = (const ext_param_t *)(d->d_blob + o_X);
  B->pair_     = (const char *)(d->d_blob + o_pr);
  B->salt_loop = (const int *)(d->d_blob + o_sl);
  B->S         = (const short *)(d->d_blob + o_S);
  B->seq       = (const char *)(d->d_blob + o_sq);
  B->Sp        = (const unsigned int *)(d->d_blob + o_Sp);
  B->f5        = (int *)(d->d_blob + o_f5);
  B->stk       = (int *)(d->d_blob + o_stk);
  B->pt        = (int *)(d->d_blob + o_pt);
  B->str       = (char *)(d->d_blob + o_str);
  B->status    = (int *)(d->d_blob + o_st);
  d->off_f5     = o_f5;
  d->off_str    = o_str;
  d->off_status = o_st;
  return d;
}

extern "C" int
rnafold_dbt_run(rnafold_dbt_t *d, const rnafold_retired_t *r, const int order)
{
  int bad = 0;

  if ((!d) || (!r) || r->negctl_freed || (r->nslots != d->nrec) || (!r->d_my_c) ||
      ((!r->d_fml_j) && (!r->d_fml_j16)) || (r->d_fml_j16 && ((!r->d_fml_b) || (!r->d_base_off_H)))) {
    fprintf(stderr, "bt_device.cu             run: the retired set does not match the prepared one\n");
    exit(EXIT_FAILURE);
  }

  pthread_once(&g_dbt_once, dbt_stream_make);

  dbt_batch_t B = d->B;
  B.c          = r->d_my_c;
  B.fml32      = r->d_fml_j16 ? NULL : r->d_fml_j;
  B.fml16      = r->d_fml_j16;
  B.fml_b      = r->d_fml_j16 ? r->d_fml_b : NULL;
  B.base_off_H = r->d_fml_j16 ? r->d_base_off_H : NULL;

  if (!d->h_f5) {
    d->h_f5     = (int *)malloc(d->slots * sizeof(int));
    d->h_str    = (char *)malloc(d->slots);
    d->h_status = (int *)malloc((size_t)d->nrec * sizeof(int));
    if ((!d->h_f5) || (!d->h_str) || (!d->h_status)) {
      fprintf(stderr, "bt_device.cu             run: out of host memory\n");
      exit(EXIT_FAILURE);
    }
  }

  dbt_kernel<<<d->nrec, dbt_block(), 0, g_dbt_stream>>>(B, order);
  DBT_CK(cudaGetLastError());
  DBT_CK(cudaMemcpyAsync(d->h_f5, d->d_blob + d->off_f5, d->slots * sizeof(int),
                         cudaMemcpyDeviceToHost, g_dbt_stream));
  DBT_CK(cudaMemcpyAsync(d->h_str, d->d_blob + d->off_str, d->slots,
                         cudaMemcpyDeviceToHost, g_dbt_stream));
  DBT_CK(cudaMemcpyAsync(d->h_status, d->d_blob + d->off_status, (size_t)d->nrec * sizeof(int),
                         cudaMemcpyDeviceToHost, g_dbt_stream));
  DBT_CK(cudaStreamSynchronize(g_dbt_stream));

  for (int H = 0; H < d->nrec; H++)
    bad += (d->h_status[H] != 0);

  return bad;
}

extern "C" const int *
rnafold_dbt_f5(const rnafold_dbt_t *d, const int H)
{
  assert(d && d->h_f5 && (H >= 0) && (H < d->nrec));
  return d->h_f5 + d->seq_off[H];
}

extern "C" const char *
rnafold_dbt_structure(const rnafold_dbt_t *d, const int H)
{
  assert(d && d->h_str && (H >= 0) && (H < d->nrec));
  return d->h_str + d->seq_off[H];
}

extern "C" int
rnafold_dbt_status(const rnafold_dbt_t *d, const int H)
{
  assert(d && d->h_status && (H >= 0) && (H < d->nrec));
  return d->h_status[H];
}

extern "C" size_t
rnafold_dbt_bytes(const rnafold_dbt_t *d)
{
  return d ? d->bytes : 0;
}

extern "C" void
rnafold_dbt_free(rnafold_dbt_t *d)
{
  if (!d)
    return;

  if (d->d_blob)
    DBT_CK(cudaFree(d->d_blob));
  free(d->n_H);
  free(d->seq_off);
  free(d->h_f5);
  free(d->h_str);
  free(d->h_status);
  free(d);
}

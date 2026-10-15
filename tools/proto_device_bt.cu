/* tools/proto_device_bt.cu -- N5 P1 (PORT_DEVICE_BACKTRACK.md §4): the MFE traceback on the device,
 * standalone, checked against upstream's own trace on the same matrices.
 *
 * For each record the host folds with upstream (RNA_GPU=0), keeps c, fML and f5, and traces with
 * upstream's backtrack. The device gets the same c and fML in the same layout the sweep leaves them in
 * (Indx(i,j) = j(j-1)/2 + i, one triangle per record), builds f5 itself (§3.3), and traces with ONE WARP
 * PER RECORD (§3.2): the interval stack in global memory, every search evaluated 32 candidates at a time
 * and ballotted, the winner the EARLIEST match in upstream's own loop order (§3.1). Bar: every f5 cell,
 * every structure and every MFE equal upstream's.
 *
 * The energies are the sweep's own device functions (int_loop_dev.h Energy()/IntLoop_X, hp_mb_dev.h
 * E_Hairpin_device / E_MLstem_device) fed from parameter structs built exactly as load_param() and
 * load_param2() build them, so a trace and the sweep cannot disagree about a loop's energy. Only the
 * exterior stem, which the sweep never evaluates on the device, is new here (e_ext, upstream's
 * vrna_E_exterior_stem() line for line).
 *
 * Model: the default one -- dangles 2, linear, no --noLP, no G-quads, no constraints, default salt.
 * Under default hard constraints every `evaluate(...)` in the upstream trace reduces to "the inner pair
 * has c != INF" (hc_reset_to_default() gives every canonical pair ALL_LOOPS, and c is INF for every other
 * one), which is what the device tests.
 *
 * Search orders reproduced (backtrack/ in this tree, 2.7.2):
 *   F5 (bt_exterior_f5.c, case 2): nibble unpaired 3' ends while f5[j] == f5[j-1]; then u = j-1 DOWN to 1:
 *       f5[j] == E_ext(type(u,j), u>1 ? S[u-1] : -1, j<n ? S[j+1] : -1) + c[u,j] + f5[u-1]
 *   C (mfe.c backtrack(), repeat1): hairpin; then the internal loop, p = i+1 UP, q = j-1 DOWN to minq
 *       (bt_internal.c); then the multibranch closing, r = i+2 UP to j-3 (bt_multibranch.c bt_mb_loop)
 *   M (bt_mb_loop_split, case 2): nibble 3', nibble 5'; single stem; split u = i+1 UP to j-2
 *
 * usage: proto_device_bt MODE NSEQ L SEED [negctl]
 *   MODE: random (ACGU) | au (A/U only: ties everywhere) | grich (GGGAGGG runs in random)
 *   negctl 1: also trace with every search's order REVERSED (last match, categories in reverse) -- must
 *             change structures on a tie-heavy fixture, and every changed one must still be at the MFE.
 * Prints one RESULT line per run. Exit 0 iff the device equals upstream (and, with negctl, the control bit).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

extern "C" {
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/mfe/exterior.h"
#include "ViennaRNA/backtrack/global.h"
#include "ViennaRNA/eval/structures.h"
#include "ViennaRNA/structures/pairtable.h"
#include "ViennaRNA/datastructures/dp_matrices.h"
#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/params/default.h"
}

/* The sweep's device headers, in int_loop.cu's order. `turn` is a macro there, defined only after
 * stub2.h, whose declarations use it as a parameter name. */
#include "interior_loopx.h"
#include "stub2.h"
#define turn 3
#include "int_loop_dev.h"
#include "hp_mb_dev.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

#define FULL 0xffffffffu
enum { BT_F5 = 1, BT_C = 2, BT_M = 3 };
enum { ORDER_UPSTREAM = 0, ORDER_REVERSED = 1, ORDER_COUNT = 2 };

/* What the exterior stem and the multibranch closing read that the sweep's structs do not carry. */
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

struct batch_t {
  int                 nrec;
  const int          *n_H;
  const long long    *tri_off;    /* nrec+1: cells of record r start here in c and fml */
  const long long    *seq_off;    /* nrec+1: n+2 slots of record r (S, seq, f5, pt)    */
  const long long    *sp_off;     /* nrec+1: packed-sequence words of record r          */
  const int          *c, *fml;
  const short        *S;          /* upstream's sequence_encoding, 0..n+1               */
  const char         *seq;        /* upstream's fc->sequence (for the special hairpins)  */
  const unsigned int *Sp;         /* 3-bit packed, the layout unpack() reads (H = 0)     */
  const char         *pair_;
  const cuda_param_t *P;
  const cuda_param2_t *P2;
  const ext_param_t  *X;
  const int          *salt_loop;
  int                *f5, *stk, *pt, *status, *ties;
};

/* The chosen candidate among K, in upstream's order (candidate 0 first). ORDER_UPSTREAM stops at the
 * first chunk with a match and takes its lowest lane; ORDER_REVERSED scans everything and takes the
 * last; ORDER_COUNT takes the first but scans everything, adding every match to *nmatch. */
template<class PRED>
__device__ __forceinline__ int
pick(const int K, const int order, int *nmatch, PRED pred)
{
  const int lane  = threadIdx.x & 31;
  int       found = -1;

  for (int base = 0; base < K; base += 32) {
    const int      k = base + lane;
    const unsigned b = __ballot_sync(FULL, (k < K) && pred(k));
    if (!b)
      continue;
    if (order == ORDER_UPSTREAM)
      return base + __ffs(b) - 1;
    *nmatch += __popc(b);
    if (order == ORDER_REVERSED)
      found = base + 31 - __clz(b);
    else if (found < 0)
      found = base + __ffs(b) - 1;
  }
  return found;
}

__global__ void
bt_kernel(const batch_t B, const int order, const int phases)
{
  const int r    = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
  const int lane = threadIdx.x & 31;
  if (r >= B.nrec)
    return;   /* whole warps only: blockDim is a multiple of 32 */

  const int           n    = B.n_H[r];
  const int          *c    = B.c + B.tri_off[r];
  const int          *fml  = B.fml + B.tri_off[r];
  const short        *S    = B.S + B.seq_off[r];
  const char         *seq  = B.seq + B.seq_off[r];
  const unsigned int *Sp   = B.Sp + B.sp_off[r];
  volatile int       *f5   = B.f5 + B.seq_off[r];
  volatile int       *stk  = B.stk + 3 * B.seq_off[r];
  int                *pt   = B.pt + B.seq_off[r];
  const char         *pair_ = B.pair_;
  const cuda_param_t *P    = B.P;
  const cuda_param2_t *P2  = B.P2;
  const ext_param_t  *X    = B.X;
  const c_tri_reader  cread = { c };
  const int           cap  = n + 2;   /* stacked intervals are disjoint: at most n, plus one empty */

  /* ---- f5 (vrna_mfe_exterior_f5, dangles 2): lanes over i, one j at a time ---- */
  if (phases & 1) {
    if (lane == 0) {
      f5[0] = 0;
      f5[1] = 0;
    }
    __syncwarp();
    for (int j = 2; j <= n; j++) {
      const int sj1  = (j < n) ? S[j + 1] : -1;
      int       best = INF;
      for (int i = 2 + lane; i <= j - 1; i += 32) {
        const int cij = c[Indx(i, j)];
        if (cij != INF) {
          const int v = f5[i - 1] + cij + e_ext(ptype_of(S, pair_, i, j), S[i - 1], sj1, X);
          best = min(best, v);
        }
      }
      if (lane == 0) {
        const int c1 = c[Indx(1, j)];
        if (c1 != INF)
          best = min(best, c1 + e_ext(ptype_of(S, pair_, 1, j), -1, sj1, X));
        best = min(best, (int)f5[j - 1]);
      }
      for (int o = 16; o > 0; o >>= 1)
        best = min(best, __shfl_xor_sync(FULL, best, o));
      if (lane == 0)
        f5[j] = best;
      __syncwarp();
    }
  }

  if (!(phases & 2))
    return;

  /* ---- the trace ---- */
  for (int k = lane; k <= n + 1; k += 32)
    pt[k] = 0;

  int sp = 0, nties = 0, fail = 0;
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
      int       nm  = 0;
      const int jj  = j, f = fij;
      const int k   = pick(j - 1, order, &nm, [&](int kk) {
        const int u   = jj - 1 - kk;
        const int cuj = c[Indx(u, jj)];
        return (cuj != INF) &&
               (f == e_ext(ptype_of(S, pair_, u, jj), (u > 1) ? S[u - 1] : -1, mm3, X) + cuj + f5[u - 1]);
      });
      if (nm > 1)
        nties++;
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
        fij = fml[Indx(i, j)];
        fi  = P2->MLbase + fml[Indx(i, j - 1)];
        if (--j == 0)
          break;
      } while (fij == fi);
      j++;
      do {
        fij = fml[Indx(i, j)];
        fi  = P2->MLbase + fml[Indx(i + 1, j)];
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
      int        nm   = 0;
      int        k    = -1;
      if (order != ORDER_UPSTREAM || !stem)
        k = pick(j - 2 - i, order, &nm, [&](int kk) {
          const int u = ii + 1 + kk;
          return fij == fml[Indx(ii, u)] + fml[Indx(u + 1, jj)];
        });
      if (stem + nm > 1)
        nties++;
      if (order == ORDER_REVERSED ? (k < 0 && stem) : stem) {
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
      int        nm   = hp ? 1 : 0;

      /* internal loop: p UP, q DOWN to minq; one ballot per p (at most MAXLOOP+1 candidates) */
      int ip = -1, iq = -1;
      if (order != ORDER_UPSTREAM || !hp) {
        const cell_inv_t ci   = cell_invariants(Sp, pair_, 0, 0, i, j);
        const int        pmax = min(j - 2, i + MAXLOOP + 1);
        for (int s = 0; s < pmax - i; s++) {
          const int p    = (order == ORDER_REVERSED) ? pmax - s : i + 1 + s;
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
          if (order == ORDER_COUNT)
            nm += __popc(b);
          if (ip < 0) {
            ip = p;
            iq = j - 1 - ((order == ORDER_REVERSED) ? 31 - __clz(b) : __ffs(b) - 1);
          }
          if (order != ORDER_COUNT)
            break;
        }
      }

      /* multibranch closing (bt_mb_loop, dangles 2): r = i+2 .. j-3 */
      int kr = -1;
      if ((order != ORDER_UPSTREAM || (!hp && ip < 0)) && (i + 2 * (turn + 2) < j)) {
        const int tm = P2->rtype[type];
        const int e  = cij - P2->MLclosing - E_MLstem_device(tm, S[j - 1], S[i + 1], P2);
        const int ii = i, jj = j;
        kr = pick(j - i - 4, order, &nm, [&](int kk) {
          const int rr = ii + 2 + kk;
          return e == fml[Indx(ii + 1, rr)] + fml[Indx(rr + 1, jj - 1)];
        });
      }
      if (nm > 1)
        nties++;

      if (order == ORDER_REVERSED) {
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

  if (lane == 0) {
    B.status[r] = fail;
    B.ties[r]   = nties;
  }
}

/* ======================================== host ======================================== */

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static char *make_seq(const char *mode, int L, unsigned *st) {
  char *s = (char *)malloc(L + 1);
  for (int i = 0; i < L; i++) {
    *st = *st * 1103515245u + 12345u;
    const unsigned x = (*st >> 8);
    if (!strcmp(mode, "au")) {
      s[i] = "AU"[x & 1];
    } else if (!strcmp(mode, "grich") && x % 9 == 0 && i + 7 < L) {
      memcpy(s + i, "GGGAGGG", 7);
      i += 6;
    } else {
      s[i] = "ACGU"[x & 3];
    }
  }
  s[L] = 0;
  return s;
}

struct host_rec_t {
  int    n;
  char  *seq;
  char  *ref;        /* upstream's structure */
  int    mfe;        /* f5[n] */
  int   *c, *fml, *f5;
  short *S;
};

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: proto_device_bt MODE NSEQ L SEED [negctl]\n");
    return 2;
  }
  const char *mode   = argv[1];
  const int   nrec   = atoi(argv[2]), L = atoi(argv[3]);
  unsigned    seed   = (unsigned)atoi(argv[4]);
  const int   negctl = (argc > 5) ? atoi(argv[5]) : 0;
  setenv("RNA_GPU", "0", 1);   /* the reference is upstream's own host fold */

  vrna_md_t md;
  vrna_md_set_default(&md);   /* dangles 2, linear, no noLP, no G-quads, default salt */

  host_rec_t *R = (host_rec_t *)calloc(nrec, sizeof(host_rec_t));
  for (int r = 0; r < nrec; r++) {
    R[r].seq = make_seq(mode, L, &seed);
    R[r].n   = L;
  }

  /* ---- upstream folds (parallel over records) ---- */
  const double tf0 = now_s();
  #pragma omp parallel for schedule(dynamic, 1)
  for (int r = 0; r < nrec; r++) {
    vrna_fold_compound_t *fc = vrna_fold_compound(R[r].seq, &md, VRNA_OPTION_MFE);
    const int             n  = (int)fc->length;
    const size_t          cells = (size_t)(n + 1) * (n + 2) / 2;
    R[r].ref = (char *)malloc(n + 1);
    vrna_mfe(fc, R[r].ref);
    if (!fc->matrices || !fc->matrices->c || !fc->matrices->fML || !fc->matrices->f5) {
      fprintf(stderr, "record %d: upstream left no matrices\n", r);
      exit(2);
    }
    R[r].mfe = fc->matrices->f5[n];
    R[r].c   = (int *)malloc(cells * sizeof(int));
    R[r].fml = (int *)malloc(cells * sizeof(int));
    R[r].f5  = (int *)malloc((n + 2) * sizeof(int));
    R[r].S   = (short *)malloc((n + 2) * sizeof(short));
    memcpy(R[r].c, fc->matrices->c, cells * sizeof(int));
    memcpy(R[r].fml, fc->matrices->fML, cells * sizeof(int));
    memcpy(R[r].f5, fc->matrices->f5, (n + 1) * sizeof(int));
    memcpy(R[r].S, fc->sequence_encoding, (n + 2) * sizeof(short));
    vrna_fold_compound_free(fc);
  }
  const double fold_s = now_s() - tf0;

  /* ---- upstream's trace alone, serially, on the same matrices: f5 + backtrack (the host's cost today,
   *      without the fetch). A fresh compound per record, its c/fML replaced by the copies. ---- */
  const int nhost = nrec < 8 ? nrec : 8;
  double    host_bt = 0;
  for (int r = 0; r < nhost; r++) {
    vrna_fold_compound_t *fc = vrna_fold_compound(R[r].seq, &md, VRNA_OPTION_MFE);
    const int             n  = (int)fc->length;
    const size_t          cells = (size_t)(n + 1) * (n + 2) / 2;
    if (!fc->matrices)
      vrna_mx_mfe_add(fc, VRNA_MX_DEFAULT, 0);
    memcpy(fc->matrices->c, R[r].c, cells * sizeof(int));
    memcpy(fc->matrices->fML, R[r].fml, cells * sizeof(int));
    char  *s  = (char *)malloc(n + 1);
    const double t0 = now_s();
    vrna_mfe_exterior_f5(fc);
    vrna_backtrack5(fc, n, s);
    host_bt += now_s() - t0;
    if (strcmp(s, R[r].ref)) {
      fprintf(stderr, "record %d: upstream's own re-trace differs from its fold -- harness error\n", r);
      exit(2);
    }
    free(s);
    vrna_fold_compound_free(fc);
  }

  /* ---- device parameters, exactly as load_param() / load_param2() / init_gpu3 build them ---- */
  vrna_fold_compound_t *fc0 = vrna_fold_compound(R[0].seq, &md, VRNA_OPTION_MFE);
  const vrna_param_t   *Pp  = fc0->params;
  cuda_param_t         *hP  = (cuda_param_t *)calloc(1, sizeof(cuda_param_t));
  memcpy(hP->stack, Pp->stack, sizeof(hP->stack));
  hP->ninio2     = Pp->ninio[2];
  hP->lxc        = (float)Pp->lxc;
  hP->TerminalAU = Pp->TerminalAU;
  memcpy(hP->bulge, Pp->bulge, sizeof(hP->bulge));
  memcpy(hP->internal_loop, Pp->internal_loop, sizeof(hP->internal_loop));
  memcpy(hP->mismatchI, Pp->mismatchI, sizeof(hP->mismatchI));
  memcpy(hP->mismatch1nI, Pp->mismatch1nI, sizeof(hP->mismatch1nI));
  memcpy(hP->mismatch23I, Pp->mismatch23I, sizeof(hP->mismatch23I));
  memcpy(hP->int11, Pp->int11, sizeof(hP->int11));
  memcpy(hP->int21, Pp->int21, sizeof(hP->int21));
  memcpy(hP->int22, Pp->int22, sizeof(hP->int22));
  hP->SaltStack   = Pp->SaltStack;
  hP->max_ninio   = MAX_NINIO;
  memcpy(hP->rtype, Pp->model_details.rtype, 8 * sizeof(int));
  hP->noGUclosure = Pp->model_details.noGUclosure;
  hP->dangles     = Pp->model_details.dangles;
  rnafold_build_salt_table(Pp, MAXLOOP + 1, hP->SaltLoop);

  cuda_param2_t *hP2 = (cuda_param2_t *)calloc(1, sizeof(cuda_param2_t));
  memcpy(hP2->hairpin, Pp->hairpin, 31 * sizeof(int));
  memcpy(hP2->mismatchH, Pp->mismatchH, sizeof(hP2->mismatchH));
  memcpy(hP2->mismatchM, Pp->mismatchM, sizeof(hP2->mismatchM));
  memcpy(hP2->MLintern, Pp->MLintern, sizeof(hP2->MLintern));
  hP2->MLclosing  = Pp->MLclosing;
  hP2->MLbase     = Pp->MLbase;
  memcpy(hP2->rtype, Pp->model_details.rtype, 8 * sizeof(int));
  hP2->TerminalAU = Pp->TerminalAU;
  hP2->lxc        = (float)Pp->lxc;
  hP2->special_hp = Pp->model_details.special_hp;
  hP2->dangles    = Pp->model_details.dangles;
  memcpy(hP2->Tetraloop_E, Pp->Tetraloop_E, 200 * sizeof(int));
  memcpy(hP2->Tetraloops, Pp->Tetraloops, 1401);
  memcpy(hP2->Triloop_E, Pp->Triloop_E, 40 * sizeof(int));
  memcpy(hP2->Triloops, Pp->Triloops, 241);
  memcpy(hP2->Hexaloop_E, Pp->Hexaloop_E, 40 * sizeof(int));
  memcpy(hP2->Hexaloops, Pp->Hexaloops, 1801);
  memcpy(hP2->stack, Pp->stack, sizeof(hP2->stack));

  ext_param_t hX;
  memcpy(hX.mismatchExt, Pp->mismatchExt, sizeof(hX.mismatchExt));
  memcpy(hX.dangle5, Pp->dangle5, sizeof(hX.dangle5));
  memcpy(hX.dangle3, Pp->dangle3, sizeof(hX.dangle3));
  hX.TerminalAU = Pp->TerminalAU;

  char pair_[8 * 8];
  for (int x = 0; x < 8; x++)
    for (int y = 0; y < 8; y++)
      pair_[x * 8 + y] = (char)Pp->model_details.pair[x][y];

  int maxn = 0;
  for (int r = 0; r < nrec; r++)
    if (R[r].n > maxn)
      maxn = R[r].n;
  int *salt = (int *)malloc((maxn + 2) * sizeof(int));
  rnafold_build_salt_table(Pp, maxn, salt);
  int *hp_len = (int *)malloc((maxn + 2) * sizeof(int));
  for (int s = 0; s < maxn + 2; s++)
    hp_len[s] = (s <= 30) ? Pp->hairpin[s] : Pp->hairpin[30] + (int)(Pp->lxc * log((double)s / 30.));

  /* ---- batch layout ---- */
  long long *tri_off = (long long *)malloc((nrec + 1) * sizeof(long long));
  long long *seq_off = (long long *)malloc((nrec + 1) * sizeof(long long));
  long long *sp_off  = (long long *)malloc((nrec + 1) * sizeof(long long));
  int       *n_H     = (int *)malloc(nrec * sizeof(int));
  tri_off[0] = seq_off[0] = sp_off[0] = 0;
  for (int r = 0; r < nrec; r++) {
    const long long n = R[r].n;
    n_H[r]         = (int)n;
    tri_off[r + 1] = tri_off[r] + (n + 1) * (n + 2) / 2;
    seq_off[r + 1] = seq_off[r] + n + 2;
    sp_off[r + 1]  = sp_off[r] + (n + 2 + 9) / 10;
  }
  const long long cells = tri_off[nrec], slots = seq_off[nrec], words = sp_off[nrec];
  int          *hc   = (int *)malloc(cells * sizeof(int));
  int          *hfml = (int *)malloc(cells * sizeof(int));
  short        *hS   = (short *)calloc(slots, sizeof(short));
  char         *hseq = (char *)calloc(slots, 1);
  unsigned int *hSp  = (unsigned int *)calloc(words, sizeof(unsigned int));
  for (int r = 0; r < nrec; r++) {
    const long long n = R[r].n, nc = (n + 1) * (n + 2) / 2;
    memcpy(hc + tri_off[r], R[r].c, nc * sizeof(int));
    memcpy(hfml + tri_off[r], R[r].fml, nc * sizeof(int));
    memcpy(hS + seq_off[r], R[r].S, (n + 2) * sizeof(short));
    memcpy(hseq + seq_off[r], R[r].seq, n);
    for (long long i = 0; i <= n + 1; i++)
      hSp[sp_off[r] + i / 10] |= ((unsigned int)R[r].S[i] & 7u) << ((i % 10) * 3);
    free(R[r].c);
    free(R[r].fml);
  }

  batch_t B;
  B.nrec = nrec;
  int *d_n; long long *d_tri, *d_seqo, *d_spo;
  int *d_c, *d_fml, *d_f5, *d_stk, *d_pt, *d_status, *d_ties, *d_salt, *d_hplen;
  short *d_S; char *d_seq, *d_pair; unsigned int *d_Sp;
  cuda_param_t *d_P; cuda_param2_t *d_P2; ext_param_t *d_X;
  CK(cudaMalloc(&d_n, nrec * sizeof(int)));
  CK(cudaMalloc(&d_tri, (nrec + 1) * sizeof(long long)));
  CK(cudaMalloc(&d_seqo, (nrec + 1) * sizeof(long long)));
  CK(cudaMalloc(&d_spo, (nrec + 1) * sizeof(long long)));
  CK(cudaMalloc(&d_c, cells * sizeof(int)));
  CK(cudaMalloc(&d_fml, cells * sizeof(int)));
  CK(cudaMalloc(&d_S, slots * sizeof(short)));
  CK(cudaMalloc(&d_seq, slots));
  CK(cudaMalloc(&d_Sp, words * sizeof(unsigned int)));
  CK(cudaMalloc(&d_f5, slots * sizeof(int)));
  CK(cudaMalloc(&d_stk, 3 * slots * sizeof(int)));
  CK(cudaMalloc(&d_pt, slots * sizeof(int)));
  CK(cudaMalloc(&d_status, nrec * sizeof(int)));
  CK(cudaMalloc(&d_ties, nrec * sizeof(int)));
  CK(cudaMalloc(&d_salt, (maxn + 2) * sizeof(int)));
  CK(cudaMalloc(&d_hplen, (maxn + 2) * sizeof(int)));
  CK(cudaMalloc(&d_pair, 64));
  CK(cudaMalloc(&d_P, sizeof(cuda_param_t)));
  CK(cudaMalloc(&d_P2, sizeof(cuda_param2_t)));
  CK(cudaMalloc(&d_X, sizeof(ext_param_t)));
  hP2->hp_len = d_hplen;   /* RNA_HP_TABLE, as init_gpu3 attaches it */
  CK(cudaMemcpy(d_n, n_H, nrec * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_tri, tri_off, (nrec + 1) * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_seqo, seq_off, (nrec + 1) * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_spo, sp_off, (nrec + 1) * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_c, hc, cells * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_fml, hfml, cells * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_S, hS, slots * sizeof(short), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_seq, hseq, slots, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_Sp, hSp, words * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_salt, salt, (maxn + 2) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_hplen, hp_len, (maxn + 2) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pair, pair_, 64, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_P, hP, sizeof(cuda_param_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_P2, hP2, sizeof(cuda_param2_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_X, &hX, sizeof(ext_param_t), cudaMemcpyHostToDevice));
  B.n_H = d_n; B.tri_off = d_tri; B.seq_off = d_seqo; B.sp_off = d_spo;
  B.c = d_c; B.fml = d_fml; B.S = d_S; B.seq = d_seq; B.Sp = d_Sp; B.pair_ = d_pair;
  B.P = d_P; B.P2 = d_P2; B.X = d_X; B.salt_loop = d_salt;
  B.f5 = d_f5; B.stk = d_stk; B.pt = d_pt; B.status = d_status; B.ties = d_ties;

  const int   threads = 128, blocks = (nrec * 32 + threads - 1) / threads;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  int   *pt     = (int *)malloc(slots * sizeof(int));
  int   *f5     = (int *)malloc(slots * sizeof(int));
  int   *status = (int *)malloc(nrec * sizeof(int));
  int   *ties   = (int *)malloc(nrec * sizeof(int));
  float  ms_all = 0, ms_f5 = 0;

  /* warm-up, then timed: f5 alone, and f5 + trace */
  bt_kernel<<<blocks, threads>>>(B, ORDER_UPSTREAM, 3);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(e0));
  bt_kernel<<<blocks, threads>>>(B, ORDER_UPSTREAM, 1);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&ms_f5, e0, e1));
  CK(cudaEventRecord(e0));
  bt_kernel<<<blocks, threads>>>(B, ORDER_UPSTREAM, 3);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaGetLastError());
  CK(cudaEventElapsedTime(&ms_all, e0, e1));

  CK(cudaMemcpy(pt, d_pt, slots * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(f5, d_f5, slots * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(status, d_status, nrec * sizeof(int), cudaMemcpyDeviceToHost));

  long long f5_bad = 0;
  int       st_bad = 0, mfe_bad = 0, failed = 0, nties = 0, tie_recs = 0;
  for (int r = 0; r < nrec; r++) {
    const int n = R[r].n;
    for (int j = 0; j <= n; j++)
      if (f5[seq_off[r] + j] != R[r].f5[j])
        f5_bad++;
    if (f5[seq_off[r] + n] != R[r].mfe)
      mfe_bad++;
    if (status[r]) {
      failed++;
      if (failed <= 3)
        fprintf(stderr, "record %d: device trace failed, code %d\n", r, status[r]);
    }
    short *ref = vrna_ptable(R[r].ref);
    int    d   = 0;
    for (int i = 1; i <= n; i++)
      if (pt[seq_off[r] + i] != ref[i])
        d = 1;
    st_bad += d;
    free(ref);
  }

  /* the count pass: same picks, every match counted (positive evidence that ties were exercised) */
  bt_kernel<<<blocks, threads>>>(B, ORDER_COUNT, 3);
  CK(cudaDeviceSynchronize());
  int *pt2 = (int *)malloc(slots * sizeof(int));
  CK(cudaMemcpy(pt2, d_pt, slots * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ties, d_ties, nrec * sizeof(int), cudaMemcpyDeviceToHost));
  int count_bad = memcmp(pt, pt2, slots * sizeof(int)) != 0;
  for (int r = 0; r < nrec; r++) {
    nties += ties[r];
    tie_recs += ties[r] > 0;
  }

  printf("proto_device_bt %s L=%d n=%d seed=%s: f5 %lld cells differ, mfe %d differ, %d/%d structures differ, "
         "%d failed, count-pass %s; ties %d in %d/%d records; device %.3f ms/batch (f5 %.3f), "
         "%.4f ms/record; host trace %.3f ms/record (f5 + backtrack, serial, %d records); fold %.1f s\n",
         mode, L, nrec, argv[4], f5_bad, mfe_bad, st_bad, nrec, failed, count_bad ? "DIFFERS" : "same",
         nties, tie_recs, nrec, ms_all, ms_f5, ms_all / nrec, 1e3 * host_bt / nhost, nhost, fold_s);
  int rc = (f5_bad || mfe_bad || st_bad || failed || count_bad) ? 1 : 0;

  if (negctl) {
    bt_kernel<<<blocks, threads>>>(B, ORDER_REVERSED, 3);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(pt2, d_pt, slots * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(status, d_status, nrec * sizeof(int), cudaMemcpyDeviceToHost));
    int nd = 0, off_mfe = 0, nfail = 0;
    for (int r = 0; r < nrec; r++) {
      const int n = R[r].n;
      nfail += status[r] != 0;
      if (memcmp(pt + seq_off[r], pt2 + seq_off[r], (n + 1) * sizeof(int)) == 0)
        continue;
      nd++;
      char *s = (char *)malloc(n + 1);
      for (int i = 1; i <= n; i++) {
        const int p = pt2[seq_off[r] + i];
        s[i - 1] = p == 0 ? '.' : (p > i ? '(' : ')');
      }
      s[n] = 0;
      const float e = vrna_eval_structure_simple(R[r].seq, s);
      if ((int)lround(e * 100.) != R[r].mfe)
        off_mfe++;
      free(s);
    }
    printf("proto_device_bt %s L=%d n=%d seed=%s NEGCTL (reversed order): %d/%d structures differ, "
           "%d off the MFE, %d failed -- %s\n", mode, L, nrec, argv[4], nd, nrec, off_mfe, nfail,
           (nd > 0 && off_mfe == 0 && nfail == 0) ? "BITES" : "DOES NOT BITE");
    if (!(nd > 0 && off_mfe == 0 && nfail == 0))
      rc |= 2;
  }
  vrna_fold_compound_free(fc0);
  return rc;
}

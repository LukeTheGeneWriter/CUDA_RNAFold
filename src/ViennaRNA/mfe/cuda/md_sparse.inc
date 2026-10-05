/* md_sparse.inc -- sparse multiloop decomposition, PORT_SPARSE_MD.md.
 *
 * STAGE S1: the candidate lists are built every row and the sparse split is computed BESIDE
 * the dense md and compared with it, cell for cell. Nothing reads the sparse result yet --
 * the dense md stays authoritative -- so this stage can change no output. It exists to
 * prove, on the device and on the GPU's own left operand, what tools/md_sparse_equiv.c
 * proved on upstream's matrices.
 *
 *   RNA_MD_SPARSE_SELFTEST=1   build the lists and compare (forces CUDA graph capture off)
 *   RNA_MD_SPARSE_NEGCTL=1     drop the scan term from the sparse side -- must go red
 *   RNA_MD_SPARSE_RHO=r        list capacity per column, min(j, ceil(r*j)+8); default 0.25
 *
 * THE RULE (section 2 of the plan). For row i and column j, with A the left operand md
 * itself reads (row i's fML WITHOUT the split term) and B the finished fML(k,j):
 *
 *   DMLi(i,j) = min( up_ml(j) ? DMLi(i,j-1) + MLbase : INF ,
 *                    min over CANDIDATES k of A(i,k-1) + fML(k,j) )
 *
 * where (k,j) is a candidate unless [up_ml(k) and fML(k,j) >= fML(k+1,j) + MLbase] or
 * [up_ml(j) and fML(k,j) >= fML(k,j-1) + MLbase].
 *
 * THE LISTS. One append-only list per (record, column j) of (k, fML(k,j)). Rows run in
 * descending i, so row k appends at most one entry to each column, after its values are final
 * (md_sparse_append, after the row closes). The test reads ROW-shaped data only: fML(k,j) and
 * fML(k,j-1) from this row's energy_min/dml, fML(k+1,j) from d_sp_prev, this file's own copy
 * of the previous row. It never reads the triangle, so RNA_ROW_BATCH's delayed triangle
 * writes cannot make it stale. Values follow the TRIANGLE's rule (md_close_row_cell): the
 * min of energy_min and dml where j-i-turn-1 >= turn+1, energy_min alone nearer the diagonal
 * -- because those are the values the dense md reads as B.
 *
 * A list that would exceed its capacity is marked -1 (overflowed) and its column falls back
 * to the dense value. Counted and printed, so the fallback has positive evidence. */

static int*                d_sp_len  = NULL;   /* per (record, column), row layout: entries, or -1 */
static int*                d_sp_k    = NULL;   /* entry k */
static int*                d_sp_v    = NULL;   /* entry fML(k,j) */
static int*                d_sp_prev = NULL;   /* previous row's triangle values, row layout */
static size_t*             d_sp_col  = NULL;   /* capacity prefix over columns 0..maxlen+1 */
static size_t*             d_sp_base = NULL;   /* per-record start in d_sp_k / d_sp_v */
static unsigned long long* d_sp_stat = NULL;   /* MD_SP_NSTAT counters, below */
static int*                d_sp_bad  = NULL;   /* the first MD_SP_NBAD mismatches, 5 ints each */
static int                 g_sp_on   = 0;      /* this chunk */
static size_t              g_sp_entries = 0;   /* capacity, entries */

#define MD_SP_NSTAT 6   /* 0 cells compared, 1 mismatches, 2 overflowed columns, 3 entries,
                           4 finite cells tested, 5 overflow fallbacks used */
#define MD_SP_NBAD  4
static unsigned long long  g_sp_tot[MD_SP_NSTAT];
static int                 g_sp_badh[MD_SP_NBAD * 5];
static int                 g_sp_nbadh = 0;
static size_t              g_sp_bytes_max = 0;
static int                 g_sp_chunks = 0;

extern "C" int hp_mb_mlbase(void);   /* hp_mb_loop.cu: the parameter set's MLbase */
static int md_sparse_on(void);       /* S2, below */

static int
md_sparse_selftest(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE_SELFTEST");
    v = (e && e[0] && e[0] != '0') ? 1 : 0;
  }
  return v;
}

static int
md_sparse_negctl(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE_NEGCTL");
    v = (e && e[0] && e[0] != '0') ? 1 : 0;
  }
  return v;
}

static double
md_sparse_rho(void)
{
  static double v = -1.0;
  if (v < 0.0) {
    const char *e = getenv("RNA_MD_SPARSE_RHO");
    v = (e && e[0]) ? atof(e) : 0.25;
    if (!(v >= 0.0)) v = 0.25;
    if (v > 1.0)     v = 1.0;
  }
  return v;
}

static size_t
md_sparse_cap(const int j)
{
  if (j <= 0) return 0;
  const size_t c = (size_t)ceil(md_sparse_rho() * (double)j) + 8;
  return c < (size_t)j ? c : (size_t)j;
}

static void
md_sparse_report(void)
{
  const unsigned long long *t = g_sp_tot;
  if (md_sparse_on()) {
    fprintf(stderr, "%-24s RNA_MD_SPARSE%s: %d chunk(s); %llu candidates of %llu finite fML(k,j) (%.2f %%); "
                    "%llu column(s) overflowed at rho %.2f, %llu md cell(s) took the dense fallback; "
                    "lists %.1f MB\n",
            __FILE__, md_sparse_negctl() ? " (NEGCTL)" : "", g_sp_chunks, t[3], t[4],
            t[4] ? 100.0 * (double)t[3] / (double)t[4] : 0.0, t[2], md_sparse_rho(), t[5],
            (double)g_sp_bytes_max / 1048576.0);
    return;
  }
  fprintf(stderr, "%-24s RNA_MD_SPARSE_SELFTEST%s: %d chunk(s), %llu cells compared, %llu MISMATCHING; "
                  "%llu candidates of %llu finite fML(k,j) (%.2f %%); %llu column(s) overflowed at "
                  "rho %.2f, %llu cell(s) took the dense fallback; lists %.1f MB\n",
          __FILE__, md_sparse_negctl() ? " (NEGCTL: scan term dropped)" : "", g_sp_chunks,
          t[0], t[1], t[3], t[4], t[4] ? 100.0 * (double)t[3] / (double)t[4] : 0.0,
          t[2], md_sparse_rho(), t[5], (double)g_sp_bytes_max / 1048576.0);
  for (int b = 0; b < g_sp_nbadh; b++)
    fprintf(stderr, "%-24s   mismatch H=%d i=%d j=%d dense %d sparse %d\n", __FILE__,
            g_sp_badh[5*b], g_sp_badh[5*b+1], g_sp_badh[5*b+2], g_sp_badh[5*b+3], g_sp_badh[5*b+4]);
}

/* init_gpu(): the chunk's lists. row_off_H is the host table, so len_H is known here. */
static void
md_sparse_alloc(const int nfiles, const size_t* row_off_H)
{
  g_sp_on = 0;
  if (!md_sparse_selftest() && !md_sparse_on()) return;

  const char *why = NULL;
  const char *knob = md_sparse_on() ? "RNA_MD_SPARSE" : "RNA_MD_SPARSE_SELFTEST";
  if (rnafold_continuous_flow())      why = "continuous flow (records on different rows)";
  else if (rnafold_slot_flow())       why = "slot flow (slots are handed over mid-sweep)";
  else if (rnafold_megakernel())      why = "the megakernel (its own fused md)";
  else if (rnafold_md_prune())        why = "RNA_MD_PRUNE (a different dense-skipping scheme)";
  else if (rnafold_md_block())        why = "the blocked md (RNA_MD_BLOCK)";
  else if (rnafold_md_tile_cb() > 0)  why = "the tile path (RNA_MD_TILE_CB)";
  if (why) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "%-24s %s REFUSED: %s -- the dense md runs\n", __FILE__, knob, why); }
    return;
  }
  if (md_sparse_on() && md_sparse_selftest()) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "%-24s RNA_MD_SPARSE_SELFTEST IGNORED: with RNA_MD_SPARSE=1 there "
                                           "is no dense md to compare with\n", __FILE__); }
  }

  int maxlen = 0;
  for (int H = 0; H < nfiles; H++) {
    const int len_H = (int)(row_off_H[H+1] - row_off_H[H]) - 1;
    if (len_H > maxlen) maxlen = len_H;
  }
  size_t *col  = (size_t *)malloc(sizeof(size_t) * (size_t)(maxlen + 2));
  size_t *base = (size_t *)malloc(sizeof(size_t) * (size_t)(nfiles + 1));
  col[0] = 0;
  for (int j = 0; j <= maxlen; j++) col[j+1] = col[j] + md_sparse_cap(j);
  base[0] = 0;
  for (int H = 0; H < nfiles; H++) {
    const int len_H = (int)(row_off_H[H+1] - row_off_H[H]) - 1;
    base[H+1] = base[H] + col[len_H + 1];
  }
  g_sp_entries = base[nfiles];

  gpuErrchk( cudaMalloc((void **)&d_sp_col,  sizeof(size_t) * (size_t)(maxlen + 2)) );
  gpuErrchk( cudaMalloc((void **)&d_sp_base, sizeof(size_t) * (size_t)(nfiles + 1)) );
  gpuErrchk( cudaMemcpy(d_sp_col,  col,  sizeof(size_t) * (size_t)(maxlen + 2), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_sp_base, base, sizeof(size_t) * (size_t)(nfiles + 1), cudaMemcpyHostToDevice) );
  free(col); free(base);

  gpuErrchk( cudaMalloc((void **)&d_sp_k,    sizeof(int) * (g_sp_entries ? g_sp_entries : 1)) );
  gpuErrchk( cudaMalloc((void **)&d_sp_v,    sizeof(int) * (g_sp_entries ? g_sp_entries : 1)) );
  gpuErrchk( cudaMalloc((void **)&d_sp_len,  sizeof(int) * g_row_total) );
  gpuErrchk( cudaMalloc((void **)&d_sp_prev, sizeof(int) * g_row_total) );
  gpuErrchk( cudaMalloc((void **)&d_sp_stat, sizeof(unsigned long long) * MD_SP_NSTAT) );
  gpuErrchk( cudaMalloc((void **)&d_sp_bad,  sizeof(int) * MD_SP_NBAD * 5) );
  gpuErrchk( cudaMemset(d_sp_len,  0, sizeof(int) * g_row_total) );
  gpuErrchk( cudaMemset(d_sp_stat, 0, sizeof(unsigned long long) * MD_SP_NSTAT) );
  {
    /* INF, exactly: the previous row of the first row is no row at all */
    int *h = (int *)malloc(sizeof(int) * g_row_total);
    for (size_t t = 0; t < g_row_total; t++) h[t] = INF;
    gpuErrchk( cudaMemcpy(d_sp_prev, h, sizeof(int) * g_row_total, cudaMemcpyHostToDevice) );
    free(h);
  }
  {
    const size_t bytes = g_sp_entries * 2 * sizeof(int) + g_row_total * 2 * sizeof(int);
    if (bytes > g_sp_bytes_max) g_sp_bytes_max = bytes;
  }
  g_sp_on = 1;
  g_sp_chunks++;
  static int said = 0;
  if (!said) {
    said = 1;
    atexit(md_sparse_report);
    if (md_sparse_on())
      fprintf(stderr, "%-24s RNA_MD_SPARSE ACTIVE: md reads only candidate split points (lists rebuilt "
                      "every row) plus a row scan, in place of the dense md (rho %.2f%s)\n",
              __FILE__, md_sparse_rho(), md_sparse_negctl() ? ", NEGCTL: scan term dropped -- WRONG ANSWERS EXPECTED" : "");
    else
      fprintf(stderr, "%-24s RNA_MD_SPARSE_SELFTEST ACTIVE: candidate lists built every row and the "
                      "sparse split compared with the dense md, cell for cell (rho %.2f%s)\n",
              __FILE__, md_sparse_rho(), md_sparse_negctl() ? ", NEGCTL: scan term dropped" : "");
  }
}

/* teardown_gpu(): fold this chunk's counters into the totals, then free by the pointer. */
static void
md_sparse_free(void)
{
  if (d_sp_stat) {
    unsigned long long s[MD_SP_NSTAT];
    int b[MD_SP_NBAD * 5];
    gpuErrchk( cudaMemcpy(s, d_sp_stat, sizeof(s), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(b, d_sp_bad,  sizeof(b), cudaMemcpyDeviceToHost) );
    const unsigned long long before = g_sp_tot[1];
    for (int t = 0; t < MD_SP_NSTAT; t++) g_sp_tot[t] += s[t];
    /* this chunk's first mismatches, while there is room */
    for (unsigned long long m = 0; m < s[1] && m < MD_SP_NBAD && g_sp_nbadh < MD_SP_NBAD; m++, g_sp_nbadh++)
      memcpy(&g_sp_badh[5 * g_sp_nbadh], &b[5 * m], 5 * sizeof(int));
    (void)before;
  }
  if (d_sp_len)  { gpuErrchk( cudaFree(d_sp_len) );  d_sp_len  = NULL; }
  if (d_sp_k)    { gpuErrchk( cudaFree(d_sp_k) );    d_sp_k    = NULL; }
  if (d_sp_v)    { gpuErrchk( cudaFree(d_sp_v) );    d_sp_v    = NULL; }
  if (d_sp_prev) { gpuErrchk( cudaFree(d_sp_prev) ); d_sp_prev = NULL; }
  if (d_sp_col)  { gpuErrchk( cudaFree(d_sp_col) );  d_sp_col  = NULL; }
  if (d_sp_base) { gpuErrchk( cudaFree(d_sp_base) ); d_sp_base = NULL; }
  if (d_sp_stat) { gpuErrchk( cudaFree(d_sp_stat) ); d_sp_stat = NULL; }
  if (d_sp_bad)  { gpuErrchk( cudaFree(d_sp_bad) );  d_sp_bad  = NULL; }
  g_sp_on = 0;
}

/* The value the TRIANGLE holds at (i,jj) once row i closes -- md_close_row_cell's rule. */
__device__ inline int
md_sp_tri_value(const int* __restrict__ energy_min, const int* __restrict__ dml,
                const size_t o, const int i, const int jj, const int turn)
{
  if (jj < i + turn + 1) return INF;
  const int e = energy_min[o + jj];
  if (jj - i - turn - 1 >= turn + 1) {
    const int d = dml[o + jj];
    return d < e ? d : e;
  }
  return e;
}

/* After row i closes: test every (k=i, j) of the row and append the candidates. One thread
 * per cell, md_close_row's grid. Each column gets at most one append per row, from the one
 * thread that owns (i,j), so there is nothing to race on. */
__global__ void
md_sparse_append_kernel(const int nfiles, const int i_row, const int turn, const int mlbase,
                        const int* __restrict__ energy_min, const int* __restrict__ dml,
                        int* __restrict__ sp_prev, int* __restrict__ sp_len,
                        int* __restrict__ sp_k, int* __restrict__ sp_v,
                        const size_t* __restrict__ sp_col, const size_t* __restrict__ sp_base,
                        const size_t* __restrict__ row_off_H, const size_t* __restrict__ size_off_H,
                        const size_t total, const int* __restrict__ i_H,
                        const char* __restrict__ up_ml_ok, const size_t* __restrict__ seq_off_H,
                        unsigned long long* __restrict__ stat)
{
  const size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= total) return;
  const int H = flatten_index_to_H(m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = (int)mj + i + turn + 1;
  const size_t o = row_off_H[H];

  const int v  = md_sp_tri_value(energy_min, dml, o, i, j,     turn);
  const int vr = md_sp_tri_value(energy_min, dml, o, i, j - 1, turn);   /* fML(i, j-1)  */
  const int vl = sp_prev[o + j];                                        /* fML(i+1, j)  */
  sp_prev[o + j] = v;   /* this row is the next row's "previous"; only this thread reads it */

  if (v >= INF/2) return;
  atomicAdd(&stat[4], 1ULL);

  const int upk = up_ml_ok[seq_off_H[H] + i] != 0;
  const int upj = up_ml_ok[seq_off_H[H] + j] != 0;
  const int domL = upk && vl < INF/2 && v >= vl + mlbase;
  const int domR = upj && vr < INF/2 && v >= vr + mlbase;
  if (domL || domR) return;

  const int n = sp_len[o + j];
  if (n < 0) return;                                   /* already overflowed */
  const size_t cap = sp_col[j + 1] - sp_col[j];
  if ((size_t)n >= cap) {
    sp_len[o + j] = -1;
    atomicAdd(&stat[2], 1ULL);
    return;
  }
  const size_t at = sp_base[H] + sp_col[j] + (size_t)n;
  sp_k[at] = i;
  sp_v[at] = v;
  sp_len[o + j] = n + 1;
  atomicAdd(&stat[3], 1ULL);
}

/* After md(i), before append(i): the sparse split for every cell of row i, compared with the
 * dense dml md just wrote. One WARP per record, walking j upwards because of the scan term.
 * A selftest, so it is written for clarity, not speed. */
__global__ void
md_sparse_selftest_kernel(const int nfiles, const int turn, const int mlbase, const int negctl,
                          const int* __restrict__ fml_i,      /* md's own left-operand pointer */
                          const int* __restrict__ dml,
                          const int* __restrict__ sp_len, const int* __restrict__ sp_k,
                          const int* __restrict__ sp_v,
                          const size_t* __restrict__ sp_col, const size_t* __restrict__ sp_base,
                          const size_t* __restrict__ row_off_H, const size_t* __restrict__ side_off_H,
                          const int* __restrict__ i_H,
                          const char* __restrict__ up_ml_ok, const size_t* __restrict__ seq_off_H,
                          unsigned long long* __restrict__ stat, int* __restrict__ bad)
{
  const int H    = blockIdx.x;
  const int lane = threadIdx.x;
  if (H >= nfiles) return;
  const long long width = (long long)side_off_H[H+1] - (long long)side_off_H[H];
  if (width <= 0) return;
  const int    i  = i_H[H];
  const size_t o  = row_off_H[H];
  const int    lo = i + turn + 2;          /* md's first k */
  int S = INF;                             /* DMLi(i, j-1), the scan carry */

  for (long long x = 0; x < width; x++) {
    const int j = (int)x + i + 2*(turn+1) + 1;      /* md_cell's j, no column shift (refused) */
    const int dense = dml[o + j];
    const int n = sp_len[o + j];
    int best = INF;
    if (n < 0) {
      best = dense;                                  /* overflowed column: the dense fallback */
    } else {
      const size_t at = sp_base[H] + sp_col[j];
      for (int e = lane; e < n; e += 32) {
        const int k = sp_k[at + e];
        if (k < lo) continue;                        /* rows i+1 .. i+turn+1: not split points */
        const int a = fml_i[o + (size_t)(k - lo)];
        const int b = sp_v[at + e];
        if (a < INF/2 && b < INF/2 && a + b < best) best = a + b;
      }
      for (int off = 16; off > 0; off >>= 1) {
        const int other = __shfl_down_sync(0xffffffffu, best, off);
        best = other < best ? other : best;
      }
      best = __shfl_sync(0xffffffffu, best, 0);
    }
    const int scan = (!negctl && up_ml_ok[seq_off_H[H] + j] && S < INF/2) ? S + mlbase : INF;
    const int s = scan < best ? scan : best;
    if (lane == 0) {
      atomicAdd(&stat[0], 1ULL);
      if (n < 0) atomicAdd(&stat[5], 1ULL);
      const int same = (s >= INF/2 && dense >= INF/2) || s == dense;
      if (!same) {
        const unsigned long long b = atomicAdd(&stat[1], 1ULL);
        if (b < MD_SP_NBAD) {
          bad[5*b] = H; bad[5*b+1] = i; bad[5*b+2] = j; bad[5*b+3] = dense; bad[5*b+4] = s;
        }
      }
    }
    S = s;
  }
}

/* md_row_chain hooks. `fml_i` is the pointer md itself read this row. */
static void
md_sparse_check(const int nfiles, const int i, const int turn, const int* fml_i)
{
  if (!g_sp_on || md_sparse_on()) return;   /* with RNA_MD_SPARSE there is no dense md to check */
  rnafold_mk_ptrs_t p;
  memset(&p, 0, sizeof(p));
  hp_mb_mk_ptrs(&p);
  md_sparse_selftest_kernel<<<nfiles, 32, 0, ISSUE_STREAM>>>(nfiles, turn, hp_mb_mlbase(),
      md_sparse_negctl(), fml_i, d_dml, d_sp_len, d_sp_k, d_sp_v, d_sp_col, d_sp_base,
      d_row_off_H, d_side_off_H, d_i_H, p.up_ml_ok, p.seq_off_H, d_sp_stat, d_sp_bad);
  gpuErrchk( cudaPeekAtLastError() );
  (void)i;
}

static void
md_sparse_append(const int nfiles, const int i, const int turn, const size_t* size_off_H)
{
  if (!g_sp_on) return;
  const size_t total = size_off_H[nfiles];
  if (total == 0) return;
  rnafold_mk_ptrs_t p;
  memset(&p, 0, sizeof(p));
  hp_mb_mk_ptrs(&p);
  const int nblocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  md_sparse_append_kernel<<<nblocks, BLOCK_SIZE, 0, ISSUE_STREAM>>>(nfiles, RNA_I_ROW(i), turn,
      hp_mb_mlbase(), d_energy_min, d_dml, d_sp_prev, d_sp_len, d_sp_k, d_sp_v, d_sp_col, d_sp_base,
      d_row_off_H, d_size_off_H, total, d_i_H, p.up_ml_ok, p.seq_off_H, d_sp_stat);
  gpuErrchk( cudaPeekAtLastError() );
}

/* ======================================================================================
 * STAGE S2: the sparse md REPLACES the dense one (RNA_MD_SPARSE=1).
 *
 * Two kernels per row, in place of modular_decomposition_kernel:
 *   md_sparse_cell_kernel   one warp per cell: the min over its column's candidates of
 *                           A(i,k-1) + fML(k,j), into dml. An overflowed column computes the
 *                           dense value instead, reading the triangle exactly as md reads B.
 *   md_sparse_scan_kernel   one block per record: DMLi(i,j) = min(DMLi(i,j-1) + c_j, best_j)
 *                           as a parallel prefix. Each step is the map x -> min(x + c_j, b_j),
 *                           c_j = MLbase where up_ml(j) and INF elsewhere, and the maps compose
 *                           associatively: (c1,b1) then (c2,b2) = (c1+c2, min(b1+c2, b2)).
 *                           Writes dml, and fm2 for circular folds.
 * Sums are INF-guarded, so a cell with no finite decomposition is exactly INF -- what upstream
 * stores -- where the dense md left a near-INF artefact (the clamp note in md_cell.inc).
 * ====================================================================================== */

static int
md_sparse_on(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE");
    v = (e && e[0] && e[0] != '0') ? 1 : 0;
  }
  return v;
}

__device__ __forceinline__ int
md_sp_add(const int a, const int b)
{
  return (a >= INF/2 || b >= INF/2) ? INF : a + b;
}

__global__ void
md_sparse_cell_kernel(const int nfiles, const int turn,
                      const int* __restrict__ fml_i,
                      const int* __restrict__ fml_j,
                      const short* __restrict__ fml_j16, const int* __restrict__ fml_b,
                      const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                      const int* __restrict__ sp_len, const int* __restrict__ sp_k,
                      const int* __restrict__ sp_v,
                      const size_t* __restrict__ sp_col, const size_t* __restrict__ sp_base,
                      const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                      const size_t* __restrict__ side_off_H, const size_t total,
                      const int* __restrict__ i_H,
                      int* __restrict__ best_out,
                      unsigned long long* __restrict__ stat)
{
  const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t    m    = (size_t)(gtid >> 5);
  const int       lane = (int)(gtid & 31);
  if (m >= total) return;                 /* a whole warp per cell, so whole warps leave */

  const int H  = flatten_index_to_H(m, side_off_H, nfiles);
  const int x  = (int)((long long)m - (long long)side_off_H[H]);
  const int i  = i_H[H];
  const int j  = x + i + 2*(turn+1) + 1;
  const int lo = i + turn + 2;
  const size_t o = row_off_H[H];
  const int n  = sp_len[o + j];
  int best = INF;

  if (n >= 0) {
    const size_t at = sp_base[H] + sp_col[j];
    for (int e = lane; e < n; e += 32) {
      const int k = sp_k[at + e];
      if (k < lo) continue;
      const int s = md_sp_add(fml_i[o + (size_t)(k - lo)], sp_v[at + e]);
      best = s < best ? s : best;
    }
  } else {
    /* overflowed column: md's own dense loop, B from the triangle */
    const long long ij0 = Indx(i, j) + (turn + 1) + 1;
    if (fml_j16) {
      const size_t bcell = base_off_H[H] + colb_off[j];
      for (int y = lane; y <= x; y += 32) {
        const int b = fml_decode(fml_j16, fml_b, tri_off_H[H] + (size_t)(ij0 + y),
                                 bcell + (size_t)((unsigned)(lo + y - 1) / FML_BLK));
        const int s = md_sp_add(fml_i[o + (size_t)y], b);
        best = s < best ? s : best;
      }
    } else {
      for (int y = lane; y <= x; y += 32) {
        const int s = md_sp_add(fml_i[o + (size_t)y], fml_j[tri_off_H[H] + (size_t)(ij0 + y)]);
        best = s < best ? s : best;
      }
    }
    if (lane == 0) atomicAdd(&stat[5], 1ULL);
  }
  for (int off = 16; off > 0; off >>= 1) {
    const int other = __shfl_down_sync(0xffffffffu, best, off);
    best = other < best ? other : best;
  }
  if (lane == 0) best_out[o + j] = best;
}

#define MD_SP_SCAN_T 256

__device__ __forceinline__ void
md_sp_compose(int &c1, int &b1, const int c2, const int b2)
{
  /* (c1,b1) then (c2,b2) */
  const int nb = md_sp_add(b1, c2);
  b1 = nb < b2 ? nb : b2;
  c1 = md_sp_add(c1, c2);
}

__global__ void
md_sparse_scan_kernel(const int nfiles, const int turn, const int mlbase, const int negctl,
                      int* __restrict__ dml,                   /* in: best_j, out: DMLi */
                      int* __restrict__ fm2,                   /* circular, or NULL */
                      const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                      const size_t* __restrict__ side_off_H, const int* __restrict__ i_H,
                      const char* __restrict__ up_ml_ok, const size_t* __restrict__ seq_off_H)
{
  __shared__ int sc[MD_SP_SCAN_T], sb[MD_SP_SCAN_T];
  const int H = blockIdx.x, t = threadIdx.x;
  if (H >= nfiles) return;
  const long long width = (long long)side_off_H[H+1] - (long long)side_off_H[H];
  if (width <= 0) return;                         /* uniform across the block */
  const int    i   = i_H[H];
  const int    j0  = i + 2*(turn+1) + 1;
  const size_t o   = row_off_H[H];
  const size_t sq  = seq_off_H[H];
  const long long per = (width + MD_SP_SCAN_T - 1) / MD_SP_SCAN_T;
  const long long x0  = (long long)t * per;
  const long long x1  = (x0 + per < width) ? x0 + per : width;

  /* this thread's segment as one map */
  int c = 0, b = INF;
  for (long long x = x0; x < x1; x++) {
    const int j  = (int)x + j0;
    const int cj = (!negctl && up_ml_ok[sq + j]) ? mlbase : INF;
    md_sp_compose(c, b, cj, dml[o + j]);
  }
  sc[t] = c; sb[t] = b;
  __syncthreads();
  /* inclusive Hillis-Steele over the segments, earlier then later */
  for (int off = 1; off < MD_SP_SCAN_T; off <<= 1) {
    int pc = 0, pb = INF;
    const int have = (t >= off);
    if (have) { pc = sc[t - off]; pb = sb[t - off]; }
    __syncthreads();
    if (have) {
      md_sp_compose(pc, pb, sc[t], sb[t]);
      sc[t] = pc; sb[t] = pb;
    }
    __syncthreads();
  }
  /* DMLi just before this segment: everything before it applied to INF, i.e. that map's b */
  int xv = (t == 0) ? INF : sb[t - 1];
  for (long long x = x0; x < x1; x++) {
    const int j  = (int)x + j0;
    const int cj = (!negctl && up_ml_ok[sq + j]) ? mlbase : INF;
    const int s  = md_sp_add(xv, cj);
    const int bj = dml[o + j];
    xv = s < bj ? s : bj;
    dml[o + j] = xv;
    if (fm2) fm2[tri_off_H[H] + Indx(i, j)] = (xv > INF/2) ? INF : xv;
  }
}

static int
md_sparse_live(void)
{
  return g_sp_on && md_sparse_on();
}

/* modular_decomposition_cuda()'s replacement for its dense launch. */
static void
md_sparse_md(const int nfiles, const int turn, const int* fml_i, const size_t total)
{
  rnafold_mk_ptrs_t p;
  memset(&p, 0, sizeof(p));
  hp_mb_mk_ptrs(&p);
  {
    const size_t nthreads = total * 32;
    const int    bs       = 256;
    const int    nb       = (int)((nthreads + bs - 1) / bs);
    md_sparse_cell_kernel<<<nb, bs, 0, ISSUE_STREAM>>>(nfiles, turn, fml_i, d_fml_j, d_fml_j16, d_fml_b,
        d_base_off_H, d_colb_off, d_sp_len, d_sp_k, d_sp_v, d_sp_col, d_sp_base,
        d_tri_off_H, d_row_off_H, d_side_off_H, total, d_i_H, d_dml, d_sp_stat);
    gpuErrchk( cudaPeekAtLastError() );
  }
  md_sparse_scan_kernel<<<nfiles, MD_SP_SCAN_T, 0, ISSUE_STREAM>>>(nfiles, turn, hp_mb_mlbase(),
      md_sparse_negctl(), d_dml, rnafold_circ_fm2_device(), d_tri_off_H, d_row_off_H, d_side_off_H,
      d_i_H, p.up_ml_ok, p.seq_off_H);
  gpuErrchk( cudaPeekAtLastError() );
}

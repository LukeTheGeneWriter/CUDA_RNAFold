/* md_sparse.inc -- sparse multiloop decomposition, PORT_SPARSE_MD.md.
 *
 * STAGE S1: the candidate lists are built every row and the sparse split is computed BESIDE
 * the dense md and compared with it, cell for cell. Nothing reads the sparse result yet --
 * the dense md stays authoritative -- so this stage can change no output. It exists to
 * prove, on the device and on the GPU's own left operand, what tools/md_sparse_equiv.c
 * proved on upstream's matrices.
 *
 *   RNA_MD_SPARSE=0|1          the sparse md (S2, below); DEFAULT ON since S5, =0 restores the dense md
 *   RNA_MD_SPARSE_SELFTEST=1   build the lists and compare (forces CUDA graph capture off)
 *   RNA_MD_SPARSE_NEGCTL=1     drop the scan term from the sparse side -- must go red
 *   RNA_MD_SPARSE_RHO=r        list capacity per column, min(j, ceil(r*j)+8); default 0.25
 *   RNA_MD_SPARSE_ENTRY=8|4    bytes per list entry (S3, below); default 8
 *   RNA_MD_SPARSE_DELTA_MAX=d  TEST ONLY: ENTRY=4 treats steps beyond +-d (default 32767) as not
 *                              fitting, to force early segments and (d = 0) slot exhaustion
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
 * to the dense value. Counted and printed, so the fallback has positive evidence.
 *
 * ENTRY FORMAT (S3, RNA_MD_SPARSE_ENTRY=8|4, default 8):
 *   8  two int arrays, k and fML(k,j).
 *   4  one uint32: k in the low 16 bits, an int16 delta in the high 16. A list is cut into
 *      segments of 32 slots -- one warp step of the readers -- and a segment's first entry
 *      is its baseline (d_sp_vb), delta 0. Every other entry stores its difference from the
 *      PREVIOUS entry, and the reader recovers values with a warp prefix sum. Not a delta from
 *      the baseline: 32 candidates span hundreds of rows of k, and on plain 1700-nt input that
 *      spread overflowed int16 in ~300 columns; consecutive candidates differ by little.
 *      Steps can still be large: a column's first candidates sit next to j (fML ~ +300) and
 *      the next may lie hundreds of rows away (~ -35000). A step outside int16 closes the
 *      segment early -- padded with k = 0, never a split point -- and opens a new one at
 *      that entry; the padded slots are counted. d_sp_last holds each column's last value
 *      for the append. Each column's slots are its capacity rounded up to 32, so a segment
 *      never straddles two columns, and the 4-B list overflows when its SLOTS run out, real
 *      entries and padding together. A k past 65535 overflows its column (counted apart).
 *      The baselines are this file's own, not int16 storage's d_fml_b: those are written when
 *      the triangle is packed, which RNA_ROW_BATCH delays past the append. */

static int*                d_sp_len  = NULL;   /* per (record, column), row layout: entries, or -1 */
static int*                d_sp_k    = NULL;   /* entry k                (ENTRY=8) */
static int*                d_sp_v    = NULL;   /* entry fML(k,j)         (ENTRY=8) */
static unsigned*           d_sp_kv4  = NULL;   /* entry k | delta << 16  (ENTRY=4) */
static int*                d_sp_vb   = NULL;   /* one baseline per 32 slots (ENTRY=4) */
static int*                d_sp_last = NULL;   /* per (record, column): last value appended (ENTRY=4) */
static int*                d_sp_prev = NULL;   /* previous row's triangle values, row layout */
static size_t*             d_sp_col  = NULL;   /* slot prefix over columns 0..maxlen+1 */
static int*                d_sp_cap  = NULL;   /* capacity per column 0..maxlen */
static size_t*             d_sp_base = NULL;   /* per-record start in the entry arrays */
static unsigned long long* d_sp_stat = NULL;   /* MD_SP_NSTAT counters, below */
static int*                d_sp_bad  = NULL;   /* the first MD_SP_NBAD mismatches, 5 ints each */
static int                 g_sp_on   = 0;      /* this chunk */
static size_t              g_sp_entries = 0;   /* slots */

#define MD_SP_NSTAT 8   /* 0 cells compared, 1 mismatches, 2 overflowed columns, 3 entries,
                           4 finite cells tested, 5 overflow fallbacks used,
                           6 of the overflowed columns, those a k past 65535 overflowed (ENTRY=4),
                           7 slots padded to close a segment early (ENTRY=4) */
#define MD_SP_NBAD  4
static unsigned long long  g_sp_tot[MD_SP_NSTAT];
static int                 g_sp_badh[MD_SP_NBAD * 5];
static int                 g_sp_nbadh = 0;
static size_t              g_sp_bytes_max = 0;
static size_t              g_sp_charged_max = 0;   /* what admission charged, same chunk sense */
static int                 g_sp_chunks = 0;

extern "C" int hp_mb_mlbase(void);   /* hp_mb_loop.cu: the parameter set's MLbase */
static int md_sparse_on(void);       /* S2, below */
static int md_sparse_explicit(void); /* S5: RNA_MD_SPARSE set by the user */

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

/* RNA_MD_SPARSE_ENTRY: 8 or 4 bytes per entry. */
static int
md_sparse_entry(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE_ENTRY");
    v = (e && e[0]) ? atoi(e) : 8;
    if (v != 8 && v != 4) {
      fprintf(stderr, "%-24s ignoring RNA_MD_SPARSE_ENTRY=%s (want 8 or 4); using 8\n", __FILE__, e);
      v = 8;
    }
  }
  return v;
}

/* RNA_MD_SPARSE_DELTA_MAX: test only, see the header. */
static int
md_sparse_delta_max(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE_DELTA_MAX");
    v = (e && e[0]) ? atoi(e) : 32767;
    if (v < 0 || v > 32767) v = 32767;
    if (v != 32767)
      fprintf(stderr, "%-24s RNA_MD_SPARSE_DELTA_MAX=%d: TEST -- 4-B steps beyond +-%d close their "
                      "segment early\n", __FILE__, v, v);
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

/* Slots a column occupies: its capacity, rounded up to a whole baseline segment for ENTRY=4. */
static size_t
md_sparse_slots(const int j)
{
  const size_t c = md_sparse_cap(j);
  return md_sparse_entry() == 4 ? (c + 31) & ~(size_t)31 : c;
}

/* Why the lists cannot run on this configuration (PORT_SPARSE_MD.md section 9), or NULL. */
static const char*
md_sparse_refusal(void)
{
  if (rnafold_continuous_flow())      return "continuous flow (records on different rows)";
  if (rnafold_slot_flow())            return "slot flow (slots are handed over mid-sweep)";
  if (rnafold_megakernel())           return "the megakernel (its own fused md)";
  if (rnafold_md_prune())             return "RNA_MD_PRUNE (a different dense-skipping scheme)";
  if (rnafold_md_block())             return "the blocked md (RNA_MD_BLOCK)";
  if (rnafold_md_tile_cb() > 0)       return "the tile path (RNA_MD_TILE_CB)";
  return NULL;
}

/* Will the sparse md run on this configuration? Asked by the int16 AUTO decision: if not, the
 * dense md streams the fML triangle and int16 pays (DRAM pressure). */
static int
md_sparse_planned(void)
{
  return md_sparse_on() && !md_sparse_refusal();
}

/* Bytes per record, for chunk admission. Mirrors md_sparse_alloc(): the slots, the
 * baselines (ENTRY=4), the row buffers (d_sp_len, d_sp_prev, and d_sp_last for ENTRY=4) and the column tables.
 * The column tables are per chunk, not per record; charging them per record over-counts,
 * which admission tolerates and an under-count would not. */
static size_t
md_sparse_bytes_per_file(const int length)
{
  if ((!md_sparse_on() && !md_sparse_selftest()) || md_sparse_refusal() || length <= 0) return 0;
  static thread_local int    last_len   = -1;   /* admission asks once per record */
  static thread_local size_t last_bytes = 0;
  if (length == last_len) return last_bytes;
  size_t slots = 0;
  for (int j = 0; j <= length; j++) slots += md_sparse_slots(j);
  const size_t entries = md_sparse_entry() == 4 ? slots * sizeof(unsigned) + (slots / 32) * sizeof(int)
                                                : slots * 2 * sizeof(int);
  last_len   = length;
  last_bytes = entries + (size_t)(length + 1) * (md_sparse_entry() == 4 ? 3 : 2) * sizeof(int)
             + (size_t)(length + 2) * (sizeof(size_t) + sizeof(int)) + 2 * sizeof(size_t)
             + sizeof(unsigned long long) * MD_SP_NSTAT + sizeof(int) * MD_SP_NBAD * 5;
  return last_bytes;
}

static void
md_sparse_report(void)
{
  const unsigned long long *t = g_sp_tot;
  char fmt[96];
  snprintf(fmt, sizeof(fmt), "%d-B entries", md_sparse_entry());
  if (md_sparse_entry() == 4)
    snprintf(fmt, sizeof(fmt), "4-B entries, %llu slot(s) padded, %llu column(s) past a 16-bit k", t[7], t[6]);
  if (md_sparse_on()) {
    if (vrna_cuda_quiet()) return;   /* the default path: a library caller asked for silence */
    fprintf(stderr, "%-24s RNA_MD_SPARSE%s: %d chunk(s); %llu candidates of %llu finite fML(k,j) (%.2f %%); "
                    "%llu column(s) overflowed at rho %.2f (%s), %llu md cell(s) took the dense fallback; "
                    "lists %.1f MB (admission charged %.1f MB)\n",
            __FILE__, md_sparse_negctl() ? " (NEGCTL)" : "", g_sp_chunks, t[3], t[4],
            t[4] ? 100.0 * (double)t[3] / (double)t[4] : 0.0, t[2], md_sparse_rho(), fmt, t[5],
            (double)g_sp_bytes_max / 1048576.0, (double)g_sp_charged_max / 1048576.0);
    return;
  }
  fprintf(stderr, "%-24s RNA_MD_SPARSE_SELFTEST%s: %d chunk(s), %llu cells compared, %llu MISMATCHING; "
                  "%llu candidates of %llu finite fML(k,j) (%.2f %%); %llu column(s) overflowed at "
                  "rho %.2f (%s), %llu cell(s) took the dense fallback; lists %.1f MB "
                  "(admission charged %.1f MB)\n",
          __FILE__, md_sparse_negctl() ? " (NEGCTL: scan term dropped)" : "", g_sp_chunks,
          t[0], t[1], t[3], t[4], t[4] ? 100.0 * (double)t[3] / (double)t[4] : 0.0,
          t[2], md_sparse_rho(), fmt, t[5], (double)g_sp_bytes_max / 1048576.0,
          (double)g_sp_charged_max / 1048576.0);
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

  const char *knob = md_sparse_on() ? "RNA_MD_SPARSE" : "RNA_MD_SPARSE_SELFTEST";
  const char *why  = md_sparse_refusal();
  if (why) {
    /* Said when the user ASKED for it. By default these configurations simply keep the dense md. */
    static int said = 0;
    if (!said && (md_sparse_explicit() || md_sparse_selftest())) {
      said = 1; fprintf(stderr, "%-24s %s REFUSED: %s -- the dense md runs\n", __FILE__, knob, why);
    }
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
  const int e4 = md_sparse_entry() == 4;
  size_t *col  = (size_t *)malloc(sizeof(size_t) * (size_t)(maxlen + 2));
  int    *cap  = (int *)malloc(sizeof(int) * (size_t)(maxlen + 1));
  size_t *base = (size_t *)malloc(sizeof(size_t) * (size_t)(nfiles + 1));
  col[0] = 0;
  for (int j = 0; j <= maxlen; j++) {
    cap[j]   = (int)md_sparse_cap(j);
    col[j+1] = col[j] + md_sparse_slots(j);   /* ENTRY=4: whole segments, so col[] stays 32-aligned */
  }
  base[0] = 0;
  for (int H = 0; H < nfiles; H++) {
    const int len_H = (int)(row_off_H[H+1] - row_off_H[H]) - 1;
    base[H+1] = base[H] + col[len_H + 1];
  }
  g_sp_entries = base[nfiles];

  gpuErrchk( cudaMalloc((void **)&d_sp_col,  sizeof(size_t) * (size_t)(maxlen + 2)) );
  gpuErrchk( cudaMalloc((void **)&d_sp_cap,  sizeof(int) * (size_t)(maxlen + 1)) );
  gpuErrchk( cudaMalloc((void **)&d_sp_base, sizeof(size_t) * (size_t)(nfiles + 1)) );
  gpuErrchk( cudaMemcpy(d_sp_col,  col,  sizeof(size_t) * (size_t)(maxlen + 2), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_sp_cap,  cap,  sizeof(int) * (size_t)(maxlen + 1), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_sp_base, base, sizeof(size_t) * (size_t)(nfiles + 1), cudaMemcpyHostToDevice) );
  free(col); free(cap); free(base);

  const size_t ne = g_sp_entries ? g_sp_entries : 1;
  if (e4) {
    gpuErrchk( cudaMalloc((void **)&d_sp_kv4, sizeof(unsigned) * ne) );
    gpuErrchk( cudaMalloc((void **)&d_sp_vb,  sizeof(int) * ((ne + 31) / 32)) );
    gpuErrchk( cudaMalloc((void **)&d_sp_last, sizeof(int) * g_row_total) );   /* written before read */
  } else {
    gpuErrchk( cudaMalloc((void **)&d_sp_k,   sizeof(int) * ne) );
    gpuErrchk( cudaMalloc((void **)&d_sp_v,   sizeof(int) * ne) );
  }
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
    /* everything cudaMalloc'd above, against what chunk admission was charged for it */
    const size_t bytes = (e4 ? ne * sizeof(unsigned) + ((ne + 31) / 32) * sizeof(int)
                             : ne * 2 * sizeof(int))
                       + g_row_total * (e4 ? 3 : 2) * sizeof(int)
                       + (size_t)(maxlen + 2) * sizeof(size_t) + (size_t)(maxlen + 1) * sizeof(int)
                       + (size_t)(nfiles + 1) * sizeof(size_t)
                       + sizeof(unsigned long long) * MD_SP_NSTAT + sizeof(int) * MD_SP_NBAD * 5;
    size_t charged = 0;
    for (int H = 0; H < nfiles; H++)
      charged += md_sparse_bytes_per_file((int)(row_off_H[H+1] - row_off_H[H]) - 1);
    if (bytes > g_sp_bytes_max) g_sp_bytes_max = bytes;
    if (charged > g_sp_charged_max) g_sp_charged_max = charged;
    if (bytes > charged)
      fprintf(stderr, "%-24s WARNING: the candidate lists take %zu bytes but chunk admission charged %zu "
                      "-- gpu_bytes_per_file() under-counts them\n", __FILE__, bytes, charged);
  }
  g_sp_on = 1;
  g_sp_chunks++;
  static int said = 0;
  if (!said) {
    said = 1;
    atexit(md_sparse_report);
    if (md_sparse_on()) {
      if (!vrna_cuda_quiet())
      fprintf(stderr, "%-24s RNA_MD_SPARSE ACTIVE: md reads only candidate split points (lists rebuilt "
                      "every row) plus a row scan, in place of the dense md (rho %.2f, %d-B entries%s)\n",
              __FILE__, md_sparse_rho(), md_sparse_entry(),
              md_sparse_negctl() ? ", NEGCTL: scan term dropped -- WRONG ANSWERS EXPECTED" : "");
    } else
      fprintf(stderr, "%-24s RNA_MD_SPARSE_SELFTEST ACTIVE: candidate lists built every row and the "
                      "sparse split compared with the dense md, cell for cell (rho %.2f, %d-B entries%s)\n",
              __FILE__, md_sparse_rho(), md_sparse_entry(), md_sparse_negctl() ? ", NEGCTL: scan term dropped" : "");
  }
  if (e4 && maxlen > 65535) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "%-24s RNA_MD_SPARSE_ENTRY=4: rows past 65535 do not fit a 16-bit k; "
                                           "their columns take the dense md\n", __FILE__); }
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
  if (d_sp_kv4)  { gpuErrchk( cudaFree(d_sp_kv4) );  d_sp_kv4  = NULL; }
  if (d_sp_vb)   { gpuErrchk( cudaFree(d_sp_vb) );   d_sp_vb   = NULL; }
  if (d_sp_last) { gpuErrchk( cudaFree(d_sp_last) ); d_sp_last = NULL; }
  if (d_sp_prev) { gpuErrchk( cudaFree(d_sp_prev) ); d_sp_prev = NULL; }
  if (d_sp_col)  { gpuErrchk( cudaFree(d_sp_col) );  d_sp_col  = NULL; }
  if (d_sp_cap)  { gpuErrchk( cudaFree(d_sp_cap) );  d_sp_cap  = NULL; }
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

/* One warp step over a list: lane `lane` loads entry s + lane of the list starting at slot
 * `at`, s a multiple of 32. Returns whether the entry exists. The WHOLE warp must call it
 * (ENTRY=4 decodes with a shuffle scan), so callers loop on a warp-uniform s.
 * sp_kv4 non-NULL selects ENTRY=4, uniform across a launch. */
__device__ __forceinline__ int
md_sp_step(const int* __restrict__ sp_k, const int* __restrict__ sp_v,
           const unsigned* __restrict__ sp_kv4, const int* __restrict__ sp_vb,
           const size_t at, const int n, const int s, const int lane, int &k, int &v)
{
  const int ok = s + lane < n;
  if (!sp_kv4) {
    if (ok) { k = sp_k[at + s + lane]; v = sp_v[at + s + lane]; }
    return ok;
  }
  const unsigned w = ok ? sp_kv4[at + s + lane] : 0u;   /* past the end: delta 0 */
  int d = (int)(short)(unsigned short)(w >> 16);
  for (int off = 1; off < 32; off <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, d, off);
    if (lane >= off) d += t;
  }
  k = (int)(w & 0xffffu);
  v = sp_vb[(at + (size_t)s) >> 5] + d;                  /* at + s is 32-aligned: a segment start */
  return ok;
}

/* After row i closes: test every (k=i, j) of the row and append the candidates. One thread
 * per cell, md_close_row's grid. Each column gets at most one append per row, from the one
 * thread that owns (i,j), so there is nothing to race on -- the ENTRY=4 baseline included:
 * it is written by the append that opens its segment, and read only by later rows' appends
 * to the same column, which are later launches. */
__global__ void
md_sparse_append_kernel(const int nfiles, const int i_row, const int turn, const int mlbase,
                        const int dmax,
                        const int* __restrict__ energy_min, const int* __restrict__ dml,
                        int* __restrict__ sp_prev, int* __restrict__ sp_len,
                        int* __restrict__ sp_k, int* __restrict__ sp_v,
                        unsigned* __restrict__ sp_kv4, int* __restrict__ sp_vb,
                        int* __restrict__ sp_last, const int* __restrict__ sp_cap,
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

  int n = sp_len[o + j];
  if (n < 0) return;                                   /* already overflowed */
  const size_t start = sp_base[H] + sp_col[j];         /* 32-aligned for ENTRY=4 */
  const int    lim   = sp_kv4 ? (int)(sp_col[j + 1] - sp_col[j]) : sp_cap[j];
  if (n >= lim || (sp_kv4 && i > 0xffff)) {
    sp_len[o + j] = -1;
    atomicAdd(&stat[2], 1ULL);
    if (n < lim) atomicAdd(&stat[6], 1ULL);           /* the 16-bit k, not the capacity */
    return;
  }
  if (sp_kv4) {
    /* the segment's first entry is its baseline; the others store the step from the entry
     * before. A step that does not fit int16 closes the segment early: the rest of it is
     * padded with k = 0 (never a split point, delta 0) and this entry opens the next. */
    int d = (n & 31) ? v - sp_last[o + j] : 0;
    if (d < -dmax || d > dmax) {
      const int n2 = (n + 31) & ~31;
      if (n2 >= lim) {
        sp_len[o + j] = -1;
        atomicAdd(&stat[2], 1ULL);
        return;
      }
      for (int p = n; p < n2; p++) sp_kv4[start + (size_t)p] = 0u;
      atomicAdd(&stat[7], (unsigned long long)(n2 - n));
      n = n2;
      d = 0;
    }
    const size_t at = start + (size_t)n;
    if (!(n & 31)) sp_vb[at >> 5] = v;
    sp_last[o + j] = v;
    sp_kv4[at] = (unsigned)i | ((unsigned)(unsigned short)(short)d << 16);
  } else {
    const size_t at = start + (size_t)n;
    sp_k[at] = i;
    sp_v[at] = v;
  }
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
                          const unsigned* __restrict__ sp_kv4, const int* __restrict__ sp_vb,
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
      for (int s0 = 0; s0 < n; s0 += 32) {
        int k = 0, b = INF;
        if (!md_sp_step(sp_k, sp_v, sp_kv4, sp_vb, at, n, s0, lane, k, b)) continue;
        if (k < lo) continue;                        /* rows i+1 .. i+turn+1: not split points */
        const int a = fml_i[o + (size_t)(k - lo)];
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
      md_sparse_negctl(), fml_i, d_dml, d_sp_len, d_sp_k, d_sp_v, d_sp_kv4, d_sp_vb, d_sp_col, d_sp_base,
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
      hp_mb_mlbase(), md_sparse_delta_max(), d_energy_min, d_dml, d_sp_prev, d_sp_len, d_sp_k, d_sp_v, d_sp_kv4, d_sp_vb,
      d_sp_last, d_sp_cap, d_sp_col, d_sp_base,
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

/* DEFAULT ON since S5 (A100: -29.3 % wall at 400 x 5601, exact on every bar). RNA_MD_SPARSE=0
 * restores the dense md. RNA_MD_SPARSE_SELFTEST=1 without an explicit RNA_MD_SPARSE means the
 * selftest (dense md authoritative, sparse computed beside it), as before the flip. */
static int
md_sparse_explicit(void)
{
  const char *e = getenv("RNA_MD_SPARSE");
  return (e && e[0]) ? 1 : 0;
}

static int
md_sparse_on(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE");
    if (e && e[0]) v = (e[0] != '0') ? 1 : 0;
    else           v = md_sparse_selftest() ? 0 : 1;
  }
  return v;
}

__device__ __forceinline__ int
md_sp_add(const int a, const int b)
{
  return (a >= INF/2 || b >= INF/2) ? INF : a + b;
}

/* L lanes per cell (RNA_MD_SPARSE_LANES, 32 by default): 32/L cells share a warp, so short
 * lists near the diagonal stop leaving most of a warp idle. ENTRY=4 needs L = 32 (its decode
 * is a whole-warp scan over one 32-slot segment). No thread returns early: the reduction's
 * shuffles name the whole warp, so cells past `total` ride along and write nothing. */
template <int L>
__global__ void
md_sparse_cell_kernel(const int nfiles, const int turn,
                      const int* __restrict__ fml_i,
                      const int* __restrict__ fml_j,
                      const short* __restrict__ fml_j16, const int* __restrict__ fml_b,
                      const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                      const int* __restrict__ sp_len, const int* __restrict__ sp_k,
                      const int* __restrict__ sp_v,
                      const unsigned* __restrict__ sp_kv4, const int* __restrict__ sp_vb,
                      const size_t* __restrict__ sp_col, const size_t* __restrict__ sp_base,
                      const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                      const size_t* __restrict__ side_off_H, const size_t total,
                      const int* __restrict__ i_H,
                      int* __restrict__ best_out,
                      unsigned long long* __restrict__ stat)
{
  const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t    mm   = (size_t)(gtid / L);
  const int       lane = (int)(gtid % L);
  const int       live = mm < total;
  const size_t    m    = live ? mm : total - 1;   /* a dead group computes a real cell, unwritten */

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
    for (int s0 = 0; s0 < n; s0 += L) {    /* uniform per group; L = 32 for ENTRY=4's shuffles */
      int k = 0, b = INF;
      if (!md_sp_step(sp_k, sp_v, sp_kv4, sp_vb, at, n, s0, lane, k, b)) continue;
      if (k < lo) continue;
      const int s = md_sp_add(fml_i[o + (size_t)(k - lo)], b);
      best = s < best ? s : best;
    }
  } else {
    /* overflowed column: md's own dense loop, B from the triangle */
    const long long ij0 = Indx(i, j) + (turn + 1) + 1;
    if (fml_j16) {
      const size_t bcell = base_off_H[H] + colb_off[j];
      for (int y = lane; y <= x; y += L) {
        const int b = fml_decode(fml_j16, fml_b, tri_off_H[H] + (size_t)(ij0 + y),
                                 bcell + (size_t)((unsigned)(lo + y - 1) / FML_BLK));
        const int s = md_sp_add(fml_i[o + (size_t)y], b);
        best = s < best ? s : best;
      }
    } else {
      for (int y = lane; y <= x; y += L) {
        const int s = md_sp_add(fml_i[o + (size_t)y], fml_j[tri_off_H[H] + (size_t)(ij0 + y)]);
        best = s < best ? s : best;
      }
    }
    if (lane == 0 && live) atomicAdd(&stat[5], 1ULL);
  }
  for (int off = L / 2; off > 0; off >>= 1) {
    const int other = __shfl_down_sync(0xffffffffu, best, off, L);
    best = other < best ? other : best;
  }
  if (lane == 0 && live) best_out[o + j] = best;
}

/* RNA_MD_SPARSE_LANES: lanes per cell, 8, 16 or 32 (default). */
static int
md_sparse_lanes(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("RNA_MD_SPARSE_LANES");
    v = (e && e[0]) ? atoi(e) : 32;
    if (v != 8 && v != 16 && v != 32) {
      fprintf(stderr, "%-24s ignoring RNA_MD_SPARSE_LANES=%s (want 8, 16 or 32); using 32\n", __FILE__, e);
      v = 32;
    }
    if (v != 32 && md_sparse_entry() == 4) {
      fprintf(stderr, "%-24s RNA_MD_SPARSE_LANES=%d REFUSED: 4-B entries decode a whole 32-slot segment "
                      "per warp -- using 32\n", __FILE__, v);
      v = 32;
    }
    if (v != 32 && !vrna_cuda_quiet())
      fprintf(stderr, "%-24s RNA_MD_SPARSE_LANES=%d: %d cells per warp\n", __FILE__, v, 32 / v);
  }
  return v;
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
  if (total == 0) return;   /* no cell this row (and the kernel's dead groups clamp to total - 1) */
  {
    const int    L        = md_sparse_lanes();
    const size_t nthreads = total * (size_t)L;
    const int    bs       = 256;
    const int    nb       = (int)((nthreads + bs - 1) / bs);
#define MD_SP_CELL(LL) md_sparse_cell_kernel<LL><<<nb, bs, 0, ISSUE_STREAM>>>(nfiles, turn, fml_i, d_fml_j, \
        d_fml_j16, d_fml_b, d_base_off_H, d_colb_off, d_sp_len, d_sp_k, d_sp_v, d_sp_kv4, d_sp_vb, d_sp_col, \
        d_sp_base, d_tri_off_H, d_row_off_H, d_side_off_H, total, d_i_H, d_dml, d_sp_stat)
    if (L == 8)       MD_SP_CELL(8);
    else if (L == 16) MD_SP_CELL(16);
    else              MD_SP_CELL(32);
#undef MD_SP_CELL
    gpuErrchk( cudaPeekAtLastError() );
  }
  md_sparse_scan_kernel<<<nfiles, MD_SP_SCAN_T, 0, ISSUE_STREAM>>>(nfiles, turn, hp_mb_mlbase(),
      md_sparse_negctl(), d_dml, rnafold_circ_fm2_device(), d_tri_off_H, d_row_off_H, d_side_off_H,
      d_i_H, p.up_ml_ok, p.seq_off_H);
  gpuErrchk( cudaPeekAtLastError() );
}

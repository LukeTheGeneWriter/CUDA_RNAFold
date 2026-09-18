/*
 *  Device discovery for the CUDA MFE backend.
 *
 *  The first .cu translation unit in the tree. It exists as much to exercise
 *  the nvcc build rule as to answer the question: everything downstream of here
 *  depends on nvcc objects linking correctly into a libtool convenience library
 *  alongside ordinary C, and that is worth proving before eight thousand lines
 *  of kernels arrive.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* rnafold_now_seconds() -- the same clock every other timer in the sweep uses,
 * so a launch total can be compared against a phase total without converting. */
extern "C" double rnafold_now_seconds(void);

/*
 *  SPIN OR BLOCK WHILE WAITING FOR THE DEVICE?
 *
 *  Established 2026-09-09, and the first answer was wrong twice.
 *
 *  The process burns ~ONE FULL CORE continuously: 103-104% of one CPU for a
 *  16 x 2600 nt run that is almost all device work. On a 12-core box that is
 *  invisible; on a 1-2 vCPU host it is the whole machine, and it decides whether
 *  folding on the CPU beside the device is possible at all -- a busy driving
 *  thread leaves no idle capacity, so a CPU folder takes time directly from the
 *  thread feeding the GPU. That is the "device waiting on the CPU" case.
 *
 *  FIRST WRONG ANSWER: "87% of it is spin-wait in the CUDA sync policy". That
 *  came from summing only the STAGE timers (0.77 s of 5.70 CPU-seconds) and
 *  calling the remainder spin -- but the PHASE timers (modular_decomp, hp_mb,
 *  int_loop, the transfers) account for it, and they measure host wall time
 *  around device work. The arithmetic left them out.
 *
 *  SECOND WRONG ANSWER: the knob below appeared to prove blocking sync does not
 *  help, when in fact it had never run -- cudaSetDeviceFlags() fails with
 *  cudaErrorSetOnActiveProcess once a context exists, and cudaGetDeviceCount()
 *  is enough to create one, so calling it after the probe was a silent no-op.
 *  Fixed (it now runs first and REPORTS failure). With the ordering correct and
 *  no error raised, CPU stays at 103% either way.
 *
 *  SO: the core is NOT consumed by the runtime's spin-wait policy. Whatever is
 *  burning it -- kernel-launch overhead, synchronous pageable copies, or the
 *  host-side row loop -- is not something this flag reaches, and switching the
 *  policy neither helps nor hurts (wall clock is a wash, ABBA-ordered).
 *
 *  The knob stays because it is one line, correct now, and the answer may differ
 *  on a host whose sync pattern differs. It is not a lever we have found value
 *  in. Anyone hoping to free a core for CPU-side folding must look at where the
 *  phase timers actually spend their host time first.
 *
 *    RNA_GPU_BLOCKING_SYNC=1   park on a wait queue instead of the CUDA default
 *    unset                     cudaDeviceScheduleAuto
 *
 *  Must be set BEFORE the context exists, which is why it lives here in the
 *  device probe rather than in init_gpu().
 */
static void
maybe_set_blocking_sync(void)
{
  static int done = 0;
  const char *e;

  if (done)
    return;

  done = 1;
  e    = getenv("RNA_GPU_BLOCKING_SYNC");

  if ((!e) || (!e[0]) || (e[0] == '0'))
    return;

  {
    /* ORDER MATTERS AND THE FIRST VERSION GOT IT WRONG. cudaSetDeviceFlags()
     * fails with cudaErrorSetOnActiveProcess once a context exists, and
     * cudaGetDeviceCount() is enough to initialise the runtime -- so calling
     * this after the probe made the knob a silent no-op. It read as "blocking
     * sync does not help" when it meant "blocking sync never happened": CPU
     * stayed at 104% in both arms, which is what gave it away.
     *
     * Report the failure rather than swallowing it. A knob that quietly does
     * nothing is worse than one that is missing. */
    const cudaError_t rc = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    if (rc != cudaSuccess) {
      fprintf(stderr,
              "device.cu                RNA_GPU_BLOCKING_SYNC requested but "
              "cudaSetDeviceFlags failed: %s. The driver will still spin.\n",
              cudaGetErrorString(rc));
      cudaGetLastError();
    }
  }
}


extern "C" unsigned int
vrna_cuda_device_count(void)
{
  int         n;
  cudaError_t err;

  /* BEFORE any runtime call, for the reason above. */
  maybe_set_blocking_sync();

  n   = 0;
  err = cudaGetDeviceCount(&n);

  /*
   * No device, no driver, or a driver/runtime mismatch are all ordinary
   * conditions on a machine that simply has no GPU. They mean "decline", not
   * "fail" -- so the error is swallowed deliberately rather than reported.
   */
  if (err != cudaSuccess) {
    cudaGetLastError();     /* clear the sticky error for any later caller */
    return 0u;
  }

  return (n > 0) ? (unsigned int)n : 0u;
}


/*
 * RNA_PHASE_SYNC -- turn the per-phase timers from an ATTRIBUTION into a
 * MEASUREMENT.
 *
 * The sweep launches its kernels asynchronously and then blocks, not at a
 * sync, but at whichever call next touches the device synchronously. In the
 * GPU-resident sweep that is hp_mb_3p_i()'s two H2D uploads of size_off_H and
 * i_H, which run BEFORE its kernel -- so the queue drains inside the `hp_mb`
 * timer, and `hp_mb` is charged for work the modular decomposition queued.
 *
 * That makes the phase split move when kernel speeds change even if no phase
 * got slower. It is the leading explanation for the long-standing "int16 makes
 * hp_mb 23-30 s slower" result (STRESS272_RESULTS.md 4, 13.6, 14.4, 15.3):
 * int16 speeds up modular_decomp's kernels, the host arrives at hp_mb's upload
 * sooner, and waits there instead. hp_mb_loop.cu contains no int16 code at all
 * -- fml_decode() is called from modular_decomposition.cu and nowhere else --
 * so there is no mechanism by which its kernel could slow down.
 *
 * With this set, every timed phase ends with a device sync, so each timer
 * holds that phase's own GPU time. WALL GOES UP: the overlap between phases is
 * destroyed, so this is a diagnostic, never a run mode. Compare SPLITS between
 * two arms both run with it, never a split from here against one without.
 *
 *   RNA_PHASE_SYNC=1   sync at every phase boundary; timers become truthful
 *   unset              normal asynchronous operation (default)
 */
extern "C" int
rnafold_phase_sync_enabled(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_PHASE_SYNC");

    v = (e && e[0] && e[0] != '0') ? 1 : 0;

    if (v)
      fprintf(stderr,
              "device.cu                RNA_PHASE_SYNC=1: syncing at every "
              "phase boundary. Phase timers are now true GPU times and the "
              "WALL IS NOT COMPARABLE to a normal run.\n");
  }

  return v;
}


/* RNA_SYNC_PROBE=k -- add k EXTRA cudaDeviceSynchronize() calls per sweep row.
 *
 * A NEGATIVE CONTROL, and the cheapest way to price a change before building
 * it. PORT_STREAM_OVERLAP_SCOPE.md wants the per-row barriers gone, which needs
 * double-buffered pinned staging and an event per slot. Before writing any of
 * that: if ADDING a barrier per row costs nothing, REMOVING one cannot pay, and
 * the whole of stage 1 is dead for the price of a five-line probe.
 *
 * The slope (seconds of wall per extra sync per row) times the number of
 * barriers a row already carries is the prize, estimated without touching a
 * single correctness-bearing line. It cannot change an answer -- a sync is a
 * wait, not a write -- so this is measurable on any arm, including one whose
 * sha we care about.
 */
extern "C" int
rnafold_sync_probe(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_SYNC_PROBE");

    v = (e && e[0]) ? atoi(e) : 0;
    if (v < 0)
      v = 0;

    if (v)
      fprintf(stderr,
              "device.cu                RNA_SYNC_PROBE=%d: %d EXTRA device syncs "
              "per sweep row (a negative control -- this can only make the run "
              "SLOWER)\n", v, v);
  }

  return v;
}


extern "C" void
rnafold_sync_probe_tick(void)
{
  const int k = rnafold_sync_probe();
  int       s;

  for (s = 0; s < k; s++) {
    const cudaError_t rc = cudaDeviceSynchronize();

    if (rc != cudaSuccess) {
      fprintf(stderr, "device.cu                RNA_SYNC_PROBE sync failed: %s\n",
              cudaGetErrorString(rc));
      cudaGetLastError();
      return;
    }
  }
}


/* ===================== RNA_STREAM_OVERLAP: the row's streams =====================
 *
 * The row loop issues six kernels and every one of them has always gone to the
 * NULL stream, so they run back to back whether or not they depend on each
 * other. Three of them do not:
 *
 *   int_loop(i)   reads c (rows > i)        -> energy_min2
 *   hp_mb_3p(i)   reads ONLY the sequence, the parameter tables and the
 *                 hard-constraint masks -- no DP value at all
 *   new_c(i)      joins both, plus DMLi1 from row i+1
 *
 * PORT_STREAM_OVERLAP_SCOPE.md 12 works the graph out and prices the two
 * overlaps it allows. They are very different sizes:
 *
 *   1  hp_mb_3p(i) beside int_loop(i)            1.89 s of 65.3   ~2 % of wall
 *   2  md(i) beside int_loop(i-1)+hp_mb(i-1)    18.4  s          ~21 % of wall
 *
 * RNA_STREAM_OVERLAP selects between them: 0 (default) is today's schedule
 * exactly -- every stream below is stream 0 and every event call returns
 * immediately -- 1 is the first, 2 is both.
 *
 * DEFAULT OFF because concurrency is the one class of change whose failure mode
 * is an intermittent wrong answer. The control is in-process: the same binary
 * folds the same batch both ways and the shas are compared.
 */
extern "C" int
rnafold_stream_overlap(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_STREAM_OVERLAP");

    /* DEFAULT CHANGED 2026-09-17 to 1 (Luke's call, on the measurement below).
     * Scaling G at 1d5eb219, 400 x 5601 on an A100: ONE sha across all six arms
     * -- levels 0, 1 and 2 byte-identical -- with level 1 at -0.4 % and level 2
     * at -0.8 %. Level 1 has been clean in every run it has ever had, including
     * the one where level 2 returned two different wrong answers; the per-chunk
     * row tables removed that race. Level 2 stays opt-in until it has been
     * through the H stress soak. "0" still forces the old single-stream
     * schedule, and is the control every sha comparison is made against. */
    v = (e && e[0]) ? atoi(e) : 1;

    if (v < 0) v = 0;
    if (v > 2) v = 2;

    if (v != 1)
      fprintf(stderr,
              "device.cu                RNA_STREAM_OVERLAP=%d: %s\n", v,
              (v == 0) ? "one stream, the pre-2026-09-17 schedule"
                       : "hp_mb_3p beside int_loop, and md(i) beside row i-1's cell work");

    if (v >= 2)
      fprintf(stderr,
              "device.cu                RNA_STREAM_OVERLAP=2 is opt-in: correct at "
              "400 x 5601 (Scaling G, 2026-09-17, one sha\n"
              "device.cu                across six arms) and worth -0.8%%, but it has not "
              "been through the stress soak. Compare shas.\n");
  }

  return v;
}


/* The two streams. Both are 0 -- the NULL stream -- until the knob turns them
 * on, which is what makes the default path bit-for-bit the old one rather than
 * a new schedule that happens to serialise. */
static cudaStream_t g_stream_cell = 0;   /* int_loop, new_c, load_my_c, fml_* */
static cudaStream_t g_stream_hp   = 0;   /* hp_mb_3p only */
static cudaStream_t g_stream_md   = 0;   /* fml_scan, the md graph, fml_prev, snapshot */
static cudaEvent_t  g_ev_md       = NULL;   /* snapshot(i) finished: DMLi1 is row i's */
/* THE THIRD EDGE, and it is the one the first level-2 build was missing.
 *
 * The hp stream has no waits of its own, so the HOST can queue hp_mb(i-1),
 * hp_mb(i-2), ... arbitrarily far ahead: a device-side wait on the cell stream
 * does not stop the host from issuing more work elsewhere, and level 2 removed
 * the per-row sync that used to bound it. Two-deep parity buffers then stop
 * being enough -- hp_mb(i-2) writes the SAME parity that fml_scan(i) is still
 * reading.
 *
 * Measured, at 400 x 5601 and not at 60 x 1500: two runs of the same level-2
 * arm returned two DIFFERENT wrong answers. One event per parity fixes it by
 * construction -- hp_mb(i) waits for the fml_scan that last read its buffer,
 * which is the one two rows earlier. */
static cudaEvent_t  g_ev_scan[2]  = { NULL, NULL };
static cudaEvent_t  g_ev_hp       = NULL;   /* hp_mb_3p(i) finished */
static cudaEvent_t  g_ev_cell     = NULL;   /* the cell chain reached a join */

extern "C" void
rnafold_streams_init(void)
{
  if (!rnafold_stream_overlap())
    return;

  if (g_stream_cell == 0) {
    /* NON-BLOCKING, deliberately: a blocking stream synchronises with the
     * legacy default stream, which would re-serialise the two the moment
     * anything at all still ran there. */
    if (cudaStreamCreateWithFlags(&g_stream_cell, cudaStreamNonBlocking) != cudaSuccess) {
      fprintf(stderr, "device.cu                stream create failed -- overlap disabled\n");
      g_stream_cell = 0;
      return;
    }
    if (cudaStreamCreateWithFlags(&g_stream_hp, cudaStreamNonBlocking) != cudaSuccess) {
      fprintf(stderr, "device.cu                stream create failed -- overlap disabled\n");
      cudaStreamDestroy(g_stream_cell);
      g_stream_cell = 0; g_stream_hp = 0;
      return;
    }
    /* Timing disabled: these exist for ordering, and a timing event costs a
     * synchronisation the schedule is trying to avoid. */
    cudaEventCreateWithFlags(&g_ev_hp,   cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_ev_cell, cudaEventDisableTiming);

    /* Level 2 only: the md chain gets a stream of its own so that row i's
     * fml_scan -> md -> fml_prev -> snapshot runs beside row i-1's int_loop and
     * hp_mb_3p. At level 1 it stays on the cell stream, where it is ordered
     * behind them exactly as it always was. */
    if (rnafold_stream_overlap() >= 2) {
      if (cudaStreamCreateWithFlags(&g_stream_md, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "device.cu                md stream create failed -- level 2 disabled\n");
        g_stream_md = 0;
      } else {
        cudaEventCreateWithFlags(&g_ev_md, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_ev_scan[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_ev_scan[1], cudaEventDisableTiming);
      }
    }
  }
}


extern "C" void
rnafold_streams_teardown(void)
{
  if (g_stream_cell) { cudaStreamDestroy(g_stream_cell); g_stream_cell = 0; }
  if (g_stream_hp)   { cudaStreamDestroy(g_stream_hp);   g_stream_hp   = 0; }
  if (g_stream_md)   { cudaStreamDestroy(g_stream_md);   g_stream_md   = 0; }
  if (g_ev_md)       { cudaEventDestroy(g_ev_md);        g_ev_md       = NULL; }
  if (g_ev_scan[0])  { cudaEventDestroy(g_ev_scan[0]);   g_ev_scan[0]  = NULL; }
  if (g_ev_scan[1])  { cudaEventDestroy(g_ev_scan[1]);   g_ev_scan[1]  = NULL; }
  if (g_ev_hp)       { cudaEventDestroy(g_ev_hp);        g_ev_hp       = NULL; }
  if (g_ev_cell)     { cudaEventDestroy(g_ev_cell);      g_ev_cell     = NULL; }
}


extern "C" cudaStream_t rnafold_stream_cell(void) { return g_stream_cell; }
extern "C" cudaStream_t rnafold_stream_hp(void)   { return g_stream_hp ? g_stream_hp : g_stream_cell; }
extern "C" cudaStream_t rnafold_stream_md(void)   { return g_stream_md ? g_stream_md : g_stream_cell; }

/* Level 2's two extra edges, from PORT_STREAM_OVERLAP_SCOPE.md 5:
 *
 *   load_my_c(i) -> fml_scan(i)     the md chain may not read row i's c until
 *                                   the cell stream has written it
 *   snapshot(i)  -> new_c(i-1)      the cell stream may not consume DMLi1 until
 *                                   the md chain has published row i's
 *
 * All four are no-ops below level 2, where the two chains share one stream and
 * program order already says this. */
extern "C" void
rnafold_stream_cell_done(void)
{
  if (g_ev_cell && g_stream_md)
    (void)cudaEventRecord(g_ev_cell, g_stream_cell);
}

extern "C" void
rnafold_stream_md_wait_cell(void)
{
  if (g_ev_cell && g_stream_md)
    (void)cudaStreamWaitEvent(g_stream_md, g_ev_cell, 0);
}

extern "C" void
rnafold_stream_md_done(void)
{
  if (g_ev_md && g_stream_md)
    (void)cudaEventRecord(g_ev_md, g_stream_md);
}

/* fml_scan(i) has finished reading row i's hp/mb parity: the buffer is free for
 * the row two later to overwrite. */
extern "C" void
rnafold_stream_scan_done(int i)
{
  if (g_ev_scan[0] && g_stream_md)
    (void)cudaEventRecord(g_ev_scan[i & 1], g_stream_md);
}

/* hp_mb_3p(i) may not write its parity until the fml_scan that last read it is
 * done -- which is row i+2's, recorded two rows ago. Unrecorded on the first
 * two rows, where the wait is a no-op, which is correct: nothing has read them
 * yet. */
extern "C" void
rnafold_stream_wait_scan(int i)
{
  if (g_ev_scan[0] && g_stream_md && g_stream_hp)
    (void)cudaStreamWaitEvent(g_stream_hp, g_ev_scan[i & 1], 0);
}

extern "C" void
rnafold_stream_wait_md(void)
{
  if (g_ev_md && g_stream_md)
    (void)cudaStreamWaitEvent(g_stream_cell, g_ev_md, 0);
}

/* hp_mb_3p(i) has finished: record it, so whoever joins can wait. No-ops when
 * the knob is off, so the call sites need no conditional of their own. */
extern "C" void
rnafold_stream_hp_done(void)
{
  if (g_ev_hp && g_stream_hp)
    (void)cudaEventRecord(g_ev_hp, g_stream_hp);
}

/* The cell chain must not run past this point until hp_mb_3p(i) is done. */
extern "C" void
rnafold_stream_wait_hp(void)
{
  if (g_ev_hp && g_stream_hp)
    (void)cudaStreamWaitEvent(g_stream_cell, g_ev_hp, 0);
}

/* Everything issued so far has to have finished -- the end of a row in the
 * schedules that still need one, and the teardown path. */
extern "C" void
rnafold_streams_sync(void)
{
  cudaError_t rc = cudaSuccess;

  if (g_stream_cell) rc = cudaStreamSynchronize(g_stream_cell);
  if ((rc == cudaSuccess) && g_stream_hp) rc = cudaStreamSynchronize(g_stream_hp);
  if ((rc == cudaSuccess) && g_stream_md) rc = cudaStreamSynchronize(g_stream_md);

  if (rc != cudaSuccess) {
    fprintf(stderr, "device.cu                stream sync failed: %s\n",
            cudaGetErrorString(rc));
    cudaGetLastError();
  }
}


extern "C" void
rnafold_phase_sync(void)
{
  if (rnafold_phase_sync_enabled()) {
    const cudaError_t rc = cudaDeviceSynchronize();

    if (rc != cudaSuccess) {
      fprintf(stderr, "device.cu                RNA_PHASE_SYNC sync failed: %s\n",
              cudaGetErrorString(rc));
      cudaGetLastError();
    }
  }
}


/*
 * RNA_LAUNCH_STATS -- per-launch DEVICE time, so a phase total can be split
 * into "the kernel" and "everything else".
 *
 * WHY THIS EXISTS. STRESS272_RESULTS.md §20 left one question open and it is
 * not answerable from phase totals. NCU says `int_loop_kernel` is IDENTICAL
 * under int16 and int32 -- 717,676.8 ns against 717,683.2 ns, every counter
 * matching -- while the `int_loop` PHASE is 25 s slower under int16 at
 * 400 x 5601. NCU profiles one kernel in isolation with clocks locked; the
 * phase timer measures a 78,000-launch loop on a live device. Something
 * between those two is the answer, and nothing in the existing instrumentation
 * looks there.
 *
 * This does. It records, per launch:
 *
 *   device_ms   cudaEvent time around the kernel ALONE
 *   host_ms     wall time around launch + event sync, i.e. device + overhead
 *   grid        so the two encodings can be compared launch-for-launch at
 *               equal work rather than in aggregate
 *
 * The decisive plot is device_ms against grid, i32 over i16. If the curves
 * coincide, the kernel really is identical at scale and the 25 s is launch
 * overhead or device state -- and sum(device_ms) against the phase total says
 * how much. If i16's curve sits above, the kernel IS slower in situ and NCU's
 * locked-clock isolation is what hid it.
 *
 * IT IS A DIAGNOSTIC, NOT A RUN MODE. Every launch ends in
 * cudaEventSynchronize, so the wall is not comparable to a normal run -- the
 * same caveat RNA_PHASE_SYNC carries, for the same reason. Compare arms that
 * both have it set.
 *
 *   RNA_LAUNCH_STATS=1          collect and print a summary at teardown
 *   RNA_LAUNCH_STATS_CSV=path   also write seq,grid,device_ms,host_ms per launch
 */

#define RNA_LS_MAX 4000000

typedef struct {
  float         device_ms;
  float         host_ms;
  unsigned int  grid;
} rna_ls_rec_t;

static rna_ls_rec_t *g_ls_rec   = NULL;
static size_t        g_ls_n     = 0;
static size_t        g_ls_drop  = 0;
static cudaEvent_t   g_ls_beg;
static cudaEvent_t   g_ls_end;
static int           g_ls_ev_ok = 0;
static double        g_ls_t0    = 0.0;


extern "C" int
rnafold_launch_stats_enabled(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_LAUNCH_STATS");

    v = (e && e[0] && e[0] != '0') ? 1 : 0;

    if (v)
      fprintf(stderr,
              "device.cu                RNA_LAUNCH_STATS=1: every instrumented "
              "launch ends in an event sync. Per-launch device time becomes "
              "available and the WALL IS NOT COMPARABLE to a normal run.\n");
  }

  return v;
}


extern "C" void
rnafold_launch_stats_begin(void)
{
  if (!rnafold_launch_stats_enabled())
    return;

  if (!g_ls_ev_ok) {
    /* cudaEventDefault, not cudaEventBlockingSync: the wait is microseconds and
     * the spin is what the rest of this process already does. */
    if ((cudaEventCreate(&g_ls_beg) != cudaSuccess) ||
        (cudaEventCreate(&g_ls_end) != cudaSuccess)) {
      fprintf(stderr, "device.cu                RNA_LAUNCH_STATS: event create "
                      "failed, stats disabled\n");
      cudaGetLastError();
      return;
    }

    g_ls_rec = (rna_ls_rec_t *)malloc(sizeof(rna_ls_rec_t) * RNA_LS_MAX);

    if (!g_ls_rec) {
      fprintf(stderr, "device.cu                RNA_LAUNCH_STATS: out of host "
                      "memory, stats disabled\n");
      return;
    }

    g_ls_ev_ok = 1;
  }

  g_ls_t0 = rnafold_now_seconds();
  cudaEventRecord(g_ls_beg);
}


extern "C" void
rnafold_launch_stats_end(unsigned int grid)
{
  float ms = 0.0f;

  if ((!rnafold_launch_stats_enabled()) || (!g_ls_ev_ok))
    return;

  cudaEventRecord(g_ls_end);

  if (cudaEventSynchronize(g_ls_end) != cudaSuccess) {
    cudaGetLastError();
    return;
  }

  if (cudaEventElapsedTime(&ms, g_ls_beg, g_ls_end) != cudaSuccess) {
    cudaGetLastError();
    return;
  }

  if (g_ls_n < RNA_LS_MAX) {
    g_ls_rec[g_ls_n].device_ms = ms;
    g_ls_rec[g_ls_n].host_ms   = (float)((rnafold_now_seconds() - g_ls_t0) * 1000.0);
    g_ls_rec[g_ls_n].grid      = grid;
    g_ls_n++;
  } else {
    g_ls_drop++;
  }
}


static int
rna_ls_cmp(const void *a, const void *b)
{
  const float x = ((const rna_ls_rec_t *)a)->device_ms;
  const float y = ((const rna_ls_rec_t *)b)->device_ms;

  return (x < y) ? -1 : ((x > y) ? 1 : 0);
}


extern "C" void
rnafold_launch_stats_report(void)
{
  double  dsum = 0.0, hsum = 0.0;
  size_t  i, half;
  const char *csv;

  if ((!rnafold_launch_stats_enabled()) || (!g_ls_ev_ok) || (g_ls_n == 0))
    return;

  for (i = 0; i < g_ls_n; i++) {
    dsum += g_ls_rec[i].device_ms;
    hsum += g_ls_rec[i].host_ms;
  }

  /* Written BEFORE the sort, so the CSV keeps launch order -- the drift within
   * a run is half the point and sorting would destroy it. */
  csv = getenv("RNA_LAUNCH_STATS_CSV");

  if (csv && csv[0]) {
    FILE *f = fopen(csv, "w");

    if (f) {
      fprintf(f, "seq,grid,device_ms,host_ms\n");

      for (i = 0; i < g_ls_n; i++)
        fprintf(f, "%lu,%u,%.6f,%.6f\n", (unsigned long)i, g_ls_rec[i].grid,
                (double)g_ls_rec[i].device_ms, (double)g_ls_rec[i].host_ms);

      fclose(f);
      fprintf(stderr, "device.cu                RNA_LAUNCH_STATS: wrote %lu "
                      "launches to %s\n", (unsigned long)g_ls_n, csv);
    } else {
      fprintf(stderr, "device.cu                RNA_LAUNCH_STATS: cannot write "
                      "%s\n", csv);
    }
  }

  /* First and last tenth, in launch order, BEFORE sorting.
   *
   * NORMALISED PER BLOCK, and that is not a nicety. Grid size GROWS through a
   * sweep -- row i launches one block per (H,j) cell and j ranges further as i
   * falls -- so the raw ms of the last tenth is several times the first tenth
   * even in a perfectly steady run. Reporting raw ms here would manufacture a
   * thermal story out of the row geometry; the first version of this line did
   * exactly that, printing "+1518%" on an idle laptop. ns/block is still not
   * like-for-like (small grids are launch-bound and read high), so this is a
   * SMELL TEST only -- the grid-matched comparison needs the CSV. */
  {
    const size_t k = (g_ls_n >= 10) ? (g_ls_n / 10) : g_ls_n;
    double f10 = 0.0, l10 = 0.0, fb = 0.0, lb = 0.0;

    for (i = 0; i < k; i++) {
      f10 += g_ls_rec[i].device_ms;
      fb  += (double)g_ls_rec[i].grid;
    }

    for (i = g_ls_n - k; i < g_ls_n; i++) {
      l10 += g_ls_rec[i].device_ms;
      lb  += (double)g_ls_rec[i].grid;
    }

    fprintf(stderr,
            "device.cu                RNA_LAUNCH_STATS int_loop_kernel: %lu "
            "launches, device %.3f s, host %.3f s, overhead %.3f s (%.1f%%)\n",
            (unsigned long)g_ls_n, dsum / 1000.0, hsum / 1000.0,
            (hsum - dsum) / 1000.0,
            (hsum > 0.0) ? 100.0 * (hsum - dsum) / hsum : 0.0);
    fprintf(stderr,
            "device.cu                RNA_LAUNCH_STATS drift (ns/block, NOT "
            "grid-matched -- smell test only): first tenth %.1f, last "
            "tenth %.1f (%.1f%%)\n",
            (fb > 0.0) ? 1e6 * f10 / fb : 0.0,
            (lb > 0.0) ? 1e6 * l10 / lb : 0.0,
            ((fb > 0.0) && (lb > 0.0) && (f10 > 0.0))
              ? 100.0 * ((l10 / lb) - (f10 / fb)) / (f10 / fb) : 0.0);
  }

  qsort(g_ls_rec, g_ls_n, sizeof(rna_ls_rec_t), rna_ls_cmp);
  half = g_ls_n / 2;

  fprintf(stderr,
          "device.cu                RNA_LAUNCH_STATS device ms: min %.4f p50 "
          "%.4f p90 %.4f p99 %.4f max %.4f%s\n",
          (double)g_ls_rec[0].device_ms,
          (double)g_ls_rec[half].device_ms,
          (double)g_ls_rec[(size_t)(g_ls_n * 0.90)].device_ms,
          (double)g_ls_rec[(size_t)(g_ls_n * 0.99)].device_ms,
          (double)g_ls_rec[g_ls_n - 1].device_ms,
          g_ls_drop ? " (TRUNCATED -- raise RNA_LS_MAX)" : "");
}


/* ===================== THE CHUNK'S ROW TABLES, uploaded once =====================
 *
 * Luke's Flow Batching, fix 3. Every sweep row used to upload three small
 * tables -- size_off_H, side_off_H, i_H -- over the SAME device buffers, with a
 * BLOCKING copy on the default stream. Two things were wrong with that:
 *
 *   1. It is traffic the device should not need. In lock-step every row's
 *      tables are a function of the record lengths and nothing else, so they
 *      are all known before the first row runs. ~22 MB at 200 x 5601.
 *
 *   2. It is the stream-overlap level 2 race (Scaling notebook G, 2026-09-17:
 *      two runs, two DIFFERENT wrong shas, with the parity-event fix in). A
 *      blocking copy on the default stream waits for the default stream only.
 *      Once level 2 removed the per-row md sync the host ran ahead of the cell,
 *      hp and md streams and overwrote tables their queued kernels had not read
 *      yet.
 *
 * So each row gets a SLOT of its own, indexed by the sweep's iteration i, and
 * a slot is written exactly once per chunk. Nothing is ever overwritten under a
 * queued reader, whatever stream it is on and however far the host runs ahead.
 *
 *   lock-step (default)  every slot filled on the host, ONE upload before row 1
 *   continuous flow /    the row is only known when its iteration arrives
 *   a schedule           (slots turn over mid-sweep), so each slot is uploaded
 *                        as its row is built -- still a fresh slot, so still
 *                        race-free by construction.
 *
 * The host half is pinned, so the one big upload goes at the pinned rate, and
 * it is the host's own source of truth for the row: fill_arrays_loop.c reads its
 * size_off_H / side_off_H out of these slots rather than keeping a second copy
 * that could drift.
 *
 * Kernels are unchanged: each file binds its table pointers to row i's slot
 * right where it used to upload, and every launch passes those pointers as
 * before. The pointers MOVE every row now, which the md graph sees as a
 * parameter update -- graph_forced_reinstantiate_count is the check that this
 * costs no re-instantiation.
 */
/* Local twins of stub2.h's pinned helpers and gpuErrchk: this file does not
 * include stub2.h, and the row tables are the only thing here that needs them. */
static void *
rt_pinned_alloc(const size_t bytes, int *pinned)
{
  void *p = NULL;

  if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) {
    *pinned = 1;
    return p;
  }

  cudaGetLastError();
  *pinned = 0;
  return malloc(bytes);
}

static void
rt_pinned_free(void *p, const int pinned)
{
  if (!p) return;
  if (pinned) cudaFreeHost(p);
  else        free(p);
}

static void
rt_check(const cudaError_t rc, const char *what)
{
  if (rc != cudaSuccess) {
    fprintf(stderr, "device.cu                row tables: %s failed: %s\n",
            what, cudaGetErrorString(rc));
    exit(EXIT_FAILURE);
  }
}

static size_t *g_rt_size_h = NULL, *g_rt_side_h = NULL;   /* host, pinned */
static int    *g_rt_ih_h   = NULL;
static int     g_rt_size_pin = 0, g_rt_side_pin = 0, g_rt_ih_pin = 0;
static size_t *g_rt_size_d = NULL, *g_rt_side_d = NULL;   /* device */
static int    *g_rt_ih_d   = NULL;
static int     g_rt_nfiles = 0, g_rt_iters = -1;

extern "C" size_t
rnafold_rowtab_bytes(const int nfiles, const int iters)
{
  const size_t rows = (size_t)(iters + 1);

  return rows * (size_t)(nfiles + 1) * sizeof(size_t) * 2
       + rows * (size_t)nfiles * sizeof(int);
}

extern "C" void
rnafold_rowtab_end(void)
{
  /* Queued kernels on any stream may still hold slot pointers -- at level 2
   * nothing syncs before the end of the sweep -- so drain the device before
   * the tables go. */
  if (g_rt_size_d)
    rt_check(cudaDeviceSynchronize(), "drain before free");
  rt_pinned_free(g_rt_size_h, g_rt_size_pin);
  rt_pinned_free(g_rt_side_h, g_rt_side_pin);
  rt_pinned_free(g_rt_ih_h,   g_rt_ih_pin);
  if (g_rt_size_d) cudaFree(g_rt_size_d);
  if (g_rt_side_d) cudaFree(g_rt_side_d);
  if (g_rt_ih_d)   cudaFree(g_rt_ih_d);
  g_rt_size_h = g_rt_side_h = NULL; g_rt_ih_h = NULL;
  g_rt_size_d = g_rt_side_d = NULL; g_rt_ih_d = NULL;
  g_rt_nfiles = 0;
  g_rt_iters  = -1;
}

/* Rows 0..iters. Row 0 is never swept; it exists so that i indexes directly. */
extern "C" void
rnafold_rowtab_begin(const int nfiles, const int iters)
{
  const size_t rows = (size_t)((iters > 0 ? iters : 0) + 1);
  const size_t ob   = rows * (size_t)(nfiles + 1) * sizeof(size_t);
  const size_t ib   = rows * (size_t)nfiles * sizeof(int);

  rnafold_rowtab_end();

  g_rt_size_h = (size_t *)rt_pinned_alloc(ob, &g_rt_size_pin);
  g_rt_side_h = (size_t *)rt_pinned_alloc(ob, &g_rt_side_pin);
  g_rt_ih_h   = (int *)rt_pinned_alloc(ib ? ib : 1, &g_rt_ih_pin);
  if ((!g_rt_size_h) || (!g_rt_side_h) || (!g_rt_ih_h)) {
    fprintf(stderr, "device.cu                row tables: host allocation of %zu bytes failed\n",
            2 * ob + ib);
    exit(EXIT_FAILURE);
  }
  /* Zero, so an unswept slot (row 0, or a row a short sweep never reaches)
   * reads as "no record has any width" rather than as garbage. */
  memset(g_rt_size_h, 0, ob);
  memset(g_rt_side_h, 0, ob);
  memset(g_rt_ih_h,   0, ib ? ib : 1);

  if ((cudaMalloc((void **)&g_rt_size_d, ob) != cudaSuccess) ||
      (cudaMalloc((void **)&g_rt_side_d, ob) != cudaSuccess) ||
      (cudaMalloc((void **)&g_rt_ih_d, ib ? ib : 1) != cudaSuccess)) {
    fprintf(stderr, "device.cu                row tables: cudaMalloc of %zu bytes failed\n",
            2 * ob + ib);
    exit(EXIT_FAILURE);
  }

  g_rt_nfiles = nfiles;
  g_rt_iters  = (int)rows - 1;
}

static void
rowtab_check(const int i)
{
  if ((i < 0) || (i > g_rt_iters) || (!g_rt_size_h)) {
    fprintf(stderr, "device.cu                row tables: row %d outside 0..%d\n",
            i, g_rt_iters);
    exit(EXIT_FAILURE);
  }
}

extern "C" size_t *rnafold_rowtab_size_host(const int i) { rowtab_check(i); return g_rt_size_h + (size_t)i * (g_rt_nfiles + 1); }
extern "C" size_t *rnafold_rowtab_side_host(const int i) { rowtab_check(i); return g_rt_side_h + (size_t)i * (g_rt_nfiles + 1); }
extern "C" int    *rnafold_rowtab_ih_host(const int i)   { rowtab_check(i); return g_rt_ih_h   + (size_t)i * g_rt_nfiles; }

extern "C" const size_t *rnafold_rowtab_size(const int i) { rowtab_check(i); return g_rt_size_d + (size_t)i * (g_rt_nfiles + 1); }
extern "C" const size_t *rnafold_rowtab_side(const int i) { rowtab_check(i); return g_rt_side_d + (size_t)i * (g_rt_nfiles + 1); }
extern "C" const int    *rnafold_rowtab_ih(const int i)   { rowtab_check(i); return g_rt_ih_d   + (size_t)i * g_rt_nfiles; }

static void
rowtab_copy(const size_t lo_o, const size_t n_o, const size_t lo_i, const size_t n_i)
{
  rt_check(cudaMemcpy(g_rt_size_d + lo_o, g_rt_size_h + lo_o, n_o * sizeof(size_t), cudaMemcpyHostToDevice), "upload");
  rt_check(cudaMemcpy(g_rt_side_d + lo_o, g_rt_side_h + lo_o, n_o * sizeof(size_t), cudaMemcpyHostToDevice), "upload");
  if (n_i)
    rt_check(cudaMemcpy(g_rt_ih_d + lo_i, g_rt_ih_h + lo_i, n_i * sizeof(int), cudaMemcpyHostToDevice), "upload");
}

/* Lock-step: every slot, once, before the sweep. Blocking, so every kernel the
 * sweep issues afterwards reads finished tables. */
extern "C" void
rnafold_rowtab_upload_all(void)
{
  const size_t rows = (size_t)(g_rt_iters + 1);

  rowtab_check(0);
  rowtab_copy(0, rows * (g_rt_nfiles + 1), 0, rows * g_rt_nfiles);
}

/* Flow: one slot, as its row is built. Blocking, and before any kernel of the
 * row is issued; the slot has never been read, so no queued work can see it
 * change. */
extern "C" void
rnafold_rowtab_upload_row(const int i)
{
  rowtab_check(i);
  rowtab_copy((size_t)i * (g_rt_nfiles + 1), (size_t)(g_rt_nfiles + 1),
              (size_t)i * g_rt_nfiles, (size_t)g_rt_nfiles);
}


/* ===================== T2a: THE EXIT PATH, one copy stream per worker =====================
 *
 * Luke's Flow Batching, T2a. After the sweep every backtrack worker pulls its
 * record's c and fML triangles off the device. They used to do it with BLOCKING
 * PAGEABLE copies on the default stream, so twelve workers queued behind each
 * other on one stream and one driver staging buffer: ~6.5 GB/s at 400 x 5601,
 * 7.2 s of wall. The scratch they copy into is now pinned (kept for the life of
 * the process, so it is pinned once, not once per chunk) and each worker copies
 * on a stream of its own, so the copies run concurrently at the pinned rate, as
 * far as the card's copy engines allow.
 *
 * Streams are grow-only and never destroyed: a worker index always maps to the
 * same stream, and there is no per-chunk create/destroy cost.
 *
 * WHAT IS PINNED, AND WHY NOT THE SCRATCH. The first version pinned each
 * worker's whole scratch pair. The fetch fell ~10x, and pinning 1.7 GB cost
 * 35 worker-seconds (~3 s of wall) under WSL, more than the copy it saved on a
 * one-chunk fold, and it grows with record length x workers. So each worker
 * gets a FIXED pinned staging buffer instead (RNA_XFER_STAGE_MB, default 8):
 * the triangle comes across in slices on the worker's stream and is memcpy'd
 * into ordinary scratch, each worker doing its own memcpy. Pin cost is
 * workers x 8 MB, once per process. Local sweep (RTX 3050, WSL): 4 MB best, 256 MB
 * worst -- pinning dominates there; re-sweep on a native host.
 */
#define RT_XFER_MAX 256
static cudaStream_t g_xfer[RT_XFER_MAX];
static void        *g_xfer_stage[RT_XFER_MAX];
static int          g_xfer_stage_pin[RT_XFER_MAX];
static int          g_xfer_n = 0;

/* RNA_XFER_STAGE_MB: 0 = "no stage, pin the worker scratch itself", N = an N-MB
 * pinned stage per worker that the worker memcpys out of, UNSET = AUTO, decided
 * by measuring what page-locking costs on THIS host.
 *
 * WHICH ONE WINS IS A PROPERTY OF THE HOST, and both arms have now been
 * measured. On the A100 (Scaling I, 2026-09-17) the staged form took fetch_mx
 * 7.24 -> 1.85 s but pushed +1.93 s into backtrack -- that is the memcpy out of
 * the stage -- so pinning the scratch should be the better half. Under WSL the
 * opposite, and not marginally: pinning ~1.7 GB of scratch cost 27-35 worker-
 * seconds, and the exit path measured 3.09 s pinned against 0.91 s staged.
 *
 * So AUTO probes instead of guessing: page-lock 16 MB once, time it, and pin the
 * scratch only if the host does it faster than RT_PIN_GBPS_MIN. The probe costs
 * ~5 ms where pinning is cheap and ~0.3 s where it is dear -- which is exactly
 * the case that is about to save seconds. Same shape as RNA_BUILD_PIPELINE's
 * memory gate: measure the host, do not assume it. */
#define RT_PIN_PROBE_BYTES (16u << 20)
/* THE THRESHOLD, and it was wrong once already.
 *
 * 0.25 s/GB was a guess with margin. Scaling I (2026-09-18, A100) then showed
 * it choosing the WRONG arm on the host that matters: every AUTO arm came back
 * with the staged signature, while the forced-pin arm was better --
 * fetch_mx 1.94 -> 0.60 s for +0.37 s of allocation, the exit path 9.61 -> 8.64
 * (-10.1%). Pinning there costs about 0.25 s/GB, exactly on the boundary.
 *
 * So: 0.8 s/GB. It picks PINNED on the A100 (~0.25 measured) and STAGED under
 * WSL (1.13 measured, where pinning lost 3.09 s against 0.91 s staged). The gap
 * between the two hosts is 4.5x, so a threshold in the middle is not a knife
 * edge -- but it IS a two-point calibration, and a third host is allowed to
 * move it. The probe is single-threaded while the pool it decides for is
 * allocated serially, which is why a rate comparison is meaningful at all. */
#define RT_PIN_SECONDS_PER_GB_MAX 0.8

static int
rt_pin_is_cheap(void)
{
  static int v = -1;

  if (v < 0) {
    void        *p     = NULL;
    const double t0    = rnafold_now_seconds();
    const int    ok    = (cudaHostAlloc(&p, RT_PIN_PROBE_BYTES, cudaHostAllocDefault) == cudaSuccess);
    const double spent = rnafold_now_seconds() - t0;
    const double per_gb = spent * (1073741824.0 / (double)RT_PIN_PROBE_BYTES);

    if (ok)
      cudaFreeHost(p);
    else
      cudaGetLastError();

    v = (ok && (per_gb < RT_PIN_SECONDS_PER_GB_MAX)) ? 1 : 0;
    fprintf(stderr,
            "device.cu                pinning costs %.2f s/GB here -> backtrack "
            "scratch %s (RNA_XFER_STAGE_MB to override)\n",
            per_gb, v ? "PINNED, no stage" : "unpinned, copied through an 8 MB stage");
  }

  return v;
}

static long
rt_stage_mb(void)
{
  static long v = -2;

  if (v == -2) {
    const char *e = getenv("RNA_XFER_STAGE_MB");

    if (e && e[0]) {
      v = atol(e);
      if (v < 0)
        v = 0;
    } else {
      v = -1;                       /* AUTO: decided on first use, below */
    }
  }

  if (v == -1)
    return rt_pin_is_cheap() ? 0 : 8;

  return v;
}

extern "C" int
rnafold_xfer_pin_scratch(void)
{
  return rt_stage_mb() == 0;
}

static size_t
rt_stage_bytes(void)
{
  const long mb = rt_stage_mb();

  return (size_t)(mb > 0 ? mb : 8) << 20;
}

/* Called once per backtrack phase, from the thread that spawns the workers,
 * BEFORE they start: creation is not thread-safe here and does not need to be. */
extern "C" void
rnafold_xfer_begin(const int n)
{
  const int want = (n < 1) ? 1 : ((n > RT_XFER_MAX) ? RT_XFER_MAX : n);

  while (g_xfer_n < want) {
    rt_check(cudaStreamCreate(&g_xfer[g_xfer_n]), "copy stream create");
    g_xfer_stage[g_xfer_n] = rnafold_xfer_pin_scratch()
                             ? NULL    /* the scratch is pinned; no stage needed */
                             : rt_pinned_alloc(rt_stage_bytes(), &g_xfer_stage_pin[g_xfer_n]);
    if ((!rnafold_xfer_pin_scratch()) && (!g_xfer_stage[g_xfer_n])) {
      fprintf(stderr, "device.cu                copy stage allocation failed\n");
      exit(EXIT_FAILURE);
    }
    g_xfer_n++;
  }
}

/* Device-to-host for backtrack worker w. w < 0 is the old blocking copy on the
 * default stream. Otherwise: slices through w's pinned stage, on w's stream. */
extern "C++" void
rnafold_d2h_w(void *dst, const void *src, const size_t bytes, const int w)
{
  if (w < 0) {
    rt_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "fetch");
    return;
  }

  if (g_xfer_n == 0)
    rnafold_xfer_begin(1);

  {
    const int          k     = w % g_xfer_n;
    const cudaStream_t s     = g_xfer[k];
    const size_t       slice = rt_stage_bytes();
    size_t             off;

    /* Pinned scratch: straight into the destination, no stage and no memcpy. */
    if (rnafold_xfer_pin_scratch()) {
      rt_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s), "fetch");
      rt_check(cudaStreamSynchronize(s), "fetch sync");
      return;
    }

    for (off = 0; off < bytes; off += slice) {
      const size_t n = (bytes - off < slice) ? (bytes - off) : slice;

      rt_check(cudaMemcpyAsync(g_xfer_stage[k], (const char *)src + off, n,
                               cudaMemcpyDeviceToHost, s), "fetch slice");
      rt_check(cudaStreamSynchronize(s), "fetch slice sync");
      memcpy((char *)dst + off, g_xfer_stage[k], n);
    }
  }
}

/* Worker w's stream. A worker beyond the pool shares one by index, which is
 * still correct: each copy is followed by a sync of its own stream. */
extern "C++" cudaStream_t
rnafold_xfer_stream(const int w)
{
  if (g_xfer_n == 0)
    rnafold_xfer_begin(1);

  return g_xfer[((w < 0) ? 0 : w) % g_xfer_n];
}

/* Pinned host ints for the scratch pool, with the malloc fallback: a host that
 * will not pin should be slow, not broken. Non-fatal, unlike
 * cuda_host_alloc_ints(). */
extern "C" int *
rnafold_pinned_ints(const size_t n, int *pinned)
{
  return (int *)rt_pinned_alloc(n * sizeof(int), pinned);
}

extern "C" void
rnafold_pinned_ints_free(int *p, const int pinned)
{
  rt_pinned_free(p, pinned);
}

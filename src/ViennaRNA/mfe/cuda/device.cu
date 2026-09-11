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

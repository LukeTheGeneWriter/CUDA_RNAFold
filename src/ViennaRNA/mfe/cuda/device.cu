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

/*
 *  gpu_batch.c -- RNAfold's GPU chunk path (cleanup C6b, decision D3).
 *
 *  Moved out of RNAfold.c, not rewritten. RNAfold.c reads the records, formats
 *  every result and owns the -j pool; this file holds records back into chunks,
 *  folds each chunk through vrna_mfe_batch() and hands the records back with
 *  their results attached. The seam is RNAfold_gpu.h:
 *
 *    gpu_batch_begin()   process_input()'s GPU decision, once per input stream
 *    gpu_batch_offer()   one record: into the chunk, or back to the caller
 *    gpu_batch_end()     the tail chunk, the pipeline drain and the reports
 *
 *  process_input()'s locals for the accumulator are the file statics below, and
 *  RUN_IN_PARALLEL(process_record, r) is rnafold_dispatch_record(r) (RNAfold.c):
 *  parallel_helpers.h DEFINES the pool's globals, so only one file may include it.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef VRNA_WITH_CUDA

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>   /* SIZE_MAX: a RAM budget with no ceiling */
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>     /* pow(): the admission model's host term */
#include <pthread.h>
#ifdef __GLIBC__
#include <malloc.h>   /* malloc_trim(): finish_gpu_batch() returns a chunk's tables */
#endif

#include "ViennaRNA/fold_vars.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/strings.h"
#include "ViennaRNA/utils/log.h"
#include "ViennaRNA/io/commands.h"
#include "ViennaRNA/sequences/utils.h"
#include "ViennaRNA/datastructures/stream_output.h"

#include "RNAfold_gpu.h"

#include "ViennaRNA/mfe/cuda/engine.h"

/* Per-chunk teardown. The device buffers are sized for one chunk's (nfiles,
 * length) by init_gpu/2/3 and are NOT reset by par_mfe() itself, so a second
 * call without these reuses dirty state -- observed as
 *   Assertion `my_c[tri_off_H[H]+ij] == INF' failed
 * on the very next chunk. The 2.3.0 driver called them at the end of every
 * chunk for exactly this reason. */
extern void teardown_gpu(void);
extern void teardown_gpu2(void);
extern void teardown_gpu3(void);

/* VRAM budgeting. Both live in the library already:
 *   gpu_bytes_per_file(len)      device bytes one record of that length needs
 *   compute_gpu_usable_bytes()   free VRAM * 0.85, capped by
 *                                RNA_GPU_VRAM_BUDGET_MB (which can only lower
 *                                it, never raise it). */
extern size_t gpu_bytes_per_file(const int length);
/* CIRCULAR: tell the VRAM model that fM2_real will be allocated, before any
 * chunk is sized against it. See modular_decomposition.cu. */
extern void   rnafold_circ_expect(const int circ);
/* N5a P0: per-chunk spans beside the RNA_RECORD_TRACE file, on its clock. */
extern void   rnafold_chunk_trace_open(const char *record_path, const double base);
/* N5a P2: a batch's backtrack may run beside the next sweep. Internal hooks,
 * mfe_cuda.c -- vrna_mfe_batch() itself is unchanged (decision 9.1). */
extern int    rnafold_bt_pipeline(void);
extern void   rnafold_bt_pipeline_attach(void (*done)(void *ctx));
extern void   rnafold_bt_pipeline_ctx(void *ctx);
extern int    rnafold_bt_pipeline_went_async(void);
extern void   rnafold_bt_pipeline_drain(void);
extern size_t rnafold_bt_pending_device_bytes(void);   /* P3 rule (b), mfe_cuda.c */
extern size_t compute_gpu_usable_bytes(void);
/* The soft limit alone, and the warning for a record over the VRAM budget (B2). */
extern size_t compute_gpu_usable_bytes_soft(void);
extern void   rnafold_vram_over_budget(const size_t need, const size_t allowed);
/* int16 AUTO prices a full-cap chunk: it needs the cap this file admits by (modular_decomposition.cu). */
extern void   rnafold_set_chunk_cells_cap(const unsigned long long cap_cells);
/* The four stage counters that mfe_cuda.c PRINTS but nothing incremented until
 * 2026-09-08. They cover work that happens in this file rather than in the CUDA
 * layer -- building and freeing fold compounds, device teardown, and output --
 * which is why they were never wired up: the counter lives with the reporter,
 * the work lives here, and nobody closed the gap. The deep profile found 61% of
 * wall at 120x5601 outside every timer; these are the first four candidates.
 * See docs/archive/PROFILE272_DEEP_RESULTS.md. */
extern double stage_build_s, stage_output_s, stage_teardown_s, stage_free_s;
extern double rnafold_now_seconds(void);



/* Fold one accumulated chunk on the GPU, then dispatch its records normally.
 *
 * The records are dispatched AFTER folding, each carrying its result, so all
 * output formatting stays in process_record() and is shared with the
 * per-record path. Nothing about ordering changes: the ostream slots were
 * requested at read time.
 */
/* May this RUN use the GPU chunk path at all?
 *
 * This is what makes the CUDA build a strict accelerator rather than a variant
 * of RNAfold with a different feature set. Anything the device path does not
 * reproduce exactly is not refused and is not approximated -- it simply goes
 * down upstream's own per-record path, which is the validated CPU code. The
 * user sees ViennaRNA 2.7.2's answers for every option; the GPU only ever
 * accelerates the cases where it agrees.
 *
 * Two layers, deliberately:
 *   here            a RUN-level decision from the model details and the
 *                   options, made once, before any record is committed;
 *   par_fill_arrays a per-batch backstop that ERRORS if an unsupported model
 *                   reaches the sweep anyway. It should now be unreachable,
 *                   and is kept precisely so that "unreachable" is enforced
 *                   rather than assumed.
 *
 * Note what is NOT excluded: the partition function, MEA, centroid, plots and
 * probability output all keep working, because process_record() still does
 * every one of them itself on its own fold compound. The chunk path supplies
 * only the MFE energy and structure. That falls out of not duplicating
 * process_record(), and it is why -p needs no special handling here.
 */
static int
gpu_path_usable(struct options *opt,
                const char    **why)
{
  vrna_md_t *md = &(opt->md);

#define NO(msg) do { if (why) *why = (msg); return 0; } while (0)
/* Gate 1's half of the RNA_ENGINE_ALLOW test hook (mfe/cuda/engine.c).
   Gate 1 is an OPTIMISATION and gate 2 is the authority, so a probe has to
   lift both or the fold never reaches the one being measured. */
#define NO_UNLESS(id, msg) \
  do { if (!vrna_cuda_engine_allow(id)) NO(msg); } while (0)

  /* Model details the sweep does not implement. Mirrors
   * vrna_cuda_engine_supports() in mfe/cuda/engine.c; kept in step with it. */
  /* DANGLE MODEL 0 ACCEPTED 2026-09-11, alongside the long-standing 2.
   *
   * d0 and d2 share the RECURSION -- mfe_multibranch.c:686 dispatches
   * ml_pair_d0 or ml_pair_d2 and BOTH read only dmli1 -- so no new DP state was
   * needed.
   *
   * AND IT NEEDED EXACTLY ONE DEVICE CHANGE, which is not where I first looked.
   * Upstream ZEROES P->mismatchM when dangles == 0 (params.c:644-646), so
   * E_MLstem() already returns the bare-stem energy under d0 and both
   * multibranch sites were ALREADY correct. Measured: 0 nonzero mismatchM
   * entries at d0 against 175 at d2, and red-teaming those two sites changes
   * nothing. The one term that genuinely differed is
   * vrna_mfe_gquad_internal_loop()'s mismatchI (mfe_gquad.c:306, `if
   * (dangles)`) -- mismatchI is NOT zeroed (158 nonzero at both models), so
   * -d0 -g was the only combination actually returning a wrong answer.
   * Red-team: 20 differing lines with that gate removed.
   *
   * d1 AND d3 STAY DECLINED, and not for want of effort: ml_pair_d1 needs
   * dmli2 as WELL as dmli1 -- a second DMLi generation the sweep does not
   * carry -- plus the `dangle_model % 2` terms at :1015 and :1376, and d3 adds
   * coaxial stacking at :1501. That is new DP state, not new arithmetic.
   *
   * The exterior loop needed nothing: vrna_mfe_exterior_f5() is upstream's own
   * and runs on the host, as does vrna_mfe_multibranch_m1() for fM1 under
   * uniq_ML, so both are dangle-correct for every model by construction.
   */
  if ((md->dangles != 0) && (md->dangles != 2))
    NO_UNLESS("dangles", "dangle model 1 or 3 (0 and 2 are accelerated)");
  /* G-QUADRUPLEXES ACCEPTED 2026-09-10 (G3).

   * The sweep now scores quadruplexes into c and fML: c_gq is carried to the
   * device (G0), extend_fm_3p()'s multibranch term is in fml_scan_kernel (G1),
   * and vrna_mfe_gquad_internal_loop()'s three bounded (p,q) sweeps are in
   * gq_internal_kernel (G2).
   *
   * The last blocker was NOT the recursion. With the energy already byte-exact,
   * the STRUCTURE still came out with a single '+' where a quadruplex belongs,
   * because vrna_backtrack_from_intervals() discards a gquad's layout when it
   * downconverts to the legacy vrna_bp_stack_t. Fixed by
   * VRNA-PATCH(bps-backtrack): the port now backtracks into a vrna_bps_t and
   * renders with vrna_db_from_bps().
   *
   * BAR: byte-identical to pristine 2.7.2 on the frozen tests/gquad/ set (6
   * records, 80-1200 nt, every one containing a quadruplex), and GPU==CPU on 20
   * further G-rich records at 70-1100 nt, alone and combined with --noLP,
   * --noGU, -4, --salt and -T. Regression test: tests/mfe_cuda_gquad.ts.
   */
  /* CIRCULAR RNA ACCEPTED 2026-09-11.

   * The sweep now persists fM2_real. It was never new arithmetic:
   * modular_decomposition_kernel already reduces min_k(fML[i,k]+fML[k+1,j])
   * into DMLi every row -- exactly what upstream's mfe_multibranch_m2_fast()
   * computes -- and the fork discarded it one row later. It now also stores it
   * into a persistent triangle, and backtrack_one_slot() calls
   * VRNA-PATCH(circular-postprocess) on it.
   *
   * COSTS A CHUNK WIDTH. fM2_real is a second full int32 triangle per record,
   * counted in modular_decomposition_bytes_per_file() under
   * rnafold_circ_expect(), so a circular batch admits proportionally fewer
   * records rather than OOMing.
   *
   * BAR: byte-identical to pristine 2.7.2 on the frozen tests/circ/ set (6
   * records, 60-600 nt, every circular energy differing from its linear one),
   * and GPU==CPU on 20 further records at 45-570 nt -- plain, chunked, int16,
   * --noLP, --noGU, --salt, -T and -4. tests/mfe_cuda_circ.ts is the
   * regression bar; RNA_CIRC_VERIFY=1 re-checks fM2_real against its own
   * definition, O(n^3), for small inputs.
   */
  /* --noClosingGU ACCEPTED 2026-09-11.

   * It was the last multi-gate option, and it was half implemented: the
   * hairpin/multibranch half already worked (rnafold_hc_opt() drops HP_LOOP and
   * MB_LOOP for a GU/UG pair exactly as hard.c:786-791 does, and new_c_kernel
   * skips both terms on gate bit 1), while the interior-loop half did nothing
   * at all -- so c was internally inconsistent rather than merely suboptimal.
   *
   * The missing half is two rules, both from mfe/mfe_internal.c: a GU/UG pair
   * may neither CLOSE nor BE ENCLOSED BY a bulge or interior loop. STACKS ARE
   * EXEMPT -- mfe_stacks() carries no such test, and vrna_E_internal() returns
   * the stack energy before consulting no_close. Energy() now skips those
   * candidates, and gq_internal_kernel returns early for a GU closing pair
   * because vrna_mfe_gquad_internal_loop() is called from inside upstream's
   * own `if (!noclose)` block.
   *
   * BAR: byte-identical to pristine 2.7.2 with --noClosingGU across the frozen
   * fixtures; RNA_HC_VERIFY=1 proves the mask half against the host's own
   * hc->mx under this flag. tests/mfe_cuda_noclosinggu.ts is the regression
   * bar.
   */

  /* NOT rejected, and each for a reason that was measured rather than assumed:
   *
   *   uniq_ML  the MFE recursion never reads fM1, so the ANSWER is right.
   *            Note the sweep leaves fM1 entirely INF (fill_arrays.c:271);
   *            RNAfold never reads it, but the LIBRARY guard in
   *            mfe/cuda/engine.c still declines uniq_ML for that reason, since
   *            a vrna_mfe_batch() caller might go on to call vrna_subopt().
   *   (logML is NOT here: it stays declined. See mfe/cuda/engine.c -- 12/12 at
   *    160 nt is a sample, and RNAfold has no --logML flag to bar it with.)
   *   noGU     reaches the sweep through the uploaded hc->mx bitmasks, which
   *            upstream has already populated.
   *
   * All three verified byte-identical against the CPU route over the full
   * reference set before the rejection was lifted; tools/verify_option_parity.sh
   * covers them.
   *
   * noLP is now ACCEPTED (2026-09-07). The earlier note here ("disagreed on 3
   * of 12 records") understated the defect badly, because it compared ENERGIES.
   * Diagnosed
   * properly 2026-09-07 by lifting both guards and re-evaluating each returned
   * structure with RNAeval -- the check that needs no oracle:
   *
   *   CPU  rec 0: reports -295.60, structure re-evaluates to -295.60, 274 pairs
   *   GPU  rec 0: reports -295.60, structure re-evaluates to  -77.18,  90 pairs
   *   GPU  rec 1: reports -294.94, structure re-evaluates to   +5.54,   6 pairs
   *
   * 8 of 8 sampled records inconsistent by 87 to 300 kcal. The matrix fill and
   * the backtrack do not agree WITH EACH OTHER -- the same failure shape as the
   * hard-constraint bug, and invisible to an energy comparison because the f5
   * value is roughly right while the structure is not.
   *
   * Cause, and it is not "ptype is not enough". noLP changes what the
   * recursion WRITES: at mfe/mfe.c:4413 upstream stores cc1[j-1]+stackEnergy
   * into c[ij] rather than new_c, carrying the unconstrained value sideways in
   * the cc/cc1 row buffers. That is what forbids a helix of length one. The
   * sweep stores new_c, so upstream's backtrack -- which branches on noLP at
   * mfe/mfe.c:4289 and calls vrna_bt_stacked_pairs() -- walks a matrix built
   * under a different convention than the one it assumes.
   *
   * fill_arrays_loop.c:215 still carries the deleted machinery as a comment,
   * marked "gcov says not used". It said that because noLP was never exercised.
   *
   * FIXED and the guard lifted: a cc/cc1 row-buffer pair, a stack_row_kernel
   * that computes upstream's vrna_eval_stack() per row, and new_c_kernel
   * writing the stacked value into c while carrying the unconstrained new_c in
   * cc. The BACKTRACK needed nothing -- vrna_backtrack_from_intervals() wraps
   * upstream's own backtrack(), which already handles noLP; it was mis-walking
   * our matrix precisely because the matrix was wrong.
   *
   * Verified: 60/60 records byte-identical to upstream --noLP (was 1/60), every
   * structure re-evaluates to its own reported energy, zero lonely pairs, seven
   * reference workloads including F_extreme and a 1600-record batch, and
   * identical across 2/3/4/7 chunks. --noLP + RNA_FML_INT16 is REFUSED at init
   * (fill_arrays.c): noLP puts near-INF FINITE values into fML that the 16-bit
   * per-block offsets cannot represent. */
  if (md->energy_set != 0)      NO_UNLESS("energy_set", "non-default energy set");
  /* salt is no longer barred: the multibranch kernel inherits it through the
   * parameter tables, and the hairpin/internal kernels each add one term from
   * a host-built table. tools/verify_salt_parity.sh is the bar. */

  /* Per-record data that reaches the recursion as constraints.
   *
   * Hard constraints were MEASURED here, not assumed, and they do not work: on
   * a 12x80 nt batch with a forced-unpaired block the sweep returned -14.30 for
   * a structure worth -5.40. See the long note in mfe/cuda/engine.c, which
   * declines them at the library level -- that is the check that actually keeps
   * a vrna_mfe_batch() caller safe. This one is a routing decision on top:
   * barring the run here keeps -C on upstream's PARALLEL per-record path,
   * whereas letting the chunk build and then be declined would fold it serially
   * inside vrna_mfe_batch().
   *
   * flush_gpu_chunk() still applies constraints to the compounds it builds. It
   * is not dead code: it is what makes the declined-batch fallback fold the
   * right thing, and it has to be in place before this bar can lift. */
  /* -C ACCEPTED 2026-09-16 -- see the long note in mfe/cuda/engine.c. The
   * device carries hc->up_hp and hc->up_int now, not just hc->up_ml, and the
   * masks are packed after the depot is materialised. */
  if (opt->probing_data)        NO_UNLESS("soft", "probing/SHAPE data");
  /* --commands ACCEPTED 2026-09-16 -- conditionally, and the condition is not
   * checked here. build_one() now applies the file to the batch's own compound,
   * so gate 2 sees whatever it queued: hard constraints are supported, soft
   * constraints and unstructured domains are declined there by the checks that
   * already exist. A command file is not one thing, and this gate cannot tell
   * which one it is without applying it. */
  if (opt->mod_params)          NO_UNLESS("mod", "modified bases");
  if (opt->ligandMotif)         NO_UNLESS("motif", "a ligand motif");

#undef NO

  if (why)
    *why = NULL;

  return 1;
}


/* NO LONGER AN ADMISSION GATE (2026-09-27). This was the 2.3.0 record-count
 * threshold, and it decided whether a chunk went to the device at all. It does not
 * any more -- rnafold_chunk_earns_gpu() asks about WORK, in cells, through the
 * admission model below. See its WHAT IT REPLACES for
 * why a record count was the wrong unit (ten 80 nt sequences beat one 5601 nt one,
 * which is backwards) and why it had to go.
 *
 * ONE user remains: cpu_slice_take(), which holds records back from a chunk to fold
 * on the host and needs a floor on how many it may leave the GPU. That is a
 * different question -- how to SPLIT a chunk already bound for the device, not
 * whether to send one -- and a record count is a defensible unit for it, because the
 * records in one chunk are of similar length by construction. */
#define VRNA_MIN_GPU_BATCH 10
/* ============ THE ADMISSION MODEL (S3, 2026-10-18) ==========================
 *
 * WHAT IT REPLACES. From 2026-09-27 until S3 the test was one floor on a chunk's
 * total matrix cells, floor = F * R_host * jobs, with F the measured probe and
 * R_host = 6.7e5 cells/s. Cells was the right unit (it replaced a record count that
 * sent ten 80 nt records to the device and kept one 5601 nt record off it), but the
 * model was wrong in three ways that PORT_SHORT_SEQUENCES.md section 8.2 measured
 * (E-A2: 1-4096 records at 60-300 nt, -j1/-j4/-j12, every arm byte-identical to the
 * CPU's, each device arm's probe logged and subtracted):
 *
 *   - It had no PER-RECORD device term. Each record on the device path costs about
 *     0.68 ms of host-side build, fetch and trace whatever its length, so below about
 *     85 nt at -j12 the device loses at every N, and the floor admitted them anyway.
 *   - It assumed the host scales linearly with -j. Here -j12 is 3.75x one job.
 *   - It charged the probe alone as the fixed cost. The first real CUDA work
 *     (context, modules) costs another ~0.34 s beyond it.
 *
 * THE MODEL. For records of c = L(L+1)/2 cells each,
 *
 *     host wall  = sum t_h(c_i) / S(min(n, jobs))     t_h(c) = K c^1.15, one job
 *     device     = fixed + sum (h + k c_i)
 *     S(j)       = j / (1 + beta (j - 1))
 *
 * and the device is used when the host wall is the larger. `fixed` is what is still
 * to be paid: the probe prior plus C0 before the device is reached, C0 after the
 * probe and before the first chunk, and only the per-chunk cost after that. Two
 * limits fall out of it, which together are the minimum GPU batch by length squared:
 *
 *   - A MINIMUM LENGTH, where t_h(c)/S(jobs) = h + k c. Below it the device loses at
 *     any N, so such a record is never offered to the device (Luke, D1). It moves
 *     with -j (about 45 nt at -j1, 95 nt at -j12 here), so it is solved once per run.
 *   - A MINIMUM BATCH above it, set by the fixed cost. The measured crossovers
 *     (200 nt: 128 records at -j12, 32 at -j1) are reproduced to within one doubling.
 *
 * MACHINE DEPENDENCE. The constants are an RTX 3050 laptop's, against 12 logical
 * cores. beta especially: a server CPU scales closer to linearly, which moves every
 * crossover toward the host. RNA_GPU_COST_MODEL overrides any of them, as
 * "K=2.65e-7,h=6.8e-4,k=8.1e-8,C0=0.34,beta=0.18", and F (reaching the device, below)
 * as well; F alone is measured per machine, by configure. The A100 calibration (D3) and the
 * tuner of PORT_ZEROCONF_SCOPE.md section 3.1 are where better numbers come from. A
 * forced RNA_GPU_WORK_FLOOR still means the old total-cells test with no minimum
 * length, so RNA_GPU_WORK_FLOOR=0 is "always the device" for every harness that
 * forces the path.
 */
#define VRNA_GPU_MODEL_K      2.65e-7  /* host, one job: s per cell^1.15 (200 nt: 23.4 ms) */
#define VRNA_GPU_MODEL_EXP    1.15
#define VRNA_GPU_MODEL_H      6.8e-4   /* device path, per record: s */
#define VRNA_GPU_MODEL_KDEV   8.1e-8   /* device path, per cell: s */
#define VRNA_GPU_MODEL_C0     0.34     /* first real CUDA work beyond the probe: s */
#define VRNA_GPU_MODEL_BETA   0.18     /* host -j scaling: S(4) = 2.6, S(12) = 4.0 */
#define VRNA_GPU_MODEL_CHUNK  0.04     /* each chunk after the first: s */
#define VRNA_GPU_PROBE_PRIOR  0.08     /* the probe, when configure did not measure it */
#define VRNA_GPU_MODEL_LMAX   65536    /* the minimum-length search gives up here */

/* F BEFORE THE PROBE: WHAT REACHING THE DEVICE COSTS DEPENDS ON HOW LONG IT IDLED
 * (PORT_SHORT_SEQUENCES.md 9.2-9.3). On the RTX 3050 laptop: 0.07 s straight after another
 * process used it, 0.7 s after 0.3-5 s idle, 1.4 s from 10 s on, flat to 120 s. The
 * 0.08 above is the hot case only, and a one-shot run is the cold one.
 *
 * So configure measures both on the machine it runs on (m4/ac_rna_cuda.m4, 3b), and the
 * model budgets the COLD one (Luke, 2026-10-20). The cost is asymmetric: two small runs
 * back to back that come out faster than budgeted lose nothing, but two that each wake
 * a hot-budgeted device and take longer than the host would have are a regression.
 *
 * A build host is often not the run host (a cluster login node), so the measured value
 * applies only where gethostname() matches the host configure ran on; elsewhere the
 * prior above. The host name costs nothing to read and touches no device. */
#if defined(VRNA_GPU_WAKE_COLD_S) && defined(VRNA_GPU_WAKE_HOST)
static const double VRNA_GPU_WAKE_COLD = VRNA_GPU_WAKE_COLD_S;
static const char   VRNA_GPU_WAKE_AT[] = VRNA_GPU_WAKE_HOST;   /* "" = any host */
#endif

/* F, in seconds: measured at the vrna_cuda_devices() probe, and reported. The model
 * does not need it after the probe -- by then it is paid. */
static double g_gpu_reach_cost = -1.0;

/* jobs, captured where opt is in scope so the model can be evaluated from a static
 * without threading opt through two call sites that do not otherwise need it. */
static int g_host_jobs = 1;

/* Per PROCESS, not per input stream: a second input file re-probes (cheaply), but
 * the context and modules C0 pays for are still there. */
static int g_device_reached = 0;
static int g_device_chunks  = 0;   /* chunks the device has taken */

static struct {
  double K, h, k, C0, beta, F;
} g_model = { VRNA_GPU_MODEL_K, VRNA_GPU_MODEL_H, VRNA_GPU_MODEL_KDEV,
              VRNA_GPU_MODEL_C0, VRNA_GPU_MODEL_BETA, VRNA_GPU_PROBE_PRIOR };

/* Where g_model.F came from, for the ON line. */
static const char *g_F_source = "default; configure did not measure it";

static int g_min_len = 0;   /* the minimum length, nt; solved by gpu_model_begin() */
static int g_short_n = 0;   /* records kept on the host for being under it */

/* ADMISSION CAP (2026-09-27, Luke: "shoot for 48 residents").
 *
 * MEASURED, and the number is a proxy for something else. Flow3 section B at
 * 96 x 5601 nt: 4 residents 40.60 s, 24 -> 22.64, 48 -> 21.12 (best), 96 -> 22.32.
 * So 48 beats both the wider and the narrower arms. But 48 is exactly n/2 in that
 * fixture, and at 200 x 5601 the best measured width was 100 -- also n/2. In both
 * cases the optimum is TWO CHUNKS, which is the minimum that lets the build
 * pipeline overlap anything (Flow4 A: one chunk overlaps 0.00 s, two overlap
 * 4.96 s of a 9.65 s build), while more chunks start paying per-chunk cost.
 *
 * WE CANNOT ASK FOR "TWO CHUNKS". The accumulator is streaming -- records arrive
 * one at a time and n is not known until the input ends -- so the policy has to be
 * expressible per-record. A width cap is that, and expressing it in CELLS rather
 * than records is what makes it length-independent: the same cap yields ~48
 * records at 5601 nt and ~586 at 1600, which is the right direction, because SM
 * fill depends on cells per row and not on how many records supply them.
 *
 * THAT RISK WAS MEASURED AND IT IS NOT REAL AT SCALE (A100, 2026-09-27). The worry
 * was that a fixed cap over-splits inputs larger than its fixture, from 200 x 5601
 * where 2 chunks beat 7. At 400 x 5601 the opposite holds, and the mechanism is the
 * build pipeline:
 *
 *     cap off      81.96 s   2 chunks   3.23 s of an 18.85 s build hidden
 *     cap default  75.60 s   9 chunks  18.05 s hidden
 *     cap 2x       73.76 s   5 chunks  15.32 s hidden
 *
 * `off` is the WORST arm. More chunks buy more overlap than they cost, because a
 * chunk boundary is the only place the build can hide -- so capping is what makes the
 * build nearly free (89 % hidden, against the 51 % this comment used to imply). 2x
 * the default wins by 1.8 s, so the cap is slightly too aggressive and the optimum
 * sits between 5 and 9 chunks: a tuning question for section 3.2, not a design flaw.
 *
 * AND THE THING THAT WAS SUPPOSED TO REPLACE IT DOES NOT. Per-record overlap under
 * continuous flow was the proposed alternative that needs no splitting. Measured at
 * all four stress shapes, `overlapped` is FLAT to within 2 % across batch, continuous
 * flow and slot flow -- flow creates no extra overlap opportunity at all. Chunk count
 * is the lever; flow is not.
 *
 * 48 * (5601*5602/2) = 7.53e8 cells. RNA_GPU_CHUNK_CELLS overrides; 0 disables.
 */
#define VRNA_GPU_CHUNK_CELLS VRNA_GPU_CHUNK_CELLS_DEFAULT   /* engine.h: the library uses it too */

static unsigned long long
rnafold_gpu_chunk_cells(void)
{
  static long long v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_GPU_CHUNK_CELLS");

    v = (long long)VRNA_GPU_CHUNK_CELLS;
    if ((e) && (e[0])) {
      const long long n = atoll(e);

      if (n >= 0) {
        v = n;
        fprintf(stderr, "%-24s RNA_GPU_CHUNK_CELLS=%lld (default %llu): chunk width "
                        "capped by matrix area%s\n", "bin/RNAfold.c", v,
                        (unsigned long long)VRNA_GPU_CHUNK_CELLS,
                        v ? "" : " -- DISABLED, the VRAM budget alone decides");
      }
    }
  }

  return (unsigned long long)v;
}


/* The CPU-fallback threshold, overridable so it can be MEASURED.
 *
 * Measured 2026-09-07 on 60 x 900 nt: this constant is what made int16 look
 * like a REGRESSION in benchmark v3's arm F. Chunk capacity comes from the
 * VRAM budget, so the final chunk holds a quantisation remainder -- and
 * int16's larger chunks leave a LARGER remainder. At a 48 MB budget int32
 * packed 14/chunk and left 4 records behind (2.6 s on the CPU) while int16
 * packed 18/chunk and left 6 (4.0 s). Both did ~2.7 s of GPU work, so the
 * whole apparent 0.86x regression was the CPU tail -- and nothing could see
 * it, because the benchmark's validity gate tested `sweeps == 0`, which
 * catches only a TOTAL fallback and never a partial one.
 *
 * Per-chunk GPU cost measured at 900 nt is 0.36 s + 0.025 s/record against
 * 0.65 s/record on the CPU, so break-even there is under ONE record: at that
 * length the fallback can never win. That is consistent with the length
 * dependence noted above -- the CPU fold and the GPU marginal cost are both
 * O(L^3) while the per-chunk fixed cost is not, so the break-even count goes
 * as L^-3. That is also why those three documented points line up: 65 at
 * 300 nt gives 8.1 at 600 and 1.0 at 1200.
 *
 * An env override rather than a new constant, deliberately: the default stays
 * behaviour-preserving, and ONE binary can be folded both ways and compared
 * against itself -- the shape that settled RNA_GPU_SWEEP and RNA_FML_INT16.
 * It announces itself for the reason the int16 gate does: a run that did not
 * apply the setting must not be able to pass for one that did.
 *
 * The VRAM site below no longer uses this at all. It asked "can free VRAM hold a
 * worthwhile batch at this length?" and answered with ten records, which on a small
 * card silently excluded every long sequence -- the ones the device helps most. It
 * now asks only "does ONE record fit", which is a capacity fact rather than a
 * policy, and the work floor answers the policy question.
 */
static int rnafold_min_gpu_batch(void);   /* defined just below */

/* A forced work floor, in triangular matrix cells, or -1 when none is set and the
 * model decides.
 *
 * RNA_GPU_WORK_FLOOR is the name, and RNA_MIN_GPU_CELLS is accepted as an alias so
 * the harnesses and notebooks that already set it keep working. Forced, it is the
 * pre-S3 test exactly: a chunk goes to the device once its total cells reach it, and
 * no record is too short. 0 means "no floor": always use the device.
 */
static double
rnafold_gpu_work_floor(void)
{
  static int    read = 0;
  static double v    = -1.0;

  if (!read) {
    const char *e = getenv("RNA_GPU_WORK_FLOOR");

    read = 1;
    if ((!e) || (!e[0]))
      e = getenv("RNA_MIN_GPU_CELLS");

    if ((e) && (e[0]) && (atof(e) >= 0.0)) {
      v = atof(e);
      fprintf(stderr, "%-24s work floor forced to %.0f cells: a chunk goes to the "
                      "device once its total matrix area reaches this\n",
              "bin/RNAfold.c", v);
    }
  }

  return v;
}


static double
gpu_model_host_s(const double cells)   /* one job */
{
  return g_model.K * pow(cells, VRNA_GPU_MODEL_EXP);
}


static double
gpu_model_dev_s(const double cells)
{
  return g_model.h + g_model.k * cells;
}


/* How much faster than one job the host folds n records: no more than n of them
 * run at once, which is the wave quantisation a few records at a high -j need. */
static double
gpu_model_speedup(const int n)
{
  const int j = (n < g_host_jobs) ? ((n > 0) ? n : 1) : g_host_jobs;

  return (double)j / (1.0 + g_model.beta * (double)(j - 1));
}


/* What the device still costs before it folds anything more. */
static double
gpu_model_fixed_s(void)
{
  if (!g_device_reached)
    return g_model.F + g_model.C0;

  return (g_device_chunks == 0) ? g_model.C0 : VRNA_GPU_MODEL_CHUNK;
}


/* Work, summed as the model needs it. */
struct gpu_work {
  int     n;
  double  cells;
  double  host_s;   /* one job */
  double  dev_s;
};


static void
gpu_work_add(struct gpu_work  *w,
             const double     L)
{
  const double c = L * (L + 1.0) / 2.0;

  w->n++;
  w->cells  += c;
  w->host_s += gpu_model_host_s(c);
  w->dev_s  += gpu_model_dev_s(c);
}


static int
gpu_work_earns(const struct gpu_work *w)
{
  const double floor_cells = rnafold_gpu_work_floor();

  if (floor_cells >= 0.0)
    return (floor_cells == 0.0) || (w->cells >= floor_cells);

  return w->host_s / gpu_model_speedup(w->n) > gpu_model_fixed_s() + w->dev_s;
}


/* D1: is a record of L nt long enough for the device at all? */
static int
gpu_record_admissible(const double L)
{
  return (rnafold_gpu_work_floor() >= 0.0) || (L >= (double)g_min_len);
}


/* Once per input stream, after g_host_jobs is known: read RNA_GPU_COST_MODEL (once
 * per process) and solve the minimum length for this -j. The marginal saving
 * t_h/S - (h + k c) is convex in c and negative at c = 0, so it crosses zero once,
 * and the first L where it is positive is the answer. */
static void
gpu_model_begin(void)
{
  static int read = 0;
  int        L;

  if (!read) {
    const char *e = getenv("RNA_GPU_COST_MODEL");

    read = 1;
#if defined(VRNA_GPU_WAKE_COLD_S) && defined(VRNA_GPU_WAKE_HOST)
    {
      char host[256] = "";

      if (!VRNA_GPU_WAKE_AT[0]) {
        g_model.F  = VRNA_GPU_WAKE_COLD;
        g_F_source = "cold, given to configure";
      } else if ((gethostname(host, sizeof(host) - 1) == 0) &&
                 (!strcmp(host, VRNA_GPU_WAKE_AT))) {
        g_model.F  = VRNA_GPU_WAKE_COLD;
        g_F_source = "cold, measured by configure";
      } else {
        g_F_source = "default; configure measured another host";
      }
    }
#endif
    if ((e) && (e[0])) {
      char *s = strdup(e), *tok, *save = NULL;

      for (tok = strtok_r(s, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        char *eq = strchr(tok, '=');

        if (!eq)
          continue;

        *eq = '\0';
        if (!strcmp(tok, "K"))
          g_model.K = atof(eq + 1);
        else if (!strcmp(tok, "h"))
          g_model.h = atof(eq + 1);
        else if (!strcmp(tok, "k"))
          g_model.k = atof(eq + 1);
        else if (!strcmp(tok, "C0"))
          g_model.C0 = atof(eq + 1);
        else if (!strcmp(tok, "beta"))
          g_model.beta = atof(eq + 1);
        else if (!strcmp(tok, "F")) {
          g_model.F  = atof(eq + 1);
          g_F_source = "RNA_GPU_COST_MODEL";
        } else
          fprintf(stderr, "%-24s RNA_GPU_COST_MODEL: unknown term '%s', ignored\n",
                  "bin/RNAfold.c", tok);
      }
      free(s);
      fprintf(stderr, "%-24s RNA_GPU_COST_MODEL: K=%g h=%g k=%g C0=%g beta=%g F=%g\n",
              "bin/RNAfold.c", g_model.K, g_model.h, g_model.k, g_model.C0, g_model.beta,
              g_model.F);
    }
  }

  g_min_len = VRNA_GPU_MODEL_LMAX;
  for (L = 1; L < VRNA_GPU_MODEL_LMAX; L++) {
    const double c = (double)L * (L + 1.0) / 2.0;

    if (gpu_model_host_s(c) / gpu_model_speedup(g_host_jobs) > gpu_model_dev_s(c)) {
      g_min_len = L;
      break;
    }
  }

  g_short_n = 0;
}


/* Does this chunk earn the device? The model, over the records it holds. Asked once
 * per chunk and only once the device is reached, so a chunk that earns it is counted
 * here: the next chunk's fixed cost is the per-chunk one, C0 having been paid. */
static int
rnafold_chunk_earns_gpu(struct record_data **chunk, const int n)
{
  struct gpu_work w = { 0, 0.0, 0.0, 0.0 };
  int             i;

  for (i = 0; i < n; i++)
    if ((chunk[i]) && (chunk[i]->sequence))
      gpu_work_add(&w, (double)strlen(chunk[i]->sequence));

  if (!gpu_work_earns(&w))
    return 0;

  g_device_chunks++;
  return 1;
}

static int
rnafold_min_gpu_batch(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_MIN_GPU_BATCH");

    v = VRNA_MIN_GPU_BATCH;

    if (e && *e) {
      const long n = atol(e);

      if (n >= 1) {
        v = (int)n;
        fprintf(stderr,
                "%-24s RNA_MIN_GPU_BATCH=%d (default %d): chunks smaller than "
                "this fold on the CPU\n",
                "bin/RNAfold.c", v, VRNA_MIN_GPU_BATCH);
      } else {
        fprintf(stderr,
                "%-24s ignoring RNA_MIN_GPU_BATCH=%s (want a positive integer)\n",
                "bin/RNAfold.c", e);
      }
    }
  }

  return v;
}


/* Will process_record() have to construct a fold compound for an
 * already-folded record? ONE predicate with two users, because they must not
 * drift:
 *
 *   1. process_record() itself, to skip the construction (that skip is the whole
 *      of stage_output -- 124.8 s of a 669 s run before it landed);
 *   2. pipeline_flush(), which is only race-free while the answer is NO.
 *
 * (2) is the subtle one. The builder thread calls vrna_fold_compound() ->
 * vrna_params(), whose SPEEDUP_PARAMS cache is unsynchronised (Defect B). The
 * fold side is clean -- vrna_mfe_batch() reaches vrna_fold_compound_prepare()
 * with VRNA_OPTION_MFE, and vrna_params_prepare() only touches the already-built
 * fc->params on that path (params.c:428-441) -- but process_record() is
 * dispatched to the -j pool from inside the fold, and if IT builds compounds
 * they race the builder. So the pipeline declines whenever this returns true.
 *
 * Conservative in the same direction as process_record()'s own use: anything not
 * enumerated counts as needing a compound, which costs speed, not correctness.
 */
int
output_needs_compound(struct options *opt)
{
  return !(opt->noPS && !opt->pf && !opt->MEA && !opt->lucky &&
           !opt->verbose && !opt->benchmark &&
           !fold_constrained && !opt->constraint_file &&
           !opt->probing_data && !opt->ligandMotif &&
           !opt->cmds && !opt->mod_params);
}


/* A chunk between its two halves.
 *
 * flush_gpu_chunk() used to do both -- build every fold compound, then fold them
 * -- in one serial pass, which means the GPU sits idle for the whole build and
 * the cores sit idle for the whole fold. At 400 x 5601 that idle is 33.9% of
 * wall and `build` is 72% of it (docs/archive/STRESS272_RESULTS.md §14).
 *
 * Splitting it here is a PURE REFACTOR and changes nothing on its own; it exists
 * so the build of one chunk can later be overlapped with the fold of the
 * previous one. `built` is what separates a batch that owns compounds from one
 * that does not, so the teardown path can tell them apart.
 */
struct gpu_batch {
  struct record_data    **chunk;
  int                     n;
  vrna_fold_compound_t  **VC;
  char                  **Str;
  float                  *EN;
  int                     built;
  double                  t_fold0;  /* N5a: the fold window's start, for a handed-off batch */
};


/* How many threads build one chunk's fold compounds (option A).
 *
 * DEFAULT "auto" (nproc) SINCE 2026-09-12. It shipped serial as a new,
 * unproven knob, and `MERGING.md` has listed it under "Default ON" that whole
 * time -- the doc was describing the intent and the code never got there.
 *
 * WHAT CHANGED IS THE SIZE OF THE PRIZE, not the confidence. `build` was 24.5%
 * of wall on a T4. On an A100 the GPU phases are 3-5x faster and `build` does
 * not move at all -- 121.4 s of a 224.9 s wall, **54%**
 * (docs/archive/STRESS272_RESULTS.md 23.5). Measured here at 60 x 2400 on 12 cores:
 * build 1.684 -> 0.260 s, **6.47x**, byte-identical at 2/3/4/6/8/12/auto.
 *
 * Scaling flattens past ~4-6 threads, which is what an allocation- and
 * bandwidth-heavy O(n^2) table build should do; the remaining gain to 12 is
 * real but shallow. `auto` rather than a tuned constant because the right
 * number is a property of the HOST, and this project has been wrong before
 * about numbers that are (see the heterogeneous scope note).
 *
 * ON OVERSUBSCRIPTION WITH -j: the output pool and these builders can overlap
 * on the pipeline path, so a `-j nproc` run briefly has 2x nproc runnable
 * threads. Left alone deliberately -- the builders are short-lived and the
 * fold they overlap is GPU-bound, so the cost is context switches rather than
 * contention. Under a thread budget (B4) the builders and the backtrack workers
 * draw from one shared count, and -j's auto size is capped at the budget; at
 * `full` this paragraph holds. RNA_BACKTRACK_THREADS subtracts
 * `cpu_queue_threads` for the same reason, but that queue is retired on this
 * branch and is always 0 here.
 *
 * "0" or "1" still forces serial, which is how the A/B is run.
 *
 * SAFE ONLY BECAUSE Defect B IS FIXED. vrna_fold_compound() reaches
 * vrna_params(), whose SPEEDUP_PARAMS cache was shared mutable state with no
 * lock; params.c now guards it. Before that fix this knob was undefined
 * behaviour that happened to look fine whenever every record shared one model
 * -- measured 0 of 160 000 tables corrupted with identical model details,
 * 25 140 of 160 000 (15.7 %) when they differ. RNAfold always passes one md, so
 * the failure would never have shown up here and would have shown up in a
 * library caller doing a temperature sweep.
 */
static int g_build_threads_explicit;

static int
rnafold_build_threads(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_BUILD_THREADS");

    if ((!e) || (!e[0]) || (!strcmp(e, "auto"))) {
      long hw = sysconf(_SC_NPROCESSORS_ONLN);
      v = (hw > 1) ? (int)hw : 1;
    } else {
      v = atoi(e);
      if (v < 1)
        v = 1;

      g_build_threads_explicit = 1;
    }

    /* B4: the thread budget caps the auto size; RNA_BUILD_THREADS wins, warned */
    v = vrna_budget_cap_threads("build threads", v, g_build_threads_explicit);

    /* ANNOUNCE IT. Every other knob in this driver that changes what runs says
     * so on stderr, for one reason: a run that did not apply a setting must not
     * be able to pass for one that did. This one was silent for three days
     * while its default disagreed with MERGING.md, and no harness could have
     * caught that. The DEFAULT is said only under --verbose (PORT_SHORT_SEQUENCES.md
     * D2); an explicit RNA_BUILD_THREADS always is, which is the case that rule is for. */
    if ((e && e[0]) || vrna_cuda_verbose())
      fprintf(stderr,
              "%-24s build threads %d%s\n", "bin/RNAfold.c", v,
              (e && e[0]) ? " (from RNA_BUILD_THREADS)" : " (auto: nproc, the default)");
  }

  return v;
}


struct build_range {
  struct gpu_batch  *b;
  struct options    *opt;
  int                lo, hi;      /* [lo, hi) */
};


static void build_one(struct gpu_batch *b, struct options *opt, int i);


static void *
build_range_main(void *p)
{
  struct build_range *r = (struct build_range *)p;
  int                 i;

  for (i = r->lo; i < r->hi; i++)
    build_one(r->b, r->opt, i);

  return NULL;
}


/* Build every fold compound in the batch. Host-only; touches no device state,
 * which is what makes it safe to run beside another batch's GPU work. */
static void
build_gpu_batch(struct gpu_batch *b,
                struct options   *opt)
{
  int                   n     = b->n;
  int                   i;

  b->VC  = (vrna_fold_compound_t **)vrna_alloc(sizeof(void *) * n);
  b->Str = (char **)vrna_alloc(sizeof(char *) * n);
  b->EN  = (float *)vrna_alloc(sizeof(float) * n);

  /* stage_build_s was DECLARED and PRINTED but never incremented, so the stage
   * line reported build=0.000 for the life of the project -- which reads as
   * "this costs nothing" when it means "nobody measured it". The deep profile
   * on 2026-09-08 found 61% of wall at 120x5601 outside every timer, and this
   * is one of the four counters that should have been covering it.
   * vrna_fold_compound() builds the ptype and hard-constraint tables, which are
   * O(n^2) per record, so this is a genuine candidate for that remainder. */
  const double t_build = rnafold_now_seconds();
  const int    nthr    = rnafold_build_threads();

  int          held;
  const int    nwant = ((nthr <= 1) || (n < 2)) ? 1 : ((nthr < n) ? nthr : n);
  const int    ngot  = vrna_budget_threads_acquire(nwant, g_build_threads_explicit, &held);

  if (ngot <= 1) {
    for (i = 0; i < n; i++)
      build_one(b, opt, i);
  } else {
    /* Records are independent: each writes only VC[i] and Str[i], and the only
     * shared thing any of them touches is the parameter cache inside
     * vrna_params(), which params.c now locks. Static contiguous ranges rather
     * than a work queue -- the records in one chunk are close in length, so the
     * imbalance is small and a queue would add a second shared structure for no
     * measured gain. */
    const int  t = ngot;   /* B4: what the shared thread budget gave */
    pthread_t  *th = (pthread_t *)vrna_alloc(sizeof(pthread_t) * t);
    struct build_range *rg =
      (struct build_range *)vrna_alloc(sizeof(struct build_range) * t);
    int        k, started = 0;

    for (k = 0; k < t; k++) {
      rg[k].b   = b;
      rg[k].opt = opt;
      rg[k].lo  = (int)((long)n * k / t);
      rg[k].hi  = (int)((long)n * (k + 1) / t);

      if (pthread_create(&th[k], NULL, build_range_main, &rg[k]) == 0)
        started++;
      else
        break;              /* fall through: this range is built inline below */
    }

    for (k = started; k < t; k++)
      build_range_main(&rg[k]);

    for (k = 0; k < started; k++)
      pthread_join(th[k], NULL);

    free(th);
    free(rg);
  }

  vrna_budget_threads_release(held);

  /* WALL time, not summed worker time. Threading a phase once made its timer go
   * NEGATIVE on this project by subtracting worker-seconds from wall-seconds
   * (trap 8, step 2a); this stays a plain wall-clock span so it keeps meaning
   * the same thing whether or not the loop above threaded. */
  stage_build_s += rnafold_now_seconds() - t_build;
  b->built       = 1;
}


/* One record's fold compound. Split out so the serial and threaded paths cannot
 * drift -- they are the same code, called from two loops. */
static void
build_one(struct gpu_batch *b,
          struct options   *opt,
          int               i)
{
  struct record_data **chunk = b->chunk;

  /* THE SEQUENCE process_record() FOLDS, not the one that was read. Found 2026-10-04
   * by the binder notebook: RNAfold converts T -> U (unless --noconv) and upper-cases
   * before folding, and this compound was built from the raw input. Every term that
   * matches sequence strings -- the special hairpins (Tetraloops, Triloops, Hexaloops)
   * -- then saw 'T' where the CPU saw 'U', so DNA input folded on the GPU gave wrong
   * structures and energies in both directions: 39 of 200 records at 2000 nt, e.g.
   * -624.60 against the CPU's -625.00. RNA input never reached it, which is why every
   * parity bar (all RNA alphabet) passed. The upper-casing is process_record()'s too,
   * applied here so the two compounds are built from the same string; T -> U is the
   * part a test has shown to matter (tests/RNAfold/gpu_input). */
  char *seq = strdup(chunk[i]->sequence);

  if (!opt->noconv)
    vrna_seq_toRNA(seq);

  vrna_seq_toupper(seq);
  b->VC[i] = vrna_fold_compound(seq, &(opt->md), VRNA_OPTION_DEFAULT);
  free(seq);

  /* The chunk path builds its OWN fold compounds, so it has to apply every
   * per-record constraint that process_record() would apply to its own. Skip
   * this and the batch folds unconstrained and returns a plausible, wrong
   * structure -- the precise failure mode the routing guard exists to
   * prevent, arriving through the driver rather than through the device.
   *
   * process_record() still applies constraints to the compound it builds for
   * the partition function, MEA and the no-solution check; these two
   * compounds are separate objects, so nothing is constrained twice. */
  if (fold_constrained)
    apply_constraints(b->VC[i],
                      opt->constraint_file,
                      (const char **)chunk[i]->rest,
                      chunk[i]->multiline_input,
                      opt->constraint_enforce,
                      opt->constraint_canonical,
                      1 /* quiet: process_record() reports each record once */);

  /* COMMAND FILES, 2026-09-16. Same argument as the constraints above, and the
   * reason --commands was silently ignored rather than declined: a command file
   * can queue HARD constraints, SOFT constraints or unstructured domains, and
   * the routing guard decides on what it FINDS IN THE COMPOUND. Never applying
   * them here meant the guard inspected a compound that had none of them, saw
   * nothing to refuse, and folded the batch unconstrained -- the driver-side
   * version of the failure the guard exists to prevent.
   *
   * With them applied, gate 2 does the right thing per record without knowing
   * anything about command files: hard constraints land in hc->depot and are
   * ACCEPTED (they have been supported since -C shipped); soft constraints land
   * in fc->sc and unstructured domains in fc->domains_up, and both are declined
   * to upstream's own path. */
  if (opt->cmds)
    vrna_commands_apply(b->VC[i], opt->cmds, VRNA_CMD_PARSE_DEFAULTS);

  /* LIGAND MOTIFS, same argument again. vrna_sc_add_hi_motif() installs a SOFT
   * constraint (RNAfold.c:3153), so gate 2 declines it -- but only if the
   * compound it inspects actually carries one. Applied here so the decision is
   * made on what the fold compound IS rather than on which flag was typed,
   * which is what makes gate 2 the authority and gate 1 merely the
   * optimisation it is supposed to be.
   *
   * Gate 1 still refuses --motif, deliberately: a motif ALWAYS lands in sc and
   * is ALWAYS declined, so building a batch for it would be pure waste. This
   * exists for the library caller and for the probe. */
  if (opt->ligandMotif)
    add_ligand_motif(b->VC[i], opt->ligandMotif, 0 /* quiet */, VRNA_OPTION_MFE);

  b->Str[i] = (char *)vrna_alloc(sizeof(char) * (strlen(chunk[i]->sequence) + 1));
}


/* Hand a folded batch's answers back to its records, free its compounds and
 * dispatch the output. Called right after the fold, or -- for a batch whose
 * backtrack was handed to the trace thread (N5a, RNA_BT_PIPELINE) -- from
 * bt_chunk_done() when the backend joins that trace. Either way on the main
 * thread. */
static void
finish_gpu_batch(struct gpu_batch *b,
                 const double      trace_fold_a,
                 const double      trace_fold_b)
{
  struct record_data    **chunk = b->chunk;
  int                     n     = b->n;
  vrna_fold_compound_t  **VC    = b->VC;
  char                  **Str   = b->Str;
  float                  *EN    = b->EN;
  int                     i;

  {
    int ti;

    for (ti = 0; ti < n; ti++) {
      chunk[ti]->trace_fold_t0 = trace_fold_a;
      chunk[ti]->trace_fold_t1 = trace_fold_b;
    }
  }

  /* stage_free_s: also a dead counter until 2026-09-08. Freeing a fold
   * compound releases the same O(n^2) tables build allocated. */
  const double t_free = rnafold_now_seconds();
  for (i = 0; i < n; i++) {
    chunk[i]->prefolded           = 1;
    chunk[i]->prefolded_energy    = EN[i];
    chunk[i]->prefolded_structure = Str[i];   /* handed over; freed with the record */
    vrna_fold_compound_free(VC[i]);
  }
#ifdef __GLIBC__
  /* Give the chunk's tables back to the OS. glibc keeps freed memory in the
   * arena that served it, and these compounds were built on the builder thread
   * and are freed here, a chunk later under N5a: without the trim the process
   * held every chunk's tables (2000 x 1200, both pipelines: 6.3 GB peak and
   * swapping on a 6.7 GB host, against 4.4 GB with this trim and no swap; N5 P0,
   * 2026-10-14). The RAM budget reads MEASURED RSS, so the untrimmed heap also
   * made it shed workers and narrow chunks for memory nothing was using. */
  malloc_trim(0);
#endif
  stage_free_s += rnafold_now_seconds() - t_free;

  /* dispatch only once every record in the chunk has its answer.
   * stage_output_s covers process_record(): formatting, the PS/DP plots when
   * they are not suppressed, and everything the partition function does when
   * -p is set. It is the fourth dead counter, and on a --noPS MFE-only run it
   * should be small -- if it is NOT, that is the finding.
   *
   * CAVEAT, and it is the trap that once made a phase timer report a NEGATIVE
   * number here: RUN_IN_PARALLEL is thpool_add_work() whenever max_threads > 1
   * (parallel_helpers.h:63), so with -j this measures DISPATCH, not the work.
   * Single-threaded it calls fun(data) inline and the number is real. Read
   * stage_output_s only from a run without -j until someone joins the pool
   * inside the timed region. */
  const double t_output = rnafold_now_seconds();
  for (i = 0; i < n; i++)
    rnafold_dispatch_record(chunk[i]);
  stage_output_s += rnafold_now_seconds() - t_output;

  free(VC);
  free(Str);
  free(EN);
  b->VC = NULL; b->Str = NULL; b->EN = NULL; b->built = 0;
}


#ifdef VRNA_WITH_CUDA
/* N5a P2 (PORT_BACKTRACK_PIPELINE.md §3.3): may this batch's backtrack run
 * beside the next sweep? Its output is then dispatched one chunk later, so the
 * order must come from the ostream slots (not --unordered), and nothing on the
 * output path may build compounds of its own (the build pipeline's rule). */
static int
bt_pipeline_wanted(struct options *opt)
{
  static int said = 0;

  if (!rnafold_bt_pipeline())
    return 0;

  if ((!opt->keep_order) || (output_needs_compound(opt))) {
    if (!said) {
      said = 1;
      fprintf(stderr, "%-24s RNA_BT_PIPELINE: backtracks stay sequential -- %s\n",
              "bin/RNAfold.c", (!opt->keep_order)
              ? "--unordered output has no ostream to keep the order"
              : "the output path builds compounds of its own");
    }

    return 0;
  }

  return 1;
}


/* The backend joined a handed-off batch's trace: its answers are in. */
static void
bt_chunk_done(void *ctx)
{
  struct gpu_batch *ab = (struct gpu_batch *)ctx;

  finish_gpu_batch(ab, ab->t_fold0, trace_now());
  free(ab->chunk);
  free(ab);
}
#endif


/* Fold a built batch, release the device and finish it (finish_gpu_batch()).
 * Everything here that touches the GPU lives on ONE thread; build_gpu_batch()
 * is the half that may run beside it. */
static void
fold_gpu_batch(struct gpu_batch *b,
               struct options   *opt)
{
  int                     n     = b->n;
  vrna_fold_compound_t  **VC    = b->VC;
  char                  **Str   = b->Str;
  float                  *EN    = b->EN;

  /* THE SEAM. Not par_mfe(): the driver asks the LIBRARY to fold a batch, and
   * the library uses whatever backend is registered -- the CUDA one here, or
   * a plain loop over vrna_mfe() if none is. Nothing below this line, and
   * nothing in this function, is CUDA-specific.
   *
   * That is what makes the diff presentable: the accelerator is a backend, not
   * a fork of the driver, and removing it leaves a correct program. */
  /* RNA_RECORD_TRACE: the device window for every record in this chunk. A batch
   * has no per-record device window, so the chunk's window is the honest answer
   * here, and the reader treats a shared window as exactly that. */
  const double trace_fold_a = trace_now();

#ifdef VRNA_WITH_CUDA
  /* N5a P2: offer the backend a context it may finish later. It owns a copy of
   * the batch -- the caller reuses `b` and, on the unpipelined path, its chunk
   * array -- and comes back through bt_chunk_done() at the join. */
  struct gpu_batch *ab = NULL;

  if (bt_pipeline_wanted(opt)) {
    ab          = (struct gpu_batch *)vrna_alloc(sizeof(*ab));
    *ab         = *b;
    ab->chunk   = (struct record_data **)vrna_alloc(sizeof(void *) * n);
    memcpy(ab->chunk, b->chunk, sizeof(void *) * n);
    ab->t_fold0 = trace_fold_a;
    rnafold_bt_pipeline_ctx(ab);
  }
#else
  (void)opt;
#endif

  vrna_mfe_batch(VC, (size_t)n, Str, EN);

  const double trace_fold_b = trace_now();

  /* Release the device state this chunk sized, before the next chunk sizes its
   * own. Without this the second chunk inherits dirty buffers. A handed-off
   * chunk's backtrack buffers were retired out of the globals and are not
   * touched here. */
  const double t_teardown = rnafold_now_seconds();
  teardown_gpu();
  teardown_gpu2();
  teardown_gpu3();
  stage_teardown_s += rnafold_now_seconds() - t_teardown;

#ifdef VRNA_WITH_CUDA
  if (ab) {
    const int went = rnafold_bt_pipeline_went_async();

    rnafold_bt_pipeline_ctx(NULL);   /* nothing left to hand off */
    if (went) {
      /* the answers arrive at the join; `ab` owns the arrays now */
      b->VC = NULL; b->Str = NULL; b->EN = NULL; b->built = 0;
      return;
    }

    free(ab->chunk);
    free(ab);
  }
#endif

  finish_gpu_batch(b, trace_fold_a, trace_fold_b);
}


/* ===================== the build/fold pipeline (option B) =====================
 *
 * One chunk deep: while the GPU folds chunk N, ONE host thread builds chunk
 * N+1's fold compounds. Prize is (chunks-1)/chunks x build -- ~19.6% of wall at
 * 400 x 5601, and exactly ZERO on a single-chunk run, because there is nothing
 * to overlap with. docs/archive/PORT_HETEROGENEOUS_SCOPE.md option B.
 *
 * WHY ONE THREAD AND NOT A POOL. vrna_fold_compound() reaches vrna_params(),
 * whose SPEEDUP_PARAMS cache is four unsynchronised file-scope statics
 * (params.c:100-106, our Defect B). A pool of builders is a data race on it; a
 * single builder never calls it concurrently with itself. The FOLD side must
 * therefore never call vrna_params() either, or the race comes back through the
 * other door -- that is asserted below rather than assumed, because it is the
 * one property this whole design rests on.
 *
 * AUTO BY DEFAULT, and that default was earned rather than assumed.
 * docs/archive/STRESS272_RESULTS.md 32.3 measured it at scale on an A100 at 400 x 5601:
 *
 *   chunks | off       | on        | wall   | peak host RSS
 *   -------+-----------+-----------+--------+-------------------
 *     13   | 112.29 s  |  96.02 s  | -14.5% |  3.07 -> 5.05 GB
 *      7   | 100.87 s  |  86.62 s  | -14.1% |  4.56 -> 8.07 GB
 *      2   |  90.36 s  |  85.26 s  |  -5.6% | 13.80 -> 20.20 GB
 *
 * So the wall is always better and the host RAM is always worse -- and at the
 * FASTEST configuration, the fewest chunks, which is also what 30.7 says to
 * run, it asks for 20.2 GB to fold 400 x 5601. A 16 GB host that works today
 * would OOM on an unconditional flip. That is why the default is not one.
 *
 *   RNA_BUILD_PIPELINE unset -> AUTO: on when the chunk about to be built fits
 *                                     in half of MemAvailable, off otherwise.
 *   RNA_BUILD_PIPELINE=1     -> forced on, host memory NOT consulted.
 *   RNA_BUILD_PIPELINE=0     -> forced off.
 *
 * The prize also shrank while nobody was looking: the validated 16.8-21.4%
 * predates RNA_BUILD_THREADS being on, which cut `build` from ~121 s to 18.4 s.
 * At two chunks a one-deep pipeline can hide at most half of that, and it hides
 * 35% of it.
 */
#define RNAFOLD_PIPELINE_OFF   0
#define RNAFOLD_PIPELINE_ON    1
#define RNAFOLD_PIPELINE_AUTO  2

static int
rnafold_build_pipeline(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_BUILD_PIPELINE");

    if ((e == NULL) || (e[0] == '\0')) {
      v = RNAFOLD_PIPELINE_AUTO;
    } else if (e[0] == '0') {
      v = RNAFOLD_PIPELINE_OFF;
      fprintf(stderr, "%-24s RNA_BUILD_PIPELINE=0: builder thread disabled\n",
              "bin/RNAfold.c");
    } else {
      v = RNAFOLD_PIPELINE_ON;
      fprintf(stderr,
              "%-24s RNA_BUILD_PIPELINE=1: builder thread forced on, host "
              "memory NOT consulted\n", "bin/RNAfold.c");
    }
  }

  return v;
}


/* Host memory available to us right now, in bytes; 0 when it cannot be known.
 *
 * MemAvailable rather than MemFree, deliberately: MemFree excludes reclaimable
 * page cache, which on a box that has just read a 400-record FASTA is most of
 * what we would actually get back, so reading it would decline the pipeline on
 * precisely the runs that benefit. sysconf() is the fallback and IS the MemFree
 * answer -- conservative in the same direction.
 */
static size_t
rnafold_host_avail_bytes(void)
{
  FILE    *f;
  char     line[256];
  size_t   kb = 0;

  /* TEST HOOK. The decline branch is otherwise unreachable on any machine with
   * room to spare -- the VRAM budget splits the chunk long before the host runs
   * short -- so it would ship untested on every box that can run the workload
   * at all. RNA_HOST_AVAIL_MB pretends there is that much available and nothing
   * else changes. Announced, because a run that lied to itself about memory
   * must not be able to pass for one that did not. */
  {
    static int   announced = 0;
    const char  *e         = getenv("RNA_HOST_AVAIL_MB");

    if (e && e[0]) {
      const long mb = atol(e);

      if (mb >= 0) {
        if (!announced) {
          fprintf(stderr,
                  "%-24s RNA_HOST_AVAIL_MB=%ld: pretending this much host "
                  "memory is available (test hook)\n", "bin/RNAfold.c", mb);
          announced = 1;
        }

        return (size_t)mb * (size_t)1048576;
      }
    }
  }

  f = fopen("/proc/meminfo", "r");

  if (f) {
    while (fgets(line, sizeof(line), f)) {
      unsigned long v;

      if (sscanf(line, "MemAvailable: %lu kB", &v) == 1) {
        kb = (size_t)v;
        break;
      }
    }

    fclose(f);
  }

  if (kb)
    return kb * (size_t)1024;

#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
  {
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long psize = sysconf(_SC_PAGESIZE);

    if ((pages > 0) && (psize > 0))
      return (size_t)pages * (size_t)psize;
  }
#endif

  return 0;
}


/* Host bytes one built fold compound holds.
 *
 * MEASURED, not derived (STRESS272 35.3). Six shapes from 40 x 1200 to
 * 10 x 12000, each folded with the build pipeline off and on and the DIFFERENCE
 * in peak RSS divided by the records in a chunk -- which is exactly what the
 * pipeline's second chunk costs:
 *
 *     bytes/record = 0.727 * L^2 + 72.5 * L      (+/- 0.2% at L >= 2400)
 *
 * The structural upper bound -- dense (n+1)^2 hard constraints plus triangular
 * ptype -- is 1.5 L^2, and the first version of this function charged 2.0 L^2
 * on the strength of two RSS points at one length. That is 2.7x the truth at
 * production lengths, and with the bar at half of MemAvailable it declined the
 * pipeline on hosts where it would have fitted three times over.
 *
 * 1.0 L^2 + 128 L is used: the measured fit with ~37% of headroom on the
 * quadratic term, which keeps the estimate conservative without being wrong by
 * a factor. Still an over-estimate at every length measured, which is the
 * direction this has to err.
 */
static size_t
rnafold_compound_bytes(size_t len)
{
  const size_t L = len + 1;

  return (L * L) + (128 * L);
}


/* AUTO's decision for the chunk about to be handed to the builder.
 *
 * The pipeline's extra live set is exactly one chunk of compounds: the builder
 * fills chunk N+1 while chunk N is still folding. Half of MemAvailable is the
 * bar, because the other half has to cover the fold side, the ostream backlog
 * and whatever else the host is doing -- and being wrong here costs an OOM
 * kill, the one failure mode a SPEED default must not introduce.
 *
 * Announced on every CHANGE of verdict rather than once: a run whose chunks
 * differ in length can legitimately pipeline some and decline others, and a
 * silent switch is exactly what makes a timing unexplainable a week later.
 */
static int
pipeline_fits_in_host_memory(struct record_data **chunk,
                             int                  n)
{
  static int  last = -1;
  size_t      need = 0;
  size_t      avail;
  int         fits;
  int         i;

  for (i = 0; i < n; i++)
    need += rnafold_compound_bytes(chunk[i]->sequence ?
                                   strlen(chunk[i]->sequence) : 0);

  avail = rnafold_host_avail_bytes();

  /* An unknown is not a yes. */
  fits = (avail > 0) && (need <= avail / 2);

  if (fits != last) {
    if (vrna_cuda_verbose())   /* an AUTO verdict: said under --verbose (PORT_SHORT_SEQUENCES.md D2) */
      fprintf(stderr,
              "%-24s build pipeline AUTO: %s -- next chunk needs ~%.3f GB, "
              "MemAvailable %.3f GB (bar is half)\n",
              "bin/RNAfold.c", fits ? "ON" : "off",
              need / 1073741824.0, avail / 1073741824.0);
    last = fits;
  }

  return fits;
}


struct builder_arg {
  struct gpu_batch  *b;
  struct options    *opt;
  double             span;      /* builder wall time, for the overlap report */
};


static void *
builder_main(void *p)
{
  struct builder_arg *a  = (struct builder_arg *)p;
  const double        t0 = rnafold_now_seconds();

  build_gpu_batch(a->b, a->opt);
  a->span = rnafold_now_seconds() - t0;
  return NULL;
}


/* How much of the build actually ran BESIDE the GPU.
 *
 * Wall clock alone cannot answer that on a machine whose ceiling is ~4% -- it is
 * under run-to-run noise -- and `accounted - wall` only mirrors wall, so it is
 * not independent evidence. min(builder span, fold span) per chunk is: it is
 * bounded by both and is zero if either did not run. Reported once at the end so
 * a pipelined run says what it achieved rather than leaving it to be inferred. */
static double g_overlap_s   = 0.0;
static double g_builder_s   = 0.0;
static int    g_pipelined_n = 0;


/* The original entry point, now just the two halves back to back. Behaviour is
 * unchanged: the split exists so a caller that wants to overlap them can. */
static void
flush_gpu_chunk(struct record_data **chunk,
                int                  n,
                struct options      *opt)
{
  struct gpu_batch b;
  int              i;

  if (n <= 0)
    return;

  trace_open();

  if (!rnafold_chunk_earns_gpu(chunk, n)) {
    /* CPU fallback. Not a separate worker queue: upstream's driver already has
     * a per-record parallel path, so an undersized chunk simply goes down it.
     * That is the fork's RNAfold_cpu_queue.c retired rather than ported --
     * MERGING.md flagged it as largely redundant once upstream grew its own
     * thread pool and vrna_ostream_t, and this is where that pays off. The
     * records keep their ostream slots, so output order is unaffected. */
    for (i = 0; i < n; i++)
      rnafold_dispatch_record(chunk[i]);

    return;
  }

  memset(&b, 0, sizeof(b));
  b.chunk = chunk;
  b.n     = n;

  build_gpu_batch(&b, opt);
  fold_gpu_batch(&b, opt);
}


/* ================= CPU slice: fold some records on the cores (option C) ======
 *
 * The accelerated path uses ~ONE core of twelve (measured: 98% of a possible
 * 1200%, and 107% with A and B both on). Everything else is waiting on the
 * device. So hold `m` records back from each chunk and hand them to the -j pool
 * before folding the rest: the cores work through them while the GPU works
 * through the chunk. docs/archive/PORT_CPU_QUEUE_SCOPE.md.
 *
 * NO NEW FOLDING CODE. A record that never entered a chunk has prefolded == 0,
 * so process_record() folds it itself -- the same path an undersized chunk has
 * always taken. What is new is only the decision to send it there.
 *
 * REQUIRES -j. Without a pool RUN_IN_PARALLEL() runs inline, so the slice would
 * be folded SERIALLY BEFORE the GPU starts, which is strictly worse than not
 * slicing at all. m is forced to 0 in that case, and that is asserted rather
 * than assumed because the failure mode is a silent slowdown, not an error.
 *
 * SAFE ONLY BECAUSE Defect B IS FIXED: several pool threads now call
 * vrna_fold_compound() at once. Before params.c was locked this was UB.
 */
static int
rnafold_cpu_slice(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_CPU_SLICE");
    v = (e && e[0] && e[0] != '0') ? 1 : 0;
  }

  return v;
}


static pthread_mutex_t  g_slice_mtx  = PTHREAD_MUTEX_INITIALIZER;
static long             g_slice_out  = 0;   /* dispatched but not yet finished */
static long             g_slice_done = 0;   /* finished, for the report */
static int              g_slice_m    = 0;   /* current slice size */


/* Called by process_record() when a sliced record is finished. */
void
cpu_slice_finished(void)
{
  pthread_mutex_lock(&g_slice_mtx);
  g_slice_out--;
  g_slice_done++;
  pthread_mutex_unlock(&g_slice_mtx);
}


static long
cpu_slice_outstanding(void)
{
  long v;

  pthread_mutex_lock(&g_slice_mtx);
  v = g_slice_out;
  pthread_mutex_unlock(&g_slice_mtx);

  return v;
}


/* Move the slice out of `chunk` and onto the pool. Returns how many records are
 * left for the GPU, compacted to the front.
 *
 * WHICH records: the SHORTEST ones. CPU fold cost goes as ~n^3 while the slice's
 * whole job is to finish before the GPU does, so a single long record in the
 * slice is the tail-imbalance failure -- it can still be folding after the device
 * has finished everything, which turns C from a disappointment into a
 * regression. Selection is a partial sort by length, not a shuffle.
 */
static int
cpu_slice_take(struct record_data **chunk,
               int                  n,
               struct options      *opt)
{
  int   m, i, k;
  long  behind;

  if ((!rnafold_cpu_slice()) || (opt->jobs <= 1) || (n < 2))
    return n;                   /* no pool: RUN_IN_PARALLEL would be inline */

  /* Adapt on OUTCOME, not on a model of the two throughputs: the ratio depends
   * on the card's clock state, which moves. If the pool cleared its last slice
   * before the device finished, it had spare capacity -- grow. If work is still
   * outstanding, it did not -- shrink by the backlog. */
  behind = cpu_slice_outstanding();

  /* Additive increase, multiplicative decrease. The first version subtracted the
   * backlog outright and OSCILLATED 0 -> jobs -> 0, averaging half the slice it
   * should have run and converging on 1. AIMD is stable for the same reason it
   * is everywhere else. */
  if (behind > 0) {
    g_slice_m /= 2;
  } else {
    const int step = (opt->jobs > 2) ? opt->jobs / 2 : 1;
    g_slice_m += step;
  }

  if (g_slice_m < 0)
    g_slice_m = 0;

  /* THE CAP, and it was set wrong the first time.
   *
   * Every record handed to the cores is a record the GPU does not get, and the
   * first version capped the slice at n/8 to protect BATCH WIDTH. That reasoning
   * came from MIN_GPU_BATCH and the chunking work -- but the stress runs
   * measured the opposite at scale: going from 5 chunks to 14 (2.8x NARROWER
   * batches) cost 0.6% of wall. Batch width is nearly free to lose once the
   * device is saturated, so n/8 was throttling the slice for a cost that is not
   * there, and the measured result was a ~4% REGRESSION with only 5% of records
   * offloaded.
   *
   * The real constraint is the TAIL: a slice still folding after the device has
   * finished everything sets the wall. That is what the outcome feedback above
   * is for, so the cap only has to keep the GPU above MIN_GPU_BATCH and leave
   * the controller room to find the balance.
   *
   * RNA_CPU_SLICE_CAP is the maximum percent of a chunk, so the trade can be
   * measured rather than argued. */
  {
    int         cap_pct = 33;
    const char *e       = getenv("RNA_CPU_SLICE_CAP");
    const int   floor_n = rnafold_min_gpu_batch();
    int         width_cap;

    if ((e) && (e[0])) {
      cap_pct = atoi(e);
      if (cap_pct < 0)   cap_pct = 0;
      if (cap_pct > 90)  cap_pct = 90;
    }

    width_cap = (int)((long)n * cap_pct / 100);

    if (g_slice_m > width_cap)
      g_slice_m = width_cap;

    if (n - g_slice_m < floor_n)
      g_slice_m = (n > floor_n) ? (n - floor_n) : 0;
  }

  m = g_slice_m;

  if (m <= 0)
    return n;

  /* Partial selection sort: pull the m shortest to the end of the array. */
  for (k = 0; k < m; k++) {
    int  best = 0;
    size_t best_len = (size_t)-1;

    for (i = 0; i < n - k; i++) {
      size_t l = strlen(chunk[i]->sequence);

      if (l < best_len) {
        best_len = l;
        best     = i;
      }
    }

    {
      struct record_data *t = chunk[best];
      chunk[best]  = chunk[n - k - 1];
      chunk[n - k - 1] = t;
    }
  }

  pthread_mutex_lock(&g_slice_mtx);
  g_slice_out += m;
  pthread_mutex_unlock(&g_slice_mtx);

  for (i = n - m; i < n; i++) {
    chunk[i]->cpu_slice = 1;
    rnafold_dispatch_record(chunk[i]);
  }

  return n - m;
}


/* One-deep pipeline state. Static because main()'s three flush sites share it. */
static struct gpu_batch *g_pending = NULL;


/* Fold and release whatever is waiting. Safe to call when nothing is. */
static void
pipeline_drain(struct options *opt)
{
  if (g_pending) {
    struct gpu_batch *b = g_pending;

    g_pending = NULL;          /* clear FIRST: fold_gpu_batch() dispatches output */
    fold_gpu_batch(b, opt);
    free(b->chunk);
    free(b);
  }
}


/* RAM budget (PORT_BUDGETS.md B3): a chunk's compounds by the fitted model, and its
 * longest record. */
static size_t
chunk_host_of(struct record_data  **chunk,
              int                 n,
              size_t              *lmax)
{
  size_t  b = 0, m = 0;
  int     i;

  for (i = 0; i < n; i++) {
    const size_t l = chunk[i]->sequence ? strlen(chunk[i]->sequence) : 0;

    b += vrna_budget_compound_bytes(l);
    if (l > m)
      m = l;
  }

  if (lmax)
    *lmax = m;

  return b;
}


/* Does a build pipeline fit the RAM budget? It holds the NEXT chunk while the pending
 * one folds and backtracks (at one worker at least), so both sets of compounds are
 * resident. Given up after the workers and before the chunk width (decision 2); an
 * explicit RNA_BUILD_PIPELINE=1 wins, with a warning (decision 1). */
static int
budget_fits_build_pipeline(struct record_data  **chunk,
                           int                 n,
                           struct options      *opt,
                           int                 explicit_on)
{
  const size_t  ram = vrna_budget_ram();
  size_t        l1 = 0, l2 = 0, need;
  static int    said = -1;
  int           fits;

  if (ram == SIZE_MAX)
    return 1;

  /* measured (RSS, which already holds the pending chunk's compounds) + the next chunk's */
  need = vrna_budget_rss() + chunk_host_of(chunk, n, &l1);
  if (g_pending)
    (void)chunk_host_of(g_pending->chunk, g_pending->n, &l2);

  need += vrna_budget_worker_bytes((l1 > l2) ? l1 : l2, opt->md.circ);
  fits  = (need <= ram);

  if (fits != said) {
    said = fits;
    if ((!fits) && explicit_on)
      fprintf(stderr, "%-24s WARNING: RNA_BUILD_PIPELINE=1 needs ~%.0f MB, over the RAM "
                      "budget's %.0f MB; the explicit setting wins\n", "bin/gpu_batch.c",
              (double)need / 1048576.0, (double)ram / 1048576.0);
    else if (!fits && !vrna_cuda_quiet())
      fprintf(stderr, "%-24s RAM budget: build pipeline off (two chunks need ~%.0f MB of "
                      "%.0f MB)\n", "bin/gpu_batch.c", (double)need / 1048576.0,
              (double)ram / 1048576.0);
  }

  return fits;
}


/* Pipelined replacement for flush_gpu_chunk(). Identical behaviour whenever the
 * pipeline is declined -- by RNA_BUILD_PIPELINE=0, by AUTO finding too little
 * host memory, or by output_needs_compound(). */
static void
pipeline_flush(struct record_data **chunk,
               int                  n,
               struct options      *opt)
{
  struct gpu_batch   *b;
  struct builder_arg  a;
  pthread_t           th;
  int                 i;

  if (n <= 0)
    return;

  /* Hand part of the chunk to the cores first, so they are already working by
   * the time the device starts. Returns the number of records still bound for
   * the GPU, with those records compacted to the front of `chunk`. */
  n = cpu_slice_take(chunk, n, opt);

  if (n <= 0)
    return;                     /* the whole chunk went to the cores */

  /* Declined when the output path builds compounds of its own: those run on the
   * -j pool from inside the fold and would race the builder thread through
   * vrna_params(). Serial is slower, not wrong. */
  {
    const int mode = rnafold_build_pipeline();
    const int fits = budget_fits_build_pipeline(chunk, n, opt, mode == RNAFOLD_PIPELINE_ON);
    const int want = (mode == RNAFOLD_PIPELINE_ON) ||
                     ((mode == RNAFOLD_PIPELINE_AUTO) && fits &&
                      pipeline_fits_in_host_memory(chunk, n));

    if ((!want) || (output_needs_compound(opt))) {
      flush_gpu_chunk(chunk, n, opt);
      return;
    }
  }

  if (!rnafold_chunk_earns_gpu(chunk, n)) {
    /* Too small for the device, exactly as flush_gpu_chunk() decides. Dispatch
     * down the per-record path; ordering is by ostream slot (requested at READ
     * time, main loop), not by dispatch order, so this may precede a pending
     * chunk's output without reordering the file. */
    for (i = 0; i < n; i++)
      rnafold_dispatch_record(chunk[i]);

    return;
  }

  b        = (struct gpu_batch *)vrna_alloc(sizeof(*b));
  memset(b, 0, sizeof(*b));
  b->n     = n;
  /* The caller REUSES its gpu_chunk[] array as soon as this returns, so the
   * batch has to own a copy of the pointers -- not the records, which outlive
   * it either way. */
  b->chunk = (struct record_data **)vrna_alloc(sizeof(void *) * n);
  memcpy(b->chunk, chunk, sizeof(void *) * n);

  a.b   = b;
  a.opt = opt;

  /* RAM budget: while the builder holds `b`, the folding chunk's backtrack workers are
   * sized beside it (vrna_budget_workers). */
  vrna_budget_held(VRNA_BUDGET_HELD_NEXT_CHUNK, chunk_host_of(b->chunk, n, NULL));

  if (pthread_create(&th, NULL, builder_main, &a) != 0) {
    vrna_budget_held(VRNA_BUDGET_HELD_NEXT_CHUNK, 0);
    /* No thread available: degrade to the serial shape rather than fail. */
    build_gpu_batch(b, opt);
    pipeline_drain(opt);
    g_pending = b;
    return;
  }

  /* THE OVERLAP. The previous chunk folds on the GPU while `b` builds on the
   * builder thread. Note the device is IDLE again by the time this returns --
   * fold_gpu_batch() tears it down -- which is what keeps main()'s
   * compute_gpu_usable_bytes() query correct without any change there. */
  {
    const double t_fold = rnafold_now_seconds();
    double       fold_span;

    pipeline_drain(opt);
    fold_span = rnafold_now_seconds() - t_fold;

    pthread_join(th, NULL);   /* `a` is a stack local; joining keeps it alive */
    vrna_budget_held(VRNA_BUDGET_HELD_NEXT_CHUNK, 0);

    /* Zero on the first chunk, where pipeline_drain() had nothing to fold --
     * which is exactly the (chunks-1)/chunks the prize is bounded by. */
    g_overlap_s += (a.span < fold_span) ? a.span : fold_span;
    g_builder_s += a.span;
    g_pipelined_n++;
  }

  g_pending = b;
}


/* ===================== the accumulator: process_input()'s locals ===========
 * Reset by gpu_batch_begin() for every input stream, as the locals were. */
static struct record_data **gpu_chunk;
static int                   gpu_chunk_n;
static unsigned long long    gpu_chunk_cells;   /* admission cap, see VRNA_GPU_CHUNK_CELLS */
static int                   gpu_chunk_cap;   /* allocated slots in gpu_chunk[] */
static int                   gpu_enabled;
static int                   gpu_hard_cap;   /* RNA_GPU_CHUNK, testing override */
static unsigned int          gpu_chunk_len;
static size_t                chunk_usable_bytes;
static int                   chunk_started;
static size_t                chunk_bytes;
static size_t                chunk_host;        /* RAM budget (B3): the chunk's compounds */
static size_t                chunk_host_lmax;   /* and its longest record */

/* THE DEVICE IS REACHED LAZILY (PORT_SHORT_SEQUENCES.md S2, 2026-10-18).
 *
 * W. B. Langdon: one short sequence typed at the prompt got "fml_j AUTO: ..." and
 * no answer until Ctrl-D, and a short input paid the GPU's wake-up for a fold the
 * CPU finishes first. Measured here: ~100 ms per invocation with RNA_GPU=0 against
 * ~290-360 ms by default, for a 24 nt record that then folded on the CPU anyway.
 * The probe that creates the CUDA context ran in gpu_batch_begin(), before a single
 * record had been read.
 *
 * Now nothing touches CUDA until the input holds enough work to earn the device.
 * Until then records are HELD here, host side, and an input that ends first is
 * folded on the per-record path having never created a context. Interactive input
 * (S1) is never held at all: a terminal user is owed each answer before the next
 * prompt, so a record either earns the device on its own and is folded at once,
 * alone, or goes to the per-record path as stock ViennaRNA does. */
#define GPU_MODE_OFF      0   /* settled on the CPU: RNA_GPU=0, no device, or declined */
#define GPU_MODE_PENDING  1   /* may use the device; not reached yet */
#define GPU_MODE_DECLINED 2   /* options the sweep cannot run; not reached yet */
#define GPU_MODE_ON       3

static int                   gpu_mode;
static int                   gpu_interactive;   /* stdin is a terminal */
static const char           *gpu_why;           /* GPU_MODE_DECLINED: the option */
static struct record_data  **gpu_hold;          /* PENDING: read, not yet dispatched */
static int                   gpu_hold_n;
static int                   gpu_hold_cap;
static struct gpu_work       gpu_hold_work;
static struct gpu_work       gpu_shadow_work;   /* DECLINED: what the device was missing */


void
gpu_batch_begin(struct options *opt,
                int            interactive)
{
  gpu_chunk = NULL;
  gpu_chunk_n = 0;
  gpu_chunk_cells = 0;
  gpu_chunk_cap = 0;
  gpu_enabled = 0;
  gpu_hard_cap = 0;
  gpu_chunk_len = 0;
  chunk_usable_bytes = 0;
  chunk_started = 0;
  chunk_bytes = 0;
  chunk_host = 0;
  chunk_host_lmax = 0;
  gpu_mode = GPU_MODE_OFF;
  gpu_interactive = interactive;
  gpu_why = NULL;
  gpu_hold = NULL;
  gpu_hold_n = 0;
  gpu_hold_cap = 0;
  memset(&gpu_hold_work, 0, sizeof(gpu_hold_work));
  memset(&gpu_shadow_work, 0, sizeof(gpu_shadow_work));
  vrna_cuda_set_verbose(opt->verbose);

  /* The GPU chunk accumulator: a THIRD dispatch path alongside the serial and
   * per-record-parallel ones. Records are held back until a chunk's worth have
   * been read, folded together by par_mfe(), then dispatched normally with
   * their results attached.
   *
   * Ordering is unaffected: vrna_ostream_request() is still called at READ
   * time, in input order, so the output queue is unchanged no matter when a
   * record is actually dispatched. That is what lets the byte-identical bar
   * compare this path against the per-record path directly.
   *
   * Deliberately simple for now -- a fixed chunk size, uniform lengths, no
   * VRAM budgeting and no CPU-queue fallback. The chunker, the mixed-length
   * join mask and the slot machinery are all still in mfe/cuda and can be
   * reconnected once this path is proven correct. */

  /* The VRAM budget, replacing the fixed record count.
   *
   * chunk_usable_bytes is queried ONCE PER CHUNK, after the previous chunk's
   * teardown, so it sees genuinely free VRAM rather than counting the last
   * chunk's still-resident buffers as unavailable. chunk_bytes is what the
   * accepted records cost: one buffer set per record, gpu_bytes_per_file() each. */

  {
    /* GATE 3 IS GONE (2026-09-27). It used to read
     *
     *   gpu_enabled = (vrna_cuda_devices() > 0) && (e) && (e[0]);
     *
     * where e is getenv("RNA_GPU_CHUNK") -- so the accelerator engaged only if an
     * environment variable was SET TO SOMETHING. Nothing in upstream ViennaRNA's
     * documented install and use (./configure && make && sudo make install, then
     * RNAfold < in.fa) sets it, so every ordinary user got the CPU path, silently:
     * measured 2026-09-27, unset produced zero sweeps and ZERO BYTES of stderr.
     *
     * It was never a policy. RNA_GPU_CHUNK is a chunk-WIDTH override that acquired
     * presence-as-enable as an undocumented second job, which left the flag in the
     * absurd position that its "off" state (unset) and its most useful state ("0",
     * meaning no width cap) were different values with opposite effects.
     *
     * Now: if the library was built with a backend and a device is present and this
     * run's options are supported, the device is used. RNA_GPU_CHUNK keeps its real
     * job and nothing else. See PORT_ZEROCONF_SCOPE.md section 11.
     */
    const char *e   = getenv("RNA_GPU_CHUNK");
    const char *off = getenv("RNA_GPU");
    const char *why = NULL;

    g_host_jobs  = (opt->jobs > 0) ? opt->jobs : 1;
    gpu_model_begin();   /* the minimum length moves with -j */
    gpu_hard_cap = (e && e[0]) ? atoi(e) : 0;   /* 0 or less: budget decides */
    if (gpu_hard_cap < 0)
      gpu_hard_cap = 0;

    /* THE EXPLICIT OFF SWITCH. Deleting gate 3 removed the only way to ask for the
     * CPU path at run time, and two different people need one:
     *
     *   a USER who wants stock behaviour from an accelerated build -- to compare, to
     *   work around something, or because the machine's GPU is busy; and
     *
     *   every VERIFICATION HARNESS in tools/, whose CPU reference WAS gate 3. Those
     *   harnesses produce the accelerator's correctness bar by folding the same
     *   binary twice and demanding identical output, and the reference arm was
     *   simply "run without RNA_GPU_CHUNK set". With gate 3 gone that arm would
     *   quietly become a second GPU run, and the bar would compare the GPU against
     *   itself -- a check that can no longer fail. So this is not a convenience; it
     *   is what keeps those bars meaningful, and the harnesses now say RNA_GPU=0
     *   where they used to say nothing.
     *
     * It is announced like any other decision, so a run that used the CPU cannot be
     * mistaken for one that used the device, or the reverse. Since S2 it never
     * touches CUDA at all: RNA_GPU=0 is now exactly stock ViennaRNA's cost. */
    if ((off) && (off[0]) && (!strcmp(off, "0"))) {
      gpu_mode = GPU_MODE_OFF;
      fprintf(stderr, "%-24s GPU acceleration OFF: RNA_GPU=0 -- folding on the CPU\n",
              "bin/RNAfold.c");
    } else if (!gpu_path_usable(opt, &why)) {
      gpu_mode = GPU_MODE_DECLINED;   /* reached, and said, only if it would have mattered */
      gpu_why  = why;
    } else {
      gpu_mode = GPU_MODE_PENDING;
    }
  }
}


/* Reach the device: the probe, the announcement and the backend. Was the body of
 * gpu_batch_begin() until S2 made it lazy; called at most once per input stream, by
 * gpu_batch_offer(), when the input first holds enough work to earn the device. A
 * DECLINED run reaches it only to say what the unused device was missing. Settles
 * gpu_mode, and returns whether the device is now in use. */
static int
gpu_activate(struct options *opt)
{
  const char *why = gpu_why;
  int         ndev;

  {
    /* F, MEASURED. cudaGetDeviceCount() is what brings up the driver and creates
     * the context, so this call IS the cost of reaching the device, and it is the
     * dominant term in the admission floor -- ~0.6 s here against 0.042 s for
     * init_gpu() proper. Timing it costs nothing because the call has to happen
     * anyway, and it makes the floor respond to a cold driver, a shared cluster
     * GPU or a MIG slice instead of asserting a laptop's number. */
    {
      const double t_probe = trace_now();

      ndev             = vrna_cuda_devices();
      g_gpu_reach_cost = trace_now() - t_probe;
      gpu_enabled      = (ndev > 0) && (gpu_mode == GPU_MODE_PENDING);
    }

    /* REGISTER BEFORE ANNOUNCING (C10). The backend can refuse -- a build whose host and
     * device code disagree on a struct layout -- and an ON line printed first would then
     * be a false claim. One of only TWO places this driver names CUDA -- the other is
     * the vrna_cuda_devices() probe above. After this call the driver asks for batches
     * through vrna_mfe_batch() and the library decides; every other line of the chunking
     * machinery is backend-agnostic, which is the claim that actually matters and the one
     * MERGING.md repeats. CIRCULAR: the VRAM model must know BEFORE the first chunk is
     * sized, because fM2_real is a second full triangle per record. */
    if (gpu_enabled) {
      rnafold_circ_expect(opt->md.circ);
      if (!vrna_cuda_register_batch_backend()) {
        gpu_enabled = 0;
        why         = "the CUDA backend refused to register (see the line above)";
      }
    }

    if ((!gpu_enabled) && (!why))
      why = "no CUDA device is visible";

    /* ANNOUNCE THE DECISION -- but only when there is a decision to announce.
     *
     * It used to be behind --verbose, on the reasoning that a user who asked for
     * --gquad wants their answer rather than a lecture. That is wrong in one
     * direction: a user who silently gets 1x instead of 32x cannot tell, and neither
     * can a harness -- which is how this project spent 50 minutes of A100 time on a
     * build that could not have used the device.
     *
     * But it would be equally wrong to shout on every run of a CUDA-enabled package
     * installed on a machine with no GPU. That is the bioconda case, it is the
     * COMMON case, and there stock ViennaRNA says nothing and so must we. So:
     *
     *   engaged                 -> one line. Positive evidence a silent fallback
     *                              cannot fake, which is what the harnesses need.
     *   device present, declined-> one line with the reason. This is the dangerous
     *                              case: the hardware is right there and unused.
     *   no device at all        -> silent unless --verbose. Nothing surprising
     *                              happened; there was nothing to decline.
     */
    /* Which GPU, by name (Bill Langdon, 2026-10-19): device 0, the one the backend
     * uses. Asked only once a device answered, so a machine without one pays nothing. */
    const char *dev_name = (ndev > 0) ? vrna_cuda_device_name() : NULL;

    if (!dev_name)
      dev_name = "the GPU";

    if (gpu_enabled) {
      g_device_reached = 1;
      if (rnafold_gpu_work_floor() >= 0.0)
        fprintf(stderr, "%-24s GPU acceleration ON: %s (device 0 of %d); reaching the "
                        "device measured %.2f s; a chunk needs %.3g matrix cells (forced)\n",
                "bin/RNAfold.c", dev_name, ndev, g_gpu_reach_cost, rnafold_gpu_work_floor());
      else
        fprintf(stderr, "%-24s GPU acceleration ON: %s (device 0 of %d); reaching the "
                        "device measured %.2f s (budgeted %.2f s, %s); with %d job%s, "
                        "records under %d nt fold on the host\n", "bin/RNAfold.c", dev_name,
                ndev, g_gpu_reach_cost, g_model.F, g_F_source,
                g_host_jobs, (g_host_jobs == 1) ? "" : "s", g_min_len);
    } else if (ndev > 0) {
      /* A device is present and unused: the hardware is sitting there. (RNA_GPU=0,
       * the user asking for the CPU, is said in gpu_batch_begin() and never gets
       * here.) */
      fprintf(stderr, "%-24s GPU acceleration OFF: %s -- folding on the CPU; %s is "
                      "unused\n", "bin/RNAfold.c", why ? why : "unavailable", dev_name);
    } else if (opt->verbose) {
      vrna_log_info("no CUDA device visible; folding on the CPU path");
    }

    if (gpu_enabled) {
      /* N5a: RNA_BT_PIPELINE=1 lets a chunk's backtrack run beside the next
       * sweep; its answers come back through bt_chunk_done(). */
      if (rnafold_bt_pipeline())
        rnafold_bt_pipeline_attach(&bt_chunk_done);
    }
  }

  gpu_mode = gpu_enabled ? GPU_MODE_ON : GPU_MODE_OFF;
  return gpu_enabled;
}


/* GPU_BATCH_TAKEN: the record is in the chunk. GPU_BATCH_NOT_TAKEN: the GPU path is
 * off for this run. GPU_BATCH_TOO_BIG: free VRAM cannot hold even this one record;
 * the caller folds it on the pool and skips the rest of its loop body, exactly as
 * the `continue` here used to. */
static int
offer_into_chunk(struct record_data *record,
                 struct options     *opt)
{
  {
      /* A chunk must be UNIFORM LENGTH.
       *
       * The sweep shares one triangular layout across the batch, sized from the
       * chunk, and the mixed-length join mask that would relax that lives in
       * the chunker which is not reconnected yet. Feeding it a mixed chunk
       * reaches the device and fails there -- observed as
       *   Assertion `ij>=0 && (size_t)ij < tri_off_H[H+1]-tri_off_H[H]' failed
       *   Assertion `my_c[tri_off_H[H]+ij] == INF' failed
       *   CUDA error: an illegal memory access was encountered
       * on the mixed-length reference inputs. Flushing on a length change is
       * what the 2.3.0 driver did for the same reason, and it keeps this simple
       * path honest until the real chunker is wired back in.
       *
       * Note those were device-side ASSERTS, not silent corruption: this build
       * has no NDEBUG, and they turned an out-of-bounds read into a precise
       * diagnostic. Worth remembering when NDEBUG becomes the release default
       * (plan item I1) that the verification build must keep them. */
      const unsigned int this_len = (unsigned int)strlen(record->sequence);

      /* RNA_GPU_UNIFORM_CHUNKS=1 restores flushing on a length change.
       *
       * Kept as an A/B switch rather than deleted, because the reason mixed
       * chunks first failed here turned out to be the MISSING PER-CHUNK
       * TEARDOWN, not mixed lengths: the 2.3.0 driver dropped its
       * `vc->length != chunk_length` clause deliberately, so records of
       * different lengths accumulate into one batch and the sweep's join mask
       * handles the short ones. Its only flush trigger is the VRAM budget.
       * Having the switch means the claim can be re-tested rather than
       * believed. */
      size_t bytes_projected;
      int    need_flush;

      if ((gpu_chunk_n > 0) && (this_len != gpu_chunk_len) &&
          (getenv("RNA_GPU_UNIFORM_CHUNKS"))) {
        pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
        gpu_chunk_n = 0;
        chunk_bytes = 0;
        chunk_host = chunk_host_lmax = 0;
        gpu_chunk_cells = 0;
        chunk_started = 0;
      }

      /* Would this record still fit the chunk's VRAM budget? */
      bytes_projected = chunk_bytes + gpu_bytes_per_file((int)this_len);

      need_flush = (!chunk_started) || (bytes_projected > chunk_usable_bytes);

      if ((gpu_hard_cap > 0) && (gpu_chunk_n >= gpu_hard_cap))
        need_flush = 1;

      /* ADMISSION CAP. Flush before this record if it would push the chunk past
       * the cell budget -- but never flush an empty chunk, or a single record
       * larger than the cap could never be folded at all. */
      {
        const unsigned long long cap = rnafold_gpu_chunk_cells();
        const unsigned long long c   = (unsigned long long)this_len *
                                       ((unsigned long long)this_len + 1ull) / 2ull;

        if ((cap > 0) && (gpu_chunk_n > 0) && (gpu_chunk_cells + c > cap))
          need_flush = 1;
      }

      /* RAM BUDGET (PORT_BUDGETS.md B3): chunk width is shed LAST (decision 2), so a record
       * joins while the chunk fits at ONE backtrack worker and with no pipeline -- the
       * workers and the pipelines are given up first, where they are decided. */
      {
        const size_t ram = vrna_budget_ram();

        if ((ram != SIZE_MAX) && (gpu_chunk_n > 0)) {
          const size_t lmax = (this_len > chunk_host_lmax) ? this_len : chunk_host_lmax;
          /* measured (RSS) + this chunk's compounds, not yet built + one worker */
          const size_t need = vrna_budget_rss() + chunk_host +
                              vrna_budget_compound_bytes(this_len) +
                              vrna_budget_worker_bytes(lmax, opt->md.circ);
          static int   said = 0;

          if (need > ram) {
            need_flush = 1;
            if ((!said) && (!vrna_cuda_quiet())) {
              said = 1;
              fprintf(stderr, "%-24s RAM budget: chunks narrowed to fit %.0f MB (at one "
                              "backtrack worker and no pipeline)\n", "bin/gpu_batch.c",
                      (double)ram / 1048576.0);
            }
          }
        }
      }

      if (need_flush) {
        pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
        gpu_chunk_n = 0;
        chunk_bytes = 0;
        chunk_host = chunk_host_lmax = 0;
        gpu_chunk_cells = 0;

        /* Query AFTER the flush: flush_gpu_chunk() tears the device state down,
         * so this sees free VRAM rather than counting the previous chunk's
         * buffers as unavailable. */
        rnafold_set_chunk_cells_cap(rnafold_gpu_chunk_cells());   /* before the first query: int16 AUTO */
        chunk_usable_bytes = compute_gpu_usable_bytes();
        /* RULE (b) AT THE DRIVER (PORT_BUDGETS.md 9.7): never narrow a chunk for the
         * backtrack pipeline. A trace handed off under P2 may still hold its retired set
         * on the device; size as if it were gone (the same 0.85 margin both VRAM limits
         * leave), and the engine joins it before the sweep if this chunk then does not
         * fit beside it (cuda_batch_cb). Measured before: 6 -> 9 chunks at 2800 x 40,
         * RNA_VRAM_BUDGET=1100. Nothing pending (pipeline off): no change. */
        chunk_usable_bytes += (size_t)((double)rnafold_bt_pending_device_bytes() * 0.85);
        chunk_started      = 1;

        /* Degraded case: free VRAM cannot hold even ONE record of this length, so
         * there is no batch to form and the record has to go down the per-record
         * path. There is no choice being made here -- it is a capacity fact.
         *
         * THIS USED TO DEMAND ROOM FOR VRNA_MIN_GPU_BATCH (ten) RECORDS, on the
         * reasoning that folding one chunk at a time would be slower than not using
         * the GPU. That reasoning was derived for SHORT records, where the per-chunk
         * fixed cost dominates, and it is false for long ones: gpu_crossover.sh
         * measures a single 1600 nt record at 2.69x. Worse, it silently excluded
         * exactly the sequences the device helps most -- on a 4 GB card, ten 8000 nt
         * records need ~5 GB, so no 8000 nt sequence could ever reach the device,
         * while a user with a bigger card saw it work. A gate whose behaviour turns
         * on the size of the card is not a gate anyone can reason about.
         *
         * The admission model above already answers "is this worth the device"; this only
         * answers "does it fit". */
        if (gpu_bytes_per_file((int)this_len) > chunk_usable_bytes) {
          /* Over the VRAM BUDGET (RNA_VRAM_BUDGET) but within what is free: warn, and
           * fold it ALONE on the device -- the budget sizes chunks, it does not exile a
           * record (PORT_BUDGETS.md). Sizing the chunk to exactly this record makes the
           * next record flush it. Only a record over the soft limit goes to the cores. */
          const size_t need = gpu_bytes_per_file((int)this_len);

          if (need > compute_gpu_usable_bytes_soft())
            return GPU_BATCH_TOO_BIG;   /* the caller dispatches it and moves on */

          rnafold_vram_over_budget(need, chunk_usable_bytes);
          chunk_usable_bytes = need;
        }
      }

      /* grow both arrays as needed -- the chunk size is decided by bytes now,
       * so it is not known in advance */
      if (gpu_chunk_n == gpu_chunk_cap) {
        gpu_chunk_cap = gpu_chunk_cap ? gpu_chunk_cap * 2 : 64;
        gpu_chunk     = (struct record_data **)vrna_realloc(gpu_chunk,
                                                            sizeof(void *) * gpu_chunk_cap);
      }
      chunk_bytes += gpu_bytes_per_file((int)this_len);
      chunk_host  += vrna_budget_compound_bytes(this_len);
      if (this_len > chunk_host_lmax)
        chunk_host_lmax = this_len;
      if ((gpu_chunk_n == 0) && (vrna_budget_ram() != SIZE_MAX)) {
        /* one record alone over the RAM budget: warn, and fold it anyway (decision) */
        const size_t need = vrna_budget_rss() + chunk_host +
                            vrna_budget_worker_bytes(this_len, opt->md.circ);
        static int   warned = 0;

        if ((need > vrna_budget_ram()) && (warned++ < 5))
          fprintf(stderr, "%-24s WARNING: a %u nt record needs ~%.0f MB of host memory, over "
                          "the RAM budget's %.0f MB; folding it anyway\n", "bin/gpu_batch.c",
                  this_len, (double)need / 1048576.0, (double)vrna_budget_ram() / 1048576.0);
      }

      gpu_chunk_len            = this_len;
      gpu_chunk[gpu_chunk_n++] = record;
      gpu_chunk_cells += (unsigned long long)this_len *
                         ((unsigned long long)this_len + 1ull) / 2ull;
  }

  return GPU_BATCH_TAKEN;
}


/* S1: fold what the chunk holds NOW, and wait for it, so an interactive user sees
 * the answer before the next prompt. The build and backtrack pipelines both leave
 * work pending by design; both are drained, and the next record starts a fresh
 * chunk (chunk_started = 0 re-reads free VRAM, as after any flush). */
static void
flush_chunk_now(struct options *opt)
{
  pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
  pipeline_drain(opt);
  rnafold_bt_pipeline_drain();
  gpu_chunk_n     = 0;
  chunk_bytes     = 0;
  chunk_host      = chunk_host_lmax = 0;
  gpu_chunk_cells = 0;
  chunk_started   = 0;
}


static void
hold_dispatch_all(void)
{
  int i;

  for (i = 0; i < gpu_hold_n; i++)
    rnafold_dispatch_record(gpu_hold[i]);

  gpu_hold_n = 0;
  memset(&gpu_hold_work, 0, sizeof(gpu_hold_work));
}


int
gpu_batch_offer(struct record_data *record,
                struct options     *opt)
{
  const double L = (double)strlen(record->sequence);

  if (gpu_mode == GPU_MODE_OFF)
    return GPU_BATCH_NOT_TAKEN;

  /* S3, D1: under the minimum length the device loses at any batch size, so the
   * record is never offered to it -- not held, not counted toward waking it. */
  if ((!gpu_interactive) && (!gpu_record_admissible(L))) {
    g_short_n++;
    return GPU_BATCH_NOT_TAKEN;
  }

  if (gpu_mode == GPU_MODE_DECLINED) {
    /* Never held: the sweep cannot run these options. The device is reached only
     * once the input is big enough that its absence mattered, so the one-line
     * warning appears exactly when it is worth reading -- and a short -d3 run pays
     * no wake-up just to be told. gpu_activate() settles the mode to OFF. */
    gpu_work_add(&gpu_shadow_work, L);
    if (gpu_work_earns(&gpu_shadow_work))
      (void)gpu_activate(opt);

    return GPU_BATCH_NOT_TAKEN;
  }

  /* S1: a terminal is never kept waiting. The record is its own chunk, so the model
   * is asked about it alone, at one job: that is also the minimum-length test. */
  if (gpu_interactive) {
    struct gpu_work w = { 0, 0.0, 0.0, 0.0 };
    int             r;

    gpu_work_add(&w, L);
    if (!gpu_work_earns(&w))
      return GPU_BATCH_NOT_TAKEN;   /* the per-record path, at once */

    if ((gpu_mode == GPU_MODE_PENDING) && (!gpu_activate(opt)))
      return GPU_BATCH_NOT_TAKEN;

    r = offer_into_chunk(record, opt);
    if (r == GPU_BATCH_TAKEN)
      flush_chunk_now(opt);

    return r;
  }

  if (gpu_mode == GPU_MODE_PENDING) {
    int i;

    if (gpu_hold_n == gpu_hold_cap) {
      gpu_hold_cap = gpu_hold_cap ? gpu_hold_cap * 2 : 64;
      gpu_hold     = (struct record_data **)vrna_realloc(gpu_hold,
                                                         sizeof(void *) * gpu_hold_cap);
    }

    gpu_hold[gpu_hold_n++] = record;
    gpu_work_add(&gpu_hold_work, L);

    if (!gpu_work_earns(&gpu_hold_work))
      return GPU_BATCH_TAKEN;       /* held; gpu_batch_end() dispatches it if the input stops */

    if (!gpu_activate(opt)) {
      hold_dispatch_all();          /* no device after all: every held record, at once */
      return GPU_BATCH_TAKEN;
    }

    /* Reached. The held records join chunks in input order, exactly as if they had
     * been offered now; ostream slots were requested at read time, so the output
     * order does not depend on any of this. */
    for (i = 0; i < gpu_hold_n; i++)
      if (offer_into_chunk(gpu_hold[i], opt) == GPU_BATCH_TOO_BIG)
        rnafold_dispatch_record(gpu_hold[i]);

    gpu_hold_n = 0;
    memset(&gpu_hold_work, 0, sizeof(gpu_hold_work));
    return GPU_BATCH_TAKEN;
  }

  return offer_into_chunk(record, opt);
}


void
gpu_batch_end(struct options *opt)
{
  /* whatever is left over at EOF. A short final chunk is the normal case, and
   * flush_gpu_chunk() sends it down the per-record path if it does not carry enough
   * WORK to earn the device -- the tail-case fallback the 2.3.0 driver had, expressed
   * once rather than at each call site. It was a record count until 2026-09-27; the
   * tail of a long-sequence input is now correctly kept on the device, which is where
   * a handful of 5601 nt records belongs. */

  /* S2: the input ended before it held enough work to earn the device. Everything
   * held goes down the per-record path, and no CUDA context was ever created. */
  hold_dispatch_all();
  free(gpu_hold);
  gpu_hold     = NULL;
  gpu_hold_cap = 0;

  if ((g_short_n > 0) && (vrna_cuda_verbose()))
    fprintf(stderr, "%-24s %d record%s under %d nt folded on the host: too short to "
                    "earn the device at %d job%s\n", "bin/gpu_batch.c", g_short_n,
            (g_short_n == 1) ? "" : "s", g_min_len, g_host_jobs,
            (g_host_jobs == 1) ? "" : "s");

  if (gpu_enabled) {
    pipeline_flush(gpu_chunk, gpu_chunk_n, opt);
    /* The pipeline leaves the last chunk built but unfolded by construction --
     * there was no following chunk to overlap it with. Nothing else drains it,
     * and skipping this loses the tail silently rather than loudly. */
    pipeline_drain(opt);
    /* N5a: and the last handed-off backtrack, for the same reason. */
    rnafold_bt_pipeline_drain();

    if (g_pipelined_n > 0)
      fprintf(stderr,
              "%-24s build pipeline: %d chunks, builder %.3f s, "
              "OVERLAPPED %.3f s (%.0f%% of builder time hidden behind the GPU)\n",
              "RNAfold.c", g_pipelined_n, g_builder_s, g_overlap_s,
              (g_builder_s > 0.0) ? 100.0 * g_overlap_s / g_builder_s : 0.0);

    /* EVERY record the slice took, not only the finished ones (2026-10-16). This prints
     * before the pool is joined, so the last sliced records can still be folding: it
     * reported 3 where the record trace showed 4 routed to the cores, and the N5 bar
     * that counts device-traced + core-folded = all records caught it. */
    {
      long done, out;

      pthread_mutex_lock(&g_slice_mtx);
      done = g_slice_done;
      out  = g_slice_out;
      pthread_mutex_unlock(&g_slice_mtx);

      if ((rnafold_cpu_slice()) || (done + out > 0))
        fprintf(stderr,
                "%-24s cpu slice: %ld records folded on the cores (%ld still folding "
                "at this report), final slice size %d%s\n",
                "RNAfold.c", done + out, out, g_slice_m,
                (opt->jobs > 1) ? "" : "  <-- -j not set, so the slice stayed 0");
    }

    free(gpu_chunk);
  }
}

#endif /* VRNA_WITH_CUDA */

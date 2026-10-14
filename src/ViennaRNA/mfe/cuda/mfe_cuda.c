/** \file $Revision: 1.111 $ **/

/*
                  minimum free energy
                  RNA secondary structure prediction

                  c Ivo Hofacker, Chrisoph Flamm
                  original implementation by
                  Walter Fontana
                  g-quadruplex support and threadsafety
                  by Ronny Lorenz

                  Vienna RNA package
*/

/*
WBL  5 Feb 2018 vienna_rna/rf/rf_cuda2 allow nfiles sequences to be processed in parallel
WBL 29 Dec 2017 vienna_rna/rf/rf3, integrate CUDA changes 
WBL 29 Nov 2017 for vienna_rna/rf/rf, based on r1.82
WBL 12 Aug 2017 Revert to ViennaRNA-2.3.0/src/ViennaRNA/mfe.c add #GA
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>   /* SIZE_MAX: a RAM budget with no ceiling */
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/strings.h"
#include "ViennaRNA/utils/structures.h"
#include "ViennaRNA/params/default.h"
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/constraints/basic.h"
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/mfe/cuda/engine.h"   /* vrna_cuda_keeping_matrices() */
#include "ViennaRNA/constraints/soft.h"
#include "ViennaRNA/eval/gquad.h"
#include "ViennaRNA/mfe/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/loops/all.h"
#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/backtrack/global.h"   /* PORT: vrna_backtrack_from_intervals() */
#include "ViennaRNA/mfe/multibranch.h"    /* vrna_mfe_multibranch_m1(), for fM1 */
#include "ViennaRNA/params/salt.h"        /* vrna_salt_loop_int(), for the salt table */

#include <assert.h>
#ifdef STUB
#include "stub2.h"
//#include "stub.h"
#endif /*STUB*/


/* make this interface backward compatible with RNAlib < 2.2.0 */
#define VRNA_BACKWARD_COMPAT

#define MAXSECTORS        500     /* dimension for a backtrack array */

/*
#################################
# GLOBAL VARIABLES              #
#################################
*/

/*
#################################
# PRIVATE VARIABLES             #
#################################
*/

// New Jul 2026: per-row phase timing for par_fill_arrays()'s row loop
// (fill_arrays_loop.c) -- answers, with real numbers instead of guesses,
// how much of a fold's wall-clock is GPU kernel+transfer time
// (int_loop/hp_mb/load_my_c/modular_decomp) vs. pure host-side CPU time
// (the three nested for(H) for(j) combination loops that stitch GPU
// outputs together into my_c/my_fML each row). Same
// accumulate-and-print-once-at-exit shape as modular_decomposition.cu's
// existing graph_mgmt_seconds/print_graph_update_stats() -- duplicated
// here rather than shared across the two translation units, matching this
// codebase's established per-file self-containment convention.
static inline double
now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Stage-attribution timers, 2026-08-22. The "phase" timers below only ever
// covered the sweep's row loop; measurement on Continuous_Flow_Batching showed
// 20-31% of wall sitting OUTSIDE them on every workload, so these account for
// the rest. Non-static and declared in stub2.h because RNAfold.c (a separate
// translation unit) owns the record-building and output stages.
double stage_build_s      = 0.0; //vrna_fold_compound(): ptype + hc, both O(n^2)

// GPU-resident sweep, step 5a. RNA_GPU_SWEEP=1 makes the sweep consume the
// device results the kernels have been producing since steps 2-4, and skips
// both the three per-row host loops and the six per-row transfers that exist
// only to feed them (~364 GB a run at 400x5601).
//
// A GATE, not a demolition, and deliberately so: the host loops read the D2H'd
// buffers, so deleting the copies outright would take RNA_ROW_VERIFY with them
// -- leaving end-to-end fold output as the only check, which is strictly weaker
// (it proves values right WHERE THEY ARE READ, not everywhere). With a gate, one
// binary folds both ways and the two can be compared directly, which is the
// same self-comparison that settled the host-threading question.
//
// DEFAULTS ON as of 2026-08-30 (it was opt-in while it was being proven).
// RNA_GPU_SWEEP=0 restores the host sweep; RNA_ROW_VERIFY implies it, since
// the verify has nothing to compare against otherwise. Measured on Colab at
// 400x5601: 464.8 -> 333.0 s, 1.396x, output byte-identical. Cached: this
// is read once, not 16803 times, and it must be CONSTANT for the run -- the
// CUDA graph captures a different node topology in each mode, and a mode that
// changed per row would force a reinstantiate every row.
PUBLIC int
rnafold_gpu_sweep(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_GPU_SWEEP");
    if(e && e[0]) {
      v = strcmp(e,"0") ? 1 : 0;          // explicit setting wins, either way
    } else if(getenv("RNA_ROW_VERIFY")) {
      // RNA_ROW_VERIFY checks the device kernels against the HOST loops -- and
      // device mode is exactly the mode that does not run them. The verify
      // functions would early-out on their NULL host pointers and print
      // nothing, which reads identically to "verified clean". Default to the
      // host sweep so the tool keeps working; an explicit RNA_GPU_SWEEP=1
      // still overrides, above.
      v = 0;
      fprintf(stderr,"%-24s RNA_ROW_VERIFY is set: using the HOST sweep so there "
                     "is something to verify against (set RNA_GPU_SWEEP "
                     "explicitly to override)\n", __FILE__);
    } else {
      v = 1;                              // DEFAULT ON since 2026-08-30
    }
  }
  return v;
}
// int16 fml_j (INT16_FML_SCOPE.md). OFF by default and constant for the run,
// the same shape as RNA_GPU_SWEEP above and for the same reason: one binary
// folds both ways, so the encoding is compared against ITSELF rather than
// against a remembered result. The device buffers differ between modes, so this
// cannot vary per row.
//
// ON, d_fml_j becomes a 16-bit offset from a per-64-entry baseline. The bound
// is provable rather than sampled -- the most negative stack37 entry is -340
// and a 64-entry window admits at most 32 stacked pairs, so |offset| <= 10880
// against int16's 32766 -- but that bound is for the DEFAULT parameter table,
// so the pack kernel range-checks and traps rather than wrapping.
// Set by rnafold_fml_int16_vet_params() when the LOADED parameter table cannot
// support the encoding. Sticky: once a table has been rejected the gate stays
// shut for the rest of the process, because the tables are global and a later
// batch cannot un-load them.
static int g_fml_int16_unsafe = 0;
static int g_fml_int16_auto   = -1;   /* -1 undecided; see rnafold_fml_int16_auto_decide() */

// DEFAULT ON since 2026-09-27. Measured on an A100 at 5601 nt, byte-identical at
// every shape: md -8.8 % at 8 records, -15.5 % at 32, -17.7 % at 96, -18.4 % at
// 200; wall -1.8 % from 32 records up. VRAM also halves (12946 -> 10168 MB at 96
// records), which independently raises what admission can admit. The wall gain is
// far smaller than the md gain because md is ~43 % of the wall and the host-side
// build is ~21 % -- see PORT_ZEROCONF_SCOPE.md.
//
// WHY IT IS *AUTO* RATHER THAN MERELY ON. int16 cannot be honoured in some
// combinations -- today --noLP (rnafold_fml_int16_stand_down()); earlier
// RNA_MD_PRUNE and RNA_SLOT_FLOW, both deleted in the cleanup -- and refusing
// them with exit() would have made a working command line a hard failure the
// moment the default flipped. A default must not do that.
//
// So the asymmetry, which is the same one PORT_ZEROCONF_SCOPE.md argues for
// --enable-cuda: an EXPLICIT request that cannot be honoured is an error, while a
// DEFAULT that cannot be honoured steps aside and says so. rnafold_fml_int16_explicit()
// is what the refusal sites test to tell the two apart.
//
//   RNA_FML_INT16=1  force on; conflicts are errors
//   RNA_FML_INT16=0  force off
//   (unset)          AUTO: on, and quietly off where it is not supported
PUBLIC int
rnafold_fml_int16_explicit(void) {
  const char *e = getenv("RNA_FML_INT16");

  return (e && e[0] && strcmp(e, "0")) ? 1 : 0;
}

PUBLIC int
rnafold_fml_int16(void) {
  static int v = -1;

  if(v < 0) {
    const char *e = getenv("RNA_FML_INT16");

    /* unset -> on. "0" -> off. anything else -> on. */
    v = (e && e[0]) ? (strcmp(e, "0") ? 1 : 0) : 1;

    if (!vrna_cuda_quiet() && e && e[0]) {
      /* explicit; AUTO says what it chose in rnafold_fml_int16_auto_decide() */
      fprintf(stderr,"%-24s fml_j is %s (RNA_FML_INT16 set explicitly)\n", __FILE__,
              v ? "16-bit offsets from a per-64 baseline" : "full int32");
    }
  }

  /* AUTO (unset) defers to rnafold_fml_int16_auto_decide(); until it has run, on. */
  return v && !g_fml_int16_unsafe && (rnafold_fml_int16_explicit() || g_fml_int16_auto != 0);
}

/* AUTO, REFINED 2026-10-06 (Luke: "make the int32/16 switch based on VRAM limitation and DRAM
 * pressure"), on the A100 N2 run: under sparse md, int32 is 1.3-2.7 % FASTER than int16 --
 * md no longer streams the fML triangle, so int16's pack and decode are pure cost -- and
 * int16's VRAM saving buys nothing while the cell cap, not VRAM, sizes the chunks.
 *
 * So, unset, int16 is kept only where it pays:
 *   DRAM pressure  the dense md will run (sparse md off or refused), and it streams the
 *                  triangle: int16 halves those bytes (md -20.7 % at 400 x 5601, 10-03).
 *   VRAM limit     chunks are sized by VRAM -- no cell cap (the library path), or a full-cap
 *                  chunk would not fit in int32 (a 4 GB card at the default cap).
 * Otherwise int32. Decided ONCE, from the first compute_gpu_usable_bytes() of the run, which
 * both admission paths call before they price a record: the device buffers differ between
 * the modes, so it cannot change per chunk. Reads before the decision see int16; that can only
 * UNDER-state a record's bytes when int32 is then chosen, and int32 is only chosen when a whole
 * full-cap chunk fits with room to spare, so admission stays safe. */
PUBLIC void
rnafold_fml_int16_auto_decide(const size_t usable_bytes, const unsigned long long cap_cells,
                              const size_t full_cap_bytes32, const int dense_md)
{
  if (g_fml_int16_auto >= 0) return;
  const char *why;
  if (dense_md)             { g_fml_int16_auto = 1; why = "the dense md streams the fML triangle (DRAM pressure)"; }
  else if (cap_cells == 0)  { g_fml_int16_auto = 1; why = "chunks are sized by VRAM (no cell cap)"; }
  else if (full_cap_bytes32 > usable_bytes) { g_fml_int16_auto = 1; why = "a full-cap chunk would not fit in VRAM as int32"; }
  else                      { g_fml_int16_auto = 0; why = "sparse md, and a full-cap chunk fits in VRAM as int32"; }
  if (!rnafold_fml_int16_explicit() && getenv("RNA_FML_INT16") == NULL && !vrna_cuda_quiet())
    fprintf(stderr, "%-24s fml_j AUTO: %s -- %s (full-cap chunk %.1f GB as int32, %.1f GB usable)\n",
            __FILE__, g_fml_int16_auto ? "int16" : "int32", why,
            (double)full_cap_bytes32 / 1073741824.0, (double)usable_bytes / 1073741824.0);
}

/* A DEFAULT that cannot be honoured steps aside; an EXPLICIT request does not.
 * Must be called before init_gpu() commits the layout. */
PUBLIC void
rnafold_fml_int16_stand_down(const char *why) {
  if (!rnafold_fml_int16())
    return;                         /* already off, or already stood down */
  if (rnafold_fml_int16_explicit()) {
    fprintf(stderr,"%-24s RNA_FML_INT16=1 cannot be honoured: %s. Unset it.\n",
            __FILE__, why);
    exit(EXIT_FAILURE);
  }
  if (!vrna_cuda_quiet())
    fprintf(stderr,"%-24s fml_j falls back to int32: %s. (int16 is the default; set "
                   "RNA_FML_INT16=1 to make this an error instead.)\n",
            __FILE__, why);
  g_fml_int16_unsafe = 1;           /* sticky, same reasoning as the vet path */
}

// The int16 fML offset bound is (FML_BLK/2) * |most negative stack entry|: a
// window of B consecutive positions admits at most B/2 stacked pairs. For the
// DEFAULT table the worst entry is -340, giving 32 * 340 = 10880 against int16's
// 32766 -- about 3x headroom, which is why the encoding is length-independent by
// construction rather than by extrapolation.
//
// **A -P FILE REPLACES stack37, SO THAT BOUND IS AN ASSUMPTION, NOT A PROOF.**
// Measured 2026-09-10 with every negative stack entry set to -2000: the encoding
// overflowed, and the pack kernel's range check did NOT save us --
//
//   * its `assert(0)` is a NO-OP, because release builds define NDEBUG. The
//     comment above it said "TRAP, never wrap"; under -DNDEBUG it printed and
//     then wrapped.
//   * device-side printf goes to the process STDOUT, so 48 781 diagnostic lines
//     landed IN THE FOLD OUTPUT, and
//   * the answer underneath was wrong anyway on 8 of 12 records, by up to
//     31.8 kcal/mol -- well-formed structures, plausible energies.
//
// So the check has to happen HERE, on the host, before any folding: a kernel is
// the wrong place to decide a whole run is impossible. Declining to int32 is the
// same shape as every other routing guard in this port -- the user gets the
// right answer, more slowly, and is told why.
//
// The bound counts STACKING ONLY; tetraloop bonuses and dangles also contribute,
// so it is the right shape of argument rather than a finished proof. The pack
// kernel's range check therefore stays as a backstop -- now one that actually
// stops (see pack_fml_kernel).
PUBLIC void
rnafold_fml_int16_vet_params(const int worst_stack, const int blk) {
  const long long bound = (long long)(blk/2) * (long long)(worst_stack < 0 ? -worst_stack
                                                                           : worst_stack);
  // No "already done" flag: this is called once per batch and costs 64 int
  // comparisons, and a vrna_mfe_batch() caller CAN load a different table
  // between batches. The stickiness lives in g_fml_int16_unsafe, deliberately:
  // once a table has been rejected the gate stays shut, because the device
  // buffers for the run were sized for int32 the moment we declined.
  if(g_fml_int16_unsafe) return;
  if(bound <= 32766) return;

  g_fml_int16_unsafe = 1;
  // Only worth saying if the user actually asked for int16.
  {
    const char *e = getenv("RNA_FML_INT16");
    if(e && e[0] && strcmp(e,"0"))
      fprintf(stderr,
              "%-24s RNA_FML_INT16 DECLINED: this parameter table's worst stack "
              "entry is %d, so the fML offset bound is %d/2 * %d = %lld, past "
              "int16's 32766. Folding in int32 instead -- the answer is "
              "unaffected. See INT16_FML_SCOPE.md.\n",
              __FILE__, worst_stack, blk,
              worst_stack < 0 ? -worst_stack : worst_stack, bound);
  }
}

// The chunk's record capacities, owned here because par_fill_arrays() must see
// EXACTLY the table par_mfe() allocated the buffers from.
static size_t *g_cap_H = NULL;
static int     g_cap_n = 0;

PUBLIC const size_t *
rnafold_chunk_capacities(const int nfiles) {
  assert(g_cap_H != NULL && nfiles <= g_cap_n);
  (void)nfiles;
  return g_cap_H;
}

double stage_prepare_s    = 0.0; //vrna_fold_compound_prepare()
double stage_prefill_s    = 0.0; //par_fill_arrays()'s pre-sweep host matrix INF fill
double stage_backtrack_s  = 0.0; //backtrack(), single- or multi-threaded
double stage_output_s     = 0.0; //printing folds
double stage_gpuinit_s    = 0.0; //init_gpu/2/3
double stage_teardown_s   = 0.0; //teardown_gpu/2/3
double stage_btpin_worker_s = 0.0; //T2a: worker-seconds spent pinning backtrack scratch (inside backtrack)
double stage_btfetch_worker_s = 0.0; //T2a: worker-seconds in the fetch itself
double stage_free_s       = 0.0; //vrna_fold_compound_free()

// gpuinit came out at 199 s -- 25.6% of the 776 s Colab benchmark and the
// second-largest item overall -- with no idea which part of init_gpu/2/3 it
// is. These split it three ways per function, plus the two cross-cutting
// suspects: the per-record O(n^2) host bitmask packing loops, and cudaMalloc
// (which matters here because teardown frees ~20 GB per chunk in a reported
// 0.000 s, so the cost may simply be deferred into the next chunk's mallocs).
double stage_ig1_s        = 0.0; //init_gpu   (modular_decomposition.cu)
double stage_ig2_s        = 0.0; //init_gpu2  (int_loop.cu)
double stage_ig3_s        = 0.0; //init_gpu3  (hp_mb_loop.cu)
double stage_ig_pack_s    = 0.0; //host-side bitmask / sequence packing loops
double stage_ig_malloc_s  = 0.0; //cudaMalloc inside the three init functions

// See stub2.h. Default 0 = always use the host packing path, so anything that
// forgets to set it stays correct rather than silently wrong.
//
// It is no longer set by any caller: par_fill_arrays() DERIVES it from the batch
// just before init_gpu (see the long note there for why a caller-set flag was
// the wrong shape). The initialiser stays 0 so that a path which somehow reaches
// init_gpu2/init_gpu3 without passing through that derivation is slow rather
// than wrong.
//
// HISTORY, because this was two silent wrong answers deep and the second one
// looked exactly like the fix for the first. On 2.3.0 RNAfold.c set this
// (0b4bcf3e); the 2.7.2 port dropped that line, which read as a one-line
// regression worth ~20% of wall. It was not: enabling it on 2.7.2 unchanged
// returned a WRONG ANSWER on 9 of 10 records, because the device replica had
// two divergences from 2.7.2's hard.c that nothing read and nothing checked --
// the i==j diagonal open-coded CLOSING_LOOPS where upstream writes ALL_LOOPS,
// and max_bp_span was passed as ONE SCALAR for a batch in which it is PER
// RECORD. Both are fixed (stub2.h rnafold_hc_opt(), hp_mb_loop.cu init_gpu3);
// RNA_HC_VERIFY=1 now reports 532928 words x 4 masks, zero mismatching.
int g_hc_seq_derived = 0;

/* Salt correction table, shared by the hairpin and internal-loop kernels.
 *
 * Indexed by BACKBONE COUNT, upstream's `L`: a hairpin of `size` uses
 * [size + 1] (eval/hairpin.h:357), an internal loop uses [backbones] where
 * backbones = nl + ns + 2 (eval/internal.h:644). Upstream stores only
 * SaltLoop[MAXLOOP + 2] and evaluates the closed form vrna_salt_loop_int()
 * beyond that. An internal loop can never exceed MAXLOOP so the table always
 * suffices there, but a HAIRPIN is unbounded -- which is the only reason this
 * function exists. Rather than port exp/log into a kernel and then have to
 * defend a double-precision match, the closed form is evaluated here, on the
 * host, by upstream's own function, for every index a batch could ask for.
 *
 * `out` must hold n_max + 2 ints. All zero when salt is at its default, which
 * is what makes the kernels' unconditional add correct in the common case.
 */
PUBLIC void
rnafold_build_salt_table(const vrna_param_t *P, const int n_max, int *out)
{
  const double salt = P->model_details.salt;
  int          L;

  if (salt == VRNA_MODEL_DEFAULT_SALT) {
    memset(out, 0, (size_t)(n_max + 2) * sizeof(int));
    return;
  }

  for (L = 0; L < n_max + 2; L++) {
    out[L] = (L <= MAXLOOP + 1)
             ? P->SaltLoop[L]
             : vrna_salt_loop_int(L,
                                  salt,
                                  P->temperature + K0,
                                  P->model_details.backbone_length);
  }
}


double rnafold_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* N5a P0 (PORT_BACKTRACK_PIPELINE.md §5): one line per chunk beside the
 * RNA_RECORD_TRACE file, so the GPU's idle time between sweeps is read off
 * chunk by chunk instead of inferred from the run's totals. RNAfold opens it
 * with its own clock base, so the two files share a time axis.
 *
 * Under it, the sweep's end is a device drain. Without one, sweep_t1 would be
 * when the last launch was ISSUED, and the backtrack's first synchronous fetch
 * would absorb the sweep's tail and be blamed for it. The drain costs nothing
 * the fetch was not about to wait for anyway.
 *
 * CSV: chunk,nrec,max_len,cells,enter,init_t1,sweep_t1,bt_t1,exit,
 *      bt_threads,fetch_worker_s,busy_worker_s
 *   enter..init_t1    device allocation and upload (stage_gpuinit_s)
 *   init_t1..sweep_t1 the sweep, GPU busy
 *   sweep_t1..bt_t1   fetch + trace on the host, GPU IDLE -- N5a's prize
 *   exit -> next enter, from the other file's view: teardown, free, output
 *                     dispatch and any wait for the builder, GPU idle too
 * Times are seconds on CLOCK_MONOTONIC minus RNAfold's trace base. */
static FILE  *g_chunk_fp    = NULL;
static double g_chunk_base  = 0.0;
static int    g_chunk_k     = 0;
static double g_bt_fetch_ws = 0.0;   /* the last backtrack_all()'s worker sums */
static double g_bt_busy_ws  = 0.0;
static int    g_bt_threads  = 0;

PUBLIC void
rnafold_chunk_trace_open(const char *record_path, const double base) {
  char *p;

  if(g_chunk_fp || !record_path || !record_path[0])
    return;
  p = vrna_strdup_printf("%s.chunks.csv", record_path);
  g_chunk_fp = fopen(p, "w");
  if(!g_chunk_fp) {
    fprintf(stderr, "%-24s RNA_RECORD_TRACE: cannot open %s -- no chunk spans\n",
            __FILE__, p);
  } else {
    g_chunk_base = base;
    fprintf(g_chunk_fp, "chunk,nrec,max_len,cells,enter,init_t1,sweep_t1,bt_t1,exit,"
                        "bt_threads,fetch_worker_s,busy_worker_s\n");
  }
  free(p);
}

/* a span edge, or an empty field when the schedule has no such edge */
PRIVATE void
chunk_trace_t(const double t) {
  if(t > 0.0) fprintf(g_chunk_fp, ",%.6f", t - g_chunk_base);
  else        fprintf(g_chunk_fp, ",");
}

PRIVATE void
chunk_trace_line(const int nrec, const int max_len, const size_t cells,
                 const double enter, const double init_t1, const double sweep_t1,
                 const double bt_t1, const double t_exit,
                 const int bt_threads, const double fetch_ws, const double busy_ws) {
  /* Under RNA_BT_PIPELINE a chunk's line is written when its trace is JOINED,
   * so bt_t1 can come after exit: that gap is the overlap itself. */
  fprintf(g_chunk_fp, "%d,%d,%d,%zu", g_chunk_k++, nrec, max_len, cells);
  chunk_trace_t(enter);
  chunk_trace_t(init_t1);
  chunk_trace_t(sweep_t1);
  chunk_trace_t(bt_t1);
  chunk_trace_t(t_exit);
  if(bt_t1 > 0.0)
    fprintf(g_chunk_fp, ",%d,%.6f,%.6f\n", bt_threads, fetch_ws, busy_ws);
  else
    fprintf(g_chunk_fp, ",,,\n");
  fflush(g_chunk_fp);   /* a handful of lines; never lose them to an abort */
}

/* N5a (PORT_BACKTRACK_PIPELINE.md): RNA_BT_PIPELINE=1 retires each chunk's
 * backtrack buffers into a rnafold_retired_t (bt_retired.h) when its sweep
 * ends, and fetches from there on non-blocking copy streams. Opt-in until the
 * A100 run (decision 9.2). For a driver that attached (RNAfold) the trace then
 * runs beside the next chunk's sweep (P2, bt_async_join()); for anyone else it
 * runs in today's order (P1). Neither may change one byte of output. */
PUBLIC int
rnafold_bt_pipeline(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_BT_PIPELINE");
    v = (e && e[0] && strcmp(e,"0")) ? 1 : 0;
    if(v) fprintf(stderr,"%-24s RNA_BT_PIPELINE=1: each chunk's backtrack buffers are "
                         "retired at the end of its sweep and, for RNAfold, traced "
                         "beside the next chunk's sweep\n", __FILE__);
  }
  return v;
}

/* positive evidence: chunks that went through retire -> backtrack -> release */
static int g_bt_retired_n = 0;

/* ===================== N5a P2: the trace runs beside the next sweep ==========
 *
 * PORT_BACKTRACK_PIPELINE.md §3.3/§3.4. With RNA_BT_PIPELINE=1 and a driver that
 * has attached a "chunk done" callback (RNAfold, rnafold_bt_pipeline_attach()),
 * par_mfe() retires the chunk at the end of its sweep, starts its backtrack on
 * a trace thread and RETURNS: the caller tears down, builds and sweeps the next
 * chunk while this one is traced. vrna_mfe_batch() itself is unchanged
 * (decision 9.1); only a driver that attached knows the answers come later.
 *
 * The rules, each one load-bearing:
 *   - ONE backtrack at a time. The scratch pool (bt_pool_get) and the copy
 *     streams are shared, so every par_mfe() joins the pending trace after its
 *     own sweep and before any backtrack of its own. At most two triangle sets
 *     are on the device: the one being traced and the one being swept.
 *   - The callback runs on the MAIN thread, at the join -- never on the trace
 *     thread. It frees fold compounds and dispatches output, and RNAfold's
 *     output guards only lock under -j, so it must not run beside the driver.
 *     Output for chunk k therefore appears after sweep k+1; its ORDER is kept by
 *     the ostream slots reserved at read time.
 *   - The release (cudaFree, which synchronises the device) also happens at the
 *     join, where the sweep has just drained anyway.
 *   - Only a call the engine marks as allowed may hand off: one whose records
 *     are the last piece of its vrna_mfe_batch() (rnafold_bt_async_allow()).
 *   - A driver that has handed off must call rnafold_bt_pipeline_drain() before
 *     it reads the last answers or exits. */
typedef void (*rnafold_bt_done_fn)(void *ctx);

static rnafold_bt_done_fn g_bt_done_cb    = NULL;
static void              *g_bt_next_ctx   = NULL;
static int                g_bt_async_ok   = 0;
static int                g_bt_went_async = 0;

typedef struct {
  pthread_t                    th;
  int                          live;
  rnafold_retired_t           *ret;
  int                          nfiles, length, cpu_queue_threads;
  size_t                       cells;
  const vrna_fold_compound_t **VC;
  const char                 **Structure;
  float                       *EN;
  int                         *energy;     /* heap: par_mfe()'s is a VLA */
  size_t                      *tri_off_H;  /* heap copy, nfiles+1 */
  void                        *ctx;
  double                       enter, init_t1, sweep_t1, exit_t, t_start, bt_t1;
  rnafold_bt_done_fn           cb;          /* the driver's, or NULL: an internal hand-off (P3) */
  int                          bt_threads;
  double                       fetch_ws, busy_ws;
} bt_async_t;

static bt_async_t g_bt_async;
static int        g_bt_async_n      = 0;     /* chunks traced beside a sweep */
static int        g_bt_internal_n   = 0;     /* of which inside one vrna_mfe_batch() (P3) */
static double     g_bt_traced_s     = 0.0;   /* their trace wall, summed */
static double     g_bt_wait_s       = 0.0;   /* waited for a trace at a join */
static double     g_bt_drain_wait_s = 0.0;   /* waited for the last one at the end */

PUBLIC void
rnafold_bt_pipeline_attach(rnafold_bt_done_fn done) {
  g_bt_done_cb = done;
}

/* The driver's context for the NEXT batch. Cleared on hand-off; a batch that
 * did not go async leaves rnafold_bt_pipeline_went_async() at 0. */
PUBLIC void
rnafold_bt_pipeline_ctx(void *ctx) {
  g_bt_next_ctx   = ctx;
  g_bt_went_async = 0;
}

PUBLIC int
rnafold_bt_pipeline_went_async(void) {
  return g_bt_went_async;
}

/* engine.c: may the coming par_mfe() call hand its trace off? RNAFOLD_BT_ASYNC_DRIVER (the
 * last sub-chunk of a call: to a driver that attached, P2) or RNAFOLD_BT_ASYNC_INTERNAL (any
 * other sub-chunk, P3: no callback; the trace writes straight into the caller's arrays and
 * the engine joins it before the call returns), 0 = trace in place. */
PUBLIC void
rnafold_bt_async_allow(const int ok) {
  g_bt_async_ok = ok;
}

/* P3, rule (b) (PORT_BUDGETS.md 9.3): device bytes the pending trace's retired set holds,
 * so the engine can tell whether it narrows the next sub-chunk. 0 when none is pending. */
PUBLIC size_t
rnafold_bt_pending_device_bytes(void) {
  return g_bt_async.live ? rnafold_retired_bytes(g_bt_async.ret) : 0;
}

PRIVATE void bt_async_join(const int at_drain);

static int g_bt_early_n = 0;   /* P3 rule (b): joined early rather than narrow a sub-chunk */

/* P3: join the pending trace now. RNAFOLD_BT_JOIN_END: the engine's end-of-call drain, an
 * internal trace only (a driver hand-off stays pending for its driver). RNAFOLD_BT_JOIN_EARLY:
 * rule (b), whatever is pending, because it would narrow the next sub-chunk. */
PUBLIC void
rnafold_bt_pipeline_join(const int why) {
  if(!g_bt_async.live)
    return;
  if((why == RNAFOLD_BT_JOIN_END) && g_bt_async.cb)
    return;
  if(why == RNAFOLD_BT_JOIN_EARLY)
    g_bt_early_n++;
  bt_async_join(0);
}

PRIVATE void
backtrack_all(const int nfiles, const vrna_fold_compound_t **VC,
              const char **Structure, int *energy, float *EN,
              const size_t *tri_off_H, const int cpu_queue_threads,
              const rnafold_retired_t *ret);

PRIVATE void *
bt_async_main(void *p) {
  bt_async_t *a = (bt_async_t *)p;

  a->t_start = rnafold_now_seconds();
  backtrack_all(a->nfiles, a->VC, a->Structure, a->energy, a->EN, a->tri_off_H,
                a->cpu_queue_threads, a->ret);
  a->bt_t1      = rnafold_now_seconds();
  /* backtrack_all() leaves these for the chunk trace; copy them while they
   * are still this chunk's -- the join reads them from here */
  a->bt_threads = g_bt_threads;
  a->fetch_ws   = g_bt_fetch_ws;
  a->busy_ws    = g_bt_busy_ws;
  return NULL;
}

/* Wait for the pending trace, write its chunk line, release its retired set,
 * then hand its answers to the driver -- on this, the main, thread. */
PRIVATE void
bt_async_join(const int at_drain) {
  bt_async_t   *a = &g_bt_async;
  double        t0, w;
  void         *ctx;
  rnafold_bt_done_fn cb;

  if(!a->live)
    return;

  t0 = rnafold_now_seconds();
  pthread_join(a->th, NULL);
  w  = rnafold_now_seconds() - t0;
  if(at_drain) g_bt_drain_wait_s += w;
  else         g_bt_wait_s       += w;
  g_bt_traced_s += a->bt_t1 - a->t_start;

  if(g_chunk_fp)
    chunk_trace_line(a->nfiles, a->length, a->cells, a->enter, a->init_t1, a->sweep_t1,
                     a->bt_t1, a->exit_t, a->bt_threads, a->fetch_ws, a->busy_ws);

  rnafold_retired_release(a->ret);
  free(a->energy);
  free(a->tri_off_H);
  ctx     = a->ctx;
  cb      = a->cb;
  a->live = 0;
  g_bt_retired_n++;

  if(cb)
    cb(ctx);
}

PUBLIC void
rnafold_bt_pipeline_drain(void) {
  bt_async_join(1);
}

PRIVATE void
rnafold_bt_pipeline_report(void) {
  const double hidden = g_bt_traced_s - g_bt_wait_s - g_bt_drain_wait_s;

  /* tools/make_nb_n5a.py parses this line with a strict regex: keep its shape */
  fprintf(stderr, "%-24s backtrack pipeline: %d chunk(s) retired, %d traced beside the "
                  "next sweep: traced %.3f s, waited %.3f s at joins + %.3f s at the end, "
                  "OVERLAPPED %.3f s\n", __FILE__, g_bt_retired_n, g_bt_async_n,
          g_bt_traced_s, g_bt_wait_s, g_bt_drain_wait_s, hidden > 0.0 ? hidden : 0.0);
  fprintf(stderr, "%-24s backtrack pipeline inside vrna_mfe_batch(): %d sub-chunk(s) traced "
                  "beside the next, %d joined early so the next was not narrowed\n", __FILE__,
          g_bt_internal_n, g_bt_early_n);
}

PRIVATE void
print_stage_timing_stats(void) {
  if (vrna_cuda_quiet())
    return;
  if (rnafold_bt_pipeline())
    rnafold_bt_pipeline_report();
  fprintf(stderr,
    "%-24s backtrack scratch (worker-s): alloc=%.3f fetch=%.3f\n",
    __FILE__, stage_btpin_worker_s, stage_btfetch_worker_s);
  fprintf(stderr,
    "%-24s stage timing (s): build=%.3f prepare=%.3f prefill=%.3f backtrack=%.3f "
    "output=%.3f gpuinit=%.3f teardown=%.3f free=%.3f || non-sweep total=%.3f\n",
    __FILE__, stage_build_s, stage_prepare_s, stage_prefill_s, stage_backtrack_s,
    stage_output_s, stage_gpuinit_s, stage_teardown_s, stage_free_s,
    stage_build_s + stage_prepare_s + stage_prefill_s + stage_backtrack_s
    + stage_output_s + stage_gpuinit_s + stage_teardown_s + stage_free_s);
  fprintf(stderr,
    "%-24s gpuinit breakdown (s): init_gpu=%.3f init_gpu2=%.3f init_gpu3=%.3f "
    "|| of which pack=%.3f cudaMalloc=%.3f other=%.3f\n",
    __FILE__, stage_ig1_s, stage_ig2_s, stage_ig3_s,
    stage_ig_pack_s, stage_ig_malloc_s,
    stage_ig1_s + stage_ig2_s + stage_ig3_s - stage_ig_pack_s - stage_ig_malloc_s);
}

static double phase_int_loop_s          = 0.0;
static double phase_hp_mb_s             = 0.0;
static double phase_new_c_host_s        = 0.0;
static double phase_load_my_c_s         = 0.0;
static double phase_fml_host_s          = 0.0;
static double phase_modular_decomp_s    = 0.0;
// Was phase_my_fml_update_host_s. That loop is gone (see fill_arrays_loop.c);
// what remains under this timer is the row-shaped fml_prev cache write, which
// is genuinely host-combine work. The post-sweep D2H that replaced the rest of
// it is timed separately below and counted as transfer, not host, time.
static double phase_fml_prev_host_s     = 0.0;
// Both post-sweep triangle fetches, fML and my_c (fetch_fML/fetch_my_c).
static double phase_fetch_mx_s          = 0.0;

static void
print_phase_timing_stats(void) {
  const double gpu_transfer_total = phase_int_loop_s + phase_hp_mb_s
                                   + phase_load_my_c_s + phase_modular_decomp_s
                                   + phase_fetch_mx_s;
  const double host_combine_total = phase_new_c_host_s + phase_fml_host_s
                                   + phase_fml_prev_host_s;
  if (!vrna_cuda_quiet())
    fprintf(stderr,
    "%-24s phase timing (s): int_loop=%.3f hp_mb=%.3f load_my_c=%.3f "
    "modular_decomp=%.3f fetch_mx=%.3f | new_c_host=%.3f fml_host=%.3f fml_prev_host=%.3f "
    "|| GPU+transfer total=%.3f host-combine total=%.3f\n",
    __FILE__, phase_int_loop_s, phase_hp_mb_s, phase_load_my_c_s,
    phase_modular_decomp_s, phase_fetch_mx_s, phase_new_c_host_s, phase_fml_host_s,
    phase_fml_prev_host_s, gpu_transfer_total, host_combine_total);

  // RNA_LAUNCH_STATS, printed immediately after the phase line so the two can be
  // read together: the kernel's own device time against the phase that contains
  // it. Silent unless the knob is set.
  rnafold_launch_stats_report();
}

/*
#################################
# PRIVATE FUNCTION DECLARATIONS #
#################################
*/

PRIVATE int           fill_arrays(vrna_fold_compound_t *vc){exit(99);}//use par_fill_arrays, code now included via fill_arrays.c
/* Deleted with the 2.3.0 copies they belonged to: fill_arrays_circ(),
 * fill_arrays_comparative(), fill_arrays_comparative_circ(),
 * backtrack_comparative(), backtrack() and mfe_cuda_vrna_mfe().
 * Circular and comparative folding, and all backtracking, are upstream's --
 * reached through vrna_mfe() and vrna_backtrack_from_intervals(). */

/*
#################################
# BEGIN OF FUNCTION DEFINITIONS #
#################################
*/

//callback_backtrack essentially existing code but packaged to make easier to call from nfiles in parallel version
PRIVATE float
callback_backtrack_bts(const vrna_fold_compound_t* vc,
		   const int energy,
		   char *structure, //out
		   vrna_bts_t bt_stack) { /* interval stack, possibly PRE-SEEDED by
                                           * postprocess_circular() */
    /* call user-defined recursion status callback function */

  float   mfe = (float)(INF/100.);
  char    *ss;
  vrna_bps_t        bp;
  const int length  = (int) vc->length;

    if(vc->stat_cb)
      vc->stat_cb(vc, VRNA_STATUS_MFE_POST, vc->auxdata);  /* PORT: 2.7.2 added the fold compound as first argument */

    if(structure && vc->params->model_details.backtrack){
      /* vrna_bps_t, NOT the legacy vrna_bp_stack_t.
       *
       * The legacy form carries only .i and .j. A G-quadruplex needs its
       * layer/linker layout (bp.L, bp.l[3]) to render, and
       * vrna_backtrack_from_intervals() DISCARDS both when it downconverts --
       * so through that entry a gquad can only ever come out as a single '+'
       * where the box belongs, however correct the energy is. Measured in
       * PORT_GQUAD_SPEC.md G2: 2 '+' characters against upstream's 14, on a
       * fold whose energy was byte-exact.
       *
       * vrna_backtrack_from_intervals_bps() is the VRNA-PATCH that fills this
       * directly. It grows on demand, so the old "guess how many G's" sizing
       * is gone rather than merely enlarged. */
      bp = vrna_bps_init(4*(1+length/2));

      switch(vc->type){
        /* PORT TO 2.7.2: backtrack_comparative() is flagged for deletion with
         * the comparative forward recursion it belongs to. A comparative fold
         * compound must never reach this function at all -- the routing guard
         * declines them, so they are folded and backtracked by upstream. The
         * branch becomes an explicit refusal rather than a dangling call:
         * reaching it means the guard has a hole, and this project's standing
         * rule is that an accelerator refuses rather than answers a fold
         * compound it does not support. */
        case VRNA_FC_TYPE_COMPARATIVE:
          vrna_log_error("%s: comparative fold compound reached the CUDA "
                         "backtrack path; the routing guard should have "
                         "declined it", __FILE__);
          vrna_bps_free(bp);
          return (float)(INF/100.);

        case VRNA_FC_TYPE_SINGLE:     /* fall through */

        /* PORT TO 2.7.2: upstream's own entry, not this file's copy.
         *
         * vrna_backtrack_from_intervals() is PUBLIC in 2.7.2
         * (backtrack/global.h:38) and documented as "backtrack a secondary
         * structure with pre-evaluated structure components" -- which is
         * exactly the GPU case: the sweep has filled c/fML, and what remains
         * is to retrace them. The fork's own backtrack() was a 2.3.0 copy
         * calling eight deprecated vrna_BT_* wrappers, and at least one of
         * them no longer works against 2.7.2's matrices ("backtracking failed
         * in fML" / "in repeat", then a segfault). Same lesson as the private
         * vrna_mfe() copy: the copy exists because 2.3.0 had no seam, and it
         * goes as soon as upstream offers the entry point. */
        default:                      vrna_backtrack_from_intervals_bps(vc, bp, bt_stack);
                                      break;
      }

      /* vrna_db_from_bps() expands a quadruplex via
       * vrna_db_insert_gq(structure, i, bp.L, bp.l, length); the legacy
       * vrna_db_from_bp_stack() has no layout to expand and writes one '+'. */
      ss = vrna_db_from_bps(bp, length);
      strncpy(structure, ss, length + 1);
      free(ss);
      vrna_bps_free(bp);
    }

    if (vc->params->model_details.backtrack_type=='C')
      mfe = (float) vc->matrices->c[vc->jindx[length]+1]/100.;
    else if (vc->params->model_details.backtrack_type=='M')
      mfe = (float) vc->matrices->fML[vc->jindx[length]+1]/100.;
    else
      mfe = (float) energy/100.;

    if(vc->type == VRNA_FC_TYPE_COMPARATIVE)
      mfe /= (float)vc->n_seq;

  return mfe;
}

/* New Jul 2026: parallelizes the backtrack loop below across CPU threads.
 * Each sequence's backtrack only reads that sequence's own vc (already
 * fully filled in host memory by par_fill_arrays()) and writes its own
 * bt_stack (thread-local, declared inside backtrack_worker) and Structure[i]/
 * EN[i] slot -- verified no shared mutable state anywhere in backtrack()'s
 * call chain (exterior_loops.c/multibranch_loops.c/hairpin_loops.c/
 * interior_loops.c), so no locking is needed beyond the atomic work-claim
 * counter below.
 *
 * RNA_BACKTRACK_THREADS is deliberately a separate knob from RNA_CPU_THREADS
 * (RNAfold_cpu_queue.c's pool for whole off-batch sequences) -- "auto"
 * sizing subtracts cpu_queue_threads (RNAfold.c's already-resolved
 * RNA_CPU_THREADS count, passed in) from hardware concurrency so the two
 * pools don't oversubscribe cores when both are active at once, without
 * this file re-deriving RNA_CPU_THREADS' own parsing/disable-condition
 * logic.
 *   "0"          -> disabled, exactly the original serial loop.
 *   unset        -> "auto" (DEFAULT CHANGED 2026-08-30; was serial).
 *   "auto"       -> max(1, min(nfiles, hw_concurrency - cpu_queue_threads)).
 *   "<N>"        -> exactly N threads, capped at nfiles.
 */
PRIVATE int
backtrack_thread_count(const int nfiles, const int cpu_queue_threads) {
  static int        env_read = 0;
  static const char *env     = NULL;
  if(!env_read) {
    env      = getenv("RNA_BACKTRACK_THREADS");
    env_read = 1;
  }
  // DEFAULT CHANGED 2026-08-30: unset now means "auto", not serial. Measured on
  // Colab (400x5601): serial-everything 573.2 s vs both pools auto 464.8 s, with
  // every config byte-identical -- so the old default cost 1.23x for no
  // correctness benefit. "0" still forces serial for anyone who wants it.
  const char *v = (env && env[0]) ? env : "auto";
  if(!strcmp(v, "0")) return 1;

  int n;
  if(!strcmp(v, "auto")) {
    long hw = sysconf(_SC_NPROCESSORS_ONLN);
    if(hw < 1) hw = 1;
    n = (int)hw - cpu_queue_threads;
  } else {
    n = atoi(v);
  }
  if(n < 1) n = 1;
  if(n > nfiles) n = nfiles;
  return n;
}

PRIVATE float
callback_backtrack(const vrna_fold_compound_t* vc, int s, const int energy,
                   char *structure, sect bt_stack[MAXSECTORS]);

/* RNA_CIRC_VERIFY -- see the check in backtrack_one_slot(). O(n^3) per record,
 * so it is for small inputs and never a run mode. */
PRIVATE int
rnafold_circ_verify(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_CIRC_VERIFY");
    v = (e && e[0] && strcmp(e,"0")) ? 1 : 0;
    if(v) fprintf(stderr,"%-24s RNA_CIRC_VERIFY=1: checking fM2_real against its "
                         "definition, O(n^3) per record\n", __FILE__);
  }
  return v;
}

// One reusable c/fML pair, sized to the longest record in the chunk. This is
// the point of the whole arrangement: host matrix memory becomes
// (workers) x 125.6 MB at 5601nt instead of (records in chunk) x 125.6 MB,
// which at a VRAM-filling chunk width was about as much host RAM as VRAM.
typedef struct {
  int    *c;           /* T2a: persistent across chunks -- see bt_pool_get() */
  int    *fML;
  /* CIRCULAR: fM2_real, pooled for the same reason as the other two -- it is a
   * third full triangle per record, and holding one per RECORD instead of one
   * per WORKER would cost as much host RAM again as c and fML together. */
  int    *fM2;
  size_t  cells;   /* capacity, in ints, of each of the above */
  int     pin_c, pin_fML, pin_fM2;  /* which of the three are pinned (else malloc) */
  int     w;       /* T2a: this worker's copy stream */
  double  fetch_s; /* per-worker, so the timer needs no lock */
  double  busy_s;  /* likewise: this worker's active span, for attribution */
  double  pin_s;   /* T2a: time spent (re)allocating the pinned scratch */
} bt_scratch_t;

PRIVATE void
bt_scratch_free(bt_scratch_t *sc) {
  rnafold_pinned_ints_free(sc->c,   sc->pin_c);
  rnafold_pinned_ints_free(sc->fML, sc->pin_fML);
  rnafold_pinned_ints_free(sc->fM2, sc->pin_fM2);
  sc->c = sc->fML = sc->fM2 = NULL;
  sc->cells = 0;
}

PRIVATE void
bt_scratch_ensure(bt_scratch_t *sc, const size_t cells, const int want_fm2) {
  /* want_fm2 rather than always: fM2_real is a THIRD full triangle and a
   * linear fold must not pay for it. At 5601 nt that is 125.6 MB per worker,
   * which on a 12-worker pool is 1.5 GB of host RAM for a matrix nothing
   * would read. */
  if((sc->cells >= cells) && ((!want_fm2) || (sc->fM2)))
    return;

  /* T2a: NOT zeroed -- every cell is overwritten by the fetch that follows (the
   * copy is the record's whole triangle), so vrna_alloc()'s calloc was pure
   * cost. PINNED by default since 2026-09-17 (RNA_XFER_STAGE_MB=0): the copy
   * lands here directly, with no stage and no memcpy out of one. Under
   * RNA_XFER_STAGE_MB=N it is ordinary memory and the worker copies through an
   * N-MB pinned stage -- which is what WSL wants, where pinning 1.7 GB cost 35
   * worker-seconds. See rnafold_d2h_w() in device.cu. */
  const double t_pin = rnafold_now_seconds();
  const int    pin    = rnafold_xfer_pin_scratch();
  bt_scratch_free(sc);
  if(pin) {
    sc->c   = rnafold_pinned_ints(cells, &sc->pin_c);
    sc->fML = rnafold_pinned_ints(cells, &sc->pin_fML);
    sc->fM2 = want_fm2 ? rnafold_pinned_ints(cells, &sc->pin_fM2) : NULL;
  } else {
    sc->c   = (int *) malloc(sizeof(int) * cells);
    sc->fML = (int *) malloc(sizeof(int) * cells);
    sc->fM2 = want_fm2 ? (int *) malloc(sizeof(int) * cells) : NULL;
    sc->pin_c = sc->pin_fML = sc->pin_fM2 = 0;
  }
  if((!sc->c) || (!sc->fML) || (want_fm2 && (!sc->fM2))) {
    fprintf(stderr, "%-24s backtrack scratch: allocation of %zu ints failed\n",
            __FILE__, cells);
    exit(EXIT_FAILURE);
  }
  sc->cells = cells;
  sc->pin_s += rnafold_now_seconds() - t_pin;
}

/* The pool outlives the chunk: a 12-worker pool at 5601 nt is 1.5 GB, and
 * allocating and first-touching it per chunk is paid again for nothing.
 * Grow-only, and kept for the life of the process. */
static bt_scratch_t *g_bt_pool   = NULL;
static int           g_bt_pool_n = 0;

/* RAM budget (PORT_BUDGETS.md B3): the grow-only pool is right for throughput and wrong
 * under a budget -- a pool grown before the budget took effect would never shrink, and
 * a library caller would keep it all after its fold returned. So while a RAM budget is
 * in force the slots beyond `keep` are freed. Never while a trace is in flight: the
 * backtrack pipeline's trace thread uses the same pool. */
PRIVATE void
bt_pool_trim(const int keep) {
  if(g_bt_async.live)
    return;
  for(int t = keep; t < g_bt_pool_n; t++)
    bt_scratch_free(&g_bt_pool[t]);
  if(keep < g_bt_pool_n)
    g_bt_pool_n = keep;
}

/* engine.c, at the end of a batch: under a RAM budget the process does not keep its
 * backtrack scratch between calls. */
PUBLIC void
rnafold_bt_pool_release_under_budget(void) {
  if(vrna_budget_ram() != SIZE_MAX)
    bt_pool_trim(0);
}

PRIVATE bt_scratch_t *
bt_pool_get(const int n) {
  if(vrna_budget_ram() != SIZE_MAX)
    bt_pool_trim(n);
  if(n > g_bt_pool_n) {
    g_bt_pool = (bt_scratch_t *) vrna_realloc(g_bt_pool, sizeof(bt_scratch_t) * n);
    memset(g_bt_pool + g_bt_pool_n, 0, sizeof(bt_scratch_t) * (n - g_bt_pool_n));
    g_bt_pool_n = n;
  }
  for(int t=0; t<n; t++) {
    g_bt_pool[t].w       = t;
    g_bt_pool[t].fetch_s = 0.0;
    g_bt_pool[t].busy_s  = 0.0;
    g_bt_pool[t].pin_s   = 0.0;
  }
  rnafold_xfer_begin(n);
  return g_bt_pool;
}

/* Size every worker's scratch for the largest record it could be handed, HERE,
 * on one thread, before any worker starts.
 *
 * Page-locking serialises in the kernel: twelve workers pinning their own
 * scratch on first touch cost 29 worker-seconds for 1.7 GB, where doing it
 * serially costs the page-locking alone. It also takes the allocation out of
 * the backtrack phase's own timing, where it was being charged as if it were
 * per-record work. Grow-only and kept for the process, so a second chunk pays
 * nothing. */
PRIVATE void
bt_pool_prepare(bt_scratch_t *pool, const int n, const int nfiles,
                const vrna_fold_compound_t **VC) {
  size_t cells_max = 0;
  int    want_fm2  = 0;

  for(int i=0; i<nfiles; i++) {
    const size_t len   = (size_t)VC[i]->length;
    const size_t cells = (len + 1)*(len + 2)/2;
    if(cells > cells_max) cells_max = cells;
    if(VC[i]->params->model_details.circ) want_fm2 = 1;
  }
  if(cells_max == 0) return;

  for(int t=0; t<n; t++)
    bt_scratch_ensure(&pool[t], cells_max, want_fm2);
}

typedef struct {
  const vrna_fold_compound_t **VC;
  const char                 **Structure;
  int                         *energy;  /* now an OUTPUT: filled here, not by par_fill_arrays() */
  float                        *EN;
  int                           nfiles;
  int                          *next_i; /* shared work-claim counter */
  const size_t                *tri_off_H;
  /* N5a: when set, the triangles are fetched from this retired chunk, never
   * from the live globals; ret_id is the chunk the caller believes it is. */
  const rnafold_retired_t     *ret;
  int                           ret_id;
} backtrack_pool_args_t;

typedef struct {
  backtrack_pool_args_t *shared;
  bt_scratch_t          *sc;     /* this worker's own */
} backtrack_thread_arg_t;

// Post-processing for one record: give it a scratch pair, pull its two
// triangles off the GPU into that pair, finish the exterior loop, read the
// MFE, backtrack, then hand the scratch back. VC[]'s own c/fML pointers are
// NULL outside this window (par_mfe() releases them up front), which is what
// keeps peak host RAM at one record's worth per worker.
// `slot` is where the record's triangles live on the device and `idx` is what
// indexes VC/Structure/energy/EN. They are equal (one record per buffer, see
// backtrack_one()); they were separate for slot flow, deleted in the cleanup (C4).
PRIVATE void
backtrack_fetch_slot(backtrack_pool_args_t *a, const int idx, const int slot, bt_scratch_t *sc) {
  vrna_fold_compound_t *vc = (vrna_fold_compound_t *) a->VC[idx];
  const size_t lo    = a->tri_off_H[slot];
  // The record's own triangle: Indx(i,j)=j*(j-1)/2+i does not depend on length,
  // so the cell count is the record's, and must fit the buffer it was given.
  const size_t len   = (size_t)vc->length;
  const size_t cells = (len + 1)*(len + 2)/2;
  assert(vc->type == VRNA_FC_TYPE_SINGLE);
  assert(cells > 0);
  assert(cells <= a->tri_off_H[slot+1] - lo);  // the occupant must fit its slot

  bt_scratch_ensure(sc, cells, vc->params->model_details.circ);
  vc->matrices->c   = sc->c;
  vc->matrices->fML = sc->fML;
  /* CIRCULAR: fM2_real is allocated by upstream under ALLOC_CIRC and freed by
   * par_mfe() alongside c/fML, so it is reattached from the pool here exactly
   * like them. Only under md.circ -- otherwise the pointer stays whatever
   * upstream left it (NULL), and postprocess_circular() never runs. */
  if(vc->params->model_details.circ)
    vc->matrices->fM2_real = sc->fM2;

  const double t0 = rnafold_now_seconds();
  if(a->ret) {
    /* N5a: a retired chunk. The id check is what catches a set handed to the
     * wrong chunk's records once P2 has two sets alive at once. */
    if(a->ret->id != a->ret_id) {
      fprintf(stderr,"%-24s RETIRED SET MISMATCH: record %d expects chunk %d, "
                     "set is chunk %d\n", __FILE__, idx, a->ret_id, a->ret->id);
      exit(EXIT_FAILURE);
    }
    rnafold_retired_fetch_my_c_w(a->ret, sc->c,   lo, cells, sc->w);
    rnafold_retired_fetch_fml_w(a->ret,  sc->fML, lo, cells, slot, sc->w);
    if(vc->params->model_details.circ)
      rnafold_retired_fetch_fm2_w(a->ret, sc->fM2, lo, cells, sc->w);
  } else {
  fetch_my_c_one_w(sc->c,   lo, cells, sc->w);
  // _H, not the bare form: the int16 path needs the record index to find its
  // baselines. Widening happens here, at the device boundary -- the host's fML
  // is int32 and every consumer of it expects that.
  fetch_fML_one_Hw(sc->fML, lo, cells, slot, sc->w);
  /* CIRCULAR: the third triangle. Same slice, same offsets -- fM2_real shares
   * c/fML's layout exactly, which is why the device store needed no new
   * indexing. */
  if(vc->params->model_details.circ)
    fetch_fm2_one_w(sc->fM2, lo, cells, sc->w);
  }
  sc->fetch_s += rnafold_now_seconds() - t0;
}

// Phase 4b: the SECOND half of retiring a record -- everything after the two
// triangles have landed in the scratch. It is pure host work: the circ check,
// fM1, the exterior loop, the MFE read and the backtrack all read sc->c/sc->fML
// and make no CUDA call at all. That is what makes it safe to run on a worker
// thread. (It was split from the fetch for slot flow, where the fetch had to stay
// on the sweep's thread; slot flow is gone, the seam is kept.)
//
// `slot` is deliberately not a parameter -- it addresses the device, and
// nothing below this line does.
PRIVATE void
backtrack_finish_slot(backtrack_pool_args_t *a, const int idx, bt_scratch_t *sc) {
  vrna_fold_compound_t *vc  = (vrna_fold_compound_t *) a->VC[idx];
  const size_t          len = (size_t)vc->length;

  /*
   * RNA_CIRC_VERIFY (default off, O(n^3) -- small inputs only).
   *
   * Checks the device's fM2_real against its own DEFINITION using the fML this
   * same fetch just brought back:
   *
   *     fM2_real[i][j] == min over k in [i+1, j-2] of ( fML[i][k] + fML[k+1][j] )
   *
   * which is what mfe_multibranch_m2_fast() computes (mfe_multibranch.c:1205-1245).
   * No oracle and no second fold: it diffs the transported value against the
   * thing it is supposed to be, which is the self-comparison shape this project
   * keeps returning to. The k range is the part worth testing -- an off-by-one
   * at either end is a silent wrong answer, not a crash.
   */
  if((vc->params->model_details.circ) && (rnafold_circ_verify())) {
    const int   *idx  = vc->jindx;
    unsigned int i, j, bad = 0;

    for(i = 1; i + 1 <= len; i++) {
      for(j = i + 1; j <= len; j++) {
        int want = INF, kk;

        for(kk = (int)i + 1; kk + 2 <= (int)j; kk++) {
          const int a = sc->fML[idx[kk] + (int)i];
          const int b = sc->fML[idx[j] + kk + 1];
          if((a != INF) && (b != INF) && (a + b < want))
            want = a + b;
        }

        if(sc->fM2[idx[j] + (int)i] != want) {
          if(bad < 5)
            fprintf(stderr, "RNA_CIRC_VERIFY: fM2_real[%u][%u] device=%d "
                            "definition=%d\n", i, j,
                    sc->fM2[idx[j] + (int)i], want);
          bad++;
        }
      }
    }

    fprintf(stderr, "RNA_CIRC_VERIFY: record len %u -- %u cells disagree\n",
            len, bad);
    if(bad)
      exit(EXIT_FAILURE);
  }

  /*
   * fM1 under uniq_ML -- "multibranch loop part with exactly one branch".
   *
   * The sweep never computes it: fill_arrays.c prefills it to INF and no kernel
   * touches it, which is correct for the MFE because the recursion never reads
   * fM1. vrna_subopt() does read it, though, so a vrna_mfe_batch() caller that
   * folded with uniq_ML and then called subopt would have been reading a matrix
   * that was entirely INF. That is what kept uniq_ML declined in the library
   * routing guard.
   *
   * It costs no device work at all. fM1 is a pure function of `c` -- just
   * fetched into the scratch above -- and of its own earlier cells, so one host
   * pass here reconstructs it exactly. The cell-by-cell helper is upstream's
   * own, and the loop order is upstream's own too (mfe.c: i descending, j
   * ascending), so the dependency direction is inherited rather than reasoned
   * about.
   *
   * Cost is O(n^2) calls, each re-initialising a soft-constraint wrapper, and
   * it is paid ONLY when uniq_ML is set -- which for plain MFE folding it is
   * not. RNAfold sets it only for --ImFeelingLucky.
   */
  if ((vc->params->model_details.uniq_ML) &&
      (vc->matrices->fM1)) {
    int       *fM1  = vc->matrices->fM1;
    const int *indx = vc->jindx;
    int       i, j;

    for (i = 1; i <= (int)len; i++)
      fM1[indx[i] + i] = INF;

    for (i = (int)len - 1; i >= 1; i--)
      for (j = i + 1; j <= (int)len; j++)
        fM1[indx[j] + i] = vrna_mfe_multibranch_m1(vc, (unsigned int)i, (unsigned int)j);
  }

  vrna_mfe_exterior_f5(vc);          /* PORT: was E_ext_loop_5(); fills f5 from c */
  a->energy[idx] = vc->matrices->f5[vc->length];     /* was par_fill_arrays()'s job */

  sect bt_stack[MAXSECTORS]; /* thread-local */

  /*
   * CIRCULAR RNA. vrna_mfe() runs postprocess_circular() unconditionally on
   * md.circ right after the inside-engine branch, so a fold that goes through
   * upstream gets it free. vrna_mfe_batch() never returns through that path,
   * which is the whole reason -c was declined -- see
   * VRNA-PATCH(circular-postprocess).
   *
   * It REPLACES the energy (the exterior-loop f5 answer is the LINEAR one) and
   * SEEDS the backtrack stack itself, which is the part PORT_CIRC_SPEC.md
   * flagged as "not obviously mechanical": the backtrack must continue from a
   * pre-seeded stack rather than starting from f5[n]. That is why
   * vrna_backtrack_from_intervals_bps() takes a vrna_bts_t.
   */
  if(vc->params->model_details.circ) {
    vrna_bts_t cbt = vrna_bts_init(MAXSECTORS);

    a->energy[idx] = vrna_mfe_postprocess_circular(vc, cbt);
    a->EN[idx]     = callback_backtrack_bts(vc, a->energy[idx],
                                            a->Structure[idx], cbt);
    vrna_bts_free(cbt);
  } else
  a->EN[idx] = callback_backtrack(vc, 0, a->energy[idx], a->Structure[idx], bt_stack);

  // Detach before anything can free the compound: the scratch outlives it and
  // is reused, so leaving these set would hand vrna_mx_mfe_free() a pointer it
  // does not own. free(NULL) in there is a no-op.
  /* The scratch pair is pooled, so the record must let go of it. Normally it is
   * left with NULL c/fML -- par_mfe() freed its own before the sweep. Under
   * vrna_cuda_keep_matrices() it gets private copies instead, the extents upstream's
   * vrna_mfe() leaves, so a caller can backtrack or read the matrices afterwards. */
  if (vrna_cuda_keeping_matrices()) {
    const size_t cells = (len + 1) * (len + 2) / 2;
    vc->matrices->c   = (int *) vrna_alloc(sizeof(int) * cells);
    vc->matrices->fML = (int *) vrna_alloc(sizeof(int) * cells);
    memcpy(vc->matrices->c,   sc->c,   sizeof(int) * cells);
    memcpy(vc->matrices->fML, sc->fML, sizeof(int) * cells);
    if (vc->params->model_details.circ) {
      vc->matrices->fM2_real = (int *) vrna_alloc(sizeof(int) * cells);
      memcpy(vc->matrices->fM2_real, sc->fM2, sizeof(int) * cells);
    } else {
      vc->matrices->fM2_real = NULL;
    }
    return;
  }
  vc->matrices->c   = NULL;
  vc->matrices->fML = NULL;
  vc->matrices->fM2_real = NULL;   /* CIRCULAR: pooled, same reason */
}

// The two halves back to back: backtrack_all()'s workers each own one record
// start to finish, so there is nothing to hand over and no reason to split it.
PRIVATE void
backtrack_one_slot(backtrack_pool_args_t *a, const int idx, const int slot, bt_scratch_t *sc) {
  backtrack_fetch_slot(a, idx, slot, sc);
  backtrack_finish_slot(a, idx, sc);
}

/* The linear entry: an EMPTY interval stack, which is what the old sect[]
 * form always passed (every call site used s == 0). Circular seeds its own and
 * calls callback_backtrack_bts() directly. */
PRIVATE float
callback_backtrack(const vrna_fold_compound_t* vc,
                   int s,
                   const int energy,
                   char *structure,
                   sect bt_stack[MAXSECTORS]) {
  vrna_bts_t bts = vrna_bts_init(MAXSECTORS);
  float      mfe;

  /* s has always been 0 at every call site -- the sweep fills the matrices and
   * the backtrack starts from f5[n] -- so there is nothing to carry over. The
   * parameter stays for signature compatibility and is asserted, not trusted. */
  assert(s == 0);
  (void)s; (void)bt_stack;

  mfe = callback_backtrack_bts(vc, energy, structure, bts);
  vrna_bts_free(bts);

  return mfe;
}


PRIVATE void
backtrack_one(backtrack_pool_args_t *a, const int idx, bt_scratch_t *sc) {
  backtrack_one_slot(a, idx, idx, sc);   // one record per buffer: they are the same
}

PRIVATE void *
backtrack_worker(void *arg) {
  backtrack_thread_arg_t *t = (backtrack_thread_arg_t *) arg;
  const double t0 = rnafold_now_seconds();
  int idx;
  while((idx = __sync_fetch_and_add(t->shared->next_i, 1)) < t->shared->nfiles)
    backtrack_one(t->shared, idx, t->sc);
  // This worker's own active span, ending when it runs out of records to claim.
  // Paired with fetch_s below to split the phase's WALL between fetch and
  // backtracking -- see the attribution note at the end of the phase.
  t->sc->busy_s += rnafold_now_seconds() - t0;
  return NULL;
}

// Staggered_Row_Batching Phase 2a: see stub2.h for what row_off_H/tri_off_H
// mean. While every chunk RNAfold.c hands us is still built uniform-length
// (its flush trigger hasn't been relaxed yet), these tables are numerically
// identical to the old H*(length+1)/Hoff(H,length) formulas -- asserted
// below as a regression gate on the table-building logic itself, ahead of
// any real consumer switching over to it in a later phase.
// The one place the capacity rule lives: a record's buffers are sized to its
// own length. par_mfe() publishes the table (rnafold_chunk_capacities()) and
// par_fill_arrays() reads it, because both address the SAME device buffers.
PUBLIC void
compute_slot_capacities(const int nfiles, const vrna_fold_compound_t **VC,
                        const int length, size_t* cap_H) {
  (void)length;
  for(int H=0; H<nfiles; H++)
    cap_H[H] = (size_t)VC[H]->length;
}

PUBLIC void
compute_batch_offsets(const int nfiles, const size_t* cap_H,
                       size_t* row_off_H, size_t* tri_off_H) {
  // Built from the capacity table, which is VC[H]->length (compute_slot_capacities()).
  row_off_H[0] = 0;
  tri_off_H[0] = 0;
  for(int H=0; H<nfiles; H++) {
    const size_t len = cap_H[H];
    row_off_H[H+1] = row_off_H[H] + (len + 1);
    tri_off_H[H+1] = tri_off_H[H] + (len + 1)*(len + 2)/2;
  }
#ifndef NDEBUG
  {
    // Staggered_Row_Batching Phase 6c: this used to assert the tables matched
    // the old uniform H*(length+1) / H*(length+1)*(length+2)/2 formulas
    // exactly -- a deliberate regression gate on the table logic, valid only
    // while every chunk was single-length. Chunks are now genuinely mixed, so
    // that gate would fire on correct input. Replaced with the invariant that
    // actually still holds: each H's block is exactly its own length's worth,
    // and both tables are strictly increasing prefix sums.
    for(int H=0; H<nfiles; H++) {
      const size_t len = cap_H[H];
      assert(row_off_H[H+1] - row_off_H[H] == len + 1);
      assert(tri_off_H[H+1] - tri_off_H[H] == (len + 1)*(len + 2)/2);
      assert(row_off_H[H+1] > row_off_H[H]);
      assert(tri_off_H[H+1] > tri_off_H[H]);
    }
  }
#endif
}

// Staggered_Row_Batching Phase 4: see stub2.h for the full explanation of
// why this is declared there (near compute_batch_offsets()) and defined
// here rather than inline next to flatten_index_to_H() -- fill_arrays_loop.c
// calls this from a plain .c translation unit that never sees anything
// inside stub2.h's __CUDACC__ guard, so this function (and its self-check)
// can't reference flatten_index_to_H() the way an in-.cu-file caller could.
// Debug-build self-check: monotonic non-decreasing (guaranteed by width_H
// being size_t, i.e. non-negative, but cheap to assert explicitly) and the
// final total matches an independently-recomputed sum -- catches a
// duplicated/dropped term, the main way a loop like this actually breaks.
// flatten_index_to_H()'s own binary-search correctness was already verified
// separately (a standalone throwaway test, 8 cases, every flat index
// cross-checked against a brute-force reference) when it was first written.
PUBLIC void
compute_flatten_offsets(const int nfiles, const size_t* width_H, size_t* flat_off_H) {
  flat_off_H[0] = 0;
  for(int H=0; H<nfiles; H++) flat_off_H[H+1] = flat_off_H[H] + width_H[H];
#ifndef NDEBUG
  {
    size_t independent_total = 0;
    for(int H=0; H<nfiles; H++) {
      assert(flat_off_H[H+1] >= flat_off_H[H]);
      independent_total += width_H[H];
    }
    assert(flat_off_H[nfiles] == independent_total);
  }
#endif
}

// The post-sweep phase: every record's triangles fetched, finished and
// backtracked. Everything it needs is an argument, so N5a P2 can run it on a
// trace thread for the previous chunk.
PRIVATE void
backtrack_all(const int nfiles, const vrna_fold_compound_t **VC,
              const char **Structure, int *energy, float *EN,
              const size_t *tri_off_H, const int cpu_queue_threads,
              const rnafold_retired_t *ret) {
  int i;
  // Post-processing. Each record now gets its triangles fetched into a
  // pooled scratch pair, its exterior loop finished and its MFE read, then
  // backtracks, then releases the scratch -- see backtrack_one(). One pool
  // slot per worker, so peak host matrix memory is n_bt_threads records'
  // worth rather than nfiles.
  const double t_bt = rnafold_now_seconds();
  /* RAM budget (PORT_BUDGETS.md B3, decision 2): backtrack workers are shed first. What
   * the process holds is MEASURED (its RSS: base, this chunk's compounds, memory the
   * allocator kept), less the scratch pool it is about to reuse, plus what the driver
   * is still adding beside it (a build pipeline's next chunk). */
  size_t max_len = 0;
  for(int H = 0; H < nfiles; H++)
    if((size_t)VC[H]->length > max_len) max_len = VC[H]->length;
  size_t need_now = 0;
  if(vrna_budget_ram() != SIZE_MAX) {
    size_t pool = 0;
    for(int t = 0; t < g_bt_pool_n; t++)
      pool += g_bt_pool[t].cells * sizeof(int) *
              ((g_bt_pool[t].c ? 1 : 0) + (g_bt_pool[t].fML ? 1 : 0) + (g_bt_pool[t].fM2 ? 1 : 0));
    const size_t rss = vrna_budget_rss();
    need_now = ((rss > pool) ? rss - pool : 0) + vrna_budget_held_total();
  }
  const char *bt_env      = getenv("RNA_BACKTRACK_THREADS");
  const int   bt_explicit = (bt_env && bt_env[0] && strcmp(bt_env, "auto")) ? 1 : 0;
  /* B4: the thread budget caps the pool, the RAM budget may shed more, and the workers
   * are then DRAWN from the shared count -- a build pipeline's builders or a trace
   * thread's workers may hold some of it right now. */
  const int n_bt_want =
    vrna_budget_workers(vrna_budget_cap_threads("backtrack workers",
                                                backtrack_thread_count(nfiles, cpu_queue_threads),
                                                bt_explicit),
                        need_now, max_len, VC[0]->params->model_details.circ, bt_explicit);
  int bt_held;
  const int n_bt_threads = vrna_budget_threads_acquire(n_bt_want, bt_explicit, &bt_held);
  int next_i = 0;
  backtrack_pool_args_t targ = { VC, Structure, energy, EN, nfiles, &next_i, tri_off_H,
                                 ret, ret ? ret->id : -1 };
  bt_scratch_t *pool = bt_pool_get(n_bt_threads);   /* T2a: persistent, maybe pinned */
  bt_pool_prepare(pool, n_bt_threads, nfiles, VC);  /* serially, before any worker */
  if(n_bt_threads <= 1) {
    const double t_serial = rnafold_now_seconds();
    for(i=0;i<nfiles;i++) backtrack_one(&targ, i, &pool[0]);
    pool[0].busy_s += rnafold_now_seconds() - t_serial;
  } else {
    pthread_t *bt_threads = (pthread_t *) vrna_alloc(sizeof(pthread_t) * n_bt_threads);
    backtrack_thread_arg_t *targs =
      (backtrack_thread_arg_t *) vrna_alloc(sizeof(backtrack_thread_arg_t) * n_bt_threads);
    for(int t=0; t<n_bt_threads; t++) { targs[t].shared = &targ; targs[t].sc = &pool[t]; }
    int started = 0;
    for(; started < n_bt_threads; started++) {
      if(pthread_create(&bt_threads[started], NULL, backtrack_worker, &targs[started]) != 0) {
        fprintf(stderr, "mfe_cuda.c: pthread_create failed for backtrack worker %d, "
                         "continuing with fewer threads\n", started);
        break;
      }
    }
    if(started == 0) {
      /* pthread_create failed on the very first thread -- fall back to serial */
      const double t_serial = rnafold_now_seconds();
      for(i=0;i<nfiles;i++) backtrack_one(&targ, i, &pool[0]);
      pool[0].busy_s += rnafold_now_seconds() - t_serial;
    } else {
      for(int t=0; t<started; t++)
        pthread_join(bt_threads[t], NULL);
    }
    free(targs);
    free(bt_threads);
  }
  vrna_budget_threads_release(bt_held);
  // The fetch is timed per worker so it needs no lock. Fold the slots in and
  // charge it to transfer rather than to backtracking, which it dwarfs. Note
  // the local, not the global: phase_fetch_mx_s accumulates across chunks.
  //
  // FIXED 2026-08-27, after the first RNA_BACKTRACK_THREADS=auto run on Colab
  // printed `backtrack=-87.783` and `fetch_mx=101.445` against a phase that
  // actually took 13.66 s of wall. pool[].fetch_s is WORKER-seconds: with W
  // workers running concurrently their sum can exceed the phase's wall clock
  // outright, and this used to subtract that sum straight from the wall --
  // worker-seconds minus wall-seconds, a category error that goes negative as
  // soon as W > 1. (The wall was still right; only the split of it was not.)
  // So split the phase's WALL in the ratio the workers actually spent it.
  // Both parts are then non-negative and sum to the phase exactly, and with a
  // single worker busy_worker_s IS the phase, so this reduces to the old
  // subtraction and leaves every previously-recorded number unchanged.
  double fetch_worker_s = 0.0, busy_worker_s = 0.0;
  for(int t=0; t<n_bt_threads; t++) {
    fetch_worker_s += pool[t].fetch_s;
    busy_worker_s  += pool[t].busy_s;
    stage_btpin_worker_s   += pool[t].pin_s;
    stage_btfetch_worker_s += pool[t].fetch_s;
  }
  g_bt_fetch_ws = fetch_worker_s;   /* N5a P0 chunk spans */
  g_bt_busy_ws  = busy_worker_s;
  g_bt_threads  = n_bt_threads;
  /* the pool is kept: see bt_pool_get() */
  const double bt_phase_s = rnafold_now_seconds() - t_bt;
  double fetch_this_chunk = (busy_worker_s > 0.0)
                          ? bt_phase_s * (fetch_worker_s / busy_worker_s)
                          : 0.0;
  if(fetch_this_chunk > bt_phase_s) fetch_this_chunk = bt_phase_s;  /* can't exceed the phase */
  phase_fetch_mx_s  += fetch_this_chunk;
  stage_backtrack_s += bt_phase_s - fetch_this_chunk;
}

//except par_fill_arrays(), do each file sequentially as before
PUBLIC void
par_mfe(const int nfiles,
	const vrna_fold_compound_t** VC,
	const char** Structure,
	float* EN, //out
	const int cpu_queue_threads) {

  /* N5a P0 chunk spans; see rnafold_chunk_trace_open() */
  const double tr_enter = rnafold_now_seconds();
  double       tr_init  = 0.0, tr_sweep = 0.0, tr_bt = 0.0;
  int          bt_handed = 0;   /* N5a P2: this call's trace went to the trace thread */

  //start GPU as early as possible so allow maximise overlap GPU with CPU
  // Staggered_Row_Batching Phase 6c/6d: the length handed to init_gpu*() sizes
  // every whole-batch device buffer that is not already table-driven -- most
  // importantly int_loop.cu's d_S, whose 10-bases-per-word packing is still
  // laid out as nfiles*(length+2) (Phase 2f, deliberately deferred). Taking
  // VC[0]'s length here was correct only while chunks were single-length; with
  // mixed lengths it under-sizes d_S whenever VC[0] is not the longest record,
  // and the bases past VC[0]->length of every longer sequence are then never
  // packed -- which surfaces as unpack()'s `out>=0 && out<=4` device assert
  // rather than as a wrong answer. Must be the batch maximum, matching
  // par_fill_arrays()'s sweep bound.
  int length = 0;
  for(int H=0; H<nfiles; H++)
    if((int)VC[H]->length > length) length = (int)VC[H]->length;
  const vrna_md_t* md = &(VC[0]->params->model_details);
  const int turn      = md->min_loop_size;
  // Vet the parameter table against the int16 bound HERE, before init_gpu(),
  // because init_gpu() is where the mode is COMMITTED: it allocates d_fml_j16
  // instead of d_fml_j, and after that a decline cannot be honoured. Doing this
  // in load_param() alone was too late by exactly one call -- the decline
  // printed, and the already-allocated int16 path packed anyway and hit the
  // pack kernel's trap. load_param() still calls it, harmlessly, as the second
  // line of defence for any entry point that does not come through here.
  {
    const vrna_param_t *P = VC[0]->params;
    int worst = 0;
    for(int a=0; a<=NBPAIRS; a++)
      for(int b=0; b<=NBPAIRS; b++) {
        const int v = P->stack[a][b];
        if(v < worst && v > -INF/2) worst = v;
      }
    rnafold_fml_int16_vet_params(worst, FML_BLK);
  }

  /* AND noLP, for the same reason and in the same place. noLP puts near-INF
   * FINITE values into fML (c[ij] takes cc1[j-1]+stackEnergy, which is INF at the
   * top of the triangle, and the asymmetric INF guard turns that into ~1e7 rather
   * than INF) -- so a block whose first-written entry is one of those gets a 1e7
   * baseline and ordinary energies no longer fit int16. The provable bound holds;
   * its PREMISE that fML carries no near-INF finite cells is what noLP breaks.
   *
   * This used to exit() from par_fill_arrays, which was correct while int16 was
   * opt-in and became a REGRESSION the moment it became the default: --noLP is an
   * ordinary user option and it stopped working. It has to be resolved HERE,
   * before init_gpu() commits the layout, because par_fill_arrays runs after the
   * buffers are already int16. */
  if (md->noLP)
    rnafold_fml_int16_stand_down("--noLP puts near-INF finite values into fML, "
                                 "which the 16-bit per-block offsets cannot "
                                 "represent");

  /* A NON-DEFAULT SALT USED TO STAND INT16 DOWN, and no longer does (2026-10-02).
   *
   * 069981ff added the stand-down because verify_option_parity.sh's salt_hi arm once
   * gave a different answer under int16, mechanism unknown. It never reproduced:
   *
   *   - HEAD, laptop: 255 arms of -g and salt (0.05..5.0 M, with -c, -T, -d0, -g,
   *     --helical-rise) x 15 knob arms, int16 forced -- all equal to the CPU, and the
   *     c triangle identical to int32's;
   *   - 069981ff ITSELF with the stand-down patched out: its own parity script on its
   *     own fixture passes, and 120 repeated runs (plain + salt 1.5/0.2) are 0 wrong;
   *   - A100 (CUDA_RNAFold_DeadWarps.ipynb, section S): 4 salts x 2 fixtures x
   *     overlap 0/1/2 x 3 = 72/72 equal to the CPU and to int32's c triangle.
   *
   * Salt also breaks no premise of the encoding: RNA_TRI_CHECKSUM counts 0 near-INF
   * finite cells under every salt (-C --enforceConstraint: 10 845, so the counter
   * sees them). The likeliest past cause is the md_snapshot_dml race 7982a147 fixed
   * after 069981ff, which sat on every run's path -- unconfirmed, it did not show in
   * those 120 runs either. If a salt run ever differs again, RNA_TRI_CHECKSUM's c hash
   * against RNA_FML_INT16=0 is the first thing to compare. */

  /* AND HARD CONSTRAINTS -- a SHIPPED crash, found 2026-09-30 by the Python
   * binding's GPU default and then reproduced on the CLI:
   *
   *     RNAfold -C --enforceConstraint  (8 x 700 nt, one enforced (1,700) pair)
   *     RNA_FML_INT16 range: H=0 (i=1,j=34) value 9999960 baseline 480 ... ABORTING
   *
   * Enforcing a pair forbids everything that conflicts with it, and the forbidden
   * cells reach fML as near-INF FINITE values -- the same premise noLP breaks, so
   * the 16-bit per-block offsets cannot hold them and pack_fml_cell() traps. Live
   * since int16 became the default (069981ff); under WSL the trap does not even
   * abort, it wedges the process in the driver.
   *
   * Standing down on ANY depot is conservative: it is not yet known which constraint
   * shapes are safe, and int32 is always right. Unlike salt (above), -C has a FOUND
   * mechanism, so this one stays until the safe shapes are known --
   * finding the precise condition belongs in INT16_FML_SCOPE.md. */
  for (int H = 0; H < nfiles; H++)
    if ((VC[H]->hc != NULL) && (VC[H]->hc->depot != NULL)) {
      rnafold_fml_int16_stand_down("hard constraints put near-INF finite values into "
                                   "fML (an enforced pair forbids its conflicts), "
                                   "which the 16-bit per-block offsets cannot "
                                   "represent");
      break;
    }
  // Every LAYOUT table below is built from the capacity table; every BOUND
  // stays on the record's own VC[H]->length. The two are the same number.
  size_t cap_H[nfiles];
  compute_slot_capacities(nfiles, VC, length, cap_H);
  // Publish it: par_fill_arrays() reads this instead of recomputing -- see
  // rnafold_chunk_capacities().
  if(g_cap_n < nfiles) {
    free(g_cap_H);
    g_cap_H = (size_t *) vrna_alloc(sizeof(size_t) * nfiles);
    g_cap_n = nfiles;
  }
  memcpy(g_cap_H, cap_H, sizeof(size_t) * nfiles);
  size_t row_off_H[nfiles+1], tri_off_H[nfiles+1]; //Phase 2a, see compute_batch_offsets() above
  compute_batch_offsets(nfiles, cap_H, row_off_H, tri_off_H);
  /* Can the device DERIVE hc->mx and ptype from the sequence, instead of the
   * host packing them in O(n^2) per record? That packing is 87% of gpuinit and
   * gpuinit is 22.9% of wall at 400x5601 (docs/archive/STRESS272_RESULTS.md), so this is the
   * largest single lever in the port.
   *
   * DERIVED HERE RATHER THAN SET BY THE CALLER, deliberately. On 2.3.0 this was
   * an assignment in RNAfold.c; the 2.7.2 port dropped that line and nothing
   * noticed, because a lost assignment to a default-0 flag is invisible -- it
   * only makes you slower. Anything a caller must remember can be forgotten in
   * exactly that silent direction, so the library works it out from the batch it
   * was handed.
   *
   * It is sound because the LIBRARY GUARD already refuses everything that could
   * perturb hc->mx after the default reset: hard constraints (hc->depot), soft
   * constraints, SHAPE/probing, ligand motifs and command files are all declined
   * by vrna_cuda_engine_supports() before a compound can reach this function.
   * That leaves noLP, which IS accepted and which zeroes ptype entries for
   * isolated pairs -- so the masks stop being a pure function of the sequence
   * and the host loops have to run. sanity() already asserts noLP agrees across
   * the batch; assert the rest of the precondition here rather than trusting the
   * guard from a distance, because this is the file that would be wrong.
   *
   * Correctness bar is RNA_HC_VERIFY=1, which builds both ways and compares
   * every word of all four masks -- 532928 words x 4, zero mismatching. */
  {
    const vrna_md_t *md0 = &(VC[0]->params->model_details);
    int h;
    int derivable = !md0->noLP;

    /* A DEPOT IS THE OTHER WAY hc->mx STOPS BEING A FUNCTION OF THE SEQUENCE,
     * and this predicate did not ask. It read noLP alone and leaned on the
     * routing guard to have refused everything else -- which was true, and is
     * exactly the kind of precondition that is held from a distance until
     * somebody relaxes the guard. Asked directly now, per record: anything
     * carrying a hard-constraint depot or a soft constraint packs the host way.
     * Slower, never wrong, and it makes the derive path safe to leave on. */
    for(h = 0; h < nfiles; h++) {
      if(VC[h] == NULL) continue;
      if(VC[h]->sc != NULL) derivable = 0;
      if((VC[h]->hc != NULL) && (VC[h]->hc->depot != NULL)) derivable = 0;
      assert(VC[h]->params->model_details.noLP == md0->noLP);
    }

    g_hc_seq_derived = derivable;
  }

  /* PREPARE BEFORE ANYTHING READS hc->mx OR ptype -- and this ordering is the
   * whole of what -C needed.
   *
   * vrna_constraints_add() only QUEUES: hc->depot holds the constraint and
   * hc->mx, ptype and the up_* arrays stay byte-identical to an unconstrained
   * compound until vrna_fold_compound_prepare() materialises them. This loop
   * used to run AFTER init_gpu3(), which packs the device's hard-constraint
   * masks -- so the masks were built from the UNCONSTRAINED matrix.
   *
   * For every model the guard admits today that is invisible, because the
   * pre-prepare hc->mx already equals the final one. Under -C it is a wrong
   * answer: measured 2026-09-16 with RNA_ENGINE_ALLOW=hc, the device came back
   * BETTER THAN LEGAL on 8 of 8 records and its structures re-evaluated to a
   * different energy than it reported.
   *
   * And RNA_HC_VERIFY reported "0 mismatching words" on exactly those folds,
   * because it too rebuilds from hc->mx at pack time: it was comparing two
   * unconstrained matrices. The verifier only means something from here on.
   */
  if(VC[0]->type == VRNA_FC_TYPE_SINGLE) {
    int i;
    const double t_prepare = rnafold_now_seconds();
    for(i=0;i<nfiles;i++) {
      if(!vrna_fold_compound_prepare(VC[i], VRNA_OPTION_MFE)){
	vrna_message_warning("vrna_mfe@mfe.c: Failed to prepare vrna_fold_compound");
	EN[i] = INF/100.0F;
	exit(1);
      }
      /* call user-defined recursion status callback function */
      if(VC[i]->stat_cb){ VC[i]->stat_cb(VC[i], VRNA_STATUS_MFE_PRE, VC[i]->auxdata);}  /* PORT: 2.7.2 signature */
    }
    stage_prepare_s += rnafold_now_seconds() - t_prepare;
  }


  const double t_gpuinit = rnafold_now_seconds();
  init_gpu(nfiles,length,tri_off_H,row_off_H);
  init_gpu2(nfiles,VC, turn, length, 512, tri_off_H, row_off_H, cap_H);
  init_gpu3(nfiles,VC, turn, length, 512, row_off_H, cap_H);
  /* RNA_STREAM_OVERLAP: create the row's streams and events once per chunk,
   * after the device exists and before any row is issued. A no-op at the
   * default, where every launch stays on the NULL stream. */
  rnafold_streams_init();
  tr_init = rnafold_now_seconds();
  stage_gpuinit_s += tr_init - t_gpuinit;
  rnafold_vram_sample();   /* RNA_VRAM_BUDGET: this chunk's buffers are all allocated */

  if(VC[0]->type == VRNA_FC_TYPE_SINGLE) {
    int i;
    /*
     * G-quadruplex stage G0: carry c_gq to the device.
     *
     * Here and not earlier, because c_gq is created by
     * vrna_fold_compound_prepare() -- dp_matrices.c:547, gated on md.gquad --
     * which now runs above init_gpu3() rather than below it, so that the
     * hard-constraint masks are packed from a materialised hc->mx. Here and not later, because the c/fML free below does not
     * touch c_gq but everything after this is the sweep.
     *
     * A no-op today: nothing reads the table until G1/G2, and -g is still
     * declined by the routing guard, so this only ever fires for a caller that
     * reached vrna_mfe_batch() with gquad set some other way. It is wired in
     * now so the transfer has its own bar (tests/mfe_cuda_gquad.ts) separately
     * from the recursion work -- if the table does not arrive intact, every
     * later stage debugs the wrong thing.
     */
    (void)rnafold_gq_upload(nfiles, VC);

    // Release every record's c/fML now. Nothing reads either between here and
    // backtrack_one(), which reattaches a pooled scratch pair per record --
    // new_c_host stopped writing My_c in a1430bd, fml_host stopped writing
    // My_fML in 89e5721, and the dead prefill went in 4b2a18b, so the sweep
    // never touches them. At 5601nt this is 125.6 MB per record reclaimed,
    // roughly 1:1 with the record's VRAM, and it is what stops host RAM being
    // as binding a constraint on chunk width as the GPU's own memory.
    for(i=0;i<nfiles;i++) {
      vrna_fold_compound_t *v = (vrna_fold_compound_t *) VC[i];
      free(v->matrices->c);   v->matrices->c   = NULL;
      free(v->matrices->fML); v->matrices->fML = NULL;
    }

    int energy[nfiles];
    par_fill_arrays(nfiles,VC,energy);
    rnafold_vram_sample();   /* and at the sweep's end, beside any retained chunk */
    if(g_chunk_fp) {
      rnafold_device_drain();   /* sweep_t1 = the device finished, not the last issue */
      tr_sweep = rnafold_now_seconds();
    }

    /* N5a P2: one backtrack at a time (the scratch pool is shared), so the
     * previous chunk's trace is joined HERE -- after this sweep, which it has
     * been running beside, and before anything of this chunk's is traced. */
    bt_async_join(0);

    if(rnafold_bt_pipeline()) {
      /* N5a P1: retire, then backtrack from the retired set. */
      rnafold_retired_t *ret = rnafold_retire_chunk(nfiles, tri_off_H);
      bt_async_t        *a   = &g_bt_async;

      const int to_driver = (g_bt_async_ok == RNAFOLD_BT_ASYNC_DRIVER) && g_bt_done_cb &&
                            g_bt_next_ctx;
      const int internal  = (g_bt_async_ok == RNAFOLD_BT_ASYNC_INTERNAL);

      if(to_driver || internal) {
        /* P2: hand the trace off and return; the caller sweeps the next chunk. P3: the
         * caller is engine.c, which sweeps its next sub-chunk and joins this one before
         * vrna_mfe_batch() returns. */
        a->ret               = ret;
        a->nfiles            = nfiles;
        a->length            = length;
        a->cpu_queue_threads = cpu_queue_threads;
        a->cells             = tri_off_H[nfiles];
        a->VC                = VC;
        a->Structure         = Structure;
        a->EN                = EN;
        a->energy            = (int *) vrna_alloc(sizeof(int) * nfiles);
        a->tri_off_H         = (size_t *) vrna_alloc(sizeof(size_t) * (nfiles + 1));
        memcpy(a->tri_off_H, tri_off_H, sizeof(size_t) * (nfiles + 1));
        a->cb                = to_driver ? g_bt_done_cb : NULL;
        a->ctx               = to_driver ? g_bt_next_ctx : NULL;
        a->enter             = tr_enter;
        a->init_t1           = tr_init;
        a->sweep_t1          = tr_sweep;
        a->exit_t            = 0.0;

        if(pthread_create(&a->th, NULL, bt_async_main, a) == 0) {
          a->live         = 1;
          bt_handed       = 1;
          if(rnafold_bt_negctl() == 2)   /* negative control: give the set away */
            rnafold_retired_negctl_free(ret);
          if(to_driver) {
            g_bt_next_ctx   = NULL;
            g_bt_went_async = 1;
          } else {
            g_bt_internal_n++;
          }
          g_bt_async_n++;
        } else {
          /* no thread to be had: P1's sequential order, still correct */
          free(a->energy);
          free(a->tri_off_H);
        }
      }

      if(!a->live || (a->ret != ret)) {
        backtrack_all(nfiles, VC, Structure, energy, EN, tri_off_H, cpu_queue_threads, ret);
        rnafold_retired_release(ret);
        g_bt_retired_n++;
      }
    } else {
      backtrack_all(nfiles, VC, Structure, energy, EN, tri_off_H, cpu_queue_threads, NULL);
    }
    if(g_chunk_fp)
      tr_bt = rnafold_now_seconds();

    if(bt_handed) {
      /* handed off: the line is written when the trace is joined */
      g_bt_async.exit_t = rnafold_now_seconds();
    } else if(g_chunk_fp) {
      chunk_trace_line(nfiles, length, tri_off_H[nfiles], tr_enter, tr_init, tr_sweep, tr_bt,
                       rnafold_now_seconds(), g_bt_threads, g_bt_fetch_ws, g_bt_busy_ws);
    }
  } else {
  /* PORT TO 2.7.2 -- WIRED THROUGH THE SEAM.
   *
   * This used to call mfe_cuda_vrna_mfe(), this file's private copy of
   * upstream vrna_mfe(). It now calls the real one. That is the whole point of
   * the inside-engine seam (vrna_gr_set_inside_engine, ViennaRNA/grammar/mfe.h):
   * a record is folded by upstream's own vrna_mfe(), which consults whatever
   * engine is attached to the fold compound and otherwise fills the matrices
   * itself. Circular RNA, comparative fold compounds, backtracking and output
   * are upstream's again rather than reimplemented here.
   *
   * Nothing is lost by the swap: the copy's own header comment records that it
   * was reachable only for VRNA_FC_TYPE_COMPARATIVE and "NOT safe to call for
   * VRNA_FC_TYPE_SINGLE".
   */
  for(int i=0;i<nfiles;i++) {
    EN[i] = vrna_mfe(VC[i], Structure[i]);
  }}
}


/**
*** fill "c", "fML" and "f5" arrays and return  optimal energy
**/
#include "fill_arrays.c"




/* New Jul 2026: removed this file's own vrna_backtrack_from_intervals --
 * confirmed dead code (nothing in this file called it; callback_backtrack
 * below uses its own backtrack()/backtrack_comparative() directly), and its
 * externally-linked duplicate of mfe.c's real vrna_backtrack_from_intervals
 * (same name, same signature) caused a "multiple definition" link error
 * once mfe.o started getting pulled into RNAfold's link (for
 * vrna_mfe_cpu(), added for the CPU worker queue -- see mfe.c). The real
 * one (mfe.c) is still reachable via -lRNA for anything that needs it. */

/**
*** trace back through the "c", "f5" and "fML" arrays to get the
*** base pairing list. No search for equivalent structures is done.
*** This is fast, since only few structure elements are recalculated.
***
*** normally s=0.
*** If s>0 then s items have been already pushed onto the bt_stack
**/


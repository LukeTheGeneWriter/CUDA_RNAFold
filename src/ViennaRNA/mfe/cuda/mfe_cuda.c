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
#include "ViennaRNA/constraints/soft.h"
#include "ViennaRNA/eval/gquad.h"
#include "ViennaRNA/mfe/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/loops/all.h"
#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/backtrack/global.h"   /* PORT: vrna_backtrack_from_intervals() */
#include "ViennaRNA/mfe/multibranch.h"    /* vrna_mfe_multibranch_m1(), for fM1 */

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
// Continuous flow phase B. OFF by default: with it off every record sits on
// the same row as today and the kernels' assert(i == i_row) self-proof stays
// live, so one binary can be A/B'd against itself -- the same shape that
// settled RNA_GPU_SWEEP.
//
// ON, each record starts at ITS OWN top row on the first iteration instead of
// waiting for the shared counter to come down to it, and retires when it
// reaches row 1 instead of the whole batch finishing together. Same total work
// (each record still computes exactly its own triangle, one row per iteration,
// in the same order); what changes is WHEN each record's rows are computed, so
// the early iterations stop being nearly empty. The answers must not move --
// that is the verification bar.
PUBLIC int
rnafold_continuous_flow(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_CONTINUOUS_FLOW");
    v = (e && e[0] && strcmp(e,"0")) ? 1 : 0;
    // Phase C3: slot flow IMPLIES continuous flow -- a slot can only take its
    // next record early if records are on their own rows. This has to be
    // decided HERE and not just inside the sweep, because RNA_I_ROW() asks this
    // function whether there is still a single row for the kernels to assert
    // against. Getting that wrong trips the assert rather than folding wrong,
    // which is how it was caught.
    // k=1 is the degenerate schedule -- one record per slot, no turnover -- and
    // still needs per-record rows, so the implication starts at 1, not 2. That
    // makes k=1 a test of the schedule plumbing in isolation from the handover.
    if(!v && rnafold_slot_flow() >= 1) v = 1;
    if(v) fprintf(stderr,"%-24s RNA_CONTINUOUS_FLOW=1: records start at their own "
                         "top row and retire at their own last row\n", __FILE__);
  }
  return v;
}

// Continuous flow phase C1. A SLOT's capacity and its OCCUPANT's length are the
// same number today, and the code uses them interchangeably. They stop being the
// same the moment a slot outlives its first occupant (phase C2), so C1 separates
// them while they still agree -- which is what makes the separation provable.
//
// RNA_SLOT_CAPACITY=max sizes every slot to the chunk maximum instead of its own
// record's length. It WASTES VRAM and is not a mode anyone should run for real;
// it exists as the end-to-end proof that a record folds correctly in an
// over-sized slot, which is the load-bearing assumption of all of phase C. It
// must stay byte-identical.
PUBLIC int
rnafold_slot_capacity_max(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_SLOT_CAPACITY");
    v = (e && (e[0]=='m'||e[0]=='M')) ? 1 : 0;
    if(v) fprintf(stderr,"%-24s RNA_SLOT_CAPACITY=max: every slot sized to the chunk "
                         "maximum, not to its own record (test mode, wastes VRAM)\n", __FILE__);
  }
  return v;
}

// Continuous flow phase C2: SLOT TURNOVER self-check (RNA_SLOT_TURNOVER=1).
// After a chunk folds normally, every record is re-admitted into a DIFFERENT
// slot -- record r moves to slot r-1, cyclically -- and the chunk is folded a
// second time. Slot capacities were sized for both occupants up front, so most
// records end up in a slot LARGER than themselves, which is the arrangement all
// of phase C depends on. Each slot's second result must equal the first result
// of the record that moved into it.
//
// It doubles the work and discards the second answer, so it is a test mode, not
// a feature. What it proves is the thing C3 needs and cannot assume: that
// refilling a slot leaves NO trace of its previous occupant.
PUBLIC int
rnafold_slot_turnover(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_SLOT_TURNOVER");
    v = (e && e[0] && strcmp(e,"0")) ? 1 : 0;
    if(v) fprintf(stderr,"%-24s RNA_SLOT_TURNOVER=1: every chunk is folded twice, the "
                         "second time with every record in a different slot (test mode)\n",
                  __FILE__);
  }
  return v;
}

// Continuous flow phase C3: RNA_SLOT_FLOW=k folds the chunk in ceil(nfiles/k)
// SLOTS instead of one slot per record. Each slot runs a queue of records back
// to back: as soon as one reaches its last row the slot fetches it, backtracks
// it, and takes the next -- mid-sweep, while every other slot keeps going.
//
// k=2 halves the slot count (and so the VRAM) for the same records; k=1 or unset
// is today's behaviour. The output must not change, only the schedule.
PUBLIC int
rnafold_slot_flow(void) {
  static int v = -1;
  if(v < 0) {
    const char *e = getenv("RNA_SLOT_FLOW");
    v = (e && e[0]) ? atoi(e) : 0;
    if(v < 0) v = 0;
    if(v >= 1) fprintf(stderr,"%-24s RNA_SLOT_FLOW=%d: chunks run in 1/%d as many slots, "
                             "each slot taking its next record mid-sweep\n", __FILE__, v, v);
  }
  return v;
}

// The chunk's slot capacities, owned here because par_fill_arrays() must see
// EXACTLY the table par_mfe() allocated the buffers from. Recomputing them
// there would be wrong under turnover: a slot's capacity covers both of its
// occupants, so it depends on the whole chunk, not on the pass's own records.
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
int g_hc_seq_derived = 0;

double rnafold_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

PRIVATE void
print_stage_timing_stats(void) {
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
  fprintf(stderr,
    "%-24s phase timing (s): int_loop=%.3f hp_mb=%.3f load_my_c=%.3f "
    "modular_decomp=%.3f fetch_mx=%.3f | new_c_host=%.3f fml_host=%.3f fml_prev_host=%.3f "
    "|| GPU+transfer total=%.3f host-combine total=%.3f\n",
    __FILE__, phase_int_loop_s, phase_hp_mb_s, phase_load_my_c_s,
    phase_modular_decomp_s, phase_fetch_mx_s, phase_new_c_host_s, phase_fml_host_s,
    phase_fml_prev_host_s, gpu_transfer_total, host_combine_total);
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
callback_backtrack(const vrna_fold_compound_t* vc,
		   int     s,
		   const int energy,
		   char *structure, //out
		   sect    bt_stack[MAXSECTORS]) { /* stack of partial structures for backtracking */
    /* call user-defined recursion status callback function */

  float   mfe = (float)(INF/100.);
  char    *ss;
  vrna_bp_stack_t   *bp;
  const int length  = (int) vc->length;

    if(vc->stat_cb)
      vc->stat_cb(vc, VRNA_STATUS_MFE_POST, vc->auxdata);  /* PORT: 2.7.2 added the fold compound as first argument */

    if(structure && vc->params->model_details.backtrack){
      bp = (vrna_bp_stack_t *)vrna_alloc(sizeof(vrna_bp_stack_t) * (4*(1+length/2))); /* add a guess of how many G's may be involved in a G quadruplex */

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
          free(bp);
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
        default:                      vrna_backtrack_from_intervals(vc, bp, bt_stack, s);
                                      break;
      }

      ss = vrna_db_from_bp_stack(bp, length);
      strncpy(structure, ss, length + 1);
      free(ss);
      free(bp);
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

// One reusable c/fML pair, sized to the longest record in the chunk. This is
// the point of the whole arrangement: host matrix memory becomes
// (workers) x 125.6 MB at 5601nt instead of (records in chunk) x 125.6 MB,
// which at a VRAM-filling chunk width was about as much host RAM as VRAM.
typedef struct {
  int    *c;
  int    *fML;
  size_t  cells;   /* capacity, in ints, of each of the two above */
  double  fetch_s; /* per-worker, so the timer needs no lock */
  double  busy_s;  /* likewise: this worker's active span, for attribution */
} bt_scratch_t;

PRIVATE void
bt_scratch_ensure(bt_scratch_t *sc, const size_t cells) {
  if(sc->cells >= cells) return;
  free(sc->c);
  free(sc->fML);
  sc->c     = (int *) vrna_alloc(sizeof(int) * cells);
  sc->fML   = (int *) vrna_alloc(sizeof(int) * cells);
  sc->cells = cells;
}

typedef struct {
  const vrna_fold_compound_t **VC;
  const char                 **Structure;
  int                         *energy;  /* now an OUTPUT: filled here, not by par_fill_arrays() */
  float                        *EN;
  int                           nfiles;
  int                          *next_i; /* shared work-claim counter */
  const size_t                *tri_off_H;
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
// Phase C3: `slot` is where the record's triangles LIVE on the device, which is
// the record's own index only when there is one record per slot. `idx` stays the
// record index: it is what indexes VC/Structure/energy/EN, and so what keeps the
// output in input order however the records were scheduled.
PRIVATE void
backtrack_one_slot(backtrack_pool_args_t *a, const int idx, const int slot, bt_scratch_t *sc) {
  vrna_fold_compound_t *vc = (vrna_fold_compound_t *) a->VC[idx];
  const size_t lo    = a->tri_off_H[slot];
  // Phase C1: the OCCUPANT's own triangle, not the slot's. Indx(i,j)=j*(j-1)/2+i
  // does not depend on length, so a record living in a larger slot occupies
  // exactly the prefix -- and taking the slot's size here would both copy cells
  // this record does not own and overrun host matrices sized to its own length.
  const size_t len   = (size_t)vc->length;
  const size_t cells = (len + 1)*(len + 2)/2;
  assert(vc->type == VRNA_FC_TYPE_SINGLE);
  assert(cells > 0);
  assert(cells <= a->tri_off_H[slot+1] - lo);  // the occupant must fit its slot

  bt_scratch_ensure(sc, cells);
  vc->matrices->c   = sc->c;
  vc->matrices->fML = sc->fML;

  const double t0 = rnafold_now_seconds();
  fetch_my_c_one(sc->c,   lo, cells);
  fetch_fML_one (sc->fML, lo, cells);
  sc->fetch_s += rnafold_now_seconds() - t0;

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
  a->EN[idx] = callback_backtrack(vc, 0, a->energy[idx], a->Structure[idx], bt_stack);

  // Detach before anything can free the compound: the scratch outlives it and
  // is reused, so leaving these set would hand vrna_mx_mfe_free() a pointer it
  // does not own. free(NULL) in there is a no-op.
  vc->matrices->c   = NULL;
  vc->matrices->fML = NULL;
}

PRIVATE void
backtrack_one(backtrack_pool_args_t *a, const int idx, bt_scratch_t *sc) {
  backtrack_one_slot(a, idx, idx, sc);   // one record per slot: they are the same
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
// Continuous flow phase C1: the one place the capacity rule lives. par_mfe() and
// par_fill_arrays() both build the offset tables independently (Phase 2c) and
// they address the SAME device buffers, so both must derive capacities the same
// way -- hence a shared function rather than the rule written out twice.
PUBLIC void
compute_slot_capacities(const int nfiles, const vrna_fold_compound_t **VC,
                        const int length, size_t* cap_H) {
  const int cap_max = rnafold_slot_capacity_max();
  for(int H=0; H<nfiles; H++)
    cap_H[H] = (size_t)(cap_max ? length : (int)VC[H]->length);
  // Phase C2: under the turnover self-check slot H hosts VC[H] and then
  // VC[(H+1)%nfiles], so it has to be big enough for both. This is the rule the
  // real scheduler (C3) generalises: a slot's capacity covers every record its
  // queue will ever hold.
  if(rnafold_slot_turnover() && nfiles >= 2)
    for(int H=0; H<nfiles; H++) {
      const size_t other = (size_t)VC[(H+1)%nfiles]->length;
      if(other > cap_H[H]) cap_H[H] = other;
    }
}

PUBLIC void
compute_batch_offsets(const int nfiles, const size_t* cap_H,
                       size_t* row_off_H, size_t* tri_off_H) {
  // Phase C1: built from the SLOT CAPACITY table, not from VC[H]->length. The
  // two are equal by default, so every table below is bit-for-bit what it was.
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

// Continuous flow phase C3: what happens the moment a record finishes its last
// row. Its triangles are still in its slot on the device and are about to be
// overwritten by the next occupant, so they are fetched HERE, and the record is
// finished off immediately -- exterior loop, MFE, backtrack -- into its own
// output slot, which keeps the output in input order whatever the schedule did.
//
// Serial, on the sweep's own thread: the sweep waits for one record's fetch and
// backtrack at each handover. Overlapping that with the sweep is the obvious
// next step and is deliberately not part of this one -- correctness first.
typedef struct {
  backtrack_pool_args_t *args;
  bt_scratch_t          *sc;
  int                    retired;
} retire_ctx_t;

PRIVATE void
on_retire_cb(void *ctx, int slot, int record) {
  retire_ctx_t *r = (retire_ctx_t *) ctx;
  backtrack_one_slot(r->args, record, slot, r->sc);
  r->retired++;
}

// Continuous flow phase C2: the post-sweep phase, lifted out of par_mfe()
// unchanged so it can be run for a SECOND set of occupants in the same slots.
// Everything it needs is now an argument; the only edit to the body was the
// de-indent and declaring its own loop variable.
PRIVATE void
backtrack_all(const int nfiles, const vrna_fold_compound_t **VC,
              const char **Structure, int *energy, float *EN,
              const size_t *tri_off_H, const int cpu_queue_threads) {
  int i;
  // Post-processing. Each record now gets its triangles fetched into a
  // pooled scratch pair, its exterior loop finished and its MFE read, then
  // backtracks, then releases the scratch -- see backtrack_one(). One pool
  // slot per worker, so peak host matrix memory is n_bt_threads records'
  // worth rather than nfiles.
  const double t_bt = rnafold_now_seconds();
  const int n_bt_threads = backtrack_thread_count(nfiles, cpu_queue_threads);
  int next_i = 0;
  backtrack_pool_args_t targ = { VC, Structure, energy, EN, nfiles, &next_i, tri_off_H };
  bt_scratch_t *pool = (bt_scratch_t *) vrna_alloc(sizeof(bt_scratch_t) * n_bt_threads);
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
    free(pool[t].c);
    free(pool[t].fML);
  }
  free(pool);
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
  // Continuous flow phase C1: the per-slot capacity, in nucleotides. Every
  // LAYOUT table below is built from this; every BOUND stays on the occupant's
  // own VC[H]->length. Equal by default -- see rnafold_slot_capacity_max().
  // Continuous flow phase C3: build the chunk's SCHEDULE first, because it
  // decides how many slots there are and therefore the shape of everything
  // below. Records are dealt round-robin in DESCENDING length order, so each
  // slot's queue is non-increasing and its capacity is simply its first
  // record's length -- which makes every later occupant fit by construction,
  // with no allocator and nothing to defragment.
  const int flow_k   = rnafold_slot_flow();
  const int use_flow = (flow_k >= 1 && nfiles >= 2);
  const int slots    = use_flow ? ((nfiles + flow_k - 1) / flow_k) : nfiles;

  int order[nfiles], queue[nfiles], qoff[slots+1];
  const vrna_fold_compound_t *VCsl_buf[slots];
  const vrna_fold_compound_t **VCsl = VC;
  if(use_flow) {
    for(int a=0;a<nfiles;a++) order[a] = a;
    for(int a=0;a<nfiles;a++) {          // selection sort, descending length
      int best = a;
      for(int b=a+1;b<nfiles;b++)
        if(VC[order[b]]->length > VC[order[best]]->length) best = b;
      const int t = order[a]; order[a] = order[best]; order[best] = t;
    }
    // Deal: slot s takes order[s], order[s+slots], order[s+2*slots], ...
    int w = 0;
    for(int s=0;s<slots;s++) {
      qoff[s] = w;
      for(int p=s; p<nfiles; p+=slots) queue[w++] = order[p];
    }
    qoff[slots] = w;
    assert(w == nfiles);
    for(int s=0;s<slots;s++) VCsl_buf[s] = VC[queue[qoff[s]]];
    VCsl = VCsl_buf;
  }

  const int nslots = slots;
  size_t cap_H[nslots];
  compute_slot_capacities(nslots, VCsl, length, cap_H);
  if(use_flow)
    for(int s=0;s<nslots;s++) {         // the whole queue must fit the slot
      for(int q=qoff[s]; q<qoff[s+1]; q++) {
        const size_t l = (size_t)VC[queue[q]]->length;
        if(l > cap_H[s]) cap_H[s] = l;
      }
    }
  // Publish it: par_fill_arrays() reads this instead of recomputing -- see
  // rnafold_chunk_capacities().
  if(g_cap_n < nslots) {
    free(g_cap_H);
    g_cap_H = (size_t *) vrna_alloc(sizeof(size_t) * nslots);
    g_cap_n = nslots;
  }
  memcpy(g_cap_H, cap_H, sizeof(size_t) * nslots);
  size_t row_off_H[nslots+1], tri_off_H[nslots+1]; //Phase 2a, see compute_batch_offsets() above
  compute_batch_offsets(nslots, cap_H, row_off_H, tri_off_H);
  const double t_gpuinit = rnafold_now_seconds();
  init_gpu(nslots,length,tri_off_H,row_off_H);
  init_gpu2(nslots,VCsl, turn, length, 512, tri_off_H, row_off_H, cap_H);
  init_gpu3(nslots,VCsl, turn, length, 512, row_off_H, cap_H);
  stage_gpuinit_s += rnafold_now_seconds() - t_gpuinit;

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
    if(use_flow) {
      // Continuous flow phase C3: the sweep itself retires each record as it
      // finishes, so there is no separate post-sweep phase -- on_retire_cb()
      // fetches and backtracks one record at each handover, and again for every
      // slot's final occupant when the sweep ends. One scratch pair serves them
      // all, which is what keeps host matrix memory at ONE record's worth.
      const double t_bt = rnafold_now_seconds();
      bt_scratch_t sc; memset(&sc, 0, sizeof(sc));
      backtrack_pool_args_t targ = { VC, Structure, energy, EN, nfiles, NULL, tri_off_H };
      retire_ctx_t rctx = { &targ, &sc, 0 };
      rnafold_schedule_t sched;
      sched.slots     = nslots;
      sched.length    = length;
      sched.qoff      = qoff;
      sched.queue     = queue;
      sched.VC_all    = VC;
      sched.on_retire = on_retire_cb;
      sched.ctx       = &rctx;

      par_fill_arrays(nslots, VCsl, energy, &sched);

      // Every record must have been retired exactly once -- a slot left holding
      // a record would silently produce an unwritten structure.
      if(rctx.retired != nfiles)
        fprintf(stderr,"%-24s SCHEDULE ERROR: %d of %d records retired\n",
                __FILE__, rctx.retired, nfiles);
      fprintf(stderr,"%-24s slot flow: %d records through %d slots, %d retired, "
                     "%zu triangle cells\n",
              __FILE__, nfiles, nslots, rctx.retired, tri_off_H[nslots]);
      free(sc.c); free(sc.fML);
      const double bt_phase_s = rnafold_now_seconds() - t_bt;
      phase_fetch_mx_s  += sc.fetch_s > bt_phase_s ? bt_phase_s : sc.fetch_s;
      stage_backtrack_s += bt_phase_s - (sc.fetch_s > bt_phase_s ? bt_phase_s : sc.fetch_s);
    } else {
    par_fill_arrays(nfiles,VC,energy,NULL);

    backtrack_all(nfiles, VC, Structure, energy, EN, tri_off_H, cpu_queue_threads);
    }

    // Continuous flow phase C2: the slot-turnover self-check. See
    // rnafold_slot_turnover(). Record r moves to slot r-1 (cyclically), every
    // slot is refilled from its new occupant, and the chunk is folded again;
    // slot s must then reproduce exactly what record (s+1) produced the first
    // time. nfiles is unchanged across the two passes on purpose -- d_S packs
    // ten bases per word with H as the fastest index, so its layout depends on
    // the slot COUNT and a pass with a different one would address it wrongly.
    if(rnafold_slot_turnover() && nfiles >= 2) {
      float *EN1  = (float *) vrna_alloc(sizeof(float) * nfiles);
      char **Str1 = (char **) vrna_alloc(sizeof(char *) * nfiles);
      for(i=0;i<nfiles;i++) { EN1[i] = EN[i]; Str1[i] = strdup(Structure[i]); }

      const vrna_fold_compound_t **VC2 =
        (const vrna_fold_compound_t **) vrna_alloc(sizeof(vrna_fold_compound_t *) * nfiles);
      char **Str2   = (char **) vrna_alloc(sizeof(char *) * nfiles);
      float *EN2    = (float *) vrna_alloc(sizeof(float) * nfiles);
      int   *energy2= (int *)   vrna_alloc(sizeof(int)   * nfiles);
      for(i=0;i<nfiles;i++) {
        VC2[i]  = VC[(i+1) % nfiles];
        Str2[i] = (char *) vrna_alloc(VC2[i]->length + 1);
      }

      // The occupants changed, so every piece of sequence-derived device
      // content has to be rebuilt. The sweep's own state (d_fml_j, d_dml,
      // d_dml1, d_fml_prev and all the host row buffers) is already reset by
      // par_fill_arrays() on every call, and d_my_c's INF prefill comes back
      // with refill_gpu2(); between them that is the complete reset list.
      refill_gpu2(nfiles, VC2, turn, length, 512, tri_off_H, row_off_H, cap_H);
      refill_gpu3(nfiles, VC2, turn, length, 512, row_off_H, cap_H);

      par_fill_arrays(nfiles, VC2, energy2, NULL);
      backtrack_all(nfiles, VC2, (const char **) Str2, energy2, EN2, tri_off_H, cpu_queue_threads);

      int bad = 0;
      for(i=0;i<nfiles;i++) {
        const int src = (i+1) % nfiles;
        if(EN2[i] != EN1[src] || strcmp(Str2[i], Str1[src]) != 0) {
          if(++bad <= 5)
            fprintf(stderr,"%-24s SLOT TURNOVER MISMATCH slot %d (record %d): "
                           "second %.2f vs first %.2f\n",
                    __FILE__, i, src, EN2[i], EN1[src]);
        }
      }
      fprintf(stderr,"%-24s slot turnover: %d records refolded in rotated slots, "
                     "%d mismatching\n", __FILE__, nfiles, bad);

      for(i=0;i<nfiles;i++) { free(Str1[i]); free(Str2[i]); }
      free(Str1); free(Str2); free(EN1); free(EN2); free(energy2); free(VC2);
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


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

#include "ViennaRNA/utils.h"
#include "ViennaRNA/energy_par.h"
#include "ViennaRNA/data_structures.h"
#include "ViennaRNA/fold_vars.h"
#include "ViennaRNA/params.h"
#include "ViennaRNA/constraints.h"
#include "ViennaRNA/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/unstructured_domains.h"
#include "ViennaRNA/loop_energies.h"
#include "ViennaRNA/mfe.h"

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

static double phase_int_loop_s          = 0.0;
static double phase_hp_mb_s             = 0.0;
static double phase_new_c_host_s        = 0.0;
static double phase_load_my_c_s         = 0.0;
static double phase_fml_host_s          = 0.0;
static double phase_modular_decomp_s    = 0.0;
static double phase_my_fml_update_host_s = 0.0;

static void
print_phase_timing_stats(void) {
  const double gpu_transfer_total = phase_int_loop_s + phase_hp_mb_s
                                   + phase_load_my_c_s + phase_modular_decomp_s;
  const double host_combine_total = phase_new_c_host_s + phase_fml_host_s
                                   + phase_my_fml_update_host_s;
  fprintf(stderr,
    "%-24s phase timing (s): int_loop=%.3f hp_mb=%.3f load_my_c=%.3f "
    "modular_decomp=%.3f | new_c_host=%.3f fml_host=%.3f my_fml_update_host=%.3f "
    "|| GPU+transfer total=%.3f host-combine total=%.3f\n",
    __FILE__, phase_int_loop_s, phase_hp_mb_s, phase_load_my_c_s,
    phase_modular_decomp_s, phase_new_c_host_s, phase_fml_host_s,
    phase_my_fml_update_host_s, gpu_transfer_total, host_combine_total);
}

/*
#################################
# PRIVATE FUNCTION DECLARATIONS #
#################################
*/

PRIVATE int           fill_arrays(vrna_fold_compound_t *vc){exit(99);}//use par_fill_arrays, code now included via fill_arrays.c
PRIVATE void          fill_arrays_circ(vrna_fold_compound_t *vc, sect bt_stack[], int *bt);
PRIVATE void          backtrack(vrna_fold_compound_t *vc, vrna_bp_stack_t *bp_stack, sect bt_stack[], int s);

PRIVATE int           fill_arrays_comparative(vrna_fold_compound_t *vc);
PRIVATE void          fill_arrays_comparative_circ(vrna_fold_compound_t *vc, sect bt_stack[], int *bt);
PRIVATE void          backtrack_comparative(vrna_fold_compound_t *vc, vrna_bp_stack_t *bp_stack, sect bt_stack[], int s);
PRIVATE float          mfe_cuda_vrna_mfe(vrna_fold_compound_t *vc, char *structure); // renamed from vrna_mfe, see definition

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
      vc->stat_cb(VRNA_STATUS_MFE_POST, vc->auxdata);

    if(structure && vc->params->model_details.backtrack){
      bp = (vrna_bp_stack_t *)vrna_alloc(sizeof(vrna_bp_stack_t) * (4*(1+length/2))); /* add a guess of how many G's may be involved in a G quadruplex */

      switch(vc->type){
        case VRNA_FC_TYPE_COMPARATIVE:  backtrack_comparative(vc, bp, bt_stack, s);
                                      break;

        case VRNA_FC_TYPE_SINGLE:     /* fall through */

        default:                      backtrack(vc, bp, bt_stack, s);
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
 *   unset or "0" -> disabled, exactly the original serial loop.
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
  if(!env || !env[0] || !strcmp(env, "0")) return 1;

  int n;
  if(!strcmp(env, "auto")) {
    long hw = sysconf(_SC_NPROCESSORS_ONLN);
    if(hw < 1) hw = 1;
    n = (int)hw - cpu_queue_threads;
  } else {
    n = atoi(env);
  }
  if(n < 1) n = 1;
  if(n > nfiles) n = nfiles;
  return n;
}

typedef struct {
  const vrna_fold_compound_t **VC;
  const char                 **Structure;
  const int                   *energy;
  float                        *EN;
  int                           nfiles;
  int                          *next_i; /* shared work-claim counter */
} backtrack_pool_args_t;

PRIVATE void *
backtrack_worker(void *arg) {
  backtrack_pool_args_t *a = (backtrack_pool_args_t *) arg;
  int idx;
  while((idx = __sync_fetch_and_add(a->next_i, 1)) < a->nfiles) {
    assert(a->VC[idx]->type == VRNA_FC_TYPE_SINGLE);
    sect bt_stack[MAXSECTORS]; /* stack of partial structures for backtracking, thread-local */
    a->EN[idx] = callback_backtrack(a->VC[idx], 0, a->energy[idx], a->Structure[idx], bt_stack);
  }
  return NULL;
}

//except par_fill_arrays(), do each file sequentially as before
PUBLIC void
par_mfe(const int nfiles,
	const vrna_fold_compound_t** VC,
	const char** Structure,
	float* EN, //out
	const int cpu_queue_threads) {

  //start GPU as early as possible so allow maximise overlap GPU with CPU
  const int length    = VC[0]->length;
  const vrna_md_t* md = &(VC[0]->params->model_details);
  const int turn      = md->min_loop_size;
  init_gpu(nfiles,length);
  init_gpu2(nfiles,VC, turn, length, 512);
  init_gpu3(nfiles,VC, turn, length, 512);

  if(VC[0]->type == VRNA_FC_TYPE_SINGLE) {
    int i;
    for(i=0;i<nfiles;i++) {
      if(!vrna_fold_compound_prepare(VC[i], VRNA_OPTION_MFE)){
	vrna_message_warning("vrna_mfe@mfe.c: Failed to prepare vrna_fold_compound");
	EN[i] = INF/100.0F;
	exit(1);
      }
      /* call user-defined recursion status callback function */
      if(VC[i]->stat_cb){ VC[i]->stat_cb(VRNA_STATUS_MFE_PRE, VC[i]->auxdata);}
    }

    int energy[nfiles];
    par_fill_arrays(nfiles,VC,energy);

    const int n_bt_threads = backtrack_thread_count(nfiles, cpu_queue_threads);
    if(n_bt_threads <= 1) {
      for(i=0;i<nfiles;i++) {
        assert(VC[i]->type == VRNA_FC_TYPE_SINGLE);
        sect bt_stack[MAXSECTORS]; /* stack of partial structures for backtracking */
        EN[i] = callback_backtrack(VC[i], 0, energy[i],Structure[i],bt_stack);
      }
    } else {
      int next_i = 0;
      backtrack_pool_args_t targ = { VC, Structure, energy, EN, nfiles, &next_i };
      pthread_t *bt_threads = (pthread_t *) vrna_alloc(sizeof(pthread_t) * n_bt_threads);
      int started = 0;
      for(; started < n_bt_threads; started++) {
        if(pthread_create(&bt_threads[started], NULL, backtrack_worker, &targ) != 0) {
          fprintf(stderr, "mfe_cuda.c: pthread_create failed for backtrack worker %d, "
                           "continuing with fewer threads\n", started);
          break;
        }
      }
      if(started == 0) {
        /* pthread_create failed on the very first thread -- fall back to serial */
        for(i=0;i<nfiles;i++) {
          assert(VC[i]->type == VRNA_FC_TYPE_SINGLE);
          sect bt_stack[MAXSECTORS];
          EN[i] = callback_backtrack(VC[i], 0, energy[i],Structure[i],bt_stack);
        }
      } else {
        for(int t=0; t<started; t++)
          pthread_join(bt_threads[t], NULL);
      }
      free(bt_threads);
    }
  } else {
  for(int i=0;i<nfiles;i++) {
    EN[i] = mfe_cuda_vrna_mfe(VC[i], Structure[i]);
  }}
}

/* New Jul 2026: renamed from vrna_mfe + made PRIVATE (static). This file's
 * mfe.h include declares the real vrna_mfe() as an external PUBLIC symbol;
 * this file used to define ANOTHER external vrna_mfe() of its own (same
 * name), which happened to not conflict only because nothing previously
 * forced the linker to also pull mfe.c's real vrna_mfe() into RNAfold's
 * link. Adding vrna_mfe_cpu() to mfe.c (for the CPU worker queue) changed
 * that -- satisfying its own undefined-symbol requirement drags the whole
 * mfe.o archive member in, including mfe.c's real vrna_mfe(), which then
 * collided with this one at link time ("multiple definition"). Renaming +
 * making this PRIVATE removes the collision entirely: it's only ever
 * called from this file's own par_mfe() (the VRNA_FC_TYPE_COMPARATIVE
 * branch just above), so it never needed external linkage in the first
 * place. NOT safe to call for VRNA_FC_TYPE_SINGLE -- see fill_arrays()
 * stub below (exit(99)); still only exercised via the comparative branch. */
PRIVATE float
mfe_cuda_vrna_mfe(vrna_fold_compound_t *vc,
          char *structure){
#ifdef GA
  modify_params(vc);
#endif

  int     energy, s;
  sect    bt_stack[MAXSECTORS]; /* stack of partial structures for backtracking */

  s       = 0;
  const float mfe = (float)(INF/100.);

  if(vc){
    if(!vrna_fold_compound_prepare(vc, VRNA_OPTION_MFE)){
      vrna_message_warning("vrna_mfe@mfe.c: Failed to prepare vrna_fold_compound");
      return mfe;
    }

    /* call user-defined recursion status callback function */
    if(vc->stat_cb)
      vc->stat_cb(VRNA_STATUS_MFE_PRE, vc->auxdata);

    switch(vc->type){
      case VRNA_FC_TYPE_SINGLE:     energy = fill_arrays(vc);
                                    if(vc->params->model_details.circ){
                                      fill_arrays_circ(vc, bt_stack, &s);
                                      energy = vc->matrices->Fc;
                                    }
                                    break;

      case VRNA_FC_TYPE_COMPARATIVE:  energy = fill_arrays_comparative(vc);
                                    if(vc->params->model_details.circ){
                                      fill_arrays_comparative_circ(vc, bt_stack, &s);
                                      energy = vc->matrices->Fc;
                                    }
                                    break;

      default:                      vrna_message_warning("unrecognized fold compound type");
                                    return mfe;
                                    break;
    }

    return callback_backtrack(vc,s,energy,structure,bt_stack);
  }
  return mfe;
}

/**
*** fill "c", "fML" and "f5" arrays and return  optimal energy
**/
#include "fill_arrays.c"

#include "circfold.inc"


/**
*** the actual forward recursion to fill the energy arrays
**/
PRIVATE int
fill_arrays_comparative(vrna_fold_compound_t *vc){

  char              *hard_constraints;
  unsigned short    **a2s;
  short             **S, **S5, **S3;
  int               i, j, turn, energy, stackEnergy, new_c, s, *type, tt, *cc,
                    *cc1, *Fmi, *DMLi, *DMLi1, *DMLi2, n_seq, length, *indx,
                    *c, *f5, *fML, *ggg, *pscore, dangle_model;
  vrna_param_t      *P;
  vrna_md_t         *md;
  vrna_hc_t         *hc;
  vrna_sc_t         **sc;

  n_seq             = vc->n_seq;
  length            = vc->length;
  S                 = vc->S;
  S5                = vc->S5;             /* S5[s][i] holds next base 5' of i in sequence s */
  S3                = vc->S3;             /* Sl[s][i] holds next base 3' of i in sequence s */
  a2s               = vc->a2s;
  P                 = vc->params;
  md                = &(P->model_details);
  indx              = vc->jindx;          /* index for moving in the triangle matrices c[] and fMl[] */
  c                 = vc->matrices->c;    /* energy array, given that i-j pair */
  f5                = vc->matrices->f5;   /* energy of 5' end */
  fML               = vc->matrices->fML;  /* multi-loop auxiliary energy array */
  ggg               = vc->matrices->ggg;
  pscore            = vc->pscore;         /* precomputed array of pair types */
  dangle_model      = md->dangles;
  turn              = md->min_loop_size;
  hc                = vc->hc;
  sc                = vc->scs;
  hard_constraints  = hc->matrix;

  /* allocate some memory for helper arrays */
  type  = (int *) vrna_alloc(n_seq*sizeof(int));
  cc    = (int *) vrna_alloc(sizeof(int)*(length+2)); /* linear array for calculating canonical structures */
  cc1   = (int *) vrna_alloc(sizeof(int)*(length+2)); /*   "     "        */
  Fmi   = (int *) vrna_alloc(sizeof(int)*(length+1)); /* holds row i of fML (avoids jumps in memory) */
  DMLi  = (int *) vrna_alloc(sizeof(int)*(length+1)); /* DMLi[j] holds MIN(fML[i,k]+fML[k+1,j])  */
  DMLi1 = (int *) vrna_alloc(sizeof(int)*(length+1)); /*             MIN(fML[i+1,k]+fML[k+1,j])  */
  DMLi2 = (int *) vrna_alloc(sizeof(int)*(length+1)); /*             MIN(fML[i+2,k]+fML[k+1,j])  */


  if((turn < 0) || (turn > length))
    turn = length;

  /* init energies */
  for (j=1; j<=length; j++){
    Fmi[j]=DMLi[j]=DMLi1[j]=DMLi2[j]=INF;
    for (i=(j>turn?(j-turn):1); i<j; i++) {
      c[indx[j]+i] = fML[indx[j]+i] = INF;
    }
  }

  /* begin recursions */
  for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */
    for (j = i+turn+1; j <= length; j++) {
      int ij, psc;
      ij = indx[j]+i;

      for (s=0; s<n_seq; s++) {
        type[s] = md->pair[S[s][i]][S[s][j]];
        if (type[s]==0) type[s]=7;
      }

      psc = pscore[indx[j]+i];
      if (hard_constraints[ij]) {   /* a pair to consider */
        new_c = INF;

        /* hairpin ----------------------------------------------*/
        energy  = vrna_E_hp_loop(vc, i, j);
        new_c   = MIN2(new_c, energy);

        /* check for multibranch loops */
        energy  = vrna_E_mb_loop_fast(vc, i, j, DMLi1, DMLi2);
        new_c   = MIN2(new_c, energy);

        /* check for interior loops */
        energy  = vrna_E_int_loop(vc, i, j);
        new_c   = MIN2(new_c, energy);

        /* remember stack energy for --noLP option */
        if(md->noLP){
          stackEnergy = vrna_E_stack(vc, i, j);
          new_c       = MIN2(new_c, cc1[j-1]+stackEnergy);
          cc[j]       = new_c - psc; /* add covariance bonnus/penalty */
          c[ij]       = cc1[j-1] + stackEnergy - psc;
        } else {
          c[ij]       = new_c - psc; /* add covariance bonnus/penalty */
        }
      } /* end >> if (pair) << */

      else c[ij] = INF;

      /* done with c[i,j], now compute fML[i,j] */
      assert(0); //protect vrna_E_ml_stems_fast
      //fML[ij] = Fmi[j] = vrna_E_ml_stems_fast(vc, i, j, Fmi, DMLi);

    } /* END for j */

    {
      int *FF; /* rotate the auxilliary arrays */
      FF = DMLi2; DMLi2 = DMLi1; DMLi1 = DMLi; DMLi = FF;
      FF = cc1; cc1=cc; cc=FF;
      for (j=1; j<=length; j++) {cc[j]=Fmi[j]=DMLi[j]=INF; }
    }
  } /* END for i */
  /* calculate energies of 5' and 3' fragments */

  f5[0] = 0;
  for(j = 1; j <= turn + 1; j++){
    if(hc->up_ext[j]){
      energy = f5[j-1];
      if((energy < INF) && sc)
        for(s=0;s < n_seq; s++){
          if(sc[s]){
            if(sc[s]->energy_up)
              energy += sc[s]->energy_up[a2s[s][j]][1];
          }
        }
    } else {
      energy = INF;
    }
    f5[j] = energy;
  }

  switch(dangle_model){
    case 0:   for(j = turn + 2; j <= length; j++){
                f5[j] = INF;

                if(hc->up_ext[j]){
                  energy = f5[j-1];
                  if((energy < INF) && sc)
                    for(s=0; s < n_seq; s++){
                      if(sc[s]){
                        if(sc[s]->energy_up)
                          energy += sc[s]->energy_up[a2s[s][j]][1];
                      }
                    }
                  f5[j] = MIN2(f5[j], energy);
                }

                if (hard_constraints[indx[j]+1] & VRNA_CONSTRAINT_CONTEXT_EXT_LOOP){
                  if(c[indx[j]+1] < INF){
                    energy = c[indx[j]+1];
                    for(s = 0; s < n_seq; s++){
                      tt = md->pair[S[s][1]][S[s][j]];
                      if(tt==0) tt=7;
                      energy += E_ExtLoop(tt, -1, -1, P);
                    }
                    f5[j] = MIN2(f5[j], energy);
                  }

                  if(md->gquad){
                    if(ggg[indx[j]+1] < INF)
                      f5[j] = MIN2(f5[j], ggg[indx[j]+1]);
                  }
                }

                for(i = j - turn - 1; i > 1; i--){
                  if(hard_constraints[indx[j]+i] & VRNA_CONSTRAINT_CONTEXT_EXT_LOOP){
                    if(c[indx[j]+i]<INF){
                      energy = f5[i-1] + c[indx[j]+i];
                      for(s = 0; s < n_seq; s++){
                        tt = md->pair[S[s][i]][S[s][j]];
                        if(tt==0) tt=7;
                        energy += E_ExtLoop(tt, -1, -1, P);
                      }
                      f5[j] = MIN2(f5[j], energy);
                    }

                    if(md->gquad){
                      if(ggg[indx[j]+i] < INF)
                        f5[j] = MIN2(f5[j], f5[i-1] + ggg[indx[j]+i]);
                    }
                  }
                }
              }
              break;

    default:  for(j = turn + 2; j <= length; j++){
                f5[j] = INF;

                if(hc->up_ext[j]){
                  energy = f5[j-1];
                  if((energy < INF) && sc)
                    for(s=0; s < n_seq; s++){
                      if(sc[s]){
                        if(sc[s]->energy_up)
                          energy += sc[s]->energy_up[a2s[s][j]][1];
                      }
                    }
                  f5[j] = MIN2(f5[j], energy);
                }

                if(hard_constraints[indx[j]+1] & VRNA_CONSTRAINT_CONTEXT_EXT_LOOP){
                  if (c[indx[j]+1]<INF) {
                    energy = c[indx[j]+1];
                    for(s = 0; s < n_seq; s++){
                      tt = md->pair[S[s][1]][S[s][j]];
                      if(tt==0) tt=7;
                      energy += E_ExtLoop(tt, -1, (j<length) ? S3[s][j] : -1, P);
                    }
                    f5[j] = MIN2(f5[j], energy);
                  }

                  if(md->gquad){
                    if(ggg[indx[j]+1] < INF)
                      f5[j] = MIN2(f5[j], ggg[indx[j]+1]);
                  }
                }

                for(i = j - turn - 1; i > 1; i--){
                  if(hard_constraints[indx[j]+i] & VRNA_CONSTRAINT_CONTEXT_EXT_LOOP){
                    if (c[indx[j]+i]<INF) {
                      energy = f5[i-1] + c[indx[j]+i];
                      for(s = 0; s < n_seq; s++){
                        tt = md->pair[S[s][i]][S[s][j]];
                        if(tt==0) tt=7;
                        energy += E_ExtLoop(tt, S5[s][i], (j < length) ? S3[s][j] : -1, P);
                      }
                      f5[j] = MIN2(f5[j], energy);
                    }

                    if(md->gquad){
                      if(ggg[indx[j]+i] < INF)
                        f5[j] = MIN2(f5[j], f5[i-1] + ggg[indx[j]+i]);
                    }
                  }
                }
              }
              break;
  }
  free(type);
  free(cc);
  free(cc1);
  free(Fmi);
  free(DMLi);
  free(DMLi1);
  free(DMLi2);
  return(f5[length]);
}

#include "ViennaRNA/alicircfold.inc"

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
PRIVATE void
backtrack(vrna_fold_compound_t *vc,
          vrna_bp_stack_t *bp_stack,
          sect bt_stack[],
          int s){

  unsigned char   type;
  char            *string, *ptype, backtrack_type;
  int             i, j, ij, k, length, no_close, b, *my_c, *indx, noLP, noGUclosure;
  vrna_param_t    *P;

  b               = 0;
  length          = vc->length;
  my_c            = vc->matrices->c;
  indx            = vc->jindx;
  P               = vc->params;
  noLP            = P->model_details.noLP;
  noGUclosure     = P->model_details.noGUclosure;
  string          = vc->sequence;
  ptype           = vc->ptype;
  backtrack_type  = P->model_details.backtrack_type;

  if (s==0) {
    bt_stack[++s].i = 1;
    bt_stack[s].j = length;
    bt_stack[s].ml = (backtrack_type=='M') ? 1 : ((backtrack_type=='C')? 2: 0);
  }
  while (s>0) {
    int ml, cij;
    int canonical = 1;     /* (i,j) closes a canonical structure */

    /* pop one element from stack */
    i  = bt_stack[s].i;
    j  = bt_stack[s].j;
    ml = bt_stack[s--].ml;

    switch(ml){
      /* backtrack in f5 */
      case 0:   {
                  int p, q;
                  if(vrna_BT_ext_loop_f5(vc, &j, &p, &q, bp_stack, &b)){
                    if(j > 0){
                      bt_stack[++s].i = 1;
                      bt_stack[s].j   = j;
                      bt_stack[s].ml  = 0;
                    }
                    if(p > 0){
                      i = p;
                      j = q;
                      goto repeat1;
                    }

                    continue;
                  } else {
                    vrna_message_error("backtracking failed in f5 for sequence:\n%s\n", string);
                  }
                }
                break;

      /* trace back in fML array */
      case 1:   {
                  int p, q, comp1, comp2;
                  if(vrna_BT_mb_loop_split(vc, &i, &j, &p, &q, &comp1, &comp2, bp_stack, &b)){
                    if(i > 0){
                      bt_stack[++s].i = i;
                      bt_stack[s].j   = j;
                      bt_stack[s].ml  = comp1;
                    }
                    if(p > 0){
                      bt_stack[++s].i = p;
                      bt_stack[s].j   = q;
                      bt_stack[s].ml  = comp2;
                    }

                    continue;
                  } else {
                    vrna_message_error("backtracking failed in fML for sequence:\n%s\n", string);
                  }
                }
                break;

      /* backtrack in c */
      case 2:   bp_stack[++b].i = i;
                bp_stack[b].j   = j;
                goto repeat1;

      default:  vrna_message_error("Backtracking failed due to unrecognized DP matrix!");
                break;
    }

  repeat1:

    /*----- begin of "repeat:" -----*/
    ij = indx[j]+i;

    if (canonical)
      cij = my_c[ij];

    type = (unsigned char)ptype[ij];

    if (noLP)
      if(vrna_BT_stack(vc, &i, &j, &cij, bp_stack, &b)){
        canonical = 0;
        goto repeat1;
      }

    canonical = 1;

    no_close = (((type==3)||(type==4))&&noGUclosure);

    if (no_close) {
      if (cij == FORBIDDEN) continue;
    } else {
      if(vrna_BT_hp_loop(vc, i, j, cij, bp_stack, &b))
        continue;
    }

    if(vrna_BT_int_loop(vc, &i, &j, cij, bp_stack, &b)){
      if(i < 0)
        continue;
      else
        goto repeat1;
    }

    /* (i.j) must close a multi-loop */
    int comp1, comp2;

    if(vrna_BT_mb_loop(vc, &i, &j, &k, cij, &comp1, &comp2)){
      bt_stack[++s].i = i;
      bt_stack[s].j   = k;
      bt_stack[s].ml  = comp1;
      bt_stack[++s].i = k + 1;
      bt_stack[s].j   = j;
      bt_stack[s].ml  = comp2;
    } else {
      vrna_message_error("backtracking failed in repeat for sequence:\n%s\n", string);
    }

    /* end of repeat: --------------------------------------------------*/

  } /* end of infinite while loop */

  bp_stack[0].i = b;    /* save the total number of base pairs */
}


/**
*** backtrack in the energy matrices to obtain a structure with MFE
**/
PRIVATE void
backtrack_comparative(vrna_fold_compound_t *vc,
                      vrna_bp_stack_t *bp_stack,
                      sect bt_stack[],
                      int s) {

  /*------------------------------------------------------------------
    trace back through the "c", "f5" and "fML" arrays to get the
    base pairing list. No search for equivalent structures is done.
    This inverts the folding procedure, hence it's very fast.
    ------------------------------------------------------------------*/
   /* normally s=0.
     If s>0 then s items have been already pushed onto the sector stack */

  unsigned short  **a2s;
  short           **S, **S5, **S3, *S_cons;
  int             i, j, k, p, q, turn, energy, en, c0, l1, minq, maxq,
                  type_2, tt, mm, b, cov_en, *type, n_seq, length, *indx,
                  *c, *f5, *fML, *pscore, *ggg, *rtype, dangle_model, with_gquad;
  vrna_param_t    *P;
  vrna_md_t       *md;
  vrna_hc_t       *hc;
  vrna_sc_t       **sc;

  n_seq         = vc->n_seq;
  length        = vc->length;
  S             = vc->S;
  S5            = vc->S5;     /*S5[s][i] holds next base 5' of i in sequence s*/
  S3            = vc->S3;     /*Sl[s][i] holds next base 3' of i in sequence s*/
  a2s           = vc->a2s;
  P             = vc->params;
  md            = &(P->model_details);
  indx          = vc->jindx;     /* index for moving in the triangle matrices c[] and fMl[]*/
  c             = vc->matrices->c;     /* energy array, given that i-j pair */
  f5            = vc->matrices->f5;     /* energy of 5' end */
  fML           = vc->matrices->fML;     /* multi-loop auxiliary energy array */
  pscore        = vc->pscore;     /* precomputed array of pair types */
  ggg           = vc->matrices->ggg;
  S_cons        = vc->S_cons;
  rtype         = &(md->rtype[0]);
  dangle_model  = md->dangles;
  with_gquad    = md->gquad;
  hc            = vc->hc;
  sc            = vc->scs;
  turn          = md->min_loop_size;
  b             = 0;
  cov_en        = 0;

  type  = (int *) vrna_alloc(n_seq*sizeof(int));

  if((turn < 0) || (turn > length))
    turn = length;

  if (s==0) {
    bt_stack[++s].i = 1;
    bt_stack[s].j = length;
    bt_stack[s].ml = (md->backtrack_type=='M') ? 1 : ((md->backtrack_type=='C')?2:0);
  }
  while (s>0) {
    int ss, ml, fij, fi, cij, traced, i1, j1, jj=0, gq=0;
    int canonical = 1;     /* (i,j) closes a canonical structure */
    i  = bt_stack[s].i;
    j  = bt_stack[s].j;
    ml = bt_stack[s--].ml;   /* ml is a flag indicating if backtracking is to
                              occur in the fML- (1) or in the f-array (0) */
    if (ml==2) {
      bp_stack[++b].i = i;
      bp_stack[b].j   = j;
      cov_en += pscore[indx[j]+i];
      goto repeat1_comparative;
    }

    if (j < i+turn+1) continue; /* no more pairs in this interval */

    if(ml != 0){
      fij = fML[indx[j]+i];
      fi  = (hc->up_ml[j]) ? fML[indx[j-1]+i] + n_seq*P->MLbase : INF;
    } else {
      fij = f5[j];
      fi  = (hc->up_ext[j]) ? f5[j-1] : INF;
    }

    if(sc)
      for(ss = 0; ss < n_seq; ss++)
        if(sc[ss]){
          if(sc[ss]->energy_up)
            fi += sc[ss]->energy_up[a2s[ss][j]][1];
        }

    if (fij == fi) {  /* 3' end is unpaired */
      bt_stack[++s].i = i;
      bt_stack[s].j   = j-1;
      bt_stack[s].ml  = ml;
      continue;
    }

    if (ml == 0) { /* backtrack in f5 */
      switch(dangle_model){
        case 0:   /* j or j-1 is paired. Find pairing partner */
                  for (i=j-turn-1,traced=0; i>=1; i--) {
                    int en;
                    jj = i-1;

                    if (hc->matrix[indx[j] + i] & VRNA_CONSTRAINT_CONTEXT_EXT_LOOP){
                      en = c[indx[j]+i] + f5[i-1];
                      for(ss = 0; ss < n_seq; ss++){
                        type[ss] = md->pair[S[ss][i]][S[ss][j]];
                        if (type[ss]==0) type[ss] = 7;
                        en += E_ExtLoop(type[ss], -1, -1, P);
                      }
                      if (fij == en) traced=j;
                    }

                    if(with_gquad){
                      if(fij == f5[i-1] + ggg[indx[j]+i]){
                        /* found the decomposition */
                        traced = j; jj = i - 1; gq = 1;
                        break;
                      }
                    }

                    if (traced) break;
                  }
                  break;
        default:  /* j or j-1 is paired. Find pairing partner */
                  for (i=j-turn-1,traced=0; i>=1; i--) {
                    int en;
                    jj = i-1;
                    if (hc->matrix[indx[j] + i] & VRNA_CONSTRAINT_CONTEXT_EXT_LOOP){
                      en = c[indx[j]+i] + f5[i-1];
                      for(ss = 0; ss < n_seq; ss++){
                        type[ss] = md->pair[S[ss][i]][S[ss][j]];
                        if (type[ss]==0) type[ss] = 7;
                        en += E_ExtLoop(type[ss], (i>1) ? S5[ss][i]: -1, (j < length) ? S3[ss][j] : -1, P);
                      }
                      if (fij == en) traced=j;
                    }

                    if(with_gquad){
                      if(fij == f5[i-1] + ggg[indx[j]+i]){
                        /* found the decomposition */
                        traced = j; jj = i - 1; gq = 1;
                        break;
                      }
                    }

                    if (traced) break;
                  }
                  break;
      }

      if (!traced) vrna_message_error("backtrack failed in f5");
      /* push back the remaining f5 portion */
      bt_stack[++s].i = 1;
      bt_stack[s].j   = jj;
      bt_stack[s].ml  = ml;

      /* trace back the base pair found */
      j=traced;

      if(with_gquad && gq){
        /* goto backtrace of gquadruplex */
        goto repeat_gquad_comparative;
      }

      bp_stack[++b].i = i;
      bp_stack[b].j   = j;
      cov_en += pscore[indx[j]+i];
      goto repeat1_comparative;
    }
    else { /* trace back in fML array */
      if(hc->up_ml[i]){
        en = fML[indx[j]+i+1] + n_seq * P->MLbase;

        if(sc)
          for(ss = 0; ss < n_seq; ss++)
            if(sc[ss]){
              if(sc[ss]->energy_up)
                en += sc[ss]->energy_up[a2s[ss][i]][1];
            }

        if(en == fij) { /* 5' end is unpaired */
          bt_stack[++s].i = i+1;
          bt_stack[s].j   = j;
          bt_stack[s].ml  = ml;
          continue;
        }
      }

      if(md->gquad){
        if(fij == ggg[indx[j]+i] + n_seq * E_MLstem(0, -1, -1, P)){
          /* go to backtracing of quadruplex */
          goto repeat_gquad_comparative;
        }
      }

      if(hc->matrix[indx[j] + i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC){
        cij = c[indx[j]+i];
        if(dangle_model){
          for(ss = 0; ss < n_seq; ss++){
            tt = md->pair[S[ss][i]][S[ss][j]];
            if(tt==0) tt=7;
            cij += E_MLstem(tt, S5[ss][i], S3[ss][j], P);
          }
        }
        else{
          for(ss = 0; ss < n_seq; ss++){
            tt = md->pair[S[ss][i]][S[ss][j]];
            if(tt==0) tt=7;
            cij += E_MLstem(tt, -1, -1, P);
          }
        }

        if (fij==cij){
          /* found a pair */
          bp_stack[++b].i = i;
          bp_stack[b].j   = j;
          cov_en += pscore[indx[j]+i];
          goto repeat1_comparative;
        }
      }

      for (k = i+1+turn; k <= j-2-turn; k++)
        if (fij == (fML[indx[k]+i]+fML[indx[j]+k+1]))
          break;

      bt_stack[++s].i = i;
      bt_stack[s].j   = k;
      bt_stack[s].ml  = ml;
      bt_stack[++s].i = k+1;
      bt_stack[s].j   = j;
      bt_stack[s].ml  = ml;

      if (k>j-2-turn) vrna_message_error("backtrack failed in fML");
      continue;
    }

  repeat1_comparative:

    /*----- begin of "repeat:" -----*/
    if (canonical)  cij = c[indx[j]+i];

    for (ss=0; ss<n_seq; ss++) {
      type[ss] = md->pair[S[ss][i]][S[ss][j]];
      if (type[ss]==0) type[ss] = 7;
    }

    if (md->noLP)
      if (cij == c[indx[j]+i]) {
        /* (i.j) closes canonical structures, thus
           (i+1.j-1) must be a pair                */
        for (ss=0; ss<n_seq; ss++) {
          type_2 = md->pair[S[ss][j-1]][S[ss][i+1]];  /* j,i not i,j */
          if (type_2==0) type_2 = 7;
          cij -= P->stack[type[ss]][type_2];
          if(sc){
            if(sc[ss]->energy_bp)
              cij -= sc[s]->energy_bp[indx[j] + i];
          }
        }
        cij += pscore[indx[j]+i];
        bp_stack[++b].i = i+1;
        bp_stack[b].j   = j-1;
        cov_en += pscore[indx[j-1]+i+1];
        i++; j--;
        canonical=0;
        goto repeat1_comparative;
      }
    canonical = 1;
    cij += pscore[indx[j]+i];

    /* does (i,j) close a hairpin loop ? */
    if(vrna_BT_hp_loop(vc, i, j, cij, bp_stack, &b))
      continue;

    for (p = i+1; p <= MIN2(j-2-turn,i+MAXLOOP+1); p++) {
      minq = j-i+p-MAXLOOP-2;
      if (minq<p+1+turn) minq = p+1+turn;
      if(hc->up_int[i+1] < (p - i - 1)) break;

      for (q = j-1; q >= minq; q--) {

        if(hc->up_int[q+1] < (j - q - 1)) break;

        if (c[indx[q]+p]>=INF) continue;

        for (ss=energy=0; ss<n_seq; ss++) {
          int u1 = a2s[ss][p-1] - a2s[ss][i];
          int u2 = a2s[ss][j-1] - a2s[ss][q];
          type_2 = md->pair[S[ss][q]][S[ss][p]];  /* q,p not p,q */
          if (type_2==0) type_2 = 7;
          energy += E_IntLoop(u1, u2, type[ss], type_2, S3[ss][i], S5[ss][j], S5[ss][p], S3[ss][q], P);

        }

        if(sc)
          for(ss = 0; ss < n_seq; ss++)
            if(sc[ss]){
              int u1 = a2s[ss][p-1] - a2s[ss][i];
              int u2 = a2s[ss][j-1] - a2s[ss][q];
/*
              int u1 = p - i - 1;
              int u2 = j - q - 1;
*/
              if(u1 + u2 == 0)
                if(sc[ss]->energy_stack){
                  if(S[ss][i] && S[ss][j] && S[ss][p] && S[ss][q]){ /* don't allow gaps in stack */
                    energy +=   sc[ss]->energy_stack[a2s[ss][i]]
                              + sc[ss]->energy_stack[a2s[ss][p]]
                              + sc[ss]->energy_stack[a2s[ss][q]]
                              + sc[ss]->energy_stack[a2s[ss][j]];
                  }
                }
              if(sc[ss]->energy_bp)
                energy += sc[ss]->energy_bp[indx[j] + i];

              if(sc[ss]->energy_up)
                energy +=   sc[ss]->energy_up[a2s[ss][i] + 1][u1]
                          + sc[ss]->energy_up[a2s[ss][q] + 1][u2];
            }

        traced = (cij == energy+c[indx[q]+p]);
        if (traced) {
          bp_stack[++b].i = p;
          bp_stack[b].j   = q;
          cov_en += pscore[indx[q]+p];
          i = p, j = q;
          goto repeat1_comparative;
        }
      }
    }

    /* end of repeat: --------------------------------------------------*/

    /* (i.j) must close a multi-loop */

    i1 = i+1;
    j1 = j-1;

    if(with_gquad){
      /*
        The case that is handled here actually resembles something like
        an interior loop where the enclosing base pair is of regular
        kind and the enclosed pair is not a canonical one but a g-quadruplex
        that should then be decomposed further...
      */
      mm = 0;
      for(ss=0;ss<n_seq;ss++){
        tt = type[ss];
        if(tt == 0) tt = 7;
        if(dangle_model == 2)
          mm += P->mismatchI[tt][S3[ss][i]][S5[ss][j]];
        if(tt > 2)
          mm += P->TerminalAU;
      }

      for(p = i + 2;
        p < j - VRNA_GQUAD_MIN_BOX_SIZE;
        p++){
        if(S_cons[p] != 3) continue;
        l1    = p - i - 1;
        if(l1>MAXLOOP) break;
        minq  = j - i + p - MAXLOOP - 2;
        c0    = p + VRNA_GQUAD_MIN_BOX_SIZE - 1;
        minq  = MAX2(c0, minq);
        c0    = j - 1;
        maxq  = p + VRNA_GQUAD_MAX_BOX_SIZE + 1;
        maxq  = MIN2(c0, maxq);
        for(q = minq; q < maxq; q++){
          if(S_cons[q] != 3) continue;
          c0  = mm + ggg[indx[q] + p] + n_seq * P->internal_loop[l1 + j - q - 1];
          if(cij == c0){
            i=p;j=q;
            goto repeat_gquad_comparative;
          }
        }
      }
      p = i1;
      if(S_cons[p] == 3){
        if(p < j - VRNA_GQUAD_MIN_BOX_SIZE){
          minq  = j - i + p - MAXLOOP - 2;
          c0    = p + VRNA_GQUAD_MIN_BOX_SIZE - 1;
          minq  = MAX2(c0, minq);
          c0    = j - 3;
          maxq  = p + VRNA_GQUAD_MAX_BOX_SIZE + 1;
          maxq  = MIN2(c0, maxq);
          for(q = minq; q < maxq; q++){
            if(S_cons[q] != 3) continue;
            if(cij == mm + ggg[indx[q] + p] + n_seq * P->internal_loop[j - q - 1]){
              i = p; j=q;
              goto repeat_gquad_comparative;
            }
          }
        }
      }
      q = j1;
      if(S_cons[q] == 3)
        for(p = i1 + 3; p < j - VRNA_GQUAD_MIN_BOX_SIZE; p++){
          l1    = p - i - 1;
          if(l1>MAXLOOP) break;
          if(S_cons[p] != 3) continue;
          if(cij == mm + ggg[indx[q] + p] + n_seq * P->internal_loop[l1]){
            i = p; j = q;
            goto repeat_gquad_comparative;
          }
        }
    }

    if(hc->matrix[indx[j] + i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP){
      mm = n_seq*P->MLclosing;
      switch(dangle_model){
        case 0:   for(ss = 0; ss < n_seq; ss++){
                    tt = rtype[type[ss]];
                    mm += E_MLstem(tt, -1, -1, P);
                  }
                  break;
        default:  for(ss = 0; ss < n_seq; ss++){
                    tt = rtype[type[ss]];
                    mm += E_MLstem(tt, S5[ss][j], S3[ss][i], P);
                  }
                  break;
      }

      if(sc)
        for(ss = 0; ss < n_seq; ss++)
          if(sc[ss]){
            if(sc[ss]->energy_bp)
              mm += sc[ss]->energy_bp[indx[j] + i];
          }

      bt_stack[s+1].ml  = bt_stack[s+2].ml = 1;

      for (k = i1+turn+1; k < j1-turn-1; k++){
        if(cij == fML[indx[k]+i1] + fML[indx[j1]+k+1] + mm) break;
      }

      if (k<=j-3-turn) { /* found the decomposition */
        bt_stack[++s].i = i1;
        bt_stack[s].j   = k;
        bt_stack[++s].i = k+1;
        bt_stack[s].j   = j1;
      } else {
          vrna_message_error("backtracking failed in repeat");
      }
    } else
      vrna_message_error("backtracking failed in repeat");

    continue; /* this is a workarround to not accidentally proceed in the following block */

  repeat_gquad_comparative:
    /*
      now we do some fancy stuff to backtrace the stacksize and linker lengths
      of the g-quadruplex that should reside within position i,j
    */
    {
      int cnt1, l[3], L, size;
      size = j-i+1;

      for(L=0; L < VRNA_GQUAD_MIN_STACK_SIZE;L++){
        if(S_cons[i+L] != 3) break;
        if(S_cons[j-L] != 3) break;
      }

      if(L == VRNA_GQUAD_MIN_STACK_SIZE){
        /* continue only if minimum stack size starting from i is possible */
        for(; L<=VRNA_GQUAD_MAX_STACK_SIZE;L++){
          if(S_cons[i+L-1] != 3) break; /* break if no more consecutive G's 5' */
          if(S_cons[j-L+1] != 3) break; /* break if no more consecutive G'1 3' */
          for(    l[0] = VRNA_GQUAD_MIN_LINKER_LENGTH;
                  (l[0] <= VRNA_GQUAD_MAX_LINKER_LENGTH)
              &&  (size - 4*L - 2*VRNA_GQUAD_MIN_LINKER_LENGTH - l[0] >= 0);
              l[0]++){
            /* check whether we find the second stretch of consecutive G's */
            for(cnt1 = 0; (cnt1 < L) && (S_cons[i+L+l[0]+cnt1] == 3); cnt1++);
            if(cnt1 < L) continue;
            for(    l[1] = VRNA_GQUAD_MIN_LINKER_LENGTH;
                    (l[1] <= VRNA_GQUAD_MAX_LINKER_LENGTH)
                &&  (size - 4*L - VRNA_GQUAD_MIN_LINKER_LENGTH - l[0] - l[1] >= 0);
                l[1]++){
              /* check whether we find the third stretch of consectutive G's */
              for(cnt1 = 0; (cnt1 < L) && (S_cons[i+2*L+l[0]+l[1]+cnt1] == 3); cnt1++);
              if(cnt1 < L) continue;

              /*
                the length of the third linker now depends on position j as well
                as the other linker lengths... so we do not have to loop too much
              */
              l[2] = size - 4*L - l[0] - l[1];
              if(l[2] < VRNA_GQUAD_MIN_LINKER_LENGTH) break;
              if(l[2] > VRNA_GQUAD_MAX_LINKER_LENGTH) continue;
              /* check for contribution */
              if(ggg[indx[j]+i] == E_gquad_ali(i, L, l, (const short **)S, n_seq, P)){
                int a;
                /* fill the G's of the quadruplex into base pair stack */
                for(a=0;a<L;a++){
                  bp_stack[++b].i = i+a;
                  bp_stack[b].j   = i+a;
                  bp_stack[++b].i = i+L+l[0]+a;
                  bp_stack[b].j   = i+L+l[0]+a;
                  bp_stack[++b].i = i+L+l[0]+L+l[1]+a;
                  bp_stack[b].j   = i+L+l[0]+L+l[1]+a;
                  bp_stack[++b].i = i+L+l[0]+L+l[1]+L+l[2]+a;
                  bp_stack[b].j   = i+L+l[0]+L+l[1]+L+l[2]+a;
                }
                goto repeat_gquad_comparative_exit;
              }
            }
          }
        }
      }
      vrna_message_error("backtracking failed in repeat_gquad_comparative");
    }
  repeat_gquad_comparative_exit:
    __asm("nop");

  }

  bp_stack[0].i = b;    /* save the total number of base pairs */
  free(type);
}


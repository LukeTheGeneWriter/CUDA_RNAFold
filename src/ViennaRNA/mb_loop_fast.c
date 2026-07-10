
/* Modifications for eventual CUDA version $Revision: 1.2 $
WBL  3 Dec 2017 investigate data dependence in E_mb_loop_fast
  split off multibranch_loops.c r1.10 for time being
WBL 20 Jan 2017 Add const, remove unsupported options
WBL 15 Jan 2017 Add STUB
 */

#include <assert.h>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include "ViennaRNA/utils.h"
#include "ViennaRNA/fold_vars.h"
#include "ViennaRNA/energy_par.h"
#include "ViennaRNA/constraints.h"
#include "ViennaRNA/exterior_loops.h"
#include "ViennaRNA/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/unstructured_domains.h"
#include "ViennaRNA/multibranch_loops.h"

#ifdef STUB
#include "stub2.h"
#endif /*STUB*/

#ifdef ON_SAME_STRAND
#undef ON_SAME_STRAND
#endif

#define ON_SAME_STRAND(I,J,C)  (((I)>=(C))||((J)<(C)))

//Renamed E_mb_loop_fast

PUBLIC int
mb_loop_fast( vrna_fold_compound_t *vc,
                int i,
                int j/*,
                int *dmli1,
                int *dmli2*/){

  unsigned char type, tt;
  char          *ptype, *hc, eval_loop, el;
  short         S_i1, S_j1, *S;
  int           decomp, en, e, cp, *indx, *hc_up, *fc, ij, hc_decompose,
                dangle_model, *rtype;
  vrna_sc_t     *sc;
  vrna_param_t  *P;
  vrna_ud_t     *domains_up;
#ifdef WITH_GEN_HC
  vrna_callback_hc_evaluate *f;
#endif

  cp            = vc->cutpoint;
  ptype         = vc->ptype;
  S             = vc->sequence_encoding;
  indx          = vc->jindx;
  hc            = vc->hc->matrix;
  hc_up         = vc->hc->up_ml;
  sc            = vc->sc;
  fc            = vc->matrices->fc;
  P             = vc->params;
  ij            = indx[j] + i;
  hc_decompose  = hc[ij];
  dangle_model  = P->model_details.dangles;
  rtype         = &(P->model_details.rtype[0]);
  type          = (unsigned char)ptype[ij];
  domains_up    = vc->domains_up;
  /* init values */
  e             = 0;//INF;
  decomp        = 0;//INF;

#ifdef WITH_GEN_HC
  f  = vc->hc->f;
#endif

  if(cp < 0){
    S_i1    = S[i+1];
    S_j1    = S[j-1];
  } else {
    S_i1  = ON_SAME_STRAND(i, i + 1, cp) ? S[i+1] : -1;
    S_j1  = ON_SAME_STRAND(j - 1, j, cp) ? S[j-1] : -1;
  }

  if((S_i1 >= 0) && (S_j1 >= 0)){ /* regular multi branch loop */

    eval_loop = hc_decompose & VRNA_CONSTRAINT_CONTEXT_MB_LOOP;
    el        = eval_loop;

#ifdef WITH_GEN_HC
    if(f)
      el = (f(i, j, i+1, j-1, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

    /* new closing pair (i,j) with mb part [i+1,j-1] */
    if(el){
      //decomp = dmli1[j-1];
      tt = rtype[type];

      if(tt == 0)
        tt = 7;

      if(decomp != INF){
	assert(dangle_model==2); //ensure path gcov shows to always be taken
        switch(dangle_model){
          /* no dangles */
          case 0:   decomp += E_MLstem(tt, -1, -1, P);
                    assert(!sc); //ensure path gcov shows to always be taken
                    if(sc){
                      if(sc->energy_bp)
                        decomp += sc->energy_bp[ij];

                      if(sc->f)
                        decomp += sc->f(i, j, i+1, j-1, VRNA_DECOMP_PAIR_ML, sc->data);
                    }
                    break;

          /* double dangles */
          case 2:   decomp += E_MLstem(tt, S_j1, S_i1, P);
                    if(sc){
                      if(sc->energy_bp)
                        decomp += sc->energy_bp[ij];

                      if(sc->f)
                        decomp += sc->f(i, j, i+1, j-1, VRNA_DECOMP_PAIR_ML, sc->data);
                    }
                    break;

          /* normal dangles, aka dangles = 1 || 3 */
          default:  decomp += E_MLstem(tt, -1, -1, P);
                    if(sc){
                      if(sc->energy_bp)
                        decomp += sc->energy_bp[ij];

                      if(sc->f)
                        decomp += sc->f(i, j, i+1, j-1, VRNA_DECOMP_PAIR_ML, sc->data);
                    }
                    break;
        }
      }
    }

    if(dangle_model % 2){  /* dangles == 1 || dangles == 3 */
      assert(0); //ensure path gcov shows to always be taken
#ifndef STUB
      el = eval_loop;
      el = (hc_up[i + 1] > 0) ? el : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        el = (f(i, j, i+2, j-1, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

      /* new closing pair (i,j) with mb part [i+2,j-1] */
      if(el){
        if(dmli2[j-1] != INF){
          tt = rtype[type];

          if(tt == 0)
            tt = 7;

          en = dmli2[j-1] + E_MLstem(tt, -1, S_i1, P) + P->MLbase;
          if(sc){
            if(sc->energy_up)
              en += sc->energy_up[i+1][1];

            if(sc->energy_bp)
              en += sc->energy_bp[ij];

            if(sc->f)
              en += sc->f(i, j, i+2, j-1, VRNA_DECOMP_PAIR_ML, sc->data);
          }
          decomp = MIN2(decomp, en);
        }
      }

      el = eval_loop;
      el = ((hc_up[i + 1] > 0) && (hc_up[j-1] > 0)) ? el : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        el = (f(i, j, i+2, j-2, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

      /* new closing pair (i,j) with mb part [i+2.j-2] */
      if(el){
        if(dmli2[j-2] != INF){
          tt = rtype[type];

          if(tt == 0)
            tt = 7;

          en = dmli2[j-2] + E_MLstem(tt, S_j1, S_i1, P) + 2*P->MLbase;
          if(sc){
            if(sc->energy_up)
              en += sc->energy_up[i+1][1]
                    + sc->energy_up[j-1][1];

            if(sc->energy_bp)
              en += sc->energy_bp[ij];

            if(sc->f)
              en += sc->f(i, j, i+2, j-2, VRNA_DECOMP_PAIR_ML, sc->data);
          }
          decomp = MIN2(decomp, en);
        }
      }

      el = eval_loop;
      el = (hc_up[j-1] > 0) ? el : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        el = (f(i, j, i+1, j-2, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

      /* new closing pair (i,j) with mb part [i+1, j-2] */
      if(el){
        if(dmli1[j-2] != INF){
          tt = rtype[type];

          if(tt == 0)
            tt = 7;

          en = dmli1[j-2] + E_MLstem(tt, S_j1, -1, P) + P->MLbase;
          if(sc){
            if(sc->energy_up)
              en += sc->energy_up[j-1][1];

            if(sc->energy_bp)
              en += sc->energy_bp[ij];

            if(sc->f)
              en += sc->f(i, j, i+1, j-2, VRNA_DECOMP_PAIR_ML, sc->data);
          }
          decomp = MIN2(decomp, en);
        }
      }
#endif /*not STUB*/

    } /* end if dangles % 2 */

    if(decomp != INF)
      e = decomp + P->MLclosing;

  } /* end regular multibranch loop */

  if(!ON_SAME_STRAND(i, j, cp)){ /* multibrach like cofold structure with cut somewhere between i and j */
    assert(dangle_model==2); //ensure path gcov shows to always be taken
#ifndef STUB
    eval_loop = hc_decompose & VRNA_CONSTRAINT_CONTEXT_MB_LOOP;
    el        = eval_loop;

#ifdef WITH_GEN_HC
    if(f)
      el = (f(i, j, i+1, j-1, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

    if(el){
      if((fc[i+1] != INF) && (fc[j-1] != INF)){
        decomp = fc[i+1] + fc[j-1];
        tt = rtype[type];

        if(tt == 0)
          tt = 7;

        switch(dangle_model){
          case 0:   decomp += E_ExtLoop(tt, -1, -1, P);
                    break;
          case 2:   decomp += E_ExtLoop(tt, S_j1, S_i1, P);
                    break;
          default:  decomp += E_ExtLoop(tt, -1, -1, P);
                    break;
        }
      }
    }

    if(dangle_model % 2){ /* dangles == 1 || dangles == 3 */
      el  = eval_loop;
      el  = (hc_up[i+1] > 0) ? el : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        el = (f(i, j, i+2, j-1, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

      if(el){
        if((fc[i+2] != INF) && (fc[j-1] != INF)){
          tt = rtype[type];

          if(tt == 0)
            tt = 7;

          en     = fc[i+2] + fc[j-1] + E_ExtLoop(tt, -1, S_i1, P);
          decomp = MIN2(decomp, en);
        }
      }

      el  = eval_loop;
      el  = (hc_up[j-1] > 0) ? el : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        el = (f(i, j, i+1, j-2, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

      if(el){
        if((fc[i+1] != INF) && (fc[j-2] != INF)){
          tt = rtype[type];

          if(tt == 0)
            tt = 7;

          en     = fc[i+1] + fc[j-2] + E_ExtLoop(tt, S_j1, -1, P);
          decomp = MIN2(decomp, en);
        }
      }

      el  = eval_loop;
      el  = ((hc_up[i+1] > 0) && (hc_up[j-1] > 0)) ? el : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        el = (f(i, j, i+2, j-2, VRNA_DECOMP_PAIR_ML, vc->hc->data)) ? el : (char)0;
#endif

      if(el){
        if((fc[i+2] != INF) && (fc[j-2] != INF)){
          tt = rtype[type];

          if(tt == 0)
            tt = 7;

          en     = fc[i+2] + fc[j-2] + E_ExtLoop(tt, S_j1, S_i1, P);
          decomp = MIN2(decomp, en);
        }
      }
    }

    e = MIN2(e, decomp);
#endif /*not STUB*/
  }
  return e;
}

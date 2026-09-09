/*
 *  The CUDA MFE backend's attachment point and routing guard.
 *
 *  This file deliberately contains no CUDA code. It decides WHETHER a fold
 *  compound may go to the device, and that decision is the part which has to be
 *  right: an accelerator that answers a fold compound it does not fully support
 *  returns a plausible, self-consistent, WRONG structure. That failure mode has
 *  already been observed on this project's 2.3.0 base, where -g produced
 *  answers up to 31 kcal/mol short of the optimum while passing every
 *  self-consistency check, and -c silently returned the linear answer.
 *
 *  So the rule here is: enumerate what is SUPPORTED, decline everything else,
 *  and never let an unrecognised model detail fall through to the device.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/grammar/mfe.h"
#include "ViennaRNA/mfe/global.h"      /* vrna_mfe_batch_backend_set() */
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/constraints/soft.h"

#include "ViennaRNA/intern/grammar_dat.h"

#include "ViennaRNA/mfe/cuda/engine.h"

#ifdef VRNA_WITH_CUDA
/* implemented in device.cu */
unsigned int vrna_cuda_device_count(void);
#endif

/*
 * Whether a device-side matrix fill exists yet. The seam, the routing guard and
 * the build glue land before the kernels do, so until the sweep is ported this
 * is 0 and the engine declines every fold compound -- including the ones it
 * fully supports. That keeps the intermediate states of the port honest: the
 * backend can be attached, and the answer is still upstream's own.
 */
#define VRNA_CUDA_HAVE_SWEEP 0


PUBLIC unsigned int
vrna_cuda_devices(void)
{
#ifdef VRNA_WITH_CUDA
  return vrna_cuda_device_count();
#else
  return 0;
#endif
}


PUBLIC unsigned int
vrna_cuda_engine_supports(vrna_fold_compound_t  *fc,
                          const char            **reason)
{
  const char  *why = NULL;
  vrna_md_t   *md;
  unsigned int i, j;

#define DECLINE(msg) do { why = (msg); goto done; } while (0)

  if (fc == NULL)
    DECLINE("no fold compound");

  /* comparative folding has a different recursion entirely */
  if (fc->type != VRNA_FC_TYPE_SINGLE)
    DECLINE("not a single-sequence fold compound");

  /* multistrand adds fms5/fms3 and per-nucleotide strand bookkeeping */
  if (fc->strands > 1)
    DECLINE("multiple strands");

  md = &(fc->params->model_details);

  /*
   * Model details. Each of these changes the recursion or the energies in a
   * way the device path does not reproduce today.
   */
  if (md->dangles != 2)
    DECLINE("dangle model other than 2");

  if (md->gquad)
    DECLINE("G-quadruplexes (see PORT_GQUAD_SPEC.md)");

  if (md->circ)
    DECLINE("circular RNA (see PORT_CIRC_SPEC.md)");

  /* noLP is ACCEPTED as of 2026-09-07. It was declined because the sweep
   * filtered ptype for it but never applied the recursion constraint, which
   * left the matrix fill and the backtrack disagreeing with each other by 87
   * to 300 kcal/mol -- while the reported ENERGY agreed with upstream on 45 of
   * 60 records, which is why an energy comparison called it "3 of 12".
   * new_c_kernel now writes cc1[j-1]+stackEnergy into c and carries the
   * unconstrained value in cc, exactly as mfe/mfe.c:4413 does.
   * Bar: tests/mfe_cuda_nolp.ts and tools/verify_nolp_parity.sh, neither of
   * which is an energy comparison. See PORT_NOLP_SPEC.md. */

  if (md->noGUclosure)
    DECLINE("noClosingGU");

  /* logML stays DECLINED. It measured 12/12 at 160 nt in the scope probe, but
   * that is a 12-sequence sample and RNAfold has NO --logML flag, so there is
   * no way to bar it from the CLI at all -- the parity check that appeared to
   * cover it was comparing two empty outputs from a rejected option. Lifting a
   * guard needs evidence; this has a sample and a broken test. */
  if (md->logML)
    DECLINE("logarithmic multibranch loop scaling");

  /*
   * uniq_ML is ACCEPTED as of the fM1 post-pass in mfe_cuda.c.
   *
   * It was declined for a real reason, not a suspected one: the MFE answer is
   * correct under uniq_ML because the recursion never reads fM1, but the sweep
   * left fM1 entirely INF, and this guard is what vrna_mfe_batch() consults. A
   * caller could fold a batch with uniq_ML and then call vrna_subopt(), which
   * does read fM1.
   *
   * backtrack_one_slot() now reconstructs fM1 on the host from the `c` triangle
   * it has just fetched, using upstream's own per-cell helper and upstream's
   * own loop order. tests/mfe_cuda_fm1.ts compares the result against a
   * compound folded entirely by upstream, cell for cell.
   */

  if (md->energy_set != 0)
    DECLINE("non-default energy set");

  /*
   * NON-STANDARD BASE PAIRS (--nsp) are declined, and this closes a LIVE wrong
   * answer rather than documenting a known gap: measured on 30 x 80-1240 nt,
   * --nsp=GA disagreed with upstream on 28 records, deterministically, worse on
   * 27 and better on 1. Not ignored -- partially applied, which is the worst of
   * the three possibilities because it looks like it is working.
   *
   * The mechanism is in int_loop.cu's Energy(), not here. It resolves the two
   * interior-loop pair types as
   *
   *     type   = Ptype(...,i,j)        raw, with no 0->7 promotion
   *     type_2 = Ptype(...,q,p)        the index swapped, in place of rtype[]
   *
   * where upstream uses vrna_get_ptype() (alphabet.c:482, tt==0 ? 7 : tt) and
   * then rtype[]. BOTH shortcuts are exact identities at default settings --
   * rtype[] is BUILT as rtype[pair[i][j]] = pair[j][i] (model.c:1100), and a
   * ptype-0 cell is refused by the hard constraint mask before Energy() runs --
   * which is why the port has been byte-identical without them.
   *
   * --nsp breaks the first identity at model.c:1104, where rtype[7] is FORCED
   * to 7 after the derivation loop. An asymmetric spec (--nsp=GA, no leading
   * '-') leaves pair[A][G] == 0 while pair[G][A] == 7, so upstream reads row 7
   * of stack/int11/int21/int22/mismatchI and the device reads row 0. Row 7 is
   * the non-standard row (NST/NSM, default.c:53-56); row 0 is not.
   *
   * TEST THE EFFECT, NOT THE FIELD. md->nonstandards is only one of the routes
   * in -- the deprecated global, a copied md, or a hand-built one all reach the
   * same place -- and every one of them lands in md->pair via vrna_md_update().
   * Default BP_pair (pair_mat.h:21-30) holds only 0..6, so a 7 anywhere in the
   * pair table means non-standard pairs are enabled, however they got there.
   * This is the -C lesson mirrored: there the queued depot was the honest
   * thing to test and the materialised matrix was not; here the materialised
   * table is honest and the request field is the one that can be bypassed.
   *
   * Lifting this needs Energy() fixed AND a byte-identical run over MIXED
   * lengths for both --nsp=GA and --nsp="-GA" (the symmetric form masks the
   * rtype divergence and would pass on its own). See
   * PORT_NSP_PARAMFILE_SCOPE.md.
   */
  for (i = 0; i <= MAXALPHA; i++)
    for (j = 0; j <= MAXALPHA; j++)
      if (md->pair[i][j] == 7)
        DECLINE("non-standard base pairs (--nsp)");

  /*
   * NOT md->window_size: vrna_fold_compound() sets both window_size and
   * max_bp_span to the sequence length for an ordinary GLOBAL fold, so testing
   * `window_size > 0` declines everything, including the default model. What
   * actually distinguishes a sliding-window fold is the hard constraint layout.
   */
  if ((fc->hc != NULL) && (fc->hc->type != VRNA_HC_DEFAULT))
    DECLINE("sliding window hard constraints");

  /* a genuinely restricted span, as opposed to the default span == length */
  if ((md->max_bp_span > 0) && ((unsigned int)md->max_bp_span < fc->length))
    DECLINE("restricted base pair span");

  /*
   * Salt is ACCEPTED. The hot multibranch kernel needed nothing at all for it:
   * params.c:640-645 folds SaltMLbase/SaltMLclosing into MLbase, MLclosing and
   * MLintern at parameter-init time, and modular_decomposition.cu reads only
   * those, so it inherits the correction without knowing salt exists. Hairpins
   * and internal loops each take one added term, from a table the host builds
   * with upstream's own vrna_salt_loop_int(). See PORT_SALT_SPEC.md and
   * tools/verify_salt_parity.sh.
   */



  /*
   * Constraints. Soft constraints reach the recursion as arbitrary host
   * callbacks, which cannot run in a kernel at all; hard constraint callbacks
   * are the same problem.
   */
  if (fc->sc != NULL)
    DECLINE("soft constraints");

  if ((fc->hc != NULL) && (fc->hc->f != NULL))
    DECLINE("hard constraint callback");

  /*
   * HARD CONSTRAINTS, in their plain bitmask form, are declined too -- and this
   * one closed a live hole rather than documenting a known gap.
   *
   * Until this check existed, nothing here fired for a -C style constraint: it
   * leaves hc->type at VRNA_HC_DEFAULT and hc->f at NULL, so a constrained fold
   * compound passed the guard and vrna_mfe_batch() folded it on the device.
   * Measured on a 12x80 nt batch with a forced-unpaired block, the device
   * returned -14.30 for a structure worth -5.40 -- its matrix fill and its
   * backtrack did not even agree with each other.
   *
   * The sweep is not simply ignoring the constraint matrix; it packs bitmasks
   * from hc->mx and honours some of it. It does not reproduce upstream's full
   * set of loop-context checks, and the fork's own host sweep is wrong in the
   * SAME way, which is why RNA_ROW_VERIFY reports 58128 cells checked and zero
   * mismatches on the very folds that come out wrong. Device-against-host
   * cannot see a defect the two share; only upstream's fill_arrays can.
   *
   * The depot is the right thing to test. vrna_constraints_add() only QUEUES a
   * constraint -- hc->mx, ptype and up_* are all still byte-identical to an
   * unconstrained compound until vrna_fold_compound_prepare() materialises them
   * -- and this guard runs BEFORE par_mfe() prepares. So comparing hc->mx here
   * would compare two identical matrices and accept everything, which is
   * precisely the shape of the empty-vs-empty checks that have already fooled
   * this project. hc->depot is non-NULL from the moment a constraint is queued.
   *
   * tools/verify_constraint_parity.sh is the bar this has to pass before the
   * check can be lifted; PORT_ACCELERATION_SCOPE.md records the cost.
   */
  if ((fc->hc != NULL) && (fc->hc->depot != NULL))
    DECLINE("hard structure constraints (see tools/verify_constraint_parity.sh)");

  if (fc->domains_up != NULL)
    DECLINE("unstructured domains (ligand motifs)");

  if (fc->domains_struc != NULL)
    DECLINE("structured domains");

  /*
   * Auxiliary grammar rules are combined into the recursion cell by cell on the
   * host. A device fill would silently drop them. The engine itself lives in
   * the same structure, so only the RULE arrays are checked here.
   */
  if (fc->aux_grammar != NULL) {
    if ((vrna_array_size(fc->aux_grammar->f)) ||
        (vrna_array_size(fc->aux_grammar->c)) ||
        (vrna_array_size(fc->aux_grammar->m)) ||
        (vrna_array_size(fc->aux_grammar->m1)) ||
        (vrna_array_size(fc->aux_grammar->m2)) ||
        (vrna_array_size(fc->aux_grammar->aux)))
      DECLINE("auxiliary grammar rules");
  }

done:
#undef DECLINE

  if (reason)
    *reason = why;

  return (why == NULL) ? 1 : 0;
}


PRIVATE int
cuda_engine_cb(vrna_fold_compound_t *fc,
               int                  *energy,
               void                 *data)
{
  const char *reason = NULL;

  (void)data;
  (void)energy;

  if (!vrna_cuda_engine_supports(fc, &reason))
    return 0;                     /* decline: the library folds it itself */

#if VRNA_CUDA_HAVE_SWEEP
  return vrna_cuda_fill_matrices(fc, energy);
#else
  /*
   * Supported, but there is no device fill yet. Declining is the only honest
   * answer: returning anything here would be inventing one.
   */
  return 0;
#endif
}


#ifdef VRNA_WITH_CUDA
/* the batch sweep, declared in mfe/cuda/stub2.h (internal, not installed) */
extern void
par_mfe(const int                     nfiles,
        const vrna_fold_compound_t  **VC,
        const char                  **Structure,
        float                        *EN,
        const int                     cpu_queue_threads);
#endif


PRIVATE int
cuda_batch_cb(vrna_fold_compound_t  **fcs,
              size_t                  n,
              char                  **structures,
              float                  *energies,
              void                   *data)
{
#ifdef VRNA_WITH_CUDA
  const char  *reason = NULL;
  size_t      i;

  (void)data;

  if ((fcs == NULL) || (n == 0) || (structures == NULL) || (energies == NULL))
    return 0;

  /* Decline the WHOLE batch unless every record is supported. Splitting it
   * would be a silent policy decision about which records the caller gets
   * accelerated; declining leaves that choice with the caller, which already
   * knows how to fold them singly. */
  for (i = 0; i < n; i++) {
    if (!vrna_cuda_engine_supports(fcs[i], &reason))
      return 0;
  }

  par_mfe((int)n, (const vrna_fold_compound_t **)fcs,
          (const char **)structures, energies, 0);

  return 1;
#else
  (void)fcs; (void)n; (void)structures; (void)energies; (void)data;
  return 0;
#endif
}


PUBLIC unsigned int
vrna_cuda_register_batch_backend(void)
{
  if (vrna_cuda_devices() == 0)
    return 0;

  return vrna_mfe_batch_backend_set(&cuda_batch_cb, NULL);
}


PUBLIC unsigned int
vrna_cuda_attach(vrna_fold_compound_t *fc)
{
  if (fc == NULL)
    return 0;

  if (vrna_cuda_devices() == 0)
    return 0;

  return vrna_gr_set_inside_engine(fc, &cuda_engine_cb, NULL, NULL, NULL);
}

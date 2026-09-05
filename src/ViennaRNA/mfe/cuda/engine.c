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

  if (md->noLP)
    DECLINE("noLP");

  if (md->noGUclosure)
    DECLINE("noClosingGU");

  if (md->logML)
    DECLINE("logarithmic multibranch loop scaling");

  if (md->uniq_ML)
    DECLINE("unique multibranch loop decomposition");

  if (md->energy_set != 0)
    DECLINE("non-default energy set");

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

  if (md->salt != VRNA_MODEL_DEFAULT_SALT)
    DECLINE("salt correction (see PORT_SALT_SPEC.md)");

  /*
   * Constraints. Soft constraints reach the recursion as arbitrary host
   * callbacks, which cannot run in a kernel at all; hard constraint callbacks
   * are the same problem. Only the bitmask form of hard constraints, derived
   * from the sequence, is reproducible on the device.
   */
  if (fc->sc != NULL)
    DECLINE("soft constraints");

  if ((fc->hc != NULL) && (fc->hc->f != NULL))
    DECLINE("hard constraint callback");

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

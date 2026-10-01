#ifndef VIENNA_RNA_PACKAGE_MFE_CUDA_ENGINE_H
#define VIENNA_RNA_PACKAGE_MFE_CUDA_ENGINE_H

#include "ViennaRNA/fold_compound.h"

/**
 *  @file     ViennaRNA/mfe/cuda/engine.h
 *  @brief    A CUDA backend for the MFE matrix fill
 *
 *  The backend attaches to a fold compound through the inside-engine seam
 *  (vrna_gr_set_inside_engine(), ViennaRNA/grammar/mfe.h). It fills the same
 *  matrices the default implementation fills, and DECLINES any fold compound
 *  whose model it does not fully support, in which case the library computes
 *  the answer itself exactly as if no backend were attached.
 *
 *  Declining is the default for anything unrecognised. An accelerator that
 *  answers a fold compound it does not fully support returns a different
 *  structure rather than an error, which is far worse than being slow.
 */

/**
 *  @brief  TEST HOOK: is the named routing-guard check lifted?
 *
 *  Reads RNA_ENGINE_ALLOW, a comma-separated list of check ids (or "all"), and
 *  announces every lift on stderr. It exists so that a declined option can be
 *  MEASURED from a shipped binary instead of a scratch build -- five options
 *  moved from DECLINED to ACCELERATED that way. A lifted check routes a fold to
 *  a device path known not to support it: the answer may be silently wrong.
 *
 *  @param  id  The check id, e.g. "hc", "energy_set", "soft", "motif"
 *  @return     Non-zero if that check should be skipped
 */
int
vrna_cuda_engine_allow(const char *id);

/**
 *  @brief  Is a usable CUDA device present?
 *
 *  @return The number of usable devices; 0 if the library was built without
 *          CUDA support or no device is available.
 */
unsigned int
vrna_cuda_devices(void);


/**
 *  @brief  Attach the CUDA MFE backend to a fold compound
 *
 *  Binds the backend through vrna_gr_set_inside_engine(). Whether any given
 *  fold compound is actually handled on the device is decided per call, at
 *  fold time, by vrna_cuda_engine_supports().
 *
 *  @param  fc  The fold compound to attach the backend to
 *  @return     Non-zero on success, 0 if the backend could not be attached
 *              (no CUDA support compiled in, no device, or an engine is
 *              already bound to @p fc)
 */
unsigned int
vrna_cuda_attach(vrna_fold_compound_t *fc);


/**
 *  @brief  Register the CUDA backend as the library's batch MFE backend
 *
 *  After this, vrna_mfe_batch() folds supported batches on the device and
 *  everything else exactly as the library would anyway. A caller therefore
 *  never has to name CUDA at any point after this one call -- which is the
 *  whole idea: the accelerator is a backend, not a fork of the driver.
 *
 *  @return Non-zero on success; 0 if there is no usable device or the library
 *          was built without CUDA support
 */
unsigned int
vrna_cuda_register_batch_backend(void);


/**
 *  @brief  Would the CUDA backend handle this fold compound?
 *
 *  Exposed separately from the engine callback so that callers batching many
 *  records can ask the question BEFORE committing a record to a GPU batch, and
 *  so the decision can be tested directly.
 *
 *  @param  fc      The fold compound to test
 *  @param  reason  If non-NULL, receives a static string naming the first
 *                  unsupported property found, or NULL when supported
 *  @return         Non-zero if the backend would handle @p fc
 */
unsigned int
vrna_cuda_engine_supports(vrna_fold_compound_t  *fc,
                          const char            **reason);


/**
 *  @brief  Leave each record's MFE matrices populated after a device fold
 *
 *  The batch backend releases every record's host c/fML before the sweep and
 *  backtracks each record from a pooled scratch pair, so after a device fold the
 *  fold compound's c and fML are NULL -- fine for a caller that only wants the
 *  structure and energy, fatal for one that then calls vrna_backtrack5() or reads
 *  the matrices, as upstream's vrna_mfe() lets it. With this switch on, each
 *  record gets its own copies of the device's triangles (c, fML, and fM2_real when
 *  circular), exactly the extents vrna_mfe() would have left. Costs one triangle
 *  copy per record; off by default. Used by the Python binding's fold_compound.mfe().
 *
 *  @param  on  Non-zero to keep the matrices, zero for the default behaviour
 */
void
vrna_cuda_keep_matrices(int on);


/**
 *  @brief  Whether vrna_cuda_keep_matrices() is on
 */
int
vrna_cuda_keeping_matrices(void);


/**
 *  @brief  How many batches the device has folded in this process
 *
 *  Positive evidence that a call used the GPU: a fold that falls back to the host
 *  gives the identical answer, so the answer alone never proves which path ran.
 *  Always 0 in a build without CUDA.
 */
unsigned long
vrna_cuda_device_batches(void);

#endif

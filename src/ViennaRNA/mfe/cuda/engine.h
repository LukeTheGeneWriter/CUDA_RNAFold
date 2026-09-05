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

#endif

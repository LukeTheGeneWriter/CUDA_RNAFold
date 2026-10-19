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
 *  @brief  Total memory of the device the backend uses, in bytes; 0 without one
 */
size_t
vrna_cuda_device_memory(void);


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


/**
 *  @brief  Whether RNA_GPU=0 is set, the run-time off switch
 *
 *  The batch backend checks this at every batch and declines when it is set, so the
 *  host folds the batch with upstream's vrna_mfe().
 */
int
vrna_cuda_switched_off(void);


/**
 *  @brief  Silence the backend's routine diagnostics
 *
 *  The backend reports its configuration, sweep shape and timings on stderr: one
 *  line per decision. RNAfold and the verification harnesses depend on those
 *  lines, so they are on by default. A library caller such as the Python binding
 *  can turn them off. Errors, refusals, and lines that only appear when an RNA_*
 *  knob is set still print.
 *
 *  @param  on  Non-zero to silence, zero for the default behaviour
 */
void
vrna_cuda_set_quiet(int on);


/**
 *  @brief  Whether vrna_cuda_set_quiet() is on
 */
int
vrna_cuda_quiet(void);


/**
 *  @brief  Say the AUTO decisions a default run makes (int16 / int32, ...) on stderr
 *
 *  Off by default: a run that took the defaults prints only whether the device was
 *  used, as stock ViennaRNA prints nothing. RNAfold turns it on under --verbose.
 *  An EXPLICIT knob still announces itself regardless.
 */
void
vrna_cuda_set_verbose(int on);


int
vrna_cuda_verbose(void);


/**
 *  @name   Resource budgets (budget.c, PORT_BUDGETS.md)
 *
 *  How much host RAM, device memory and how many threads a run may take. A budget
 *  is a number (MB for RAM and VRAM, a count for threads) or a fraction of the
 *  machine's total: "quarter", "half", "threequarter", "full". It only lowers the
 *  ceiling: every consumer keeps its own soft limit, and "full" is no ceiling.
 *  Precedence: vrna_budget_set(), then RNA_RAM_BUDGET / RNA_VRAM_BUDGET /
 *  RNA_THREAD_BUDGET, then vrna_budget_default().
 *  @{
 */
#define VRNA_BUDGET_RAM     0
#define VRNA_BUDGET_VRAM    1
#define VRNA_BUDGET_THREADS 2

/** @brief  The chunk cell cap's default: 48 x (5601 x 5602 / 2) cells. RNAfold and the library
 *          share it; RNA_GPU_CHUNK_CELLS overrides, 0 disables (bin/gpu_batch.c explains it) */
#define VRNA_GPU_CHUNK_CELLS_DEFAULT 753000000ull

/** @brief  Set one budget; NULL or "" returns it to the environment / default. 1 = ok, 0 = unparseable */
int
vrna_budget_set(int         which,
                const char  *spec);


/** @brief  The program's default for all three (RNAfold "full", Python "half"). 1 = ok */
int
vrna_budget_default(const char *spec);


/** @brief  Host RAM ceiling in bytes; SIZE_MAX = none */
size_t
vrna_budget_ram(void);


/** @brief  Device memory ceiling in bytes for a card of @p device_total bytes; SIZE_MAX = none */
size_t
vrna_budget_vram(size_t device_total);


/** @brief  Thread budget, at least 1 */
int
vrna_budget_threads(void);


/** @brief  Whether a resource's budget is "full" (no ceiling) */
int
vrna_budget_is_full(int which);


/** @brief  MemTotal, MemAvailable (honours the RNA_HOST_AVAIL_MB test hook) and hardware threads */
size_t
vrna_budget_host_total(void);


size_t
vrna_budget_host_available(void);


int
vrna_budget_hw_threads(void);


/** @brief  One stderr line describing the budgets in force, once per process, and only when one is not full */
void
vrna_budget_report(size_t device_total);


/** @brief  The fitted host model (PORT_BUDGETS.md 8.4), in bytes */
size_t
vrna_budget_compound_bytes(size_t length);


size_t
vrna_budget_worker_bytes(size_t  length,
                         int     circ);


size_t
vrna_budget_hostfold_bytes(size_t length);


size_t
vrna_budget_base_bytes(int gpu);


/** @brief  Host memory a driver holds beside the chunk being backtracked (slots below) */
#define VRNA_BUDGET_HELD_NEXT_CHUNK 0   /* a build pipeline's next chunk */
#define VRNA_BUDGET_HELD_RETAINED   1   /* a chunk retained for the backtrack pipeline */
#define VRNA_BUDGET_HELD_CARRY      2   /* the previous chunk, freed but kept by the allocator */
void
vrna_budget_held(int     slot,
                 size_t  bytes);


size_t
vrna_budget_held_total(void);


/** @brief  This process's resident set now, in bytes (0 if unreadable) */
size_t
vrna_budget_rss(void);


/** @brief  How many backtrack workers fit the RAM budget beside @p need_now bytes (at least 1; @p want when none is set) */
int
vrna_budget_workers(int     want,
                    size_t  need_now,
                    size_t  max_length,
                    int     circ,
                    int     explicit_setting);


/** @brief  A pool's size capped at the thread budget (announced once per @p who); an explicit setting wins, with a warning */
int
vrna_budget_cap_threads(const char  *who,
                        int         want,
                        int         explicit_setting);


/** @brief  Draw @p want workers from the shared thread budget (an auto pool waits while it is spent and may get fewer); hand @p held back to _release() */
int
vrna_budget_threads_acquire(int want,
                            int explicit_setting,
                            int *held);


void
vrna_budget_threads_release(int held);


/** @} */

#endif

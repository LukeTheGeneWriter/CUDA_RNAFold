#ifndef RNAFOLD_CPU_QUEUE_H
#define RNAFOLD_CPU_QUEUE_H

/* New Jul 2026: CPU worker-pool for RNAfold's heterogeneous GPU+CPU
 * dispatch. Sequences routed here (see RNAfold.c's dispatch decision at the
 * point each vrna_fold_compound_t is built) fold concurrently with the GPU
 * batch path (par_mfe()), using vrna_mfe_cpu() (src/ViennaRNA/mfe.c) --
 * NOT vrna_mfe(), which inside this binary resolves to mfe_cuda.c's
 * version and is unsafe for VRNA_FC_TYPE_SINGLE fold compounds.
 *
 * v1 scope, deliberately: CPU-folded results print as their own block
 * after the GPU batch's output (not interleaved back into original input
 * order), and skip PS-plot output. Both are flagged fast-follows, not
 * oversights.
 */

#include <stdio.h>
#include "ViennaRNA/data_structures.h"

/* Starts the worker pool. n_threads<=0 disables the queue entirely --
 * rnafold_cpu_queue_submit() becomes a no-op returning 0 in that case, and
 * the caller should route every sequence to the GPU batch path instead.
 * md is snapshotted by value (read-only afterward, safe for concurrent
 * reads across worker threads) -- must already be fully configured (as
 * RNAfold.c's own md is, by the time the read loop starts). output is the
 * FILE* worker threads print to; a single internal mutex serializes prints
 * so concurrent workers don't interleave/garble each other's output.
 */
void rnafold_cpu_queue_init(int n_threads, const vrna_md_t *md, FILE *output);

/* Submit one sequence for CPU folding. seq_id/orig_sequence/rec_sequence
 * are NOT retained -- the queue strdup's its own copies, so the caller may
 * free/reuse its originals immediately after this call returns. Blocks
 * (briefly) if the bounded queue is momentarily full -- ordinary
 * producer/consumer backpressure, not an error.
 * Returns 1 if accepted (queue active), 0 if the queue is disabled
 * (n_threads<=0 at init) -- caller should fall back to the GPU batch path.
 */
int rnafold_cpu_queue_submit(const char *seq_id, const char *orig_sequence, const char *rec_sequence);

/* Signals shutdown, joins all worker threads (draining whatever's still
 * queued first), then prints CPU-folded results. Must be called exactly
 * once, after every GPU chunk (RNAfold.c streams the input and folds/
 * prints/frees it in chunks rather than all at once -- see
 * process_gpu_chunk() there) has finished printing its own output (see the
 * v1 output-ordering scope note above). Safe to call even if the queue was
 * disabled (n_threads<=0) -- no-op in that case.
 */
void rnafold_cpu_queue_shutdown(void);

#endif

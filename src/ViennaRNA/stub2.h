/* Modifications for eventual CUDA version $Revision: 1.13 $
WBL  8 Jan 2018 Extend linkage for CUDA interface in modular_decomposition.cu
WBL  3 Dec 2017 investigate data dependence in E_mb_loop_fast
  split off multibranch_loops.c r1.10 for time being
*/

#ifdef __cplusplus
extern "C" void
#else
PUBLIC void
#endif
choose_gpu(int argc, char **argv); //updates argc and argv

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_fML(const int nfiles,const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_fML(const int nfiles,
	 const int i, const int turn, const int length,
	 const int* energy_min);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_min_fML(const int nfiles, const int i, const int turn, const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
modular_decomposition_i(const int nfiles,
			const int i, const int turn, const int length,
		      //const int* indx,
		      //const int ijsize,  //for sanity checks
		      //const int* my_fML,
			int* DMLi);

// CUDA-graph-captured fusion of load_fML() + modular_decomposition_i() +
// load_min_fML() -- see definition in modular_decomposition.cu for why these
// three (and only these three) can be captured as a single graph.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_fML_modular_decomposition_load_min_fML(const int nfiles,
					     const int i, const int turn, const int length,
					     const int* energy_min, //in
					     int* DMLi);            //out

PUBLIC int
extend_fm_3p( const int i,
              const int j,
              const int *fm,
              const vrna_fold_compound_t *vc);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu(const int nfiles, const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu3(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size);

// teardown_gpu()/teardown_gpu2()/teardown_gpu3(): free the nfiles/length-
// scaled device buffers allocated by init_gpu()/init_gpu2()/init_gpu3() and
// reset each file's one-time-init guard, so the next batch's init_gpu*()
// call actually re-allocates at the new nfiles instead of no-op'ing. One per
// .cu file, mirroring each file's own init_gpu*() -- see modular_decomposition.cu/
// int_loop.cu/hp_mb_loop.cu for what each frees and what it deliberately
// leaves allocated (the fixed-size, nfiles/length-independent buffers).
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
teardown_gpu(void);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
teardown_gpu2(void);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
teardown_gpu3(void);

// *_bytes_per_file(): bytes of device memory the owning file needs for one
// additional sequence at the given length -- used by compute_max_gpu_batch()
// to size a GPU batch against free VRAM. Each file owns its own formula
// (mirrors its init_gpu*() cudaMalloc sizes exactly) rather than duplicating
// it elsewhere.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ size_t
#else
PUBLIC size_t
#endif
modular_decomposition_bytes_per_file(const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ size_t
#else
PUBLIC size_t
#endif
int_loop_bytes_per_file(const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ size_t
#else
PUBLIC size_t
#endif
hp_mb_loop_bytes_per_file(const int length);

// Queries free VRAM (cudaMemGetInfo) and returns the largest nfiles that
// fits within a safety margin of it at the given length, summing all three
// files' per-file costs -- or 0 if even min_batch sequences wouldn't fit
// (signal: route the remainder to the CPU queue instead). See
// modular_decomposition.cu for the safety-margin/grid-limit details.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ int
#else
PUBLIC int
#endif
compute_max_gpu_batch(const int length, const int min_batch);

// Per-row (fixed i, all j, all H) hairpin/multibranch/3'-extension energy
// kernel -- see hp_mb_loop.cu for why these three (and only these three) can
// be computed fresh each i instead of precomputed as a full nfiles*ijsize
// array.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
hp_mb_3p_i(const int nfiles, const vrna_fold_compound_t **VC,
	   const int i, const int turn, const int length,
	   int* energy_hp_row, int* energy_mb_row, int* energy_3p00_row); //all out

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_my_c(const int nfiles,
	  const int i, const int turn, const int length,
	  const int* min_e); //in

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
int_loop_i(const int nfiles,
	   const vrna_fold_compound_t **VC,
	   const int i, const int turn, const int length,
	   /*const int* indx, const int ijsize,
	   const char* hard_constraints, const int* my_c,*/
	   int* energy_min ); //out


PUBLIC void
par_mfe(const int nfiles,
	const vrna_fold_compound_t** VC,
	const char** Structure,
	float* EN, //out
	const int cpu_queue_threads); //RNAfold.c's already-resolved RNA_CPU_THREADS count, for RNA_BACKTRACK_THREADS=auto sizing

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
sanity(const vrna_fold_compound_t* vc0, const vrna_fold_compound_t* vc);

PRIVATE void
par_fill_arrays(const int nfiles, const vrna_fold_compound_t **VC, int* Energy);

int
mb_loop_fast( vrna_fold_compound_t *vc,
                int i,
                int j);

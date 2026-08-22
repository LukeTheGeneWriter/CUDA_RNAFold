/* Modifications for eventual CUDA version $Revision: 1.14 $
WBL  5 Aug 2026 Allow arrays to exceed two billion elements (shared Indx()/Hoff())
WBL  8 Jan 2018 Extend linkage for CUDA interface in modular_decomposition.cu
WBL  3 Dec 2017 investigate data dependence in E_mb_loop_fast
  split off multibranch_loops.c r1.10 for time being
*/

// Staggered_Row_Batching Phase 3: compute_flatten_offsets() below uses
// assert() -- int_loop.cu includes this header before its own <assert.h>,
// so this header needs to be self-sufficient rather than relying on
// inclusion order in whichever file includes it.
#include <assert.h>

#ifdef __cplusplus
extern "C" void
#else
PUBLIC void
#endif
choose_gpu(int argc, char **argv); //updates argc and argv

// Page-locked host allocation -- see cuda_host_alloc_ints()/cuda_host_free()
// definitions in modular_decomposition.cu for why. Lets plain .c files
// (fill_arrays.c) get pinned buffers without needing to #include
// <cuda_runtime.h> themselves, same as every other .c<->.cu boundary here.
#ifdef __cplusplus
extern "C" int*
#else
PUBLIC int*
#endif
cuda_host_alloc_ints(const size_t n);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ char*
#else
PUBLIC char*
#endif
cuda_host_alloc_bytes(const size_t n);

#ifdef __cplusplus
extern "C" void
#else
PUBLIC void
#endif
cuda_host_free(void* p);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_fML(const int nfiles,const int length,
         const size_t tri_off_H_total, const size_t row_off_H_total); //tri_off_H[nfiles]/row_off_H[nfiles], compute_batch_offsets()

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_fML(const int nfiles,
	 const int i, const int turn, const int length,
	 const int* energy_min,
	 const size_t* size_off_H); //in, nfiles+1 entries -- Staggered_Row_Batching Phase 4, see compute_flatten_offsets()

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_min_fML(const int nfiles, const int i, const int turn, const int length,
             const size_t total); //side_off_H[nfiles] -- d_side_off_H already uploaded by modular_decomposition_cuda() earlier this row

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
			int* DMLi,
			const size_t* row_off_H,  //in, nfiles+1 entries -- see compute_batch_offsets()
			const size_t* side_off_H); //in, nfiles+1 entries -- Staggered_Row_Batching Phase 4, see compute_flatten_offsets()

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
					     int* DMLi,             //out
					     const size_t* row_off_H,  //in, nfiles+1 entries
					     const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 4
					     const size_t* side_off_H); //in, nfiles+1 entries -- Staggered_Row_Batching Phase 4

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
init_gpu(const int nfiles, const int length,
         const size_t* tri_off_H, const size_t* row_off_H); //in, both nfiles+1 entries -- see compute_batch_offsets()

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size,
          const size_t* tri_off_H, const size_t* row_off_H); //in, both nfiles+1 entries -- see compute_batch_offsets()

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu3(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size,
          const size_t* row_off_H); //in, nfiles+1 entries -- see compute_batch_offsets()

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
// additional sequence at the given length -- summed by gpu_bytes_per_file()
// below, which RNAfold.c budgets a GPU chunk against free VRAM with. Each
// file owns its own formula
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

// Staggered_Row_Batching Phase 6b: replaces compute_max_gpu_batch() -- see
// modular_decomposition.cu for the full explanation. gpu_bytes_per_file()
// sums all three files' per-record cost at the given length (call once per
// candidate record); compute_gpu_usable_bytes() queries free VRAM
// (cudaMemGetInfo) and returns the safety-margined usable budget for a
// fresh chunk (call once per chunk start, not per record).
#ifdef __cplusplus
extern "C" /*PUBLIC*/ size_t
#else
PUBLIC size_t
#endif
gpu_bytes_per_file(const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ size_t
#else
PUBLIC size_t
#endif
compute_gpu_usable_bytes(void);

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
	   int* energy_hp_row, int* energy_mb_row, int* energy_3p00_row, //all out
	   char* gate_row, //out -- bit0 = hc->matrix[ij]!=0, bit1 = ptype[ij] is GU/UG
	   const size_t* size_off_H); //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_my_c(const int nfiles,
	  const int i, const int turn, const int length,
	  const int* min_e, //in
	  const size_t* size_off_H); //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5

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
	   int* energy_min, //out
	   const size_t* size_off_H); //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5


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

// Staggered_Row_Batching Phase 2a: builds row_off_H[]/tri_off_H[] (each
// nfiles+1 entries, exclusive prefix sums over VC[]'s real per-H lengths) --
// row_off_H[H] is where H's row-shaped data (length_H[H]+1 ints) starts
// within a flattened H-block buffer; tri_off_H[H] is where H's
// triangle-shaped data (Hoff(1,length_H[H]) ints) starts. See mfe_cuda.c for
// the exact formulas and the degenerate-under-uniform-length self-check.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
compute_batch_offsets(const int nfiles, const vrna_fold_compound_t **VC,
                       size_t* row_off_H, size_t* tri_off_H); //out, both nfiles+1

// Staggered_Row_Batching Phase 4: host-side builder for
// flatten_index_to_H()'s table (declared further down, __CUDACC__-guarded
// since it's __host__ __device__ and only ever called from .cu files) --
// given any per-H "current active width" array (formula is caller-specific,
// e.g. length_H[H]-i-turn for load_fML, length_H[H]-i-2*turn-2 shared by
// fmli/load_min_fML/modular_decomposition, see fill_arrays_loop.c), builds
// the exclusive-prefix-sum offset table. Declared here (not next to
// flatten_index_to_H()) and defined in mfe_cuda.c (not inline) specifically
// so it's callable from fill_arrays_loop.c -- a plain .c file, not compiled
// by nvcc, which can't see anything inside the __CUDACC__ guard below. Its
// debug-build self-check therefore can't call flatten_index_to_H() either
// (same reason) -- correctness of the binary search itself was already
// verified via a standalone throwaway test when flatten_index_to_H() was
// first written (Phase 3); this just re-checks the prefix-sum arithmetic.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
compute_flatten_offsets(const int nfiles, const size_t* width_H, size_t* flat_off_H); //out, nfiles+1

// Staggered_Row_Batching 2026-08-22: bulk copy of the GPU's my_fML triangle
// into each record's own matrices->fML after the sweep, replacing
// my_fml_update_host's per-row mirroring. fML_H is nfiles pointers, one per
// record; tri_off_H is the nfiles+1 table from compute_batch_offsets().
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
fetch_fML(const int nfiles, int** fML_H, const size_t* tri_off_H);

// Staggered_Row_Batching 2026-08-22: the my_c twin of fetch_fML(), living in
// int_loop.cu because that file owns d_my_c. Replaces new_c_host's per-row
// strided My_c(H,ij) stores.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
fetch_my_c(const int nfiles, int** c_H, const size_t* tri_off_H);

// Stage-attribution timers (mfe_cuda.c). The row-loop "phase" timers cover
// only the sweep; these account for the 20-31% of wall that sits outside it.
extern double stage_build_s, stage_prepare_s, stage_prefill_s, stage_backtrack_s;
extern double stage_output_s, stage_gpuinit_s, stage_teardown_s, stage_free_s;
double rnafold_now_seconds(void);

PRIVATE void
par_fill_arrays(const int nfiles, const vrna_fold_compound_t **VC, int* Energy);

int
mb_loop_fast( vrna_fold_compound_t *vc,
                int i,
                int j);

// Shared by int_loop.cu and modular_decomposition.cu (previously each
// defined its own, or open-coded j*(j-1)/2+i / H*((length+1)*(length+2)/2)
// in 32-bit int -- see "Langdon's 2026 indexing bug" in 2f35ecc). Widened to
// long long here so nfiles*ijsize can exceed 2^31 without every call site
// needing its own size_t/long long cast.
#ifdef __CUDACC__
__host__ __device__
inline long long Indx(const int i, const int j) { //j*(j-1)/2+i
  const long long j_1 = j-1; //force 64 bit calculation
  return j*j_1/2+i;
}

__host__ __device__
inline long long
Hoff(const int H, const int length){ //H*((length+1)*(length+2)/2)
  const long long l1 = length+1;
  const long long l2 = length+2;
  return H*(l1*l2)/2;
}

// Staggered_Row_Batching Phase 3: given a flat thread/work index and a
// per-H offset table (exclusive prefix-sum of each H's "current active
// width" for whatever launch this is -- built host-side by
// compute_flatten_offsets(), declared further up this file since it's
// called from plain .c files too, not just here), returns which H that
// flat index belongs to. Binary search over the (nfiles+1)-entry table: a
// standard CSR-row-pointer lookup, O(log nfiles) per thread. The caller
// computes its position within H via idx - flat_off_H[H]. This is the
// "flatten-and-offset" building block every kernel launch in Phases 4-5
// uses -- while every H still shares one uniform active width (today, and
// through the rest of Phase 2), flat_off_H[] degenerates to H*width
// exactly like row_off_H[]/tri_off_H[] did before mixed lengths existed.
__host__ __device__
inline int flatten_index_to_H(const size_t idx, const size_t* flat_off_H, const int nfiles) {
  int lo = 0, hi = nfiles;
  while(lo+1 < hi) {
    const int mid = lo + (hi-lo)/2;
    if(flat_off_H[mid] <= idx) lo = mid; else hi = mid;
  }
  return lo;
}

// Suggests a block size for `kernel` via CUDA's own occupancy heuristic
// (cudaOccupancyMaxPotentialBlockSize) instead of a hardcoded #define, so
// launch configuration adapts to whatever GPU is actually present rather
// than whichever one the constant was tuned against. Host-only -- the
// occupancy calculator isn't callable from device code. Falls back to
// `fallback` (the caller's previous hardcoded value) if the query itself
// fails, so a startup error here degrades to the old fixed behavior instead
// of an uninitialized/zero block size reaching a kernel launch.
// IMPORTANT CAVEAT, measured 2026-08-21: cudaOccupancyMaxPotentialBlockSize
// maximises occupancy *per SM*, which silently assumes the grid is large
// enough to fill the device. Several kernels here launch grids of 8-10 blocks
// on a 20-SM GPU, and for those the heuristic is actively harmful: a bigger
// block divides the same total thread count into fewer blocks, so most SMs sit
// idle. ncu on modular_decomposition_kernel measured block 640 -> grid 10,
// 9% SM throughput, 24% achieved occupancy -- parallelism-starved, not
// compute- or bandwidth-bound. This is the same trap that made int_loop_kernel
// prefer BLOCK_SIZE=32 over 256 despite far lower occupancy (see int_loop.cu).
// `env_name`, when given, lets a specific kernel's block size be overridden at
// runtime (RNA_*_BLOCK_SIZE), matching this codebase's env-var convention and
// making the choice measurable instead of assumed. Must be a power of two in
// [32,1024]; anything else is reported and ignored.
template <typename KernelT>
__host__ inline int
rnafold_choose_block_size(KernelT kernel, const int fallback,
                           const char* env_name = NULL,
                           const size_t dynamic_smem_bytes = 0,
                           const int block_size_limit = 0) {
  if(env_name) {
    const char* v = getenv(env_name);
    if(v && *v) {
      const int want = atoi(v);
      if(want >= 32 && want <= 1024 && (want & (want-1)) == 0) {
        fprintf(stderr, "%-24s %s=%d overriding the occupancy heuristic\n",
                __FILE__, env_name, want);
        return want;
      }
      fprintf(stderr, "%-24s ignoring %s=%s (want a power of two in [32,1024])\n",
              __FILE__, env_name, v);
    }
  }
  int min_grid_size = 0, block_size = 0;
  const cudaError_t error = cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, dynamic_smem_bytes, block_size_limit);
  if (error != cudaSuccess || block_size <= 0) {
    fprintf(stderr, "rnafold_choose_block_size: cudaOccupancyMaxPotentialBlockSize "
            "returned error %s (code %d) -- falling back to block size %d\n",
            cudaGetErrorString(error), error, fallback);
    return fallback;
  }
  return block_size;
}
#endif /*__CUDACC__*/

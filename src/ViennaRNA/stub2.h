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
					     const size_t* side_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 4
					     const int* i_H); //in, nfiles entries -- continuous flow phase A2, per-record row index

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
	   const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
           const int* i_H); //in, nfiles entries -- continuous flow phase A, per-record row index

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_my_c(const int nfiles,
	  const int i, const int turn, const int length,
	  const int* min_e, //in
	  const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	  const int* i_H); //in, nfiles entries -- continuous flow phase A2, per-record row index

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
	   const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	   const int* i_H); //in, nfiles entries -- continuous flow phase A3, per-record row index


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

// Per-record variants for the scratch-pool backtracking path (mfe_cuda.c).
// tri_lo is tri_off_H[H]; cells is tri_off_H[H+1]-tri_off_H[H].
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
fetch_fML_one(int* dst, const size_t tri_lo, const size_t cells);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
fetch_my_c_one(int* dst, const size_t tri_lo, const size_t cells);

// Stage-attribution timers (mfe_cuda.c). The row-loop "phase" timers cover
// only the sweep; these account for the 20-31% of wall that sits outside it.
// extern "C": these live in mfe_cuda.c (compiled as C) but the .cu files use
// them too, and nvcc compiles those as C++ -- without this the .cu objects ask
// the linker for mangled names that do not exist.
#ifdef __cplusplus
extern "C" {
#endif
extern double stage_build_s, stage_prepare_s, stage_prefill_s, stage_backtrack_s;
extern double stage_output_s, stage_gpuinit_s, stage_teardown_s, stage_free_s;
double rnafold_now_seconds(void);

// Non-zero when hc->matrix and ptype are pure functions of the sequence, i.e.
// no user constraints/SHAPE/ligand-motifs/commands and noLP off. Set once by
// RNAfold.c. When set, init_gpu2/init_gpu3 skip their O(n^2) host packing
// loops and let the GPU derive the five bitmasks from the sequence instead --
// that packing measured 197.4 s of a 769 s Colab run, 25.7% of wall.
extern int g_hc_seq_derived;

// GPU-resident sweep, step 5a: non-zero when RNA_GPU_SWEEP selects the
// device-resident path -- the three per-row host loops are skipped and the six
// per-row transfers that fed them are not issued. Constant for the run (the
// CUDA graph's captured topology differs between modes). DEFAULTS ON since
// 2026-08-30; RNA_GPU_SWEEP=0 restores the host sweep.
//
// STEP 5b, and why dropping the per-row syncs under this flag is safe. Each
// per-row cudaDeviceSynchronize() existed to make the D2H that followed it
// safe. In device mode there is no D2H, and device-side ordering does not
// depend on those syncs at all:
//   - every per-row kernel launches on the legacy NULL stream;
//   - graph_stream is created with a plain cudaStreamCreate(), i.e. a BLOCKING
//     stream, so NULL-stream work and graph_stream work are implicitly ordered
//     in both directions;
//   - the build passes no --default-stream per-thread, so those legacy
//     semantics genuinely apply (checked, not assumed).
// So load_my_c_kernel still sees new_c_kernel's d_new_e, the graph trio still
// sees fml_scan_kernel's d_energy_min, and md_snapshot_dml() still runs after
// the trio -- by stream order rather than by host round trip.
//
// Errors are still caught: cudaPeekAtLastError() after each launch catches
// launch failures immediately, and the row's remaining blocking copies force
// execution failures (including the .cu files' live device asserts) to surface
// within the same row -- later than before, but not silently.
PUBLIC int rnafold_gpu_sweep(void);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
int_loop_hccc_buffers(unsigned int** d_out, const size_t** off_out);

// GPU-resident sweep, step 1. Accessors handing a .cu file's row-shaped device
// buffers to the kernels being built in hp_mb_loop.cu, plus the per-row DMLi
// snapshot that stands in for the host's DMLi1 rotation. Each buffer stays
// owned -- allocated, INF-prefilled, VRAM-budgeted and freed -- by the file
// that declares it; only a pointer crosses. int_loop_hccc_buffers() above is
// the precedent.
//
// NOTHING READS THESE YET. Step 1 is behaviour-neutral by construction so that
// byte-identical output can be established across the verification matrix
// before any host loop moves. Any out-parameter may be NULL.
//
// Validity: int_loop_row_buffers between init_gpu2()/teardown_gpu2();
// md_row_buffers and md_snapshot_dml between init_gpu()/teardown_gpu().
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
int_loop_row_buffers(int** energy_min2_out, int** new_e_out);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
md_row_buffers(int** dml_out, int** dml1_out, int** fml_prev_out,
               int** energy_min_out);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
md_snapshot_dml(void);
#ifdef __cplusplus
}
#endif

// GPU-resident sweep, step 2: the device twin of fill_arrays_loop.c's
// fml_prev_host. Launches fml_prev_kernel over this row, and when
// RNA_ROW_VERIFY is set compares the result cell-for-cell against the host
// array still being computed beside it, reporting (H, i, j) and both values on
// each of the first 20 mismatches plus a per-chunk total.
//
// The host loop still runs and its output is still what the sweep consumes --
// the device result is written to d_fml_prev and read by nothing yet. Deleting
// the host loop waits until fml_scan_kernel needs d_fml_prev, and until
// RNA_ROW_VERIFY has been clean across the whole matrix.
//
// fml_prev_host is only dereferenced under RNA_ROW_VERIFY, so a caller that has
// no host array may pass NULL provided the flag is unset.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
fml_prev_i(const int nfiles, const int i, const int turn,
           const int* fml_prev_host,
           const size_t* row_off_H, const size_t* size_off_H,
           const int* i_H); //in, nfiles entries -- continuous flow phase A, per-record row index

// GPU-resident sweep, step 3: the device twin of new_c_host, the largest of the
// three per-row host loops. Elementwise, no recurrence over j.
//
// MUST be called after new_c_host and before load_my_c(): load_my_c uploads the
// host's new_C over d_new_e, so the verify readback has to happen first -- and
// that upload landing afterwards is exactly what keeps this step
// behaviour-neutral while both paths run.
//
// new_C_host is dereferenced only under RNA_ROW_VERIFY.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
new_c_i(const int nfiles, const int i, const int turn, const int noGUclosure,
        const int* new_C_host,
        const size_t* row_off_H, const size_t* size_off_H,
           const int* i_H); //in, nfiles entries -- continuous flow phase A, per-record row index

// GPU-resident sweep, step 4: the device twin of fml_host -- the one loop of
// the three that is a recurrence along j rather than elementwise. Implemented
// as an inclusive scan over affine min-plus maps, one block per record.
//
// MUST be called after fml_host and before the load_fML/modular_decomposition/
// load_min_fML trio: that trio uploads the host's energy_min over d_energy_min,
// so the verify readback has to precede it, and that upload landing afterwards
// is what keeps the step behaviour-neutral while both paths run.
//
// energy_min_host is dereferenced only under RNA_ROW_VERIFY.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
fml_scan_i(const int nfiles, const int i, const int turn,
           const int* energy_min_host,
           const size_t* row_off_H, const size_t* size_off_H,
           const int* i_H); //in, nfiles entries -- continuous flow phase A, per-record row index

// gpuinit attribution (mfe_cuda.c). See there for why.
#ifdef __cplusplus
extern "C" {
#endif
extern double stage_ig1_s, stage_ig2_s, stage_ig3_s;
extern double stage_ig_pack_s, stage_ig_malloc_s;
#ifdef __cplusplus
}
#endif

// Wraps the cudaMalloc calls inside init_gpu/2/3 so their cost is separable
// from the host packing loops. Same gpuErrchk behaviour as the calls it
// replaces -- only the timing is added.
#ifdef __CUDACC__
#define TIMED_CUDAMALLOC(pp, sz) do {                                   \
    const double _tm = rnafold_now_seconds();                           \
    gpuErrchk( cudaMalloc((void **)(pp), (sz)) );                       \
    stage_ig_malloc_s += rnafold_now_seconds() - _tm;                   \
  } while(0)
#endif

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
// Invert a flat triangular index into (i,j), the inverse of ViennaRNA's
// jindx[j]+i == j*(j-1)/2+i. Column j occupies flat indices
// j*(j-1)/2+1 .. j*(j+1)/2, so f determines j uniquely. The sqrt is a seed
// only -- double has 53 bits of mantissa against the ~1.3e8 argument here, but
// the two correction loops make the result exact regardless, which matters
// because a single off-by-one would silently shift every predicate.
__device__
inline void rnafold_tri_unflatten(const long long f, int* pi, int* pj) {
  int j = (int)((1.0 + sqrt(1.0 + 8.0*(double)(f-1))) * 0.5);
  while((long long)j*(j-1)/2 >= f) j--;
  while((long long)(j+1)*j/2 <  f) j++;
  *pj = j;
  *pi = (int)(f - (long long)j*(j-1)/2);
}

// Device replica of hc_reset_to_default()'s VRNA_FC_TYPE_SINGLE case
// (constraints_hard.c:747). Returns the hc->matrix byte for flat index f of a
// record of length len_H, or 0 for the slack slots the host loop leaves at
// their calloc value (f==0, and anything past len*(len+1)/2).
//
// Deliberately mirrors the host control flow line for line rather than being
// "simplified": every branch here corresponds to one there, so the two can be
// compared by eye as well as by memcmp.
__device__
inline unsigned char rnafold_hc_opt(const long long f, const int len_H,
                                    const int turn, const int max_bp_span,
                                    const int noGU, const int noGUclosure,
                                    const short* __restrict__ S_H,
                                    const char*  __restrict__ pair) {
  const long long cells = (long long)len_H*(len_H+1)/2;
  if(f < 1 || f > cells) return 0u;                 //slack: calloc'd zero on the host
  int i, j; rnafold_tri_unflatten(f, &i, &j);

  if(i == j)                                        //loop 1: unpaired, all contexts
    return (unsigned char)(  VRNA_CONSTRAINT_CONTEXT_EXT_LOOP
                           | VRNA_CONSTRAINT_CONTEXT_HP_LOOP
                           | VRNA_CONSTRAINT_CONTEXT_INT_LOOP
                           | VRNA_CONSTRAINT_CONTEXT_MB_LOOP);

  //loop 2 writes only i in [1, j-turn-1] of columns j > turn+1
  if(!(j > turn + 1 && i < j - turn)) return 0u;

  int max_span = max_bp_span;
  if((max_span < 5) || (max_span > len_H)) max_span = len_H;
  if((j - i + 1) > max_span) return 0u;

  const int t = (int)pair[S_H[i]*8 + S_H[j]];
  if(t == 0) return 0u;
  if(t == 3 || t == 4) {
    if(noGU) return 0u;
    if(noGUclosure)
      return (unsigned char)(VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS
                             & ~(VRNA_CONSTRAINT_CONTEXT_HP_LOOP
                                 | VRNA_CONSTRAINT_CONTEXT_MB_LOOP));
  }
  return (unsigned char)VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS;
}

// Device replica of vrna_ptypes() (alphabet.c:143) with noLP off. Its
// anti-diagonal walk covers exactly {(i,j) : j-i >= turn+1} -- verified by
// simulating the loop for n = 12/20/41/60, zero cells missing or extra -- so
// the walk order is irrelevant and each cell is just md->pair[S[i]][S[j]].
__device__
inline int rnafold_ptype(const long long f, const int len_H, const int turn,
                         const short* __restrict__ S_H,
                         const char*  __restrict__ pair) {
  const long long cells = (long long)len_H*(len_H+1)/2;
  if(f < 1 || f > cells) return 0;
  int i, j; rnafold_tri_unflatten(f, &i, &j);
  if(j - i < turn + 1) return 0;
  return (int)pair[S_H[i]*8 + S_H[j]];
}

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

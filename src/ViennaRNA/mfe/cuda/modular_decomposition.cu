//WBL 11 Dec 2017 $Revision: 1.89 $ CUDA GGGP ViennaRNA-2.3.0 rf/rf/
//Helper for fill_arrays.c 
//based on fill_arrays_loop.c r1.10

//Plan is to have all CUDA code here.
//Atleast whilst only convert modular decomposition to CUDA

//LAW 14 Jul 2026 added WBL diff till 2026
//WBL  5 Jul 2026 see if can get to compile with CUDA 13.*
//WBL 15 May 2023 add use_cuda
//WBL  4 Mar 2018 add optional pause to fake persistence mode
//WBL 17 Feb 2018 simplify GPU startup message and send to stderr
//WBL  2 Jan 2018 
//WBL 31 Dec 2017 try without atomicMin by use of one block per column
//WBL 30 Dec 2017 for debug make modular_decomposition_ij as in modular_decomposition.c r1.6
//WBL 29 Dec 2017 ../rf_cuda try tuning


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <time.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/strings.h"
#include "ViennaRNA/utils/structures.h"
#include "ViennaRNA/params/default.h"
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/constraints/basic.h"
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/constraints/soft.h"
#include "ViennaRNA/eval/gquad.h"
#include "ViennaRNA/mfe/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/loops/all.h"
#include "ViennaRNA/mfe/global.h"

//https://devtalk.nvidia.com/default/topic/1012969/cuda-programming-and-performance/texture-unit-in-pascal-architecture/2
#ifdef MIN2
#undef MIN2
#endif
#define MIN2(x,y) min(x,y)

//#include <assert.h>
//#ifdef STUB
#include "stub2.h"
//#include "stub.h"
//#endif /*STUB*/

// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h> //Commented out in 2026, nothing here is used
//#include <helper_cuda.h> //Commented out in 2026 to get started

//BLOCK_SIZE must be power of two 32 or greater
#define BLOCK_SIZE 64

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk2(ans,first) { gpuAssert((ans), __FILE__, __LINE__,first); }
inline void gpuAssert(cudaError_t code, const char *file, const int line, const bool first=false, const bool abort=true)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA error: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
     if(first)
     fprintf(stderr,"CUDA error: on first kernel.\nWas %s compiled with correct compute levels?\n", file);
     if (abort) exit(code);
   }
}

/*taken from helper_string.h
//https://people.maths.ox.ac.uk/gilesm/cuda/prac1/helper_string.h getCmdLineArgumentInt
inline int getCmdLineArgumentInt(int argc, char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
		    //printf("getCmdLineArgumentInt(%d,%s) removing argc %d %s ",
		    //	   argc,argv[0],i,argv[i]);
		    //fflush(stdout);

                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
		    //WBL 15 Dec 2017 remove argv[i]
		    //for(int k=i; k < argc-1; k++) argv[k] = argv[k+1];
		    /* fails for unknown reason
		      for(int k=i; k < argc-1; k++){
		      printf("getCmdLineArgumentInt(%d,%s) argv[%d]%s becomes %s\n",
			     argc,argv[0],k,argv[k],argv[k+1]);
		    }
		    printf("getCmdLineArgumentInt(%d,%s) argv[%d] set null\n",
			     argc,argv[0],argc-1);
		    argv[argc-1] = NULL;
		    argc--;
		    **
		    argv[i][0] = '\0'; //ensure -device= is not visible to rest of RNAfold
		    //printf("`%s' value %d\n", &string_argv[length + auto_inc],value);
                }
                else
                {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound)
    {
        return value;
    }
    else
    {
        return 0;
    }
}
*/   //End comment block from WBL
int use_cuda = 0;
//C interface to CUDA code
extern "C" void
choose_gpu(int argc, char **argv) {
//based on CUDA 9.0 0Samples/natrixMul.cu
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    //Eg --device=1 (for second GPU)
    //Eg --persistence=1 (to pause)
    int devID = 0;
    /*int persistence = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device");
        cudaSetDevice(devID);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "persistence"))
    {
        persistence = getCmdLineArgumentInt(argc, argv, "persistence");
    }
	*/
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);
	/*
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }
	*/
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
      fprintf(stderr,"RNAfold GPU Device %d: \"%s\" with compute capability %d.%d\n",
	      devID, deviceProp.name, deviceProp.major, deviceProp.minor); //BLOCK_SIZE);
    }
	//ok zero copy will be really inefficient but makes a start
	//https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
    //https://devtalk.nvidia.com/default/topic/895513/cudamalloc-slow/?offset=1
    //njuffa suggests calling cudafree can get GPU warmed up early
	if (!deviceProp.canMapHostMemory){
		fprintf(stderr,"RNAfold GPU Device %d: \"%s\" does not support zero copy\n", devID, deviceProp.name);
		use_cuda = 0;
		return; //for the time being RNAFold will use pthreads instead
	}
	error = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (error != cudaSuccess){
		printf("failed to set cudaDeviceMapHost for zero copy error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		use_cuda = 0;
		return; //for the time being, RNAFold will use pthreads instead
	}

    error = cudaFree(0);
    if (error != cudaSuccess)
    {
        printf("cudaFree(0) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    /*https://devtalk.nvidia.com/default/topic/1030608/cuda-programming-and-performance/why-2-9-seconds-to-start-tesla-k20/
    if(persistence){
      fprintf(stderr,"pausing\n");
      pause();
      fprintf(stderr,"pause returned, exiting with errno %d, %s line %d\n", 
                     errno, __FILE__,__LINE__);
      exit(errno);
    }*/
	use_cuda = 1;
}

void int_Memcpy(int* out, const int* in, const size_t size, const cudaMemcpyKind dir, const int error_report) { // 32-bit signed integer overflow bug fix
  const cudaError_t error = cudaMemcpy(out, in, size*sizeof(int), dir);
  if (error != cudaSuccess)  {
    printf("cudaMemcpy(%p,%p,%zu,%d) returned error %s (code %d), %s line(%d)\n", // 32-bit signed integer overflow bug fix
	   out,in,
	   size*sizeof(int),dir,cudaGetErrorString(error), error, __FILE__, error_report);
    exit(EXIT_FAILURE);
  }
}

// Async twin of int_Memcpy for use inside captured CUDA graph regions.
// NB: a cudaSuccess return here only means the copy was recorded (or, outside
// capture, enqueued) successfully -- NOT that it will succeed when the graph
// is later replayed. Real data/runtime errors from replayed graph work only
// surface at the cudaStreamSynchronize() after cudaGraphLaunch().
void int_MemcpyAsync(int* out, const int* in, const size_t size, const cudaMemcpyKind dir, cudaStream_t stream, const int error_report) {
  const cudaError_t error = cudaMemcpyAsync(out, in, size*sizeof(int), dir, stream);
  if (error != cudaSuccess)  {
    printf("cudaMemcpyAsync(%p,%p,%zu,%d) returned error %s (code %d), %s line(%d)\n",
	   out,in,
	   size*sizeof(int),dir,cudaGetErrorString(error), error, __FILE__, error_report);
    exit(EXIT_FAILURE);
  }
}

// New Jul 2026: page-locked host allocation for buffers that are the
// direct source/destination of a cudaMemcpy every row (DMLi/DMLi1/DMLi2
// here, and fill_arrays.c's energy_min/energy_hp_row/energy_mb_row/
// energy_3p00_row/new_C) -- pageable host memory forces the CUDA driver to
// stage every one of those transfers through an internal pinned buffer
// first; allocating these ourselves as pinned skips that copy. Hard-fails
// on allocation failure via gpuErrchk, matching every other CUDA
// allocation in this codebase -- see compute_gpu_usable_bytes()'s
// safety_margin, tightened alongside this change specifically to keep
// pinned-memory pressure down at the largest batch sizes rather than
// adding a fallback-to-pageable path here.
PUBLIC int*
cuda_host_alloc_ints(const size_t n) {
  int* p;
  gpuErrchk( cudaHostAlloc((void **) &p, n*sizeof(int), cudaHostAllocDefault) );
  return p;
}

// Byte-sized twin of the above, for row buffers holding flags rather than
// energies -- see gate_row in hp_mb_loop.cu. Pinned for the same reason the
// int version is: it is copied back once per sweep row.
PUBLIC char*
cuda_host_alloc_bytes(const size_t n) {
  char* p;
  gpuErrchk( cudaHostAlloc((void **) &p, n*sizeof(char), cudaHostAllocDefault) );
  return p;
}

PUBLIC void
cuda_host_free(void* p) {
  gpuErrchk( cudaFreeHost(p) );
}

int first = 1;

// Forward declarations so init_gpu() (below) can probe fmli_kernel's/
PUBLIC void pack_fml(const int nfiles, const int i, const int turn,
                     const int length, const size_t* size_off_H);

// modular_decomposition_kernel's occupancy before they're defined further
// down this file, and so that probe runs once, up front, rather than lazily
// at first launch like this file's other kernels get further down. That's
// deliberate, not just style: both are launched from inside a CUDA-graph
// capture region (load_fML_modular_decomposition_load_min_fML() below).
// cudaOccupancyMax*() take no stream argument and don't enqueue anything, so
// they should be legal regardless of another stream's capture state -- but
// init_gpu() already runs once per batch, well before any capture begins,
// so there's no reason to lean on that instead.
__global__ void fmli_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
        int* __restrict__ fml_i,
  const int* __restrict__ fml_j,
  const int* __restrict__ fml_row,   // int16 path, NULL when off
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H);   // continuous flow phase A2
// TILE = how many threads cooperate on one output cell's y-reduction; see the
// kernel's own comment further down for why that split exists at all. TILE=1
// is exactly the pre-2026-08-22 one-thread-per-cell kernel, kept reachable
// via RNA_MD_TILE=1 as an A/B baseline.
template <int TILE>
__global__ void modular_decomposition_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
  const int* __restrict__ fml_i, const int* __restrict__ fml_j,
  const short* __restrict__ fml_j16, const int* __restrict__ fml_b,
  const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
  int* __restrict__ dml,
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H);   // continuous flow phase A2

// Block sizes for the above, chosen once in init_gpu() below instead of the
// BLOCK_SIZE=64 constant these used to hardcode (tuned against one GPU, the
// L4). Neither kernel has shared memory or a reduction tying it to a
// specific size.
static int g_block_size_fmli = 0;
static int g_block_size_md   = 0;
// How many threads cooperate on one modular_decomposition_kernel output cell
// (its TILE template parameter). 32 = one warp per cell, the point of the
// 22 Aug 2026 rewrite; RNA_MD_TILE overrides it for A/B measurement, with
// RNA_MD_TILE=1 reproducing the previous one-thread-per-cell kernel exactly.
// Chosen in init_gpu() rather than at launch because the occupancy probe
// needs the specific template instantiation.
static int g_md_tile = 32;

// Same shape as rnafold_choose_block_size()'s env handling in stub2.h, but
// for the tile width -- kept here rather than there because TILE is specific
// to this one kernel.
static int
rnafold_choose_md_tile(void) {
  const char* v = getenv("RNA_MD_TILE");
  if(v && *v) {
    const int want = atoi(v);
    if(want >= 1 && want <= 32 && (want & (want-1)) == 0) return want;
    fprintf(stderr, "%-24s ignoring RNA_MD_TILE=%s (want a power of two in [1,32])\n",
            __FILE__, v);
  }
  return 32;
}

//int* d_indx; //indx no longer used
int* d_energy_min;
int* d_fml_i;  //my_fML
int* d_fml_j;  //my_fML
// int16 fml_j (RNA_FML_INT16). Allocated INSTEAD of d_fml_j when the gate is on,
// so the VRAM saving is real and measurable rather than shadowed by keeping both.
//   d_fml_j16   the offsets themselves, same triangular layout as d_fml_j
//   d_fml_b     one int32 baseline per FML_BLK consecutive entries WITHIN a
//               column -- blocks must not straddle a column boundary, because
//               the end of column j and the start of column j+1 are the two
//               extremes of the whole energy range and a straddling block would
//               have to span it
//   d_fml_row   the CURRENT row's cells in full int32. Required, not an
//               optimisation: fmli_kernel reads row i between the two writers,
//               at its pre-MIN2 value, so a cell is not final until its row
//               completes and cannot be packed before then.
short*  d_fml_j16;
int*    d_fml_b;
int*    d_fml_row;
// colb_off[j] = number of baseline slots used by columns 1..j-1, so a column's
// blocks start on a baseline boundary. A pure function of j and FML_BLK, hence
// ONE table for the whole chunk rather than one per record.
size_t* d_colb_off;
size_t* d_base_off_H;   // per-record start in d_fml_b
static size_t g_base_total = 0;
int* d_dml;  //DMLi
// GPU-resident sweep, step 1. Two row-shaped buffers the device side of the
// sweep needs once new_c_host/fml_host/fml_prev_host become kernels. They are
// allocated, INF-prefilled and maintained from here on but READ BY NOTHING
// YET: this step is deliberately behaviour-neutral, so the whole verification
// matrix can confirm byte-identical output before any host loop moves.
//
//   d_dml1      the device twin of the host's DMLi1 -- row i+1's DMLi, which
//               new_c_kernel will read as e_mb's DMLi1[j-1].
//   d_fml_prev  the device twin of the host's fml_prev -- row i's final fML,
//               which fml_scan_kernel reads one row later as fML(i+1,j).
//
// The host keeps DMLi1 by ROTATING three buffers. This does not: it takes one
// device-to-device copy per row instead (md_snapshot_dml(), below). That is a
// deliberate departure. Rotating would change the pointer baked into the
// captured CUDA-graph node every single row, and the graph currently gets away
// with 16785 cheap cudaGraphExecUpdate()s against only 3 forced
// reinstantiates; a 3.3 MB D2D copy costs ~0.18 s across a whole run and
// leaves the graph's parameters untouched. Correctness before cleverness on
// the piece that is hardest to debug.
//
// There is deliberately no d_dml2: the host allocates, prefills and rotates
// DMLi2, but nothing reads it. Confirming and removing that is a separate
// cleanup and is not bundled here.
int* d_dml1;
int* d_fml_prev;
// Staggered_Row_Batching Phase 2d: own device copies (per this codebase's
// established convention of each .cu file owning independent device state)
// of tri_off_H[]/row_off_H[] -- d_fml_j's per-H triangle-block start and
// d_energy_min/d_fml_i/d_dml's per-H row-block start, replacing the
// H-tightest H+X*nfiles convention throughout this file.
static size_t* d_tri_off_H;
static size_t* d_row_off_H;
// Staggered_Row_Batching Phase 6d: row_off_H[nfiles], the real total extent of
// the row-shaped buffers here (d_energy_min, d_dml). Cached at init_gpu() time
// -- equals the old uniform nfiles*(length+1) only while chunks are
// uniform-length, and that formula over-runs the allocation once they aren't.
static size_t  g_row_total = 0;
// Staggered_Row_Batching Phase 4: per-row "current active width" tables for
// load_fML/fmli_kernel/modular_decomposition_kernel/load_min_fML_kernel's
// flat grids -- rebuilt+reuploaded every sweep row i (see fill_arrays_loop.c),
// unlike tri_off_H/row_off_H above which are fixed for the whole chunk. Only
// two distinct shapes needed: size_off_H is load_fML's own
// (length_H[H]-i-turn); side_off_H is shared by fmli_kernel/
// modular_decomposition_kernel/load_min_fML_kernel (length_H[H]-i-2*turn-2 --
// verified algebraically identical across all three from their pre-Phase-4
// bound-check arithmetic). Both buffers are allocated once per chunk here
// (init_gpu()) and just overwritten each row, same as d_energy_min already is.
static size_t* d_size_off_H;
static size_t* d_side_off_H;
// Continuous flow phase A2: this file's own copy of the per-record row index,
// read by all four kernels below. int_loop.cu and hp_mb_loop.cu each carry an
// identical one -- per-translation-unit device tables are this codebase's
// existing convention (d_size_off_H above is duplicated the same way).
//
// Uploaded OUTSIDE the CUDA-graph capture region (see
// load_fML_modular_decomposition_load_min_fML() below), with a blocking
// cudaMemcpy on the NULL stream, which is what lets fill_arrays_loop.c pass a
// plain stack table: this file's standing hazard -- an async H2D captured into
// the graph must have a persistent host source -- does not apply to a copy that
// has completed before capture even begins.
static int*    d_i_H;
static int*    i_H_shadow   = NULL;
static int     i_H_shadow_n = 0;
static void    i_H_shadow_reset(void);   // defined below; called from init/teardown above it
//int* h_dml;  //DMLi
//unsigned int mem_size_buf; //bytes in h_dml and d_dml
//int* fml_j;  //my_fML

// CUDA Graph plumbing for the load_fML -> modular_decomposition -> load_min_fML
// chain (fill_arrays_loop.c has no host CPU logic between those three calls,
// which is what makes them capturable as a single graph). graph_stream must
// stay a *blocking* stream (not cudaStreamNonBlocking): that's what makes the
// legacy-default-stream ordering rule implicitly wait for int_loop_i()'s /
// load_my_c()'s still-fully-synchronous NULL-stream work before the captured
// chain runs each iteration, with no extra synchronization code needed.
cudaStream_t    graph_stream     = 0;
cudaGraphExec_t graph_exec       = NULL;
int             graph_exec_valid = 0;

// Diagnostic-only counters/timer for the graph exec update-vs-reinstantiate
// path, to measure (rather than guess) how often cudaGraphExecUpdate() is
// actually cheap-patching vs. falling back to a full destroy+instantiate,
// and how much cumulative CPU time that bookkeeping (capture+update/
// instantiate+destroy, NOT the launch+sync of the real GPU work) costs
// across a whole fold. Printed once at process exit via atexit(). Remove
// once the CUDA-graph-vs-old-branch timing question is settled.
long   graph_update_success_count      = 0;
long   graph_first_instantiate_count   = 0;
long   graph_forced_reinstantiate_count = 0;
double graph_mgmt_seconds              = 0.0;

static void
print_graph_update_stats(void) {
  fprintf(stderr,
    "%-24s CUDA graph stats: %ld update() succeeded, %ld first-time instantiate, "
    "%ld forced reinstantiate (update failed), %.3f s cumulative capture/update/"
    "instantiate/destroy overhead (excludes launch+sync)\n",
    __FILE__, graph_update_success_count, graph_first_instantiate_count,
    graph_forced_reinstantiate_count, graph_mgmt_seconds);
}

static inline double
graph_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

PUBLIC void
init_gpu(const int nfiles, const int length,
         const size_t* tri_off_H, const size_t* row_off_H) {
  if(!first) return;
  const double _t_ig1 = rnafold_now_seconds();
  fprintf(stderr,"%-24s init_gpu(%d, %d)\n",__FILE__,nfiles,length);
  cudaError_t error;
  // graph_stream is nfiles/length-independent -- guarded on its own initial
  // value (0), not on `first`, so teardown_gpu() can reset first=1 between
  // GPU batches without this recreating (and leaking the handle to) a fresh
  // stream every batch. Created exactly once for the whole process.
  if(graph_stream == 0) gpuErrchk( cudaStreamCreate(&graph_stream) );

  TIMED_CUDAMALLOC(&d_tri_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_tri_off_H, tri_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
  TIMED_CUDAMALLOC(&d_row_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_row_off_H, row_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  // Staggered_Row_Batching Phase 4: allocated here (fixed size for the whole
  // chunk), but not populated here -- unlike tri_off_H/row_off_H these change
  // every sweep row i, so the real upload happens per-row in load_fML()/
  // modular_decomposition_cuda() instead.
  TIMED_CUDAMALLOC(&d_size_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  TIMED_CUDAMALLOC(&d_side_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  TIMED_CUDAMALLOC(&d_i_H, (size_t)nfiles*sizeof(int));
  i_H_shadow_reset();          // fresh buffer: the shadow must not claim it is current

  // Staggered_Row_Batching Phase 2d: allocation sizes now the real per-H sum
  // (row_off_H[nfiles]/tri_off_H[nfiles]) instead of a uniform nfiles*(...)
  // multiply -- see compute_batch_offsets(), mfe_cuda.c.
  const size_t mem_size_len = row_off_H[nfiles] * sizeof(int);
  const size_t ijsize_len   = tri_off_H[nfiles] * sizeof(int);
  // Staggered_Row_Batching Phase 6d: cached for load_fML()/
  // modular_decomposition_cuda(), which transfer whole row-shaped buffers each
  // row but never receive the offset table.
  g_row_total = row_off_H[nfiles];

  error = cudaMalloc((void **) &d_energy_min, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_energy_min %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  error = cudaMalloc((void **) &d_fml_i, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_fml_i %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  if(!rnafold_fml_int16()) {
    error = cudaMalloc((void **) &d_fml_j, ijsize_len);
    if (error != cudaSuccess)  {
        printf("cudaMalloc d_fml_j %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
  	     ijsize_len, cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
  } else {
    // Slot flow hands a slot to a new occupant MID-SWEEP, and reset_slot_md()
    // cannot put that slot's baselines back to UNSET from the tri_lo/tri_n it
    // is given. Encoding the new occupant against the old one's baselines is a
    // silent wrong answer, so the pairing is refused rather than approximated.
    if(rnafold_slot_flow() >= 1) {
      fprintf(stderr,"%-24s RNA_FML_INT16 and RNA_SLOT_FLOW cannot be combined yet: "
                     "a slot handover would leave stale baselines. Unset one.\n",
              __FILE__);
      exit(EXIT_FAILURE);
    }

    // colb_off is built to the LONGEST record; every shorter one indexes a
    // prefix of it, exactly as the triangular Indx() already does.
    int maxlen = 0;
    for(int H=0;H<nfiles;H++) {
      const size_t cells = tri_off_H[H+1]-tri_off_H[H];
      int n = 0; while(((size_t)(n+1)*(n+2))/2 < cells) n++;
      if(n > maxlen) maxlen = n;
    }
    size_t* colb = (size_t*)malloc((size_t)(maxlen+2)*sizeof(size_t));
    colb[0] = colb[1] = 0;
    for(int j=1;j<=maxlen;j++)
      colb[j+1] = colb[j] + (size_t)((j + FML_BLK - 1)/FML_BLK);
    TIMED_CUDAMALLOC(&d_colb_off, (size_t)(maxlen+2)*sizeof(size_t));
    gpuErrchk( cudaMemcpy(d_colb_off, colb, (size_t)(maxlen+2)*sizeof(size_t), cudaMemcpyHostToDevice) );

    size_t* boff = (size_t*)malloc((size_t)(nfiles+1)*sizeof(size_t));
    boff[0] = 0;
    for(int H=0;H<nfiles;H++) {
      const size_t cells = tri_off_H[H+1]-tri_off_H[H];
      int n = 0; while(((size_t)(n+1)*(n+2))/2 < cells) n++;
      boff[H+1] = boff[H] + colb[n+1];
    }
    g_base_total = boff[nfiles];
    TIMED_CUDAMALLOC(&d_base_off_H, (size_t)(nfiles+1)*sizeof(size_t));
    gpuErrchk( cudaMemcpy(d_base_off_H, boff, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
    free(colb); free(boff);

    TIMED_CUDAMALLOC(&d_fml_j16, tri_off_H[nfiles]*sizeof(short));
    TIMED_CUDAMALLOC(&d_fml_b,   g_base_total*sizeof(int));
    TIMED_CUDAMALLOC(&d_fml_row, mem_size_len);
  }

  error = cudaMalloc((void **) &d_dml, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_dml %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  // GPU-resident sweep, step 1 -- see the declarations above. Same row shape
  // and lifetime as d_dml, so they are allocated, prefilled, sized into the
  // VRAM budget and freed alongside it.
  error = cudaMalloc((void **) &d_dml1, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_dml1 %zu returned error %s (code %d), line(%d)\n",
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  error = cudaMalloc((void **) &d_fml_prev, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_fml_prev %zu returned error %s (code %d), line(%d)\n",
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  // See the forward declarations above for why this happens here rather
  // than lazily at first launch.
  g_block_size_fmli = rnafold_choose_block_size(fmli_kernel, BLOCK_SIZE, "RNA_FMLI_BLOCK_SIZE");
  g_md_tile = rnafold_choose_md_tile();
  // Each instantiation is a distinct kernel with its own register/occupancy
  // profile, so the probe has to name the one that will actually launch.
#define MD_BS(T) rnafold_choose_block_size(modular_decomposition_kernel<T>, BLOCK_SIZE, "RNA_MD_BLOCK_SIZE")
  switch(g_md_tile) {
    case  1: g_block_size_md = MD_BS(1);  break;
    case  2: g_block_size_md = MD_BS(2);  break;
    case  4: g_block_size_md = MD_BS(4);  break;
    case  8: g_block_size_md = MD_BS(8);  break;
    case 16: g_block_size_md = MD_BS(16); break;
    default: g_block_size_md = MD_BS(32); break;
  }
#undef MD_BS
  // The kernel's tile-to-shuffle-group alignment assumes whole warps per
  // block. rnafold_choose_block_size() only ever yields a power of two >= 32
  // (occupancy heuristic or env override, both validated), so this holds
  // today -- asserted so a future change there fails loudly here instead of
  // silently corrupting the reduction for TILE < 32.
  assert(g_block_size_md % 32 == 0);
  fprintf(stderr,"%-24s fmli_kernel block size %d, modular_decomposition_kernel block size %d (both were hardcoded %d), md tile %d\n",
	  __FILE__, g_block_size_fmli, g_block_size_md, BLOCK_SIZE, g_md_tile);

  stage_ig1_s += rnafold_now_seconds() - _t_ig1;
  first = 0;
  return;
}

// Frees the 4 nfiles/length-scaled device buffers allocated by init_gpu()
// and resets `first` so the next init_gpu() call actually re-runs (rather
// than no-op'ing) at a new batch's nfiles. graph_stream is deliberately left
// alone -- cheap to keep, no reason to tear it down between batches (see the
// comment on its creation above). Defensively destroys graph_exec if valid,
// since its captured nodes reference the buffers freed right below --
// belt-and-suspenders per the multi-batch design doc; the existing
// update-or-reinstantiate fallback in
// load_fML_modular_decomposition_load_min_fML() would likely self-heal even
// without this, but there's no reason to leave a stale exec referencing
// about-to-be-freed memory during the "GPU unloaded" window between batches.
PUBLIC void
teardown_gpu(void) {
  if(first) return; // never initialized (or already torn down) -- nothing to free
  gpuErrchk( cudaFree(d_energy_min) );
  gpuErrchk( cudaFree(d_fml_i) );
  if(!rnafold_fml_int16()) {
    gpuErrchk( cudaFree(d_fml_j) );
  } else {
    gpuErrchk( cudaFree(d_fml_j16) );
    gpuErrchk( cudaFree(d_fml_b) );
    gpuErrchk( cudaFree(d_fml_row) );
    gpuErrchk( cudaFree(d_colb_off) );
    gpuErrchk( cudaFree(d_base_off_H) );
  }
  gpuErrchk( cudaFree(d_dml) );
  gpuErrchk( cudaFree(d_dml1) );
  gpuErrchk( cudaFree(d_fml_prev) );
  gpuErrchk( cudaFree(d_tri_off_H) );
  gpuErrchk( cudaFree(d_row_off_H) );
  gpuErrchk( cudaFree(d_size_off_H) );
  gpuErrchk( cudaFree(d_i_H) );
  i_H_shadow_reset();          // the device buffer is gone; the shadow must not outlive it
  gpuErrchk( cudaFree(d_side_off_H) );
  if(graph_exec_valid) {
    gpuErrchk( cudaGraphExecDestroy(graph_exec) );
    graph_exec_valid = 0;
  }
  first = 1;
}

// Bytes of device memory this file needs for one additional sequence at the
// given length -- the 3 mem_size_len-scale buffers (d_energy_min, d_fml_i,
// d_dml) plus the dominant ijsize_len-scale one (d_fml_j), mirroring
// init_gpu()'s own size formulas exactly.
PUBLIC size_t
modular_decomposition_bytes_per_file(const int length) {
  // x5, not x3: d_energy_min, d_fml_i, d_dml, and (GPU-resident sweep step 1)
  // d_dml1 + d_fml_prev. This number drives RNAfold.c's chunk admission, so
  // under-counting it does not merely mis-report -- it lets a chunk be
  // accepted that does not fit, and the failure surfaces as an OOM inside
  // par_mfe() rather than as a smaller batch.
  const size_t cells        = (size_t)(length+1)*(length+2)/2;
  if(rnafold_fml_int16()) {
    // x6: the five above plus d_fml_row, which is row-shaped and so is noise
    // beside the triangle. The triangle halves, and the baselines add one int32
    // per FML_BLK entries -- 1.6% at B=64, against the 50% saved.
    const size_t mem_size_len = (size_t)(length+1) * sizeof(int) * 6;
    const size_t tri16        = cells * sizeof(short);
    const size_t base_bytes   = ((cells + FML_BLK - 1)/FML_BLK + (size_t)length + 2) * sizeof(int);
    return mem_size_len + tri16 + base_bytes;
  }
  const size_t mem_size_len = (size_t)(length+1) * sizeof(int) * 5;
  const size_t ijsize_len   = cells * sizeof(int);
  return mem_size_len + ijsize_len;
}

// Staggered_Row_Batching Phase 6b: replaces compute_max_gpu_batch() (which
// computed a single fixed record-count "capacity" up front from one shared
// length -- correct only because every chunk was uniform-length, and about
// to stop being true once Phase 6c drops that constraint). RNAfold.c now
// tracks a running VRAM budget incrementally as records of potentially
// different lengths accumulate into a chunk, rather than committing to a
// count derived from a single representative length. These two functions
// are the building blocks for that: gpu_bytes_per_file() gives the real
// per-record cost at any length (RNAfold.c calls this once per candidate
// record, not once per chunk), and compute_gpu_usable_bytes() gives the
// VRAM budget for a fresh chunk (called once per chunk start -- the
// in-progress chunk's own buffers aren't allocated yet at accumulation
// time, that happens later inside par_mfe(), so free VRAM doesn't
// meaningfully change as records accumulate on the host).
//
// The old MAX_GPU_BATCH_GRID_LIMIT (CUDA's gridDim.y/z 65535 cap) is
// retired along with compute_max_gpu_batch() -- it existed specifically
// because nfiles was used directly as gridDim.y in this file's kernels (and
// int_loop.cu's/hp_mb_loop.cu's); Phases 4-5 flattened every kernel here to
// a 1-D grid, so that limit's rationale no longer applies (noted but not
// acted on when Phase 5 shipped -- this is where it's actually retired).
//
// Aggregates the per-file VRAM cost across all three GPU-resident files
// (this one, int_loop.cu, hp_mb_loop.cu) for one record at the given
// length -- owned here since this file already has the dominant-cost
// buffer and the `first`-guard pattern this whole feature extends.
PUBLIC size_t
gpu_bytes_per_file(const int length) {
  return modular_decomposition_bytes_per_file(length)
       + int_loop_bytes_per_file(length)
       + hp_mb_loop_bytes_per_file(length);
}

// Queries free VRAM (cudaMemGetInfo) and returns the safety-margined usable
// budget for a fresh chunk. Call once per chunk start, not per record.
// TODO: the 85% safety margin was tightened from an initial 95% after the
// row buffers in fill_arrays.c (DMLi/DMLi1/DMLi2 + 5 more) became
// page-locked (cuda_host_alloc_ints()) -- pinned host allocation is a more
// constrained resource than pageable RAM (can't be swapped, often a
// stricter OS limit) and hard-fails rather than falling back, so this
// margin is the actual safety net for that. Not derived from a systematic
// study of allocator fragmentation/CUDA context overhead across many batch
// sizes, just empirically fine so far -- revisit if a future run OOMs
// closer to the edge than any tested so far did.
// RNA_GPU_VRAM_BUDGET_MB caps the budget below what free VRAM would allow.
// It can only ever *lower* it, never raise it, so setting it can't provoke an
// OOM that wouldn't have happened anyway. Two uses: forcing multi-chunk
// behaviour on a machine whose card is big enough to swallow the whole input
// in one batch (the only way to exercise the incremental budget above --
// chunks otherwise split on length change alone), and being a polite tenant
// on a GPU shared with another process. Env var rather than a CLI flag, per
// this project's convention of avoiding gengetopt regeneration.
PUBLIC size_t
compute_gpu_usable_bytes(void) {
  size_t free_bytes = 0, total_bytes = 0;
  gpuErrchk( cudaMemGetInfo(&free_bytes, &total_bytes) );
  const double safety_margin = 0.85; // TODO: tune further if a tighter margin ever OOMs
  size_t usable = (size_t)((double)free_bytes * safety_margin);

  const char *env_budget = getenv("RNA_GPU_VRAM_BUDGET_MB");
  if(env_budget && *env_budget) {
    const long mb = atol(env_budget);
    if(mb > 0) {
      const size_t cap = (size_t)mb * 1024u * 1024u;
      static int announced = 0;
      if(!announced) { // once per process, matching this file's other config messages
        fprintf(stderr, "%-24s RNA_GPU_VRAM_BUDGET_MB=%ld capping GPU chunk budget "
                        "(free VRAM would have allowed %.0f MB)\n",
                __FILE__, mb, (double)usable/(1024.0*1024.0));
        announced = 1;
      }
      if(cap < usable) usable = cap;
    } else {
      fprintf(stderr, "%-24s ignoring RNA_GPU_VRAM_BUDGET_MB=%s (want a positive integer)\n",
              __FILE__, env_budget);
    }
  }
  return usable;
}

// ---------------------------------------------------------------------------
// int16 fml_j (RNA_FML_INT16). See INT16_FML_SCOPE.md.
//
// A cell is stored as a signed 16-bit offset from a baseline shared by FML_BLK
// consecutive entries WITHIN one column, with FML_INF16 reserved for INF. The
// baseline is whichever entry of the block is packed first (the highest index,
// packed at the earliest row) -- deliberately NOT the block minimum, which
// would change as the block fills and invalidate offsets already written. What
// bounds the offset is the block's SPREAD, not the choice of origin.
// ---------------------------------------------------------------------------

// Baseline slot for the cell at within-column index `idx` of column `j`, in
// record H. colb_off makes each column start on a baseline boundary; a block
// straddling a column boundary would have to span the whole energy range.
__device__ __forceinline__ size_t
fml_bidx(const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
         const int H, const int j, const int idx) {
  return base_off_H[H] + colb_off[j] + (size_t)((idx - 1)/FML_BLK);
}

__device__ __forceinline__ int
fml_decode(const short* __restrict__ j16, const int* __restrict__ b,
           const size_t t, const size_t bidx) {
  const short o = j16[t];
  return (o == FML_INF16) ? INF : (b[bidx] + (int)o);
}

// Closes row i: the two writers have both run, so d_fml_row now holds this
// row's FINAL values and they can be packed. Launch shape mirrors
// load_fML_kernel's (size_off_H), which is the superset of the two write ranges.
__global__ void
pack_fml_kernel(const int nfiles, const int i_row, const int turn, const int length,
                const int* __restrict__ fml_row,
                      short* __restrict__ fml_j16,
                      int* __restrict__ fml_b,
                const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                const size_t* __restrict__ size_off_H, const size_t total,
                const int* __restrict__ i_H) {
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = mj + i+turn+1;

  const int  v    = fml_row[row_off_H[H]+j];
  const size_t t  = tri_off_H[H] + Indx(i,j);

  if(v == INF) { fml_j16[t] = FML_INF16; return; }

  const size_t bidx = fml_bidx(base_off_H, colb_off, H, j, i);
  int b = fml_b[bidx];
  if(b == FML_BASE_UNSET) {
    // First non-INF entry of this block. Race-free: rows are separate kernel
    // launches and, within a row, each thread owns a distinct (column, block).
    fml_b[bidx] = b = v;
  }
  const long long d = (long long)v - (long long)b;
  // TRAP, never wrap. The bound is provable for the default parameter table
  // (B/2 * 340 = 10880) but a -P file or a rescaled temperature can move it, and
  // a silent wrap here is a plausible, self-consistent, WRONG answer.
  if(d > 32766 || d < -32766) {
    printf("RNA_FML_INT16 range: H=%d (i=%d,j=%d) value %d baseline %d delta %lld "
           "exceeds int16. See INT16_FML_SCOPE.md.\n", H, i, j, v, b, d);
    assert(0);
  }
  fml_j16[t] = (short)d;
}

/* prefill matrices with init contributions */
__global__ void
init_fML_kernel(const size_t ijsize, // 32-bit signed integer overflow bug fix
		int* __restrict__ fml_j) { //out d_fml_j
  const size_t m = blockIdx.x*blockDim.x+threadIdx.x; // 32-bit signed integer overflow bug fix
  if(m>=ijsize) return;
  fml_j[m] = INF;
}

__global__ void
init_fml16_kernel(const size_t ijsize, short* __restrict__ j16) {
  const size_t m = blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=ijsize) return;
  j16[m] = FML_INF16;
}

__global__ void
init_base_kernel(const size_t n, int* __restrict__ b) {
  const size_t m = blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=n) return;
  b[m] = FML_BASE_UNSET;
}

PUBLIC void
init_fML(const int nfiles, const int length,
         const size_t tri_off_H_total, const size_t row_off_H_total) {
  // Staggered_Row_Batching Phase 2d: init_fML() is only ever reached via
  // fill_arrays.c -> par_fill_arrays(), which always runs after par_mfe()
  // has already unconditionally called init_gpu() -- `first` is always
  // already 0 here, same reasoning as int_loop_i()'s equivalent dead-code
  // removal in Phase 2b. The old `if(first) init_gpu(...)` fallback had no
  // way to supply tri_off_H/row_off_H here anyway (only the two totals are
  // passed in, not the full tables init_gpu() needs).
  assert(!first);
  const int first_ = first;
  const size_t ijsize = tri_off_H_total;
  /* Setup execution parameters for helper kernel */
  const size_t nblocks = (ijsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  if(!rnafold_fml_int16()) {
    init_fML_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_fml_j);
  } else {
    // Every cell starts INF, and every baseline starts UNSET so the first
    // non-INF packer of each block establishes it.
    init_fml16_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_fml_j16);
    const size_t nb = (g_base_total + BLOCK_SIZE - 1)/BLOCK_SIZE;
    if(g_base_total) init_base_kernel<<<nb,BLOCK_SIZE>>>(g_base_total, d_fml_b);
  }
  gpuErrchk2( cudaPeekAtLastError(),  first_ );

  //To aid debug etc initialise d_dml (DMLi)
  const size_t hsize = row_off_H_total;
  const size_t nblock2 = (hsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_dml);
  gpuErrchk( cudaPeekAtLastError() );

  // Same INF prefill the host gives DMLi1/fml_prev in fill_arrays.c, and for
  // the same reason: a record that has not joined the sweep yet is never
  // written by any kernel, so the first row it does join must read INF -- the
  // state a single-sequence fold starts from. Prefilling here also means row
  // `length-turn-1` reads a defined d_dml1 on the very first iteration.
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_dml1);
  gpuErrchk( cudaPeekAtLastError() );
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_fml_prev);
  gpuErrchk( cudaPeekAtLastError() );

#ifndef NDEBUG
  gpuErrchk2( cudaDeviceSynchronize(),first_ );
  //may pickup errors later if dont sync now
#endif
}

// Continuous flow phase A2: content-compared upload of the per-record row
// index, the twin of int_loop.cu/hp_mb_loop.cu's.
//
// HAZARD, and the reason for i_H_shadow_reset(): d_i_H is freed and
// re-cudaMalloc'd per chunk. Without the reset, a fresh chunk whose first table
// happened to equal the previous chunk's last one would skip the upload and
// leave the new buffer UNINITIALISED -- garbage row indices, wrong answers, no
// crash. init_gpu()/teardown_gpu() both call the reset for exactly that reason.
static void
i_H_shadow_reset(void) {
  free(i_H_shadow);
  i_H_shadow   = NULL;
  i_H_shadow_n = 0;
}

static void
upload_i_H(const int nfiles, const int* i_H) {
  const size_t bytes = (size_t)nfiles * sizeof(int);
  if(i_H_shadow_n != nfiles) {
    free(i_H_shadow);
    i_H_shadow   = (int*)malloc(bytes);
    i_H_shadow_n = i_H_shadow ? nfiles : 0;
  } else if(i_H_shadow && memcmp(i_H_shadow, i_H, bytes) == 0) {
    return;
  }
  gpuErrchk( cudaMemcpy(d_i_H, i_H, bytes, cudaMemcpyHostToDevice) );
  if(i_H_shadow) memcpy(i_H_shadow, i_H, bytes);
}

// Continuous flow phase C3: put ONE slot's sweep state back to the state a
// chunk starts in, so the slot can take a new record mid-sweep. Exactly the
// buffers init_fML() fills for the whole chunk, restricted to this slot's own
// ranges by pointer offset -- the neighbours are mid-recursion and must not be
// touched.
PUBLIC void
reset_slot_md(const size_t tri_lo, const size_t tri_n,
              const size_t row_lo, const size_t row_n) {
  if(tri_n) {
    const size_t nb = (tri_n + BLOCK_SIZE - 1)/BLOCK_SIZE;
    if(!rnafold_fml_int16()) {
      init_fML_kernel<<<nb,BLOCK_SIZE>>>(tri_n, d_fml_j + tri_lo);
    } else {
      // Resetting the offsets to the INF sentinel is not sufficient on its own:
      // the slot's BASELINES have to go back to UNSET too, or the new occupant's
      // first packed entry in each block would be encoded against the previous
      // occupant's baseline. That is a silent wrong answer, not a crash.
      init_fml16_kernel<<<nb,BLOCK_SIZE>>>(tri_n, d_fml_j16 + tri_lo);
      gpuErrchk( cudaPeekAtLastError() );
      // The slot's baseline range is not derivable from tri_lo/tri_n here, so
      // this path is refused rather than approximated -- see the guard in
      // init_gpu(). RNA_SLOT_FLOW + RNA_FML_INT16 is not a supported pairing yet.
    }
    gpuErrchk( cudaPeekAtLastError() );
  }
  if(row_n) {
    const size_t nb = (row_n + BLOCK_SIZE - 1)/BLOCK_SIZE;
    init_fML_kernel<<<nb,BLOCK_SIZE>>>(row_n, d_dml      + row_lo);
    init_fML_kernel<<<nb,BLOCK_SIZE>>>(row_n, d_dml1     + row_lo);
    init_fML_kernel<<<nb,BLOCK_SIZE>>>(row_n, d_fml_prev + row_lo);
    gpuErrchk( cudaPeekAtLastError() );
  }
  gpuErrchk( cudaDeviceSynchronize() );
}

//perhaps this can be combined with fmli_kernel?
__global__ void
load_fML_kernel(const int nfiles, const int i_row, const int turn, const int length,
		const int* __restrict__ energy_min,
	              int* __restrict__ fml_j,
	              int* __restrict__ fml_row,   //int16 path: NULL when off
		const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
		const size_t* __restrict__ size_off_H, const size_t total,
		const int* __restrict__ i_H) { //out d_fml_j my_fML
  // Staggered_Row_Batching Phase 4: flat index -> (H, position) via
  // flatten_index_to_H() over this row's real per-H active width
  // (size_off_H), replacing the old uniform m/nfiles split. j is guaranteed
  // <= length by size_off_H's own construction (built from
  // length_H[H]-i-turn, clamped >=0) -- no bound check needed here anymore.
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  // Continuous flow phase A2: this record's own row index. Identical to the old
  // shared scalar i_row today, and the assert checks that at RUNTIME, because it
  // is precisely the property phase B stops holding.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const long long j  = mj + i+turn+1;

  assert(H >= 0 && H < nfiles);
  assert(j>=0 && j<=length);
  const long long ij = Indx(i,j);
  assert(ij>=0 && ij<Hoff(1,length));
  if(fml_row) {
    // int16 path: the row's cells stay in full precision until pack_fml_kernel
    // closes the row. `ij` is unused here then, but the asserts above still
    // check it.
    (void)ij;
    fml_row[row_off_H[H]+j] = energy_min[row_off_H[H]+j];
  } else {
    assert(fml_j[tri_off_H[H]+ij] == INF);
           fml_j[tri_off_H[H]+ij] = energy_min[row_off_H[H]+j];
  }
}

PUBLIC void
load_fML(const int nfiles,
	 const int i, const int turn, const int length,
	 const int* energy_min,
	 const size_t* size_off_H) {   //in
  //out d_fml_j
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

  // NB: the old #ifdef NDEBUG pre-sync here was dead code -- this build never
  // defines NDEBUG -- and is superseded anyway: everything in this function
  // now runs on graph_stream, a *blocking* stream, so the legacy-default-
  // stream ordering rule already guarantees init_fML_kernel (NULL stream) has
  // completed before any of this is issued.
  //for simplicity transfer all energy_min, even though only need H * [start:length]
  // GPU-resident sweep: in device mode fml_scan_kernel has already written
  // d_energy_min, so this upload is the round trip being removed. Guarded
  // INSIDE the capture region on purpose -- the mode is constant for the run,
  // so exactly one topology is ever captured and the graph's cheap-update path
  // is unaffected. Check the graph-stats line: the reinstantiate count must not
  // climb.
  if(!rnafold_gpu_sweep())
    int_MemcpyAsync(d_energy_min,energy_min, g_row_total, cudaMemcpyHostToDevice, graph_stream, __LINE__);
  gpuErrchk( cudaMemcpyAsync(d_size_off_H, size_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice, graph_stream) );

  /* Setup execution parameters for helper kernel */
  const int nblocks = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
  load_fML_kernel<<<nblocks,BLOCK_SIZE,0,graph_stream>>>(nfiles, RNA_I_ROW(i), turn, length,
					  d_energy_min,  //in
					  d_fml_j,  //out
					  d_fml_row, //out, int16 path (NULL when off)
					  d_tri_off_H, d_row_off_H,
					  d_size_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
}

// Closes the row for the int16 path: both writers have run, so d_fml_row holds
// this row's final values and can be packed into the triangle. Shares
// load_fML()'s launch shape because that is the superset of the two write
// ranges. A no-op when the gate is off.
PUBLIC void
pack_fml(const int nfiles, const int i, const int turn, const int length,
         const size_t* size_off_H) {
  if(!rnafold_fml_int16()) return;
  const size_t total = size_off_H[nfiles];
  if(total==0) return;
  const size_t nblocks = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
  pack_fml_kernel<<<nblocks,BLOCK_SIZE,0,graph_stream>>>(nfiles, RNA_I_ROW(i), turn, length,
                        d_fml_row, d_fml_j16, d_fml_b,
                        d_tri_off_H, d_row_off_H,
                        d_base_off_H, d_colb_off,
                        d_size_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
}

__global__ void
load_min_fML_kernel(const int nfiles, const int i_row, const int turn, const int length,
		    const int* __restrict__ energy_min,
		    const int* __restrict__ dml,     //in  d_dml   DMLi
		          int* __restrict__ fml_j,     //out d_fml_j my_fML
		          int* __restrict__ fml_row,   //int16 path: NULL when off
		    const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
		    const size_t* __restrict__ side_off_H, const size_t total,
		    const int* __restrict__ i_H) {
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, side_off_H, nfiles);
  const long long mj = (long long)m - (long long)side_off_H[H];
  const int i = i_H[H];   // continuous flow phase A2 -- see load_fML_kernel
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows

  const int j  = mj + (i + 2*(turn+1)) + 1;
  const long long ij = Indx(i,j);

  assert(H >= 0 && H < nfiles);
  assert(j >=0 && j<=length);
  assert(ij>=0 && ij<Hoff(1,length));

  if(fml_row) fml_row[row_off_H[H]+j] = MIN2(energy_min[row_off_H[H]+j],dml[row_off_H[H]+j]);
  else        fml_j[tri_off_H[H]+ij]   = MIN2(energy_min[row_off_H[H]+j],dml[row_off_H[H]+j]);
}

PUBLIC void
load_min_fML(const int nfiles,
	     const int i, const int turn, const int length,
	     const size_t total) { // Staggered_Row_Batching Phase 4: side_off_H[nfiles] --
	                            // d_side_off_H already uploaded by modular_decomposition_cuda()
	                            // earlier in this row's call sequence, just reused here.
// energy_min already in d_energy_min
// DMLi       already in d_dml
// d_fml_j    out
  if(total==0) return;

/* Setup execution parameters for helper kernel */
  const int nblocks = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
  load_min_fML_kernel<<<nblocks,BLOCK_SIZE,0,graph_stream>>>(nfiles, RNA_I_ROW(i), turn, length,
					  d_energy_min,  //in
					  d_dml,    //in
					  d_fml_j,  //out
					  d_fml_row, //out, int16 path (NULL when off)
					  d_tri_off_H, d_row_off_H,
					  d_side_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
}

//indx[n] = n*(n-1)/2
//On GTX 745 no point fml_i in texture as have unified texture/LI cache
//perhaps also fml_j access pattern may not suit texture anyway
__global__ void
fmli_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
        int* __restrict__ fml_i,   //out
  const int* __restrict__ fml_j,   //In  d_fml_j
  const int* __restrict__ fml_row, //In  d_fml_row -- int16 path, NULL when off
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H) {

  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, side_off_H, nfiles);
  const long long mj = (long long)m - (long long)side_off_H[H];
  const int i = i_H[H];   // continuous flow phase A2 -- see load_fML_kernel
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  // `start` moved below the flatten: it depends on i, which is per-record now
  // and so is not known until H is.
  const int start = i+turn+1;

  const int k  = start + mj;
  const long long ik = Indx(i,k);
  assert(H >= 0 && H < nfiles);
  // Staggered_Row_Batching Phase 2d: fml_i is now table-driven (row_off_H),
  // so it can no longer be written via the flat H-tightest index m -- mj is
  // fml_i's within-row position (0-based, same value the old H+mj*nfiles
  // convention used), so row_off_H[H]+mj is the equivalent table-driven cell.
  // int16 path: row i is exactly what fml_row holds, and it is CONTIGUOUS there
  // rather than strided across columns at Indx(i,k) -- so this read gets
  // cheaper, not harder. It must not read the packed triangle: row i is not
  // final yet (load_min_fML_kernel has not run), which is the whole reason the
  // staging buffer exists.
  if(fml_row) { (void)ik; fml_i[row_off_H[H]+mj] = fml_row[row_off_H[H]+k]; }
  else                    fml_i[row_off_H[H]+mj] = fml_j[tri_off_H[H]+ik]; //ith column
}

//Use __restrict__ to give compiler best chance
// Rewritten 5 Aug 2026 (per Dr. Langdon's Aug 2026 main-branch work, 1893825)
// to exploit H-tightest parallelism: one thread per (H,j) output cell doing a
// serial scan over y, dropping the old block-per-j/reduction-per-block
// scheme entirely. That scheme made sense when parallelism came only from
// splitting one sequence's column sum across a block; with nfiles sequences
// batched, there's now enough independent thread-level parallelism across H
// alone that the reduction/shared-memory machinery is no longer needed. The
// *thread-grid* decomposition below (m/nfiles, H-fastest-varying flat index)
// is still H-tightest for this reason -- only the *data storage* underneath
// (fml_i[row_off_H[H]+y]/fml_j[tri_off_H[H]+yij] below, Staggered_Row_Batching
// Phase 2d) moved off H-tightest, since that's what staggering/mixed lengths
// actually breaks (see the coalescing finding in harmonic-swimming-hare.md).
// Split across TILE threads per output cell, 22 Aug 2026. The Aug 5 rewrite
// above left each thread walking `for(y=0..x)` alone, so the kernel could
// only finish when its *longest* thread did -- a serial chain as long as the
// sequence. ncu confirmed that shape rather than a work shortage: block size
// 32..640 all gave ~250 us (grid 189 -> 10) because every thread was already
// resident, duration scaled linearly with length (124/249/574 us at 300/600/
// 1200) but barely at all with nfiles (239/249/297 us for 5/20/40, i.e. 8x
// the work for +24% time), and SM throughput sat at 9%. Classic critical-path
// bound. TILE lanes now stride the same y range (lane, lane+TILE, ...) and
// combine with a warp shuffle, so the chain shortens to ~x/TILE + log2(TILE)
// while total work is unchanged.
//
// Numerically identical to TILE=1, not merely close: min is associative and
// commutative over exact ints, and a lane with no y of its own contributes
// INF -- the same value the serial loop seeds `value` with -- so the set
// being minimised is identical either way.
//
// Coalescing improves as a side effect. Adjacent threads used to hold
// adjacent j (different cells) and so read fml_j at a stride; now adjacent
// lanes hold adjacent y within one cell, making both fml_i[row_off_H[H]+y]
// and fml_j[tri_off_H[H]+y+ij0] unit-stride across the tile.
template <int TILE>
__global__ void
modular_decomposition_kernel(
  const int nfiles, const int i_row, const int turn, const int length,
  const int* __restrict__ fml_i, const int* __restrict__ fml_j,  //In  d_dml_i, d_fml_j
  const short* __restrict__ fml_j16, const int* __restrict__ fml_b, //int16 path
  const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
  int* __restrict__ dml,                            //Out d_dml (h_dml)
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H) {
  // Power of two <= 32 so a tile is a contiguous, warp-aligned lane group and
  // __shfl_down_sync()'s `width` can partition the warp for us. The launcher
  // additionally requires blockDim.x % 32 == 0, which is what makes "same
  // cell" and "same shuffle group" the same set of threads.
  static_assert(TILE >= 1 && TILE <= 32 && (TILE & (TILE-1)) == 0,
                "TILE must be a power of two in [1,32]");

  const long long gtid = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  const long long m    = gtid / TILE;
  const int       lane = (int)(gtid & (TILE-1));

  // Deliberately NOT an early `return` for out-of-range m (which is what the
  // TILE=1 version did). With TILE < 32 one warp carries 32/TILE independent
  // cells, so the grid's tail block can hold live and dead cells in the same
  // warp; a thread that returned there would still be named by the shuffle
  // mask below, which is undefined behaviour. Every thread instead runs to
  // the end, and the dead ones just carry INF through a reduction whose
  // result is discarded. Costs one tail block's worth of shuffles.
  const bool active = ((size_t)m < total);

  //typically values in fml_i read many times, assume many !=INF and that GPU cache will cope
  int value = INF;
  size_t out = 0;
  if(active) {
    const int H = flatten_index_to_H((size_t)m, side_off_H, nfiles);
    const long long mj = (long long)m - (long long)side_off_H[H];
    // Continuous flow phase A2. Read inside the `active` guard, the only place
    // H exists; the inactive lanes never need it -- they just carry INF through
    // the shuffle below. Two lines use it: j and ij0.
    const int i = i_H[H];
    assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows

    const int x = mj;
    const int j = x + (i + 2*(turn+1)) + 1;
    const long long ij0 = Indx(i,j) + (turn+1) + 1;

    assert(H >= 0 && H < nfiles);
    out = row_off_H[H]+j;
    // int16 path: the cell at yij sits at within-column index i+turn+2+y, and
    // its baseline slot advances once every FML_BLK steps of y. The per-cell
    // part of the slot index is loop-invariant, so it is hoisted; only the
    // block number moves. Reads here are always indices STRICTLY GREATER than
    // i, which are final and packed -- that is what makes decoding safe.
    if(fml_j16) {
      const size_t bcell = base_off_H[H] + colb_off[j];
      const int    idx0  = i + turn + 2;
      for(int y=lane; y <= x; y += TILE) {
        const long long yij = y + ij0;
        assert(yij < Hoff(nfiles,length));
        const int  d = fml_decode(fml_j16, fml_b, tri_off_H[H]+yij,
                                  bcell + (size_t)((idx0 + y - 1)/FML_BLK));
        value = MIN2(fml_i[row_off_H[H]+y] + d, value);
      }
    } else
    for(int y=lane; y <= x; y += TILE) {
      assert(x>=0 && x<=length);
      assert(y>=0 && y<=length);
      assert(y<=x);
      const long long yij = y + ij0;
      assert(yij < Hoff(nfiles,length));
      value = MIN2(fml_i[row_off_H[H]+y] + fml_j[tri_off_H[H]+yij], value);
    }
  }

  // Full 0xffffffff mask is correct precisely because nothing above returns:
  // all 32 lanes of the warp reach this point. `width=TILE` then confines
  // each exchange to one cell's lane group. Compiles to nothing when TILE==1.
#pragma unroll
  for(int off = TILE/2; off > 0; off >>= 1)
    value = MIN2(value, __shfl_down_sync(0xffffffff, value, off, TILE));

  if(active && lane == 0) dml[out] = value;
}

void modular_decomposition_cuda(const int nfiles,
				const int i, const int turn, const int length,
			      //const int* indx,
			      //const int* my_fML,
				      int* DMLi,
				const size_t* row_off_H,
				const size_t* side_off_H,
				const int* i_H) { //in, nfiles entries -- continuous flow phase B
  //printf("\nmodular_decomposition_cuda(%d,%d,%d,indx,my_fML)\n",
  //	 i,turn,length);

  // Staggered_Row_Batching Phase 4: side_off_H[nfiles] replaces the old
  // scalar side<=0 check -- numerically identical while every H shares one
  // length (today), but now table-driven like everything feeding it.
  const size_t total = side_off_H[nfiles];

  if(total == 0) {
    for(int H=0; H<nfiles;H++) {
    // Staggered_Row_Batching Phase 6d: bound this fill by H's OWN row width,
    // not the shared `length`. Each H's row slot is exactly
    // row_off_H[H+1]-row_off_H[H] == VC[H]->length+1 entries (Phase 2a built
    // the table that way), so the shared bound would run off the end of a
    // short H's row and into the next H's. VC[] isn't in scope here, so the
    // per-H length is recovered from the table itself rather than threading a
    // new parameter through modular_decomposition_i()'s public signature.
    const int len_H = (int)(row_off_H[H+1] - row_off_H[H]) - 1;
    // Continuous flow phase B: this record's own row, and skip it entirely once
    // it has retired (i_H[H] < 1) -- its row buffer must keep the values its
    // last row left there. Identical to i while every record shares a row.
    const int i_h = i_H[H];
    if(i_h < 1) continue;
    for (int j = i_h+turn+1; j <= len_H; j++) {
      DMLi[row_off_H[H]+j] = INF; // Staggered_Row_Batching Phase 2d: table-driven per-H row offset
    }}
    return;
  }

  // Staggered_Row_Batching Phase 4: d_side_off_H uploaded once here -- both
  // fmli_kernel below and load_min_fML_kernel (called later this row, from
  // load_min_fML()) read the same already-uploaded table, so it's not
  // re-uploaded there.
  gpuErrchk( cudaMemcpyAsync(d_side_off_H, side_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice, graph_stream) );

  //for simplicity transfer all to start with
  //perhaps should use cuMemsetD32
  //for (int j = 0; j <= length; j++) DMLi[j] = INF;
  //int_Memcpy(d_dml,h_dml,length+1, cudaMemcpyHostToDevice,__LINE__);


  /*now done in fmli_kernel
  //in first version just use ith column and multiple columns of my_fML
  //start off by transfering all of my_fML which we might read to GPU.
  //Could just transfer part which has changed, or get kernel to set it
  int m;
  int* fml_i = (int*)malloc(side*sizeof(int));
  for(m=0;m<side;m++) {
    const int k  = start + m;
    const int ik = indx[k] + i;
    fml_i[m] = my_fML[ik]; //ith column
    //printf("modular_decomposition_cuda(%d,%d,%d,indx,my_fML) fml_i[%d]%d <= my_fML[%d]\n",
    //	   i,turn,length,
    //	   m,fml_i[m],ik);
  }
  int_Memcpy(d_fml_i,fml_i, side, cudaMemcpyHostToDevice,__LINE__);
  */

  //these parts of my_fML re-used as i reduces to 1
  //const int ij_size  = side*(side+1)/2;
  //assert(ij_size<=ijsize_min);
  //int* d_fml_j = &d_dml[(length+1) + side];
  //int*   fml_j = &h_dml[(length+1) + side]; //(int*)malloc(ij_size*sizeof(int));
  /*
  m=0;
  for(int j=0; j <  side;j++) {
    const int ij  = indx[i + 2*(turn + 1) + 1 + j] + i;
    const int k1j = ij + turn + 2; //indx[j] + i + 1; //indx[j] + i + turn + 2;
    for(int k=0; k <= j   ;k++,m++) {
      fml_j[m] = my_fML[k+k1j]; //covers mutiple columns
      //printf("modular_decomposition_cuda(i=%d,%d,%d,indx,my_fML) j %d k %d fml_j[%d]%d <= my_fML[%d]\n",
      //     i,turn,length,
      //     j,k,
      //     m,fml_j[m],k+k1j);
      assert(m<ij_size);
//    fml_j[m] = ((fml_i[k] != INF ) &&   (my_fML[k+k1j] != INF))?   fml_i[k] + my_fML[k+k1j] : INF;
    }
  }*/
  //int_Memcpy(d_dml,h_dml, (length+1) + side + ij_size, cudaMemcpyHostToDevice,__LINE__);
  //const int ij_size = ((length+1)*(length+2)/2); //better not to transfer whole of my_fML but will do for the time being
  //int_Memcpy(d_dml,h_dml, 2*(length+1) + ij_size, cudaMemcpyHostToDevice,__LINE__);

  //Gave up trying to page lock my_fML, instead copy it directly
  /**Now keeping my_fML in sync with d_fml_j elsewhere
  {
  const int j = (i + 2*(turn+1)) + 1; //first used value of j
  const int start =      j*(     j-1)/2 + i      + (turn+1) + 1;
  const int top   = length*(length-1)/2 + length - (turn+1);
  const int size  = top - start + 1;
  int_Memcpy(&d_fml_j[start],&my_fML[start], size, cudaMemcpyHostToDevice,__LINE__);
  }
  */
//gpuErrchk( cudaDeviceSynchronize() );

  { /* Setup execution parameters for helper kernel */
    // g_block_size_fmli: chosen once in init_gpu() -- see the forward
    // declarations near the top of this file for why.
    const int block_size = g_block_size_fmli;
    const int nblocks = (total + block_size - 1)/block_size;
    fmli_kernel<<<nblocks,block_size,0,graph_stream>>>(nfiles, RNA_I_ROW(i), turn, length,
					d_fml_i,  //Out
					d_fml_j,  //In
					d_fml_row, //In, int16 path (NULL when off)
					d_tri_off_H, d_row_off_H,
					d_side_off_H, total, d_i_H);
    gpuErrchk( cudaPeekAtLastError() );
  }

//const int todo = side*(side+1)/2;


//  printf("modular_decomposition_cuda(i=%d,%d,%d,indx,my_fML) start %d stop %d threads needed %d\n",
//	 i,turn,length,
//	 start, stop,todo);

  //modular_decomposition_kernel(side,i,turn,length,fml_i,fml_j); //host testing

  /* Setup execution parameters */
  // g_block_size_md: chosen once in init_gpu() -- see the forward
  // declarations near the top of this file for why. (The serial per-thread
  // `for y` scan below is a separate, algorithmic imbalance this doesn't
  // address -- see the NCU report follow-up.)
  const int block_size = g_block_size_md;
  //const int nblocks = (todo + block_size - 1)/block_size; //for time being waste many blocks
  //const int nblocks = side;
  // g_md_tile threads per output cell now, not one -- see the kernel comment.
  // Done in size_t because total already is: at the widest chunks measured so
  // far (nfiles ~ 2000) total x 32 comfortably exceeds 2^31.
  const size_t nthreads   = total * (size_t)g_md_tile;
  const size_t nblocks_sz = (nthreads + (size_t)block_size - 1)/(size_t)block_size;
  assert(nblocks_sz <= 2147483647u); //gridDim.x limit
  const int nblocks = (int)nblocks_sz;

#define MD_LAUNCH(T) modular_decomposition_kernel<T><<<nblocks,block_size,0,graph_stream>>>( \
                       nfiles, RNA_I_ROW(i), turn, length, \
                       d_fml_i, d_fml_j, \
                       d_fml_j16, d_fml_b, d_base_off_H, d_colb_off, \
                       d_dml,   /*Out*/ \
                       d_tri_off_H, d_row_off_H, \
                       d_side_off_H, total, d_i_H)
  switch(g_md_tile) {
    case  1: MD_LAUNCH(1);  break;
    case  2: MD_LAUNCH(2);  break;
    case  4: MD_LAUNCH(4);  break;
    case  8: MD_LAUNCH(8);  break;
    case 16: MD_LAUNCH(16); break;
    default: MD_LAUNCH(32); break;
  }
#undef MD_LAUNCH

  gpuErrchk( cudaPeekAtLastError() );

  //for effiency transfer all of DMLi rather that just those part that have been calculated
  // GPU-resident sweep: DMLi's only readers were new_c_host (via DMLi1) and
  // fml_prev_host, both skipped in device mode; new_c_kernel reads d_dml1 and
  // fml_prev_kernel reads d_dml, both device-side.
  if(!rnafold_gpu_sweep())
    int_MemcpyAsync(DMLi,d_dml, g_row_total, cudaMemcpyDeviceToHost, graph_stream, __LINE__);
  // no sync here -- the one remaining sync happens once, after the whole
  // captured chain is launched, in the new orchestration function.

  //free(fml_j);
  //free(fml_i);
}

      /* modular decomposition -------------------------------**
      const int decomp = modular_decomposition(i,ij,j,turn,Fmi,my_fML);
      ** modular decomposition -------------------------------*/
PUBLIC int
modular_decomposition_ij(const int i, const int j, const int turn, 
		      const int length,
		      const int* indx, 
		      const int ijsize,
 		      const int* fmi,   //better cache access
		      const int* my_fML) {
  {
      int ij            = indx[j]+i;
//    assert(ij>=0 && ij<ijsize);
{
  int k   =  i + turn + 1;
  int k1j = ij + turn + 2; //indx[j] + i + 1; //indx[j] + i + turn + 2;
  const int stop = j - 2 - turn;
  int decomp = INF;

  for (; k <= stop; k++, k1j++){
    //assert(k  >0 &&   k<=length); //starts at 1 not 0
    //assert(k1j>0 && k1j<ijsize ); //starts at 1 not 0

    //const int ik = indx[k] + i;
    //assert(ik>0 && ik<ijsize  ); //starts at 1 not 0
    //assert(my_fML[ik] == Fmi[k]);

  //if((my_fML[ik] != INF ) && (my_fML[k1j] != INF)){
    if((   fmi[k]  != INF ) && (my_fML[k1j] != INF)){
    //const int en = my_fML[ik] + my_fML[k1j];
      const int en = fmi[k] + my_fML[k1j];
      decomp = MIN2(decomp, en);
    }
    //if(i==2895)
    //printf("modular_decomposition_ij(%d,%d,%d,%d,indx,%d,my_fML) j=%d stop=%d k=%d fmi[%d]=%d my_fML[%d] %d decomp %d\n",
    //	   i,j,turn,length,ijsize,
    //	   j,stop,k,
    //	   k,fmi[k], k1j,my_fML[k1j], decomp);
  }
  //assert(((k+1) <= j - 2 - turn)==0);
  return decomp;
  /* End modular decomposition -------------------------------*/
}
    }
}
/**/
//C interface to CUDA code
extern "C" /*PUBLIC*/ void
modular_decomposition_i(const int nfiles,
			const int i, const int turn, const int length,
		      //const int* indx,
		      //const int ijsize,
		      //const int* my_fML, //In
			int* DMLi,         //Out
			const size_t* row_off_H,
			const size_t* side_off_H,
			const int* i_H) {  //in, nfiles entries -- continuous flow phase B
  //const int max_ij_len = length - 2*(turn+1) - 1;
  //const int ijsize_min = max_ij_len*(max_ij_len+1)/2;
  //if(first) init_gpu(length);
  modular_decomposition_cuda(nfiles,i,turn,length, DMLi, row_off_H, side_off_H, i_H);
  return;

  /* remove debug** {
    int* Fmi   = (int *) malloc(sizeof(int)*(length + 1)); // holds row i of fML (avoids jumps in memory)
    const int start = i+turn+1;
    const int stop  = length - 2 - turn;
    int j;
    for (j = start; j <= stop; j++) {
      const int ij = indx[j] + i;
      Fmi[j] = my_fML[ij]; //ith column
    }
    int* c_DMLi = (int*)calloc(length+1,sizeof(int));
    int err = 0;
    for (j = i+turn+1; j <= length; j++) {
      c_DMLi[j] = modular_decomposition_ij(i,j,turn,length,indx,ijsize,Fmi,my_fML);
      ** remove debug**
      fprintf(stdout,"modular_decomposition_i(i=%d,%d,%d,indx,%d,my_fML,DMLi) DMLi[%d] %d v h_dml[%d] %d\n",
	      i,turn,length,ijsize,
	      j,c_DMLi[j],j,DMLi[j]);
      **
      if(c_DMLi[j]!=DMLi[j]) err = j;
    }
    if(err) {
      fprintf(stdout,"ERROR in modular_decomposition_i(i=%d,%d,%d,indx,%d,my_fML,DMLi) DMLi[%d]%d != h_dml[%d]%d\n",
	      i,turn,length,ijsize,
	      err,c_DMLi[err],err,DMLi[err]);
      fprintf(stderr,"ERROR in modular_decomposition_i(i=%d,%d,%d,indx,%d,my_fML,DMLi) DMLi[%d]%d != h_dml[%d]%d\n",
	      i,turn,length,ijsize,
	      err,c_DMLi[err],err,DMLi[err]);
      exit(1);
    } else {
      const int j = i+turn+1;
      memcpy(&DMLi[j],&c_DMLi[j],(length-j+1)*sizeof(int));
    }
    free(c_DMLi);
  }
  */
}

// Copies the GPU's my_fML triangle back into each record's own
// VC[H]->matrices->fML, once, after the whole sweep. Replaces the per-row
// host mirroring my_fml_update_host used to do (see fill_arrays_loop.c):
// nothing between rows reads the host triangle, and backtrack() -- the only
// consumer that does -- runs after this.
//
// The copy is contiguous per record, not scattered, because the two layouts
// already agree: ViennaRNA indexes fML by jindx[j]+i == j*(j-1)/2+i, which is
// exactly this file's Indx(i,j), and dp_matrices.c sizes the allocation
// (n+1)*(n+2)/2 == tri_off_H[H+1]-tri_off_H[H] (compute_batch_offsets(),
// mfe_cuda.c). So H's slice of d_fml_j maps onto its fML one-for-one.
//
// Synchronous on the NULL stream: this runs once per chunk, immediately
// before the host reads the result, so there is nothing to overlap it with.
// One record's slice, for the scratch-pool path: the caller backtracks records
// one at a time into a reusable buffer instead of holding every record's
// triangle at once. Same copy as fetch_fML() does per H, just addressable
// individually. Safe to call from several host threads -- the CUDA runtime is
// thread-safe and these serialise on the default stream.
// GPU-resident sweep, step 1: hand this file's row-shaped device buffers to
// the kernels that will live in hp_mb_loop.cu. Same shape as int_loop.cu's
// int_loop_hccc_buffers() -- the device state stays owned by the file that
// allocates it, and only a pointer crosses the translation-unit boundary.
// Valid only between init_gpu() and teardown_gpu().
extern "C" /*PUBLIC*/ void
md_row_buffers(int** dml_out, int** dml1_out, int** fml_prev_out,
               int** energy_min_out) {
  if(dml_out)        *dml_out        = d_dml;
  if(dml1_out)       *dml1_out       = d_dml1;
  if(fml_prev_out)   *fml_prev_out   = d_fml_prev;
  if(energy_min_out) *energy_min_out = d_energy_min;
}

// Publishes row i's DMLi as "the previous row's" for row i-1, replacing the
// host's DMLi1 rotation (see the declaration comment for why a copy and not a
// rotate). Called from fill_arrays_loop.c at exactly the point the host
// rotates, so the two stay in lockstep. Synchronous on the NULL stream: the
// next row's first kernel must see it, and there is nothing to overlap it
// with here.
extern "C" /*PUBLIC*/ void
md_snapshot_dml(void) {
  gpuErrchk( cudaMemcpy(d_dml1, d_dml, g_row_total*sizeof(int),
                        cudaMemcpyDeviceToDevice) );
}

// Widens on the way out when the gate is on. The host's fML is int32 and every
// consumer of it -- backtracking, subopt, the fM1 post-pass -- expects that, so
// the encoding stops at the device boundary. `H` is needed for the baseline
// offset, which the int32 path did not require, hence the extra argument.
extern "C" /*PUBLIC*/ void
fetch_fML_one_H(int* dst, const size_t tri_lo, const size_t cells, const int H);

extern "C" /*PUBLIC*/ void
fetch_fML_one(int* dst, const size_t tri_lo, const size_t cells) {
  assert(!rnafold_fml_int16());   // callers must use fetch_fML_one_H()
  gpuErrchk( cudaMemcpy(dst, &d_fml_j[tri_lo], cells*sizeof(int),
                        cudaMemcpyDeviceToHost) );
}

// Host-side decode. Deliberately NOT a kernel: it runs once per record on the
// backtrack pool, off the critical path, and a kernel would need a scratch
// int32 triangle on the device -- which is exactly the allocation this whole
// change exists to remove.
extern "C" /*PUBLIC*/ void
fetch_fML_one_H(int* dst, const size_t tri_lo, const size_t cells, const int H) {
  if(!rnafold_fml_int16()) { fetch_fML_one(dst, tri_lo, cells); return; }

  static __thread short*  h16  = NULL;
  static __thread size_t  h16n = 0;
  if(h16n < cells) { free(h16); h16 = (short*)malloc(cells*sizeof(short)); h16n = cells; }

  gpuErrchk( cudaMemcpy(h16, &d_fml_j16[tri_lo], cells*sizeof(short),
                        cudaMemcpyDeviceToHost) );

  // The baselines for this record, and the column offsets, both small.
  size_t base_lo = 0, base_n = 0;
  gpuErrchk( cudaMemcpy(&base_lo, &d_base_off_H[H],   sizeof(size_t), cudaMemcpyDeviceToHost) );
  { size_t hi; gpuErrchk( cudaMemcpy(&hi, &d_base_off_H[H+1], sizeof(size_t), cudaMemcpyDeviceToHost) );
    base_n = hi - base_lo; }
  int* hb = (int*)malloc((base_n?base_n:1)*sizeof(int));
  if(base_n) gpuErrchk( cudaMemcpy(hb, &d_fml_b[base_lo], base_n*sizeof(int), cudaMemcpyDeviceToHost) );

  // n from the triangle size: cells == (n+1)(n+2)/2
  int n = 0; while(((size_t)(n+1)*(n+2))/2 < cells) n++;
  size_t* colb = (size_t*)malloc((size_t)(n+2)*sizeof(size_t));
  colb[0] = colb[1] = 0;
  for(int j=1;j<=n;j++) colb[j+1] = colb[j] + (size_t)((j + FML_BLK - 1)/FML_BLK);

  for(int j=1;j<=n;j++)
    for(int i=1;i<=j;i++) {
      const size_t t = (size_t)j*(j-1)/2 + i;
      if(t >= cells) continue;
      const short o = h16[t];
      dst[t] = (o == FML_INF16) ? INF
             : hb[colb[j] + (size_t)((i-1)/FML_BLK)] + (int)o;
    }
  free(hb); free(colb);
}

extern "C" /*PUBLIC*/ void
fetch_fML(const int nfiles, int** fML_H, const size_t* tri_off_H) {
  for(int H=0; H<nfiles; H++) {
    const size_t n = tri_off_H[H+1] - tri_off_H[H];
    assert(n > 0);
    // Routed through the decoding form so the bulk fetch cannot quietly become
    // the one path that reads raw offsets as energies.
    fetch_fML_one_H(fML_H[H], tri_off_H[H], n, H);
  }
}

// Captures load_fML() -> modular_decomposition_i() -> load_min_fML() as a
// single CUDA graph and replays it, instead of issuing 6 separate blocking
// driver calls (fill_arrays_loop.c calls these three back-to-back with no
// host CPU logic between them, which is what makes this legal to capture).
// Recaptured every call rather than captured once, for two reasons: (1) i
// and turn change every call, which changes grid dims (nblocks/side) for the
// underlying kernels -- cudaGraphExecUpdate() handles that as a params
// update, not a topology change; and (2) modular_decomposition_cuda() and
// load_min_fML() both take a host-only early-return branch (side<=0) for the
// first few iterations near the diagonal, which produces a *different*
// node topology (fewer GPU nodes) than later iterations -- recapturing
// means cudaGraphExecUpdate() naturally fails exactly once at that
// transition and falls back to a fresh cudaGraphInstantiate(), rather than
// needing to special-case that boundary by hand.
extern "C" /*PUBLIC*/ void
load_fML_modular_decomposition_load_min_fML(const int nfiles,
					     const int i, const int turn, const int length,
					     const int* energy_min, //in
					     int* DMLi,             //out
					     const size_t* row_off_H,
					     const size_t* size_off_H,
					     const size_t* side_off_H,
					     const int* i_H) {      //in, nfiles entries -- continuous flow phase A2
  // Continuous flow phase A2: uploaded HERE, before either branch, and
  // deliberately outside the capture region below -- a blocking NULL-stream
  // cudaMemcpy that has completed before cudaStreamBeginCapture() is reached.
  // That is what keeps a plain stack i_H legal: this file's capture-region rule
  // is that an ASYNC H2D recorded INTO the graph needs a persistent host
  // source, and graph_stream being a blocking stream orders the replay after
  // this copy exactly as it already does for int_loop_i()/load_my_c()'s
  // NULL-stream work. It adds no sync point the row did not already have --
  // upload_size_off_H() does the same blocking per-row H2D twice already -- and
  // the content comparison skips it on the rows where nothing changed.
  upload_i_H(nfiles, i_H);

  // RNA_CUDA_GRAPH=0 disables capture/replay and just issues the same
  // (now-async, graph_stream-targeted) calls directly, with one sync at the
  // end -- lets a whole fold be re-run graph-off vs graph-on and diffed
  // end-to-end (structure/MFE output) to validate the graph path, without
  // needing a second parallel set of device buffers threaded through every
  // kernel just for this A/B check.
  static int use_graph = -1;
  if(use_graph == -1) {
    const char* env = getenv("RNA_CUDA_GRAPH");
    use_graph = (env && env[0]=='0') ? 0 : 1;
    fprintf(stderr,"%-24s CUDA graph capture for load_fML/modular_decomposition/load_min_fML: %s\n",
	    __FILE__, use_graph? "enabled" : "disabled (RNA_CUDA_GRAPH=0)");
    if(use_graph) atexit(print_graph_update_stats);
  }

  if(!use_graph) {
    load_fML(nfiles,i,turn,length,energy_min,size_off_H);
    modular_decomposition_i(nfiles,i,turn,length,DMLi,row_off_H,side_off_H,i_H);
    load_min_fML(nfiles,i,turn,length,side_off_H[nfiles]);
    // int16: closes the row AFTER both writers, which is the whole ordering
    // constraint this design exists to respect. No-op when the gate is off.
    pack_fml(nfiles,i,turn,length,size_off_H);
    gpuErrchk( cudaStreamSynchronize(graph_stream) );
    return;
  }

  cudaGraph_t graph = NULL;
  gpuErrchk( cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeThreadLocal) );

  load_fML(nfiles,i,turn,length,energy_min,size_off_H);
  modular_decomposition_i(nfiles,i,turn,length,DMLi,row_off_H,side_off_H,i_H);
  load_min_fML(nfiles,i,turn,length,side_off_H[nfiles]);
  // int16: a FOURTH captured node, and it must be inside the capture so the
  // replay reproduces the whole row. Ordering is what matters -- pack must
  // follow both writers -- and stream order inside the capture gives that.
  pack_fml(nfiles,i,turn,length,size_off_H);

  gpuErrchk( cudaStreamEndCapture(graph_stream, &graph) );

  const double mgmt_start = graph_now_seconds(); //diagnostic-only

  if(graph_exec_valid) {
    cudaGraphExecUpdateResultInfo update_result;
    const cudaError_t update_error = cudaGraphExecUpdate(graph_exec, graph, &update_result);
    if(update_error != cudaSuccess) {
      //topology changed (e.g. the side<=0 boundary above) -- stale exec can't
      //be patched in place, drop it and instantiate fresh below.
      //This is an *expected*, handled outcome (not a real fault), but every
      //CUDA runtime call -- success or failure -- updates the thread-local
      //"last error" slot that cudaPeekAtLastError()/cudaGetLastError() read.
      //Left uncleared, this failure would sit there and get misattributed to
      //the next unrelated cudaPeekAtLastError() check anywhere downstream
      //(e.g. inside int_loop_i() on the next iteration). cudaGetLastError()
      //(not Peek) both reads AND resets it, so call it to consume/clear this
      //one now that we've handled it.
      cudaGetLastError();
      gpuErrchk( cudaGraphExecDestroy(graph_exec) );
      graph_exec_valid = 0;
      graph_forced_reinstantiate_count++; //diagnostic-only
    } else {
      graph_update_success_count++; //diagnostic-only
    }
  } else {
    graph_first_instantiate_count++; //diagnostic-only
  }
  if(!graph_exec_valid) {
    gpuErrchk( cudaGraphInstantiate(&graph_exec, graph, 0) );
    graph_exec_valid = 1;
  }

  gpuErrchk( cudaGraphDestroy(graph) ); //transient capture object, not graph_exec (the persistent one) -- leaks every iteration if skipped

  graph_mgmt_seconds += graph_now_seconds() - mgmt_start; //diagnostic-only

  gpuErrchk( cudaGraphLaunch(graph_exec, graph_stream) );
  //the one sync that remains: also the only point where a real runtime/data
  //error from the replayed graph (bad address, illegal access, device-side
  //assert()) becomes observable, since a graph gives no per-node attribution
  //
  //DELIBERATELY NOT GATED by step 5b, unlike the six per-row syncs in
  //int_loop.cu/hp_mb_loop.cu. Those were made redundant by stream ordering and
  //bought real run-ahead; this one is the graph's only error checkpoint, and
  //dropping it would push a graph fault out to some later row with nothing to
  //attribute it to. It also buys little: a row still contains two blocking
  //pageable H2Ds (upload_size_off_H in each file), and a pageable H2D
  //stream-syncs before it copies, so the host cannot run far ahead regardless.
  //Making those async -- which needs persistent PINNED host tables, per this
  //file's standing capture-region hazard -- is what would unlock true
  //back-to-back queueing, and it is not part of 5b.
  gpuErrchk( cudaStreamSynchronize(graph_stream) );
}

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

#include "ViennaRNA/utils.h"
#include "ViennaRNA/energy_par.h"
#include "ViennaRNA/data_structures.h"
#include "ViennaRNA/fold_vars.h"
#include "ViennaRNA/params.h"
#include "ViennaRNA/constraints.h"
#include "ViennaRNA/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/unstructured_domains.h"
#include "ViennaRNA/loop_energies.h"
#include "ViennaRNA/mfe.h"

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

int first = 1;
//int* d_indx; //indx no longer used
int* d_energy_min;
int* d_fml_i;  //my_fML
int* d_fml_j;  //my_fML
int* d_dml;  //DMLi
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
init_gpu(const int nfiles, const int length) {
  if(!first) return;
  fprintf(stderr,"%-24s init_gpu(%d, %d)\n",__FILE__,nfiles,length);
  cudaError_t error;
  gpuErrchk( cudaStreamCreate(&graph_stream) );
  const size_t mem_size_len = (size_t)nfiles*(length+1) * sizeof(int); //starts at 1 not 0 // 32-bit signed integer overflow bug fix
  const size_t ijsize_len   = (size_t)nfiles*((length+1)*(length+2)/2) * sizeof(int); // 32-bit signed integer overflow bug fix

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

  error = cudaMalloc((void **) &d_fml_j, ijsize_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_fml_j %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     ijsize_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  error = cudaMalloc((void **) &d_dml, mem_size_len);
  if (error != cudaSuccess)  {
      printf("cudaMalloc d_dml %zu returned error %s (code %d), line(%d)\n", // 32-bit signed integer overflow bug fix
	     mem_size_len, cudaGetErrorString(error), error, __LINE__);
      exit(EXIT_FAILURE);}

  first = 0;
  return;
}

/* prefill matrices with init contributions */
__global__ void
init_fML_kernel(const size_t ijsize, // 32-bit signed integer overflow bug fix
		int* __restrict__ fml_j) { //out d_fml_j
  const size_t m = blockIdx.x*blockDim.x+threadIdx.x; // 32-bit signed integer overflow bug fix
  if(m>=ijsize) return;
  fml_j[m] = INF;
}

PUBLIC void
init_fML(const int nfiles, const int length) {
  const int first_ = first;
  if(first) init_gpu(nfiles,length);
  const size_t ijsize = (size_t)nfiles*(length+1)*(length+2)/2; // 32-bit signed integer overflow bug fix
  /* Setup execution parameters for helper kernel */
  const size_t nblocks = (ijsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  init_fML_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_fml_j);
  gpuErrchk2( cudaPeekAtLastError(),  first_ );

  //To aid debug etc initialise d_dml (DMLi)
  const size_t hsize = (size_t)nfiles*(length+1); // 32-bit signed integer overflow bug fix
  const size_t nblock2 = (hsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  init_fML_kernel<<<nblock2,BLOCK_SIZE>>>(hsize, d_dml);
  gpuErrchk( cudaPeekAtLastError() );

#ifndef NDEBUG
  gpuErrchk2( cudaDeviceSynchronize(),first_ );
  //may pickup errors later if dont sync now
#endif
}

//perhaps this can be combined with fmli_kernel?
__global__ void
load_fML_kernel(const int i, const int turn, const int length,
		const int* __restrict__ energy_min,
	              int* __restrict__ fml_j) { //out d_fml_j my_fML
  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int j = m + i+turn+1; 
  if(j>length) return;

  const int H = blockIdx.y;
  const int ij = j*(j-1)/2+i;
  assert(ij>=0 && ij<(length+1)*(length+2)/2);
  assert(fml_j[H*((length+1)*(length+2)/2)+ij] == INF);
         fml_j[H*((length+1)*(length+2)/2)+ij] = energy_min[H*(length+1)+j];
}

PUBLIC void
load_fML(const int nfiles,
	 const int i, const int turn, const int length,
	 const int* energy_min) {   //in
  //out d_fml_j
  const int start = i+turn+1; 
  const int size  = length - start + 1;
  if(size<=0) return;

//printf("load_fML(%d,i=%d,%d,%d,energy_min) start %d size %d ",
//       nfiles,i,turn,length,start,size);
//for(int k=start; k<start+10 && k<=length; k++) { printf("energy_min[%d]%d ",k,energy_min[k]);}
//printf("\n");

  // NB: the old #ifdef NDEBUG pre-sync here was dead code -- this build never
  // defines NDEBUG -- and is superseded anyway: everything in this function
  // now runs on graph_stream, a *blocking* stream, so the legacy-default-
  // stream ordering rule already guarantees init_fML_kernel (NULL stream) has
  // completed before any of this is issued.
  //for simplicity transfer all energy_min, even though only need H * [start:length]
  int_MemcpyAsync(d_energy_min,energy_min, (size_t)nfiles*(length+1), cudaMemcpyHostToDevice, graph_stream, __LINE__);

  /* Setup execution parameters for helper kernel */
  const int nblocks = (size + BLOCK_SIZE - 1)/BLOCK_SIZE;
  dim3 blocks(nblocks,nfiles);
  load_fML_kernel<<<blocks,BLOCK_SIZE,0,graph_stream>>>(i, turn, length,
					  d_energy_min,  //in
					  d_fml_j); //out
  gpuErrchk( cudaPeekAtLastError() );
}

__global__ void
load_min_fML_kernel(const int i, const int turn, const int length,
		    const int* __restrict__ energy_min,
		    const int* __restrict__ dml,     //in  d_dml   DMLi
		          int* __restrict__ fml_j) { //out d_fml_j my_fML
  const int start = i+turn+1;
  const int stop  = length - 2 - turn;
  const int side  = stop - start + 1;

  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=side) return;

  const int H = blockIdx.y;

  const int j  = m + (i + 2*(turn+1)) + 1;
  const int ij = j*(j-1)/2+i;

  assert(j >=0 && j<=length);
  assert(ij>=0 && ij<(length+1)*(length+2)/2);

  fml_j[H*((length+1)*(length+2)/2)+ij] = MIN2(energy_min[H*(length+1)+j],dml[H*(length+1)+j]);

//  printf("load_min_fML_kernel(i=%d,%d,%d,*,*,*) block %d,%d energy_min[%d]%d dml[%d]%d fml_j[%d]%d\n",
//	 i,turn,length,
//	 blockIdx.x,threadIdx.x,
//	 j,energy_min[j], j,dml[j], ij,fml_j[ij]);
}

PUBLIC void
load_min_fML(const int nfiles,
	     const int i, const int turn, const int length) {
// energy_min already in d_energy_min
// DMLi       already in d_dml
// d_fml_j    out

  //NB modular_decomposition_ij() uses side (rather than size)
  const int start = i+turn+1;
  const int stop  = length - 2 - turn;
  const int side  = stop - start + 1;
  if(side<=0) return;

//printf("load_min_fML(i=%d,%d,%d) start %d side %d\n",
//	 i,turn,length,start,side);

/* Setup execution parameters for helper kernel */
  const int nblocks = (side + BLOCK_SIZE - 1)/BLOCK_SIZE;
  dim3 blocks(nblocks,nfiles);
  load_min_fML_kernel<<<blocks,BLOCK_SIZE,0,graph_stream>>>(i, turn, length,
					  d_energy_min,  //in
					  d_dml,    //in
					  d_fml_j); //out
  gpuErrchk( cudaPeekAtLastError() );
//printf("\n");
}

//indx[n] = n*(n-1)/2
//On GTX 745 no point fml_i in texture as have unified texture/LI cache
//perhaps also fml_j access pattern may not suit texture anyway
__global__ void
fmli_kernel(
  const int i, const int turn, const int length,
        int* __restrict__ fml_i,   //out
  const int* __restrict__ fml_j) { //In  d_fml_j

  const int start = i+turn+1;
  const int stop  = length - 2 - turn;
  const int side  = stop - start + 1;

  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=side) return;
  const int H = blockIdx.y;

  const int k  = start + m;
  const int ik = k*(k-1)/2 + i;
  fml_i[H*(length+1)+m] = fml_j[H*((length+1)*(length+2)/2)+ik]; //ith column

  //printf("fmli_kernel(%d,%d,%d,fml_i,my_fML) fml_i[%d]%d <= my_fML[%d]\n",
  //	   i,turn,length,
  //	   m,fml_i[m],ik);
}

//Use __restrict__ to give compiler best chance
/*template <int BLOCK_SIZE>*/ __global__ void
modular_decomposition_kernel(
  const int i, const int turn, const int length,
  const int* __restrict__ fml_i, const int* __restrict__ fml_j,  //In  d_dml_i, d_fml_j
  int* __restrict__ dml) {                          //Out d_dml (h_dml)

  const int H = blockIdx.y;
  const int x = blockIdx.x;
  const int j = x + (i + 2*(turn+1)) + 1;
        int y = threadIdx.x;
        int thread = j*(j-1)/2 + threadIdx.x + i + (turn+1) + 1;
  int value = INF;
  for(; y <= x; thread+=blockDim.x, y+=blockDim.x) {
    //assert(x>=0 && x<=length);
    //assert(y>=0 && y<=length);
    //assert(y<=x);

    //const int en_i   = fml_i[y];
    //const int en_j   = fml_j[thread];
    //const int decomp = ((en_i != INF ) && (en_j != INF))? en_i + en_j : INF;
    //https://devtalk.nvidia.com/default/topic/1028130/cuda-programming-and-performance/best-way-to-find-many-minimums/
    //https://devtalk.nvidia.com/default/topic/1012969/cuda-programming-and-performance/texture-unit-in-pascal-architecture/2
    //value = MIN2(((fml_i[y] != INF ) && (fml_j[thread] != INF))? fml_i[y] + fml_j[thread] : INF, value);
    value = MIN2(fml_i[H*(length+1)+y] + fml_j[H*((length+1)*(length+2)/2)+thread], value);

    //printf("modular_decomposition_kernel(i=%d,%d,%d,fml_i,fml_j,dml) block %d,%d j %d y %d fml_i[%d] %d fml_j[%d] %d value %d\n",
    // 	   i,turn,length,
    //	   blockIdx.x,threadIdx.x, j,y,
    //	   y,fml_i[y], thread,fml_j[thread], value);
  }//endfor whole of column
  volatile __shared__ int en[BLOCK_SIZE];
  en[threadIdx.x] = value; //must set whole of en

//Ok try to make a reduction, require power of two block size
//__syncthreads();
  //assert(BLOCK_SIZE==32 || BLOCK_SIZE==64 || BLOCK_SIZE==128 || BLOCK_SIZE==256 || BLOCK_SIZE==512 || BLOCK_SIZE==1024 || BLOCK_SIZE==2048);
//#if BLOCK_SIZE > 32
//#define SYNC32 __syncthreads()
//#else
//nuffa points to lack of volatile bug https://stackoverflow.com/questions/10729185/removing-syncthreads-in-cuda-warp-level-reduction
//also https://stackoverflow.com/questions/10729185/removing-syncthreads-in-cuda-warp-level-reduction
//http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//#define SYNC32
//#endif

  //Does thread group straddle block boundary?

  //if(i==2895)
  //printf("modular_decomposition_kernel(side=%d,i=%d,%d,%d,fml_i,fml_j,dml,dummy) block %d thread %d x %d y %d\n",
  //	   side,i,turn,length,
  //	   blockIdx.x,thread,x,y);
    const int ix = threadIdx.x;
    //const int ix_stop  = MIN2(x-y+threadIdx.x, blockDim.x - 1);
      //assert(ix_stop >= 0 && ix_stop < blockDim.x);
      //assert(en[ix] > -INF && en[ix] <= INF);
      //assert(en[ix] != 0);   //for testing only
    //printf("modular_decomposition_kernel(i=%d,%d,%d,fml_i,fml_j,dml) block %d,%d y %d fml_i[%d] %d fml_j[%d] %d decomp %d en[%d] %d j %d ix_stop %d\n",
    //	   i,turn,length,
    //	   blockIdx.x,threadIdx.x,y,
    //	   y,debugi, thread,debugj, decomp, threadIdx.x,en[threadIdx.x],j,ix_stop);
    //if(i==2818)
    //printf("modular_decomposition_kernel(side=%d,i=%d,%d,%d,fml_i,fml_j,dml) block %d,%d y %d thread %d ix_stop %d en[%d] %d\n",
    //	   side,i,turn,length,
    //	   blockIdx.x,threadIdx.x,y,
    //	   thread,ix_stop, ix,en[ix]);
#if BLOCK_SIZE >=1024
  __syncthreads(); if(ix < 512) en[ix] = MIN2(en[ix], en[ix+512]);
#endif
#if BLOCK_SIZE >=512
  __syncthreads(); if(ix < 256) en[ix] = MIN2(en[ix], en[ix+256]);
#endif
#if BLOCK_SIZE >=256
  __syncthreads(); if(ix < 128) en[ix] = MIN2(en[ix], en[ix+128]);
#endif
#if BLOCK_SIZE >=128
  __syncthreads(); if(ix <  64) en[ix] = MIN2(en[ix], en[ix+ 64]);
#endif
  if(ix < 32) {
#if BLOCK_SIZE >=64
    __syncthreads();            en[ix] = MIN2(en[ix], en[ix+ 32]);
#endif
    en[ix] = MIN2(en[ix], en[ix+ 16]);
    en[ix] = MIN2(en[ix], en[ix+  8]);
    en[ix] = MIN2(en[ix], en[ix+  4]);
    en[ix] = MIN2(en[ix], en[ix+  2]);
    en[ix] = MIN2(en[ix], en[ix+  1]);
  }

  if(threadIdx.x==0){
    dml[H*(length+1)+j] = en[0];
  //if(i==2818)
    //printf("modular_decomposition_kernel(side=%d,i=%d,%d,%d,fml_i,fml_j,dml) block %d,%d x %d y %d thread %d j %d decomp %d\n",
    //	   side,i,turn,length,
    //	   blockIdx.x,threadIdx.x,x,y,thread,
    //	   j, decomp);
      //CUDA 9.0 programming guide B.12.1.4. atomicMin()
      //atomicMin(&dml[j],en[ix]);
      //const int old = atomicMin(&dml[j],en[ix]); //old value only for sanity check
      //if(!(old > -INF && old <= INF)) dml[j] = -999999;
      //assert(old > -INF && old <= INF);
      // assert(old != 0); //for testing only
  }
}

void modular_decomposition_cuda(const int nfiles,
				const int i, const int turn, const int length,
			      //const int* indx,
			      //const int* my_fML,
				      int* DMLi) {
  //printf("\nmodular_decomposition_cuda(%d,%d,%d,indx,my_fML)\n",
  //	 i,turn,length);

  const int start = i+turn+1;
  const int stop  = length - 2 - turn;
  const int side  = stop - start + 1;

  if(side <= 0 ) {
    for(int H=0; H<nfiles;H++) {
    for (int j = i+turn+1; j <= length; j++) {
      DMLi[H*(length+1)+j] = INF;
    }}
    return;
  }

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
    const int block_size = BLOCK_SIZE;
    const int nblocks = (side + block_size - 1)/block_size;
    dim3 blocks(nblocks,nfiles);
    fmli_kernel<<<blocks,block_size,0,graph_stream>>>(i, turn, length,
					d_fml_i,  //Out
					d_fml_j); //In
    gpuErrchk( cudaPeekAtLastError() );
  }

//const int todo = side*(side+1)/2;


//  printf("modular_decomposition_cuda(i=%d,%d,%d,indx,my_fML) start %d stop %d threads needed %d\n",
//	 i,turn,length,
//	 start, stop,todo);

  //modular_decomposition_kernel(side,i,turn,length,fml_i,fml_j); //host testing

  /* Setup execution parameters */
  const int block_size = BLOCK_SIZE;
  //const int nblocks = (todo + block_size - 1)/block_size; //for time being waste many blocks
  //const int nblocks = side;
  const int nblocks = length - (i + 2*(turn+1));
  
//  printf("launch modular_decomposition_kernel<<<%d,%d>>>(side=%d,i=%d,%d,%d... threads %d todo %d\n",
//	 nblocks,block_size,
//	 side,i,turn,length,
//	 nblocks*block_size,todo);

  dim3 blocks(nblocks,nfiles);
  modular_decomposition_kernel<<<blocks,block_size,0,graph_stream>>>(i, turn, length,
					   d_fml_i, d_fml_j,
					   d_dml); //Out

  gpuErrchk( cudaPeekAtLastError() );

  //for effiency transfer all of DMLi rather that just those part that have been calculated
  int_MemcpyAsync(DMLi,d_dml, (size_t)nfiles*(length+1), cudaMemcpyDeviceToHost, graph_stream, __LINE__);
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
			int* DMLi) {       //Out
  //const int max_ij_len = length - 2*(turn+1) - 1;
  //const int ijsize_min = max_ij_len*(max_ij_len+1)/2;
  //if(first) init_gpu(length);
  modular_decomposition_cuda(nfiles,i,turn,length, DMLi);
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
					     int* DMLi) {           //out
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
    load_fML(nfiles,i,turn,length,energy_min);
    modular_decomposition_i(nfiles,i,turn,length,DMLi);
    load_min_fML(nfiles,i,turn,length);
    gpuErrchk( cudaStreamSynchronize(graph_stream) );
    return;
  }

  cudaGraph_t graph = NULL;
  gpuErrchk( cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeThreadLocal) );

  load_fML(nfiles,i,turn,length,energy_min);
  modular_decomposition_i(nfiles,i,turn,length,DMLi);
  load_min_fML(nfiles,i,turn,length);

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
  gpuErrchk( cudaStreamSynchronize(graph_stream) );
}

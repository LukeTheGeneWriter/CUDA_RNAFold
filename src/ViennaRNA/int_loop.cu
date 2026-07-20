#define Version "$Revision: 1.110 $ "
//WBL 11 Jan 2018 CUDA GGGP ViennaRNA-2.3.0 rf/rf_cuda2
//Helper for fill_arrays.c 
//based on ViennaRNA-2.3.0/src/ViennaRNA/interior_loops.c (Nov  1  2016) 

//WBL 17 Feb 2018 clean for production (cf r1.75), remove tick
//    keep source code of small unused kernels for the timebeing but remove calling them.
//WBL 11 Feb 2018 use own timing rather than nvidia profiling tools
//WBL  6 Feb 2018 split interior_loopx.h into separate non-divergent kernels
//WBL 28 Jan 2018 process nfiles in one go

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include "ViennaRNA/fold_vars.h"
#include "ViennaRNA/utils.h"
#include "ViennaRNA/constraints.h"
#include "ViennaRNA/exterior_loops.h"
#include "ViennaRNA/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/unstructured_domains.h"
#include "ViennaRNA/interior_loops.h"

//use GPU primitives in CUDA code
#undef MIN2
#define MIN2(x,y) min(x,y)
#undef MAX2
#define MAX2(x,y) max(x,y)
//ViennaRNA/energy_par.c
#define MAX_NINIO 300
#include           "interior_loopx.h"

#include "stub2.h"
#include <assert.h>

//Avoiding passing turn as a kernel parameter make only a tiny saving
//allow GPU compile to optimise
//ViennaRNA/model.h  min_loop_size == TURN

#define turn 3

/*Unused to host C code to check answers given by GPU code
PRIVATE int
E_int_loop( const vrna_fold_compound_t *vc,
            const int i,
            const int j);
*/
/********************************************************************
Begin CUDA code
********************************************************************/

int first2 = 1; //avoid id clash with modular_decomposition.cu

//cf ViennaRNA-2.3.0/src/ViennaRNA/params.h
//ensure contents follow multiple of 128 bytes
typedef struct  cuda_param_s       cuda_param_t; //only fields in vrna_param_t that are read
struct cuda_param_s {
//int     id;
  int     stack[NBPAIRS+1][NBPAIRS+1];
  int     bulge[MAXLOOP+1];
  int     ninio2; //ninio[5];
  int     internal_loop[MAXLOOP+1];
  int     TerminalAU;
  float   lxc; /*double*/
  int     pad1[31];
//int     hairpin[31];
//int     mismatchExt[NBPAIRS+1][5][5];
  int     mismatchI[NBPAIRS+1][5][5];
  int     pad2[24];
  int     mismatch1nI[NBPAIRS+1][5][5];
  int     pad3[24];
  int     mismatch23I[NBPAIRS+1][5][5];
  int     pad4[24];
//int     mismatchH[NBPAIRS+1][5][5];
//int     mismatchM[NBPAIRS+1][5][5];
//int     dangle5[NBPAIRS+1][5];
//int     dangle3[NBPAIRS+1][5];
  int     int11[NBPAIRS+1][NBPAIRS+1][5][5];
  int     int21[NBPAIRS+1][NBPAIRS+1][5][5][5];
  int     int22[NBPAIRS+1][NBPAIRS+1][5][5][5][5];
//int     MLbase;
//int     MLintern[NBPAIRS+1];
//int     MLclosing;
//int     DuplexInit;
//int     Tetraloop_E[200];
//char    Tetraloops[1401];
//int     Triloop_E[40];
//char    Triloops[241];
//int     Hexaloop_E[40];
//char    Hexaloops[1801];
//int     TripleC;
//int     MultipleCA;
//int     MultipleCB;
//int     gquad [VRNA_GQUAD_MAX_STACK_SIZE + 1]
//              [3*VRNA_GQUAD_MAX_LINKER_LENGTH + 1];
//
//double  temperature;            /**<  @brief  Temperature used for loop contribution scaling */
//
//vrna_md_t model_details;   /**<  @brief  Model details to be used in the recursions */
};

cuda_param_t* d_param;
char*         d_pair; //[NBPAIRS+1][NBPAIRS+1];
unsigned int* d_hccc; //read via Hc
short*        d_S;    //S[length+2]
int*          d_my_c;
int*          d_energy_min2; //share with modular_decomposition.cu ?
int*          d_new_e;
//no longer in use
//int*        d_energy_min20; //alternative calculation of d_energy_min2
//int*        d_buf;  //intermediate energy result GPU only

#define BLOCK_SIZE 512

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk2(ans,first) { gpuAssert((ans), __FILE__, __LINE__,first); }
inline void gpuAssert(cudaError_t code, const char *file, const int line, const bool first=false, const bool abort=true)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA error: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
     if(first) fprintf(stderr,"CUDA error: on first kernel.\n", file);
     if (abort) exit(code);
   }
}

#define Assert(ans) { Assert_((ans), __FILE__, __LINE__); }
inline void Assert_(bool test, const char *file, const int line) {
  if(test) return;
  fprintf(stderr,"Assert failed %s %d\n", file, line);
  exit(1);
}

/* prefill matrices with init contributions */
__global__ void
init_my_c_kernel(const size_t ijsize, // 32-bit signed integer overflow bug fix
		 int* __restrict__ my_c) {
  const size_t m = blockIdx.x*blockDim.x+threadIdx.x; // 32-bit signed integer overflow bug fix
  if(m>=ijsize) return;
  my_c[m] = INF;
}

PUBLIC void
init_my_c(const size_t ijsize) { // 32-bit signed integer overflow bug fix
  /* Setup execution parameters for helper kernel */
  const size_t nblocks = (ijsize + BLOCK_SIZE - 1)/BLOCK_SIZE; // 32-bit signed integer overflow bug fix
  init_my_c_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_my_c);
  gpuErrchk2( cudaPeekAtLastError(),  first2 );
#ifndef NDEBUG
  gpuErrchk2( cudaDeviceSynchronize(),first2 );
  //may pickup errors later if dont sync now
#endif
}

PUBLIC void
sanity(const vrna_fold_compound_t* vc0, const vrna_fold_compound_t* vc) {
  //check when processing multiple files they have the same parameters
  //in principle could adapt code to cope with differences but not done yet
  //Initially use Assert to make sure compiler does not optimise away checks
  Assert(vc0->length == vc->length);
  //params assumed to be ok since all loaded from same .par file but some checks anyway
  const vrna_param_t* P0 = vc0->params;
  const vrna_param_t* P  =  vc->params;
  Assert(P0->MLbase == P->MLbase);
  const vrna_md_t *md0 = &(vc0->params->model_details);
  const vrna_md_t *md  = &( vc->params->model_details);
  Assert(memcmp(md0->pair,md->pair,21*21*sizeof(int))==0);

  Assert(md0->noGUclosure   == md->noGUclosure);
  Assert(md0->noLP          == md->noLP);
  Assert(md0->uniq_ML       == md->uniq_ML);
  Assert(md0->dangles       == md->dangles);
  Assert(md0->min_loop_size == md->min_loop_size);
  Assert(md->min_loop_size  == turn);

  Assert(P0->TerminalAU == P->TerminalAU);
  Assert(P0->ninio[2]   == P->ninio[2]);
  Assert(P0->lxc        == P->lxc);
}


void load_param(const vrna_param_t *P){
  //Even though only used once make copy on host as probably easier to debug than many cudaMemcpy
  cuda_param_t* H = (cuda_param_t*) malloc(sizeof(cuda_param_s));

  memcpy(H->stack,        P->stack,        (NBPAIRS+1)*(NBPAIRS+1)*sizeof(int));
  H->ninio2     =         P->ninio[2];
  H->lxc        =  (float)P->lxc;
  H->TerminalAU =         P->TerminalAU;
  memcpy(H->bulge,        P->bulge,        (MAXLOOP+1)*sizeof(int));
  memcpy(H->internal_loop,P->internal_loop,(MAXLOOP+1)*sizeof(int));
  memcpy(H->mismatchI,    P->mismatchI,    (NBPAIRS+1)*5*5*sizeof(int));
  memcpy(H->mismatch1nI,  P->mismatch1nI,  (NBPAIRS+1)*5*5*sizeof(int));
  memcpy(H->mismatch23I,  P->mismatch23I,  (NBPAIRS+1)*5*5*sizeof(int));
  memcpy(H->int11,        P->int11,        (NBPAIRS+1)*(NBPAIRS+1)*5*5*sizeof(int));
  memcpy(H->int21,        P->int21,        (NBPAIRS+1)*(NBPAIRS+1)*5*5*5*sizeof(int));
  memcpy(H->int22,        P->int22,        (NBPAIRS+1)*(NBPAIRS+1)*5*5*5*5*sizeof(int));

  const cudaError_t error = cudaMemcpy(d_param,H,sizeof(cuda_param_s),cudaMemcpyHostToDevice);
  if (error != cudaSuccess)  {
    printf("cudaMemcpy(d_param,H,%lu,cudaMemcpyHostToDevice) returned error %s (code %d), %s line(%d)\n", 
	   sizeof(cuda_param_s), cudaGetErrorString(error), error, __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

  free(H);
}

#define bitsperint (8*sizeof(unsigned int))

//make hccc oversized to simplify bounds checks in nthsetindex
#define Hc_ints(length) (((length*(length+1))/2+2 + (MAXLOOP+1)*(MAXLOOP+2)/2 + bitsperint - 1)/bitsperint)

PUBLIC void
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size) {
  if(!first2) return;
  fprintf(stderr,"%-24s init_gpu2(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

  assert(turn_ == turn);
  assert(MAX_NINIO == 300); //ViennaRNA/energy_par.c
  //printf("%s %s d_param is %lu bytes, NBPAIRS %d MAXLOOP %d BLOCK_SIZE %d\n",
  //	 __FILE__,Version,sizeof(cuda_param_s),NBPAIRS,MAXLOOP,block_size);

  // d_param/d_pair are nfiles/length-independent -- guarded on their own
  // one-time check (d_param starts NULL, a zero-initialized global) rather
  // than on first2, so teardown_gpu2() can reset first2=1 between GPU
  // batches without this block re-allocating (and leaking) them every batch.
  if(!d_param) {
    gpuErrchk( cudaMalloc((void **) &d_param, sizeof(cuda_param_s)) );
    load_param(VC[0]->params);

    char pair_[NBPAIRS+1][NBPAIRS+1];
    for(int x=0;x<21;x++){
    for(int y=0;y<21;y++) {
      const vrna_md_t *md = &(VC[0]->params->model_details);
      if(x < NBPAIRS+1 && y < NBPAIRS+1) {
        pair_[x][y] = md->pair[x][y];
        assert(pair_[x][y] >= 0 && pair_[x][y] < 8);
      }
      else assert(md->pair[x][y]==0);
    }}
    const size_t pair_size = (NBPAIRS+1)*(NBPAIRS+1)*sizeof(char); // 32-bit signed integer overflow bug fix
    gpuErrchk( cudaMalloc((void **) &d_pair, pair_size) );
    gpuErrchk( cudaMemcpy(d_pair,pair_,pair_size,cudaMemcpyHostToDevice) );
  }

  size_t size = (size_t)nfiles*Hc_ints(length)*sizeof(unsigned int); // 32-bit signed integer overflow bug fix
  gpuErrchk( cudaMalloc((void **) &d_hccc, size) );
  unsigned int* hccc   = (unsigned int*) calloc((size_t)nfiles*Hc_ints(length),sizeof(unsigned int)); // 32-bit signed integer overflow bug fix
  for(int H=0;H<nfiles;H++){
    assert(bitsperint==(1+0x1f));
    unsigned int mask;
    for(int i=0;i<(length*(length+1))/2+2;i++){ //leave padding as zero
      mask = ((i & 0x1f) == 0)? 1 : mask << 1;
      const int I = H*Hc_ints(length)+i/bitsperint;
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC) hccc[I] |= mask;
    }
  }
  gpuErrchk( cudaMemcpy(d_hccc,hccc,(size_t)nfiles*Hc_ints(length)*sizeof(unsigned int),cudaMemcpyHostToDevice) ); // 32-bit signed integer overflow bug fix
  free(hccc);

  size = (size_t)nfiles*(length+2)*sizeof(short); // 32-bit signed integer overflow bug fix
  gpuErrchk( cudaMalloc((void **) &d_S, size) );
  short* buff = (short*) malloc(size); //could use cudaMallocHost
  for(int H=0;H<nfiles;H++){
    const int len = (length+2);
    memcpy(&buff[H*len],VC[H]->sequence_encoding,len*sizeof(short));
  }
  gpuErrchk( cudaMemcpy(d_S,buff,size,cudaMemcpyHostToDevice) );
  free(buff);

  size = (size_t)nfiles*((length+1)*(length+2)/2)*sizeof(int); // 32-bit signed integer overflow bug fix
  gpuErrchk( cudaMalloc((void **) &d_my_c, size) );
  init_my_c((size_t)nfiles*(length+1)*(length+2)/2); // 32-bit signed integer overflow bug fix

  size = (size_t)nfiles*(length+1)*sizeof(int); // 32-bit signed integer overflow bug fix
  gpuErrchk( cudaMalloc((void **) &d_new_e, size) );

  gpuErrchk( cudaMalloc((void **) &d_energy_min2, size) );
  /*no longer in use
  gpuErrchk( cudaMalloc((void **) &d_energy_min20,size) );

  size = nfiles*length*sizeof(int);
  gpuErrchk( cudaMalloc((void **) &d_buf, size) );
  */
  first2 = 0;
}

// Frees the 5 nfiles/length-scaled device buffers allocated by init_gpu2()
// and resets first2 so the next init_gpu2() call re-runs at a new batch's
// nfiles. d_param/d_pair are deliberately left allocated -- they're
// nfiles/length-independent (see the one-time guard in init_gpu2() above)
// and never need resizing between batches.
PUBLIC void
teardown_gpu2(void) {
  if(first2) return; // never initialized (or already torn down) -- nothing to free
  gpuErrchk( cudaFree(d_hccc) );
  gpuErrchk( cudaFree(d_S) );
  gpuErrchk( cudaFree(d_my_c) );
  gpuErrchk( cudaFree(d_new_e) );
  gpuErrchk( cudaFree(d_energy_min2) );
  first2 = 1;
}

// Bytes of device memory this file needs for one additional sequence at the
// given length -- d_hccc/d_S/d_my_c/d_new_e/d_energy_min2, mirroring
// init_gpu2()'s own size formulas exactly. d_param/d_pair are excluded --
// fixed-size, paid once regardless of batch count.
PUBLIC size_t
int_loop_bytes_per_file(const int length) {
  const size_t hccc_bytes         = Hc_ints(length)*sizeof(unsigned int);
  const size_t s_bytes            = (size_t)(length+2)*sizeof(short);
  const size_t my_c_bytes         = (size_t)(length+1)*(length+2)/2*sizeof(int);
  const size_t new_e_bytes        = (size_t)(length+1)*sizeof(int);
  const size_t energy_min2_bytes  = (size_t)(length+1)*sizeof(int);
  return hccc_bytes + s_bytes + my_c_bytes + new_e_bytes + energy_min2_bytes;
}

//perhaps this can be combined with other kernels?
__global__ void
load_my_c_kernel(const int i, /*const int turn,*/ const int length,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c) { //out
  const size_t H = blockIdx.y; // Langdon's 2026 indexing bug
  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int j = m + i+turn+1;
  if(j>length) return;

  const int ij = j*(j-1)/2+i;
  const size_t ijsize = (size_t)(length+1)*(length+2)/2; // Langdon's 2026 indexing bug
  assert(ij>=0 && (size_t)ij<ijsize);
  assert(my_c[H*ijsize+ij] == INF); // Langdon's 2026 indexing bug
         my_c[H*ijsize+ij] = new_e[H*(length+1)+j]; // Langdon's 2026 indexing bug
}

PUBLIC void
load_my_c(const int nfiles,
	  const int i, const int turn_, const int length,
	  const int* new_e) {   //in
  //out d_my_c
  const int start = i+turn+1; 
  const int size  = length - start + 1;
  if(size<=0) return;

#ifdef NDEBUG
  //check here in case of earlier errors
  gpuErrchk( cudaDeviceSynchronize() );
#endif
  //for simplicity transfer all new_e, even though only need H * [start:length]
  gpuErrchk( cudaMemcpy(d_new_e,new_e,nfiles*(length+1)*sizeof(int),cudaMemcpyHostToDevice) );


  /* Setup execution parameters for helper kernel */
  const int nblocks = (size + BLOCK_SIZE - 1)/BLOCK_SIZE;

  dim3 blocks(nblocks,nfiles);

  load_my_c_kernel<<<blocks,BLOCK_SIZE>>>(i, /*turn,*/ length,
					   d_new_e,  //in
					   d_my_c); //out
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

//#include "ptype.cu"
//Was
//WBL 13 Jan 2018 From ViennaRNA-2.3.0/src/ViennaRNA/alphabet.c Revision: 1.9

//Modification:
//WBL 14 Jan 2018 just single element of ptype[ij]

//Based on ViennaRNA-2.3.0/src/ViennaRNA/utils.c
//replace ptypes array
__device__ inline unsigned char
Ptype(const short* __restrict__ S, const char* __restrict__ pair,//[8][8],
      const int i, const int j) {

  //assert(i>=0 && i<=length);
  //assert(j>=0 && j<=length);
  assert(S[i]>=0 && S[i]<8);
  assert(S[j]>=0 && S[j]<8);

  return pair[S[i]*8 + S[j]];

  //assert(ptype>=0 && ptype<8);
  /*
  printf("my_ptype(S,md,%d,%d, %d,%d) ptype %d S[%d]%d S[%d]%d ",
	 i,j,real_ptype,length,ptype,i,S[i], j,S[j]);
  if(counter==0) {
  for(int l=0;l<8;l++) {
    printf("\n");
    for(int k=0;k<8;k++) {
      printf("pair[%d][%d]%d ",l,k,md->pair[l][k]);
    }
  }}
  printf("\n");
  if(counter++>1000) exit(1);
  */
}
//end_include "ptype.cu"

//since q traditionally counts down this is smallest value
__device__ inline int 
Min_q(const int i, const int j, const int turn_) { //max_q
  return MAX2(i+turn+2, j - MAXLOOP - 1);
}
__device__ inline int 
Max_p(const int i, const int j, const int q, 
		 const int turn_/*, const int hc_top*/) {
  const int j_q = j - q - 1;
  int max_p = i + 1;
  int tmp   = i + 1 + MAXLOOP - j_q;
  max_p     = MAX2(max_p, tmp);
  tmp       = q - turn;
  max_p     = MIN2(max_p, tmp);
//tmp       = i + 1 + hc_top; //makes no difference
  return MIN2(max_p, tmp);
}

__device__ inline int
Indx(const int i, const int j) { 
  return j*(j-1)/2+i;
}

#undef BLOCK_SIZE
//Like modular_decomposition.cu have one block per j value
//each block has (MAXLOOP+1)*(MAXLOOP+2)/2 worker threads
//present reduction code needs BLOCK_SIZE to be at least 32 and a power of 2
#define BLOCK_SIZE 32

//emulate hc[pq] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC;
__device__ inline
int Hc(const int ij, const unsigned int* __restrict__ hccc){
  const int I = ij/bitsperint;
  const unsigned int m = hccc[I];
  //const int shift = (ij - I*bitsperint);
  const int ans = (m >> (ij - I*bitsperint)) & 1;
  /*
  if(blockIdx.x==4 && threadIdx.x==15) 
    printf("%d,%d Hc(%d,hccc) I %d m %08x shift %d ans %d\n",
	   blockIdx.x,threadIdx.x, ij, I,m,shift,ans);
  */
  return ans;
}

//setpq to minimise number of threads prevented by hccc from doing anything
//but then threads tend to take different paths through Energy() leading to
//divergence. ALternative small kernels minimise thread divergence but tend
//to each have too little work to be efficient. 
#include "nth.h"
//output the location of the rankth set bit in hccc
//return false if no such bit (inside mask_size)
__device__ inline
int setpq(const int i,
	  const int j,
	  const int maxcol,
	  const unsigned int* __restrict__ hccc,
	  const int work,  //0...31..511 assumed to step forward
	  unsigned int& mask, //search context
	  int& row_start,  //search context
	  int& done,       //search context
	  int& column,     //output
	  int& row) {      //output
  assert(bitsperint==32);
  assert(maxcol<=MAXLOOP);
do {
  const int rank = work - done;
  if(row_start > 0 && mask == 0) {
    column++;
    row_start = 0;
  }
  if(column > maxcol) {
    return false;
  }
  const int mask_size = column + 1;
  assert(mask_size>0);
  assert(mask_size<=bitsperint+1);
  if(row_start == 0) {
    const int pq = Indx(i,j+column);
    const int I =     pq/bitsperint;
    const int x = pq - I*bitsperint;
    mask = hccc[I];
    mask = mask >> x; //remove bits below pq
    if(mask_size+x > 32 ) {//get top bits
      unsigned int m2 = hccc[I+1];
      m2 = m2 & (~((~0) << (mask_size+x-32))); //clear bits above mask_size
      m2 = m2 << (32-x);                       //avoid over writing lower bits in mask already in use
      mask = mask | m2;                        //splice two parts of column mask together
    } else {
      mask = mask & (~((~0)<< mask_size));     //clear bits above mask_size
    }
  }//endif read new column mask

  int popc;
  row = find_nth_set_bit(mask,rank,popc);
  if(row>=0) {
    row_start = row + 1;
    done += 1+rank;
    mask = mask & (~((1 << row_start) - 1));//clear self and bits below row
    return true;
  }//else did not find, try next column
  assert(popc <= mask_size);
  assert(mask_size<32); assert(mask < (1 << mask_size));
  done += popc;
  column++;
  row_start = 0;
 } while (true);
}

//interface to interior_loopx.h via IntLoop_X()
__device__ inline int
Energy(const int i, const int j, const int q, const int p, 
	  /*const char* hard_constraints,*/ const int* my_c,
	  /*const int* hc_up, const char* hc, const unsigned int* __restrict__ hccc,*/
	  const short* __restrict__ S, const char* __restrict__ pair_,//[NBPAIRS+1][NBPAIRS+1],
	  const cuda_param_t __restrict__ *P,
	  //const int n1,
          //const int ns,
          //const int nl,
          //const int type,
          //const int type_2,
          //const int si1,
          //const int sj1,
          //const int sp1,
          //const int sq1,
	  //Remainder are in const vrna_param_t *P,
	  //approx in order of how much gcov says they are used
	  const int TerminalAU,
	  const int ninio2,
	  const int bulge[MAXLOOP+1],
	  const int internal_loop[MAXLOOP+1],
	  const float lxc,
	  const int mismatchI[NBPAIRS+1][5][5],
	  const int mismatch1nI[NBPAIRS+1][5][5],
	  const int mismatch23I[NBPAIRS+1][5][5],
	  //gcov says p->stack,P->int11,P->int21,P->int22 seldom used
	  const int stack[NBPAIRS+1][NBPAIRS+1],
	  const int int11[NBPAIRS+1][NBPAIRS+1][5][5],
	  const int int21[NBPAIRS+1][NBPAIRS+1][5][5][5],
	  const int int22[NBPAIRS+1][NBPAIRS+1][5][5][5][5]){

  //const int j_q = j - q - 1;
  //assert(q+1<length+2);
  //this should not be needed as using Hc if(hc_up[q+1] < j_q) return INF;

  int energy = INF;
	  const int pp = p -(i+1);


	  const int pq = Indx(p,q);
	  assert(pp == p-(i+1));
	  //assert(pq+pp == pq);
	  //assert(pq > 0 && pq < (length*(length+1))/2+2);
	  /*now using setpq() so this test should be redundant
	  const char eval_loop = Hc(pq,hccc);
	  if(!eval_loop) {
	    printf("%d,%d Energy(%d,%d,%d,%d...) pq %d fails\n",
		   blockIdx.x,threadIdx.x,
		   i,j,q,p, pq);
	  }
	  assert(eval_loop);
	  if(eval_loop)*/{
	    energy = my_c[pq];
	    if(energy != INF){
	      //assert(ptype[pq]>=0 && ptype[pq]<8);
	      //const unsigned char type_2 = rtype[(unsigned char)ptype[pq]];
	      const unsigned char type   = Ptype(S,pair_,i,j);
	      //assert(type == Ptype(S,pair_,i,j));
	      assert(type<8);
	      const unsigned char type_2 = Ptype(S,pair_,q,p);
	      assert(type_2<8);
	      //assert(i+pp  >=0 && i+pp  <length+2);

	      const int u1 = p - 1 - i; //u1 = p1 - i;
	      const int u2 = j - 1 - q; //u2 = j1 - q;

	      const int ns = (u1>u2)? u2 : u1;
	      const int nl = (u1>u2)? u1 : u2;

	      energy += IntLoop_X(u1, ns, nl, type, type_2, 
				  S[i+1], S[j-1], S[i+pp], S[q + 1],
				  TerminalAU,ninio2,
				  P->bulge,P->internal_loop,lxc,
				  mismatchI,
				  mismatch1nI,
				  mismatch23I,
				  P->stack,
				  P->int11,
				  P->int21,
				  P->int22);
	      //if(i==2000) printf("\n");
	    }//endif c[pq+] != INF
	  }//endif hc[pq+] & ...
	  return energy;
}

//Series of small kernels no longer in use, each dedicate to a path through IntLoop_X() to avoid divergence
//based on interior_loopx.h r1.16
__global__ void
int_loop_nl0_kernel(const int nfiles,
		    const int i, /*const int turn,*/ const int length,
		    const int /*__restrict__*/ stack[NBPAIRS+1][NBPAIRS+1],
		    const char* __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		    const short* __restrict__ SH, //[H,length+2]
		    const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		          int* __restrict__ nl0) {  //out
  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  if(Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure nl==0
    const int p = i+1;
    const int q = j-1;
    const int pq = Indx(p,q);
    if(Hc(pq,Hccc)) {
      //do energy = my_c[pq] if(energy != INF) and energy += in xxx_kernel

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8);

      assert(p - 1 - i == 0 && j - 1 - q == 0); //if (nl == 0) {
      //if(v)printf("n1==0 stack[%d][%d]",type,type_2);
      energy = stack[type][type_2];  /* stack */
    }
  }
  nl0[m] = energy;
}		     

//based on interior_loopx.h r1.16
__global__ void
int_loop_ns0_kernel(const int i, /*const int turn,*/ const int length,
		    const int TerminalAU,
		    const int bulge[MAXLOOP+1],
		    const int  /*__restrict__*/ stack[NBPAIRS+1][NBPAIRS+1],
		    const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		    const short* __restrict__ SH, //[H,length+2]
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		    const int*   __restrict__ my_c,
		            int* __restrict__ ns0) {  //out
  assert(blockDim.x > 2*(MAXLOOP+1));
  //using up to 2*MAXLOOP+1 threads so BLOCK_SIZE=64
  //might as well use Grid to determine H
  const size_t H = blockIdx.y; // Langdon's 2026 indexing bug -- widening H to
  //size_t promotes every H*(...) product below to size_t arithmetic
  //automatically, per C's usual arithmetic conversions
  const int j = blockIdx.x + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  const int nl = (threadIdx.x & 0x1f);
  if(nl>0 && nl<=MAXLOOP && // nl==0 done elsewhere
     Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure ns==0,                  u2 == 0 : u1 == 0
    const int p = (threadIdx.x<32)? i+1 + nl : i+1;
    const int q = (threadIdx.x<32)? j-1      : j-1 - nl;

    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    /*
    if(q < min_q){
      printf("%d,%d int_loop_ns0_kernel(%d,%d,%d,...) H %d j %d q %d p fails min_q %d\n",
	     blockIdx.x,threadIdx.x,
	     i,turn,length,H,j,q,p,min_q);
    }
    if(p > max_p){
      printf("%d,%d int_loop_ns0_kernel(%d,%d,%d,...) H %d j %d q %d p fails max_p %d\n",
	     blockIdx.x,threadIdx.x,
	     i,turn,length,H,j,q,p,max_p);
    }
    */
    if(q >= min_q && q<=length && p <= max_p) {
      const int pq = Indx(p,q);
      //printf("%d.%d,%d int_loop_ns0_kernel(%d,%d,%d,...) H %d j %d q %d p %d pq %d\n",
      //     blockIdx.x,blockIdx.y,threadIdx.x,
      //     i,turn,length,H,j,q,p,pq);
      assert(q >= min_q);
      assert(p <= max_p);
      assert(q <= length);
      assert(p <= length);
      assert(nl >  0);
      assert(nl <= MAXLOOP);
      if(Hc(pq,Hccc)) {
      const size_t hpq = pq + H*((length+1)*(length+2)/2); // Langdon's 2026 indexing bug -- hpq itself must be size_t too, not just H, or the correctly-widened RHS truncates right back on assignment
      energy = my_c[hpq];
      if(energy != INF) {

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8); 

      assert( p - 1 - i == 0 || j - 1 - q == 0); //if (ns == 0)
      assert((p - 1 - i == 0 && j - 1 - q <= MAXLOOP) || //nl<=MAXLOOP)
	     (p - 1 - i <= MAXLOOP && j - 1 - q == 0));

      energy += bulge[nl];
      if (nl==1) {
      //if(v)printf("nl==1 stack[%d][%d] ",type,type_2);
      //if(v)printf("ns==0 bulge[%d] %d %d ",nl,type,type_2);
      energy += stack[type][type_2];
      }
      else {
      //if(v)printf("ns==0 bulge[%d] %d %d ",nl,type,type_2);
      //if(v)printf("nl!=1 %d %d TerminalAU?? ",type,type_2);
	if (type>2) energy += TerminalAU;
	if (type_2>2) energy += TerminalAU;
      }
    }}}
  }
  volatile __shared__ int en[64];
  en[threadIdx.x] = energy; //must set whole of en

#define ix threadIdx.x
  if(ix < 32) {
    __syncthreads();            en[ix] = MIN2(en[ix], en[ix+ 32]);
    en[ix] = MIN2(en[ix], en[ix+ 16]);
    en[ix] = MIN2(en[ix], en[ix+  8]);
    en[ix] = MIN2(en[ix], en[ix+  4]);
    en[ix] = MIN2(en[ix], en[ix+  2]);
    en[ix] = MIN2(en[ix], en[ix+  1]);
  }
#undef ix
  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(threadIdx.x==0) ns0[blockId] = en[0];
}		     

//based on interior_loopx.h r1.16
__global__ void
int_loop_1xn_kernel(const int i, /*const int turn,*/ const int length,
		    const int ninio2,
		    const int internal_loop[MAXLOOP+1],
		    const int mismatch1nI[NBPAIRS+1][5][5],
		    const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		    const short* __restrict__ SH, //[H,length+2]
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		    const int*   __restrict__ my_c,
		            int* __restrict__ out_1xn) {
  assert(blockDim.x > 2*(MAXLOOP+1));
  //using up to 2*(MAXLOOP-2) threads so BLOCK_SIZE=64
  //might as well use Grid to determine H
  const size_t H = blockIdx.y; // Langdon's 2026 indexing bug -- widening H to
  //size_t promotes every H*(...) product below to size_t arithmetic
  //automatically, per C's usual arithmetic conversions
  const int j = blockIdx.x + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  const int nl = (threadIdx.x & 0x1f);
  if(nl>2 && nl<=MAXLOOP && // nl==0,1,2, done elsewhere
     Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure ns==1,                  u2 == 1 : u1 == 1
    const int p = (threadIdx.x<32)? i+1 + nl : i+1 + 1;
    const int q = (threadIdx.x<32)? j-1-1    : j-1 - nl;

    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p) {
      const int pq = Indx(p,q);
      assert(q >= min_q);
      assert(p <= max_p);
      assert(q <= length);
      assert(p <= length);
      assert(nl >  0);
      assert(nl <= MAXLOOP);
      if(Hc(pq,Hccc)) {
      const size_t hpq = pq + H*((length+1)*(length+2)/2); // Langdon's 2026 indexing bug -- hpq itself must be size_t too, not just H, or the correctly-widened RHS truncates right back on assignment
      energy = my_c[hpq];
      if(energy != INF) {

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8); 

      assert( p - 1 - i == 1 || j - 1 - q == 1); //if (ns == 1)
      assert((p - 1 - i == 1           && j - 1 - q + 1 <= MAXLOOP) || //nl+1<=MAXLOOP
	     (p - 1 - i + 1 <= MAXLOOP && j - 1 - q == 1));

      const int si1 = S[i+1];
      const int sj1 = S[j-1];
      const int sq1 = S[q+1];
      const int sp1 = S[p-1];
      const int ns  = 1;
      /* 1xn loop */
      energy+= internal_loop[nl+1] +
               MIN2(MAX_NINIO, (nl-ns)*ninio2) +
               mismatch1nI[type][si1][sj1] + mismatch1nI[type_2][sq1][sp1];
      }
    }}
  }
  volatile __shared__ int en[64];
  en[threadIdx.x] = energy; //must set whole of en

#define ix threadIdx.x
  if(ix < 32) {
    __syncthreads();            en[ix] = MIN2(en[ix], en[ix+ 32]);
    en[ix] = MIN2(en[ix], en[ix+ 16]);
    en[ix] = MIN2(en[ix], en[ix+  8]);
    en[ix] = MIN2(en[ix], en[ix+  4]);
    en[ix] = MIN2(en[ix], en[ix+  2]);
    en[ix] = MIN2(en[ix], en[ix+  1]);
  }
#undef ix
  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(threadIdx.x==0) out_1xn[blockId] = en[0];
}		     
//based on interior_loopx.h r1.16
__global__ void
int_loop_int11_kernel(const int nfiles,
		      const int i, /*const int turn,*/ const int length,
		      const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		      const short* __restrict__ SH, //[H,length+2]
		      const int int11[NBPAIRS+1][NBPAIRS+1][5][5],
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		            int* __restrict__ int11_out) {

  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  if(Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure ns==1 and nl==1
    const int p = i+1+1;
    const int q = j-1-1;
    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p) {
    const int pq = Indx(p,q);
    assert(q >= min_q);
    assert(p <= max_p);
    assert(q <= length);
    assert(p <= length);
    if(Hc(pq,Hccc)) {
      //do energy = my_c[pq] if(energy != INF) and energy += in xxx_kernel

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8);

      assert(p - 1 - i == 1); //if (ns==1) and (nl==1)
      assert(j - 1 - q == 1);

      const int si1 = S[i+1];
      const int sj1 = S[j-1];

      energy = int11[type][type_2][si1][sj1];
    }}
  }
  int11_out[m] = energy;
}		     

//based on interior_loopx.h r1.16
__global__ void
int_loop_int21_kernel(const int n1, //n1==1 or n1!=1
		      const int nfiles,
		      const int i, /*const int turn,*/ const int length,
		      const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		      const short* __restrict__ SH, //[H,length+2]
		      const int int21[NBPAIRS+1][NBPAIRS+1][5][5][5],
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		            int* __restrict__ int21_out) {

  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  if(Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure ns==1 and nl==2 (n1 == 1 given by input)
    const int p = (n1==1)? i+1+1 : i+1+2;
    const int q = (n1==1)? j-1-2 : j-1-1;
    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p) {
    const int pq = Indx(p,q);
    assert(q >= min_q);
    assert(p <= max_p);
    assert(q <= length);
    assert(p <= length);
    if(Hc(pq,Hccc)) {
      //do energy = my_c[pq] if(energy != INF) and energy += in xxx_kernel

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8);

      if(n1==1) {//if (ns==1) and (nl==2)
	assert(p - 1 - i == 1); 
	assert(j - 1 - q == 2);
      } else {
	assert(p - 1 - i == 2);
	assert(j - 1 - q == 1);
      }
      #define si1 S[i+1]
      #define sj1 S[j-1]
      #define sq1 S[q+1]
      #define sp1 S[p-1]

      energy = (n1==1)? int21[type][type_2][si1][sq1][sj1] : //n1==1
                        int21[type_2][type][sq1][si1][sp1];  //n1!=1
      #undef si1
      #undef sj1
      #undef sq1
      #undef sp1
    }}
  }
  int21_out[m] = energy;
}		     

//based on interior_loopx.h r1.16
__global__ void
int_loop_int22_kernel(const int nfiles, /* 2x2 loop */
		      const int i, /*const int turn,*/ const int length,
		      const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		      const short* __restrict__ SH, //[H,length+2]
		      const int int22[NBPAIRS+1][NBPAIRS+1][5][5][5][5],
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		            int* __restrict__ int22_out) {

  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  if(Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure ns==2 and nl==2
    const int p = i+1+2;
    const int q = j-1-2;
    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p) {
    const int pq = Indx(p,q);
    assert(q >= min_q);
    assert(p <= max_p);
    assert(q <= length);
    assert(p <= length);
    if(Hc(pq,Hccc)) {
      //do energy = my_c[pq] if(energy != INF) and energy += in xxx_kernel

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8);

      assert(p - 1 - i == 2); //if (ns==2) and (nl==2)
      assert(j - 1 - q == 2);

      const int si1 = S[i+1];
      const int sj1 = S[j-1];
      const int sq1 = S[q+1];
      const int sp1 = S[p-1];

      energy = int22[type][type_2][si1][sp1][sq1][sj1]; /* 2x2 loop */
    }}
  }
  int22_out[m] = energy;
}		     

//based on interior_loopx.h r1.16
__global__ void
int_loop_nl3_kernel(const int n1, //n1==2 or n1==3 flag
		    const int nfiles,
		    const int i, /*const int turn,*/ const int length,
		    const int ninio2,
		    const int internal_loop5,
		    const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		    const short* __restrict__ SH, //[H,length+2]
		    const int mismatch23I[NBPAIRS+1][5][5],
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		            int* __restrict__ nl3_out) {

  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];
  if(Hc(ij,Hccc)) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    //ensure ns==2 and nl==3 (n1 == 3 given by input)
    const int p = (n1==3)? i+1+3 : i+1+2;
    const int q = (n1==3)? j-1-2 : j-1-3;
    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p) {
    const int pq = Indx(p,q);
    assert(q >= min_q);
    assert(p <= max_p);
    assert(q <= length);
    assert(p <= length);
    if(Hc(pq,Hccc)) {
      //do energy = my_c[pq] if(energy != INF) and energy += in xxx_kernel

      const short* S = &SH[H*(length+2)];
      const unsigned char type   = Ptype(S,pair,i,j);
      assert(type<8);
      const unsigned char type_2 = Ptype(S,pair,q,p);
      assert(type_2<8);

      if(n1==3) {//if (ns==2) and (nl==3)
	assert(p - 1 - i == 3); 
	assert(j - 1 - q == 2);
      } else {
	assert(p - 1 - i == 2);
	assert(j - 1 - q == 3);
      }
      const int si1 = S[i+1];
      const int sj1 = S[j-1];
      const int sq1 = S[q+1];
      const int sp1 = S[p-1];
      /* 2x3 loop */
      energy = internal_loop5+ninio2 +
	       mismatch23I[type][si1][sj1] + mismatch23I[type_2][sq1][sp1];
    }}
  }
  nl3_out[m] = energy;
}		     

//based on interior_loopx.h r1.16
__global__ void
int_loop_I_kernel(const int i, /*const int turn,*/ const int length,
		  const int ninio2,
		  const int internal_loop[MAXLOOP+1],
		  const int mismatchI[NBPAIRS+1][5][5],
		  const float lxc,
		  const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		  const short* __restrict__ SH, //[H,length+2]
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		  const int* __restrict__ my_c,
		        int* __restrict__ out_I) {
  //using up to 29*29-3 threads, for simpliciity BLOCK_SIZE=1024
  //might as well use Grid to determine H
  const size_t H = blockIdx.y; // Langdon's 2026 indexing bug -- widening H to
  //size_t promotes every H*(...) product below to size_t arithmetic
  //automatically, per C's usual arithmetic conversions
  const int j = blockIdx.x + i+turn+1;

  int energy = INF;
  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];

  if(!Hc(ij,Hccc)) {
    const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    if(threadIdx.x==0) out_I[blockId] = INF;
    return;

  } else { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    const int di = threadIdx.x & 0x01f;
    const int dj = threadIdx.x /32;
    assert(di>=0 && di <32);
    assert(dj>=0 && dj <32);
    if(di<=MAXLOOP && dj <=MAXLOOP) {
    const int p = i+1 + di;
    const int q = j-1 - dj;

    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p){
    const int u1 = p - 1 - i; //u1 = p1 - i;
    const int u2 = j - 1 - q; //u2 = j1 - q;

    const int ns = (u1>u2)? u2 : u1;
    const int nl = (u1>u2)? u1 : u2;
    if(ns>2 || (ns==2 && nl>3)) { //avoid data calculated by other kernels     
      const int pq = Indx(p,q);
      assert(q >= min_q);
      assert(p <= max_p);
      assert(q <= length);
      assert(p <= length);
      assert(nl >  0);
      assert(nl <= MAXLOOP);
      if(Hc(pq,Hccc)) {
      const size_t hpq = pq + H*((length+1)*(length+2)/2); // Langdon's 2026 indexing bug -- hpq itself must be size_t too, not just H, or the correctly-widened RHS truncates right back on assignment
      energy = my_c[hpq];
      if(energy != INF) {

	const short* S = &SH[H*(length+2)];
	const unsigned char type   = Ptype(S,pair,i,j);
	assert(type<8);
	const unsigned char type_2 = Ptype(S,pair,q,p);
	assert(type_2<8); 

	const int si1 = S[i+1];
	const int sj1 = S[j-1];
	const int sq1 = S[q+1];
	const int sp1 = S[p-1];

	/* generic interior loop (no else here!)*/
	const int u = nl + ns;

	energy += (u <= MAXLOOP) ? (internal_loop[u]) : (internal_loop[30]+(int)(lxc*log((u)/30.0f)));

	energy += MIN2(MAX_NINIO, (nl-ns)*ninio2);

	energy += mismatchI[type][si1][sj1] + mismatchI[type_2][sq1][sp1];
      }}
    }}}
  }

#define BLOCK_SIZE_ 1024
  assert(blockDim.x == BLOCK_SIZE_); //for simplicity must be a power of 2
  volatile __shared__ int en[BLOCK_SIZE_];
  en[threadIdx.x] = energy; //must set whole of en

#define ix threadIdx.x
#if BLOCK_SIZE_ >=1024
  __syncthreads(); if(ix < 512) en[ix] = MIN2(en[ix], en[ix+512]);
#endif
#if BLOCK_SIZE_ >=512
  __syncthreads(); if(ix < 256) en[ix] = MIN2(en[ix], en[ix+256]);
#endif
#if BLOCK_SIZE_ >=256
  __syncthreads(); if(ix < 128) en[ix] = MIN2(en[ix], en[ix+128]);
#endif
#if BLOCK_SIZE_ >=128
  __syncthreads(); if(ix <  64) en[ix] = MIN2(en[ix], en[ix+ 64]);
#endif
  if(ix < 32) {
#if BLOCK_SIZE_ >=64
    __syncthreads();            en[ix] = MIN2(en[ix], en[ix+ 32]);
#endif
    en[ix] = MIN2(en[ix], en[ix+ 16]);
    en[ix] = MIN2(en[ix], en[ix+  8]);
    en[ix] = MIN2(en[ix], en[ix+  4]);
    en[ix] = MIN2(en[ix], en[ix+  2]);
    en[ix] = MIN2(en[ix], en[ix+  1]);
  }
#undef ix
#undef BLOCK_SIZE_
  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  if(threadIdx.x==0) out_I[blockId] = en[0];
}		     
//based on interior_loopx.h r1.16
__global__ void
int_loop_I1_kernel(const int dj, //2...MAXLOOP
		   const int i, /*const int turn,*/ const int length,
		  const int ninio2,
		  const int internal_loop[MAXLOOP+1],
		  const int mismatchI[NBPAIRS+1][5][5],
		  const float lxc,
		  const char*  __restrict__ pair,//[NBPAIRS+1][NBPAIRS+1],
		  const short* __restrict__ SH, //[H,length+2]
	     const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		  const int* __restrict__ my_c,
		        int* __restrict__ out_I) {
  //if(gridDim.x>=1) return;

  //using up to 29 threads, for simpliciity BLOCK_SIZE=32
  //might as well use Grid to determine H
  const size_t H = blockIdx.y; // Langdon's 2026 indexing bug -- widening H to
  //size_t promotes every H*(...) product below to size_t arithmetic
  //automatically, per C's usual arithmetic conversions
  const int j = blockIdx.x + i+turn+1;

  const int ij = Indx(i,j);
  const unsigned int* Hccc = &hccc[H*Hc_ints(length)];

  //if(gridDim.x>=2000) return;
  if(!Hc(ij,Hccc)) {
    const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    if(threadIdx.x==0) out_I[blockId] = INF;
    return;

  } else { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    unsigned int flag2 = 1<<threadIdx.x;
    unsigned int flag   = 0;
    int energy = INF;
    const int di = threadIdx.x; //& 0x01f;
    assert(di>=0 && di <32);
    assert(dj>=2 && dj <=MAXLOOP);
    if(di<=MAXLOOP) {
    const int p = i+1 + di;
    const int q = j-1 - dj;

    const int min_q = Min_q(i,j,turn);
    const int max_p = Max_p(i,j,q,turn);
    if(q >= min_q && q<=length && p <= max_p){
    const int u1 = p - 1 - i; //u1 = p1 - i;
    const int u2 = j - 1 - q; //u2 = j1 - q;

    const int ns = (u1>u2)? u2 : u1;
    const int nl = (u1>u2)? u1 : u2;
    if(ns>2 || (ns==2 && nl>3)) { //avoid data calculated by other kernels     
      const int pq = Indx(p,q);
      assert(q >= min_q);
      assert(p <= max_p);
      assert(q <= length);
      assert(p <= length);
      assert(nl >  0);
      assert(nl <= MAXLOOP);
      if(Hc(pq,Hccc)) {
      const size_t hpq = pq + H*((length+1)*(length+2)/2); // Langdon's 2026 indexing bug -- hpq itself must be size_t too, not just H, or the correctly-widened RHS truncates right back on assignment
      energy = my_c[hpq];
      if(energy != INF) {
	flag = 1;

	const short* S = &SH[H*(length+2)];
	const unsigned char type   = Ptype(S,pair,i,j);
	assert(type<8);
	const unsigned char type_2 = Ptype(S,pair,q,p);
	assert(type_2<8); 

	const int si1 = S[i+1];
	const int sj1 = S[j-1];
	const int sq1 = S[q+1];
	const int sp1 = S[p-1];

	/* generic interior loop (no else here!)*/
	const int u = nl + ns;

	energy += (u <= MAXLOOP) ? (internal_loop[u]) : (internal_loop[30]+(int)(lxc*log((u)/30.0f)));

	energy += MIN2(MAX_NINIO, (nl-ns)*ninio2);

	energy += mismatchI[type][si1][sj1] + mismatchI[type_2][sq1][sp1];
      }}
    }}}

    if(0 && gridDim.x>=2000) {
    if(threadIdx.x==0) out_I[0] = energy;
    return;
    }

#define BLOCK_SIZE_ 32
  assert(blockDim.x == BLOCK_SIZE_); //for simplicity must be a power of 2
#if NDEBUG || BLOCK_SIZE_ >=64
  volatile __shared__ int en[BLOCK_SIZE_+16];
#else
  volatile __shared__ int en[BLOCK_SIZE_+16]; //avoid addressing errors although en[32..47] contains junk
#endif

  en[threadIdx.x] = flag2;   //use all threads, must set whole of en
  __syncthreads(); 
  int countb = 0;
  if(gridDim.x==2000) if(threadIdx.x==0) for(int k=0;k<blockDim.x;k++) countb = countb | en[k];

  en[threadIdx.x] = flag;   //use all threads, must set whole of en
  __syncthreads(); 
  int countf = 0;
  if(gridDim.x==2000) if(threadIdx.x==0) for(int k=0;k<blockDim.x;k++) if(en[k])      countf++;

  en[threadIdx.x] = energy; //use all threads, must set whole of en
  __syncthreads(); 
  int count = 0;
  if(gridDim.x==2000) if(threadIdx.x==0) for(int k=0;k<blockDim.x;k++) if(en[k]!=INF) count++;


  if(gridDim.x==2000) {
    if(threadIdx.x==0 && (blockIdx.x%100)==0 ) {
      //printf("%d int_loop_I1_kernel( %d i=%d,%d,%d...) H %d j %d count %d countf %d threadIdx.x==0 p %d q %d min_q %d max_p %d u1 %d u2 %d ns %d nl %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
      printf("%d int_loop_I1_kernel( %d i=%d,%d,%d...) H %d j %d count %d countf %d countb %08x blockDim.x %d, %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
	     blockIdx.x,
	     dj,i,length,ninio2,
	     H,j,count,countf,countb,blockDim.x,
	     //p,q,
	     //min_q,max_p,u1,u2,ns,nl,
	     en[0],en[1],en[2],en[3],en[4],en[5],en[6],en[7],en[8],en[9],en[10],en[11],en[12],en[13],en[14],en[15],en[16],en[17],en[18],en[19],en[20],en[21],en[22],en[23],en[24],en[25],en[26],en[27],en[28],en[29],en[30],en[31]);
    }}

#define ix threadIdx.x
#if BLOCK_SIZE_ >=1024
  __syncthreads(); if(ix < 512) en[ix] = MIN2(en[ix], en[ix+512]);
#endif
#if BLOCK_SIZE_ >=512
  __syncthreads(); if(ix < 256) en[ix] = MIN2(en[ix], en[ix+256]);
#endif
#if BLOCK_SIZE_ >=256
  __syncthreads(); if(ix < 128) en[ix] = MIN2(en[ix], en[ix+128]);
#endif
#if BLOCK_SIZE_ >=128
  __syncthreads(); if(ix <  64) en[ix] = MIN2(en[ix], en[ix+ 64]);
#endif
  if(ix < 32) {
#if BLOCK_SIZE_ >=64
    __syncthreads();            en[ix] = MIN2(en[ix], en[ix+ 32]);
#endif
    en[ix] = MIN2(en[ix], en[ix+ 16]);
    en[ix] = MIN2(en[ix], en[ix+  8]);
    en[ix] = MIN2(en[ix], en[ix+  4]);
    en[ix] = MIN2(en[ix], en[ix+  2]);
    en[ix] = MIN2(en[ix], en[ix+  1]);
  }
#undef ix
#undef BLOCK_SIZE_
  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  //if(gridDim.x>=2000) {
  //  if(threadIdx.x==0) out_I[0] = en[0];
  //  return;
  //}
  if(threadIdx.x==0) out_I[blockId] = en[0];
  }
}		     
#undef MAX_NINIO

//No longer used,  two kernals to combine output of small kernels above
//Run after kernals which calculate only one p,q and so do not use my_c
__global__ void
int_loop_min_kernel(const int dp, const int dq, //0,0, 1,1 (int11) or 2,2 (int22) 2,3 3,2
		    const int nfiles,
		    const int i, /*const int turn,*/ const int length,
		    const int* __restrict__ my_c,
		    const int* __restrict__ energy_in,
		    const int* __restrict__ energy_min0,  //sanity check
			  int* __restrict__ energy_min) { //In,Out
  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  //initialise output on first use when dp and dq are zero
  //if(dp==0 && dq==0) energy_min[hj] = INF;

  const int p = i+1+dp;
  const int q = j-1-dq;
  const int min_q = Min_q(i,j,turn);
  const int max_p = Max_p(i,j,q,turn);
  if(q >= min_q && q<=length && p <= max_p) {
    const size_t pq   = Indx(p,q) + (size_t)H*((length+1)*(length+2)/2); // Langdon's 2026 indexing bug
    const int c_   = my_c[pq];
    const int delta = energy_in[m];
    if(c_ != INF && delta != INF) {
      const int energy = c_ + delta;
      const int hj = H*(length+1)+j;
#ifdef CHECK
      const int sanity = energy_min0[hj];
      if(energy<sanity) {
	printf("%d,%d int_loop_min_kernel(%d,%d,%d,i=%d,%d,%d...) stride %d H %d j %d p %d q %d my_c[%d]%d energy_in[%d]%d energy_min0[%d]%d energy_min[%d]%d\n",
	       blockIdx.x,threadIdx.x,
	       dp,dq,nfiles,i,turn,length,	 
	       stride,H,j,p,q,
	       pq,my_c[pq], m,energy_in[m], hj,energy_min0[hj], hj,energy_min[hj]);
	assert(0);
      }
#endif //CHECK
      if(energy<energy_min[hj]) energy_min[hj] = energy;
    }
  }
}


//Run after kernals which calculate min of multiple p,q
__global__ void
int_loop_min_kernel2(const int equality, //0 or 1
		     const int nfiles,
		     const int i, /*const int turn,*/ const int length,
		     const int* __restrict__ energy_in,
		     const int* __restrict__ energy_min0,  //sanity check
		 	   int* __restrict__ energy_min) { //In,Out
  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = length - (i+turn+1) + 1;
  if(m >= stride*nfiles) return;
  const int H = m / stride;
  const int j = m - H*stride + i+turn+1;

  const int energy = energy_in[m];
  const int hj = H*(length+1)+j;
#ifdef CHECK
  const int sanity = energy_min0[hj];
  if((equality && MIN2(energy,energy_min[hj]) != sanity) || energy < sanity) {
    printf("%d,%d int_loop_min_kernel2(%d,%d,i=%d,%d,%d...) stride %d H %d j %d energy_in[%d]%d energy_min0[%d]%d energy_min[%d]%d\n",
	   blockIdx.x,threadIdx.x,
	   equality,nfiles,i,turn,length,	 
	   stride,H,j,
	   m,energy_in[m], hj,energy_min0[hj], hj,energy_min[hj]);
    assert(0);
  }
#endif //CHECK
  if(energy<energy_min[hj]) energy_min[hj] = energy;
}


__global__ void
int_loop_kernel(const int i, /*const int turn,*/ const int length,
		const int TerminalAU, const int ninio2,
		const cuda_param_t* __restrict__ P, const float lxc,
		const char* __restrict__ pair_, //[NBPAIRS+1][NBPAIRS+1],
		const short* __restrict__ S,    //[length+2]
		const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		const int* __restrict__ my_c,
		      int* __restrict__ energy_min) { //out

  int energy = INF;
  const size_t H = blockIdx.y; // Langdon's 2026 indexing bug -- widens the
  //&my_c[H*((length+1)*(length+2)/2)] subscript below to size_t arithmetic
  const int j = blockIdx.x + i+turn+1;


  const int ij = Indx(i,j);
  if(Hc(ij,&hccc[H*Hc_ints(length)])) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    /* we evaluate this pair */
    unsigned int mask; //search context
    int row_start = 0; //search context
    int done = 0;      //search context
    int column = 0;
    int row = 0;
    const int p0 = i+1;
    const int q0 = Min_q(i,j,turn);
    const int maxcol = MIN2(MAXLOOP,(j - 1) - q0);
    for(int work = threadIdx.x; setpq(p0,q0,maxcol,&hccc[H*Hc_ints(length)],work,mask,row_start,done,column,row); work += BLOCK_SIZE) {
      const int p = p0 + row;
      const int q = q0 + column;
      const int energy2 = Energy(i,j,q,p,
		    &my_c[H*((length+1)*(length+2)/2)],
		    &S[H*(length+2)],pair_,P,
		    TerminalAU,ninio2,
		    P->bulge,P->internal_loop,lxc,
		    P->mismatchI,
		    P->mismatch1nI,
		    P->mismatch23I,
		    P->stack,
		    P->int11,
		    P->int21,
		    P->int22);

    energy = MIN2(energy,energy2);
  }//endfor

#if NDEBUG || BLOCK_SIZE >=64
  volatile __shared__ int en[BLOCK_SIZE];
#else
  //avoid cuda-memcheck reporting addressing errors although en[32..47] contains junk
  volatile __shared__ int en[BLOCK_SIZE+16];
#endif
  en[threadIdx.x] = energy; //must set whole of en

#define ix threadIdx.x
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
  energy = en[0];
#undef ix
  }//endif VRNA_CONSTRAINT_CONTEXT_INT_LOOP
  if(threadIdx.x==0) energy_min[H*(length+1)+j] = energy;
}

//Host (ie non-GPU) code
PRIVATE void
int_loop_cuda(const int nfiles,
	      const int i, /*const int turn,*/ const int length,
	      const vrna_param_t *P,
	      int* energy_min) { //out
  //cf modular_decomposition.cu r1.79
  const int nblocks = length - (i+turn);
  if(nblocks<=0) return;

  //Using gridDim for convenience but imposes a 65535 limit on length (or nfiles)
  dim3 blocks(nblocks,nfiles);

  int_loop_kernel<<<blocks,BLOCK_SIZE>>>(i, /*turn,*/ length,
					  P->TerminalAU,P->ninio[2],
					  d_param,P->lxc,
					  d_pair,
					  d_S,
					  d_hccc, 
					  d_my_c,
					  d_energy_min2); //Out

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  //printf("int_loop_kernel<<<%d.%d,%d>>>(i=%d...) ok\n",blocks.x,blocks.y,BLOCK_SIZE,i);
  
  gpuErrchk( cudaMemcpy(energy_min,d_energy_min2, nfiles*(length+1)*sizeof(int),cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaDeviceSynchronize() );

  /*used to have alternative code to launch int_loop_nl0_kernel etc here */
  return;
}
#undef bitsperint

#undef MIN2
//ViennaRNA/utils.h
#define MIN2(A, B)      ((A) < (B) ? (A) : (B))
#undef MAX2
//ViennaRNA/utils.h
#define MAX2(A, B)      ((A) > (B) ? (A) : (B))
#undef turn

PUBLIC void
int_loop_i(const int nfiles,
	   const vrna_fold_compound_t **VC,
	   const int i, const int turn_, const int length,
	   /*const int* indx, const int ijsize,
	   const char* hard_constraints, const int* my_c,*/
	   int* energy_min ) { //out
  if(first2) init_gpu2(nfiles,VC, turn_, length, BLOCK_SIZE);


  int_loop_cuda(nfiles,i,/*turn,*/length,VC[0]->params, energy_min);
  return;
  /* normal code to run calculation on host to check answers given by GPU
  int new_e[length+1];
  int_loop_cuda(i,turn,length,my_c, new_e);

  int j;
  for (j = i+turn+1; j <= length; j++) {
    const int ij  = indx[j]+i;
    assert(ij>=0 && ij<ijsize);
    const int hc_decompose  = hard_constraints[ij];

    if (hc_decompose) {   // we evaluate this pair **

      // check for interior loops **
      energy_min[j] = E_int_loop(vc, i, j); //vrna_E_int_loop(vc, i, j);
    } // end >> if (pair) << **
  }

  int err = 0;
  for (j = i+turn+1; j <= length; j++) {
    if(new_e[j] != energy_min[j]) {
      printf("new_e[%d]%d != energy_min[%d]%d\n",j,new_e[j], j,energy_min[j]);
      err = 1;
    } else {
//    printf("new_e[%d]%d\n",j,new_e[j]);
    }
  }
  if(err) exit(1);
*/
}

//Unused to host C code to check answers given by GPU code
PRIVATE int
E_int_loop( const vrna_fold_compound_t *vc,
            const int i,
            const int j){

  unsigned char     type, type_2;
  char              *hc, *hc_pq, eval_loop;
  char              *ptype, *ptype_pq;
  short             *S, S_i1, S_j1, *S_p1, *S_q1;
  int               q, p, j_q, p_i, pq, *c_pq, max_q, max_p, tmp,
                    *rtype, /*noGUclosure, **no_close,*/ energy, cp, //en,
                    *indx, *hc_up, ij, hc_decompose, e, *c, //*ggg,
                    //with_gquad, 
                    turn;
  vrna_sc_t         *sc;
  vrna_param_t      *P;
  vrna_md_t         *md;
  vrna_mx_mfe_t     *matrices;
//vrna_ud_t         *domains_up;
//#ifdef WITH_GEN_HC
//vrna_callback_hc_evaluate *f;
//#endif

  cp            = vc->cutpoint;
  indx          = vc->jindx;
  hc            = vc->hc->matrix;
  hc_up         = vc->hc->up_int;
  P             = vc->params;
  matrices      = vc->matrices;
  ij            = indx[j] + i;
  hc_decompose  = hc[ij];
  e             = INF;
  c             = vc->matrices->c;
//ggg           = vc->matrices->ggg;
  md            = &(P->model_details);
//with_gquad    = md->gquad;
  turn          = md->min_loop_size;
//domains_up    = vc->domains_up;

//#ifdef WITH_GEN_HC
//f = vc->hc->f;
//#endif

  /* CONSTRAINED INTERIOR LOOP start */
  if(hc_decompose & VRNA_CONSTRAINT_CONTEXT_INT_LOOP){
    /* prepare necessary variables */
    rtype       = &(md->rtype[0]);
//  noGUclosure = md->noGUclosure;
    max_q       = i+turn+2;
    max_q       = MAX2(max_q, j - MAXLOOP - 1);

    ptype     = vc->ptype;
    type      = (unsigned char)ptype[ij];
//  no_close  = (((type==3)||(type==4))&&noGUclosure);
    S         = vc->sequence_encoding;

    S_i1      = S[i+1];
    S_j1      = S[j-1];
    sc        = vc->sc;

  /*if(type == 0) gcov says branch never taken
      type = 7;*/

  /*if(domains_up && domains_up->energy_cb){
      exit(1); gcov says branch never taken
      for(q = j - 1; q >= max_q; q--){
        j_q = j - q - 1;

        if(hc_up[q+1] < j_q) break;

        pq        = indx[q] + i + 1;
        p_i       = 0;
        max_p     = i + 1;
        tmp       = i + 1 + MAXLOOP - j_q;
        max_p     = MAX2(max_p, tmp);
        tmp       = q - turn;
        max_p     = MIN2(max_p, tmp);
        tmp       = i + 1 + hc_up[i + 1];
        max_p     = MIN2(max_p, tmp);
        hc_pq     = hc + pq;
        c_pq      = c + pq;

        ptype_pq  = ptype + pq;
        S_p1      = S + i;
        S_q1      = S + q + 1;

        for(p = i+1; p <= max_p; p++){
          eval_loop = *hc_pq & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC;
#ifdef WITH_GEN_HC
          if(f)
            eval_loop = (f(i, j, p, q, VRNA_DECOMP_PAIR_IL, vc->hc->data)) ? eval_loop : (char)0;
#endif
          ** discard this configuration if (p,q) is not allowed to be enclosed pair of an interior loop **
          if(eval_loop){
            energy = *c_pq;
            if(energy != INF){
              type_2 = rtype[(unsigned char)*ptype_pq];

              if(type_2 == 0)
                type_2 = 7;

              if (noGUclosure)
                if (no_close||(type_2==3)||(type_2==4))
                  if ((p>i+1)||(q<j-1)) continue;  ** continue unless stack **

              energy += eval_interior_loop( vc, i, j, p, q);
              e = MIN2(e, energy);
            }
          }
          hc_pq++;    ** get hc[pq + 1] **
          c_pq++;     ** get c[pq + 1] **
          p_i++;      ** increase unpaired region [i+1...p-1] **

          ptype_pq++; ** get ptype[pq + 1] **
          S_p1++;

          pq++;
        } ** end q-loop **
      } ** end p-loop **
    } else */{

      for(q = j - 1; q >= max_q; q--){
        j_q = j - q - 1;

        if(hc_up[q+1] < j_q) break; //appears to be needed despite that gcov says it has no impact

        pq        = indx[q] + i + 1;
        p_i       = 0;
        max_p     = i + 1;
        tmp       = i + 1 + MAXLOOP - j_q;
        max_p     = MAX2(max_p, tmp);
        tmp       = q - turn;
        max_p     = MIN2(max_p, tmp);
        tmp       = i + 1 + hc_up[i + 1];
        max_p     = MIN2(max_p, tmp);
        hc_pq     = hc + pq;
        c_pq      = c + pq;

        ptype_pq  = ptype + pq;
        S_p1      = S + i;
        S_q1      = S + q + 1;

        for(p = i+1; p <= max_p; p++){
          eval_loop = *hc_pq & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC;
//#ifdef WITH_GEN_HC
//        if(f)
//          eval_loop = (f(i, j, p, q, VRNA_DECOMP_PAIR_IL, vc->hc->data)) ? eval_loop : (char)0;
//#endif
          /* discard this configuration if (p,q) is not allowed to be enclosed pair of an interior loop */
          if(eval_loop){
            energy = *c_pq;
            if(energy != INF){
              type_2 = rtype[(unsigned char)*ptype_pq];

	      /* gcov says if never taken
              if (noGUclosure)
		exit(1);
                if (no_close||(type_2==3)||(type_2==4))
                  if ((p>i+1)||(q<j-1)) continue;  ** continue unless stack **

              if(type_2 == 0)
                type_2 = 7;
			 */
              energy += ubf_eval_int_loop(i, j, p, q,
                                          i + 1, j - 1, p - 1, q + 1,
                                          S_i1, S_j1, *S_p1, *S_q1,
                                          type, type_2, rtype,
                                          ij, cp,
                                          P, sc);
	      /*
	      printf("ubf_eval_int_loop( %d %d %d %d ...) %d %d %d c[%d]%d gives %d\n",
		     i, j, p, q, p-(i+1), q-max_q, (p-(i+1))+(q-max_q),int(c_pq-c),*c_pq,energy);
	      stop = 1;
	      */
              e = MIN2(e, energy);
            }
          }
          hc_pq++;    /* get hc[pq + 1] */
          c_pq++;     /* get c[pq + 1] */
          p_i++;      /* increase unpaired region [i+1...p-1] */

          ptype_pq++; /* get ptype[pq + 1] */
          S_p1++;

          pq++;
        } /* end q-loop */
      } /* end p-loop */
    }

    /*gcov says branch never taken
    if(with_gquad){
      ** include all cases where a g-quadruplex may be enclosed by base pair (i,j) **
      if ((!no_close) && ((cp < 0) || ON_SAME_STRAND(i, j, cp))) {
        energy = E_GQuad_IntLoop(i, j, type, S, ggg, indx, P);
        e = MIN2(e, energy);
      }
    }
    */
  }

  return e;
}

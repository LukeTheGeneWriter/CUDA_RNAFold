#define Version "$Revision: 1.188 $ "
//Helper for fill_arrays.c 
//based on ViennaRNA-2.3.0/src/ViennaRNA/interior_loops.c (Nov  1  2016) 

//Modifications (reverse order):
//WBL 28 Aug 2026 Clean for commit.
//WBL 27 Aug 2026 Revert to r1.184 as pair.h slower Reorder d_my_c
//WBL 26 Aug 2026 Clean for commit. Remove VC[H]->jindx lookup overhead
//WBL 25 Aug 2026 Add load_min_dmli_kernel, int_loop_mls uses fml_j not prev_fml
//WBL 19 Aug 2026 Add int_loop_mls_kernel2
//WBL  9 Aug 2026 prepare to use int_loop_mls_kernel
//WBL  8 Aug 2026 Add int_loop_mls based on part of fill_arrays_loop.c
//WBL 31 Jul 2026 make H tightest index d_energy_min2/energy_min
//WBL 30 Jul 2026 swap H,j blockIdx x,y in int_loop_kernel
//WBL 29 Jul 2026 for int_loop_kernel, pack d_S ten per word H fastest index
//WBL 28 Jul 2026 Merge LukeTheGeneWriter/CUDA_RNAFold Commit 6da6612
//    Removed 11 unused int_loop_*_kernel to make maintenance easier
//WBL 19 Jul 2026 Reorder d_new_e, (d_my_c still todo)
//WBL 19 Jul 2026 Allow arrays to exceed two billion elements
//WBL 12 Jul 2026 for CUDA 13 Luke Williams
//WBL 17 Feb 2018 clean for production (cf r1.75), remove tick
//    keep source code of small unused kernels for the timebeing but remove calling them.
//WBL 11 Feb 2018 use own timing rather than nvidia profiling tools
//WBL  6 Feb 2018 split interior_loopx.h into separate non-divergent kernels
//WBL 28 Jan 2018 process nfiles in one go
//WBL 11 Jan 2018 CUDA GGGP ViennaRNA-2.3.0 rf/rf_cuda2

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
unsigned int* d_S;    //S[length+2] pack 10 per word
int*          d_my_c;
int*          d_energy_min2; //share with modular_decomposition.cu ?
int*          d_new_e;
//int*        d_energy_3p_00;
//int*        d_energy_3p_en;
//int*        d_energy_mls;
struct energy_3p* d_energies; //only one row
int*          d_out_fml; //only to transfer part of My_fML
//no longer in use
//int*        d_energy_min20; //alternative calculation of d_energy_min2
//int*        d_buf;  //intermediate energy result GPU only

#define BLOCK_SIZE 512
#define BLOCK_SIZE2 1024

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk2(ans,first) { gpuAssert((ans), __FILE__, __LINE__,first); }
inline void gpuAssert(cudaError_t code, const char *file, const int line, const bool first=false, const bool abort=true)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA error: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
     if(first) fprintf(stderr,"CUDA error: on first kernel. %s\n", file);
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
init_my_c_kernel(const long long ijsize,
		 int* __restrict__ my_c) {
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=ijsize) return;
  my_c[m] = INF;
}

PUBLIC void
init_my_c(const long long ijsize) {
  /* Setup execution parameters for helper kernel */
  const long long nblocks = (ijsize + BLOCK_SIZE - 1)/BLOCK_SIZE;
#ifndef NDEBUG
  printf("init_my_c_kernel<<<%lld,%d>>>(%lld,%p)\n",
	 nblocks,BLOCK_SIZE,ijsize, d_my_c);fflush(NULL);
#endif
  init_my_c_kernel<<<nblocks,BLOCK_SIZE>>>(ijsize, d_my_c);
  gpuErrchk2( cudaPeekAtLastError(),  first2 );
#ifndef NDEBUG
  gpuErrchk2( cudaDeviceSynchronize(),first2 );
  //may pickup errors later if dont sync now
  printf("init_my_c_kernel<<<%lld,%d>>> ok\n",nblocks,BLOCK_SIZE);fflush(NULL);
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
//#define Hc_ints(length) (((length*(length+1))/2+2 + (MAXLOOP+1)*(MAXLOOP+2)/2 + bitsperint - 1)/bitsperint)
__host__ __device__ inline
long long Hc_ints(const int length) {
  const long long l1 = length+1; //force 64 bit calculation
  return ((length*l1)/2+2 + (MAXLOOP+1)*(MAXLOOP+2)/2 + bitsperint - 1)/bitsperint;
}

int getbs(const char* envname, const int def) {
  const char* s = getenv(envname);
  if(s==NULL) return def;
  const int i = atoi(s);
  return (i)? i : def;
}

extern int load_min_fML_kernel_bs;
extern int fmli_kernel_bs;
extern int modular_decomposition_kernel_bs;
//extern int load_fML_kernel_bs;
int int_loop_kernel_bs = 32;
int load_my_c_kernel_bs = 512;
int int_loop_mls_kernel_bs = 32;
int int_loop_mls_kernel2_bs = BLOCK_SIZE2;
int int_loop_dmli_kernel_bs = 32;

void put10(const unsigned int word, const int H, const int nfiles, const int i, const int size, unsigned int* out){
  assert(word <= 04444444444); //max legit value in octal
  const int I = (H + nfiles*i)/10;
  assert(I>=0 && I < size);
  assert(out[I] == 0xffffffff);
  out[I] = word;
}

PUBLIC void
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size) {
  if(!first2) return;
  fprintf(stderr,"%-26s init_gpu2(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

  assert(turn_ == turn);
  assert(MAX_NINIO == 300); //ViennaRNA/energy_par.c
  //printf("%s %s d_param is %lu bytes, NBPAIRS %d MAXLOOP %d BLOCK_SIZE %d\n",
  //	 __FILE__,Version,sizeof(cuda_param_s),NBPAIRS,MAXLOOP,block_size);

  load_min_fML_kernel_bs =          getbs("load_min_fML_kernel",64);
  fmli_kernel_bs =                  getbs("fmli_kernel",64);
  modular_decomposition_kernel_bs = getbs("modular_decomposition_kernel",64);
//load_fML_kernel_bs =              getbs("load_fML_kernel",64);
  int_loop_kernel_bs =              getbs("int_loop_kernel",32);
  load_my_c_kernel_bs =             getbs("load_my_c_kernel",512);
  int_loop_mls_kernel_bs =          getbs("int_loop_mls_kernel",32);
  int_loop_mls_kernel2_bs =         getbs("int_loop_mls_kernel2",BLOCK_SIZE2);
  int_loop_dmli_kernel_bs =         getbs("int_loop_dmli_kernel",32);

  /*if(modular_decomposition_kernel_bs != 64) {
    fprintf(stderr,"variable modular_decomposition_kernel_bs not implemented %d\n",
	    modular_decomposition_kernel_bs); exit(99);}*/
  if(int_loop_kernel_bs != 32) {
    fprintf(stderr,"variable int_loop_kernel_bs not implemented %d\n",
	    int_loop_kernel_bs); exit(99);}
  if(int_loop_mls_kernel2_bs != BLOCK_SIZE2) {
    fprintf(stderr,"variable int_loop_mls_kernel2_bs not implemented %d\n",
	    int_loop_mls_kernel2_bs); exit(99);}

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
  size_t size = (NBPAIRS+1)*(NBPAIRS+1)*sizeof(char);
  gpuErrchk( cudaMalloc((void **) &d_pair, size) );
  gpuErrchk( cudaMemcpy(d_pair,pair_,size,cudaMemcpyHostToDevice) );

  size = (size_t)nfiles*Hc_ints(length)*sizeof(unsigned int); // 32-bit signed integer overflow bug fix
  gpuErrchk( cudaMalloc((void **) &d_hccc, size) );
  unsigned int* hccc   = (unsigned int*) calloc((size_t)nfiles*Hc_ints(length),sizeof(unsigned int)); // 32-bit signed integer overflow bug fix
  for(int H=0;H<nfiles;H++){
    assert(bitsperint==(1+0x1f));
    unsigned int mask;
    const long long l1  = length+1;
    const long long top = length*l1/2+2;
    for(long long i=0;i<top;i++){ //leave padding as zero
      mask = ((i & 0x1f) == 0)? 1 : mask << 1;
      const long long I = H*Hc_ints(length)+i/bitsperint;
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC) hccc[I] |= mask;
    }
  }
  gpuErrchk( cudaMemcpy(d_hccc,hccc,(size_t)nfiles*Hc_ints(length)*sizeof(unsigned int),cudaMemcpyHostToDevice) ); // 32-bit signed integer overflow bug fix
  free(hccc);

   
  //ten S per word, length+2
  assert(sizeof(unsigned int) == 4);
  size = ((nfiles * (length+2) + 9)/10) * sizeof(unsigned int);
  gpuErrchk( cudaMalloc((void **) &d_S, size) );
  unsigned int* buff = (unsigned int*) malloc(size); //could use cudaMallocHost
#ifndef NDEBUG
  memset(buff,0xff,size);
#endif
  const int len = (length+2);
  int H0 = 0;
  int i0 = 0;
  int j = 0; //0 to 9
  unsigned int word;
  for(int i=0;i<len;i++){
  for(int H=0;H<nfiles;H++){
    //pack ten per word, cf unpack()
    if(j==0) {word = 0; H0 = H; i0 = i;}
    const unsigned int s = VC[H]->sequence_encoding[i];
    assert(s <= 4);
    assert(j >= 0 && j < 10);
    const unsigned int s2 = s << (j*3);
    word = word | s2;
    j++;
    if(j >= 10) {
      j=0; put10(word,H, nfiles,i, size/4,buff);
    }
  }}
  if(j>0)  put10(word,H0,nfiles,i0,size/4,buff);
  for(size_t i=0;i<size/4;i++) assert(buff[i] <= 04444444444);
  gpuErrchk( cudaMemcpy(d_S,buff,size,cudaMemcpyHostToDevice) );
  free(buff);
  
  size = Hoff(nfiles,length); //nfiles*(length+1)*(length+2)/2
  gpuErrchk( cudaMalloc((void **) &d_my_c, size*sizeof(int)) );
  init_my_c(size);

  size = (size_t)nfiles*(length+1)*sizeof(int); // 32-bit signed integer overflow bug fix
  gpuErrchk( cudaMalloc((void **) &d_new_e, size) );

  gpuErrchk( cudaMalloc((void **) &d_energy_min2, size) );
  //const long long ijsize = (length+1)*(length+2)/2;
  //const long long size2  = nfiles*ijsize*sizeof(int);
  //gpuErrchk( cudaMalloc((void **) &d_energy_3p_00, size2) );
  //gpuErrchk( cudaMalloc((void **) &d_energy_3p_en, size2) );
  //gpuErrchk( cudaMalloc((void **) &d_energy_mls,   size2) );
  const int size3 = nfiles*((length+1) - (1+turn+1))*sizeof(energy_3p);
  gpuErrchk( cudaMalloc((void **) &d_energies,size3) );
  /*no longer in use 
  gpuErrchk( cudaMalloc((void **) &d_energy_min20,size) );

  size = nfiles*length*sizeof(int);
  gpuErrchk( cudaMalloc((void **) &d_buf, size) );
  */
  //use d_out_fml pending better intergration of fill_arrays_loop and GPU for My_fML
  const int mem_size_len = nfiles*(length+1) * sizeof(int); //oversize for simplicity
  gpuErrchk( cudaMalloc((void **) &d_out_fml,mem_size_len) );
  first2 = 0;
  //printf("%-24s init_gpu2 done\n",__FILE__);fflush(NULL);
}

//perhaps this can be combined with other kernels?
/*now define const int turn,*/
__global__ void
load_my_c_kernel(const int i, const int length, const int nfiles,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c) { //out
  const long long m  = blockIdx.x*blockDim.x + threadIdx.x;
  const long long mj = m / nfiles;
  const int       H  = m - mj * nfiles;
  const long long j  = mj + i+turn+1; 
  if(j < i+turn+1 || j > length) return;

  assert(H >= 0 && H < nfiles);
  const long long ij = Indx(i,j);
  const long long indx = H+ij*nfiles;
  assert(ij>=0 && ij<Hoff(1,length)); //(length+1)*(length+2)/2
  assert(my_c[indx] == INF);
  const int eindx = m;
  assert(eindx == H + (j-(i+turn+1))*nfiles);
  assert(eindx < nfiles*(length - (i+turn+1) + 1));
  assert(eindx < nfiles*(length -   (turn+1)));
         my_c[indx] = new_e[eindx];
}

PUBLIC void
load_my_c(const int nfiles,
	  const int i, const int turn_, const int length,
	  const int* new_e) {   //in
  //out d_my_c
  const int start = i+turn+1; 
  const int size  = length - start + 1;
  if(size<=0) return;

  assert(turn_ == turn);
#ifdef NDEBUG
  //check here in case of earlier errors
  gpuErrchk( cudaDeviceSynchronize() );
#endif
  //transfer only used part of new_e
  gpuErrchk( cudaMemcpy(d_new_e,new_e,nfiles*size*sizeof(int),cudaMemcpyHostToDevice) );

  /* Setup execution parameters for helper kernel */
  const int nblocks = (nfiles*size + load_my_c_kernel_bs - 1)/load_my_c_kernel_bs;

  //dim3 blocks(nblocks,nfiles);

#ifndef NDEBUG
  printf("load_my_c_kernel<<<%d,%d>>>(%d,%d,%d,d_new_e,d_my_c)\n",
	 nblocks,load_my_c_kernel_bs,i,length,nfiles);//,d_new_e,d_my_c);
#endif
  load_my_c_kernel<<<nblocks,load_my_c_kernel_bs>>>(i, /*turn,*/ length, nfiles,
					   d_new_e,  //in
					   d_my_c); //out
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

__device__ inline
int unpack(const unsigned int* S, const int H, const int nfiles, const int i){ //ten per word
  assert(H>=0 && H < nfiles);
  const int k = H + nfiles*i;
  const int I = k/10;
  const int shift = (k - I*10)*3;
  assert(shift >= 0 && shift <= 32-3);
  const int out = (S[I] >> shift) & 7;
  assert(out>=0 && out <= 4);
  return out;
}

//#include "ptype.cu"
//Was
//WBL 13 Jan 2018 From ViennaRNA-2.3.0/src/ViennaRNA/alphabet.c Revision: 1.9

//Modification:
//WBL 14 Jan 2018 just single element of ptype[ij]

//Based on ViennaRNA-2.3.0/src/ViennaRNA/utils.c
//replace ptypes array
__device__ inline unsigned char
Ptype(const unsigned int* __restrict__ S,const char* __restrict__ pair,//[8][8],
      const int H, const int nfiles, const int i, const int j) {

  const int si = unpack(S,H,nfiles,i);
  const int sj = unpack(S,H,nfiles,j);
  //assert(i>=0 && i<=length);
  //assert(j>=0 && j<=length);
  assert(si>=0 && si<8);
  assert(sj>=0 && sj<8);

  return pair[si*8 + sj];

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


#undef BLOCK_SIZE
//Like modular_decomposition.cu have one block per j value
//each block has (MAXLOOP+1)*(MAXLOOP+2)/2 worker threads
//present reduction code needs BLOCK_SIZE to be at least 32 and a power of 2
#define BLOCK_SIZE 32

//emulate hc[pq] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC;
__device__ inline
int Hc(const long long ij, const unsigned int* __restrict__ hccc){
  const long long I = ij/bitsperint;
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
    const long long pq = Indx(i,j+column);
    const long long I =     pq/bitsperint;
    const long long x = pq - I*bitsperint;
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
Energy(const int H, const int nfiles, const int i, const int j, const int q, const int p,
	  /*const char* hard_constraints,*/
          const int*          __restrict__ my_c,
	  /*const int* hc_up, const char* hc, const unsigned int* __restrict__ hccc,*/
	  const unsigned int* __restrict__ S,
	  const char*         __restrict__ pair_,//[NBPAIRS+1][NBPAIRS+1],
	  const cuda_param_t* __restrict__ P,
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


	  const long long pq = Indx(p,q);
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
	    energy = my_c[H+pq*nfiles];
	    if(energy != INF){
	      //assert(ptype[pq]>=0 && ptype[pq]<8);
	      //const unsigned char type_2 = rtype[(unsigned char)ptype[pq]];
	      const unsigned char type   = Ptype(S,pair_,H,nfiles,i,j);
	      //assert(type == Ptype(S,pair_,i,j));
	      assert(type<8);
	      const unsigned char type_2 = Ptype(S,pair_,H,nfiles,q,p);
	      assert(type_2<8);
	      //assert(i+pp  >=0 && i+pp  <length+2);

	      const int u1 = p - 1 - i; //u1 = p1 - i;
	      const int u2 = j - 1 - q; //u2 = j1 - q;

	      const int ns = (u1>u2)? u2 : u1;
	      const int nl = (u1>u2)? u1 : u2;

	      const int si1 = unpack(S,H,nfiles,i+1);
	      const int sj1 = unpack(S,H,nfiles,j-1);
	      const int sp1 = unpack(S,H,nfiles,i+pp);
	      const int sq1 = unpack(S,H,nfiles,q + 1);

	      energy += IntLoop_X(u1, ns, nl, type, type_2, 
				  si1, sj1, sp1, sq1,
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
/*Removed r1.126 (28 Jul 2026) to make maintenance easier
If need
  int_loop_nl0_kernel,
  int_loop_ns0_kernel,
  int_loop_1xn_kernel,
  int_loop_int11_kernel,
  int_loop_min_kernel,
  int_loop_min_kernel2,
  int_loop_int21_kernel,
  int_loop_int22_kernel,
  int_loop_nl3_kernel,
  int_loop_I_kernel,
  int_loop_I1_kernel,
restore from ViennaRNA-2.3.0cuda.tar.gz github or other backup*/

__global__ void
int_loop_kernel(const int nfiles, const int i, /*const int turn,*/ const int length,
		const int TerminalAU, const int ninio2,
		const cuda_param_t* __restrict__ P, const float lxc,
		const char* __restrict__ pair_, //[NBPAIRS+1][NBPAIRS+1],
		const unsigned int* __restrict__ S,    //[length+2] packed
		const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		const int* __restrict__ my_c,
		      int* __restrict__ energy_min) { //out

  int energy = INF;
  const int H = blockIdx.x;
  const int j = blockIdx.y + i+turn+1;

  assert(H >= 0 && H < nfiles);
  assert(j > (i+turn) && j <= length);

  const long long ij = Indx(i,j);
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
      const int energy2 = Energy(H,nfiles,i,j,q,p,
		    my_c,
		    S,pair_,P,
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
  if(threadIdx.x==0) energy_min[H+j*nfiles] = energy;
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
  dim3 blocks(nfiles,nblocks);
#ifndef NDEBUG
  printf("int_loop_kernel<<<[%d,%d],%d>>>(%d,i=%d,length=%d,TerminalAU=%d,ninio2=%d,...)\n",
	 nfiles,nblocks,int_loop_kernel_bs,
	 nfiles,i, /*turn,*/ length,
	 P->TerminalAU,P->ninio[2]);
  assert(int_loop_kernel_bs == BLOCK_SIZE); //assumes compiler knows block size
#endif
  int_loop_kernel<<<blocks,int_loop_kernel_bs>>>(nfiles,i, /*turn,*/ length,
					  P->TerminalAU,P->ninio[2],
					  d_param,P->lxc,
					  d_pair,
					  d_S,
					  d_hccc, 
					  d_my_c,
					  d_energy_min2); //Out

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
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

PUBLIC void
int_loop_i(const int nfiles,
	   const vrna_fold_compound_t **VC,
	   const int i, const int turn_, const int length,
	   /*const int* indx, const int ijsize,
	   const char* hard_constraints, const int* my_c,*/
	   int* energy_min ) { //out
  assert(turn_ == turn);
  if(first2) init_gpu2(nfiles,VC, turn_, length, BLOCK_SIZE);


  int_loop_cuda(nfiles,i,/*turn,*/length,VC[0]->params, energy_min);
  return;
}

extern "C"
__host__ __device__
long long Hindx(const int H, const int nfiles,
		const int i, const int j, const int length){
  //(apart from ignoring turn at ends) pack densely
  assert(H >= 0);
  assert(nfiles >= 0);
  assert(i > 0);
  assert(j >= i);
  assert(H < nfiles);
  assert(i <= length);
  assert(j <= length);
  const long long I = (i-1)*(length+1) - i*(i-1)/2 + (j-i);
  return H + I*nfiles;
}

__global__ void
int_loop_mls_kernel(const int i, const int length, const int nfiles,
	         const int* __restrict__ my_c,
		 const int en,
	         const struct energy_3p* __restrict__ E,
		 const int* __restrict__ fml_j,        /*My_fML*/
		       int* __restrict__ energy_min) { /*out (incomplete)*/

  const long long m  = blockIdx.x*blockDim.x + threadIdx.x;
  const long long mj = m / nfiles;
  const int       H  = m - mj * nfiles;
  const long long j  = mj + i+turn+1; 
  if(j < i+turn+1 || j > length) return;

  const int start = i+turn+1; 
#ifndef NDEBUG
  const int size  = length - start + 1;
  assert(size>0);
  assert(H >= 0 && H < nfiles);
#endif

      const int ii = i+1; //previous i (for i counts down)

      /* done with c[i,j], now compute fML[i,j] and fM1[i,j] */

      //my_fML[ij] = vrna_E_ml_stems_fast(vc, i, j, Fmi, DMLi);

      /*  extension with one unpaired nucleotide at the right (3' site)
	  or full branch of (i,j)
      */
      //from extend_fm_3p()...

      //WBL Aug 2026 use single assigment.
      //although not needed, avoid oddity if My_c(H,ij) == INF
      //NB ensure large array indexes are wide enough
      const long long ij      = Indx(i,j);
#ifndef NDEBUG
      const long long ijsize  = (length+1)*(length+2)/2;
      assert(ij>=0 && ij<ijsize);
#endif
      const long long cindx   = H+ij*nfiles; //d_my_c_indx
      const long long offset  = Hindx(0,nfiles,i,start,length);
      const int       Hij     = Hindx(H,nfiles,i,j,length) - offset;
      assert(cindx   >= 0 && cindx   < Hoff(nfiles,length));
      assert(Hij >= 0     && Hij     < nfiles*size); //size3
      assert(Hij == m);
      const int c       = my_c[cindx]; //My_c(H,ij == copy_d_my_c[d_my_c_indx]
      const long long indx_p1 = H+Indx(ii,j)*nfiles;
      assert(indx_p1 >= 0 && indx_p1 < Hoff(nfiles,length));
      const int fML_p1  = ((ii <= length-turn-1) && (j >= ii+turn+1 && j <= length))? fml_j[indx_p1] : INF;

      const int e00 = ((c      != INF) && (E[Hij].energy_3p_00 != INF))? c      + E[Hij].energy_3p_00 : INF;
      //en0 depends on j - 1 so done in 2nd kernel
      //end from extend_fm_3p()...

      //const int e0 = extend_fm_3p(i, j, my_fML, vc);

      const int e3 = (fML_p1 != INF)? fML_p1 + en : INF;

      const int e4 = E[Hij].energy_mls;

      const int min3 = MIN2(e00,MIN2(e3,e4)); //e1 e31.
//    } /* end of j-loop */
//
      assert(H+j*nfiles >= (i+turn+1)*nfiles && H+j*nfiles < nfiles*(length+1));
      energy_min[H+j*nfiles] = min3; //save scratch value, perhaps put in shared memory?
}
__global__ void
int_loop_mls_kernel2(const int i, const int length, const int nfiles,
		     const struct energy_3p* __restrict__ E,
		                        int* __restrict__ energy_min, /*in out*/
                                        int* __restrict__ fml_j) { /*out*/
//r1.165 Wed 19 Aug 13:03:45 BST 2026 disapointing. So:
//give each H its own block
//use all threads to put min4 into shared memory, then ripple min

  const int start = i+turn+1; 
#ifndef NDEBUG
  const int size  = length - start + 1;
  assert(size>0);
#endif
  const int H = blockIdx.x;
  
  assert(H >= 0 && H < nfiles);
  const long long offset  = Hindx(0,nfiles,i,start,length);
#ifndef NDEBUG
  const long long ijsize  = (length+1)*(length+2)/2;
#endif

  volatile __shared__ int energy_3p_en[BLOCK_SIZE2]; //limit 1024
  volatile __shared__ int min3[BLOCK_SIZE2];
  const int ix = threadIdx.x;
  assert(ix>=0 && ix < BLOCK_SIZE2);
  assert(blockDim.x == BLOCK_SIZE2);
  
/* create second H,j-loop */
  int fML_m1 = INF; //previous value, use only for thread 0

  for (int j = start+ix; j <= length; j += blockDim.x) {
    const int       Hij     = Hindx(H,nfiles,i,j,length) - offset;
    assert(Hij >= 0     && Hij     < nfiles*size); //size3
    energy_3p_en[ix] = E[Hij].energy_3p_en;
    assert(H+j*nfiles >= (i+turn+1)*nfiles && H+j*nfiles < nfiles*(length+1));
    min3[ix] = energy_min[H+j*nfiles]; //use saved scratch value from int_loop_mls_kernel
    __syncthreads();

    if(ix==0) {//ripple MIN2 so only one thread for this part of j loop
      const int j0 = j;
      for (int x = 0; x<blockDim.x && x+j0 <= length; x++) {
	const int j = x+j0;
	assert(x>=0 && x < BLOCK_SIZE2);
	const int en0  = ((fML_m1 != INF) && (energy_3p_en[x] != INF))? fML_m1 + energy_3p_en[x] : INF;
	const int min4 = MIN2(en0,min3[x]);

	const long long ij      = Indx(i,j);
	assert(ij>=0 && ij<ijsize);
	const long long indx    = H+ij*nfiles;
	assert(fml_j[indx] == INF);
	//set outputs (overwrite scratch value), save My_fML for next j value
	energy_min[H+j*nfiles] = min4;
	fml_j[indx]            = min4;
	fML_m1 = min4;
      }//endfor x
    }//endif thread 0)
    __syncthreads();
  }/* end of 2nd H,j-loop */
}
#define My_fML(H,ij)            VC[H]->matrices->fML[ij]
void
int_loop_mls(const int nfiles,
	     const vrna_fold_compound_t **VC,
	     const int i, /*const int turn,*/ const int length,
	     const long long ijsize,
	     const int* new_C,
	     const int en,
	     const struct energy_3p* energies,
	     int* energy_min) { //out also My_fML
  //in d_my_c
  assert(turn == 3);
  const int start = i+turn+1; 
  const int size  = length - start + 1;
  if(size<=0) return;

#ifdef NDEBUG
  //check here in case of earlier errors
  gpuErrchk( cudaDeviceSynchronize() );
#endif
  const long long offset = Hindx(0,nfiles,i,start,length);
  const int       size3  = nfiles*size*sizeof(energy_3p);
#ifndef NDEBUG
  printf("int_loop_mls offset %lld size3 %d bytes to GPU\n",offset,size3);
#endif
  gpuErrchk( cudaMemcpy(d_energies,&energies[offset],size3,cudaMemcpyHostToDevice) );
  
  /* Setup execution parameters for helper kernel */
  const int nblocks = (nfiles*size + int_loop_mls_kernel_bs - 1)/int_loop_mls_kernel_bs;

#ifndef NDEBUG
  printf("int_loop_mls_kernel<<<%d,%d>>>",
	 nblocks,int_loop_mls_kernel_bs);
  printf("(%d,%d,%d,d_my_c,%d,d_energies(3p_00,mls),d_dml,d_energy_min)\n",
	 i,length,nfiles,en);
#endif
  int_loop_mls_kernel<<<nblocks,int_loop_mls_kernel_bs>>>(i, /*turn,*/ length, nfiles,
						    d_my_c, en,
						    d_energies,
						    d_fml_j,       /*My_fML*/
						    d_energy_min); /*out (incomplete)*/
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  //Use second kernel to ensure all threads have finished writing min3 to d_energy_min
  
  /* Setup execution parameters for helper kernel */
  assert(int_loop_mls_kernel2_bs == BLOCK_SIZE2); //for simplicity start with fixed __shared
  //use of __shared memory sets number of blocks

#ifndef NDEBUG
  printf("int_loop_mls_kernel2<<<%d,%d>>>",
	 nfiles,int_loop_mls_kernel2_bs);
  printf("(%d,%d,%d,d_energies(energy_3p_en),d_energy_min,d_fml_j)\n",
	 i,length,nfiles);
#endif
  int_loop_mls_kernel2<<<nfiles,int_loop_mls_kernel2_bs>>>(i, /*turn,*/ length, nfiles,
						    d_energies,
						    d_energy_min, /*in out*/
						    d_fml_j);     /*out*/
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  {
  //todo optimise setting My_fML(H,ij) and energy_min
  //todo move copy to fill_arrays.c
  //fixed malloc mem_size_len might avoid heap fragmentation?
  const int mem_size_len = nfiles*(length+1) * sizeof(int); //starts at 1 not 0
  int* copy_d_energy_min = (int*) malloc(mem_size_len);
  const int start = (i+turn+1)*nfiles;
  const int len   = (length+1)*nfiles - start;
  gpuErrchk( cudaMemcpy(&copy_d_energy_min[start],&d_energy_min[start],len*sizeof(int),cudaMemcpyDeviceToHost) );
    for (int H=0;H<nfiles; H++) {
    for (int j = i+turn+1; j <= length; j++) {
      const int ij = Indx(i,j); //Avoid lookup of VC[H]->jindx
      assert(My_fML(H,ij) == INF);
      assert(H+j*nfiles < mem_size_len/sizeof(int));
      My_fML(H,ij) = energy_min[H+j*nfiles] = copy_d_energy_min[H+j*nfiles];
    }}
  free(copy_d_energy_min);
  }//end copy to CPU
}

__global__ void
int_loop_dmli_kernel(const int i, const int length, const int nfiles,
		     const int* __restrict__ energy_min,
		     const int* __restrict__ dml,     //in  d_dml   DMLi
		           int* __restrict__ fml_j,   /*out*/
		           int* __restrict__ out_fml) { /*transfer My_fML*/


  const int m  = blockIdx.x*blockDim.x + threadIdx.x;
  const int mj = m / nfiles;
  const int H  = m - mj * nfiles;
  const int j  = mj + i+turn+1; 
  if(j < i+turn+1 || j > length) return;

#ifndef NDEBUG
  const int start = i+turn+1; 
  const int size  = length - start + 1;
  assert(size>0);
  assert(H >= 0 && H < nfiles);
  //NB ensure large array indexes are wide enough
  const long long ijsize  = (length+1)*(length+2)/2;
#endif
  const long long ij      = Indx(i,j);
  assert(ij>=0 && ij<ijsize);
  const long long indx    = H+ij*nfiles;
  const int       x       = H+j*nfiles;
  assert(x >= nfiles*start);
  assert(x <  nfiles*(length+1));
  out_fml[x] = fml_j[indx] = MIN2(energy_min[x], dml[x]); //My_fML = MIN2(energy_min[H+j*nfiles],DMLi[H+j*nfiles])
}
void
int_loop_DMLi(const int nfiles,
	      const int i, /*const int turn,*/ const int length,
	      const long long ijsize,
	      const int* energy_min,
	      const int* DMLi,
	      const vrna_fold_compound_t **VC) { //out My_fML
  //in d_my_c
  assert(turn == 3);
  const int start = i+turn+1; 
  const int size  = length - start + 1;
  if(size<=0) return;

#ifdef NDEBUG
  //check here in case of earlier errors
  gpuErrchk( cudaDeviceSynchronize() );
#endif
  /* Setup execution parameters for helper kernel */
  const int nblocks = (nfiles*size + int_loop_dmli_kernel_bs - 1)/int_loop_dmli_kernel_bs;

#ifndef NDEBUG
  printf("int_loop_dmli_kernel<<<%d,%d>>>",
	 nblocks,int_loop_dmli_kernel_bs);
  printf("(%d,%d,%d,d_energy_min,d_dml,d_fml_j)\n",
	 i,length,nfiles);
#endif
  int_loop_dmli_kernel<<<nblocks,int_loop_dmli_kernel_bs>>>(i, /*turn,*/ length, nfiles,
							   d_energy_min,
							   d_dml,    //DMLi
							   d_fml_j, /*My_fML*/
							   d_out_fml); /*part of My_fML*/
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  {
  //todo optimise setting My_fML(H,ij)
  const int mem_size_len = nfiles*(length+1) * sizeof(int); //starts at 1 not 0
  int* out_fml = (int*) malloc(mem_size_len);
  const int start = (i+turn+1)*nfiles;
  const int len   = (length+1)*nfiles - start;
  //copy only updated part of My_fML
  gpuErrchk( cudaMemcpy(&out_fml[start],&d_out_fml[start],len*sizeof(int),cudaMemcpyDeviceToHost) );
    for (int H=0;H<nfiles; H++) {
    for (int j = i+turn+1; j <= length; j++) {
      const int ij   = Indx(i,j); //Avoid lookup of VC[H]->jindx
      const int indx = H+j*nfiles;
      assert(ij == (VC[H]->jindx[j]+i));
      assert(ij   >= 0 && ij   <        ijsize);
      assert(indx >= start && indx < start+len);
      My_fML(H,ij) = out_fml[indx];
    }}
  free(out_fml);
  }//end copy to CPU
}//end int_loop_DMLi
#undef My_fML
#undef turn

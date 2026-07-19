//New Jul 2026: GPU port of the hairpin-loop / multibranch-loop / 3'-extension
//energy precompute that used to live in fill_arrays.c as 4 full nfiles*ijsize
//host arrays (energy_hp, energy_mb, energy_3p_00; a 5th, energy_3p_en, is
//cheap enough to stay an inline host expression in fill_arrays_loop.c).
//
//None of the 3 computed here need a whole-triangle precompute: every read
//site in fill_arrays_loop.c consumes a value during the SAME outer i
//iteration that produces it, and none of the 3 source computations depend on
//any other row's DP state (mb_loop_fast()'s DMLi1 read is dead code on this
//fork's dangle_model==2 path -- see mb_loop_fast.c). So, mirroring
//int_loop_i()/load_my_c() in int_loop.cu, this file computes them fresh, one
//row (fixed i, all j, all H) at a time, called from fill_arrays_loop.c's main
//loop -- never a persistent nfiles*ijsize buffer on host or device.
//
//This file owns its own independent device state rather than sharing
//int_loop.cu's/modular_decomposition.cu's, matching the established
//convention that each .cu file here is self-contained.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ViennaRNA/utils.h"
#include "ViennaRNA/data_structures.h"
#include "ViennaRNA/params.h"
#include "ViennaRNA/constraints.h"

// CUDA runtime
#include <cuda_runtime.h>

#undef MIN2
#define MIN2(x,y) min(x,y)

#include "stub2.h"

#define BLOCK_SIZE 512

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, const int line, const bool abort=true)
{
   if (code != cudaSuccess)
   {
     fprintf(stderr,"CUDA error: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
     if (abort) exit(code);
   }
}

int first3 = 1; //avoid id clash with int_loop.cu's first2 / modular_decomposition.cu's first

//Only the parameter-table fields E_Hairpin/E_MLstem actually read on this
//fork's fixed dangle_model==2, cp==-1, with_ud==0, sc==NULL path (verified
//against hairpin_loops.h/multibranch_loops.h/mb_loop_fast.c/fill_arrays.c
//directly, not re-derived from memory).
typedef struct  cuda_param2_s cuda_param2_t;
struct cuda_param2_s {
  int   hairpin[31];
  int   mismatchH[NBPAIRS+1][5][5];
  int   mismatchM[NBPAIRS+1][5][5];
  int   MLintern[NBPAIRS+1];
  int   MLclosing;
  int   MLbase;
  int   rtype[8];
  int   TerminalAU;
  float lxc;
  int   special_hp; //bool, but stored as int for simple memcpy from vrna_md_t
  int   Tetraloop_E[200];
  char  Tetraloops[1401];
  int   Triloop_E[40];
  char  Triloops[241];
  int   Hexaloop_E[40];
  char  Hexaloops[1801];
};

cuda_param2_t* d_param2;
char*  d_pair2;      //[NBPAIRS+1][NBPAIRS+1], pair-type lookup, same content as int_loop.cu's d_pair
short* d_S2;          //sequence_encoding, [nfiles][length+2]
char*  d_sequence;    //raw nucleotide letters, [nfiles][length+2], for the tri/tetra/hexaloop string scan
unsigned int* d_hccc_mb;    //bit-packed VRNA_CONSTRAINT_CONTEXT_MB_LOOP per (H,ij)
unsigned int* d_hccc_mbenc; //bit-packed VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC per (H,ij)
//Kernels can only write device memory -- these are the device-side row
//outputs the kernel actually writes; hp_mb_3p_i() cudaMemcpy's them back to
//the caller's host row buffers afterward, same pattern as int_loop_cuda()'s
//d_energy_min2 in int_loop.cu.
int* d_energy_hp_row;
int* d_energy_mb_row;
int* d_energy_3p00_row;
//NB: energy_hp needs no hard-constraint bitmask at all -- E_Hairpin() has no
//hc dependency, and fill_arrays_loop.c's read site already gates on
//hc_decompose/no_close identically to how fill_arrays.c used to gate the
//write, so the two cancel out; see fill_arrays_loop.c.

#define bitsperint (8*sizeof(unsigned int))
#define Hc_ints2(length) (((length*(length+1))/2+2 + bitsperint - 1)/bitsperint)

void load_param2(const vrna_param_t *P){
  cuda_param2_t* H = (cuda_param2_t*) malloc(sizeof(cuda_param2_t));

  memcpy(H->hairpin,     P->hairpin,     31*sizeof(int));
  memcpy(H->mismatchH,   P->mismatchH,   (NBPAIRS+1)*5*5*sizeof(int));
  memcpy(H->mismatchM,   P->mismatchM,   (NBPAIRS+1)*5*5*sizeof(int));
  memcpy(H->MLintern,    P->MLintern,    (NBPAIRS+1)*sizeof(int));
  H->MLclosing  = P->MLclosing;
  H->MLbase     = P->MLbase;
  memcpy(H->rtype,       P->model_details.rtype, 8*sizeof(int));
  H->TerminalAU = P->TerminalAU;
  H->lxc        = (float)P->lxc;
  H->special_hp = P->model_details.special_hp;
  memcpy(H->Tetraloop_E, P->Tetraloop_E, 200*sizeof(int));
  memcpy(H->Tetraloops,  P->Tetraloops,  1401*sizeof(char));
  memcpy(H->Triloop_E,   P->Triloop_E,   40*sizeof(int));
  memcpy(H->Triloops,    P->Triloops,    241*sizeof(char));
  memcpy(H->Hexaloop_E,  P->Hexaloop_E,  40*sizeof(int));
  memcpy(H->Hexaloops,   P->Hexaloops,   1801*sizeof(char));

  gpuErrchk( cudaMemcpy(d_param2,H,sizeof(cuda_param2_t),cudaMemcpyHostToDevice) );
  free(H);
}

PUBLIC void
init_gpu3(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size) {
  if(!first3) return;
  fprintf(stderr,"%-24s init_gpu3(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

  gpuErrchk( cudaMalloc((void **) &d_param2, sizeof(cuda_param2_t)) );
  load_param2(VC[0]->params);

  char pair_[NBPAIRS+1][NBPAIRS+1];
  for(int x=0;x<21;x++){
  for(int y=0;y<21;y++) {
    const vrna_md_t *md = &(VC[0]->params->model_details);
    if(x < NBPAIRS+1 && y < NBPAIRS+1) pair_[x][y] = md->pair[x][y];
  }}
  size_t size = (NBPAIRS+1)*(NBPAIRS+1)*sizeof(char);
  gpuErrchk( cudaMalloc((void **) &d_pair2, size) );
  gpuErrchk( cudaMemcpy(d_pair2,pair_,size,cudaMemcpyHostToDevice) );

  //VRNA_CONSTRAINT_CONTEXT_MB_LOOP / _ENC bitmasks, same packing technique as
  //int_loop.cu's d_hccc (which only packs the INT_LOOP_ENC bit) -- see
  //init_gpu2() there. hc->matrix packs several independent context bits into
  //the same byte (constraints_hard.h), so this is just extracting two more.
  size = (size_t)nfiles*Hc_ints2(length)*sizeof(unsigned int);
  gpuErrchk( cudaMalloc((void **) &d_hccc_mb,    size) );
  gpuErrchk( cudaMalloc((void **) &d_hccc_mbenc, size) );
  unsigned int* hccc_mb    = (unsigned int*) calloc((size_t)nfiles*Hc_ints2(length),sizeof(unsigned int));
  unsigned int* hccc_mbenc = (unsigned int*) calloc((size_t)nfiles*Hc_ints2(length),sizeof(unsigned int));
  for(int H=0;H<nfiles;H++){
    unsigned int mask_mb, mask_mbenc;
    for(int i=0;i<(length*(length+1))/2+2;i++){
      mask_mb    = ((i & 0x1f) == 0)? 1 : mask_mb    << 1;
      mask_mbenc = ((i & 0x1f) == 0)? 1 : mask_mbenc << 1;
      const int I = H*Hc_ints2(length)+i/bitsperint;
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP)     hccc_mb[I]    |= mask_mb;
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC) hccc_mbenc[I] |= mask_mbenc;
    }
  }
  gpuErrchk( cudaMemcpy(d_hccc_mb,   hccc_mb,   (size_t)nfiles*Hc_ints2(length)*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_hccc_mbenc,hccc_mbenc,(size_t)nfiles*Hc_ints2(length)*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  free(hccc_mb);
  free(hccc_mbenc);

  size = (size_t)nfiles*(length+2)*sizeof(short);
  gpuErrchk( cudaMalloc((void **) &d_S2, size) );
  short* Sbuff = (short*) malloc(size);
  for(int H=0;H<nfiles;H++) memcpy(&Sbuff[H*(length+2)],VC[H]->sequence_encoding,(length+2)*sizeof(short));
  gpuErrchk( cudaMemcpy(d_S2,Sbuff,size,cudaMemcpyHostToDevice) );
  free(Sbuff);

  size = (size_t)nfiles*(length+2)*sizeof(char);
  gpuErrchk( cudaMalloc((void **) &d_sequence, size) );
  char* seqbuff = (char*) calloc((size_t)nfiles*(length+2),sizeof(char));
  for(int H=0;H<nfiles;H++) {
    const size_t len = strlen(VC[H]->sequence);
    memcpy(&seqbuff[H*(length+2)],VC[H]->sequence,len);
  }
  gpuErrchk( cudaMemcpy(d_sequence,seqbuff,size,cudaMemcpyHostToDevice) );
  free(seqbuff);

  size = (size_t)nfiles*(length+1)*sizeof(int);
  gpuErrchk( cudaMalloc((void **) &d_energy_hp_row,   size) );
  gpuErrchk( cudaMalloc((void **) &d_energy_mb_row,   size) );
  gpuErrchk( cudaMalloc((void **) &d_energy_3p00_row, size) );

  first3 = 0;
}

__device__ inline unsigned char
Ptype2(const short* __restrict__ S, const char* __restrict__ pair, const int i, const int j) {
  return pair[S[i]*8 + S[j]];
}

__device__ inline int
Indx2(const int i, const int j) {
  return j*(j-1)/2+i;
}

//emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP(_ENC) -- same technique as
//int_loop.cu's Hc(), against whichever of the two bitmasks is passed in
__device__ inline int
Hc2(const int ij, const unsigned int* __restrict__ hccc){
  const int I = ij/bitsperint;
  const unsigned int m = hccc[I];
  return (m >> (ij - I*bitsperint)) & 1;
}

//Replicates E_Hairpin(), hairpin_loops.h:103-145, exactly -- including the
//easy-to-get-backwards asymmetry where a triloop (size==3) miss returns
//early, *skipping* the mismatchH add below, while tetraloop/hexaloop misses
//fall through *to* it. `seq` points at the same offset E_Hairpin's host
//caller passes (vc->sequence+i-1) -- device equivalent is &d_sequence[H*(length+2)+i-1].
__device__ inline int
E_Hairpin_device(const int size, const int type, const int si1, const int sj1,
                  const char* __restrict__ seq, const cuda_param2_t* __restrict__ P) {
  int energy;

  if(size <= 30) energy = P->hairpin[size];
  else            energy = P->hairpin[30] + (int)(P->lxc*log((double)size/30.)); //double log() to match host precision (hairpin_loops.h:116)

  if(size < 3) return energy; /* should only be the case when folding alignments */

  if(P->special_hp){
    if(size == 4){
      char tl[7] = {0};
      for(int k=0;k<6;k++) tl[k] = seq[k];
      for(int off=0; off+7<=1401; off+=7){
        if(P->Tetraloops[off]==0) break; //end of populated entries
        int match=1;
        for(int k=0;k<6;k++) if(P->Tetraloops[off+k]!=tl[k]) { match=0; break; }
        if(match) return P->Tetraloop_E[off/7];
      }
    }
    else if(size == 6){
      char tl[9] = {0};
      for(int k=0;k<8;k++) tl[k] = seq[k];
      for(int off=0; off+9<=1801; off+=9){
        if(P->Hexaloops[off]==0) break;
        int match=1;
        for(int k=0;k<8;k++) if(P->Hexaloops[off+k]!=tl[k]) { match=0; break; }
        if(match) return P->Hexaloop_E[off/9];
      }
    }
    else if(size == 3){
      char tl[6] = {0,0,0,0,0,0};
      for(int k=0;k<5;k++) tl[k] = seq[k];
      for(int off=0; off+6<=241; off+=6){
        if(P->Triloops[off]==0) break;
        int match=1;
        for(int k=0;k<5;k++) if(P->Triloops[off+k]!=tl[k]) { match=0; break; }
        if(match) return P->Triloop_E[off/6];
      }
      return energy + (type>2 ? P->TerminalAU : 0);
    }
  }
  energy += P->mismatchH[type][si1][sj1];
  return energy;
}

//Replicates E_MLstem(), multibranch_loops.h:168-186. si1/sj1 are always >=0
//on every call site reachable from energy_mb/energy_3p_00 in this fork
//(cp==-1 enforced -- verified against mb_loop_fast.c/fill_arrays.c), so the
//dangle5/dangle3 branches are dead and deliberately not ported.
__device__ inline int
E_MLstem_device(const int type, const int si1, const int sj1, const cuda_param2_t* __restrict__ P) {
  int energy = P->mismatchM[type][si1][sj1];
  if(type > 2) energy += P->TerminalAU;
  energy += P->MLintern[type];
  return energy;
}

__global__ void
hp_mb_3p_kernel(const int i, const int turn, const int length,
                 const short* __restrict__ S,
                 const char*  __restrict__ seq,
                 const char*  __restrict__ pair,
                 const unsigned int* __restrict__ hccc_mb,
                 const unsigned int* __restrict__ hccc_mbenc,
                 const cuda_param2_t* __restrict__ P,
                       int* __restrict__ energy_hp_row,
                       int* __restrict__ energy_mb_row,
                       int* __restrict__ energy_3p00_row) {
  const size_t H = blockIdx.y;
  const int m = blockIdx.x*blockDim.x+threadIdx.x;
  const int j = m + i+turn+1;
  if(j>length) return;

  const short* S_H   = &S[H*(length+2)];
  const char*  seq_H = &seq[H*(length+2)];
  const int ij = Indx2(i,j);
  const unsigned int* Hccc_mb    = &hccc_mb[H*Hc_ints2(length)];
  const unsigned int* Hccc_mbenc = &hccc_mbenc[H*Hc_ints2(length)];

  int type = (int)Ptype2(S_H,pair,i,j);
  if(type == 0) type = 7;

  //energy_hp: vrna_E_hp_loop()/vrna_eval_hp_loop() do no hc check of their
  //own (hairpin_loops.c:226-293) -- fill_arrays_loop.c's read site already
  //gates on hc_decompose/no_close identically to how fill_arrays.c used to
  //gate the write, so compute unconditionally here (see file header comment).
  {
    const int u = j-i-1;
    energy_hp_row[H*(length+1)+j] =
      E_Hairpin_device(u, type, S_H[i+1], S_H[j-1], &seq_H[i-1], P);
  }

  //energy_mb: mb_loop_fast.c:92-148 (dangle_model==2, cp==-1, sc==NULL path)
  //-- decomp starts at 0 (NOT INF), only overwritten if the MB_LOOP bit is set.
  {
    int decomp = 0;
    if(Hc2(ij,Hccc_mb)){
      int tt = P->rtype[type];
      if(tt == 0) tt = 7;
      decomp = E_MLstem_device(tt, S_H[j-1], S_H[i+1], P) + P->MLclosing;
    }
    energy_mb_row[H*(length+1)+j] = decomp;
  }

  //energy_3p_00: inlined extend_fm_3p() fragment, fill_arrays.c:551-565
  //(cp==-1 forced, so ON_SAME_STRAND(...) is always true) -- default INF,
  //overwritten only if the MB_LOOP_ENC bit is set.
  {
    int e00 = INF;
    if(Hc2(ij,Hccc_mbenc)){
      const short s_i1 = (i==1) ? S_H[length] : S_H[i-1];
      e00 = E_MLstem_device(type, s_i1, S_H[j+1], P);
    }
    energy_3p00_row[H*(length+1)+j] = e00;
  }
}

PUBLIC void
hp_mb_3p_i(const int nfiles, const vrna_fold_compound_t **VC,
           const int i, const int turn, const int length,
           int* energy_hp_row, int* energy_mb_row, int* energy_3p00_row) { //all out, size nfiles*(length+1)
  const int start = i+turn+1;
  const int size  = length - start + 1;
  //size<=0 is unreachable for the i range fill_arrays_loop.c's main loop
  //actually uses (start=i+turn+1 <= length always holds there) -- same dead
  //guard load_fML() carries for the identical reason, kept for symmetry.
  if(size<=0) return;

  const int nblocks = (size + BLOCK_SIZE - 1)/BLOCK_SIZE;
  dim3 blocks(nblocks,nfiles);
  hp_mb_3p_kernel<<<blocks,BLOCK_SIZE>>>(i, turn, length,
                                          d_S2, d_sequence, d_pair2,
                                          d_hccc_mb, d_hccc_mbenc, d_param2,
                                          d_energy_hp_row, d_energy_mb_row, d_energy_3p00_row);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  const size_t rowsize = (size_t)nfiles*(length+1)*sizeof(int);
  gpuErrchk( cudaMemcpy(energy_hp_row,  d_energy_hp_row,  rowsize,cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(energy_mb_row,  d_energy_mb_row,  rowsize,cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(energy_3p00_row,d_energy_3p00_row,rowsize,cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaDeviceSynchronize() );
}

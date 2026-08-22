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
unsigned int* d_S;    //S[length+2] packed 10 bases (3 bits each) per word
int*          d_my_c;
int*          d_energy_min2; //share with modular_decomposition.cu ?
int*          d_new_e;
// Staggered_Row_Batching Phase 2b: device copy of compute_batch_offsets()'s
// tri_off_H[] (mfe_cuda.c/stub2.h) -- d_my_c's per-H triangle-block start,
// nfiles+1 entries, uploaded once per chunk in init_gpu2(). Replaces
// Hoff(H,length) wherever d_my_c is indexed on-device: Hoff() assumes every
// H shares one length, tri_off_H[] doesn't.
static size_t*       d_tri_off_H;
// Staggered_Row_Batching Phase 2c: device copy of row_off_H[] -- d_new_e's
// per-H row start, replacing H*(length+1) wherever d_new_e is indexed.
static size_t*       d_row_off_H;
// Staggered_Row_Batching Phase 6d: row_off_H[nfiles], the true total extent of
// every row-shaped buffer in this file. Cached at init_gpu2() time because the
// per-row entry points (load_my_c(), int_loop_cuda()) transfer whole row
// buffers but never receive the offset table itself. Equals the old
// nfiles*(length+1) exactly while chunks stay uniform-length; once they don't,
// that formula over-reads/over-writes past the real allocation.
static size_t        g_row_total = 0;
// Staggered_Row_Batching Phase 2e: d_hccc's per-H block start. Computed
// locally in init_gpu2() (not compute_batch_offsets()) since Hc_ints()'s
// MAXLOOP padding is a private detail of this file's bit-packing, not a
// general row/triangle shape shared elsewhere.
static size_t*       d_hc_off_H;
// Staggered_Row_Batching Phase 5: per-row block-count table shared by
// int_loop_kernel and load_my_c_kernel (both this file, same "size"
// formula reused verbatim from Phase 4's load_fML) -- allocated once per
// chunk here, uploaded fresh by whichever of int_loop_cuda()/load_my_c()
// runs first each row (each uploads independently rather than assuming
// the other already did, since -- unlike Phase 4's graph-captured
// load_fML/modular_decomposition/load_min_fML sequence -- these two are
// separate synchronous launches with no fixed relative ordering guarantee
// worth depending on).
static size_t*       d_size_off_H;
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
  //length-equality assert removed (Staggered_Row_Batching Phase 1): mixed-length
  //batches are a real target now, not a bug -- the other 8 checks below still
  //guard the single shared d_param/d_pair GPU buffers, which genuinely must
  //match across every H in a batch regardless of per-H length.
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

// Pack ten sequence bases (each 0..4, so 3 bits suffices) per 32-bit word,
// H fastest-varying -- word for host position i, files H0..H0+9 -- so that
// unpack() below is called with adjacent-H threads reading the same word.
// Per Dr. Langdon's Aug 2026 main-branch work (54b7c31): was 1 base per
// short (2 bytes) before.
void put10(const unsigned int word, const int H, const int nfiles, const int i, const int size, unsigned int* out){
  assert(word <= 04444444444); //max legit value in octal (ten 3-bit fields)
  const int I = (H + nfiles*i)/10;
  assert(I>=0 && I < size);
  assert(out[I] == 0xffffffff);
  out[I] = word;
}

PUBLIC void
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size,
          const size_t* tri_off_H, const size_t* row_off_H) { //in, nfiles+1 entries each, mfe_cuda.c
  if(!first2) return;
  const double _t_ig2 = rnafold_now_seconds();
  fprintf(stderr,"%-24s init_gpu2(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

  assert(turn_ == turn);
  assert(MAX_NINIO == 300); //ViennaRNA/energy_par.c

  TIMED_CUDAMALLOC(&d_tri_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_tri_off_H, tri_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
  TIMED_CUDAMALLOC(&d_row_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_row_off_H, row_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
  //printf("%s %s d_param is %lu bytes, NBPAIRS %d MAXLOOP %d BLOCK_SIZE %d\n",
  //	 __FILE__,Version,sizeof(cuda_param_s),NBPAIRS,MAXLOOP,block_size);

  // d_param/d_pair are nfiles/length-independent -- guarded on their own
  // one-time check (d_param starts NULL, a zero-initialized global) rather
  // than on first2, so teardown_gpu2() can reset first2=1 between GPU
  // batches without this block re-allocating (and leaking) them every batch.
  if(!d_param) {
    TIMED_CUDAMALLOC(&d_param, sizeof(cuda_param_s));
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
    TIMED_CUDAMALLOC(&d_pair, pair_size);
    gpuErrchk( cudaMemcpy(d_pair,pair_,pair_size,cudaMemcpyHostToDevice) );
  }

  // Staggered_Row_Batching Phase 2e: per-H table (own shape -- Hc_ints()'s
  // padding is this file's own private constant, not row_off_H/tri_off_H's
  // shape) built from each H's real length, table-driven instead of a
  // uniform Hc_ints(length) multiply.
  size_t hc_off_H[nfiles+1];
  hc_off_H[0] = 0;
  for(int H=0;H<nfiles;H++) hc_off_H[H+1] = hc_off_H[H] + Hc_ints(VC[H]->length);
  TIMED_CUDAMALLOC(&d_hc_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_hc_off_H, hc_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  // Staggered_Row_Batching Phase 5: allocated here (fixed size for the whole
  // chunk), not populated here -- this changes every sweep row i, uploaded
  // fresh per-row by int_loop_cuda()/load_my_c() instead.
  TIMED_CUDAMALLOC(&d_size_off_H, (size_t)(nfiles+1)*sizeof(size_t));

  size_t size = hc_off_H[nfiles]*sizeof(unsigned int);
  TIMED_CUDAMALLOC(&d_hccc, size);
  unsigned int* hccc   = (unsigned int*) calloc(hc_off_H[nfiles],sizeof(unsigned int));
  const double _t_pk1 = rnafold_now_seconds();
  for(int H=0;H<nfiles;H++){
    assert(bitsperint==(1+0x1f));
    unsigned int mask;
    // Staggered_Row_Batching Phase 6a: bounded by this H's own length, not
    // the shared `length` scalar -- VC[H]->hc->matrix is only ever sized to
    // VC[H]->length, so using the shared length here would read past a
    // shorter H's real allocation the moment `length` stops meaning "every
    // H's length" (i.e. once mixed lengths actually reach this function).
    // hccc[] itself is calloc'd, so the untouched tail for a shorter H
    // (both this triangle's own remaining slots and Hc_ints()'s MAXLOOP
    // padding) stays correctly zero either way.
    const int length_H = (int)VC[H]->length;
    for(int i=0;i<(length_H*(length_H+1))/2+2;i++){ //leave padding as zero
      mask = ((i & 0x1f) == 0)? 1 : mask << 1;
      const long long I = (long long)hc_off_H[H]+i/bitsperint; // Langdon's 2026 indexing bug -- host-side hccc population, missed by 2f35ecc's kernel-scoped fix
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC) hccc[I] |= mask;
    }
  }
  stage_ig_pack_s += rnafold_now_seconds() - _t_pk1;
  gpuErrchk( cudaMemcpy(d_hccc,hccc,hc_off_H[nfiles]*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  free(hccc);

  // Ten bases per word, H fastest index (see put10()/unpack()).
  assert(sizeof(unsigned int) == 4);
  size = ((size_t)nfiles * (length+2) + 9)/10 * sizeof(unsigned int);
  TIMED_CUDAMALLOC(&d_S, size);
  unsigned int* buff = (unsigned int*) malloc(size); //could use cudaMallocHost
  const double _t_pk2 = rnafold_now_seconds();
#ifndef NDEBUG
  memset(buff,0xff,size);
#endif
  {
    const int len = (length+2);
    int H0 = 0;
    int i0 = 0;
    int j = 0; //0 to 9
    unsigned int word = 0;
    for(int i=0;i<len;i++){
    for(int H=0;H<nfiles;H++){
      if(j==0) {word = 0; H0 = H; i0 = i;}
      // Staggered_Row_Batching Phase 6a: guard against reading past a
      // shorter H's own sequence_encoding (sized to VC[H]->length+2) once
      // the shared `length`/`len` above can exceed an individual H's real
      // length -- substitutes 0 (safe: this file's own d_S 10-per-word
      // repacking for genuinely mixed lengths is deliberately deferred,
      // Phase 2f, so this is a minimal safety guard only, not a fix for
      // that packing scheme itself. Phase 6d's active/join mask ensures a
      // position this far past H's own length is never actually consumed
      // in a real energy calculation regardless.)
      const unsigned int s = (i <= (int)VC[H]->length+1) ? VC[H]->sequence_encoding[i] : 0;
      assert(s <= 4);
      assert(j >= 0 && j < 10);
      word = word | (s << (j*3));
      j++;
      if(j >= 10) {
        j=0; put10(word,H, nfiles,i, size/4,buff);
      }
    }}
    if(j>0) put10(word,H0,nfiles,i0,size/4,buff);
  }
#ifndef NDEBUG
  for(size_t i=0;i<size/4;i++) assert(buff[i] <= 04444444444);
#endif
  stage_ig_pack_s += rnafold_now_seconds() - _t_pk2;
  gpuErrchk( cudaMemcpy(d_S,buff,size,cudaMemcpyHostToDevice) );
  free(buff);

  { const size_t my_c_elems = tri_off_H[nfiles]; //sum of each H's own triangle size
    size = my_c_elems*sizeof(int);
    TIMED_CUDAMALLOC(&d_my_c, size);
    init_my_c(my_c_elems);
  }

  // Staggered_Row_Batching Phase 6d: cached for the per-row entry points that
  // transfer these buffers whole but never see the offset table.
  g_row_total = row_off_H[nfiles];

  size = g_row_total*sizeof(int); // 32-bit signed integer overflow bug fix
  TIMED_CUDAMALLOC(&d_new_e, size);

  // Staggered_Row_Batching Phase 2d: table-driven total (row_off_H[nfiles]),
  // matching int_loop_kernel_body.inc's row_off_H[H]+j write below -- equals
  // the old uniform nfiles*(length+1) exactly while chunks stay uniform-length,
  // diverges once they don't.
  TIMED_CUDAMALLOC(&d_energy_min2, g_row_total*sizeof(int));
  /*no longer in use
  TIMED_CUDAMALLOC(&d_energy_min20, size);

  size = nfiles*length*sizeof(int);
  TIMED_CUDAMALLOC(&d_buf, size);
  */
  stage_ig2_s += rnafold_now_seconds() - _t_ig2;
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
  gpuErrchk( cudaFree(d_tri_off_H) );
  gpuErrchk( cudaFree(d_row_off_H) );
  gpuErrchk( cudaFree(d_hc_off_H) );
  gpuErrchk( cudaFree(d_size_off_H) );
  first2 = 1;
}

// Bytes of device memory this file needs for one additional sequence at the
// given length -- d_hccc/d_S/d_my_c/d_new_e/d_energy_min2, mirroring
// init_gpu2()'s own size formulas exactly. d_param/d_pair are excluded --
// fixed-size, paid once regardless of batch count.
PUBLIC size_t
int_loop_bytes_per_file(const int length) {
  const size_t hccc_bytes         = Hc_ints(length)*sizeof(unsigned int);
  // Ten bases packed per word now (put10()/unpack() in init_gpu2()) --
  // marginal cost per file is ~1/10th of one unsigned int per base, rounded
  // up to stay a conservative (over-, not under-) estimate.
  const size_t s_bytes            = ((size_t)(length+2)*sizeof(unsigned int) + 9)/10;
  const size_t my_c_bytes         = (size_t)(length+1)*(length+2)/2*sizeof(int);
  const size_t new_e_bytes        = (size_t)(length+1)*sizeof(int);
  const size_t energy_min2_bytes  = (size_t)(length+1)*sizeof(int);
  return hccc_bytes + s_bytes + my_c_bytes + new_e_bytes + energy_min2_bytes;
}

// Copies the GPU's my_c triangle back into each record's own
// VC[H]->matrices->c, once, after the whole sweep -- the my_c twin of
// modular_decomposition.cu's fetch_fML(), and the same reasoning: d_my_c is an
// exact mirror (init_my_c() fills it with INF, load_my_c_kernel writes
// d_my_c[tri_off_H[H]+Indx(i,j)] = new_C[row_off_H[H]+j], which is precisely
// what the host store used to write), the layouts coincide
// (jindx[j]+i == Indx(i,j), allocation (n+1)*(n+2)/2 == the tri_off_H stride),
// and nothing on the host reads the triangle until E_ext_loop_5()/backtrack().
// One record's slice -- the my_c twin of fetch_fML_one(); see there.
extern "C" /*PUBLIC*/ void
fetch_my_c_one(int* dst, const size_t tri_lo, const size_t cells) {
  gpuErrchk( cudaMemcpy(dst, &d_my_c[tri_lo], cells*sizeof(int),
                        cudaMemcpyDeviceToHost) );
}

extern "C" /*PUBLIC*/ void
fetch_my_c(const int nfiles, int** c_H, const size_t* tri_off_H) {
  for(int H=0; H<nfiles; H++) {
    const size_t n = tri_off_H[H+1] - tri_off_H[H];
    assert(n > 0);
    gpuErrchk( cudaMemcpy(c_H[H], &d_my_c[tri_off_H[H]], n*sizeof(int),
                          cudaMemcpyDeviceToHost) );
  }
}

//perhaps this can be combined with other kernels?
__global__ void
load_my_c_kernel(const int nfiles, const int i, /*const int turn,*/ const int length,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c,
		 const size_t* __restrict__ tri_off_H, //in
		 const size_t* __restrict__ row_off_H, //in
		 const size_t* __restrict__ size_off_H, const size_t total) { //in
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const long long j  = mj + i+turn+1;

  const long long ij = Indx(i,j);
  // Staggered_Row_Batching Phase 6d: bound the check by THIS H's own triangle
  // extent. `length` is now max(VC[H]->length) across the batch, so
  // Hoff(1,length) would let a short H's ij run past its own block undetected
  // -- the check would still pass while the write below silently landed in the
  // next H's triangle. Identical to Hoff(1,length) while lengths are uniform.
  assert(ij>=0 && (size_t)ij < tri_off_H[H+1]-tri_off_H[H]);
  assert(my_c[tri_off_H[H]+ij] == INF);
         my_c[tri_off_H[H]+ij] = new_e[row_off_H[H]+j];
}

PUBLIC void
load_my_c(const int nfiles,
	  const int i, const int turn_, const int length,
	  const int* new_e,
	  const size_t* size_off_H) {   //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
  //out d_my_c
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

#ifdef NDEBUG
  //check here in case of earlier errors
  gpuErrchk( cudaDeviceSynchronize() );
#endif
  //for simplicity transfer all new_e, even though only need H * [start:length]
  gpuErrchk( cudaMemcpy(d_new_e,new_e,g_row_total*sizeof(int),cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_size_off_H, size_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );


  /* Setup execution parameters for helper kernel */
  // Block size picked once from the actual GPU present (see stub2.h's
  // rnafold_choose_block_size()) instead of the BLOCK_SIZE constant this
  // used to hardcode -- BLOCK_SIZE=512 was tuned against one GPU (the L4);
  // this kernel has no shared memory or reduction tying it to a specific
  // size, so there's no reason not to let CUDA pick per-device.
  static int block_size = 0;
  if(!block_size) {
    block_size = rnafold_choose_block_size(load_my_c_kernel, BLOCK_SIZE, "RNA_LOAD_MY_C_BLOCK_SIZE");
    fprintf(stderr,"%-24s load_my_c_kernel block size %d (was hardcoded %d)\n",
	    __FILE__, block_size, BLOCK_SIZE);
  }
  const int nblocks = (total + block_size - 1)/block_size;

  load_my_c_kernel<<<nblocks,block_size>>>(nfiles, i, /*turn,*/ length,
					   d_new_e,  //in
					   d_my_c,   //out
					   d_tri_off_H,  //in
					   d_row_off_H,  //in
					   d_size_off_H, total);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

// Unpack one base (0..4) from d_S's ten-per-word, H-fastest packing -- see
// put10() in init_gpu2().
__device__ inline
int unpack(const unsigned int* S, const int H, const int nfiles, const int i){
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
Ptype(const unsigned int* __restrict__ S, const char* __restrict__ pair,//[8][8],
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

//Indx()/Hoff() now shared via stub2.h (see "Langdon's 2026 indexing bug")

#undef BLOCK_SIZE
//Like modular_decomposition.cu have one block per j value
//each block has (MAXLOOP+1)*(MAXLOOP+2)/2 worker threads
//present reduction code needs BLOCK_SIZE to be at least 32 and a power of 2
//
// BLOCK_SIZE=32 below is now just the fallback/default -- Hc()/
// decode_column()/Energy() above are shared by every instantiation
// regardless of block size.
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

#include "nth.h"

// Cooperative-scan design (replaces the old setpq(), removed here --
// recoverable from git history, commit 50cfa8a, if wanted for reference).
// setpq() gave every thread its own *private*, incremental walk through the
// candidate bitmask -- correct, but redundant: since MAXLOOP=30 bounds this
// per-(i,j)-cell search space to at most 31 columns, and thread ranks are
// handed round-robin across that space, nearly every thread ended up
// re-walking almost the whole thing independently, so aggregate scanning
// work scaled with BLOCK_SIZE instead of staying flat (this is what made
// BLOCK_SIZE selection for int_loop_kernel actively unsafe -- see the
// STOPGAP comment below and the "Cooperative Column Scan" design doc).
//
// decode_column() is the same per-column bit-extraction setpq() used to do
// lazily/incrementally, but now called cooperatively: int_loop_kernel_body
// .inc has up to MAXLOOP+1 (<=31, always within one warp) threads each
// decode exactly one column, once, into shared memory, then builds a
// shared prefix-sum-of-popcounts table. Every thread's rank lookup then
// becomes stateless -- binary-search-free linear scan over that small
// shared table (<=31 entries) to find its column, then a single
// find_nth_set_bit() call -- with no per-thread incremental state (mask/
// row_start/done) left to carry between iterations at all.
__device__ inline
unsigned int decode_column(const int p0, const int q0, const int column,
			    const unsigned int* __restrict__ hccc) {
  assert(bitsperint==32);
  const int mask_size = column + 1;
  assert(mask_size>0 && mask_size<=bitsperint+1);
  const int pq = Indx(p0, q0+column);
  const int I  = pq/bitsperint;
  const int x  = pq - I*bitsperint;
  unsigned int mask = hccc[I];
  mask = mask >> x; //remove bits below pq
  if(mask_size+x > 32) { //get top bits
    unsigned int m2 = hccc[I+1];
    m2 = m2 & (~((~0) << (mask_size+x-32))); //clear bits above mask_size
    m2 = m2 << (32-x);                       //avoid over writing lower bits in mask already in use
    mask = mask | m2;                        //splice two parts of column mask together
  } else {
    mask = mask & (~((~0)<< mask_size));     //clear bits above mask_size
  }
  return mask;
}

//interface to interior_loopx.h via IntLoop_X()
__device__ inline int
Energy(const int H, const int nfiles, const int i, const int j, const int q, const int p,
	  /*const char* hard_constraints,*/ const int* my_c,
	  /*const int* hc_up, const char* hc, const unsigned int* __restrict__ hccc,*/
	  const unsigned int* __restrict__ S, const char* __restrict__ pair_,//[NBPAIRS+1][NBPAIRS+1],
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
	      const unsigned char type   = Ptype(S,pair_,H,nfiles,i,j);
	      //assert(type == Ptype(S,pair_,H,nfiles,i,j));
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
	      const int sq1 = unpack(S,H,nfiles,q+1);

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

/*Removed 5 Aug 2026 (per Dr. Langdon's Aug 2026 main-branch cleanup, r1.126):
  int_loop_nl0_kernel, int_loop_ns0_kernel, int_loop_1xn_kernel,
  int_loop_int11_kernel, int_loop_int21_kernel, int_loop_int22_kernel,
  int_loop_nl3_kernel, int_loop_I_kernel, int_loop_I1_kernel,
  int_loop_min_kernel, int_loop_min_kernel2 -- each a divergence-avoiding
  path through IntLoop_X(), superseded by int_loop_kernel below and never
  called. Restore from git history (2f35ecc or earlier) if needed.*/

// Four instantiations of int_loop_kernel, one per candidate block size --
// see int_loop_kernel_body.inc for why this is a repeated #include rather
// than a single definition or a C++ template.
#define BLOCK_SIZE 32
#include "int_loop_kernel_body.inc"
#undef BLOCK_SIZE

#define BLOCK_SIZE 64
#include "int_loop_kernel_body.inc"
#undef BLOCK_SIZE

#define BLOCK_SIZE 128
#include "int_loop_kernel_body.inc"
#undef BLOCK_SIZE

#define BLOCK_SIZE 256
#include "int_loop_kernel_body.inc"
// leave BLOCK_SIZE defined as the fallback/default (32) used below
#define BLOCK_SIZE 32

// Block-size history: int_loop_choose_block_size() (timed microbenchmark,
// tried the occupancy API before that) was removed after both proved
// actively harmful -- they only ever sampled the kernel's first
// (always-tiny, nblocks=1) launch, so they couldn't see that aggregate
// scanning work in the old setpq()-based search scaled with BLOCK_SIZE
// instead of staying flat (confirmed via NCU: the timed benchmark picked
// BLOCK_SIZE=256, and total measured GPU time came out 2.3-2.9x worse than
// plain BLOCK_SIZE=32). That per-thread incremental scan is gone now --
// setpq() was replaced by decode_column() plus the cooperative
// decode/prefix-sum/lookup in int_loop_kernel_body.inc (see the
// "Cooperative Column Scan" design doc).
//
// Re-measured against real hardware 2026-08-20 (local RTX 3050, CC 8.6, via
// ncu --set basic, RNA_INT_LOOP_BLOCK_SIZE=32 vs 256, sampled across grid
// sizes from ~40 blocks up to ~54000): BLOCK_SIZE=32 remains the better
// choice, but for a *different* reason than the old setpq() liability above
// -- that liability really is gone (aggregate work no longer scales with
// BLOCK_SIZE), and occupancy at BLOCK_SIZE=256 is indeed dramatically
// higher (36-62% vs 5-33% depending on grid size) exactly as the old
// reasoning here predicted. But higher occupancy didn't translate into
// less time: BLOCK_SIZE=256 was slower at every grid size tested, from
// 1.07x slower at small grids up to 2.35x slower at grid~500, narrowing
// back to ~1.09x slower even at grid~54000 (a genuinely large batch of
// long sequences) -- never faster, at any scale tried. The likely cause:
// each (i,j) cell's interior-loop search space is inherently small
// (bounded by MAXLOOP=30), so the cooperative decode/prefix-sum/lookup
// below only ever needs one warp's worth of work regardless of BLOCK_SIZE
// -- a bigger block just adds synchronization/scheduling overhead (the
// extra warp_min combine + __syncthreads() under #if BLOCK_SIZE > 32) for
// threads that have nothing to do. Real occupancy gains for this kernel
// come from more *blocks* in flight (bigger batches, more concurrent
// (H,j) cells -- exactly what staggering/mixed-length batching is for),
// not from bigger blocks. See RNA_INT_LOOP_BLOCK_SIZE just below to
// re-test 64/128/256 directly if this ever needs re-checking on different
// hardware, without reintroducing an in-process benchmark.

//Host (ie non-GPU) code
PRIVATE void
int_loop_cuda(const int nfiles,
	      const int i, /*const int turn,*/ const int length,
	      const vrna_param_t *P,
	      int* energy_min,
	      const size_t* size_off_H) { //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
  //cf modular_decomposition.cu r1.79
  // Staggered_Row_Batching Phase 5: size_off_H[nfiles] replaces the old
  // scalar nblocks<=0 check -- numerically identical while every H shares
  // one length (today). Flat 1-D grid instead of dim3(nblocks,nfiles) --
  // also lifts the old implicit nfiles<=65535 sub-limit from using nfiles
  // as gridDim.y (gridDim.x supports far more).
  const size_t flat_nblocks = size_off_H[nfiles];
  if(flat_nblocks==0) return;

  gpuErrchk( cudaMemcpy(d_size_off_H, size_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  dim3 blocks((unsigned int)flat_nblocks);

  // Default is still the conservative STOPGAP (BLOCK_SIZE==32, hardcoded
  // not auto-tuned) -- see the comment above this kernel's four #include
  // instantiations. RNA_INT_LOOP_BLOCK_SIZE lets a live run force any of
  // the four candidates directly (32/64/128/256) to measure the
  // cooperative-scan rewrite against real hardware, same pattern as this
  // file's other RNA_*-prefixed env knobs (e.g. RNA_CUDA_GRAPH in
  // modular_decomposition.cu) -- deliberately *not* an in-process
  // benchmark picking automatically, since that's exactly the mechanism
  // that caused the last two regressions.
  static int block_size = 0;
  if(!block_size) {
    block_size = BLOCK_SIZE; // == 32, the STOPGAP default
    const char* env = getenv("RNA_INT_LOOP_BLOCK_SIZE");
    if(env) {
      const int requested = atoi(env);
      if(requested==32 || requested==64 || requested==128 || requested==256) {
	block_size = requested;
      } else {
	fprintf(stderr,"%-24s RNA_INT_LOOP_BLOCK_SIZE=%s not one of 32/64/128/256 -- ignoring, using %d\n",
		__FILE__, env, block_size);
      }
    }
    fprintf(stderr,"%-24s int_loop_kernel block size %d%s\n",
	    __FILE__, block_size,
	    env ? " (from RNA_INT_LOOP_BLOCK_SIZE)" : " (STOPGAP default -- see comment above)");
  }

  switch(block_size) {
    case 256: int_loop_kernel_256<<<blocks,256>>>(nfiles, i, /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_energy_min2); break; //Out
    case 128: int_loop_kernel_128<<<blocks,128>>>(nfiles, i, /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_energy_min2); break; //Out
    case  64: int_loop_kernel_64<<<blocks, 64>>>(nfiles, i, /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_energy_min2); break; //Out
    default:  int_loop_kernel_32<<<blocks, 32>>>(nfiles, i, /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_energy_min2); break; //Out
  }

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  //printf("int_loop_kernel<<<%d.%d,%d>>>(i=%d...) ok\n",blocks.x,blocks.y,block_size,i);

  gpuErrchk( cudaMemcpy(energy_min,d_energy_min2, g_row_total*sizeof(int),cudaMemcpyDeviceToHost) );
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
	   int* energy_min,
	   const size_t* size_off_H ) { //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
  // Staggered_Row_Batching Phase 2b: this used to have a defensive
  // `if(first2) init_gpu2(...)` fallback here, but int_loop_i() is only ever
  // reached via fill_arrays_loop.c -> par_fill_arrays() -> par_mfe(), which
  // unconditionally calls init_gpu2() (with the real tri_off_H table) before
  // par_fill_arrays() ever runs -- first2 is always already 0 by the time
  // this line executes, so the fallback was dead code. Removed rather than
  // given a fabricated tri_off_H it has no access to here.
  assert(!first2);

  int_loop_cuda(nfiles,i,/*turn,*/length,VC[0]->params, energy_min, size_off_H);
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

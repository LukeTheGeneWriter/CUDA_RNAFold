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
#include "ViennaRNA/datastructures/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/strings.h"
#include "ViennaRNA/utils/structures.h"
#include "ViennaRNA/constraints/basic.h"
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/constraints/soft.h"
#include "ViennaRNA/loops/external.h"
#include "ViennaRNA/eval/gquad.h"
#include "ViennaRNA/mfe/gquad.h"
#include "ViennaRNA/structured_domains.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/loops/internal.h"
#include "ViennaRNA/params/default.h"   /* MAX_NINIO -- see the note below */

//use GPU primitives in CUDA code
#undef MIN2
#define MIN2(x,y) min(x,y)
#undef MAX2
#define MAX2(x,y) max(x,y)
// The interior-loop asymmetry cap. This was `#define MAX_NINIO 300` until
// 2026-09-09, and the #define was a silent wrong answer waiting for -P:
// MAX_NINIO is not a constant, it is a WRITABLE library global
// (params/default.c:70, declared extern at params/default.h:70) and reading a
// parameter file overwrites it (params/io.c:671, the NINIO block's third
// value; io.c:1601 writes it back out again).
//
// The check that used to sit in init_gpu2() -- `assert(MAX_NINIO == 300)` --
// could not fail, because with the #define in scope it expanded to
// assert(300 == 300). It read as a guard against exactly the hazard it did
// not test. See PORT_NSP_PARAMFILE_SCOPE.md §2.2.
//
// The declaration comes from ViennaRNA/params/default.h, which is included
// explicitly above. It was ALREADY reaching this file transitively (via
// loops/internal.h -> eval/internal.h, and again via interior_loopx.h ->
// energy_par.h), which is exactly why the old #define was dangerous: it sat
// between two includes of the same header and only escaped mangling
// `extern int MAX_NINIO;` into `extern int 300;` because the include guard had
// already fired.
#include           "interior_loopx.h"

#include "stub2.h"
#include "gquad_dev.h"
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
/* The device helpers below moved to int_loop_dev.h so the fused megakernel
 * can call exactly this code; see the header. */
#include "int_loop_dev.h"


cuda_param_t* d_param;
char*         d_pair; //[NBPAIRS+1][NBPAIRS+1];
/* RNA_STREAM_OVERLAP (device.cu). Both return the NULL stream until the knob is
 * on, so every launch below is exactly where it has always been by default. */
extern "C" cudaStream_t rnafold_stream_cell(void);
extern "C" cudaStream_t rnafold_stream_hp(void);

unsigned int* d_hccc; //read via Hc

// HARD CONSTRAINTS, THE HALF THAT IS NOT IN hc->mx.
//
// hc->mx says whether a PAIR is legal. Whether a base may be left UNPAIRED
// lives in hc->up_hp / up_int / up_ml / up_ext, and until 2026-09-16 this
// device carried only up_ml. That is invisible until a constraint forces a
// base to PAIR: no bit in the four masks moves, and the sweep goes on allowing
// interior loops whose unpaired span covers it. Measured, with the guard
// lifted: better-than-legal answers on the two --enforceConstraint shapes.
//
// NULL whenever no record in the batch carries a hard-constraint depot, which
// is every production fold today -- the kernels then skip the test entirely
// rather than reading an array that says "everything is allowed" at the cost of
// two loads per candidate in the hottest loop in the project.
//
// One byte per position, clamped at 31: the only comparisons are against an
// interior-loop unpaired run, which MAXLOOP bounds at 30.
unsigned char* d_up_int = NULL;
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
// Luke's Flow Batching, fix 3: this row's slot in the chunk's row tables
// (device.cu). NOT owned here and never written here -- bind_row_tables(i)
// points them at row i's slot, which is written once per chunk, so no queued
// kernel on any stream can see the table change under it. Replaces the per-row
// blocking uploads over a shared buffer that raced at RNA_STREAM_OVERLAP=2.
static const size_t* d_size_off_H = NULL;
static const int*    d_i_H        = NULL;
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
  // The int16 fML bound is derived from the LOADED table, not from the literal
  // -340 of the default one -- a -P file replaces stack37 and can push the
  // offset past int16. Vetted here because this is the first point at which the
  // real table is in hand, and long before any cell is packed. See
  // rnafold_fml_int16_vet_params().
  {
    int worst = 0;   /* most negative entry; INF guards the unused NN slots */
    for(int a=0; a<=NBPAIRS; a++)
      for(int b=0; b<=NBPAIRS; b++) {
        const int v = P->stack[a][b];
        if(v < worst && v > -INF/2) worst = v;
      }
    rnafold_fml_int16_vet_params(worst, FML_BLK);
  }
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
  H->SaltStack  =         P->SaltStack;
  H->max_ninio  =         MAX_NINIO;   //the LIVE global, not a literal
  memcpy(H->rtype,        P->model_details.rtype, 8*sizeof(int));
  H->noGUclosure =        P->model_details.noGUclosure;
  H->dangles     =        P->model_details.dangles;
  //n_max = MAXLOOP+1 fills exactly MAXLOOP+3 entries (n_max+2)
  rnafold_build_salt_table(P, MAXLOOP+1, H->SaltLoop);

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

// Continuous flow phase C2: SLOT REFILL. A slot has to be able to take a
// DIFFERENT record, which means redoing every piece of sequence-derived device
// content this file owns -- the hard-constraint bitmasks, the packed sequence,
// and d_my_c's INF prefill -- WITHOUT reallocating anything.
//
// It is done by re-entering init_gpu2() with the allocations suppressed rather
// than by a second copy of the packing code, deliberately: a duplicated packer
// that drifts from the original is exactly the kind of bug that produces wrong
// energies with no crash. SLOT_ALLOC() is the only difference between the two
// paths, and the buffers it skips are unchanged in size because the refill
// keeps the same nfiles and the same capacity table.
static int g_refill2 = 0;
// Phase C3: a MID-SWEEP refill touches one slot while every other slot still
// holds live data, so the whole-buffer d_my_c prefill has to be suppressed and
// replaced by a range fill over that slot's triangle alone. Clobbering the
// neighbours here would be silent: they are mid-recursion, not INF.
static int g_slot_only = 0;
#define SLOT_ALLOC(pp, sz) do { if(!g_refill2) TIMED_CUDAMALLOC(pp, sz); } while(0)

PUBLIC void
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size,
          const size_t* tri_off_H, const size_t* row_off_H, //in, nfiles+1 entries each, mfe_cuda.c
          const size_t* cap_H) { //in, nfiles slot capacities in nt -- continuous flow phase C1
  if(!first2) return;
  const double _t_ig2 = rnafold_now_seconds();
  fprintf(stderr,"%-24s init_gpu2(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

  assert(turn_ == turn);

  SLOT_ALLOC(&d_tri_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_tri_off_H, tri_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
  SLOT_ALLOC(&d_row_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_row_off_H, row_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );
  //printf("%s %s d_param is %lu bytes, NBPAIRS %d MAXLOOP %d BLOCK_SIZE %d\n",
  //	 __FILE__,Version,sizeof(cuda_param_s),NBPAIRS,MAXLOOP,block_size);

  // d_param/d_pair are nfiles/length-independent -- guarded on their own
  // one-time check (d_param starts NULL, a zero-initialized global) rather
  // than on first2, so teardown_gpu2() can reset first2=1 between GPU
  // batches without this block re-allocating (and leaking) them every batch.
  // ALLOCATE once, UPLOAD every batch.
  //
  // These are independent of nfiles and length, which is why the allocation is
  // guarded on d_param being NULL rather than on first2 -- reallocating per
  // batch would leak. They are NOT independent of the MODEL, and uploading them
  // once per process was a silent wrong-answer bug: a second batch folded with
  // different model details (a different temperature, say) got the FIRST
  // batch's energy tables, with no error and perfectly plausible structures.
  //
  //   temperature 25, as the first par_mfe() call   12/12 records match
  //   temperature 25, after one batch at 37 C        0/12 records match
  //
  // Invisible from RNAfold, where the model is constant for a whole run, and
  // reachable the moment anything folds two batches with different models in
  // one process -- which is exactly what a vrna_mfe_batch() caller or a Python
  // binding does. The upload is a few KB against a whole batch, so doing it
  // every time costs nothing worth measuring.
  if(!d_param) {
    SLOT_ALLOC(&d_param, sizeof(cuda_param_s));
    const size_t pair_size = (NBPAIRS+1)*(NBPAIRS+1)*sizeof(char); // 32-bit signed integer overflow bug fix
    SLOT_ALLOC(&d_pair, pair_size);
  }
  {
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
    const size_t pair_size = (NBPAIRS+1)*(NBPAIRS+1)*sizeof(char);
    gpuErrchk( cudaMemcpy(d_pair,pair_,pair_size,cudaMemcpyHostToDevice) );
  }

  // Staggered_Row_Batching Phase 2e: per-H table (own shape -- Hc_ints()'s
  // padding is this file's own private constant, not row_off_H/tri_off_H's
  // shape) built from each H's real length, table-driven instead of a
  // uniform Hc_ints(length) multiply.
  size_t hc_off_H[nfiles+1];
  hc_off_H[0] = 0;
  // Phase C1: LAYOUT, so it is built from the slot capacity. The packing loop
  // below stays bounded by the occupant's own VC[H]->length -- a bigger slot
  // just leaves the tail of its bitmask block zero, and Indx(i,j) never
  // addresses it for this record.
  for(int H=0;H<nfiles;H++) hc_off_H[H+1] = hc_off_H[H] + Hc_ints(cap_H[H]);
  SLOT_ALLOC(&d_hc_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_hc_off_H, hc_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  // Staggered_Row_Batching Phase 5: allocated here (fixed size for the whole
  // chunk), not populated here -- this changes every sweep row i, uploaded
  // fresh per-row by int_loop_cuda()/load_my_c() instead.
  // (d_size_off_H / d_i_H: the chunk's row tables now, bound per row.)

  size_t size = hc_off_H[nfiles]*sizeof(unsigned int);
  SLOT_ALLOC(&d_hccc, size);
  // When hc->matrix is sequence-derived, hp_mb_loop.cu's pack_hc_kernel fills
  // d_hccc for us (it owns the plain sequence encoding and the pair table, and
  // init_gpu3 runs after this function). Skipping this loop is most of the
  // point: it and its init_gpu3 twin were 197.4 s of a 769 s Colab run.
  if(g_hc_seq_derived) {
    gpuErrchk( cudaMemset(d_hccc, 0, hc_off_H[nfiles]*sizeof(unsigned int)) );
  } else {
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
    // PORT TO 2.7.2: hc->matrix (triangular, walked contiguously) became
    // hc->mx (dense row-major, n*i+j). The BIT POSITION must stay the
    // triangular index Indx(i,j) because that is what the kernels look up;
    // only the fetch changes. See the fuller note in hp_mb_loop.cu.
    const int length_H = (int)VC[H]->length;
    for(int j=1;j<=length_H;j++){                    //leave padding as zero
      for(int i=1;i<=j;i++){
        const size_t t = (size_t)j*(j-1)/2 + i;      // Indx(i,j)
        const long long I = (long long)hc_off_H[H] + t/bitsperint; // Langdon's 2026 indexing bug -- host-side hccc population, missed by 2f35ecc's kernel-scoped fix
        const unsigned int bit = 1u << (t % bitsperint);
        if(VC[H]->hc->mx[(size_t)length_H*i + j] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC) hccc[I] |= bit;
      }
    }
  }
  stage_ig_pack_s += rnafold_now_seconds() - _t_pk1;
  gpuErrchk( cudaMemcpy(d_hccc,hccc,hc_off_H[nfiles]*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  free(hccc);

  // up_int, and only when some record actually constrains something.
  {
    int any_depot = 0;
    for(int H=0;H<nfiles;H++)
      if(VC[H] && VC[H]->hc && VC[H]->hc->depot) { any_depot = 1; break; }

    if(any_depot) {
      const size_t nrow = row_off_H[nfiles];
      unsigned char* upi = (unsigned char*) calloc(nrow, sizeof(unsigned char));
      for(int H=0;H<nfiles;H++){
        if(!VC[H]) continue;
        const int length_H = (int)VC[H]->length;
        // hc->up_int is allocated (n+2) entries (constraints/hard.c:182) and is
        // 1-based; index 0 is unused here, exactly as up_ml's packing does.
        for(int k=1;k<=length_H;k++){
          const unsigned int v = VC[H]->hc->up_int[k];
          upi[row_off_H[H]+k] = (unsigned char)((v > 31u) ? 31u : v);
        }
      }
      SLOT_ALLOC(&d_up_int, nrow*sizeof(unsigned char));
      gpuErrchk( cudaMemcpy(d_up_int, upi, nrow*sizeof(unsigned char), cudaMemcpyHostToDevice) );
      free(upi);
      fprintf(stderr,"%-24s hard-constraint up_int uploaded (%zu bytes): "
                     "interior loops are span-checked this run\n", __FILE__,
              nrow*sizeof(unsigned char));
    } else {
      d_up_int = NULL;
    }
  }
  }

  // Ten bases per word, H fastest index (see put10()/unpack()).
  assert(sizeof(unsigned int) == 4);
  size = ((size_t)nfiles * (length+2) + 9)/10 * sizeof(unsigned int);
  SLOT_ALLOC(&d_S, size);
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
    SLOT_ALLOC(&d_my_c, size);
    if(!g_slot_only) init_my_c(my_c_elems);   // phase C3: see g_slot_only
  }

  // Staggered_Row_Batching Phase 6d: cached for the per-row entry points that
  // transfer these buffers whole but never see the offset table.
  g_row_total = row_off_H[nfiles];

  size = g_row_total*sizeof(int); // 32-bit signed integer overflow bug fix
  SLOT_ALLOC(&d_new_e, size);

  // Staggered_Row_Batching Phase 2d: table-driven total (row_off_H[nfiles]),
  // matching int_loop_kernel_body.inc's row_off_H[H]+j write below -- equals
  // the old uniform nfiles*(length+1) exactly while chunks stay uniform-length,
  // diverges once they don't.
  SLOT_ALLOC(&d_energy_min2, g_row_total*sizeof(int));
  /*no longer in use
  SLOT_ALLOC(&d_energy_min20, size);

  size = nfiles*length*sizeof(int);
  SLOT_ALLOC(&d_buf, size);
  */
  stage_ig2_s += rnafold_now_seconds() - _t_ig2;
  first2 = 0;
}

// Continuous flow phase C2: re-run init_gpu2()'s CONTENT for a chunk whose slots
// have taken new occupants. Same nfiles, same capacity table, same buffers --
// only the records differ. first2 is forced back on so the body runs; the
// SLOT_ALLOC guard is what stops it reallocating over the live pointers.
PUBLIC void
refill_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_,
            const int length, const int block_size,
            const size_t* tri_off_H, const size_t* row_off_H, const size_t* cap_H) {
  assert(!first2);            // must be a live chunk, not a fresh one
  g_refill2 = 1;
  first2    = 1;
  init_gpu2(nfiles, VC, turn_, length, block_size, tri_off_H, row_off_H, cap_H);
  g_refill2 = 0;
}

// Continuous flow phase C3: refill ONE slot, mid-sweep, without disturbing the
// others. The sequence-derived content this file owns is either per-slot already
// or recomputed identically for the unchanged slots (d_S is repacked whole: its
// ten-bases-per-word layout interleaves the slots, so a single slot cannot be
// rewritten in isolation without read-modify-write of shared words. The repack
// reproduces every other slot's bytes exactly, so it is safe -- it is just work,
// and Phase 2f's per-record d_S layout is what would remove it).
//
// The one thing that must NOT be done whole-buffer is d_my_c's INF prefill --
// hence g_slot_only and the range fill below.
PUBLIC void
refill_slot2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_,
             const int length, const int block_size,
             const size_t* tri_off_H, const size_t* row_off_H, const size_t* cap_H,
             const int slot) {
  g_slot_only = 1;
  refill_gpu2(nfiles, VC, turn_, length, block_size, tri_off_H, row_off_H, cap_H);
  g_slot_only = 0;

  const size_t lo = tri_off_H[slot];
  const size_t n  = tri_off_H[slot+1] - lo;
  const size_t nblocks = (n + BLOCK_SIZE - 1)/BLOCK_SIZE;
  init_my_c_kernel<<<nblocks,BLOCK_SIZE>>>(n, d_my_c + lo);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
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
  if(d_up_int) { gpuErrchk( cudaFree(d_up_int) ); d_up_int = NULL; }
  gpuErrchk( cudaFree(d_S) );
  gpuErrchk( cudaFree(d_my_c) );
  gpuErrchk( cudaFree(d_new_e) );
  gpuErrchk( cudaFree(d_energy_min2) );
  gpuErrchk( cudaFree(d_tri_off_H) );
  gpuErrchk( cudaFree(d_row_off_H) );
  gpuErrchk( cudaFree(d_hc_off_H) );
  d_size_off_H = NULL;   // borrowed from the row tables, not ours to free
  d_i_H        = NULL;
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

// T2a (device.cu): device-to-host for backtrack worker w, through w's own
// copy stream and pinned stage. w < 0 is the old blocking default-stream copy.
void rnafold_d2h_w(void* dst, const void* src, const size_t bytes, const int w);
#define d2h_w rnafold_d2h_w

extern "C" /*PUBLIC*/ void
fetch_my_c_one_w(int* dst, const size_t tri_lo, const size_t cells, const int w) {
  d2h_w(dst, &d_my_c[tri_lo], cells*sizeof(int), w);
}

extern "C" /*PUBLIC*/ void
fetch_my_c_one(int* dst, const size_t tri_lo, const size_t cells) {
  fetch_my_c_one_w(dst, tri_lo, cells, -1);
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

// Lets hp_mb_loop.cu's pack_hc_kernel fill this file's d_hccc in the same pass
// that builds its own four masks -- one evaluation of the hc predicate instead
// of two, and no duplicate sequence/pair upload here. The extents differ
// (Hc_ints() pads by MAXLOOP where Hc_ints2() does not), so the offset table
// goes across too.
extern "C" /*PUBLIC*/ void
int_loop_hccc_buffers(unsigned int** d_out, const size_t** off_out) {
  *d_out   = d_hccc;
  *off_out = d_hc_off_H;
}

// GPU-resident sweep, step 1: this file's two row-shaped device buffers, for
// the kernels that will live in hp_mb_loop.cu. Same pattern as the accessor
// above -- ownership stays here, only a pointer crosses.
//   d_energy_min2  int_loop_kernel's per-row output, which new_c_kernel reads
//                  as its starting `new_c` (today it is D2H'd into the host's
//                  energy_min purely so new_c_host can read it).
//   d_new_e        load_my_c_kernel's input, which new_c_kernel will write
//                  directly (today new_c_host writes the host's new_C and
//                  load_my_c() uploads it).
// Valid only between init_gpu2() and teardown_gpu2().
extern "C" /*PUBLIC*/ void
int_loop_row_buffers(int** energy_min2_out, int** new_e_out) {
  if(energy_min2_out) *energy_min2_out = d_energy_min2;
  if(new_e_out)       *new_e_out       = d_new_e;
}

// Point this file's tables at row i's slot. Costs nothing and issues nothing:
// the slot was uploaded before the row's first kernel (device.cu).
static void
bind_row_tables(const int i) {
  d_size_off_H = rnafold_rowtab_size(i);
  d_i_H        = rnafold_rowtab_ih(i);
}

//perhaps this can be combined with other kernels?
// load_my_c's cell, shared with the fused per-record megakernel.
#include "int_loop_cells.inc"

__global__ void
load_my_c_kernel(const int nfiles, const int i_row, /*const int turn,*/ const int length,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c,
		 const size_t* __restrict__ tri_off_H, //in
		 const size_t* __restrict__ row_off_H, //in
		 const size_t* __restrict__ size_off_H, const size_t total, //in
		 const int* __restrict__ i_H) {
  // The arithmetic lives in int_loop_cells.inc so the megakernel runs exactly this code.
  load_my_c_cell(nfiles, i_row, length, new_e, my_c, tri_off_H, row_off_H, size_off_H, total, i_H,
                 blockIdx.x*blockDim.x+threadIdx.x);
}

PUBLIC void
load_my_c(const int nfiles,
	  const int i, const int turn_, const int length,
	  const int* new_e,
	  const size_t* size_off_H,     //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	  const int* i_H) {             //in, nfiles entries -- continuous flow phase A2
  //out d_my_c
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

#ifdef NDEBUG
  //check here in case of earlier errors
  gpuErrchk( cudaDeviceSynchronize() );
#endif
  //for simplicity transfer all new_e, even though only need H * [start:length]
  // GPU-resident sweep: in device mode new_c_kernel has already written d_new_e
  // directly, so uploading the host's copy over it is exactly the round trip
  // being removed.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaMemcpy(d_new_e,new_e,g_row_total*sizeof(int),cudaMemcpyHostToDevice) );
  bind_row_tables(i);


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

  load_my_c_kernel<<<nblocks,block_size,0,rnafold_stream_cell()>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
					   d_new_e,  //in
					   d_my_c,   //out
					   d_tri_off_H,  //in
					   d_row_off_H,  //in
					   d_size_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
  // Level 2: publish "row i's c is written" for the md chain to wait on.
  rnafold_stream_cell_done();
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );
}


//#include "ptype.cu"
//Was
//WBL 13 Jan 2018 From ViennaRNA-2.3.0/src/ViennaRNA/alphabet.c Revision: 1.9

//Modification:
//WBL 14 Jan 2018 just single element of ptype[ij]

//end_include "ptype.cu"

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
// BLOCK_SIZE=32 below is a placeholder for the shared device helpers only --
// Hc()/decode_column()/Energy() below are compiled once and used by every
// instantiation regardless of block size, so its value is irrelevant to them.
// It is NOT the default any more: that is INT_LOOP_DEFAULT_BLOCK_SIZE (64),
// named beside the four instantiations further down.
#define BLOCK_SIZE 32


#include "nth.h"


// H1: THE PART OF Energy() THAT DEPENDS ONLY ON THE CELL, NOT THE CANDIDATE.
//
// `type`, `si1` and `sj1` are functions of (i,j) alone, so they are constant for
// every candidate of one (H,j) cell -- and Energy() was recomputing all three
// per candidate. Each is a packed-sequence read: Ptype() is two unpack()s plus a
// pair_ lookup, and unpack() is an index computation, a global load and a
// shift/mask. Call it four redundant loads and thirty redundant instructions per
// candidate, against an IntLoop_X() body of perhaps fifty to a hundred.
//
// ONE HELPER, TWO KERNELS. The block-per-cell twin and the warp kernel both
// hoist it to exactly the same place -- outside their work loop -- through this
// function, so the two cannot drift. Byte-identity is by construction: the same
// values, computed once instead of N times.
//
// Whether nvcc's LICM had already done this is a question the SASS answers, not
// a question this comment should assert. See PORT_WARP_SCAN_SCOPE.md H1.



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
#undef BLOCK_SIZE

// ============ RNA_INT_LOOP_WARP: one WARP per cell, no shared, no barrier ====
//
// A SEPARATE KERNEL, for the third time and the same reason (gq_internal_kernel,
// modular_decomposition_smem_kernel): int_loop_kernel is 20-30% of GPU time and
// has a recorded history of regressions from being edited. This is an
// experiment. It is byte-identical to its twin or it is wrong.
//
// WHAT THE MEASUREMENT SAID, and this kernel is the shape it pointed at.
// STRESS272_RESULTS.md 22.5 profiled int_loop_kernel's warp stalls at three
// block sizes:
//
//     stall              bs 32    bs 64   bs 128
//     wait               23.3%    23.9%    24.4%     <- ALU dependency chain
//     barrier             0.1%     6.6%    20.7%     <- __syncthreads()
//     long_scoreboard    12.0%    14.2%    15.8%     <- global memory
//     short_scoreboard    9.4%     9.4%     8.9%     <- shared memory
//
// Memory is the THIRD cost. The first two are this kernel's own scan machinery,
// and `barrier` is why block size 128 gets MORE occupancy (86.9% against 69.8%)
// and is SLOWER: at one warp per block the two __syncthreads() are
// warp-synchronous and free, at four warps they are a barrier across warps doing
// DATA-DEPENDENT amounts of work.
//
// THREE CHANGES, each aimed at a measured stall:
//
//  1. ONE WARP PER CELL, not one block. col_mask[] and prefix[] were shared only
//     so warp 0 could publish them to the rest of the block; with the cell owned
//     by a single warp they live in REGISTERS, one column per lane. That deletes
//     both __syncthreads() (`barrier` -> 0) and both shared arrays
//     (`short_scoreboard` -> 0), and a block now holds BLOCK/32 INDEPENDENT
//     cells, so occupancy rises with block size without the barrier toll.
//
//  2. THE COLUMN LOOKUP BECOMES A 5-STEP SHUFFLE BINARY SEARCH. The twin does
//         while(column <= maxcol && prefix[column+1] <= work) column++;
//     -- restarting at 0 for EVERY work item, up to 31 dependent shared-memory
//     loads each. That is a prime suspect for `wait` and it is all of
//     `short_scoreboard`. Here the prefix lives distributed across lanes and the
//     search is 5 register-speed shuffles with no memory in the chain at all.
//
//  3. NO NEGATIVE-INDEX HAZARD. The twin needs an explicit clamp because
//     maxcol <= -2 would read prefix[maxcol+1] BEFORE the shared array (a real
//     memory-safety bug it hit once). With the prefix in registers, maxcol < 0
//     simply gives every lane popc = 0, total = 0, and no iterations.
//
// WHY THE SHUFFLES ARE SAFE. Every __shfl_*_sync below is reached by all 32
// lanes: the warp owns ONE cell, so `cell`, `Hc(ij)`, `maxcol`, `total` and
// `iters` are warp-uniform, and the early return is taken by the whole warp or
// none of it. The binary search runs a FIXED five iterations rather than
// `while (lo < hi)` for exactly this reason -- a divergent trip count would make
// the shuffles undefined, which is the kind of bug that produces a plausible
// wrong answer rather than a crash.
//
// Byte-identity is not an aspiration here, it is the definition: min is
// associative and commutative over exact ints, and this enumerates the SAME set
// of (p,q) candidates in a different order.

// H7: THE SAME QUESTION, ASKED 32 WAYS AT ONCE.
//
// H6 removed flatten_index_to_H()'s nine-deep dependent chain by deleting the
// FLAT GRID. But the flat grid is not what costs -- the SEARCH is. Flattening
// is what lets records have ragged widths (a retired record has width 0, so
// raggedness is not an edge case, it is what continuous flow does every row)
// and it is what lifts the gridDim.y <= 65535 cap. Those are worth keeping.
//
// So keep the flat index and change only the lookup. A warp has 32 lanes and,
// at this point in the kernel, nothing for them to do: every lane already
// shares `idx`. Probe 32 points of the interval at once and the search becomes
// 32-ary instead of binary -- ceil(log32(nfiles)) DEPENDENT steps rather than
// ceil(log2(nfiles)):
//
//     nfiles <=    32   1 step   (was 5)
//     nfiles <=  1024   2 steps  (was 10)
//     nfiles <= 32768   3 steps  (was 15)
//
// It reads more BYTES per step -- 32 strided 8-byte probes instead of one --
// but flat_off_H is nfiles+1 size_t (3.2 KB at nfiles=400), read by every warp,
// so it is L1-resident after the first. Bytes are not what this kernel is short
// of; the dependency chain is (STRESS272_RESULTS.md 26.6, 28.5).
//
// NO SHUFFLES ARE NEEDED. Each probe position is a pure function of
// (lo, span, lane), so once the ballot names the winning lane every lane can
// recompute that lane's position -- and the next one's -- arithmetically.
//
// The trip count is WARP-UNIFORM: `idx` is shared by the warp and `nfiles` by
// the whole grid, so every lane runs the same number of iterations and every
// __ballot_sync names all 32 lanes. (A divergent ballot is undefined, which is
// the same trap the 5-step binary search in the work loop is written around.)
__device__ inline int
flatten_index_to_H_warp(const size_t idx, const size_t* __restrict__ flat_off_H,
                        const int nfiles, const int lane) {
  int lo = 0;
  int hi = nfiles;

  while(hi - lo > 1) {
    const int span = hi - lo;

    // Lane k probes lo + floor(span*k/32). For span < 32 the tail lanes
    // duplicate the last position, which is harmless: a duplicate can only
    // re-elect a position already known to satisfy the predicate.
    const int pk = lo + (int)(((long long)span * lane) >> 5);
    const unsigned int vote =
        __ballot_sync(0xffffffff, flat_off_H[pk] <= idx);

    // flat_off_H[lo] <= idx is the loop invariant, so lane 0 always votes and
    // `vote` is never 0.
    const int win = 31 - __clz((int)vote);

    const int newlo = lo + (int)(((long long)span * win) >> 5);
    const int newhi = (win == 31) ? hi
                                  : lo + (int)(((long long)span * (win+1)) >> 5);

    // PROGRESS IS GUARANTEED, and this guard is belt-and-braces. If win < 31
    // then p(win+1) > p(win) strictly -- equal positions would have made lane
    // win+1 vote the same way, so win would not be the highest. If win == 31
    // then p(31) <= lo+span-1 < hi. Either way the interval strictly shrinks.
    lo = newlo;
    hi = (newhi > newlo) ? newhi : (newlo + 1);
  }

  return lo;
}
#define INT_LOOP_WARP_KERNEL_NAME2(cpb) int_loop_warp_kernel_##cpb
#define INT_LOOP_WARP_KERNEL_NAME(cpb) INT_LOOP_WARP_KERNEL_NAME2(cpb)

// The interior-loop cell, shared with the fused per-record megakernel: the
// kernel below is a wrapper around exactly this code.
#include "int_loop_cell.inc"

template <int CELLS_PER_BLOCK, bool GRIDY, bool WSEARCH>
__global__ void
int_loop_warp_kernel(const int nfiles, const int i_row, const int length,
                const int TerminalAU, const int ninio2,
                const cuda_param_t* __restrict__ P, const float lxc,
                const char* __restrict__ pair_,
                const unsigned int* __restrict__ S,
                const unsigned int* __restrict__ hccc,
                const unsigned char* __restrict__ up_int, //in, d_up_int -- NULL when unconstrained
                const int* __restrict__ my_c,
                const size_t* __restrict__ tri_off_H,
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ hc_off_H,
                const size_t* __restrict__ size_off_H,
                const int* __restrict__ i_H,
                      int* __restrict__ energy_min) {
  static_assert(CELLS_PER_BLOCK >= 1 && CELLS_PER_BLOCK <= 32,
                "one warp per cell; blockDim.x must be CELLS_PER_BLOCK*32");

  const int lane = (int)(threadIdx.x & 31u);
  const int wib  = (int)(threadIdx.x >> 5);             // warp within the block

  // Whole-warp exit in both grids: every lane of this warp shares the cell, so
  // no lane is left behind to be named by a shuffle mask below. This is the one
  // thing the block-per-cell twin could not do (its warps straddle cells).
  int    H;
  size_t local;                                          // cell within record H

  if(GRIDY) {
    // H6: blockIdx.y IS the record. Two adjacent size_off_H reads give the
    // record's width; no search, and the reads below do not queue behind one.
    H     = (int)blockIdx.y;
    local = (size_t)blockIdx.x * CELLS_PER_BLOCK + wib;
    if(local >= size_off_H[H+1] - size_off_H[H]) return; // the padding blocks
  } else {
    const size_t cell = (size_t)blockIdx.x * CELLS_PER_BLOCK + wib;
    if(cell >= size_off_H[nfiles]) return;
    H     = WSEARCH ? flatten_index_to_H_warp(cell, size_off_H, nfiles, lane)
                    : flatten_index_to_H(cell, size_off_H, nfiles);
    local = cell - size_off_H[H];
  }

  // The arithmetic lives in int_loop_cell.inc so the megakernel runs
  // exactly this code; only the cell derivation above is kernel-specific.
  int_loop_warp_cell(nfiles, i_row, length, TerminalAU, ninio2, P, lxc, pair_, S, hccc, up_int, my_c, tri_off_H, row_off_H, hc_off_H, size_off_H, i_H, energy_min,
                     H, local, lane);
}

// H6: WHICH RECORD IS THIS CELL IN? -- asked once per cell, answered with a
// NINE-DEEP CHAIN OF DEPENDENT GLOBAL LOADS.
//
// The grid is flat (one index over every cell of every record), so every warp
// opens by running flatten_index_to_H()'s binary search over size_off_H[] --
// 12 SASS instructions and ONE dependent LDG per iteration, ceil(log2(nfiles))
// iterations, nothing else able to issue behind it because each probe address
// depends on the previous probe's value. That is the prologue of a kernel whose
// stalls are now 44.4% `long_scoreboard` (STRESS272_RESULTS.md 27.5).
//
// The host already knows the answer. It knew it before the flattening: the
// comment in int_loop_cuda() records that this WAS a dim3(nblocks,nfiles) grid,
// and was flattened in Staggered_Row_Batching Phase 5 to allow ragged per-record
// widths and to lift an nfiles<=65535 limit. That bought generality and paid a
// dependent chain per cell.
//
// GRIDY restores blockIdx.y == H, so H is free and the five per-record table
// reads (i_H, tri_off_H, row_off_H, hc_off_H, size_off_H) become INDEPENDENT
// loads that issue together instead of queueing behind a search. The cost is
// blocks launched past a short record's width, which exit on one comparison --
// so the host picks the grid per launch and only takes it when the waste is
// small. See rnafold_int_loop_gridy() and int_loop_cuda().
// NOTE ON THE BANNER: int_loop_cuda() calls this UNCONDITIONALLY, before it
// decides which grid to use, so the knob announces itself even on a run where
// every launch takes the 2-D grid and this lookup is never reached. It was
// originally read inside the flat arm only, which meant asking for both knobs
// on a uniform workload produced no WSEARCH banner at all -- and a harness that
// checks "the knob I asked for engaged" would have failed that arm for telling
// the truth. A knob that cannot announce itself cannot be asserted on.
//
// RNA_INT_LOOP_WSEARCH=1 -- keep the flat grid, replace the per-cell binary
// search with the warp-cooperative 32-ary one above. Orthogonal to
// RNA_INT_LOOP_GRIDY, which deletes the search by deleting the flat grid; when
// both are asked for, the 2-D grid wins on the launches where its waste guard
// accepts and this covers the rest.
PUBLIC int
rnafold_int_loop_wsearch(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_INT_LOOP_WSEARCH");

    v = (e && e[0] && e[0] != '0') ? 1 : 0;

    if (v)
      fprintf(stderr, "%-24s RNA_INT_LOOP_WSEARCH=1: 32-ary warp-cooperative cell "
                      "-> record lookup, flat grid kept (experimental; sha must not "
                      "move)\n", __FILE__);
  }

  return v;
}

PUBLIC int
rnafold_int_loop_gridy(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_INT_LOOP_GRIDY");

    v = (e && e[0] && e[0] != '0') ? 1 : 0;

    if (v)
      fprintf(stderr, "%-24s RNA_INT_LOOP_GRIDY=1: blockIdx.y is the record, no "
                      "per-cell binary search (experimental; sha must not move)\n",
              __FILE__);
  }

  return v;
}

// Waste budget for the 2-D grid: a ragged chunk launches nfiles*ceil(maxw/cpb)
// blocks against sum(w_H)/cpb cells of real work, and the excess blocks exit on
// one comparison. 25% is a guess and is MEANT to be one -- the length-sorted
// chunker makes near-uniform widths the common case, and the fallback exists so
// that a chunk which is not near-uniform simply keeps today's flat grid rather
// than paying for the generality. If a workload ever trips it often, measure
// before widening it.
#define INT_LOOP_GRIDY_WASTE_NUM 5
#define INT_LOOP_GRIDY_WASTE_DEN 4

// The default, named here beside the instantiations so it cannot drift away
// from them. 64 since 2026-09-11; see the block-size history immediately below.
#define INT_LOOP_DEFAULT_BLOCK_SIZE 64

// AND A SEPARATE DEFAULT FOR THE WARP KERNEL, because the knob means a
// DIFFERENT QUANTITY there: for the twin it is threads cooperating on ONE cell,
// for the warp kernel it is 32 x the number of INDEPENDENT cells in a block.
// 64 being right for one says nothing about the other, which is why it was
// swept separately (A100, 400 x 5601, int_loop):
//
//     32 threads = 1 cell   20.65 s    <-- best
//     64         = 2        20.86 s   +1.0%
//    128         = 4        21.35 s   +3.4%
//    256         = 8        22.88 s  +10.8%
//
// Monotone, and the opposite of the prediction. There is no __syncthreads() in
// this kernel, so the expectation was that occupancy would scale with block
// size for free. It does not, because the BLOCK is still the allocation and
// retirement unit: a block holds its registers until its LAST warp finishes,
// and cells have wildly different candidate counts. Bigger blocks re-couple
// cells that the design had just decoupled.
//
// (The occupancy ceiling is identical either way -- at 58 regs/thread sm_80
// allows 32 blocks x 1 warp or 16 blocks x 2 warps, 32 warps both times. The
// difference is scheduling granularity, not capacity.)
#define INT_LOOP_WARP_DEFAULT_BLOCK_SIZE 32

// int_loop_warp_kernel -- one (H,j) cell per WARP instead of per block.
//
// DEFAULT SINCE 2026-09-12, on two architectures:
//
//     RTX 3050 (sm_86), 60 x 2400   int_loop 8.543 -> 6.727 s   -21.3%
//     A100     (sm_80), 400 x 5601  int_loop 26.23 -> 20.82 s   -20.6%
//
// with `modular_decomposition` -- which this knob cannot reach -- flat to
// +0.01% on the A100 and the sha unchanged in every arm. The A100 pair spread
// 0.15% within each arm and neither arm throttled (1410/1410 MHz both).
//
// IT WAS BUILT FROM THE STALL DATA AND THE STALL DATA CONFIRMS THE MECHANISM.
// `barrier` was 20.7% of stall cycles at block size 128 on the twin; the design
// deletes both __syncthreads() and both shared arrays by giving one warp the
// whole cell, so col_mask[] and prefix[] live in registers. Measured on the
// A100: barrier 0.933 -> 0.000 cycles per issue, EXACTLY zero, and
// short_scoreboard 1.167 -> 0.710. See STRESS272_RESULTS.md 27.
//
// RNA_INT_LOOP_WARP=0 restores the block-per-cell twin, which is kept as the
// reference implementation and as the A/B.
PUBLIC int
rnafold_int_loop_warp(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_INT_LOOP_WARP");

    v = (e && e[0]) ? (e[0] != '0') : 1;

    fprintf(stderr, "%-24s int_loop kernel: %s%s\n", __FILE__,
            v ? "warp-per-cell" : "block-per-cell (twin)",
            (e && e[0]) ? " (from RNA_INT_LOOP_WARP)" : " (the measured default)");
  }

  return v;
}

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
// Re-measured 2026-08-20 (local RTX 3050, CC 8.6, ncu --set basic,
// RNA_INT_LOOP_BLOCK_SIZE=32 vs 256, grids from ~40 blocks to ~54000):
// BLOCK_SIZE=256 was slower at every grid size tested -- 1.07x at small grids,
// 2.35x at grid~500, ~1.09x even at grid~54000 -- despite occupancy rising from
// 5-33% to 36-62%. That conclusion still stands and is reproduced below.
//
// THE STOPGAP IS RETIRED, 2026-09-11, AND THE DEFAULT IS NOW 64. The 08-20
// sweep compared 32 against 256 and never tried 64, so the conclusion it drew
// ("real occupancy gains come from more BLOCKS, not bigger blocks") was
// generalised from the two ends of the range. It is wrong in the middle.
//
// Measured at 400 x 5601 on a T4, phase-synced, i32 (STRESS272_RESULTS.md 20.1):
//
//     block size     int_loop      modular_decomp (control)     wall
//         32         102.95 s            215.4 s               523.9 s
//       * 64  *       91.80 s            216.3 s               512.4 s
//        128         110.31 s            216.1 s               528.2 s
//        256         153.56 s            215.7 s               572.5 s
//
// modular_decomp cannot be touched by this knob and is flat to 0.34% across all
// four arms, so it is a free control for device drift -- and there was none.
// The wall delta (-11.5 s) and the phase delta (-11.15 s) agree to 0.4 s, which
// is the real evidence: two independently measured quantities moved together.
//
// REPLICATED ON A SECOND ARCHITECTURE before this default changed, because
// int16's value turned out to be machine-dependent and the 16-blocks/SM limit
// that makes 64 pay is an sm_75 number. RTX 3050 (sm_86), 60 x 2400, ABBA
// within each pass so a monotone drift cancels: -18.0%, -16.4%, -13.3% across
// three passes. Output sha identical in all 12 local and all 12 Colab arms.
//
// WHY 64 AND NOT MORE, which is the part the old comment had half-right. NCU
// (the kernel's first profile ever, 20.3) puts it at 44-48% occupancy -- exactly
// the >=50% ceiling that 16 blocks/SM x 1 warp implies -- with DRAM at 4.2-5.2%
// of peak and SM throughput at 40-44%. It is latency-bound with too few warps
// resident, NOT bandwidth-bound, so lifting the block-count ceiling pays. But
// each (H,j) cell's search space is bounded by MAXLOOP=30, so past 64 threads a
// block is mostly idle and pays the cross-warp combine (#if BLOCK_SIZE > 32) for
// nothing. 64 is where occupancy doubles before the work runs out.
//
// Use RNA_INT_LOOP_BLOCK_SIZE to re-test 32/128/256 on new hardware without
// reintroducing an in-process benchmark -- which is what caused the last two
// regressions.


// ===================== G-quadruplex G2: the interior-loop term ==============
//
// The device twin of vrna_mfe_gquad_internal_loop() (mfe/mfe_gquad.c:269-470):
// "all cases where a g-quadruplex may be enclosed by base pair (i,j)".
// mfe_internal.c:637 MIN2s it into the interior-loop energy, so this MIN2s into
// d_energy_min2, which is int_loop_kernel's output for exactly that quantity.
//
// A SEPARATE KERNEL, DELIBERATELY. int_loop_kernel is 30% of GPU time
// (STRESS272_RESULTS.md 19) and has a recorded history of regressions from
// being touched without measurement. Folding three more (p,q) sweeps into it
// would put gquad-only work inside the hottest loop in the project and risk the
// default path for a feature almost nobody runs. This launches only when a c_gq
// was uploaded, so with -g off it does not exist.
//
// ONE THREAD PER (H,j) CELL, not one block: unlike the interior loop proper,
// each cell's three sweeps are short (p bounded by MAXLOOP=30, q by
// VRNA_GQUAD_MAX_BOX_SIZE=73) and heavily filtered -- upstream skips unless
// S1[p]==3 AND S1[q]==3, both must be G. Scanning is cheaper than cooperating.
__global__ void
gq_internal_kernel(const int nfiles, const int turn_,   // turn_ not turn: `turn` is a #define (:70)
                   const cuda_param_t* __restrict__ P,
                   const unsigned int* __restrict__ S,     // packed sequence_encoding
                   const char* __restrict__ pair_,
                   const size_t* __restrict__ row_off_H,
                   const size_t* __restrict__ size_off_H,
                   const int* __restrict__ i_H,
                         int* __restrict__ energy_min,     // in/out, MIN2'd
                   const int* __restrict__ gq_v,
                   const unsigned int* __restrict__ gq_col,
                   const unsigned int* __restrict__ gq_rowoff,
                   const size_t* __restrict__ gq_ent_off,
                   const size_t* __restrict__ gq_row_off) {
  const size_t k = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(k >= size_off_H[nfiles]) return;

  const int H = flatten_index_to_H(k, size_off_H, nfiles);
  const int i = i_H[H];
  const int j = (int)(k - size_off_H[H]) + i + turn_ + 1;

  // Upstream's own entry guard (mfe_gquad.c:283-285). i>0 always holds here,
  // and j <= this record's length because size_off_H[H] is built from
  // VC[H]->length - i - turn (fill_arrays_loop.c) -- so no len_H bound is
  // needed, and taking one from the BATCH max would be the 00d1e07 bug again.
  if(i + VRNA_GQUAD_MIN_BOX_SIZE >= j) return;

  // The closing pair's contribution, computed once (mfe_gquad.c:304-312).
  // vrna_get_ptype_md() PROMOTES 0 -> 7 (alphabet.c:475-477); the raw pair
  // value is not the same thing, and this is the identical trap that made
  // --nsp a live wrong answer in Energy(). dangles==2 on this fork, so the
  // mismatchI term is unconditional.
  const int si = unpack(S,H,nfiles,i+1);
  const int sj = unpack(S,H,nfiles,j-1);
  unsigned char type = Ptype(S,pair_,H,nfiles,i,j);
  if(type == 0) type = 7;

  // --noClosingGU. vrna_mfe_gquad_internal_loop() is called from INSIDE
  // mfe_internal.c's `if (!noclose)` block (:637), so a GU/UG closing pair
  // contributes no quadruplex-in-interior-loop term at all. There is no
  // enclosed-pair half here: what is enclosed is a quadruplex, not a pair.
  if(P->noGUclosure && ((type == 3) || (type == 4))) return;

  // vrna_mfe_gquad_internal_loop() gates this on `if (dangles)`
  // (mfe_gquad.c:306), so dangle model 0 closes the quadruplex with a bare
  // pair. The only dangle-dependent term in the whole interior-loop family --
  // vrna_E_internal() itself has no dangle branch at all.
  int energy = (P->dangles) ? P->mismatchI[type][si][sj] : 0;
  if(type > 2) energy += P->TerminalAU;

  int ge = INF;

#define GQ_AT(p_,q_) gq_lookup(H,(unsigned int)(p_),(unsigned int)(q_), \
                               gq_v,gq_col,gq_rowoff,gq_ent_off,gq_row_off)

  // ---- sweep 1: p == i+1, the quadruplex abuts the closing pair on the 5' side
  {
    const int p = i + 1;
    if((unpack(S,H,nfiles,p) == 3) && (p + VRNA_GQUAD_MIN_BOX_SIZE < j)) {
      int minq = p + VRNA_GQUAD_MIN_BOX_SIZE - 1;
      if(minq + 1 + MAXLOOP < j) minq = j - MAXLOOP - 1;
      int maxq = p + VRNA_GQUAD_MAX_BOX_SIZE + 1;
      if(maxq + 3 > j) maxq = j - 3;
      for(int q = minq; q < maxq; q++) {
        if(unpack(S,H,nfiles,q) != 3) continue;
        const int e_gq = GQ_AT(p,q);
        if(e_gq != INF) {
          const int u = j - q - 1;
          assert(u >= 0 && u <= MAXLOOP);
          const int c0 = energy + e_gq + P->internal_loop[u];
          if(c0 < ge) ge = c0;
        }
      }
    }
  }

  // ---- sweep 2: p from i+2, both linkers non-empty
  for(int p = i + 2; p + VRNA_GQUAD_MIN_BOX_SIZE < j; p++) {
    const int l1 = p - i - 1;
    if(l1 > MAXLOOP) break;
    if(unpack(S,H,nfiles,p) != 3) continue;
    int minq = p + VRNA_GQUAD_MIN_BOX_SIZE - 1;
    if(minq + 1 + MAXLOOP - l1 < j) minq = j - MAXLOOP + l1 - 1;
    int maxq = p + VRNA_GQUAD_MAX_BOX_SIZE + 1;
    if(maxq >= j) maxq = j - 1;
    for(int q = minq; q < maxq; q++) {
      if(unpack(S,H,nfiles,q) != 3) continue;
      const int e_gq = GQ_AT(p,q);
      if(e_gq != INF) {
        const int u = l1 + j - q - 1;
        assert(u >= 0 && u <= MAXLOOP);
        const int c0 = energy + e_gq + P->internal_loop[u];
        if(c0 < ge) ge = c0;
      }
    }
  }

  // ---- sweep 3: q == j-1, the quadruplex abuts the closing pair on the 3' side
  {
    const int q = j - 1;
    if(unpack(S,H,nfiles,q) == 3) {
      const int p0 = (i + 4 + VRNA_GQUAD_MAX_BOX_SIZE - 1 < q)
                     ? q - VRNA_GQUAD_MAX_BOX_SIZE + 1 : i + 4;
      for(int p = p0; p + VRNA_GQUAD_MIN_BOX_SIZE - 1 < j; p++) {
        const int l1 = p - i - 1;
        if(l1 > MAXLOOP) break;
        if(unpack(S,H,nfiles,p) != 3) continue;
        const int e_gq = GQ_AT(p,q);
        if(e_gq != INF) {
          assert(l1 >= 0 && l1 <= MAXLOOP);
          const int c0 = energy + e_gq + P->internal_loop[l1];
          if(c0 < ge) ge = c0;
        }
      }
    }
  }
#undef GQ_AT

  if(ge != INF) {
    const size_t o = row_off_H[H] + j;
    if(ge < energy_min[o]) energy_min[o] = ge;   // MIN2, mfe_internal.c:640
  }
}


// Launch it for this sweep row. A no-op unless rnafold_gq_upload() put a c_gq
// on the device, so the default path never reaches the kernel.
PUBLIC void
gq_internal_i(const int nfiles, const int i, const int turn_, const size_t* size_off_H,
              const int* i_H) {
  const int* gv; const unsigned int *gc, *gr; const size_t *ge_, *gro;

  if(!rnafold_gq_csr_device(&gv,&gc,&gr,&ge_,&gro)) return;

  const size_t total = size_off_H[nfiles];
  if(total == 0) return;

  bind_row_tables(i);

  const int block = 128;
  const size_t grid = (total + block - 1)/block;
  gq_internal_kernel<<<(unsigned int)grid,block>>>(
      nfiles, turn_, d_param, d_S, d_pair,
      d_row_off_H, d_size_off_H, d_i_H, d_energy_min2,
      gv, gc, gr, ge_, gro);
  gpuErrchk( cudaPeekAtLastError() );
}


//Host (ie non-GPU) code
PRIVATE void
int_loop_cuda(const int nfiles,
	      const int i, /*const int turn,*/ const int length,
	      const vrna_param_t *P,
	      int* energy_min,
	      const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	      const int* i_H) {         //in, nfiles entries -- continuous flow phase A3
  //cf modular_decomposition.cu r1.79
  // Staggered_Row_Batching Phase 5: size_off_H[nfiles] replaces the old
  // scalar nblocks<=0 check -- numerically identical while every H shares
  // one length (today). Flat 1-D grid instead of dim3(nblocks,nfiles) --
  // also lifts the old implicit nfiles<=65535 sub-limit from using nfiles
  // as gridDim.y (gridDim.x supports far more).
  const size_t flat_nblocks = size_off_H[nfiles];
  if(flat_nblocks==0) return;

  bind_row_tables(i);

  dim3 blocks((unsigned int)flat_nblocks);

  // Default 64, MEASURED on two architectures -- see the block-size history
  // above this kernel's four #include instantiations. Still a fixed constant
  // rather than an in-process benchmark: auto-tuning this kernel caused the last
  // two regressions, because it only ever sampled the first (always-tiny)
  // launch. RNA_INT_LOOP_BLOCK_SIZE forces any of the four candidates for a
  // re-test on new hardware, same pattern as this file's other RNA_*- knobs.
  static int block_size = 0;
  if(!block_size) {
    block_size = rnafold_int_loop_warp() ? INT_LOOP_WARP_DEFAULT_BLOCK_SIZE
                                         : INT_LOOP_DEFAULT_BLOCK_SIZE;
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
	    env ? " (from RNA_INT_LOOP_BLOCK_SIZE)" : " (measured default -- see comment above)");
  }

  // RNA_LAUNCH_STATS: bracket the kernel ALONE -- not the two offset uploads
  // above, which belong to the phase but not to the kernel. Splitting those two
  // apart is the entire point (STRESS272_RESULTS.md 20.4).
  rnafold_launch_stats_begin();

  // RNA_INT_LOOP_WARP: one warp per cell, so the grid is sized in WARPS rather
  // than in cells and block_size selects how many independent cells share a
  // block. Same launch-stats bracket, same grid quantity reported, so the
  // per-launch numbers stay comparable between the two kernels.
  if(rnafold_int_loop_warp()) {
    const int    cpb = block_size / 32;
    const size_t nb  = (flat_nblocks + (size_t)cpb - 1)/(size_t)cpb;
    assert(nb <= 2147483647u);

    // H6: take the 2-D grid only when it is nearly free. `maxw` is the widest
    // record in this chunk, so the 2-D grid launches nfiles*ceil(maxw/cpb)
    // blocks where the flat grid launches `nb`; the excess exit on one
    // comparison, but they are still blocks to schedule. gridDim.y is capped
    // at 65535 -- the sub-limit the flattening lifted -- so the flat grid is
    // not a fallback of convenience, it is still the general case.
    size_t maxw = 0;
    for(int H = 0; H < nfiles; H++) {
      const size_t w = size_off_H[H+1] - size_off_H[H];
      if(w > maxw) maxw = w;
    }
    const size_t nbx   = (maxw + (size_t)cpb - 1)/(size_t)cpb;
    const int    wsearch = rnafold_int_loop_wsearch();   // read UNCONDITIONALLY:
                                                        // see the banner note
    const int    gridy = rnafold_int_loop_gridy() &&
                         nfiles > 0 && nfiles <= 65535 && maxw > 0 &&
                         ((size_t)nfiles * nbx) * INT_LOOP_GRIDY_WASTE_DEN
                           <= nb * INT_LOOP_GRIDY_WASTE_NUM;

    // WHICH GRID DID IT ACTUALLY TAKE? The knob only says what was ASKED;
    // the waste guard decides, and a bar that checks the knob rather than
    // the decision is the shape of check this project keeps catching. Print
    // on every CHANGE of decision, so one line proves the 2-D grid ran and a
    // second proves the fallback is reachable.
    if(rnafold_int_loop_gridy()) {
      static int last_decision = -1;

      if(gridy != last_decision) {
        fprintf(stderr,
                "%-24s int_loop grid: %s, lookup: %s "
                "(nfiles %d, maxw %llu, blocks %llu vs flat %llu)\n",
                __FILE__, gridy ? "2-D, blockIdx.y = record" : "flat (waste guard declined)",
                gridy ? "none" : (wsearch ? "32-ary warp" : "binary"),
                nfiles, (unsigned long long)maxw,
                (unsigned long long)((size_t)nfiles * nbx), (unsigned long long)nb);
        last_decision = gridy;
      }
    }

#define IL_WARP_LAUNCH(C, G, W, GRID) int_loop_warp_kernel<C,G,W><<<GRID, 32*(C), 0, rnafold_stream_cell()>>>( \
        nfiles, RNA_I_ROW(i), length, P->TerminalAU, P->ninio[2], d_param, P->lxc, \
        d_pair, d_S, d_hccc, d_up_int, d_my_c, d_tri_off_H, d_row_off_H, d_hc_off_H, \
        d_size_off_H, d_i_H, d_energy_min2)
#define IL_WARP_DISPATCH(G, W, GRID) \
    switch(cpb) { \
      case 8: IL_WARP_LAUNCH(8, G, W, GRID); break; \
      case 4: IL_WARP_LAUNCH(4, G, W, GRID); break; \
      case 2: IL_WARP_LAUNCH(2, G, W, GRID); break; \
      default: IL_WARP_LAUNCH(1, G, W, GRID); break; \
    }
    // WSEARCH only exists on the flat side -- the 2-D grid has no search left
    // to speed up -- so the 2-D arm instantiates it false and does not double
    // the object code for nothing.
    if(gridy) {
      const dim3 g2((unsigned int)nbx, (unsigned int)nfiles);
      IL_WARP_DISPATCH(true, false, g2);
    } else {
      const dim3 g1((unsigned int)nb);
      if(wsearch) IL_WARP_DISPATCH(false, true,  g1)
      else                           IL_WARP_DISPATCH(false, false, g1)
    }
#undef IL_WARP_DISPATCH
#undef IL_WARP_LAUNCH
  } else
  // NOT an early return: the tail of this function still owns the launch-stats
  // end, the error check, the optional sync and -- in non-GPU-resident mode --
  // the D2H of energy_min. Returning here would have skipped that copy and left
  // the host reading a stale row, which is a wrong answer rather than a crash.
  switch(block_size) {
    case 256: int_loop_kernel_256<<<blocks,256,0,rnafold_stream_cell()>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_up_int,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_i_H,
						  d_energy_min2); break; //Out
    case 128: int_loop_kernel_128<<<blocks,128,0,rnafold_stream_cell()>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_up_int,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_i_H,
						  d_energy_min2); break; //Out
    case  64: int_loop_kernel_64<<<blocks, 64,0,rnafold_stream_cell()>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_up_int,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_i_H,
						  d_energy_min2); break; //Out
    default:  int_loop_kernel_32<<<blocks, 32,0,rnafold_stream_cell()>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
						  P->TerminalAU,P->ninio[2],
						  d_param,P->lxc,
						  d_pair,
						  d_S,
						  d_hccc,
						  d_up_int,
						  d_my_c,
						  d_tri_off_H,
						  d_row_off_H,
						  d_hc_off_H,
						  d_size_off_H,
						  d_i_H,
						  d_energy_min2); break; //Out
  }

  rnafold_launch_stats_end((unsigned int)flat_nblocks);

  gpuErrchk( cudaPeekAtLastError() );
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );
  //printf("int_loop_kernel<<<%d.%d,%d>>>(i=%d...) ok\n",blocks.x,blocks.y,block_size,i);

  // GPU-resident sweep: in device mode new_c_kernel reads d_energy_min2 in
  // place, so this copy has no reader. It is one of the six per-row transfers
  // being retired.
  if(!rnafold_gpu_sweep()) {
    gpuErrchk( cudaMemcpy(energy_min,d_energy_min2, g_row_total*sizeof(int),cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaDeviceSynchronize() );
  }

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
	   const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	   const int* i_H ) {        //in, nfiles entries -- continuous flow phase A3
  // Staggered_Row_Batching Phase 2b: this used to have a defensive
  // `if(first2) init_gpu2(...)` fallback here, but int_loop_i() is only ever
  // reached via fill_arrays_loop.c -> par_fill_arrays() -> par_mfe(), which
  // unconditionally calls init_gpu2() (with the real tri_off_H table) before
  // par_fill_arrays() ever runs -- first2 is always already 0 by the time
  // this line executes, so the fallback was dead code. Removed rather than
  // given a fabricated tri_off_H it has no access to here.
  assert(!first2);

  int_loop_cuda(nfiles,i,/*turn,*/length,VC[0]->params, energy_min, size_off_H, i_H);
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
  /* PORT TO 2.7.2: hc->mx is unsigned char (was char), hc->up_int is
   * unsigned int (was int). Widened here rather than cast at each use. */
  unsigned char     *hc, *hc_pq, eval_loop;
  char              *ptype, *ptype_pq;
  short             *S, S_i1, S_j1, *S_p1, *S_q1;
  unsigned int      *hc_up;
  size_t            hc_stride;      /* dense hc->mx row stride, = vc->length */
  int               q, p, j_q, p_i, pq, *c_pq, max_q, max_p, tmp,
                    *rtype, /*noGUclosure, **no_close,*/ energy, cp, //en,
                    *indx, ij, hc_decompose, e, *c, //*ggg,
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
  /* PORT TO 2.7.2: hc->matrix[indx[j]+i] became hc->mx[n*i+j]. This is host
   * code evaluating one (i,j), so the fetch is a direct translation; indx is
   * still needed below for the DP matrices, whose triangular layout upstream
   * did not change. */
  hc            = vc->hc->mx;
  hc_stride     = (size_t)vc->length;
  hc_up         = vc->hc->up_int;
  P             = vc->params;
  matrices      = vc->matrices;
  ij            = indx[j] + i;
  hc_decompose  = hc[hc_stride * i + j];
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
        /* PORT: dense hc->mx -- (p,q) is at hc[n*p + q], and advancing p by one
         * moves a whole row, not one element. c and ptype stay triangular. */
        hc_pq     = hc + hc_stride * (size_t)(i + 1) + q;
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
          hc_pq    += hc_stride;  /* dense hc->mx: next p is a whole row on */
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

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
  //Salt, appended last so no offset above it moves. Indexed by backbone
  //count: an internal loop is bounded by MAXLOOP, so backbones = nl+ns+2
  //tops out at MAXLOOP+2 and MAXLOOP+3 entries always suffice. Note that
  //index MAXLOOP+2 is one PAST upstream's own P->SaltLoop, which upstream
  //reaches by the closed form -- rnafold_build_salt_table() folds both
  //branches in, so the kernel just indexes. All zero at default salt.
  int     SaltStack;
  int     SaltLoop[MAXLOOP+3];
  //The asymmetry cap, appended last for the same reason the salt fields were:
  //no offset above it moves. Carried per-batch because a parameter file can
  //change it (params/io.c:671) -- see the note on MAX_NINIO at the top of this
  //file and PORT_NSP_PARAMFILE_SCOPE.md §2.2.
  int     max_ninio;
  //rtype[], appended last for the same reason. Energy() used to obtain type_2
  //by SWAPPING the index order (pair[S[q]][S[p]]) instead of applying rtype[].
  //That is an exact identity at default settings -- md->rtype[] is BUILT from
  //md->pair[] (model.c:1098-1100, rtype[pair[i][j]] = pair[j][i]) -- and stops
  //being one under --nsp, where model.c:1104 FORCES rtype[7] = 7 over whatever
  //the derivation loop wrote. See PORT_NSP_PARAMFILE_SCOPE.md §1.
  int     rtype[8];
  //VRNA-PATCH free: this is OURS. noGUclosure is a MODEL DETAIL, not an energy
  //parameter, and it sits here for the same reason rtype[] does -- it is read
  //by Energy() and gq_internal_kernel, both of which already take a
  //cuda_param_t* and nothing else. Appended last so no offset above it moves.
  //sanity() asserts every record in a batch agrees on it, so one batch-wide
  //value is sound.
  int     noGUclosure;
  //Dangle model, 0 or 2. Read only by gq_internal_kernel, whose mismatchI term
  //upstream gates on `if (dangles)` (mfe_gquad.c:306). Appended last, like
  //everything else here.
  int     dangles;
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
// GPU-resident sweep, step 5b. The independence the comment above insists on is
// preserved -- neither caller assumes the other ran -- but the second upload of
// an identical table is now skipped, because cudaMemcpy H2D is BLOCKING and so
// is a sync point in its own right. Compared by content, not by call order, so
// no ordering guarantee is being depended on. See upload_size_off_H() below and
// its HAZARD note about the per-chunk reallocation.
static size_t*       size_off_shadow   = NULL;
static int           size_off_shadow_n = 0;
static int           size_off_shadow_pinned = 0;  /* pinned staging, stub2.h */
static int           i_H_shadow_pinned      = 0;
static void          size_off_shadow_reset(void);  // defined below; called from init/teardown above it
// Continuous flow phase A2: this file's own copy of the per-record row index.
// hp_mb_loop.cu carries an identical one for its four kernels -- each
// translation unit keeps its own device tables here, exactly as d_size_off_H
// already does. Same content comparison, same per-chunk reallocation hazard.
static int*          d_i_H;
static int*          i_H_shadow   = NULL;
static int           i_H_shadow_n = 0;
static void          i_H_shadow_reset(void);       // defined below; called from init/teardown above it
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
  SLOT_ALLOC(&d_size_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  // Fresh buffer, holds nothing -- the shadow must not claim it is current.
  size_off_shadow_reset();
  SLOT_ALLOC(&d_i_H, (size_t)nfiles*sizeof(int));
  i_H_shadow_reset();          // fresh buffer: same hazard as size_off, same fix

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
  gpuErrchk( cudaFree(d_S) );
  gpuErrchk( cudaFree(d_my_c) );
  gpuErrchk( cudaFree(d_new_e) );
  gpuErrchk( cudaFree(d_energy_min2) );
  gpuErrchk( cudaFree(d_tri_off_H) );
  gpuErrchk( cudaFree(d_row_off_H) );
  gpuErrchk( cudaFree(d_hc_off_H) );
  gpuErrchk( cudaFree(d_size_off_H) );
  size_off_shadow_reset();   // the device buffer is gone; the shadow must not outlive it
  gpuErrchk( cudaFree(d_i_H) );
  i_H_shadow_reset();
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

// Upload size_off_H only when it differs from what the device already holds.
// Twin of hp_mb_loop.cu's function of the same name (own copy, per this
// codebase's file-ownership convention); see that one for the full rationale.
// Short version: both int_loop_cuda() and load_my_c() upload this table in the
// same sweep row, with the same contents, into the same buffer, and each upload
// is a BLOCKING cudaMemcpy -- a sync point that would defeat step 5b's removal
// of the explicit cudaDeviceSynchronize() calls.
//
// HAZARD: d_size_off_H is re-cudaMalloc'd per chunk, so the shadow MUST be
// dropped there (init_gpu2 / teardown_gpu2 both call the reset). Otherwise a
// new chunk whose first table matched the previous chunk's last would skip the
// upload into an uninitialised buffer -- wrong offsets, no crash.
// Twin of upload_size_off_H() for the per-record row index: same content
// comparison, same per-chunk reallocation hazard, same fix.
static void
i_H_shadow_reset(void) {
  rnafold_pinned_free(i_H_shadow, i_H_shadow_pinned);
  i_H_shadow   = NULL;
  i_H_shadow_n = 0;
}

static void
upload_i_H(const int nfiles, const int* i_H) {
  const size_t bytes = (size_t)nfiles * sizeof(int);
  if(i_H_shadow_n != nfiles) {
    rnafold_pinned_free(i_H_shadow, i_H_shadow_pinned);
    i_H_shadow   = (int*)rnafold_pinned_alloc(bytes, &i_H_shadow_pinned);
    i_H_shadow_n = i_H_shadow ? nfiles : 0;
  } else if(i_H_shadow && memcmp(i_H_shadow, i_H, bytes) == 0) {
    return;
  }
  /* Stage into the PINNED shadow and copy from there -- see the note on
   * rnafold_pinned_alloc() in stub2.h. The shadow had to be written anyway; all
   * that changes is that it is now also the copy source. */
  if(i_H_shadow) {
    memcpy(i_H_shadow, i_H, bytes);
    gpuErrchk( cudaMemcpy(d_i_H, i_H_shadow, bytes, cudaMemcpyHostToDevice) );
  } else {
    gpuErrchk( cudaMemcpy(d_i_H, i_H, bytes, cudaMemcpyHostToDevice) );
  }
}

static void
size_off_shadow_reset(void) {
  rnafold_pinned_free(size_off_shadow, size_off_shadow_pinned);
  size_off_shadow   = NULL;
  size_off_shadow_n = 0;
}

static void
upload_size_off_H(const int nfiles, const size_t* size_off_H) {
  const int    n     = nfiles + 1;
  const size_t bytes = (size_t)n * sizeof(size_t);

  if(size_off_shadow_n != n) {
    rnafold_pinned_free(size_off_shadow, size_off_shadow_pinned);
    size_off_shadow   = (size_t*)rnafold_pinned_alloc(bytes, &size_off_shadow_pinned);
    size_off_shadow_n = size_off_shadow ? n : 0;
  } else if(size_off_shadow && memcmp(size_off_shadow, size_off_H, bytes) == 0) {
    return;
  }

  if(size_off_shadow) {
    memcpy(size_off_shadow, size_off_H, bytes);
    gpuErrchk( cudaMemcpy(d_size_off_H, size_off_shadow, bytes, cudaMemcpyHostToDevice) );
  } else {
    gpuErrchk( cudaMemcpy(d_size_off_H, size_off_H, bytes, cudaMemcpyHostToDevice) );
  }
}

//perhaps this can be combined with other kernels?
__global__ void
load_my_c_kernel(const int nfiles, const int i_row, /*const int turn,*/ const int length,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c,
		 const size_t* __restrict__ tri_off_H, //in
		 const size_t* __restrict__ row_off_H, //in
		 const size_t* __restrict__ size_off_H, const size_t total, //in
		 const int* __restrict__ i_H) { //in
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  // Continuous flow phase A2: the row index is now this record's own. Every
  // entry equals the old shared scalar i_row today, and the assert proves that
  // at RUNTIME rather than by argument -- it is exactly the property that stops
  // holding in phase B, so a divergent table traps instead of folding silently
  // wrong.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
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
  upload_size_off_H(nfiles, size_off_H);   // skips this row's redundant re-upload
  upload_i_H(nfiles, i_H);                 // continuous flow phase A2


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

  load_my_c_kernel<<<nblocks,block_size>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
					   d_new_e,  //in
					   d_my_c,   //out
					   d_tri_off_H,  //in
					   d_row_off_H,  //in
					   d_size_off_H, total, d_i_H);
  gpuErrchk( cudaPeekAtLastError() );
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
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
// BLOCK_SIZE=32 below is a placeholder for the shared device helpers only --
// Hc()/decode_column()/Energy() below are compiled once and used by every
// instantiation regardless of block size, so its value is irrelevant to them.
// It is NOT the default any more: that is INT_LOOP_DEFAULT_BLOCK_SIZE (64),
// named beside the four instantiations further down.
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
	      //
	      // Upstream (mfe/internal.c, via vrna_get_ptype()) computes:
	      //     type   = vrna_get_ptype(ij, ptype);          // 0 -> 7
	      //     type_2 = rtype[vrna_get_ptype(pq, ptype)];   // promote, THEN rtype
	      // This site used to do neither: it took the raw pair value for `type`
	      // and obtained `type_2` by swapping the index order. Both are exact
	      // identities at default settings -- rtype[] is built from pair[], and
	      // a ptype-0 cell is refused by the hard-constraint mask before
	      // Energy() runs -- which is why the port was byte-identical for a year
	      // with them. --nsp breaks the second: model.c:1104 FORCES rtype[7]=7,
	      // so with an asymmetric spec upstream reads row 7 (the NST/NSM
	      // non-standard row) of stack/int11/int21/int22/mismatchI where this
	      // read row 0. A different energy model, PARTIALLY applied.
	      //
	      // The fork's other two type-resolution sites already did it properly:
	      // stack_row_kernel() promotes then applies rtype[], and
	      // hp_mb_3p_kernel() applies rtype[] with a documented raw-index
	      // convention. Only Energy() did neither, so under --nsp the fork
	      // disagreed with ITSELF. See PORT_NSP_PARAMFILE_SCOPE.md §1.
	      const unsigned char type_raw = Ptype(S,pair_,H,nfiles,i,j);
	      const unsigned char type     = (type_raw == 0) ? 7 : type_raw;
	      assert(type<8);
	      // p,q -- NOT q,p. The reversal is rtype[]'s job, not the index's.
	      const unsigned char t2_raw   = Ptype(S,pair_,H,nfiles,p,q);
	      assert(t2_raw<8);
	      const unsigned char type_2   = (unsigned char)P->rtype[(t2_raw == 0) ? 7 : t2_raw];
	      assert(type_2<8);
	      //assert(i+pp  >=0 && i+pp  <length+2);

	      const int u1 = p - 1 - i; //u1 = p1 - i;
	      const int u2 = j - 1 - q; //u2 = j1 - q;

	      // --noClosingGU, the interior-loop half (upstream:
	      // mfe/mfe_internal.c:305 noclose, :339/:415/:579 the type2 skips).
	      // A GU/UG pair may neither CLOSE nor BE ENCLOSED BY an interior
	      // loop or bulge. STACKS ARE EXEMPT -- upstream's mfe_stacks() has
	      // no such test at all, and vrna_E_internal() returns the stack
	      // energy BEFORE consulting no_close (eval/eval_internal.c:104-108).
	      // u1==0 && u2==0 is exactly the stack, so (u1||u2) reproduces that.
	      //
	      // WHY THE SKIP IS HERE AND NOT INSIDE IntLoop_X(), even though
	      // upstream's own guard lives in vrna_E_internal(): the caller does
	      // `energy = my_c[pq]; energy += IntLoop_X(...)`. An INF returned
	      // from the callee would be ADDED to a real negative c[pq] and land
	      // just BELOW INF, winning the min -- the same near-INF sentinel
	      // mechanism recorded in PORT_INVESTIGATIONS.md item 3. Upstream
	      // avoids it the same way, with explicit `continue`s before the
	      // call; its in-callee test is unreachable from this path.
	      //
	      // type is the PROMOTED enclosing ptype and type_2 is rtype[] of the
	      // promoted inner one, matching upstream literally. Neither
	      // transform can move a value into or out of {3,4}: 0->7 never
	      // lands on 3 or 4, and rtype[] only swaps 3<->4.
	      if (P->noGUclosure && (u1 || u2) &&
	          ((type == 3) || (type == 4) || (type_2 == 3) || (type_2 == 4)))
	        return INF;

	      const int ns = (u1>u2)? u2 : u1;
	      const int nl = (u1>u2)? u1 : u2;

	      const int si1 = unpack(S,H,nfiles,i+1);
	      const int sj1 = unpack(S,H,nfiles,j-1);
	      const int sp1 = unpack(S,H,nfiles,i+pp);
	      const int sq1 = unpack(S,H,nfiles,q+1);

	      energy += IntLoop_X(u1, ns, nl, type, type_2,
				  si1, sj1, sp1, sq1,
				  TerminalAU,ninio2,P->max_ninio,
				  P->bulge,P->internal_loop,lxc,
				  mismatchI,
				  mismatch1nI,
				  mismatch23I,
				  P->stack,
				  P->int11,
				  P->int21,
				  P->int22,
				  P->SaltStack, P->SaltLoop);
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

#define INT_LOOP_WARP_KERNEL_NAME2(cpb) int_loop_warp_kernel_##cpb
#define INT_LOOP_WARP_KERNEL_NAME(cpb) INT_LOOP_WARP_KERNEL_NAME2(cpb)

template <int CELLS_PER_BLOCK>
__global__ void
int_loop_warp_kernel(const int nfiles, const int i_row, const int length,
                const int TerminalAU, const int ninio2,
                const cuda_param_t* __restrict__ P, const float lxc,
                const char* __restrict__ pair_,
                const unsigned int* __restrict__ S,
                const unsigned int* __restrict__ hccc,
                const int* __restrict__ my_c,
                const size_t* __restrict__ tri_off_H,
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ hc_off_H,
                const size_t* __restrict__ size_off_H,
                const int* __restrict__ i_H,
                      int* __restrict__ energy_min) {
  static_assert(CELLS_PER_BLOCK >= 1 && CELLS_PER_BLOCK <= 32,
                "one warp per cell; blockDim.x must be CELLS_PER_BLOCK*32");

  const int    lane = (int)(threadIdx.x & 31u);
  const int    wib  = (int)(threadIdx.x >> 5);          // warp within the block
  const size_t cell = (size_t)blockIdx.x * CELLS_PER_BLOCK + wib;

  // Whole-warp exit: every lane of this warp shares `cell`, so no lane is left
  // behind to be named by a shuffle mask below. This is the one thing the
  // block-per-cell twin could not do (its warps straddle cells).
  if(cell >= size_off_H[nfiles]) return;

  const int H = flatten_index_to_H(cell, size_off_H, nfiles);
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = (int)(cell - size_off_H[H]) + i + turn + 1;

  const long long ij = Indx(i,j);
  int energy = INF;

  if(Hc(ij,&hccc[hc_off_H[H]])) {
    const int p0 = i+1;
    const int q0 = Min_q(i,j,turn);
    const int maxcol = MIN2(MAXLOOP,(j - 1) - q0);
    const unsigned int* __restrict__ hccc_H = &hccc[hc_off_H[H]];

    // LANE c OWNS COLUMN c, in registers. maxcol <= MAXLOOP = 30 always, so one
    // warp covers every column there can be -- which is what makes the whole
    // design possible.
    unsigned int my_mask = 0u;
    int          popc    = 0;
    if(lane <= maxcol) {
      my_mask = decode_column(p0,q0,lane,hccc_H);
      popc    = __popc(my_mask);
    }

    // Inclusive Hillis-Steele scan of popc, unchanged from the twin except that
    // the result stays in registers. Lanes past maxcol contribute 0 and cannot
    // affect lower lanes, since __shfl_up_sync only pulls from lower lanes.
    int incl = popc;
#pragma unroll
    for(int off = 1; off < 32; off <<= 1) {
      const int nv = __shfl_up_sync(0xffffffff, incl, off);
      if(lane >= off) incl += nv;
    }
    const int excl  = incl - popc;                              // prefix[c]
    const int total = __shfl_sync(0xffffffff, incl, 31);        // prefix[maxcol+1]

    const int iters = (total + 31) >> 5;                        // warp-uniform
    for(int k = 0; k < iters; k++) {
      const int  w   = (k << 5) + lane;
      const bool has = (w < total);

      // Smallest column c with prefix[c+1] > w. FIVE FIXED STEPS, not
      // `while(lo < hi)`: 2^5 = 32 covers every column, and a fixed trip count
      // keeps all 32 lanes in every shuffle. Lanes that have already converged
      // (lo == hi) still execute the shuffle and discard it.
      int lo = 0;
      int hi = (maxcol > 0) ? maxcol : 0;
#pragma unroll
      for(int s = 0; s < 5; s++) {
        const int  mid = (lo + hi) >> 1;
        const int  v   = __shfl_sync(0xffffffff, incl, mid);
        const bool act = (lo < hi);
        const bool go  = act && (v > w);
        hi = go ? mid : hi;
        lo = (act && !go) ? (mid + 1) : lo;
      }
      const int column = lo;

      // Column `column`'s mask and prefix, gathered from the lane that owns
      // them. A per-lane source index is a legal gather, and both are register
      // reads -- this is the shared-memory traffic the twin pays.
      const unsigned int cmask = __shfl_sync(0xffffffff, my_mask, column);
      const int          cbase = __shfl_sync(0xffffffff, excl,    column);

      if(has) {
        int popc_unused;
        const int row = find_nth_set_bit(cmask, w - cbase, popc_unused);
        assert(row >= 0);
        const int p = p0 + row;
        const int q = q0 + column;
        const int energy2 = Energy(H,nfiles,i,j,q,p,
                      &my_c[tri_off_H[H]],
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
      }
    }
  }

  // Warp-wide min. No shared memory and no __syncthreads() at any block size --
  // which is the whole point.
#pragma unroll
  for(int off = 16; off > 0; off >>= 1)
    energy = MIN2(energy, __shfl_down_sync(0xffffffff, energy, off));

  if(lane == 0) energy_min[row_off_H[H]+j] = energy;
}

// The default, named here beside the instantiations so it cannot drift away
// from them. 64 since 2026-09-11; see the block-size history immediately below.
#define INT_LOOP_DEFAULT_BLOCK_SIZE 64

// RNA_INT_LOOP_WARP=1 -- route to int_loop_warp_kernel, which owns one (H,j)
// cell per WARP instead of per block. An EXPERIMENT with a measured premise:
// STRESS272_RESULTS.md 22.5 puts `barrier` at 20.7% of stall cycles at block
// size 128 and `short_scoreboard` at 8.9%, both of them this kernel's own scan
// machinery, while global memory is only 12-16%. Off by default until it beats
// its twin on wall clock with an unchanged sha.
PUBLIC int
rnafold_int_loop_warp(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_INT_LOOP_WARP");

    v = (e && e[0] && e[0] != '0') ? 1 : 0;

    if (v)
      fprintf(stderr, "%-24s RNA_INT_LOOP_WARP=1: one warp per cell, no shared "
                      "memory and no __syncthreads() (experimental; sha must not "
                      "move)\n", __FILE__);
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
gq_internal_i(const int nfiles, const int turn_, const size_t* size_off_H,
              const int* i_H) {
  const int* gv; const unsigned int *gc, *gr; const size_t *ge_, *gro;

  if(!rnafold_gq_csr_device(&gv,&gc,&gr,&ge_,&gro)) return;

  const size_t total = size_off_H[nfiles];
  if(total == 0) return;

  upload_size_off_H(nfiles, size_off_H);
  upload_i_H(nfiles, i_H);

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

  upload_size_off_H(nfiles, size_off_H);   // skips this row's redundant re-upload
  upload_i_H(nfiles, i_H);                 // continuous flow phase A3 (content-deduped, as above)

  dim3 blocks((unsigned int)flat_nblocks);

  // Default 64, MEASURED on two architectures -- see the block-size history
  // above this kernel's four #include instantiations. Still a fixed constant
  // rather than an in-process benchmark: auto-tuning this kernel caused the last
  // two regressions, because it only ever sampled the first (always-tiny)
  // launch. RNA_INT_LOOP_BLOCK_SIZE forces any of the four candidates for a
  // re-test on new hardware, same pattern as this file's other RNA_*- knobs.
  static int block_size = 0;
  if(!block_size) {
    block_size = INT_LOOP_DEFAULT_BLOCK_SIZE;
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
#define IL_WARP_LAUNCH(C) int_loop_warp_kernel<C><<<(unsigned int)nb, 32*(C)>>>( \
        nfiles, RNA_I_ROW(i), length, P->TerminalAU, P->ninio[2], d_param, P->lxc, \
        d_pair, d_S, d_hccc, d_my_c, d_tri_off_H, d_row_off_H, d_hc_off_H, \
        d_size_off_H, d_i_H, d_energy_min2)
    switch(cpb) {
      case 8: IL_WARP_LAUNCH(8); break;
      case 4: IL_WARP_LAUNCH(4); break;
      case 2: IL_WARP_LAUNCH(2); break;
      default: IL_WARP_LAUNCH(1); break;
    }
#undef IL_WARP_LAUNCH
  } else
  // NOT an early return: the tail of this function still owns the launch-stats
  // end, the error check, the optional sync and -- in non-GPU-resident mode --
  // the D2H of energy_min. Returning here would have skipped that copy and left
  // the host reading a stale row, which is a wrong answer rather than a crash.
  switch(block_size) {
    case 256: int_loop_kernel_256<<<blocks,256>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
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
						  d_i_H,
						  d_energy_min2); break; //Out
    case 128: int_loop_kernel_128<<<blocks,128>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
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
						  d_i_H,
						  d_energy_min2); break; //Out
    case  64: int_loop_kernel_64<<<blocks, 64>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
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
						  d_i_H,
						  d_energy_min2); break; //Out
    default:  int_loop_kernel_32<<<blocks, 32>>>(nfiles, RNA_I_ROW(i), /*turn,*/ length,
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

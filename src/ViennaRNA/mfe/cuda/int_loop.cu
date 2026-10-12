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

/* vrna_cuda_quiet(): engine.c. On when a library caller (the Python binding) has
 * asked for the routine diagnostics below to be left out; errors still print. */
extern "C" int vrna_cuda_quiet(void);

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
/* The device helpers below live in int_loop_dev.h; see the header. */
#include "int_loop_dev.h"


cuda_param_t* d_param;
char*         d_pair; //[NBPAIRS+1][NBPAIRS+1];
/* RNA_STREAM_OVERLAP (device.cu). Both return the NULL stream until the knob is
 * on, so every launch below is exactly where it has always been by default. */
extern "C" cudaStream_t rnafold_stream_cell(void);
extern "C" cudaStream_t rnafold_stream_hp(void);

unsigned int* d_hccc; //read via Hc
// Host mirror of d_hc_off_H, for RNA_HC_VERIFY. See int_loop_hc_off_host().
static size_t* g_hc_off_host   = NULL;
static int     g_hc_off_host_n = 0;

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
// Words per record in d_S, fixed when the batch is packed. gq_internal_i() is
// launched per row and never sees the batch , so the stride is kept
// here rather than recomputed from a value that is not in scope.
static int g_s_stride = 0;
int*          d_my_c;
int*          d_energy_min2; //share with modular_decomposition.cu ?
int*          d_new_e;
// RNA_C_RING (see c_ring_store_kernel): the last 32 c rows, row-major, one slot per
// row (slot p & 31), each slot laid out like the row buffers (row_off_H[H] + j).
// g_c_ring_on is this chunk's decision -- the int_loop launch reads it to pick the
// reader -- and g_c_pend rows (the newest being g_c_pend_i) are still only in the ring.
static int*   d_c_ring    = NULL;
static int    g_c_ring_on = 0;
static int    g_c_pend    = 0;
static int    g_c_pend_i  = 0;
static long   g_c_rows    = 0;      // rows ringed / flushes issued, this chunk:
static long   g_c_flushes = 0;      //   positive evidence, printed by rnafold_c_ring_end()
static int    rnafold_c_ring(void);
static int    c_ring_asked(void);
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
  const size_t m = (size_t)blockIdx.x*blockDim.x+threadIdx.x; // 32-bit signed integer overflow bug fix
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

#define SLOT_ALLOC(pp, sz) TIMED_CUDAMALLOC(pp, sz)

/* Lyngsø S1 selftest (int_loop_lyngso.inc, included further down). */
static void ly_alloc(const int nfiles, const size_t* row_off_H);
static void ly_free(void);
static int  ly_selftest(void);
static int  ly_eval_knob(void);
static int  ly_ering(void);

PUBLIC void
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size,
          const size_t* tri_off_H, const size_t* row_off_H, //in, nfiles+1 entries each, mfe_cuda.c
          const size_t* cap_H) { //in, nfiles record capacities in nt
  if(!first2) return;
  const double _t_ig2 = rnafold_now_seconds();
  if (!vrna_cuda_quiet()) fprintf(stderr,"%-24s init_gpu2(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

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
  // Published for RNA_HC_VERIFY (see int_loop_hc_off_host). Rebuilt every call,
  // including a refill, because cap_H can change between chunks.
  if(g_hc_off_host_n != nfiles + 1) {
    free(g_hc_off_host);
    g_hc_off_host   = (size_t *) malloc((size_t)(nfiles+1)*sizeof(size_t));
    g_hc_off_host_n = g_hc_off_host ? nfiles + 1 : 0;
  }
  if(g_hc_off_host)
    memcpy(g_hc_off_host, hc_off_H, (size_t)(nfiles+1)*sizeof(size_t));
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
    // pack_hc_kernel writes only the first Hc_ints2(cap) words of each record's
    // Hc_ints(cap) block here; the rest is MAXLOOP padding that has to read as
    // zero. So this memset and that pack are a matched pair, with the SAME scope.
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

  // Ten bases per word, PER RECORD (see unpack()): record H owns
  // [H*S_STRIDE, (H+1)*S_STRIDE) words. Slightly larger than the old
  // H-fastest packing because each record rounds up to a whole word -- at
  // most 9 bases of padding per record.
  assert(sizeof(unsigned int) == 4);
  g_s_stride = S_STRIDE(length);
  size = (size_t)nfiles * S_STRIDE(length) * sizeof(unsigned int);
  SLOT_ALLOC(&d_S, size);
  unsigned int* buff = (unsigned int*) malloc(size); //could use cudaMallocHost
  const double _t_pk2 = rnafold_now_seconds();
#ifndef NDEBUG
  memset(buff,0xff,size);
#endif
  {
    const int len  = (length+2);
    const int W10  = S_STRIDE(length);
    for(int H=0;H<nfiles;H++){
      for(int w=0;w<W10;w++){
        unsigned int word = 0;

        for(int t=0;t<10;t++){
          const int i = w*10 + t;
          // Past this record's own sequence_encoding (sized to its length+2)
          // substitute 0: the batch `length` can exceed an individual H's, and
          // Phase 6d's active/join mask means such a position is never consumed
          // in a real energy calculation.
          const unsigned int sc = ((i < len) && (i <= (int)VC[H]->length+1))
                                ? VC[H]->sequence_encoding[i] : 0;

          assert(sc <= 4);
          word |= (sc << (t*3));
        }
        buff[(size_t)H*W10 + w] = word;
      }
    }
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
    init_my_c(my_c_elems);
  }

  // Staggered_Row_Batching Phase 6d: cached for the per-row entry points that
  // transfer these buffers whole but never see the offset table.
  g_row_total = row_off_H[nfiles];

  size = g_row_total*sizeof(int); // 32-bit signed integer overflow bug fix
  SLOT_ALLOC(&d_new_e, size);

  // Staggered_Row_Batching Phase 2d: table-driven total (row_off_H[nfiles]),
  // matching the int_loop kernel's row_off_H[H]+j write -- equals
  // the old uniform nfiles*(length+1) exactly while chunks stay uniform-length,
  // diverges once they don't.
  SLOT_ALLOC(&d_energy_min2, g_row_total*sizeof(int));

  // RNA_C_RING: decided and sized per chunk (a slot is this chunk's row_total), and
  // INF everywhere, so a cell outside a row's size range reads exactly what the
  // triangle's INF prefill would give.
  {
    g_c_ring_on = rnafold_c_ring();
    g_c_pend = 0; g_c_rows = 0; g_c_flushes = 0;
    if(g_c_ring_on) {
      const size_t n = (size_t)32 * g_row_total;
      TIMED_CUDAMALLOC(&d_c_ring, n*sizeof(int));
      init_my_c_kernel<<<(unsigned int)((n + BLOCK_SIZE - 1)/BLOCK_SIZE),BLOCK_SIZE>>>(n, d_c_ring);
      gpuErrchk( cudaPeekAtLastError() );
      // legacy stream; the ring's writers are on the non-blocking cell stream
      gpuErrchk( cudaDeviceSynchronize() );
    }
  }
  /*no longer in use
  SLOT_ALLOC(&d_energy_min20, size);

  size = nfiles*length*sizeof(int);
  SLOT_ALLOC(&d_buf, size);
  */
  ly_alloc(nfiles, row_off_H);   /* Lyngsø G buffers (default on; RNA_INT_LOOP_LYNGSO=0 or a refusal -> no-op) */
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
  ly_free();          // RNA_INT_LOOP_LYNGSO_SELFTEST: counters first (a sync), then by the pointer
  gpuErrchk( cudaFree(d_hccc) );
  if(d_up_int) { gpuErrchk( cudaFree(d_up_int) ); d_up_int = NULL; }
  gpuErrchk( cudaFree(d_S) );
  gpuErrchk( cudaFree(d_my_c) );
  gpuErrchk( cudaFree(d_new_e) );
  gpuErrchk( cudaFree(d_energy_min2) );
  if(d_c_ring) { gpuErrchk( cudaFree(d_c_ring) ); d_c_ring = NULL; }   // by the pointer
  g_c_ring_on = 0;
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
  // RNA_C_RING: 32 row slots. Charged when ASKED, not when granted -- the refusals
  // are decided per chunk, after the budget, and over-estimating is the safe side.
  const size_t c_ring_bytes       = c_ring_asked() ? (size_t)32*(length+1)*sizeof(int) : 0;
  // RNA_INT_LOOP_LYNGSO (or its selftest): two row buffers of G, 25 planes each (u = 6..30,
  // MAXLOOP - 5 of them). Charged when asked, like the ring, for the same reason.
  const size_t ly_bytes           = (ly_eval_knob() || ly_selftest())
                                    ? (size_t)2*(MAXLOOP - 5)*(length+1)*sizeof(int) : 0;
  // S6b: RNA_INT_LOOP_LYNGSO_ERING's 32-row e ring, the c ring's size.
  const size_t ly_e_bytes         = (ly_ering() && (ly_eval_knob() || ly_selftest()))
                                    ? (size_t)32*(length+1)*sizeof(int) : 0;
  return hccc_bytes + s_bytes + my_c_bytes + new_e_bytes + energy_min2_bytes + c_ring_bytes + ly_bytes + ly_e_bytes;
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

// N5a (bt_retired.h): hand this chunk's c triangle to a retired set. NULLing the
// global is what keeps teardown_gpu2() off it (cudaFree(NULL) is a no-op), and
// the next init_gpu2() allocates a fresh one.
void rnafold_d2h_nb_w(void* dst, const void* src, const size_t bytes, const int w);

extern "C" int *
rnafold_my_c_detach(void) {
  int *p = d_my_c;
  d_my_c = NULL;
  return p;
}

extern "C" void
rnafold_retired_fetch_my_c_w(const rnafold_retired_t *r, int *dst,
                             const size_t tri_lo, const size_t cells, const int w) {
  assert(r && r->d_my_c && (tri_lo + cells <= r->cells));
  rnafold_d2h_nb_w(dst, &r->d_my_c[tri_lo], cells*sizeof(int), w);
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

// The HOST side of the table above, for RNA_HC_VERIFY only.
//
// WHY IT EXISTS (2026-09-27). RNA_HC_VERIFY checked hccc_mb/mbenc/any/gu and NOT
// d_hccc, the interior-loop mask -- the one buffer pack_hc_kernel writes through a
// DIFFERENT offset table. So when a scoped refill left other slots' interior-loop
// masks zeroed, it reported "174692 words x 4 masks, 0 mismatching words" on a run
// whose fold output was wrong at every k. A mixed-length sha comparison caught it;
// the dedicated mask verifier did not. hp_mb_loop.cu needs this table to close that.
PUBLIC const size_t*
int_loop_hc_off_host(int* n_out) {
  if(n_out) *n_out = g_hc_off_host_n;
  return g_hc_off_host;
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
// load_my_c's cell, shared through a header (device code cannot cross a translation unit).
#include "int_loop_cells.inc"

__global__ void
load_my_c_kernel(const int nfiles, const int i_row, /*const int turn,*/ const int length,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c,
		 const size_t* __restrict__ tri_off_H, //in
		 const size_t* __restrict__ row_off_H, //in
		 const size_t* __restrict__ size_off_H, const size_t total, //in
		 const int* __restrict__ i_H) {
  // The arithmetic lives in int_loop_cells.inc.
  load_my_c_cell(nfiles, i_row, length, new_e, my_c, tri_off_H, row_off_H, size_off_H, total, i_H,
                 (size_t)blockIdx.x*blockDim.x+threadIdx.x);
}

/* ===================== RNA_C_RING: the c triangle, batched =====================
 *
 * The c twin of RNA_ROW_BATCH (modular_decomposition.cu). load_my_c writes ONE row
 * of a column-major triangle, so every element is its own sector (31.96 sectors per
 * store request, RTX 3050 ncu). Here the row goes to a 32-row ring instead -- a
 * contiguous, coalesced store -- and the triangle is written RNA_C_FLUSH_ROWS rows
 * at a time, where a column's pending rows are CONTIGUOUS (Indx(i,j) = j(j-1)/2 + i).
 *
 * WHY THE RING HAS 32 ROWS AND NOT RNA_C_FLUSH_ROWS: unlike fML (md(r) reads rows
 * >= r+turn+2, so staging 5 is free), int_loop(i) reads c rows i+1..i+31 -- MAXLOOP
 * -- starting with the row just written. So int_loop must READ the ring
 * (c_ring_reader, int_loop_dev.h), and the ring must hold every row it can reach;
 * those 31 rows are distinct mod 32. The triangle then has no reader during the
 * sweep at all (md reads c from row buffers; row fusion does write it, and is
 * refused below), so the
 * flush only has to beat the ring's own overwrite: row r's slot is reused by row
 * r-32, and with K <= 32 every flush lands, in cell-stream order, before that.
 *
 * Only reorders writes: RNA_TRI_CHECKSUM=1 must match the ring-off run exactly. */
static int
c_ring_asked(void)
{
  static int v = -1;
  /* DEFAULT ON since 2026-10-02 (A100: c writes -63 % per row, wall -1.1..-1.5 % at
   * 400x5601 and -2.5 % at 3000x1200, triangles identical). RNA_C_RING=0 turns it off. */
  if(v < 0) { const char* e = getenv("RNA_C_RING"); v = (e && e[0]) ? (e[0] != '0') : 1; }
  return v;
}

/* RNA_C_FLUSH_ROWS=K, 1..32 (default 16): rows per triangle flush. K=1 keeps the
 * ring's reads and drops the batching -- the control that splits the two effects. */
static int
c_flush_rows(void)
{
  static int v = -1;
  if(v < 0) {
    const char* e = getenv("RNA_C_FLUSH_ROWS");
    v = (e && e[0]) ? atoi(e) : 16;
    if(v < 1)  v = 1;
    if(v > 32) v = 32;
  }
  return v;
}

static int
c_env_on(const char* name)
{
  const char* e = getenv(name);
  return (e && e[0] && e[0] != '0') ? 1 : 0;
}


/* This chunk's decision. Every refusal is silent in the answer (the triangle path
 * is byte-identical), so say which, once. */
static int
rnafold_c_ring(void)
{
  if(!c_ring_asked()) return 0;
  const char* why = NULL;
  if(!rnafold_gpu_sweep())                why = "off the GPU-resident sweep the host path owns new_C";
  else if(c_env_on("RNA_ROW_FUSE"))       why = "RNA_ROW_FUSE writes the triangle from its fused kernel";
  static int said = 0;
  if(!said) {
    said = 1;
    if(why) fprintf(stderr,"%-24s RNA_C_RING REFUSED: %s -- the c triangle is written row by row\n",
                    __FILE__, why);
    else if (!vrna_cuda_quiet()) fprintf(stderr,"%-24s RNA_C_RING ACTIVE: int_loop reads c from a 32-row ring; the c "
                           "triangle is written %d rows at a time (RNA_C_FLUSH_ROWS)\n",
                    __FILE__, c_flush_rows());
  }
  return why ? 0 : 1;
}

/* Row i of c into ring slot i & 31, over the row's size range only: what lies
 * outside it stays INF, as in the triangle. */
__global__ void
c_ring_store_kernel(const int nfiles, const int i,
                    const int* __restrict__ new_e, int* __restrict__ ring, const size_t stride,
                    const size_t* __restrict__ row_off_H, const size_t* __restrict__ size_off_H,
                    const size_t total)
{
  const size_t m = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(m >= total) return;
  const int    H = flatten_index_to_H(m, size_off_H, nfiles);
  const size_t o = row_off_H[H] + (m - size_off_H[H]) + (size_t)(i + turn + 1 + (int)size_off_H[nfiles + 1]);
  ring[(size_t)(i & 31)*stride + o] = new_e[o];
}

/* Rows i_new .. i_new+npend-1 from the ring into the triangle, in one pass. The
 * shape of md_flush_rows_kernel: a block is TJ consecutive columns of row i_new's
 * size range (the widest pending row); read the tile row by row (consecutive
 * threads, consecutive j: coalesced), then write column by column (consecutive
 * threads, consecutive ROWS of one column: contiguous in the triangle). The +1 pad
 * keeps the column-wise shared read off a single bank -- with 32 pending rows,
 * TJ = 128 without it is a 32-way conflict. */
template<int TJ>
__global__ void
c_flush_rows_kernel(const int nfiles, const int i_new, const int npend,
                    const int* __restrict__ ring, const size_t stride,
                    int* __restrict__ my_c,
                    const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                    const size_t* __restrict__ size_off_H, const size_t total)
{
  __shared__ int sv[32][TJ + 1];
  __shared__ int sH[TJ], sj[TJ];
  const int    t = threadIdx.x;
  const size_t m = (size_t)blockIdx.x * TJ + t;
  int H = -1, j = 0;

  if(m < total) {
    H = flatten_index_to_H(m, size_off_H, nfiles);
    j = (int)(m - size_off_H[H]) + i_new + turn + 1 + (int)size_off_H[nfiles + 1];
    const size_t o = row_off_H[H] + j;
    for(int q = 0; q < npend; q++)
      sv[q][t] = ring[(size_t)((i_new + q) & 31)*stride + o];
  }
  sH[t] = H; sj[t] = j;
  __syncthreads();

  for(int k = t; k < TJ * npend; k += TJ) {             // row fastest: contiguous per column
    const int c = k / npend, q = k % npend;
    const int Hc = sH[c], jc = sj[c], r = i_new + q;
    if(Hc < 0 || jc < r + turn + 1) continue;           // row r has no column jc
    my_c[tri_off_H[Hc] + Indx(r, jc)] = sv[q][c];
  }
}

/* Flush the g_c_pend ringed rows, the newest being g_c_pend_i, on the cell stream --
 * the ring's writer, so the slot reuse that bounds K is ordered by the stream. */
static void
c_ring_flush(const int nfiles)
{
  const int npend = g_c_pend, i_new = g_c_pend_i;
  g_c_pend = 0;
  if(npend == 0) return;
  const size_t total = rnafold_rowtab_size_host(i_new)[nfiles];
  if(total == 0) return;
  bind_row_tables(i_new);
  const int TJ = 128;
  c_flush_rows_kernel<128><<<(unsigned int)((total + TJ - 1)/TJ),TJ,0,rnafold_stream_cell()>>>(
      nfiles, i_new, npend, d_c_ring, g_row_total, d_my_c, d_tri_off_H, d_row_off_H,
      d_size_off_H, total);
  gpuErrchk( cudaPeekAtLastError() );
  g_c_flushes++;
}

/* End of a chunk's sweep (fill_arrays_loop.c, before the row tables are released):
 * whatever is still only in the ring goes to the triangle, for the backtrack. */
extern "C" void
rnafold_c_ring_end(const int nfiles)
{
  if(!g_c_ring_on) return;
  c_ring_flush(nfiles);
  if (!vrna_cuda_quiet()) fprintf(stderr,"%-24s RNA_C_RING: %ld rows ringed, %ld triangle flushes this chunk\n",
          __FILE__, g_c_rows, g_c_flushes);
}

PUBLIC void
load_my_c(const int nfiles,
	  const int i, const int turn_, const int length,
	  const int* new_e,
	  const size_t* size_off_H,     //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	  const int* i_H) {             //in, nfiles entries -- per-record row index
  //out d_my_c
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

#ifdef NDEBUG
  //check here in case of earlier errors
  //
  // INVERTED, AND LIVE SINCE 709a072e (2026-09-05). "In case of earlier errors" is a
  // debug check, but NDEBUG is the RELEASE macro: this was dead until -DNDEBUG first
  // reached nvcc, and since then every release build has drained the whole device
  // once per sweep row, here. RNA_ROW_FUSE never calls load_my_c(), so it deletes
  // this sync as a side effect -- which confounds the fusion's measured win.
  //
  // OFF BY DEFAULT since 2026-09-30 (A100: -1.0 % alone, -3.0 % with md's row sync
  // gone, byte-identical); RNA_LOAD_MY_C_SYNC=1 restores it. Cross-stream order no longer leans on
  // it: the per-row legacy-stream work (gq_internal, the noLP cc reset, the DMLi
  // snapshot) moved onto the cell/md streams, and hp_mb waits on the scan event at
  // level 1 as well as 2 (2026-09-29).
  if(rnafold_load_my_c_sync())
    gpuErrchk( cudaDeviceSynchronize() );
#endif
  //for simplicity transfer all new_e, even though only need H * [start:length]
  // GPU-resident sweep: in device mode new_c_kernel has already written d_new_e
  // directly, so uploading the host's copy over it is exactly the round trip
  // being removed.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaMemcpy(d_new_e,new_e,g_row_total*sizeof(int),cudaMemcpyHostToDevice) );

  // RNA_C_RING: the row goes to the ring; the triangle gets it at the next flush.
  // The pending rows must be consecutive for one flush to cover them -- they are
  // (a row's size range only grows as i falls) -- but a gap is flushed, not assumed.
  if(g_c_ring_on) {
    if(g_c_pend > 0 && i != g_c_pend_i - 1) c_ring_flush(nfiles);
    bind_row_tables(i);
    c_ring_store_kernel<<<(unsigned int)((total + 255)/256),256,0,rnafold_stream_cell()>>>(
        nfiles, i, d_new_e, d_c_ring, g_row_total, d_row_off_H, d_size_off_H, total);
    gpuErrchk( cudaPeekAtLastError() );
    g_c_pend++; g_c_pend_i = i; g_c_rows++;
    if(g_c_pend >= c_flush_rows()) c_ring_flush(nfiles);
    rnafold_stream_cell_done();
    return;            // GPU-resident sweep only (rnafold_c_ring()), so no trailing sync
  }
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

  load_my_c_kernel<<<nblocks,block_size,0,rnafold_stream_cell()>>>(nfiles, i, /*turn,*/ length,
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

// ============ int_loop_warp_kernel: one WARP per cell, no shared, no barrier ====
//
// THE int_loop kernel since 2026-09-12. It replaced a block-per-cell kernel (the
// "twin" below; retired 2026-10-11 with RNA_INT_LOOP_WARP, see the pre-cleanup
// tag) and was byte-identical to it in every arm.
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


// The interior-loop cell, shared through a header: the
// kernel below is a wrapper around exactly this code.
#include "int_loop_cell.inc"

template <bool RING = false, bool LY = false>
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
                      int* __restrict__ energy_min,
                /* RNA_C_RING: read c from the 32-row ring instead of the triangle.
                 * A template flag, so the choice costs nothing in the inner loop. */
                const int* __restrict__ c_ring = NULL,
                const size_t c_ring_stride = 0,
                /* RNA_INT_LOOP_CELLS_PER_WARP: G consecutive cells per warp, 1..32.
                 * 1 = one warp per cell, exactly the code below the packed block. */
                const int G = 1,
                /* RNA_INT_LOOP_LYNGSO: G's planes for this row and row_total (int_loop_lyngso.inc) */
                const int* __restrict__ ly_G = NULL, const size_t ly_rt = 0) {
  /* ONE WARP PER BLOCK (blockDim.x == 32, INT_LOOP_WARP_BLOCK_SIZE): blockIdx.x
   * numbers the warp. */
  const int lane = (int)(threadIdx.x & 31u);

  /* DEAD CELLS DO NOT GET A WARP (G >= 2).
   *
   * RNA_INT_LOOP_WORK_STATS (since deleted) measured 62.6 % of the cells at 400x5601 with ZERO
   * candidates: hc says (i,j) cannot pair -- 1 - 6/16 on random ACGU -- so the warp
   * reads one hc bit, fails it, and still runs the 5-shuffle reduction to store INF.
   * Live cells are uniform (5-7 passes, 92 % lane efficiency), so the waste is whole
   * warps, not lanes.
   *
   * Here a warp owns G consecutive cells. Lane l < G probes cell l's hc bit -- one
   * coalesced read for all G -- and a dead cell's lane stores the INF itself, exactly
   * the value the reduction would have produced (every lane starts at INF and nothing
   * lowers it). The warp then runs the unchanged cell function on the LIVE cells only,
   * in ascending order. The set of cells written and every value written are the
   * same as at G = 1; only which warp writes them changes. `todo` comes from a
   * ballot, so the loop is warp-uniform and every shuffle inside still names all 32. */
  if(G > 1) {
    const size_t base = (size_t)blockIdx.x * (size_t)G;
    const size_t lim  = size_off_H[nfiles];
    if(base >= lim) return;                              // warp-uniform

    const size_t mine  = base + (size_t)lane;
    int          myH   = 0;
    size_t       myloc = mine;
    bool         live  = false;
    if(lane < G && mine < lim) {
      // The flat grid may straddle records within G cells: each lane finds its own.
      myH = flatten_index_to_H(mine, size_off_H, nfiles); myloc = mine - size_off_H[myH];
      const int i = i_H[myH];
      const int j = (int)myloc + i + turn + 1 + (int)size_off_H[nfiles + 1];
      live = Hc(Indx(i,j), &hccc[hc_off_H[myH]]);
      if(!live)
        energy_min[row_off_H[myH]+j] = INF;
    }
    unsigned int todo = __ballot_sync(0xffffffffu, live);
    while(todo) {
      const int src = __ffs(todo) - 1;
      todo &= todo - 1u;
      const int    H     = __shfl_sync(0xffffffffu, myH, src);
      const size_t local = (size_t)__shfl_sync(0xffffffffu, (unsigned long long)myloc, src);
      if(RING) {
        c_ring_reader rr;
        rr.base   = c_ring + row_off_H[H];
        rr.stride = c_ring_stride;
        int_loop_warp_cell_r<c_ring_reader, LY>(nfiles, i_row, length, TerminalAU, ninio2, P, lxc,
                             pair_, S, hccc, up_int, rr, row_off_H, hc_off_H,
                             size_off_H, i_H, energy_min, H, local, lane, ly_G, ly_rt);
      } else
      int_loop_warp_cell<LY>(nfiles, i_row, length, TerminalAU, ninio2, P, lxc, pair_, S, hccc, up_int, my_c, tri_off_H, row_off_H, hc_off_H, size_off_H, i_H, energy_min,
                         H, local, lane, ly_G, ly_rt);
    }
    return;
  }

  // Whole-warp exit: every lane of this warp shares the cell, so no lane is left
  // behind to be named by a shuffle mask below.
  const size_t cell = (size_t)blockIdx.x;
  if(cell >= size_off_H[nfiles]) return;
  const int    H     = flatten_index_to_H(cell, size_off_H, nfiles);
  const size_t local = cell - size_off_H[H];             // cell within record H

  // The arithmetic lives in int_loop_cell.inc; only the cell derivation above is kernel-specific.
  if(RING) {
    c_ring_reader rr;
    rr.base   = c_ring + row_off_H[H];
    rr.stride = c_ring_stride;
    int_loop_warp_cell_r<c_ring_reader, LY>(nfiles, i_row, length, TerminalAU, ninio2, P, lxc,
                         pair_, S, hccc, up_int, rr, row_off_H, hc_off_H,
                         size_off_H, i_H, energy_min, H, local, lane, ly_G, ly_rt);
  } else
  int_loop_warp_cell<LY>(nfiles, i_row, length, TerminalAU, ninio2, P, lxc, pair_, S, hccc, up_int, my_c, tri_off_H, row_off_H, hc_off_H, size_off_H, i_H, energy_min,
                     H, local, lane, ly_G, ly_rt);
}

/* RNA_INT_LOOP_CELLS_PER_WARP=G|auto (1..32; DEFAULT auto since 2026-10-02): each
 * warp owns G consecutive cells and spends its time on the live ones only -- see the
 * packed block at the top of int_loop_warp_kernel. =1 is the one-warp-per-cell kernel
 * unchanged (the opt-out). auto picks G PER LAUNCH: see rnafold_int_loop_cpw_for(). */
#define IL_CPW_AUTO 0
#define IL_CPW_AUTO_MAX 4      /* why 4 and not 8: see rnafold_int_loop_cpw_for() */
static int
rnafold_int_loop_cpw(void)
{
  static int v = -1;

  if (v < 0) {
    const char *e = getenv("RNA_INT_LOOP_CELLS_PER_WARP");

    if (!e || !e[0] || !strcmp(e, "auto")) {
      v = IL_CPW_AUTO;
      if (!vrna_cuda_quiet()) fprintf(stderr, "%-24s RNA_INT_LOOP_CELLS_PER_WARP=auto: G per row, the largest "
                      "<= %d that leaves every SM enough warps (=1 turns packing off)\n",
              __FILE__, IL_CPW_AUTO_MAX);
      return v;
    }
    v = atoi(e);
    if (v < 1)  v = 1;
    if (v > 32) v = 32;
    if (v > 1)
      fprintf(stderr, "%-24s RNA_INT_LOOP_CELLS_PER_WARP=%d: a warp owns %d cells and "
                      "runs only the ones hc lets pair\n", __FILE__, v, v);
  }

  return v;
}

/* AUTO: G FOR THIS ROW. Packing trades warps for less dead work, so it pays only
 * while the packed grid still fills the device. Measured on the laptop (RTX 3050,
 * 16 SMs, ncu, 20 x 4800, int_loop us per launch vs G=1):
 *
 *     cells/row    G=2     G=4     G=8
 *        4 050    +4 %   +15 %   +29 %
 *        8 050    -2 %    -3 %    +5 %
 *       16 050    -3 %    -5 %    -5 %
 *       32 050    -5 %    -9 %   -11 %
 *
 * Every G breaks even at the same place: ~2 000 PACKED warps, i.e. ~125 per SM. So
 * G is the largest power of two <= IL_CPW_AUTO_MAX with cells/G >= SMs * 128 --
 * G=1 on the narrow first rows of a sweep, the cap once a row is wide. The SM count
 * is the device's own, so the A100's 108 SMs move the threshold, not the rule.
 *
 * THE CAP IS 4, NOT 8: the best G also moves with LENGTH, which the width rule does
 * not see. A100 (CUDA_RNAFold_DeadWarps.ipynb B1, int_loop per row vs G=1):
 *
 *     L       G=4      G=8      G=16
 *     1200   -18.1 %  -21.8 %  -21.7 %
 *     2400   -15.2 %  -17.3 %  -14.3 %
 *     5601    -9.5 %   -7.1 %   +2.2 %      <- a mid row here is ~1.1 M cells: not starved
 *
 * Long-sequence cells cost more each, so G=8 leaves a warp too much serial work. The
 * production size is 5601, and the wall agrees: G4 -2.6 %, G8 -1.5 % at 400x5601. 4
 * gives up ~3-4 points of int_loop at short lengths; a length term could win them
 * back, but three points on one GPU is too little to fit one. The width rule is
 * moot on the A100 at production (every wide row clears 8 x SMs x 128) and still
 * protects narrow rows (laptop: G=8 on 550 cells was +159 %).
 *
 * THE LENGTH TERM WAS THEN TESTED AND IS RETIRED (A100 Queue run, 2026-10-03, section
 * C). Per row against auto, G8 is -4.6 % at 1200 nt and -2.9 % at 2400, +1.7 % at 5601;
 * but in the WALL no fixed G beat auto beyond the spread at any of the three lengths
 * (G8 at 1200: -1.7 % inside a 2.5 % spread). The cap of 4 stands. */
/* IL_CPW_AUTO_MAX is defined beside IL_CPW_AUTO above: 4. */
#define IL_CPW_WARPS_PER_SM    128
static int
rnafold_int_loop_cpw_for(const size_t cells)
{
  const int fixed = rnafold_int_loop_cpw();
  if (fixed != IL_CPW_AUTO) return fixed;

  static size_t floor_warps = 0;
  if (!floor_warps) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    floor_warps = (size_t)(sms > 0 ? sms : 1) * IL_CPW_WARPS_PER_SM;
  }
  int G = 1;
  while (G < IL_CPW_AUTO_MAX && cells / (size_t)(2 * G) >= floor_warps) G *= 2;
  return G;
}

// ONE WARP PER BLOCK. A block of this kernel is 32 x the number of INDEPENDENT
// cells it holds, and the sweep (A100, 400 x 5601, int_loop) put the best at one:
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
#define INT_LOOP_WARP_BLOCK_SIZE 32   /* the kernel assumes it: blockIdx.x is the warp */

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
                   const int s_stride,                  // d_S words per record
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
  const int j = (int)(k - size_off_H[H]) + i + turn_ + 1 + (int)size_off_H[nfiles + 1];

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
  const int si = unpack(S,H,s_stride,i+1);
  const int sj = unpack(S,H,s_stride,j-1);
  unsigned char type = Ptype(S,pair_,H,s_stride,i,j);
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
    if((unpack(S,H,s_stride,p) == 3) && (p + VRNA_GQUAD_MIN_BOX_SIZE < j)) {
      int minq = p + VRNA_GQUAD_MIN_BOX_SIZE - 1;
      if(minq + 1 + MAXLOOP < j) minq = j - MAXLOOP - 1;
      int maxq = p + VRNA_GQUAD_MAX_BOX_SIZE + 1;
      if(maxq + 3 > j) maxq = j - 3;
      for(int q = minq; q < maxq; q++) {
        if(unpack(S,H,s_stride,q) != 3) continue;
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
    if(unpack(S,H,s_stride,p) != 3) continue;
    int minq = p + VRNA_GQUAD_MIN_BOX_SIZE - 1;
    if(minq + 1 + MAXLOOP - l1 < j) minq = j - MAXLOOP + l1 - 1;
    int maxq = p + VRNA_GQUAD_MAX_BOX_SIZE + 1;
    if(maxq >= j) maxq = j - 1;
    for(int q = minq; q < maxq; q++) {
      if(unpack(S,H,s_stride,q) != 3) continue;
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
    if(unpack(S,H,s_stride,q) == 3) {
      const int p0 = (i + 4 + VRNA_GQUAD_MAX_BOX_SIZE - 1 < q)
                     ? q - VRNA_GQUAD_MAX_BOX_SIZE + 1 : i + 4;
      for(int p = p0; p + VRNA_GQUAD_MIN_BOX_SIZE - 1 < j; p++) {
        const int l1 = p - i - 1;
        if(l1 > MAXLOOP) break;
        if(unpack(S,H,s_stride,p) != 3) continue;
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
  /* ON THE CELL STREAM (2026-09-29). This MIN2s into d_energy_min2, which
   * int_loop_kernel(i) writes and new_c(i) reads, both on the cell stream. It
   * used to go to the legacy default stream, and at RNA_STREAM_OVERLAP >= 1 --
   * the default -- the cell stream is NON-BLOCKING, so nothing ordered this
   * read-modify-write against int_loop's plain stores to the same cells. */
  gq_internal_kernel<<<(unsigned int)grid,block,0,rnafold_stream_cell()>>>(
      nfiles, turn_, g_s_stride, d_param, d_S, d_pair,
      d_row_off_H, d_size_off_H, d_i_H, d_energy_min2,
      gv, gc, gr, ge_, gro);
  gpuErrchk( cudaPeekAtLastError() );
}


/* Lyngsø interior loops, S1: the selftest beside the dense kernel (PORT_LYNGSO_INTLOOP.md). */
#include "int_loop_lyngso.inc"

//Host (ie non-GPU) code
PRIVATE void
int_loop_cuda(const int nfiles,
	      const int i, /*const int turn,*/ const int length,
	      const vrna_param_t *P,
	      int* energy_min,
	      const size_t* size_off_H, //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
	      const int* i_H) {         //in, nfiles entries -- per-record row index
  //cf modular_decomposition.cu r1.79
  // Staggered_Row_Batching Phase 5: size_off_H[nfiles] replaces the old
  // scalar nblocks<=0 check -- numerically identical while every H shares
  // one length (today). Flat 1-D grid instead of dim3(nblocks,nfiles) --
  // also lifts the old implicit nfiles<=65535 sub-limit from using nfiles
  // as gridDim.y (gridDim.x supports far more).
  const size_t flat_nblocks = size_off_H[nfiles];
  if(flat_nblocks==0) return;

  bind_row_tables(i);
  ly_eval_row(nfiles, i, length);   /* RNA_INT_LOOP_LYNGSO: this row's G, before the LY cells read it */

  // RNA_LAUNCH_STATS: bracket the kernel ALONE -- not the two offset uploads
  // above, which belong to the phase but not to the kernel. Splitting those two
  // apart is the entire point (STRESS272_RESULTS.md 20.4).
  rnafold_launch_stats_begin();

  rnafold_retired_knobs();
  {
    // One warp per block, G cells per warp (RNA_INT_LOOP_CELLS_PER_WARP): the grid
    // counts WARPS, each owning G consecutive cells, so G = 1 is one per cell.
    const int    il_G = rnafold_int_loop_cpw_for(flat_nblocks);
    const size_t nb   = (flat_nblocks + (size_t)il_G - 1)/(size_t)il_G;
    assert(nb <= 2147483647u);
    const dim3   grid((unsigned int)nb);
#define IL_WARP_LAUNCH(RG, LYV) int_loop_warp_kernel<RG,LYV><<<grid, INT_LOOP_WARP_BLOCK_SIZE, 0, rnafold_stream_cell()>>>( \
        nfiles, i, length, P->TerminalAU, P->ninio[2], d_param, P->lxc, \
        d_pair, d_S, d_hccc, d_up_int, d_my_c, d_tri_off_H, d_row_off_H, d_hc_off_H, \
        d_size_off_H, d_i_H, d_energy_min2, d_c_ring, g_row_total, il_G, \
        (LYV) ? (const int*)d_ly_G[i & 1] : (const int*)NULL, (LYV) ? g_row_total : (size_t)0)
    /* RNA_C_RING picks the reader at the launch. RNA_INT_LOOP_LYNGSO picks LY: the direct
     * loops here, the generic ones from the carried table (int_loop_lyngso.inc). These
     * four are the only instances (PORT_CLEANUP_PLAN.md D2). */
    if(g_ly_eval) { if(g_c_ring_on) IL_WARP_LAUNCH(true, true);
                    else            IL_WARP_LAUNCH(false, true); }
    else if(g_c_ring_on) IL_WARP_LAUNCH(true, false);
    else                 IL_WARP_LAUNCH(false, false);
#undef IL_WARP_LAUNCH
  }

  rnafold_launch_stats_end((unsigned int)flat_nblocks);

  gpuErrchk( cudaPeekAtLastError() );
  ly_selftest_row(nfiles, i, length);   /* RNA_INT_LOOP_LYNGSO_SELFTEST; no-op otherwise */
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );

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
	   const int* i_H ) {        //in, nfiles entries -- per-record row index
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


/* ---- RNA_ROW_FUSE: the my_c triangle, for the fused row kernel -------------
 * hp_mb_3p + new_c + load_my_c are three launches per row that are element-wise
 * in j and chain within ONE cell: new_c(j) reads only hp_mb(j)'s outputs, and
 * load_my_c(j) reads only new_c(j)'s. So one thread can run all three with no
 * barrier -- see row_cells_kernel in hp_mb_loop.cu, which is where the fused
 * kernel lives because two of the three cells are defined there.
 *
 * It needs the two buffers that live on THIS side: the my_c triangle it writes
 * and that triangle's per-record base. Same shape and the same validity window
 * as int_loop_row_buffers() above. */
extern "C" /*PUBLIC*/ void
int_loop_my_c_buffers(int** my_c_out, const size_t** tri_off_H_out) {
  if(my_c_out)       *my_c_out       = d_my_c;
  if(tri_off_H_out)  *tri_off_H_out  = d_tri_off_H;
}

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
// Two more bit-packed per-(H,ij) predicates, added 2026-08-22 so new_c_host
// stops reading the ptype and hc->matrix triangles itself. Both were costing
// it a stride-~j cache miss per (H,j) -- 3.2 s of workload A between them,
// measured by stubbing them out -- and unlike the my_c/fML mirrors there was
// no row buffer already holding them, because they index *static* per-(i,j)
// data. Packing them the same way as d_hccc_mb and letting this kernel emit a
// row-shaped gate moves the random access to the GPU, where it is free.
unsigned int* d_hccc_any;   //bit-packed (hc->matrix[ij] != 0) per (H,ij)
unsigned int* d_hccc_gu;    //bit-packed (ptype[ij]==3 || ptype[ij]==4) per (H,ij)
char*         d_gate_row;   //out, row-shaped: bit0 from d_hccc_any, bit1 from d_hccc_gu
unsigned int* d_hccc_mb;    //bit-packed VRNA_CONSTRAINT_CONTEXT_MB_LOOP per (H,ij)
unsigned int* d_hccc_mbenc; //bit-packed VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC per (H,ij)
//Kernels can only write device memory -- these are the device-side row
//outputs the kernel actually writes; hp_mb_3p_i() cudaMemcpy's them back to
//the caller's host row buffers afterward, same pattern as int_loop_cuda()'s
//d_energy_min2 in int_loop.cu.
int* d_energy_hp_row;
int* d_energy_mb_row;
int* d_energy_3p00_row;
// Staggered_Row_Batching Phase 2c: device copy of row_off_H[] (own copy,
// per this file's established convention of not sharing device state with
// int_loop.cu/modular_decomposition.cu) -- replaces H*(length+1) wherever
// the 3 row buffers above are indexed.
static size_t* d_row_off_H;
// Staggered_Row_Batching Phase 6d: row_off_H[nfiles], the true total extent of
// this file's 3 row buffers. Cached at init_gpu3() time because hp_mb_3p_i()
// copies them back whole but never receives the offset table. Equals the old
// nfiles*(length+1) exactly while chunks stay uniform-length; once they don't,
// that formula runs past the real allocation.
static size_t  g_row_total = 0;
// Staggered_Row_Batching Phase 2e: d_hccc_mb/d_hccc_mbenc's per-H block
// start (own shape, Hc_ints2()-scaled -- no MAXLOOP padding, unlike
// int_loop.cu's Hc_ints(), so this can't reuse that file's d_hc_off_H) and
// d_S2/d_sequence's per-H block start (length+2 stride, distinct from
// row_off_H's length+1). Both computed locally in init_gpu3(), same
// reasoning as int_loop.cu's d_hc_off_H.
static size_t* d_hc2_off_H;
static size_t* d_seq_off_H;
// GPU-resident sweep, step 1: hc->up_ml[j] > 0, one byte per sequence
// position, indexed by seq_off_H[H]+j exactly like d_S2/d_sequence.
//
// fml_host needs this predicate twice per (H,row): once at j (the 3' unpaired
// extension) and once at i (the 5' one). It is the only input to that loop
// that is not already device-resident.
//
// A byte array, not a packed bitmask: 824 KB per chunk against a 19.8 GB
// budget, so the 8x saving buys nothing and costs bit arithmetic that could be
// wrong. Packed from the host's OWN hc->up_ml rather than re-derived from the
// sequence -- the standing rule on this codebase, and the reason the Step 1
// bitmask work needed a formal equivalence argument that this does not.
// Cheap enough to pack unconditionally, so unlike the hc masks it needs no
// g_hc_seq_derived split and works under constraints as-is.
// READ BY NOTHING YET.
static char*   d_up_ml_ok;
// Staggered_Row_Batching Phase 5: per-row block-count table for
// hp_mb_3p_kernel -- own copy (per this file's established convention),
// same "size" formula as int_loop.cu's/modular_decomposition.cu's
// (length_H[H]-i-turn, reused verbatim from Phase 4's load_fML). Allocated
// once per chunk, uploaded fresh each row by hp_mb_3p_i().
static size_t* d_size_off_H;
// Host shadow of what d_size_off_H currently holds, so a row's redundant
// re-uploads can be skipped. See upload_size_off_H() below.
static size_t* size_off_shadow   = NULL;
static int     size_off_shadow_n = 0;
static void    size_off_shadow_reset(void);  // defined below; called from init/teardown above it
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

// Builds all five packed hard-constraint bitmasks straight from the sequence,
// replacing two O(n^2)-per-record host loops that between them were 197.4 s of
// a 769 s Colab run -- 25.7% of wall -- to produce 9.81 MB per record of data
// that is fully determined by an 11 KB sequence.
//
// One thread owns one 32-cell output word and assembles it in a register, so
// there is no atomic and no read-modify-write to memory. The four masks packed
// here share hc2_off_H/Hc_ints2 extents; int_loop.cu's d_hccc has its own
// (Hc_ints pads by MAXLOOP), hence the separate offset table.
__global__ void
pack_hc_kernel(const int nfiles, const int turn, const int max_bp_span,
               const int noGU, const int noGUclosure,
               const short* __restrict__ S, const char* __restrict__ pair,
               const size_t* __restrict__ hc2_off_H,
               const size_t* __restrict__ hc_off_H,
               const size_t* __restrict__ seq_off_H,
                     unsigned int* __restrict__ mb,
                     unsigned int* __restrict__ mbenc,
                     unsigned int* __restrict__ any,
                     unsigned int* __restrict__ gu,
                     unsigned int* __restrict__ intenc,
               const size_t total_words2) {
  const size_t w = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(w >= total_words2) return;

  const int H = flatten_index_to_H(w, hc2_off_H, nfiles);
  const size_t wH = w - hc2_off_H[H];              //word index within this record
  const short* S_H = &S[seq_off_H[H]];
  //seq_off_H's stride is this record's length+2 (init_gpu3 builds it that way),
  //so no separate length table is needed.
  const int n = (int)(seq_off_H[H+1] - seq_off_H[H]) - 2;

  unsigned int m_mb=0u, m_mbenc=0u, m_any=0u, m_gu=0u, m_int=0u;
  const long long base = (long long)wH * 32;
  #pragma unroll
  for(int b=0; b<32; b++) {
    const long long f = base + b;
    const unsigned char opt = rnafold_hc_opt(f, n, turn, max_bp_span,
                                             noGU, noGUclosure, S_H, pair);
    const int pt = rnafold_ptype(f, n, turn, S_H, pair);
    m_mb    |= ((opt & VRNA_CONSTRAINT_CONTEXT_MB_LOOP)      ? 1u : 0u) << b;
    m_mbenc |= ((opt & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC)  ? 1u : 0u) << b;
    m_any   |= (opt                                          ? 1u : 0u) << b;
    m_gu    |= ((pt == 3 || pt == 4)                         ? 1u : 0u) << b;
    m_int   |= ((opt & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC) ? 1u : 0u) << b;
  }
  mb[w] = m_mb;  mbenc[w] = m_mbenc;  any[w] = m_any;  gu[w] = m_gu;

  // int_loop.cu's mask is the same bits over the same flat cells, but its
  // per-record blocks are spaced by Hc_ints() (MAXLOOP padding) rather than
  // Hc_ints2(), so it is addressed through its own table. Words past this
  // record's Hc_ints2 extent are padding and were memset to zero.
  intenc[hc_off_H[H] + wH] = m_int;
}

PUBLIC void
init_gpu3(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size,
          const size_t* row_off_H) { //in, nfiles+1 entries -- see compute_batch_offsets(), mfe_cuda.c
  if(!first3) return;
  const double _t_ig3 = rnafold_now_seconds();
  fprintf(stderr,"%-24s init_gpu3(%d,VC,%d,%d,%d)\n",__FILE__,nfiles,turn_,length,block_size);

  TIMED_CUDAMALLOC(&d_row_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_row_off_H, row_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  // d_param2/d_pair2 are nfiles/length-independent -- guarded on their own
  // one-time check (d_param2 starts NULL, a zero-initialized global) rather
  // than on first3, so teardown_gpu3() can reset first3=1 between GPU
  // batches without this block re-allocating (and leaking) them every batch.
  if(!d_param2) {
    TIMED_CUDAMALLOC(&d_param2, sizeof(cuda_param2_t));
    load_param2(VC[0]->params);

    char pair_[NBPAIRS+1][NBPAIRS+1];
    for(int x=0;x<21;x++){
    for(int y=0;y<21;y++) {
      const vrna_md_t *md = &(VC[0]->params->model_details);
      if(x < NBPAIRS+1 && y < NBPAIRS+1) pair_[x][y] = md->pair[x][y];
    }}
    const size_t pair_size = (NBPAIRS+1)*(NBPAIRS+1)*sizeof(char);
    TIMED_CUDAMALLOC(&d_pair2, pair_size);
    gpuErrchk( cudaMemcpy(d_pair2,pair_,pair_size,cudaMemcpyHostToDevice) );
  }

  //VRNA_CONSTRAINT_CONTEXT_MB_LOOP / _ENC bitmasks, same packing technique as
  //int_loop.cu's d_hccc (which only packs the INT_LOOP_ENC bit) -- see
  //init_gpu2() there. hc->matrix packs several independent context bits into
  //the same byte (constraints_hard.h), so this is just extracting two more.
  // Staggered_Row_Batching Phase 2e: table-driven per-H block start,
  // replacing the uniform Hc_ints2(length) multiply.
  size_t hc2_off_H[nfiles+1];
  hc2_off_H[0] = 0;
  for(int H=0;H<nfiles;H++) hc2_off_H[H+1] = hc2_off_H[H] + Hc_ints2(VC[H]->length);
  TIMED_CUDAMALLOC(&d_hc2_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_hc2_off_H, hc2_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  size_t size = hc2_off_H[nfiles]*sizeof(unsigned int);
  TIMED_CUDAMALLOC(&d_hccc_any, size);
  TIMED_CUDAMALLOC(&d_hccc_gu, size);
  TIMED_CUDAMALLOC(&d_hccc_mb, size);
  TIMED_CUDAMALLOC(&d_hccc_mbenc, size);
  // Sequence-derived case: pack_hc_kernel below fills all four of these (and
  // int_loop.cu's d_hccc) from the sequence once d_S2/d_pair2 exist, so the
  // O(n^2) host loop is skipped entirely. It measured 126.9 s of a 769 s Colab
  // run on its own; its int_loop twin another 71.7 s.
  if(!g_hc_seq_derived) {
  unsigned int* hccc_mb    = (unsigned int*) calloc(hc2_off_H[nfiles],sizeof(unsigned int));
  unsigned int* hccc_mbenc = (unsigned int*) calloc(hc2_off_H[nfiles],sizeof(unsigned int));
  unsigned int* hccc_any   = (unsigned int*) calloc(hc2_off_H[nfiles],sizeof(unsigned int));
  unsigned int* hccc_gu    = (unsigned int*) calloc(hc2_off_H[nfiles],sizeof(unsigned int));
  const double _t_pk3 = rnafold_now_seconds();
  for(int H=0;H<nfiles;H++){
    unsigned int mask_mb, mask_mbenc, mask2;
    // Staggered_Row_Batching Phase 6a: bounded by this H's own length, not
    // the shared `length` scalar -- same reasoning as int_loop.cu's
    // hc_off_H population loop (VC[H]->hc->matrix is only ever sized to
    // VC[H]->length; hccc_mb/hccc_mbenc are calloc'd, so the untouched
    // tail stays correctly zero).
    const int length_H = (int)VC[H]->length;
    for(int i=0;i<(length_H*(length_H+1))/2+2;i++){
      mask_mb    = ((i & 0x1f) == 0)? 1 : mask_mb    << 1;
      mask_mbenc = ((i & 0x1f) == 0)? 1 : mask_mbenc << 1;
      mask2      = ((i & 0x1f) == 0)? 1 : mask2      << 1;
      const size_t I = hc2_off_H[H]+i/bitsperint;
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP)     hccc_mb[I]    |= mask_mb;
      if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC) hccc_mbenc[I] |= mask_mbenc;
      // The two new predicates, packed from the host's OWN ptype/hc arrays
      // rather than recomputed device-side from the sequence: new_c_host used
      // exactly these values, so re-deriving them (e.g. via Ptype2()) would be
      // an extra equivalence to prove. This loop already walks hc->matrix
      // contiguously and ptype has the identical extent -- alphabet.c
      // allocates it (n*(n+1))/2+2, the same bound as this loop.
      if(VC[H]->hc->matrix[i])                                       hccc_any[I]   |= mask2;
      { const char pt = VC[H]->ptype[i];
        if(pt == 3 || pt == 4)                                       hccc_gu[I]    |= mask2; }
    }
  }
  stage_ig_pack_s += rnafold_now_seconds() - _t_pk3;
  gpuErrchk( cudaMemcpy(d_hccc_mb,   hccc_mb,   hc2_off_H[nfiles]*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_hccc_mbenc,hccc_mbenc,hc2_off_H[nfiles]*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_hccc_any,  hccc_any,  hc2_off_H[nfiles]*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_hccc_gu,   hccc_gu,   hc2_off_H[nfiles]*sizeof(unsigned int),cudaMemcpyHostToDevice) );
  free(hccc_mb);
  free(hccc_mbenc);
  free(hccc_any);
  free(hccc_gu);
  }

  // Staggered_Row_Batching Phase 2e: table-driven per-H block start (own
  // shape, length+2 stride -- distinct from row_off_H's length+1).
  size_t seq_off_H[nfiles+1];
  seq_off_H[0] = 0;
  for(int H=0;H<nfiles;H++) seq_off_H[H+1] = seq_off_H[H] + ((size_t)VC[H]->length+2);
  TIMED_CUDAMALLOC(&d_seq_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  gpuErrchk( cudaMemcpy(d_seq_off_H, seq_off_H, (size_t)(nfiles+1)*sizeof(size_t), cudaMemcpyHostToDevice) );

  // Staggered_Row_Batching Phase 5: allocated here, not populated here --
  // changes every sweep row i, uploaded fresh per-row by hp_mb_3p_i().
  TIMED_CUDAMALLOC(&d_size_off_H, (size_t)(nfiles+1)*sizeof(size_t));
  // This buffer is brand new and holds nothing. Drop the shadow so
  // upload_size_off_H() cannot mistake it for already-current -- see the
  // HAZARD note on that function.
  size_off_shadow_reset();

  size = seq_off_H[nfiles]*sizeof(short);
  TIMED_CUDAMALLOC(&d_S2, size);
  short* Sbuff = (short*) malloc(size);
  // Staggered_Row_Batching Phase 6a: copy each H's own (length+2) elements,
  // not the shared `length`'s -- seq_off_H[H+1]-seq_off_H[H] already equals
  // VC[H]->length+2 exactly (Phase 2e sized it per-H), so copying the
  // shared length here was a double bug once length can exceed a given H's
  // own: over-reading VC[H]->sequence_encoding *and* over-writing into the
  // next H's region of Sbuff.
  for(int H=0;H<nfiles;H++) memcpy(&Sbuff[seq_off_H[H]],VC[H]->sequence_encoding,((size_t)VC[H]->length+2)*sizeof(short));
  gpuErrchk( cudaMemcpy(d_S2,Sbuff,size,cudaMemcpyHostToDevice) );
  free(Sbuff);

  size = seq_off_H[nfiles]*sizeof(char);
  TIMED_CUDAMALLOC(&d_sequence, size);
  char* seqbuff = (char*) calloc(seq_off_H[nfiles],sizeof(char));
  for(int H=0;H<nfiles;H++) {
    const size_t len = strlen(VC[H]->sequence);
    memcpy(&seqbuff[seq_off_H[H]],VC[H]->sequence,len);
  }
  gpuErrchk( cudaMemcpy(d_sequence,seqbuff,size,cudaMemcpyHostToDevice) );
  free(seqbuff);

  // GPU-resident sweep, step 1 -- see the declaration. Same seq_off_H indexing
  // and (length+2) stride as d_S2/d_sequence above, so it is built in the same
  // pass and needs no offset table of its own. calloc'd, so a record's unused
  // tail reads as "not allowed", which is the safe direction: fml_host's
  // `up_ml > 0` guard yields INF there, exactly as a missing extension should.
  size = seq_off_H[nfiles]*sizeof(char);
  TIMED_CUDAMALLOC(&d_up_ml_ok, size);
  {
    char* upbuff = (char*) calloc(seq_off_H[nfiles],sizeof(char));
    for(int H=0;H<nfiles;H++) {
      // hc->up_ml is allocated (n+2) ints (constraints_hard.c:182), matching
      // this stride exactly; index 0 is unused by the sweep but copied anyway
      // so the two arrays stay index-for-index comparable.
      const int len_H = (int)VC[H]->length;
      for(int j=0;j<=len_H+1;j++)
        upbuff[seq_off_H[H]+j] = (VC[H]->hc->up_ml[j] > 0) ? 1 : 0;
    }
    gpuErrchk( cudaMemcpy(d_up_ml_ok,upbuff,size,cudaMemcpyHostToDevice) );
    free(upbuff);
  }

  // Staggered_Row_Batching Phase 6d: real total extent, not the uniform
  // nfiles*(length+1); also cached for hp_mb_3p_i()'s copy-back.
  g_row_total = row_off_H[nfiles];
  size = g_row_total*sizeof(int);
  TIMED_CUDAMALLOC(&d_energy_hp_row, size);
  TIMED_CUDAMALLOC(&d_energy_mb_row, size);
  TIMED_CUDAMALLOC(&d_energy_3p00_row, size);
  //char, not int: it carries two bits per cell and is copied back every row.
  TIMED_CUDAMALLOC(&d_gate_row, g_row_total*sizeof(char));

  // Everything pack_hc_kernel reads (d_S2, d_pair2, and the three offset
  // tables) exists by this point, which is why the launch sits at the end of
  // init_gpu3 rather than beside the allocations it fills.
  if(g_hc_seq_derived) {
    const double _t_pk = rnafold_now_seconds();
    unsigned int* d_intenc = NULL;
    const size_t* d_hcoff  = NULL;
    int_loop_hccc_buffers(&d_intenc, &d_hcoff);   //init_gpu2 already ran
    assert(d_intenc && d_hcoff);
    const vrna_md_t* md_ = &(VC[0]->params->model_details);
    const size_t total_words2 = hc2_off_H[nfiles];
    const int    bs  = 256;
    const size_t nbl = (total_words2 + bs - 1)/bs;
    assert(nbl <= 2147483647u);
    pack_hc_kernel<<<(int)nbl,bs>>>(nfiles, turn_, md_->max_bp_span,
                                    md_->noGU, md_->noGUclosure,
                                    d_S2, d_pair2,
                                    d_hc2_off_H, d_hcoff, d_seq_off_H,
                                    d_hccc_mb, d_hccc_mbenc, d_hccc_any,
                                    d_hccc_gu, d_intenc, total_words2);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    stage_ig_pack_s += rnafold_now_seconds() - _t_pk;  //same timer, for comparability

    // RNA_HC_VERIFY=1: build the masks the old way too and compare bit for
    // bit. Byte-identical fold output only proves the bits are right where
    // they are READ (j-i >= turn+1); this proves every word matches, including
    // cells nothing looks at today, so widening a read range later cannot
    // quietly expose a wrong bit. Slow by design -- diagnostics only.
    if(getenv("RNA_HC_VERIFY")) {
      const size_t nw2 = hc2_off_H[nfiles];
      unsigned int* h_mb    = (unsigned int*) calloc(nw2,sizeof(unsigned int));
      unsigned int* h_mbenc = (unsigned int*) calloc(nw2,sizeof(unsigned int));
      unsigned int* h_any   = (unsigned int*) calloc(nw2,sizeof(unsigned int));
      unsigned int* h_gu    = (unsigned int*) calloc(nw2,sizeof(unsigned int));
      for(int H=0;H<nfiles;H++){
        unsigned int mask;
        const int length_H = (int)VC[H]->length;
        for(int i=0;i<(length_H*(length_H+1))/2+2;i++){
          mask = ((i & 0x1f) == 0)? 1 : mask << 1;
          const size_t I = hc2_off_H[H]+i/bitsperint;
          if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP)     h_mb[I]    |= mask;
          if(VC[H]->hc->matrix[i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC) h_mbenc[I] |= mask;
          if(VC[H]->hc->matrix[i])                                       h_any[I]   |= mask;
          { const char pt = VC[H]->ptype[i];
            if(pt == 3 || pt == 4)                                       h_gu[I]    |= mask; }
        }
      }
      unsigned int* g = (unsigned int*) malloc(nw2*sizeof(unsigned int));
      const char*   nm[4] = {"hccc_mb","hccc_mbenc","hccc_any","hccc_gu"};
      unsigned int* dv[4] = {d_hccc_mb,d_hccc_mbenc,d_hccc_any,d_hccc_gu};
      unsigned int* hv[4] = {h_mb,h_mbenc,h_any,h_gu};
      int bad = 0;
      for(int k=0;k<4;k++){
        gpuErrchk( cudaMemcpy(g,dv[k],nw2*sizeof(unsigned int),cudaMemcpyDeviceToHost) );
        for(size_t w=0;w<nw2;w++) if(g[w]!=hv[k][w]) {
          if(bad++ < 8)
            fprintf(stderr,"RNA_HC_VERIFY MISMATCH %s word %zu: gpu=%08x host=%08x\n",
                    nm[k],w,g[w],hv[k][w]);
        }
      }
      fprintf(stderr,"%-24s RNA_HC_VERIFY: %zu words x 4 masks, %d mismatching words\n",
              __FILE__, nw2, bad);
      free(g); free(h_mb); free(h_mbenc); free(h_any); free(h_gu);
    }
  }

  stage_ig3_s += rnafold_now_seconds() - _t_ig3;
  first3 = 0;
}

// Frees the 7 nfiles/length-scaled device buffers allocated by init_gpu3()
// and resets first3 so the next init_gpu3() call re-runs at a new batch's
// nfiles. d_param2/d_pair2 are deliberately left allocated -- they're
// nfiles/length-independent (see the one-time guard in init_gpu3() above)
// and never need resizing between batches.
PUBLIC void
teardown_gpu3(void) {
  if(first3) return; // never initialized (or already torn down) -- nothing to free
  gpuErrchk( cudaFree(d_hccc_any) );
  gpuErrchk( cudaFree(d_hccc_gu) );
  gpuErrchk( cudaFree(d_gate_row) );
  gpuErrchk( cudaFree(d_hccc_mb) );
  gpuErrchk( cudaFree(d_hccc_mbenc) );
  gpuErrchk( cudaFree(d_S2) );
  gpuErrchk( cudaFree(d_sequence) );
  gpuErrchk( cudaFree(d_up_ml_ok) );
  gpuErrchk( cudaFree(d_energy_hp_row) );
  gpuErrchk( cudaFree(d_energy_mb_row) );
  gpuErrchk( cudaFree(d_energy_3p00_row) );
  gpuErrchk( cudaFree(d_row_off_H) );
  gpuErrchk( cudaFree(d_hc2_off_H) );
  gpuErrchk( cudaFree(d_seq_off_H) );
  gpuErrchk( cudaFree(d_size_off_H) );
  size_off_shadow_reset();   // the device buffer is gone; the shadow must not outlive it
  first3 = 1;
}

// Bytes of device memory this file needs for one additional sequence at the
// given length -- all row-scale (O(length), not O(length^2)) by the
// GPU-energy-precompute design this file already uses, so this is small
// relative to modular_decomposition.cu's/int_loop.cu's per-file costs.
// d_param2/d_pair2 excluded -- fixed-size, paid once regardless of batch
// count.
PUBLIC size_t
hp_mb_loop_bytes_per_file(const int length) {
  const size_t hccc_mb_bytes    = Hc_ints2(length)*sizeof(unsigned int);
  const size_t hccc_mbenc_bytes = Hc_ints2(length)*sizeof(unsigned int);
  const size_t hccc_any_bytes   = Hc_ints2(length)*sizeof(unsigned int);
  const size_t hccc_gu_bytes    = Hc_ints2(length)*sizeof(unsigned int);
  const size_t gate_row_bytes   = (size_t)(length+1)*sizeof(char);
  const size_t s2_bytes         = (size_t)(length+2)*sizeof(short);
  const size_t sequence_bytes   = (size_t)(length+2)*sizeof(char);
  const size_t hp_row_bytes     = (size_t)(length+1)*sizeof(int);
  const size_t mb_row_bytes     = (size_t)(length+1)*sizeof(int);
  const size_t p3p00_row_bytes  = (size_t)(length+1)*sizeof(int);
  // GPU-resident sweep step 1. Same (length+2) stride as sequence_bytes -- it
  // shares seq_off_H. Counted here for the same reason as everything else in
  // this function: RNAfold.c admits records into a chunk against this number,
  // so a buffer allocated but not counted is a chunk that does not fit.
  const size_t up_ml_bytes      = (size_t)(length+2)*sizeof(char);
  return hccc_mb_bytes + hccc_mbenc_bytes + hccc_any_bytes + hccc_gu_bytes
       + s2_bytes + sequence_bytes + up_ml_bytes
       + hp_row_bytes + mb_row_bytes + p3p00_row_bytes + gate_row_bytes;
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
//caller passes (vc->sequence+i-1) -- device equivalent is &d_sequence[seq_off_H[H]+i-1].
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
hp_mb_3p_kernel(const int nfiles, const int i, const int turn, const int length,
                 const short* __restrict__ S,
                 const char*  __restrict__ seq,
                 const char*  __restrict__ pair,
                 const unsigned int* __restrict__ hccc_mb,
                 const unsigned int* __restrict__ hccc_mbenc,
                 const unsigned int* __restrict__ hccc_any,
                 const unsigned int* __restrict__ hccc_gu,
                 const cuda_param2_t* __restrict__ P,
                       int* __restrict__ energy_hp_row,
                       int* __restrict__ energy_mb_row,
                       int* __restrict__ energy_3p00_row,
                      char* __restrict__ gate_row,
                 const size_t* __restrict__ row_off_H,
                 const size_t* __restrict__ hc2_off_H,
                 const size_t* __restrict__ seq_off_H,
                 const size_t* __restrict__ size_off_H, const size_t total) {
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int j = mj + i+turn+1;

  const short* S_H   = &S[seq_off_H[H]];
  const char*  seq_H = &seq[seq_off_H[H]];
  const int ij = Indx2(i,j);
  const unsigned int* Hccc_mb    = &hccc_mb[hc2_off_H[H]];
  const unsigned int* Hccc_mbenc = &hccc_mbenc[hc2_off_H[H]];

  // The gate new_c_host used to build from Ptype(H,ij)/Hard_constraints(H,ij).
  // Same two values, same ij (Indx2 here == jindx[j]+i on the host), just read
  // out of bitmasks by a GPU thread instead of out of two triangles by a host
  // loop walking them at stride ~j.
  gate_row[row_off_H[H]+j] = (char)(Hc2(ij,&hccc_any[hc2_off_H[H]])
                                 | (Hc2(ij,&hccc_gu [hc2_off_H[H]]) << 1));

  //raw_type: the 0->7 fixup must NOT be applied before the rtype[] lookup in
  //energy_mb below -- mb_loop_fast.c:74,105 uses the raw (possibly-0) type as
  //the rtype[] index and only fixes up the *result*. energy_hp/energy_3p_00
  //DO use the fixed-up type directly (hairpin_loops.c:252-255,
  //fill_arrays.c's old type_ local) -- see `type` below.
  const int raw_type = (int)Ptype2(S_H,pair,i,j);
  int type = raw_type;
  if(type == 0) type = 7;

  //energy_hp: vrna_E_hp_loop()/vrna_eval_hp_loop() do no hc check of their
  //own (hairpin_loops.c:226-293) -- fill_arrays_loop.c's read site already
  //gates on hc_decompose/no_close identically to how fill_arrays.c used to
  //gate the write, so compute unconditionally here (see file header comment).
  {
    const int u = j-i-1;
    energy_hp_row[row_off_H[H]+j] =
      E_Hairpin_device(u, type, S_H[i+1], S_H[j-1], &seq_H[i-1], P);
  }

  //energy_mb: mb_loop_fast.c:92-148 (dangle_model==2, cp==-1, sc==NULL path)
  //-- decomp starts at 0 (NOT INF), only overwritten if the MB_LOOP bit is set.
  {
    int decomp = 0;
    if(Hc2(ij,Hccc_mb)){
      int tt = P->rtype[raw_type]; //raw_type, NOT type -- see comment above
      if(tt == 0) tt = 7;
      decomp = E_MLstem_device(tt, S_H[j-1], S_H[i+1], P) + P->MLclosing;
    }
    energy_mb_row[row_off_H[H]+j] = decomp;
  }

  //energy_3p_00: inlined extend_fm_3p() fragment, fill_arrays.c:551-565
  //(cp==-1 forced, so ON_SAME_STRAND(...) is always true) -- default INF,
  //overwritten only if the MB_LOOP_ENC bit is set.
  {
    int e00 = INF;
    if(Hc2(ij,Hccc_mbenc)){
      // Staggered_Row_Batching: the i==1 wrap must land on THIS record's last
      // base, not the batch's. `length` is max(VC[H]->length) over the chunk,
      // so S_H[length] indexed a per-H block of only VC[H]->length+2 shorts:
      // for any H that isn't the longest it silently read the *next* record's
      // bases, and for the last H in a chunk it ran off the end of d_S2
      // entirely (illegal access, reproduced with descending-length input at
      // RNA_GPU_VRAM_BUDGET_MB=8/16 -- the tail chunk there is 13 records with
      // batch max 560 whose last record is only 80nt). Recovered from the
      // offset table rather than a new parameter, same technique as
      // modular_decomposition_cuda()'s len_H: seq_off_H's stride is this
      // file's own VC[H]->length+2 (see init_gpu3()).
      const int length_H = (int)(seq_off_H[H+1] - seq_off_H[H]) - 2;
      // This kernel carried no asserts at all, which is the reason the bug
      // above survived: every S_H read stayed inside the *whole* d_S2
      // allocation for all but the last H of a chunk, so nothing trapped. The
      // two reads below are the only ones in this kernel indexed by anything
      // other than i or j, so bound them against H's own block. j <= length_H
      // because size_off_H is built from VC[H]->length - i - turn per H (see
      // fill_arrays_loop.c), making j < length_H + 1.
      assert(length_H >= 0 && j <= length_H);
      const short s_i1 = (i==1) ? S_H[length_H] : S_H[i-1];
      e00 = E_MLstem_device(type, s_i1, S_H[j+1], P);
    }
    energy_3p00_row[row_off_H[H]+j] = e00;
  }
}


// ==================== GPU-resident sweep: fml_scan_kernel ====================
//
// The device twin of fill_arrays_loop.c's fml_host (46.8 s of a 458.9 s run) --
// and the only one of the three that is not elementwise. Its j-loop is a
// recurrence:
//
//     E[j] = MIN2( A[j], (E[j-1] != INF && b[j] != INF) ? E[j-1] + b[j] : INF )
//     E[j0-1] = INF,   j0 = i+turn+1
//
//     A[j]      = MIN2(c_term[j], e3_term[j])
//     c_term[j] = (e3p00[j] != INF) ? new_C[j] + e3p00[j] : INF
//     e3_term[j] = (fml_prev[j] != INF) ? fml_prev[j] + en_i : INF
//     b[j]      = (up_ml[j] > 0) ? MLbase : INF
//     en_i      = (up_ml[i] > 0) ? MLbase : INF          (row-constant per H)
//
// Read `+` as tropical: INF is +infinity and absorbing, exactly matching the
// host's `!= INF` guards. Then each element is an affine min-plus map
// f_j(x) = min(A[j], x + b[j]), and those COMPOSE:
//
//     (a_L,c_L) then (a_R,c_R)  ==  ( min(a_R, a_L + c_R),  c_L + c_R )
//
// which is associative, with identity (INF, 0). So this is an ordinary
// inclusive scan over (a,c) pairs -- no algebraic transform, no segmented-scan
// special case: b[j] == INF makes c absorbing and severs the chain by itself.
//
// TWO THINGS THAT MUST NOT BE "TIDIED", both bit-exactness hazards:
//
//  1. THE HOST'S TWO INF GUARDS ARE NOT SYMMETRIC. `en0` guards both operands;
//     `e3` guards only fml_prev. So when fml_prev[j] is a real energy and en_i
//     is INF, the host computes fml_prev[j] + 10000000 -- a large positive
//     number, NOT INF. It is never selected in practice, but it is not the same
//     value, and A[j] reproduces it exactly (see e3 below). Do not add the
//     missing guard.
//  2. The INF test is `== INF`, never `>= INF`. Item 1 can produce values above
//     INF, and absorbing those would diverge from the host.
//
// ML_BASE37 is 0 by default, so the composed c is normally 0 and the tropical
// sums cannot grow at all; even a non-default parameter file bounds them by
// length*|MLbase|, far inside int32.
//
// Launch shape: ONE BLOCK PER RECORD, tiling j with a carried E. `width` is
// block-uniform, so every __syncthreads() below is reached by the whole block
// -- including the width==0 early return for a record that has not joined the
// sweep yet.
// Default tile width. TW is a template parameter rather than a bare constant
// because it is BOTH the block size and the scan's tile width: it sizes the two
// shared arrays and sets the Hillis-Steele depth, so it must stay a
// compile-time constant and cannot be handed to rnafold_choose_block_size()'s
// occupancy heuristic the way this file's other five launch shapes are.
// RNA_FML_SCAN_THREADS overrides it; see fml_scan_tile() below.
//
// Changing TW regroups the scan but CANNOT change its result: the composition
// (a,c) is exactly associative over int here -- tropical min/add with INF
// absorbing, no floating point, and values bounded by ~1e7 (hazard 1's
// unguarded fml_prev+en_i) or by length*|MLbase| with a non-default parameter
// file, both far inside int32. So every width is bit-identical, and
// RNA_ROW_VERIFY confirms it per cell rather than taking that on trust.
#define FML_SCAN_THREADS 256

// Tropical add: INF absorbs. Mirrors the host's `(x != INF && y != INF) ? x+y : INF`
// -- equality, not >=, deliberately (see hazard 2 above).
__device__ __forceinline__ int fml_tadd(const int x, const int y) {
  return (x == INF || y == INF) ? INF : x + y;
}
__device__ __forceinline__ int fml_tmin(const int x, const int y) { return (x < y) ? x : y; }

template<int TW>
__global__ void
fml_scan_kernel(const int nfiles, const int i, const int turn,
                const int*  __restrict__ new_e,      //in  d_new_e      == host new_C
                const int*  __restrict__ e3p00,      //in  d_energy_3p00_row
                const int*  __restrict__ fml_prev,   //in  d_fml_prev   (previous row)
                const char* __restrict__ up_ml_ok,   //in  d_up_ml_ok
                const cuda_param2_t* __restrict__ P, //in  d_param2 (MLbase)
                      int*  __restrict__ energy_min, //out d_energy_min
                const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ seq_off_H,
                const size_t* __restrict__ size_off_H) {
  const int H = blockIdx.x;
  if(H >= nfiles) return;
  const long long width = (long long)size_off_H[H+1] - (long long)size_off_H[H];
  if(width <= 0) return;                  // has not joined the sweep -- whole block returns
  const size_t o  = row_off_H[H];
  const size_t so = seq_off_H[H];
  const int j0     = i + turn + 1;
  const int MLbase = P->MLbase;
  const int en_i   = up_ml_ok[so + (size_t)i] ? MLbase : INF;

  __shared__ int sa[TW];
  __shared__ int sc[TW];

  int carry = INF;                        // E[j0-1], the host's `j == i+turn+1` special case
  const int t = threadIdx.x;

  for(long long base = 0; base < width; base += TW) {
    const long long k = base + t;
    int a = INF, c = 0;                   // identity, for lanes past the end
    if(k < width) {
      const int j = j0 + (int)k;
      const int c_term = (e3p00[o+j] != INF) ? new_e[o+j] + e3p00[o+j] : INF;
      const int fp     = fml_prev[o+j];
      const int e3     = (fp != INF) ? fp + en_i : INF;   // hazard 1: en_i NOT guarded
      a = fml_tmin(c_term, e3);
      c = up_ml_ok[so + (size_t)j] ? MLbase : INF;
    }
    sa[t] = a; sc[t] = c;
    __syncthreads();

    // Hillis-Steele inclusive scan over the composition above. Left operand is
    // the earlier segment (applied first), right the later one.
    for(int d = 1; d < TW; d <<= 1) {
      int na = sa[t], nc = sc[t];
      if(t >= d) {
        na = fml_tmin(sa[t], fml_tadd(sa[t-d], sc[t]));
        nc = fml_tadd(sc[t-d], sc[t]);
      }
      __syncthreads();
      sa[t] = na; sc[t] = nc;
      __syncthreads();
    }

    if(k < width) {
      const int j = j0 + (int)k;
      energy_min[o+j] = fml_tmin(sa[t], fml_tadd(carry, sc[t]));
    }
    // Carry E across the tile boundary. Every thread reads the same lane, so
    // the new carry is block-uniform without a broadcast.
    const long long rem  = width - base;
    const int       last = (int)((rem < TW) ? (rem - 1) : (TW - 1));
    const int nextcarry  = fml_tmin(sa[last], fml_tadd(carry, sc[last]));
    __syncthreads();          // everyone has read sa/sc before the next tile overwrites them
    carry = nextcarry;
  }
}

// Upload size_off_H, but only when it differs from what the device already
// holds.
//
// GPU-resident sweep, step 5b. Four functions in this file upload this table
// before their launch -- hp_mb_3p_i, new_c_i, fml_scan_i, fml_prev_i -- and all
// four run in the SAME sweep row with the SAME table into the SAME buffer, so
// three of the four are pure duplication. That matters because cudaMemcpy H2D
// is BLOCKING and stream-ordered: each one is a sync point in its own right, so
// leaving them in place would make dropping the cudaDeviceSynchronize() calls
// worth almost nothing.
//
// Compared by CONTENT rather than by call order or a row counter. Call-order
// coupling ("only the first caller uploads") would break silently into wrong
// indices if the row body were ever reordered; content comparison cannot. The
// table genuinely changes every row (widths are length_H[H]-i-turn), so this
// skips only the within-row duplicates, never a required upload.
//
// HAZARD, and the reason for size_off_shadow_reset(): d_size_off_H is freed and
// re-cudaMalloc'd per chunk. Without an explicit reset, a fresh chunk whose
// first table happened to equal the previous chunk's last one would skip the
// upload and leave the new buffer UNINITIALISED -- garbage offsets, wrong
// answers, no crash. init_gpu3() calls the reset for exactly that reason.
static void
size_off_shadow_reset(void) {
  free(size_off_shadow);
  size_off_shadow   = NULL;
  size_off_shadow_n = 0;
}

static void
upload_size_off_H(const int nfiles, const size_t* size_off_H) {
  const int    n     = nfiles + 1;
  const size_t bytes = (size_t)n * sizeof(size_t);

  if(size_off_shadow_n != n) {              // first row of a chunk, or width changed
    free(size_off_shadow);
    size_off_shadow   = (size_t*)malloc(bytes);
    size_off_shadow_n = size_off_shadow ? n : 0;
  } else if(size_off_shadow && memcmp(size_off_shadow, size_off_H, bytes) == 0) {
    return;                                 // device already holds exactly this
  }

  gpuErrchk( cudaMemcpy(d_size_off_H, size_off_H, bytes, cudaMemcpyHostToDevice) );
  if(size_off_shadow) memcpy(size_off_shadow, size_off_H, bytes);
}

// Tile width for fml_scan_kernel, chosen once. Cached because fml_scan_i runs
// once per sweep row -- 16803 times in a 400x5601 fold -- and a getenv() per
// row would be pure waste.
//
// 256 was a guess when the scan was written, never measured. This makes it
// sweepable without a rebuild. It deliberately does NOT go through
// rnafold_choose_block_size(): that hands back whatever the occupancy heuristic
// prefers, but TW here is also the scan's tile width, so it must be one of the
// instantiated compile-time constants -- hence an explicit allow-list, and an
// invalid value warns and falls back rather than silently picking something
// that was never instantiated.
static int
fml_scan_tile(void) {
  static int v = -1;
  if(v < 0) {
    v = FML_SCAN_THREADS;
    const char* e = getenv("RNA_FML_SCAN_THREADS");
    if(e && *e) {
      const int want = atoi(e);
      if(want==32 || want==64 || want==128 || want==256 || want==512 || want==1024) {
        v = want;
        fprintf(stderr,"%-24s RNA_FML_SCAN_THREADS=%d overriding the default %d\n",
                __FILE__, want, FML_SCAN_THREADS);
      } else {
        fprintf(stderr,"%-24s ignoring RNA_FML_SCAN_THREADS=%s "
                "(want one of 32/64/128/256/512/1024)\n", __FILE__, e);
      }
    }
  }
  return v;
}

// Launches fml_scan_kernel and, under RNA_ROW_VERIFY, checks it against the
// host loop still running beside it.
//
// Call AFTER fml_host and BEFORE the load_fML/modular_decomposition/load_min_fML
// graph trio: that trio uploads the host's energy_min over d_energy_min, so the
// readback has to happen first -- and the upload landing afterwards is what
// keeps this step behaviour-neutral while both paths run.
PUBLIC void
fml_scan_i(const int nfiles, const int i, const int turn,
           const int* energy_min_host,            //in, host's own result (verify only)
           const size_t* row_off_H,               //in, nfiles+1
           const size_t* size_off_H) {            //in, nfiles+1
  if(size_off_H[nfiles]==0) return;

  int* d_new_e_ = NULL;
  int_loop_row_buffers(NULL, &d_new_e_);
  int* d_fml_prev_ = NULL; int* d_energy_min_ = NULL;
  md_row_buffers(NULL, NULL, &d_fml_prev_, &d_energy_min_);

  upload_size_off_H(nfiles, size_off_H);   // skips this row's redundant re-uploads

  // One block per record, not a flat grid: the scan carries state along j, so a
  // record's row has to stay inside one block.
  //
  // Dispatched on a runtime tile width rather than launched at a fixed one.
  // The instantiate-a-few-and-switch shape follows int_loop_kernel's 32/64/
  // 128/256 precedent, and the env override follows RNA_MD_TILE's.
#define FML_SCAN_LAUNCH(TW) \
  fml_scan_kernel<TW><<<nfiles,TW>>>(nfiles, i, turn, \
                                     d_new_e_, d_energy_3p00_row, d_fml_prev_, \
                                     d_up_ml_ok, d_param2, d_energy_min_, \
                                     d_row_off_H, d_seq_off_H, d_size_off_H)
  switch(fml_scan_tile()) {
    case   32: FML_SCAN_LAUNCH(  32); break;
    case   64: FML_SCAN_LAUNCH(  64); break;
    case  128: FML_SCAN_LAUNCH( 128); break;
    case  512: FML_SCAN_LAUNCH( 512); break;
    case 1024: FML_SCAN_LAUNCH(1024); break;
    default:   FML_SCAN_LAUNCH(FML_SCAN_THREADS); break;
  }
#undef FML_SCAN_LAUNCH
  gpuErrchk( cudaPeekAtLastError() );
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );

  static int verify = -1;
  if(verify < 0) verify = (getenv("RNA_ROW_VERIFY") != NULL);
  if(!verify || energy_min_host == NULL) return;  // device mode: no host result to compare


  static int   *mirror = NULL;
  static size_t mirror_cells = 0;
  const size_t cells = row_off_H[nfiles];
  if(mirror_cells < cells) {
    free(mirror);
    mirror = (int*) malloc(cells*sizeof(int));
    if(!mirror) { fprintf(stderr,"%-24s RNA_ROW_VERIFY: out of host memory\n",__FILE__); return; }
    mirror_cells = cells;
  }
  gpuErrchk( cudaMemcpy(mirror, d_energy_min_, cells*sizeof(int), cudaMemcpyDeviceToHost) );

  static long long checked = 0, bad = 0;
  for(int H=0; H<nfiles; H++) {
    const size_t o     = row_off_H[H];
    const size_t width = size_off_H[H+1] - size_off_H[H];
    if(width == 0) continue;
    for(size_t k=0; k<width; k++) {
      const size_t idx = o + (size_t)(i+turn+1) + k;
      checked++;
      if(mirror[idx] == energy_min_host[idx]) continue;
      if(++bad <= 20)
        fprintf(stderr,"%-24s RNA_ROW_VERIFY fml_scan MISMATCH H=%d i=%d j=%d gpu=%d host=%d\n",
                __FILE__, H, i, (int)((i+turn+1)+k), mirror[idx], energy_min_host[idx]);
    }
  }
  if(i == 1)
    fprintf(stderr,"%-24s RNA_ROW_VERIFY fml_scan: %lld cells checked, %lld mismatching\n",
            __FILE__, checked, bad);
}


// ====================== GPU-resident sweep: new_c_kernel ======================
//
// The device twin of fill_arrays_loop.c's new_c_host -- 49.7 s of a 458.9 s run,
// the largest of the three per-row host loops and the one whose inputs make the
// case for this whole port. Every value it reads is ALREADY on the device and is
// shipped to the host each row purely so the host can take a minimum:
//
//   energy_min2   int_loop_kernel's output for this row (int_loop.cu). Today it
//                 is D2H'd into the host's energy_min so new_c_host can read it.
//                 NOTE this is the interior-loop energy, not the fML extension:
//                 energy_min means two different things either side of fml_host
//                 (TRAP 1 in the row-loop notes), and this is the earlier one.
//   energy_hp_row hairpin energies, from hp_mb_3p_kernel a moment ago.
//   energy_mb_row multibranch energies, likewise.
//   gate_row      bit0 = hc->matrix[ij] != 0, bit1 = ptype[ij] is GU/UG. Also
//                 from hp_mb_3p_kernel -- which is to say new_c_host's two
//                 remaining triangle reads were ALREADY moved to the GPU in
//                 81d20a6, and then shipped back down to be read once.
//   dml1          row i+1's DMLi, published by md_snapshot_dml() at the end of
//                 the previous row (step 1).
//
// There is no recurrence over j: every cell is independent, which is why this
// one is a straight elementwise port and the scan is left for K2.
//
// e_mb: the host computes MIN2(new_c, INF) when DMLi1[j-1] is INF, which is a
// no-op, so the guard skips the min instead of materialising INF. Same result,
// one fewer add that could overflow.
//
// At the left edge j == i+turn+1 the dml1 read lands on (i+1, i+turn), inside
// the diagonal band -- a cell no row ever computes, so it holds the INF that
// init_fML_kernel put there. Same value the host's rotated DMLi1 holds, for the
// same reason; RNA_ROW_VERIFY covers it because the comparison starts at the
// first j, not the second.
__global__ void
new_c_kernel(const int nfiles, const int i, const int turn, const int noGUclosure,
             const int*  __restrict__ energy_min2,    //in  d_energy_min2
             const int*  __restrict__ energy_hp_row,  //in
             const int*  __restrict__ energy_mb_row,  //in
             const char* __restrict__ gate_row,       //in
             const int*  __restrict__ dml1,           //in  d_dml1
                   int*  __restrict__ new_e,          //out d_new_e
             const size_t* __restrict__ row_off_H,    //in
             const size_t* __restrict__ size_off_H, const size_t total) { //in
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int j = mj + i+turn+1;
  const size_t o = row_off_H[H];

  const unsigned char gate = (unsigned char)gate_row[o+j];
  if(!(gate & 1)) { new_e[o+j] = INF; return; }   // pair not evaluated

  int new_c = energy_min2[o+j];
  if(!(((gate & 2) != 0) && noGUclosure)) {
    const int hp = energy_hp_row[o+j];
    if(hp < new_c) new_c = hp;
    const int d1 = dml1[o+(j-1)];
    if(d1 != INF) {
      const int e_mb = d1 + energy_mb_row[o+j];
      if(e_mb < new_c) new_c = e_mb;
    }
  }
  new_e[o+j] = new_c;
}

// Launches new_c_kernel and, under RNA_ROW_VERIFY, checks it cell-for-cell
// against the host loop still running beside it.
//
// Call this AFTER new_c_host and BEFORE load_my_c(): load_my_c uploads the
// host's new_C over d_new_e, so the readback has to happen first, and the
// upload landing afterwards is what keeps this step behaviour-neutral -- the
// kernel's output is written and then overwritten with the identical host
// values, so the sweep cannot yet be affected by a bug here.
PUBLIC void
new_c_i(const int nfiles, const int i, const int turn, const int noGUclosure,
        const int* new_C_host,                 //in, host's own result (verify only)
        const size_t* row_off_H,               //in, nfiles+1
        const size_t* size_off_H) {            //in, nfiles+1
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

  int* d_energy_min2_ = NULL; int* d_new_e_ = NULL;
  int_loop_row_buffers(&d_energy_min2_, &d_new_e_);
  int* d_dml1_ = NULL;
  md_row_buffers(NULL, &d_dml1_, NULL, NULL);

  static int block_size = 0;
  if(!block_size) {
    block_size = rnafold_choose_block_size(new_c_kernel, BLOCK_SIZE, "RNA_NEW_C_BLOCK_SIZE");
    fprintf(stderr,"%-24s new_c_kernel block size %d\n", __FILE__, block_size);
  }

  upload_size_off_H(nfiles, size_off_H);   // skips this row's redundant re-uploads

  const size_t nblocks = (total + block_size - 1)/block_size;
  new_c_kernel<<<(int)nblocks,block_size>>>(nfiles, i, turn, noGUclosure,
                                            d_energy_min2_, d_energy_hp_row, d_energy_mb_row,
                                            d_gate_row, d_dml1_, d_new_e_,
                                            d_row_off_H, d_size_off_H, total);
  gpuErrchk( cudaPeekAtLastError() );
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );

  static int verify = -1;
  if(verify < 0) verify = (getenv("RNA_ROW_VERIFY") != NULL);
  if(!verify || new_C_host == NULL) return;  // device mode: no host result to compare


  static int   *mirror = NULL;
  static size_t mirror_cells = 0;
  const size_t cells = row_off_H[nfiles];
  if(mirror_cells < cells) {
    free(mirror);
    mirror = (int*) malloc(cells*sizeof(int));
    if(!mirror) { fprintf(stderr,"%-24s RNA_ROW_VERIFY: out of host memory\n",__FILE__); return; }
    mirror_cells = cells;
  }
  gpuErrchk( cudaMemcpy(mirror, d_new_e_, cells*sizeof(int), cudaMemcpyDeviceToHost) );

  static long long checked = 0, bad = 0;
  for(int H=0; H<nfiles; H++) {
    const size_t o     = row_off_H[H];
    const size_t width = size_off_H[H+1] - size_off_H[H];
    if(width == 0) continue;   // not joined the sweep yet
    for(size_t k=0; k<width; k++) {
      const size_t idx = o + (size_t)(i+turn+1) + k;
      checked++;
      if(mirror[idx] == new_C_host[idx]) continue;
      if(++bad <= 20)
        fprintf(stderr,"%-24s RNA_ROW_VERIFY new_c MISMATCH H=%d i=%d j=%d gpu=%d host=%d\n",
                __FILE__, H, i, (int)((i+turn+1)+k), mirror[idx], new_C_host[idx]);
    }
  }
  if(i == 1)
    fprintf(stderr,"%-24s RNA_ROW_VERIFY new_c: %lld cells checked, %lld mismatching\n",
            __FILE__, checked, bad);
}


// ===================== GPU-resident sweep: fml_prev_kernel =====================
//
// The device twin of fill_arrays_loop.c's fml_prev_host, the smallest of the
// three per-row host loops (8.1 s of a 458.9 s run) and therefore the one that
// proves the cross-file plumbing before the two that matter carry any weight.
//
// It publishes row i's FINAL fML, row-shaped, for row i-1 to read one row later
// as fML(i+1,j):
//
//     fml_prev[j] = MIN2(energy_min[j], DMLi[j])      j in [i+turn+1, len_H]
//     fml_prev[i+turn] = INF
//
// Both inputs are already device-resident at this point in the row, which is
// the whole reason this loop should never have been on the host:
//   energy_min  -> d_energy_min, uploaded inside the CUDA-graph trio a moment
//                  earlier (modular_decomposition.cu). Holds row i's fML
//                  extension, NOT the interior-loop energies -- see TRAP 1 in
//                  the row-loop notes: energy_min means two different things
//                  either side of fml_host.
//   DMLi        -> d_dml, just written by load_min_fML_kernel.
//
// The j range is exactly [i+turn+1, len_H], width len_H-i-turn, which is what
// size_off_H already encodes -- so this shares hp_mb_3p_kernel's flattening
// unchanged and needs no table of its own.
//
// The lone i+turn cell: (i, i+turn) is inside the diagonal band, a cell no row
// ever computes. Row i-1 reads it as its own j-1 at the left edge, so it has to
// read INF rather than row i+1's leftover value. The host writes it explicitly
// once per joined H; here the k==0 thread does it, which is the same set of
// records for the same reason (a record with width>=1 has joined).
__global__ void
fml_prev_kernel(const int nfiles, const int i, const int turn,
                const int* __restrict__ energy_min,   //in  d_energy_min
                const int* __restrict__ dml,          //in  d_dml
                      int* __restrict__ fml_prev,     //out d_fml_prev
                const size_t* __restrict__ row_off_H, //in
                const size_t* __restrict__ size_off_H, const size_t total) { //in
  const long long m = blockIdx.x*blockDim.x+threadIdx.x;
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int j = mj + i+turn+1;
  const size_t o = row_off_H[H];

  if(mj == 0) fml_prev[o + (i+turn)] = INF;   // the diagonal-band cell, per H

  const int a = energy_min[o+j];
  const int b = dml[o+j];
  fml_prev[o+j] = (a < b) ? a : b;            // MIN2, without the host macro
}

// Launches fml_prev_kernel, and -- while RNA_ROW_VERIFY is set -- checks it
// against the host loop that still runs beside it.
//
// The verification is the point of this step. Byte-identical fold output only
// proves the values are right WHERE THEY ARE READ; comparing the whole row
// proves every cell matches, including ones nothing looks at today, so widening
// a read range later cannot turn a latent mismatch into a wrong answer. This is
// the same argument RNA_HC_VERIFY was built on for the Step 1 bitmasks, and it
// caught real bugs there.
//
// Reports the first mismatching (H, i, j) with both values and then keeps
// counting, so one run yields the shape of the disagreement rather than just
// its existence. Deliberately fprintf + counters, never assert: asserts in the
// .c files are compiled out by -DNDEBUG, and while this one is in a .cu file
// its callers are not -- so the habit is what matters.
PUBLIC void
fml_prev_i(const int nfiles, const int i, const int turn,
           const int* fml_prev_host,              //in, host's own result (verify only)
           const size_t* row_off_H,               //in, nfiles+1
           const size_t* size_off_H) {            //in, nfiles+1
  const size_t total = size_off_H[nfiles];
  if(total==0) return;

  int* d_energy_min_ = NULL; int* d_dml_ = NULL; int* d_fml_prev_ = NULL;
  md_row_buffers(&d_dml_, NULL, &d_fml_prev_, &d_energy_min_);

  static int block_size = 0;
  if(!block_size) {
    block_size = rnafold_choose_block_size(fml_prev_kernel, BLOCK_SIZE, "RNA_FML_PREV_BLOCK_SIZE");
    fprintf(stderr,"%-24s fml_prev_kernel block size %d\n", __FILE__, block_size);
  }

  upload_size_off_H(nfiles, size_off_H);   // skips this row's redundant re-uploads

  const size_t nblocks = (total + block_size - 1)/block_size;
  fml_prev_kernel<<<(int)nblocks,block_size>>>(nfiles, i, turn,
                                               d_energy_min_, d_dml_, d_fml_prev_,
                                               d_row_off_H, d_size_off_H, total);
  gpuErrchk( cudaPeekAtLastError() );
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );

  // Cached, not a getenv() per row: this runs 16803 times in a 400x5601 fold.
  static int verify = -1;
  if(verify < 0) verify = (getenv("RNA_ROW_VERIFY") != NULL);
  if(!verify || fml_prev_host == NULL) return;  // device mode: no host result to compare


  // ---- RNA_ROW_VERIFY: pull it back and compare against the host's array ----
  static int   *mirror = NULL;
  static size_t mirror_cells = 0;
  const size_t cells = row_off_H[nfiles];
  if(mirror_cells < cells) {
    free(mirror);
    mirror = (int*) malloc(cells*sizeof(int));
    if(!mirror) { fprintf(stderr,"%-24s RNA_ROW_VERIFY: out of host memory\n",__FILE__); return; }
    mirror_cells = cells;
  }
  gpuErrchk( cudaMemcpy(mirror, d_fml_prev_, cells*sizeof(int), cudaMemcpyDeviceToHost) );

  static long long checked = 0, bad = 0;
  for(int H=0; H<nfiles; H++) {
    const size_t o     = row_off_H[H];
    const size_t width = size_off_H[H+1] - size_off_H[H];
    if(width == 0) continue;   // this H has not joined the sweep yet
    // width+1 cells: the diagonal-band cell at i+turn plus [i+turn+1, len_H].
    for(size_t k=0; k<=width; k++) {
      const size_t idx = o + (size_t)(i+turn) + k;
      checked++;
      if(mirror[idx] == fml_prev_host[idx]) continue;
      if(++bad <= 20)
        fprintf(stderr,"%-24s RNA_ROW_VERIFY fml_prev MISMATCH H=%d i=%d j=%d gpu=%d host=%d\n",
                __FILE__, H, i, (int)((i+turn)+k), mirror[idx], fml_prev_host[idx]);
    }
  }
  if(i == 1)   // last row of the sweep: report once per chunk
    fprintf(stderr,"%-24s RNA_ROW_VERIFY fml_prev: %lld cells checked, %lld mismatching\n",
            __FILE__, checked, bad);
}

PUBLIC void
hp_mb_3p_i(const int nfiles, const vrna_fold_compound_t **VC,
           const int i, const int turn, const int length,
           int* energy_hp_row, int* energy_mb_row, int* energy_3p00_row, //all out, size nfiles*(length+1)
           char* gate_row, //out, same shape but one byte per cell
           const size_t* size_off_H) { //in, nfiles+1 entries -- Staggered_Row_Batching Phase 5
  const size_t total = size_off_H[nfiles];
  //total==0 is unreachable for the i range fill_arrays_loop.c's main loop
  //actually uses (start=i+turn+1 <= length always holds there) -- same dead
  //guard load_fML() carries for the identical reason, kept for symmetry.
  if(total==0) return;

  // Block size picked once from the actual GPU present (see stub2.h's
  // rnafold_choose_block_size()) instead of the BLOCK_SIZE constant this
  // used to hardcode -- BLOCK_SIZE=512 was tuned against one GPU (the L4);
  // this kernel has no shared memory or reduction tying it to a specific
  // size, so there's no reason not to let CUDA pick per-device.
  static int block_size = 0;
  if(!block_size) {
    block_size = rnafold_choose_block_size(hp_mb_3p_kernel, BLOCK_SIZE, "RNA_HP_MB_BLOCK_SIZE");
    fprintf(stderr,"%-24s hp_mb_3p_kernel block size %d (was hardcoded %d)\n",
	    __FILE__, block_size, BLOCK_SIZE);
  }

  upload_size_off_H(nfiles, size_off_H);   // skips this row's redundant re-uploads

  const int nblocks = (total + block_size - 1)/block_size;
  hp_mb_3p_kernel<<<nblocks,block_size>>>(nfiles, i, turn, length,
                                          d_S2, d_sequence, d_pair2,
                                          d_hccc_mb, d_hccc_mbenc,
                                          d_hccc_any, d_hccc_gu, d_param2,
                                          d_energy_hp_row, d_energy_mb_row, d_energy_3p00_row,
                                          d_gate_row,
                                          d_row_off_H, d_hc2_off_H, d_seq_off_H,
                                          d_size_off_H, total);
  gpuErrchk( cudaPeekAtLastError() );
  // Step 5b: pointless once the D2H is gone; stream order already covers it.
  // Full rationale on rnafold_gpu_sweep() in stub2.h.
  if(!rnafold_gpu_sweep())
    gpuErrchk( cudaDeviceSynchronize() );

  // GPU-resident sweep: in device mode new_c_kernel and fml_scan_kernel read
  // all four of these in place. Four of the six per-row transfers, and the
  // largest group of them.
  if(!rnafold_gpu_sweep()) {
    const size_t rowsize = g_row_total*sizeof(int);
    gpuErrchk( cudaMemcpy(energy_hp_row,  d_energy_hp_row,  rowsize,cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(energy_mb_row,  d_energy_mb_row,  rowsize,cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(energy_3p00_row,d_energy_3p00_row,rowsize,cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(gate_row,       d_gate_row,       g_row_total*sizeof(char),cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaDeviceSynchronize() );
  }
}

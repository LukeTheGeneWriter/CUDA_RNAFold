/* hp_mb_dev.h -- device helpers shared with the fused per-record megakernel.
 *
 * Moved verbatim out of hp_mb_loop.cu. The megakernel calls the same
 * arithmetic the standalone kernels do, and device code cannot cross a
 * translation unit in this build. -rdc would allow the call but would stop it
 * INLINING, and these run in the innermost loops of an O(n^3) sweep.
 */
#ifndef RNAFOLD_HP_MB_DEV_H
#define RNAFOLD_HP_MB_DEV_H

/* The hard-constraint bitmask word width. Defined here as well as in the
 * owning .cu because the helpers moved above that file's own definition, and
 * an identical macro redefinition is legal. */
#ifndef bitsperint
#define bitsperint (8*sizeof(unsigned int))
#endif

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
  int   dangles;    //0 or 2; anything else is refused before the sweep starts
  int   Tetraloop_E[200];
  char  Tetraloops[1401];
  int   Triloop_E[40];
  char  Triloops[241];
  int   Hexaloop_E[40];
  char  Hexaloops[1801];
  //noLP (--noLP) only, appended last so no offset above it moves -- same
  //discipline as int_loop.cu's salt fields. Needed by stack_row_kernel for
  //upstream's vrna_eval_stack(): P->stack[type][type_2].
  int   stack[NBPAIRS+1][NBPAIRS+1];
};

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
                  const char* __restrict__ seq, const cuda_param2_t* __restrict__ P,
                  const int* __restrict__ salt_loop) {
  int energy;

  // Salt (eval/hairpin.h:352-371). Upstream branches on md.salt and then on
  // size <= MAXLOOP; both branches are folded into the table on the host, so
  // this is one load and the default-salt case adds a zero. It is added BEFORE
  // the special-hairpin lookups and carried into each of their early returns,
  // which is upstream's order -- getting that backwards would leave the
  // tetraloop/triloop/hexaloop cases uncorrected and nothing else would notice.
  const int salt = salt_loop[size+1];

  if(size <= 30) energy = P->hairpin[size];
  else            energy = P->hairpin[30] + (int)(P->lxc*log((double)size/30.)); //double log() to match host precision (hairpin_loops.h:116)

  energy += salt;

  if(size < 3) return energy; /* should only be the case when folding alignments */

  if(P->special_hp){
    if(size == 4){
      char tl[7] = {0};
      for(int k=0;k<6;k++) tl[k] = seq[k];
      for(int off=0; off+7<=1401; off+=7){
        if(P->Tetraloops[off]==0) break; //end of populated entries
        int match=1;
        for(int k=0;k<6;k++) if(P->Tetraloops[off+k]!=tl[k]) { match=0; break; }
        if(match) return P->Tetraloop_E[off/7] + salt;
      }
    }
    else if(size == 6){
      char tl[9] = {0};
      for(int k=0;k<8;k++) tl[k] = seq[k];
      for(int off=0; off+9<=1801; off+=9){
        if(P->Hexaloops[off]==0) break;
        int match=1;
        for(int k=0;k<8;k++) if(P->Hexaloops[off+k]!=tl[k]) { match=0; break; }
        if(match) return P->Hexaloop_E[off/9] + salt;
      }
    }
    else if(size == 3){
      char tl[6] = {0,0,0,0,0,0};
      for(int k=0;k<5;k++) tl[k] = seq[k];
      for(int off=0; off+6<=241; off+=6){
        if(P->Triloops[off]==0) break;
        int match=1;
        for(int k=0;k<5;k++) if(P->Triloops[off+k]!=tl[k]) { match=0; break; }
        if(match) return P->Triloop_E[off/6] + salt;
      }
      return energy + (type>2 ? P->TerminalAU : 0);
    }
  }
  energy += P->mismatchH[type][si1][sj1];
  return energy;
}

// Replicates E_MLstem(), eval/multibranch.h:163-184. si1/sj1 of -1 mean "no
// neighbour", which is how dangle model 0 asks for a bare stem; both callers
// below choose between the real bases and -1 on P->dangles. (This comment used
// to say si1/sj1 are ALWAYS >=0 -- true while d2 was the only accepted model,
// false since d0 landed 2026-09-11.)
//
// ONLY TWO OF UPSTREAM'S THREE CASES ARE HERE. The one-sided dangle5/dangle3
// cases cannot arise under d0 (passes -1,-1) or d2 (passes >=0,>=0), and d1/d3
// are declined by the engine guard and tripwired in fill_arrays.c. Testing both
// operands rather than one keeps a stray single-sided call out of
// mismatchM[type][si1][-1] -- an out-of-bounds read -- and makes it a plain 0
// instead, which the tripwire is there to catch.
//
// AND THE -1 HANDLING IS CURRENTLY REDUNDANT, deliberately. Upstream ZEROES
// P->mismatchM when dangles == 0 (params.c:644-646), so passing the real bases
// under d0 already yields the bare-stem energy. Measured, not assumed: 0 of the
// mismatchM entries are nonzero at d0 against 175 at d2. Red-teamed too --
// reverting either caller below to the unconditional d2 form changes NOTHING
// (0 differing lines on the 20-record bar).
//
// It stays because it makes this function's correctness LOCAL. As written, the
// port is right whether or not a parameter table two files away is zeroed;
// without it, d0 would silently become d2 if that zeroing ever changed, and
// nothing here would notice. Do not delete it as dead code -- it is a
// deliberate, measured redundancy.
__device__ inline int
E_MLstem_device(const int type, const int si1, const int sj1, const cuda_param2_t* __restrict__ P) {
  int energy = ((si1 >= 0) && (sj1 >= 0)) ? P->mismatchM[type][si1][sj1] : 0;
  if(type > 2) energy += P->TerminalAU;
  energy += P->MLintern[type];
  return energy;
}

// Tropical add: INF absorbs. Mirrors the host's `(x != INF && y != INF) ? x+y : INF`
// -- equality, not >=, deliberately (see hazard 2 above).
__device__ __forceinline__ int fml_tadd(const int x, const int y) {
  return (x == INF || y == INF) ? INF : x + y;
}

__device__ __forceinline__ int fml_tmin(const int x, const int y) { return (x < y) ? x : y; }

#endif /* RNAFOLD_HP_MB_DEV_H */

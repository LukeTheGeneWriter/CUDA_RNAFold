/* int_loop_dev.h -- device helpers shared with the fused per-record megakernel.
 *
 * Moved verbatim out of int_loop.cu. The megakernel calls the same
 * arithmetic the standalone kernels do, and device code cannot cross a
 * translation unit in this build. -rdc would allow the call but would stop it
 * INLINING, and these run in the innermost loops of an O(n^3) sweep.
 */
#ifndef RNAFOLD_INT_LOOP_DEV_H
#define RNAFOLD_INT_LOOP_DEV_H

/* The hard-constraint bitmask word width. Defined here as well as in the
 * owning .cu because the helpers moved above that file's own definition, and
 * an identical macro redefinition is legal. */
#ifndef bitsperint
#define bitsperint (8*sizeof(unsigned int))
#endif

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

//since q traditionally counts down this is smallest value
__device__ inline int 
Min_q(const int i, const int j, const int turn_) { //max_q
  return MAX2(i+turn+2, j - MAXLOOP - 1);
}

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

struct cell_inv_t { int type; int si1; int sj1; };

__device__ inline cell_inv_t
cell_invariants(const unsigned int* __restrict__ S, const char* __restrict__ pair_,
                const int H, const int nfiles, const int i, const int j) {
  cell_inv_t c;
  // vrna_get_ptype_md() PROMOTES 0 -> 7 (alphabet.c:475-477); the raw pair value
  // is not the same thing, and omitting it here is the trap that made --nsp a
  // live wrong answer. See PORT_NSP_PARAMFILE_SCOPE.md 1.
  const unsigned char t = Ptype(S,pair_,H,nfiles,i,j);
  c.type = (t == 0) ? 7 : (int)t;
  c.si1  = unpack(S,H,nfiles,i+1);
  c.sj1  = unpack(S,H,nfiles,j-1);
  return c;
}

/*
 *  How a cell reads the `c` triangle.
 *
 *  Energy() reads it in exactly ONE place, so which memory that read lands in
 *  is a property of this one type rather than of the recurrence. The default
 *  below is the triangle in VRAM, which is what every standalone kernel uses
 *  and which inlines to the same LDG the code had before. The megakernel
 *  substitutes a reader backed by a shared-memory window over the same cells
 *  (MAXLOOP bounds the lookback to 31 rows, so a window can hold all of them).
 *
 *  A TYPE rather than a flag on purpose: a runtime branch would sit in the
 *  innermost loop of the hottest kernel, and two copies of the recurrence
 *  would drift -- and a drifted copy here returns a plausible structure, not a
 *  crash.
 */
struct c_tri_reader {
  const int *base;                /* already offset to this record's triangle */

  __device__ __forceinline__ int
  operator()(const int p, const int q) const {
    return base[Indx(p, q)];
  }
};

//interface to interior_loopx.h via IntLoop_X()
/*
 *  Stage 1b: the same read, served out of a shared-memory window.
 *
 *  MAXLOOP bounds an interior loop to 30 unpaired bases, so cell (i,j) reads
 *  c(p,q) only for p in [i+1, i+31] and q in [j-31, j-1]. Thirty-one rows of
 *  the block's own column span therefore hold EVERY c value the block can ask
 *  for, which is what makes a window possible at all.
 *
 *  The ring has 32 slots indexed by `p & 31` rather than 31 indexed by a
 *  modulo: the 31 live rows p in [i+1, i+31] are distinct mod 32, and so is
 *  the row being evicted (i+32 lands on i&31), so a power-of-two ring is both
 *  correct and a mask instead of a division.
 *
 *  Rows advance by ONE per sweep row, so only row i+1 is fetched per row --
 *  the other thirty are already on chip. That is the part only a resident
 *  kernel can do, and the reason this belongs to the megakernel.
 */
struct c_win_reader {
  const int *sm;        /* 32 rows x `stride` columns                     */
  int        q0;        /* absolute column of the window's first entry    */
  int        stride;    /* columns per ring slot                          */

  __device__ __forceinline__ int
  operator()(const int p, const int q) const {
    return sm[((p & 31) * stride) + (q - q0)];
  }
};

template<class CREAD>
__device__ inline int
Energy(const int H, const int nfiles, const int i, const int j, const int q, const int p,
       const cell_inv_t ci,   //H1: computed once per cell by the caller
	  /*const char* hard_constraints,*/ CREAD my_c,
	  // up_int for THIS record, or NULL when the batch carries no hard
	  // constraints. See the d_up_int comment at the top of this file.
	  const unsigned char* __restrict__ up_int,
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

  // THE OTHER HALF OF A HARD CONSTRAINT (wrap_internal_hc.inc:57-67): with
  // closing pair (i,j) and inner pair (p,q), the two unpaired runs are
  // u1 = p-i-1 and u2 = j-q-1, and each must fit in the allowed run that
  // starts after the base it follows. The line this restores was commented out
  // years ago as "not needed as using Hc" -- true only while nothing could
  // force a base to pair, which is exactly what the routing guard was for.
  if(up_int){
    const int u1 = p - i - 1;
    const int u2 = j - q - 1;
    if((u1 > 0) && ((int)up_int[i+1] < u1)) return INF;
    if((u2 > 0) && ((int)up_int[q+1] < u2)) return INF;
  }

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
	    energy = my_c(p,q);   // the ONE c read -- see c_tri_reader
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
	      // H1: hoisted to cell_invariants(), computed once per cell.
	      const int type = ci.type;
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

	      const int si1 = ci.si1;    //H1: hoisted
	      const int sj1 = ci.sj1;    //H1: hoisted
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

#endif /* RNAFOLD_INT_LOOP_DEV_H */

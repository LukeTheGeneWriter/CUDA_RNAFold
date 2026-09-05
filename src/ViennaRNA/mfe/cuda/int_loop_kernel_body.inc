// Included multiple times from int_loop.cu, once per candidate BLOCK_SIZE.
// Only BLOCK_SIZE=32 is actually selected right now (see the STOPGAP
// comment above int_loop_cuda() in int_loop.cu) -- the other three stay
// instantiated so a real selection strategy can pick among them again once
// one exists. Each inclusion #define's BLOCK_SIZE to a different power of
// two first, producing a differently-named, differently-sized instantiation
// of int_loop_kernel so the launch site can pick whichever the actual GPU
// present says is best, instead of the single hardcoded BLOCK_SIZE=32 this
// used to be (tuned against one GPU, the L4).
//
// This has to be a repeated #include rather than a C++ template<int
// BLOCK_SIZE> because the block-wide reduction below leans on #if
// BLOCK_SIZE > 32 (preprocessor, evaluated once per #include) to skip the
// cross-warp combine step entirely when there's only one warp. Re-deriving
// that as if-constexpr risked subtly changing already-correct,
// warp-synchronous logic for no benefit; #include-ing this file once per
// size instead reuses it completely unmodified.
//
// Not listed as a Makefile.am prerequisite for int_loop.o: every build of
// this project is a fresh `git clone` + full `make` on Colab (see project
// memory), so there's no incremental-build staleness this file could
// introduce -- nothing here is ever a partial rebuild.

#define INT_LOOP_KERNEL_NAME2(bs) int_loop_kernel_##bs
#define INT_LOOP_KERNEL_NAME(bs) INT_LOOP_KERNEL_NAME2(bs)

__global__ void
INT_LOOP_KERNEL_NAME(BLOCK_SIZE)(const int nfiles, const int i_row, /*const int turn,*/ const int length,
		const int TerminalAU, const int ninio2,
		const cuda_param_t* __restrict__ P, const float lxc,
		const char* __restrict__ pair_, //[NBPAIRS+1][NBPAIRS+1],
		const unsigned int* __restrict__ S,    //[length+2] packed
		const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		const int* __restrict__ my_c,
		const size_t* __restrict__ tri_off_H, //in, d_my_c's per-H block start
		const size_t* __restrict__ row_off_H, //in, energy_min's per-H row start
		const size_t* __restrict__ hc_off_H,  //in, d_hccc's per-H block start
		const size_t* __restrict__ size_off_H, //in, this launch's per-H block-count table
		const int* __restrict__ i_H,           //in, continuous flow phase A3: per-record row index
		      int* __restrict__ energy_min) { //out

  int energy = INF;
  // Staggered_Row_Batching Phase 5: one block per (H,j) cell, same as
  // before -- only how blockIdx.x maps to (H,j) changed. blockIdx.x is now
  // a flat block index across every H's own block range (replacing the old
  // 2-D dim3(nblocks,nfiles) grid with blockIdx.y==H directly), via the
  // same flatten_index_to_H() mechanism Phase 4 used at thread granularity.
  // Every thread in the block redoes this lookup redundantly rather than
  // computing once and broadcasting -- cheap (O(log nfiles)) next to the
  // block's actual cooperative-scan work below, and keeps this diff to
  // exactly the top-of-kernel H/j derivation, per the plan's own scoping
  // for this phase.
  const int H = flatten_index_to_H((size_t)blockIdx.x, size_off_H, nfiles);
  // Continuous flow phase A3. This kernel was missing from the plan's list of
  // nine, but it is row-indexed like all the rest: every use of the row index
  // below comes after H is known, so the substitution is the same single line.
  // Without it phase B would compute this row at the wrong i for every record
  // but one.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const int j = ((int)blockIdx.x - (int)size_off_H[H]) + i+turn+1;

  const long long ij = Indx(i,j);
  if(Hc(ij,&hccc[hc_off_H[H]])) { //emulate hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
    /* we evaluate this pair */
    const int p0 = i+1;
    const int q0 = Min_q(i,j,turn);
    const int maxcol = MIN2(MAXLOOP,(j - 1) - q0);
    const unsigned int* __restrict__ hccc_H = &hccc[hc_off_H[H]];

    // Cooperative decode + prefix-sum (see decode_column(), int_loop.cu,
    // and the "Cooperative Column Scan" design doc). maxcol+1 <= MAXLOOP+1
    // = 31 always fits in one warp regardless of BLOCK_SIZE -- only the
    // first warp does this part; every other warp just waits at the
    // syncthreads() below.
    __shared__ unsigned int col_mask[MAXLOOP+1];
    __shared__ int prefix[MAXLOOP+2];
    if(threadIdx.x < 32) {
      const int c = (int)threadIdx.x;
      int popc = 0;
      if(c <= maxcol) {
	col_mask[c] = decode_column(p0,q0,c,hccc_H);
	popc = __popc(col_mask[c]);
      }
      // Inclusive warp scan (Hillis-Steele) of popc across all 32 lanes --
      // lanes beyond maxcol contribute 0 and never affect lower lanes,
      // since __shfl_up_sync only ever pulls from *lower*-indexed lanes.
      // All 32 lanes must execute every shuffle here regardless of c, so
      // this loop is deliberately outside the `c <= maxcol` guard above.
      int incl = popc;
#pragma unroll
      for(int offset = 1; offset < 32; offset <<= 1) {
	const int n = __shfl_up_sync(0xffffffff, incl, offset);
	if(c >= offset) incl += n;
      }
      if(c == 0) prefix[0] = 0;
      if(c <= maxcol) prefix[c+1] = incl;
    }
    __syncthreads();

    // Stateless lookup: every rank maps to exactly one (column,row) via
    // the prefix table built above -- unlike the old setpq(), there's no
    // per-thread incremental search state carried between loop iterations
    // at all, so aggregate work no longer scales with BLOCK_SIZE.
    // maxcol can go negative (q0 computed by Min_q() can exceed j-1 for
    // short/tight (i,j) spans) -- maxcol==-1 safely reads prefix[0] (always
    // zeroed above), but maxcol<=-2 would read prefix[maxcol+1] at a
    // negative index, one or more slots *before* this array in shared
    // memory. Shared memory isn't cleared between kernel launches, so that
    // read returns whatever garbage the previous block/launch left on this
    // SM -- harmless-looking on a fresh process, but a real memory-safety
    // bug: it was observed returning garbage total counts in the billions,
    // which then drove the search loop below into billions of iterations
    // and intermittently tripped the row/column bounds asserts. Clamp
    // explicitly rather than relying on prefix[0] happening to be in range.
    const int total = (maxcol >= 0) ? prefix[maxcol+1] : 0;
    for(int work = threadIdx.x; work < total; work += BLOCK_SIZE) {
      int column = 0;
      while(column <= maxcol && prefix[column+1] <= work) column++;
      assert(column <= maxcol);
      int popc_unused;
      const int row = find_nth_set_bit(col_mask[column], work - prefix[column], popc_unused);
      assert(row >= 0); //guaranteed by construction: work-prefix[column] < popcount(col_mask[column])
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
    }//endfor

    // Block-wide min reduction via warp shuffle -- replaces the old
    // volatile __shared__ tree reduction (see the "Cooperative Column
    // Scan" design doc for why this rewrite bundles both changes).
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
      energy = MIN2(energy, __shfl_down_sync(0xffffffff, energy, offset));
    //lane 0 of every warp now holds that warp's minimum.
#if BLOCK_SIZE > 32
    __shared__ int warp_min[BLOCK_SIZE/32];
    if((threadIdx.x & 31) == 0) warp_min[threadIdx.x >> 5] = energy;
    __syncthreads();
    if(threadIdx.x == 0) {
      energy = warp_min[0];
#pragma unroll
      for(int w = 1; w < BLOCK_SIZE/32; w++) energy = MIN2(energy, warp_min[w]);
    }
#endif
  }//endif VRNA_CONSTRAINT_CONTEXT_INT_LOOP
  // Staggered_Row_Batching Phase 2d: table-driven per-H row offset -- must
  // match fill_arrays_loop.c's energy_min[row_off_H[H]+j] convention, since
  // int_loop_cuda() below copies this whole buffer to the host flat.
  if(threadIdx.x==0) energy_min[row_off_H[H]+j] = energy;
}

#undef INT_LOOP_KERNEL_NAME
#undef INT_LOOP_KERNEL_NAME2

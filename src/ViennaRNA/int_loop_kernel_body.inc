// Included multiple times from int_loop.cu, once per candidate BLOCK_SIZE
// (see int_loop_choose_block_size() and stub2.h's rnafold_choose_block_size())
// -- each inclusion #define's BLOCK_SIZE to a different power of two first,
// producing a differently-named, differently-sized instantiation of
// int_loop_kernel so the launch site can pick whichever the actual GPU
// present says is best, instead of the single hardcoded BLOCK_SIZE=32 this
// used to be (tuned against one GPU, the L4).
//
// This has to be a repeated #include rather than a C++ template<int
// BLOCK_SIZE> because the reduction below leans on #if BLOCK_SIZE >= N
// (preprocessor, evaluated once per #include) to fold a variable number of
// tiers -- hand-traced correct for every power-of-two size it supports
// (32/64/128/256/512/1024), including the BLOCK_SIZE==32 shared-memory
// padding trick a few lines down. Re-deriving that as if-constexpr risked
// subtly changing already-correct, warp-synchronous logic for no benefit;
// #include-ing this file once per size instead reuses it completely
// unmodified.
//
// Not listed as a Makefile.am prerequisite for int_loop.o: every build of
// this project is a fresh `git clone` + full `make` on Colab (see project
// memory), so there's no incremental-build staleness this file could
// introduce -- nothing here is ever a partial rebuild.

#define INT_LOOP_KERNEL_NAME2(bs) int_loop_kernel_##bs
#define INT_LOOP_KERNEL_NAME(bs) INT_LOOP_KERNEL_NAME2(bs)

__global__ void
INT_LOOP_KERNEL_NAME(BLOCK_SIZE)(const int nfiles, const int i, /*const int turn,*/ const int length,
		const int TerminalAU, const int ninio2,
		const cuda_param_t* __restrict__ P, const float lxc,
		const char* __restrict__ pair_, //[NBPAIRS+1][NBPAIRS+1],
		const unsigned int* __restrict__ S,    //[length+2] packed
		const unsigned int* __restrict__ hccc,//bit array hc[ij] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP
		const int* __restrict__ my_c,
		      int* __restrict__ energy_min) { //out

  int energy = INF;
  const int H = blockIdx.y;
  const int j = blockIdx.x + i+turn+1;

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
	            &my_c[Hoff(H,length)],
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
  // H tightest index (per Dr. Langdon's Aug 2026 main-branch work, 54b7c31)
  // -- must match fill_arrays_loop.c's energy_min[H+j*nfiles] convention,
  // since int_loop_cuda() below copies this whole buffer to the host flat.
  if(threadIdx.x==0) energy_min[H+j*nfiles] = energy;
}

#undef INT_LOOP_KERNEL_NAME
#undef INT_LOOP_KERNEL_NAME2

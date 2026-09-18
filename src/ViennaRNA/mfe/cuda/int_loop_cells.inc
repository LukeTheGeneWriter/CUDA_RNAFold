/* int_loop_cells.inc -- per-cell bodies lifted out of the kernels in int_loop.cu.
 *
 * The fused per-record megakernel and the standalone kernels must run the
 * SAME arithmetic; device code cannot cross a translation unit in this build
 * (no -rdc), so a header is the only way to share it. The bodies are copied
 * verbatim -- only the thread-index prologue became a parameter.
 */

__device__ inline void
load_my_c_cell(const int nfiles, const int i_row, /*const int turn,*/ const int length,
		 const int* __restrict__ new_e,
	               int* __restrict__ my_c,
		 const size_t* __restrict__ tri_off_H, //in
		 const size_t* __restrict__ row_off_H, //in
		 const size_t* __restrict__ size_off_H, const size_t total, //in
		 const int* __restrict__ i_H,
                const long long m) {
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

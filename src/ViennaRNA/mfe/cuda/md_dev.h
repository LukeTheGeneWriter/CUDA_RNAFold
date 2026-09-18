/* md_dev.h -- device helpers shared with the fused per-record megakernel.
 *
 * Moved verbatim out of modular_decomposition.cu. The megakernel calls the same
 * arithmetic the standalone kernels do, and device code cannot cross a
 * translation unit in this build. -rdc would allow the call but would stop it
 * INLINING, and these run in the innermost loops of an O(n^3) sweep.
 */
#ifndef RNAFOLD_MD_DEV_H
#define RNAFOLD_MD_DEV_H

// Baseline slot for the cell at within-column index `idx` of column `j`, in
// record H. colb_off makes each column start on a baseline boundary; a block
// straddling a column boundary would have to span the whole energy range.
__device__ __forceinline__ size_t
fml_bidx(const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
         const int H, const int j, const int idx) {
  // Unsigned for the same reason as the hot loop in
  // modular_decomposition_kernel -- a signed power-of-two divide costs four
  // instructions to bias a negative numerator that never occurs here, and
  // callers pass a within-column index, which is >= 1. This site is O(n^2) per
  // row rather than O(n^3), so it matters even less than that one did, and
  // that one measured as no change at all.
  assert(idx >= 1);
  return base_off_H[H] + colb_off[j] + (size_t)((unsigned)(idx - 1)/FML_BLK);
}

__device__ __forceinline__ int
fml_decode(const short* __restrict__ j16, const int* __restrict__ b,
           const size_t t, const size_t bidx) {
  const short o = j16[t];
  return (o == FML_INF16) ? INF : (b[bidx] + (int)o);
}

#endif /* RNAFOLD_MD_DEV_H */

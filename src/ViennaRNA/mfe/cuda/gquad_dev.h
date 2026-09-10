/*
 * The device-side c_gq lookup, shared by gquad.cu (G0's read-back bar and the
 * G1 row expansion) and int_loop.cu (G2's interior-loop sweeps).
 *
 * One definition, because the two call sites must agree with upstream and with
 * each other. `static __device__ __forceinline__` so each translation unit gets
 * its own copy with no duplicate-symbol problem.
 *
 * IT IS A LINEAR SCAN, NOT A BINARY SEARCH. vrna_smx_csr_int_get()
 * (sparse_mx.c:66-74) walks the row's columns in order; PORT_GQUAD_SPEC.md's
 * original text called it a binary search and that is wrong. Per-lookup cost is
 * O(entries in row) -- measured 12-45 and roughly constant in n
 * (tools/upstream/gquad_rowstats_probe.c) -- so mirroring the scan exactly is
 * both correct and cheap. Matching upstream's arithmetic beats being clever.
 *
 * `rowoff` MUST already be offsets rather than counts: the CSR carries a lazy
 * `dirty` prefix sum that the host forces before upload. See
 * rnafold_gq_upload() in gquad.c.
 */
#ifndef VRNA_CUDA_GQUAD_DEV_H
#define VRNA_CUDA_GQUAD_DEV_H

#ifndef INF
#define INF 10000000
#endif

/* From params/basic.h. Restated rather than included because the CUDA
 * translation units deliberately avoid pulling RNAlib headers (they have no
 * extern "C" guards; see the note atop gquad.c). Values:
 * MIN = 4*2 + 3*1 = 11, MAX = 4*7 + 3*15 = 73. */
#ifndef VRNA_GQUAD_MIN_BOX_SIZE
#define VRNA_GQUAD_MIN_BOX_SIZE  11
#endif
#ifndef VRNA_GQUAD_MAX_BOX_SIZE
#define VRNA_GQUAD_MAX_BOX_SIZE  73
#endif

static __device__ __forceinline__ int
gq_lookup(const int                        H,
          const unsigned int               i,
          const unsigned int               j,
          const int          *__restrict__ v,
          const unsigned int *__restrict__ col,
          const unsigned int *__restrict__ rowoff,
          const size_t       *__restrict__ ent_off,
          const size_t       *__restrict__ row_off)
{
  const size_t        ro   = row_off[H];
  const size_t        base = ent_off[H];
  const unsigned int  s    = rowoff[ro + i];
  const unsigned int  d    = rowoff[ro + i + 1];
  unsigned int        p;

  for (p = s; p < d; p++)
    if (col[base + p] == j)
      return v[base + p];

  return INF;
}

#endif /* VRNA_CUDA_GQUAD_DEV_H */

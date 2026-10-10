/* N5a backtrack pipelining (PORT_BACKTRACK_PIPELINE.md §3.1): a RETIRED chunk.
 *
 * When a chunk's sweep ends, the device buffers its backtrack reads are taken
 * over from the live globals into one of these, so the next chunk's teardown
 * and init_gpu*() neither free nor reuse them while they are being fetched.
 * Everything else is torn down as before. The pointers are the chunk's own
 * allocations, and rnafold_retired_release() frees them by the pointer.
 *
 * Private to the CUDA backend: nothing here is library API (decision 9.1).
 */
#ifndef RNAFOLD_BT_RETIRED_H
#define RNAFOLD_BT_RETIRED_H

#include <stddef.h>

typedef struct {
  int     id;            /* chunk number; every fetch checks it */
  int     nslots;
  size_t  cells;         /* tri_off_H[nslots]: the retained triangles' extent */
  int    *d_my_c;
  int    *d_fml_j;       /* int32 fML, or NULL */
  short  *d_fml_j16;     /* int16 fML, or NULL -- the layout is read off the POINTER */
  int    *d_fml_b;       /* int16 baselines */
  size_t *d_base_off_H;  /* per-record start in d_fml_b */
  int    *d_fm2;         /* circular only, else NULL */
} rnafold_retired_t;

#ifdef __cplusplus
extern "C" {
#endif

/* device.cu: drain the device, detach the fetch set, and later free it. */
rnafold_retired_t *rnafold_retire_chunk(const int nslots, const size_t *tri_off_H);
void  rnafold_retired_release(rnafold_retired_t *r);

/* The owners' halves of the detach: each NULLs its globals, so its teardown
 * skips them. int_loop.cu, modular_decomposition.cu. */
int  *rnafold_my_c_detach(void);
void  rnafold_fml_detach(rnafold_retired_t *r);

/* One record's triangles from a retired set, on worker w's NON-blocking copy
 * stream. The twins of fetch_my_c_one_w() / fetch_fML_one_Hw() /
 * fetch_fm2_one_w(). */
void  rnafold_retired_fetch_my_c_w(const rnafold_retired_t *r, int *dst,
                                   const size_t tri_lo, const size_t cells, const int w);
void  rnafold_retired_fetch_fml_w(const rnafold_retired_t *r, int *dst,
                                  const size_t tri_lo, const size_t cells,
                                  const int H, const int w);
void  rnafold_retired_fetch_fm2_w(const rnafold_retired_t *r, int *dst,
                                  const size_t tri_lo, const size_t cells, const int w);

#ifdef __cplusplus
}
#endif

#endif

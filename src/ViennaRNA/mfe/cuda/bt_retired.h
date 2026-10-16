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
  int     negctl_freed;  /* RNA_BT_PIPELINE_NEGCTL=2 freed the buffers at hand-off */
  struct rnafold_dbt_s *dbt;  /* N5: the device trace's own copies (bt_device.h), or NULL for
                               * the host trace; released, and charged, with the set */
} rnafold_retired_t;

/* P3 (PORT_BUDGETS.md 9.2): what engine.c lets the coming par_mfe() do with its trace
 * (rnafold_bt_async_allow), and why it joins a pending one (rnafold_bt_pipeline_join). */
#define RNAFOLD_BT_ASYNC_DRIVER   1   /* the last sub-chunk: to a driver that attached (P2) */
#define RNAFOLD_BT_ASYNC_INTERNAL 2   /* any other sub-chunk: joined inside the call (P3) */
#define RNAFOLD_BT_JOIN_END       0   /* end of a call: an internal trace only */
#define RNAFOLD_BT_JOIN_EARLY     1   /* rule (b): it would narrow the next sub-chunk */

#ifdef __cplusplus
extern "C" {
#endif

/* device.cu: drain the device, detach the fetch set, and later free it. */
rnafold_retired_t *rnafold_retire_chunk(const int nslots, const size_t *tri_off_H);
void  rnafold_retired_release(rnafold_retired_t *r);
size_t rnafold_retired_bytes(const rnafold_retired_t *r);  /* device bytes it holds (P3) */
int   rnafold_bt_negctl(void);                           /* RNA_BT_PIPELINE_NEGCTL, 0/1/2 */
void  rnafold_retired_negctl_free(rnafold_retired_t *r); /* NEGCTL=2 only */

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

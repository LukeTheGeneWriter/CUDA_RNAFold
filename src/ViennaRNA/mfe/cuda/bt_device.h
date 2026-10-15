/* N5 device backtrack (PORT_DEVICE_BACKTRACK.md, P2): the MFE trace of a retired chunk, on the device.
 *
 * rnafold_dbt_prepare() builds a set's OWN device copies from VC[] -- sequences, parameter structs, the
 * salt and hairpin-length tables -- and its scratch (D2): the sweep frees its own at the chunk's teardown,
 * and the next chunk's init_gpu3() rewrites d_param2->hp_len, so a trace running beside the next sweep
 * cannot read them. rnafold_dbt_run() computes f5 and traces every record of the retired set, reading c and
 * fML (int32, or int16 decoded in place) where the sweep left them. rnafold_dbt_free() frees by the pointer.
 *
 * The default model only: the caller refuses everything else before calling (P2c). Private to the CUDA
 * backend, like bt_retired.h. No CUDA types in the interface, so the .c side can include it.
 */
#ifndef RNAFOLD_BT_DEVICE_H
#define RNAFOLD_BT_DEVICE_H

#include <stddef.h>
#include "bt_retired.h"

#define RNAFOLD_DBT_ORDER_UPSTREAM 0   /* upstream's search order: the answer */
#define RNAFOLD_DBT_ORDER_REVERSED 1   /* every search reversed: RNA_DEVICE_BT_NEGCTL=1 only */

#ifdef __cplusplus
extern "C" {
#endif

struct vrna_fc_s;
typedef struct rnafold_dbt_s rnafold_dbt_t;

/* One set: VC[0..nslots-1] are the records of the chunk in slot order, tri_off_H[] its nslots+1 triangle
 * offsets (the same table rnafold_retire_chunk() was given). The copies are uploaded before it returns. */
rnafold_dbt_t *rnafold_dbt_prepare(struct vrna_fc_s *const *VC, const int nslots, const size_t *tri_off_H);

/* f5 + trace for every record, on the trace's own low-priority stream; returns when the results are on
 * the host. 0 if every record traced, else the number that failed (their structure is not valid). */
int    rnafold_dbt_run(rnafold_dbt_t *d, const rnafold_retired_t *r, const int order);

const int  *rnafold_dbt_f5(const rnafold_dbt_t *d, const int H);         /* f5[0..n] of record H */
const char *rnafold_dbt_structure(const rnafold_dbt_t *d, const int H);  /* n chars + NUL */
int    rnafold_dbt_status(const rnafold_dbt_t *d, const int H);          /* 0 = traced */

size_t rnafold_dbt_bytes(const rnafold_dbt_t *d);   /* device bytes the set holds (budget, P3 rule b) */
void   rnafold_dbt_free(rnafold_dbt_t *d);

#ifdef __cplusplus
}
#endif

#endif

/*
 * G-quadruplex stage G0, host side: flatten the batch's c_gq matrices.
 *
 * The device side is gquad.cu. This is a separate translation unit for a build
 * reason, not a stylistic one: the ViennaRNA headers carry no `extern "C"`
 * guards and a .cu compiles as C++, so any RNAlib call from there comes out
 * name-mangled and fails at link. The other four .cu files never hit that
 * because none of them calls a single RNAlib function. Keeping the CSR walk in
 * C preserves that property instead of working around it.
 *
 * See PORT_GQUAD_SPEC.md "SCOPED AGAINST THE PORT" for why the table crosses to
 * the device before any recursion work: the two call sites that will read it
 * only ever READ it, so if it arrives intact a later wrong answer is a
 * recursion bug, and if it does not, every later stage debugs the wrong thing.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/params/constants.h"
#include "ViennaRNA/datastructures/array.h"
#include "ViennaRNA/datastructures/sparse_mx.h"
#include "ViennaRNA/datastructures/dp_matrices.h"

/* implemented in gquad.cu */
extern int rnafold_gq_upload_flat(const int nfiles,
                                  const size_t entries,
                                  const size_t rowslots,
                                  const int *h_v,
                                  const unsigned int *h_col,
                                  const unsigned int *h_rowoff,
                                  const size_t *h_ent_off,
                                  const size_t *h_row_off);
extern void rnafold_gq_free(void);


/*
 * Returns 1 if a table was uploaded, 0 if there was nothing to upload (gquad
 * off, or no record carries a matrix), -1 on failure. Safe to call for every
 * batch, including ones with gquad off.
 */
int
rnafold_gq_upload(const int                   nfiles,
                  const vrna_fold_compound_t  **VC)
{
  size_t        entries = 0, rowslots = 0, e_cursor = 0, r_cursor = 0;
  int           H, any = 0, rc;
  int           *h_v = NULL;
  unsigned int  *h_col = NULL, *h_rowoff = NULL;
  size_t        *h_ent_off = NULL, *h_row_off = NULL;

  rnafold_gq_free();

  if ((nfiles <= 0) || (VC == NULL))
    return 0;

  for (H = 0; H < nfiles; H++)
    if ((VC[H]) && (VC[H]->matrices) && (VC[H]->matrices->c_gq))
      any = 1;

  if (!any)
    return 0;

  h_ent_off = (size_t *)vrna_alloc(sizeof(size_t) * (nfiles + 1));
  h_row_off = (size_t *)vrna_alloc(sizeof(size_t) * (nfiles + 1));

  /*
   * Pass 1: sizes. Every record contributes a row-offset span of n+2 even when
   * it holds no entries, so the device indexing never special-cases a record.
   */
  for (H = 0; H < nfiles; H++) {
    const unsigned int n = (VC[H]) ? (unsigned int)VC[H]->length : 0u;

    h_ent_off[H]  = entries;
    h_row_off[H]  = rowslots;

    if ((VC[H]) && (VC[H]->matrices) && (VC[H]->matrices->c_gq)) {
      /*
       * FORCE THE LAZY PREFIX SUM BEFORE READING row_idx.
       *
       * vrna_smx_csr_int_get() carries a `dirty` flag and rewrites row_idx IN
       * PLACE on its first call, turning per-row COUNTS into offsets
       * (sparse_mx.c:61-65). A freshly built c_gq is dirty. Reading row_idx
       * without this yields counts that look exactly like plausible offsets --
       * monotonically small, non-negative, right length -- and every lookup
       * would then read the wrong slice of the wrong row. One get on a cell
       * that cannot exist is enough and costs nothing.
       *
       * That lazy in-place mutation is also a thread-safety hazard of the same
       * family as upstream Defect B: two threads calling get() on one dirty
       * matrix race on the prefix sum. Not live for us -- c_gq is per record
       * and a record belongs to one thread -- but worth sending upstream.
       */
      (void)vrna_smx_csr_int_get(VC[H]->matrices->c_gq, 0u, 0u, INF);
      entries += vrna_array_size(VC[H]->matrices->c_gq->v);
    }

    rowslots += (size_t)n + 2u;
  }

  h_ent_off[nfiles] = entries;
  h_row_off[nfiles] = rowslots;

  h_v      = (int *)vrna_alloc(sizeof(int) * (entries ? entries : 1));
  h_col    = (unsigned int *)vrna_alloc(sizeof(unsigned int) * (entries ? entries : 1));
  h_rowoff = (unsigned int *)vrna_alloc(sizeof(unsigned int) * (rowslots ? rowslots : 1));

  /* Pass 2: copy. */
  for (H = 0; H < nfiles; H++) {
    const unsigned int n = (VC[H]) ? (unsigned int)VC[H]->length : 0u;
    unsigned int       k;

    if ((VC[H]) && (VC[H]->matrices) && (VC[H]->matrices->c_gq)) {
      vrna_smx_csr_int_t  *m  = VC[H]->matrices->c_gq;
      const size_t        ne  = vrna_array_size(m->v);
      const size_t        cap = vrna_array_capacity(m->row_idx);

      for (k = 0; k < (unsigned int)ne; k++) {
        h_v[e_cursor + k]   = m->v[k];
        h_col[e_cursor + k] = m->col_idx[k];
      }

      /*
       * row_idx is indexed 0..n+1. Copy the whole span so the device can read
       * rowoff[i] and rowoff[i+1] for any i in 1..n with no bounds test in the
       * kernel. Past the array's own capacity the correct offset is "all
       * entries consumed", i.e. ne -- which makes those rows empty.
       */
      for (k = 0; k <= n + 1u; k++)
        h_rowoff[r_cursor + k] = (k < cap) ? m->row_idx[k] : (unsigned int)ne;

      e_cursor += ne;
    } else {
      /* No matrix for this record: an all-zero row table, so every lookup in
       * it finds an empty range and returns INF. */
      for (k = 0; k <= n + 1u; k++)
        h_rowoff[r_cursor + k] = 0u;
    }

    r_cursor += (size_t)n + 2u;
  }

  rc = rnafold_gq_upload_flat(nfiles, entries, rowslots,
                              h_v, h_col, h_rowoff, h_ent_off, h_row_off);

  free(h_v);
  free(h_col);
  free(h_rowoff);
  free(h_ent_off);
  free(h_row_off);

  return rc;
}

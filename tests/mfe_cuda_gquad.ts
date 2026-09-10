#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/params/basic.h>
#include <ViennaRNA/params/constants.h>
#include <ViennaRNA/datastructures/array.h>
#include <ViennaRNA/datastructures/sparse_mx.h>
#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/mfe/cuda/engine.h>

/*
 * G-quadruplex stage G0: does c_gq reach the device unchanged?
 *
 * -g DOES NOT WORK YET and this file does not claim it does. The routing guard
 * still declines gquad compounds; the two device call sites that will read the
 * table land in G1/G2. This bar exists on its own because both of those sites
 * only ever READ c_gq -- so if the table arrives intact, a later wrong answer
 * is a recursion bug, and if it does not, every later stage debugs the wrong
 * thing. See PORT_GQUAD_SPEC.md "SCOPED AGAINST THE PORT".
 *
 * The comparison is against vrna_smx_csr_int_get() itself, cell for cell, over
 * the FULL triangle of every record -- not against a remembered dump. Diffing
 * the transport against the thing it is supposed to reproduce is the
 * self-comparison shape this project keeps returning to.
 */

extern int rnafold_gq_upload(const int nfiles, const vrna_fold_compound_t **VC);
extern int rnafold_gq_active(void);
extern void rnafold_gq_free(void);
extern int rnafold_gq_probe(const size_t n, const int *pH, const unsigned int *pi,
                            const unsigned int *pj, int *out);

/*
 * G-runs with short linkers, interleaved with ordinary structured RNA, at
 * MIXED lengths. Every one of these must actually produce entries in c_gq --
 * asserted below, because a fixture with no quadruplexes would make this whole
 * file pass while transporting nothing.
 */
static const char *gq_seqs[] = {
  "GGGGAGGGUUAGGGGCUGGGAAAAUGCAUGCGGGGUGGGACGGGGAUGGGCUAGCUAGCAUCGAUCGGGGAGGGUUAGGGGCUGGG",
  "AUGCAUGCGGGGAGGGCUGGGGAAAGGGAUCGAUCGAUCGGGGUGGGACGGGGAUGGGUUAGCUAGCUAAGGGGAGGGUUAGGGCUGGGAAAUGCAUGC",
  "GGGGUGGGACGGGGAUGGGAAAAGCUAGCUAGCUAGCAUCGAUCGAUCGGGGAGGGUUAGGGGCUGGGCUUAUGCAUGCAUGCGGGGAGGGCUGGGGAAAGGGAUCGAUCG",
  "AUCGAUCGAUCGGGGGAGGGUUAGGGGCUGGGAAAUGCAUGCAUGCAUGCGGGGUGGGACGGGGAUGGGCUAGCUAGCUAGCAUCGGGGAGGGUUAGGGCUGGGAAAAUGCAUGCAUGCAUGC"
};

#define GQ_N (sizeof(gq_seqs) / sizeof(gq_seqs[0]))


static vrna_fold_compound_t *
gq_fc(const char *seq)
{
  vrna_md_t md;
  vrna_fold_compound_t *fc;

  vrna_md_set_default(&md);
  md.gquad = 1;

  /*
   * VRNA_OPTION_MFE, and then prepare().
   *
   * VRNA_OPTION_DEFAULT is literally 0 (fold_compound.h:398), so vrna_mx_add()
   * adds NO matrices and c_gq comes back NULL -- which reads exactly like
   * "gquad is off" rather than like a mistake. The MFE matrices, c_gq
   * included, arrive at PREPARE time (dp_matrices.c:547), which is where
   * par_mfe() gets them too (mfe_cuda.c:1090).
   */
  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);

  return fc;
}


#suite mfe_cuda_gquad

#tcase Transport

#test test_c_gq_reaches_the_device_unchanged
{
  vrna_fold_compound_t  *fc[GQ_N];
  size_t                k, n_probe = 0, cursor = 0;
  int                   *pH = NULL, *expect = NULL, *got = NULL;
  unsigned int          *pi = NULL, *pj = NULL;
  unsigned int          devices;
  unsigned long         nonINF = 0;

  for (k = 0; k < GQ_N; k++) {
    fc[k] = gq_fc(gq_seqs[k]);
    ck_assert(fc[k] != NULL);
    ck_assert(fc[k]->params->model_details.gquad == 1);
    /* The fixture must actually contain quadruplexes, or this test transports
     * nothing and passes. */
    ck_assert(fc[k]->matrices != NULL);
    ck_assert(fc[k]->matrices->c_gq != NULL);
  }

  devices = vrna_cuda_devices();
  if (devices == 0) {
    /* No device: assert the real behaviour of this configuration rather than
     * comparing the host against itself and scoring that a pass. */
    ck_assert(rnafold_gq_upload(GQ_N, (const vrna_fold_compound_t **)fc) <= 0 ||
              rnafold_gq_active() == 0);
    for (k = 0; k < GQ_N; k++)
      vrna_fold_compound_free(fc[k]);
    return;
  }

  ck_assert(rnafold_gq_upload(GQ_N, (const vrna_fold_compound_t **)fc) == 1);
  ck_assert(rnafold_gq_active() == 1);

  /* Every (i,j) of every record. */
  for (k = 0; k < GQ_N; k++) {
    const size_t n = fc[k]->length;

    n_probe += (n * (n + 1)) / 2;
  }

  pH      = (int *)vrna_alloc(sizeof(int) * n_probe);
  pi      = (unsigned int *)vrna_alloc(sizeof(unsigned int) * n_probe);
  pj      = (unsigned int *)vrna_alloc(sizeof(unsigned int) * n_probe);
  expect  = (int *)vrna_alloc(sizeof(int) * n_probe);
  got     = (int *)vrna_alloc(sizeof(int) * n_probe);

  for (k = 0; k < GQ_N; k++) {
    const unsigned int n = (unsigned int)fc[k]->length;
    unsigned int       i, j;

    for (i = 1; i <= n; i++) {
      for (j = i; j <= n; j++) {
        pH[cursor]      = (int)k;
        pi[cursor]      = i;
        pj[cursor]      = j;
        expect[cursor]  = vrna_smx_csr_int_get(fc[k]->matrices->c_gq, i, j, INF);
        if (expect[cursor] != INF)
          nonINF++;
        cursor++;
      }
    }
  }

  ck_assert(cursor == n_probe);

  /* And there must be something to find, or the comparison is INF == INF all
   * the way down -- which a completely broken transport would also pass. */
  ck_assert(nonINF > 0);

  ck_assert(rnafold_gq_probe(n_probe, pH, pi, pj, got) == 0);

  {
    size_t mismatches = 0, first = 0;

    for (k = 0; k < n_probe; k++) {
      if (got[k] != expect[k]) {
        if (mismatches == 0)
          first = k;
        mismatches++;
      }
    }

    if (mismatches) {
      fprintf(stderr,
              "c_gq transport: %zu of %zu cells differ; first at "
              "H=%d (i=%u,j=%u) host=%d device=%d\n",
              mismatches, n_probe, pH[first], pi[first], pj[first],
              expect[first], got[first]);
    }

    ck_assert(mismatches == 0);
  }

  free(pH); free(pi); free(pj); free(expect); free(got);
  rnafold_gq_free();

  for (k = 0; k < GQ_N; k++)
    vrna_fold_compound_free(fc[k]);
}


#test test_upload_is_a_noop_without_gquad
{
  /*
   * The upload must stay silent for an ordinary batch. It is called
   * unconditionally from par_mfe(), so a version that allocated or reported
   * for every fold would be a cost on the default path -- which is the one
   * thing this feature must not touch.
   */
  vrna_fold_compound_t  *fc[2];
  vrna_md_t             md;
  size_t                k;

  vrna_md_set_default(&md);          /* gquad OFF */

  for (k = 0; k < 2; k++) {
    fc[k] = vrna_fold_compound(gq_seqs[k], &md, VRNA_OPTION_MFE);
    ck_assert(fc[k] != NULL);
    (void)vrna_fold_compound_prepare(fc[k], VRNA_OPTION_MFE);
    ck_assert(fc[k]->params->model_details.gquad == 0);
    ck_assert(fc[k]->matrices->c_gq == NULL);
  }

  ck_assert(rnafold_gq_upload(2, (const vrna_fold_compound_t **)fc) == 0);
  ck_assert(rnafold_gq_active() == 0);

  for (k = 0; k < 2; k++)
    vrna_fold_compound_free(fc[k]);
}

#main-pre
    srunner_set_tap(sr, "-");

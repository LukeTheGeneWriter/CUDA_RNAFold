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
 * G-quadruplexes on the batch backend.
 *
 * Two tcases, and they test different things on purpose. `Transport` is G0: does
 * c_gq reach the device unchanged, checked against vrna_smx_csr_int_get() cell
 * for cell. `Recursion` is G3: does a -g fold through the batch backend match
 * upstream folding the same compounds one at a time.
 *
 * -g is ACCEPTED as of 2026-09-10. It was declined for a real reason -- the
 * sweep scored no quadruplex contribution at all and returned a self-consistent
 * structure 15-31 kcal/mol above the true MFE. See PORT_GQUAD_SPEC.md.
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

#tcase Recursion

/*
 * G3: -g through the batch backend, against upstream folding the same
 * compounds one at a time.
 *
 * THIS IS A STRUCTURE COMPARISON, NOT AN ENERGY ONE, AND THAT IS THE POINT.
 * At the end of G2 every ENERGY was already byte-exact against pristine 2.7.2
 * while every STRUCTURE was still wrong: the backtrack rendered a single '+'
 * where a quadruplex belongs, because vrna_backtrack_from_intervals() discards
 * bp.L/bp.l when it downconverts to the legacy vrna_bp_stack_t. An energy-only
 * bar is structurally blind to exactly that failure -- the same lesson
 * tests/mfe_cuda_nolp.ts records for --noLP.
 *
 * So this asserts the structures match AND that they actually contain
 * quadruplex notation, which is what the old rendering could not produce.
 */
#test test_batch_matches_upstream_with_gquad
{
  vrna_fold_compound_t  *ref[GQ_N], *bat[GQ_N];
  char                  *s_ref[GQ_N], *s_bat[GQ_N];
  float                 e_bat[GQ_N];
  size_t                k;
  unsigned int          devices, registered, with_gq = 0;

  for (k = 0; k < GQ_N; k++) {
    size_t n = strlen(gq_seqs[k]);

    ref[k]    = gq_fc(gq_seqs[k]);
    bat[k]    = gq_fc(gq_seqs[k]);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    ck_assert(ref[k]->params->model_details.gquad == 1);
  }

  for (k = 0; k < GQ_N; k++) {
    (void)vrna_mfe(ref[k], s_ref[k]);
    if (strchr(s_ref[k], '+'))
      with_gq++;
  }

  /* Every record must actually fold a quadruplex, or this compares two
   * gquad-free folds and proves nothing about gquads at all. */
  ck_assert(with_gq == GQ_N);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* The guard must now ACCEPT -g. Without this the test silently becomes the
     * no-device case and folds on the CPU twice -- which is exactly how it
     * would look if any of the three gates were reinstated. */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, GQ_N, s_bat, e_bat) != 0);

  for (k = 0; k < GQ_N; k++) {
    /* structure first: it is the assertion that fails when only the RENDERING
     * is broken, which was the state at the end of G2 */
    ck_assert_str_eq(s_ref[k], s_bat[k]);

    /*
     * AND THE BOX IS EXPANDED, NOT MERELY MARKED.
     *
     * ck_assert_str_eq above CANNOT catch a broken renderer, and this was
     * confirmed by red-team rather than assumed: disabling the layout expansion
     * in vrna_db_from_bps() left this test fully GREEN, because s_ref and s_bat
     * are rendered by the same function in the same process, so the damage
     * cancels on both sides. Self-comparison is the right default in this
     * project, and this is precisely its blind spot -- a defect on the SHARED
     * path is invisible to it.
     *
     * So assert an ABSOLUTE property no shared bug can fake: an expanded
     * quadruplex contains CONSECUTIVE '+' (>= VRNA_GQUAD_MIN_STACK_SIZE == 2
     * per G-tract), whereas the legacy vrna_bp_stack_t rendering produced
     * exactly one ISOLATED '+' per quadruplex. Measured on this fixture: 33
     * '+' when correct, 3 when the expansion is disabled.
     */
    {
      const char   *c;
      unsigned int  run = 0, best = 0, total = 0;

      for (c = s_bat[k]; *c; c++) {
        if (*c == '+') {
          total++;
          if (++run > best)
            best = run;
        } else {
          run = 0;
        }
      }

      ck_assert(total >= 8);   /* 4 tracts x >= 2 layers */
      ck_assert(best  >= 2);   /* a lone marker is a run of exactly 1 */
    }
  }

  for (k = 0; k < GQ_N; k++) {
    free(s_ref[k]); free(s_bat[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
  }
}


#main-pre
    srunner_set_tap(sr, "-");

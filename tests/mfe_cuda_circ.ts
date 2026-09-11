#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/eval/structures.h>
#include <ViennaRNA/params/constants.h>
#include <ViennaRNA/mfe/cuda/engine.h>

/*
 * Circular RNA (-c) through the batch backend.
 *
 * WHAT THIS FEATURE'S FAILURE ACTUALLY LOOKED LIKE, and why the test is shaped
 * this way: before 2026-09-11 the GPU returned the LINEAR answer for a circular
 * fold. Not a crash, not a refusal -- a valid structure with a plausible energy
 * that simply ignored the -c. An energy-only comparison against a remembered
 * circular number would catch it, but a comparison against the GPU's own linear
 * output would not, and neither says *why*.
 *
 * So this asserts two things:
 *   1. the batch answer matches upstream folding the same compound alone;
 *   2. the circular answer DIFFERS from the linear one for the same sequence,
 *      i.e. -c actually bit. Without (2) a path that silently dropped -c would
 *      pass (1) on any sequence whose circular and linear folds coincide.
 *
 * The underlying claim -- that DMLi IS fM2_real -- has its own check that needs
 * no oracle at all: RNA_CIRC_VERIFY=1 recomputes
 * min_k(fML[i,k]+fML[k+1,j]) from the fetched fML and diffs it cell by cell.
 * That is what caught the near-INF sentinel leak (device 9999750 where upstream
 * has exactly INF), which mattered because postprocess_circular() tests != INF.
 */

/* Mixed lengths, and long enough to have real multibranch structure -- the
 * circular post-processing is about closing a multiloop across the origin, so a
 * fixture of hairpins would exercise almost none of it. */
static const char *circ_seqs[] = {
  "GGCGCGGCACCGUCCGCGGAACAAACGGAGAAGGCAUCUUCGGAUGCCUUCUCCGUUUGUUCCGCGGACGGUGCCGCGCC",
  "AUGCAUGCAUAUAUGCGCGCAAAUUUGGGCCCAAAUUUGCGCGCAUAUAUGCAUGCAUAUAUGCGCGCAAAUUUGGGCCCAUAUGCGC",
  "GGGAAACCCUUUGGGAAACCCUUUAAAGGGCCCUUUAAAGGGCCCAAAUUUGGGCCCAAAUUUGGGCCCUUUAAAGGGCCAUGCAUGCAUGC",
  "CGCGAUAUCGCGAUAUCGCGAUAUCGCGAUAUAUGCGCGCAUAUAUGCGCGCAUAUCGCGAUAUCGCGAUAUCGCGAUAUGGCCAAUU"
};

#define CIRC_N (sizeof(circ_seqs) / sizeof(circ_seqs[0]))


static vrna_fold_compound_t *
circ_fc(const char *seq, int circ)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  md.circ = circ;

  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);

  return fc;
}


#suite mfe_cuda_circ

#tcase Recursion

#test test_batch_honours_circ
{
  vrna_fold_compound_t  *ref[CIRC_N], *bat[CIRC_N], *lin[CIRC_N];
  char                  *s_ref[CIRC_N], *s_bat[CIRC_N], *s_lin[CIRC_N];
  float                 e_bat[CIRC_N], e_ref[CIRC_N], e_lin[CIRC_N];
  size_t                k, bites = 0;
  unsigned int          devices, registered;

  for (k = 0; k < CIRC_N; k++) {
    size_t n = strlen(circ_seqs[k]);

    ref[k]    = circ_fc(circ_seqs[k], 1);
    bat[k]    = circ_fc(circ_seqs[k], 1);
    lin[k]    = circ_fc(circ_seqs[k], 0);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_lin[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    ck_assert(ref[k]->params->model_details.circ == 1);
    /* fM2_real must exist, or nothing below transports anything */
    ck_assert(ref[k]->matrices->fM2_real != NULL);
  }

  for (k = 0; k < CIRC_N; k++) {
    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
    e_lin[k] = vrna_mfe(lin[k], s_lin[k]);
    if (fabs(e_ref[k] - e_lin[k]) > 0.01)
      bites++;
  }

  /*
   * -c MUST change the answer on every record. A path that silently dropped it
   * would return the linear fold, and on a sequence whose two folds coincide
   * that is indistinguishable from success. This is the assertion that makes
   * the comparison below mean something.
   */
  ck_assert(bites == CIRC_N);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    /* No device: assert this configuration's real behaviour rather than
     * comparing upstream against itself and scoring that a pass. */
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* The guard must now ACCEPT circular. Without this the test silently
     * becomes the no-device case and folds on the CPU twice -- which is
     * exactly how it would look if any of the three gates came back. */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, CIRC_N, s_bat, e_bat) != 0);

  for (k = 0; k < CIRC_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);

    /* and the batch answer is the CIRCULAR one, not the linear one -- the
     * specific failure this feature had */
    ck_assert(fabs(e_bat[k] - e_lin[k]) > 0.01);
  }

  for (k = 0; k < CIRC_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_lin[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    vrna_fold_compound_free(lin[k]);
  }
}

#main-pre
    srunner_set_tap(sr, "-");

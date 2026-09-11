#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/params/constants.h>
#include <ViennaRNA/mfe/cuda/engine.h>

/*
 * --noClosingGU (md.noGUclosure) through the batch backend.
 *
 * WHAT THIS OPTION'S FAILURE LOOKED LIKE, and why the test is shaped this way:
 * --noClosingGU was HALF implemented for the life of the fork. The mask half
 * worked -- rnafold_hc_opt() drops HP_LOOP and MB_LOOP for a GU/UG pair exactly
 * as constraints/hard.c:786-791 does, and new_c_kernel skips both terms on gate
 * bit 1 -- while the interior-loop half did nothing at all. A half-applied rule
 * is worse than an unapplied one: c was internally INCONSISTENT, so the answer
 * was not merely suboptimal under some other model, it was not the minimum of
 * any model. Three gates kept it unreachable until 2026-09-11.
 *
 * So this asserts two things:
 *   1. the batch answer matches upstream folding the same compound alone;
 *   2. the --noClosingGU answer DIFFERS from the default one on enough records
 *      that (1) is actually testing something. Without (2) a path that silently
 *      dropped the flag would pass (1) on any sequence with no GU-closed loop
 *      in either fold -- which is most sequences, and is exactly why a
 *      GU-rich fixture is used below.
 *
 * The two halves have separate checks OUTSIDE this file, because neither is
 * visible here:
 *   - the mask half: RNA_HC_VERIFY=1 rebuilds the device masks the host way
 *     and compares bit for bit, needing no oracle at all;
 *   - the interior half: red-teamed by suppressing the skip in Energy()
 *     (int_loop.cu), which makes this test fail.
 */

/*
 * SELECTED BY MEASUREMENT, not written by hand. --noClosingGU can only bite
 * where a GU/UG pair would otherwise close a hairpin, a multiloop, a bulge or an
 * interior loop, so a fixture can easily pass whether or not the feature works.
 * A first draft of hand-written GU-heavy sequences was tried and one of six did
 * NOT bite -- the ck_assert below caught it, which is the point of having it.
 *
 * These six are the short records of tests/noclosinggu/noclosinggu_test.fa
 * (45-180 nt) for which pristine 2.7.2's fold is KNOWN to differ with and
 * without the flag. Note the 150 nt one is a plain uniform-ACGU draw: G/U
 * weighting helps, but it is the measurement that qualifies a record, not the
 * composition.
 */
static const char *ngu_seqs[] = {
  "CUGUGGGAUUUUGCUAGUUGCAGAUGCAUUCUUGUGGUGCUGUGG",
  "ACCCAGACUCUCAGGCCUGGCUGAUAGCCUAGUUGGCACGGACUGACGACUAGACUAAGC",
  "UGUUUUGGGAUGGGUUGGUUGCGUGUAUUGUUUUAGUUGGGUAGUAGGGGGAUCGGCCUGGAUGUGUAGUUCUCG",
  "GAGUUUUUUGGGUGAGUUGGUUAGAUCUGUGGGGUCUAUGUUCUGAUCUGCUUCAGGUUUGUUUUGAGUGCCUGUUGGGCGUUUAUAUUGGGUGUUUGUGUGUUAUGGUUUUAUGGGGGG",
  "AGCUACUCAAACGCACCUGUGUGGUUUACCGAGCUAACGUUCGCUAGCAAAGUGUGUGGCCCGCAUCGUGCUUCUAACUCGGGAGAAUUCUGGUGCAACGCUGCAGUCACGCAGGACGUCAAUCUGACUGAUUCACGAGUGGAUCGGCUU",
  "GGUUUGUUAAUUAGCUUUUGGUGGUUAUUGCGAACGGUGUUAGCGCGGGUUGUGUUUUUGUUUUGUUUGGGUGUUUGGGGCACUUCUGCGCGGUGUCGCAUGGUGUUGGUUGGUGUGUUGUGGUCUCUGUAGUGCGUGGAGGAGUUGUGUUGUUUUGCUUAAUAGUGUCUGUGUUGUAGG"
};

#define NGU_N (sizeof(ngu_seqs) / sizeof(ngu_seqs[0]))


static vrna_fold_compound_t *
ngu_fc(const char *seq, int noGUclosure)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  md.noGUclosure = noGUclosure;

  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);

  return fc;
}


#suite mfe_cuda_noclosinggu

#tcase Recursion

#test test_batch_honours_noclosinggu
{
  vrna_fold_compound_t  *ref[NGU_N], *bat[NGU_N], *def[NGU_N];
  char                  *s_ref[NGU_N], *s_bat[NGU_N], *s_def[NGU_N];
  float                 e_bat[NGU_N], e_ref[NGU_N], e_def[NGU_N];
  size_t                k, bites = 0;
  unsigned int          devices, registered;

  for (k = 0; k < NGU_N; k++) {
    size_t n = strlen(ngu_seqs[k]);

    ref[k]    = ngu_fc(ngu_seqs[k], 1);
    bat[k]    = ngu_fc(ngu_seqs[k], 1);
    def[k]    = ngu_fc(ngu_seqs[k], 0);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_def[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    ck_assert(ref[k]->params->model_details.noGUclosure == 1);
    ck_assert(def[k]->params->model_details.noGUclosure == 0);
  }

  for (k = 0; k < NGU_N; k++) {
    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
    e_def[k] = vrna_mfe(def[k], s_def[k]);
    if (strcmp(s_ref[k], s_def[k]) != 0)
      bites++;
  }

  /*
   * The flag must change the answer on every record. A path that silently
   * dropped it returns the default fold, and on a sequence where the two
   * coincide that is indistinguishable from success. This is the assertion
   * that makes the comparison below mean something -- and the reason the
   * fixture above is G/U-weighted rather than a random draw.
   */
  ck_assert(bites == NGU_N);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    /* No device: assert this configuration's real behaviour rather than
     * comparing upstream against itself and scoring that a pass. */
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* The guard must now ACCEPT noGUclosure. Without this the test silently
     * becomes the no-device case and folds on the CPU twice -- which is
     * exactly how it would look if any of the three gates came back. */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, NGU_N, s_bat, e_bat) != 0);

  for (k = 0; k < NGU_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);

    /* and the batch answer honoured the flag rather than returning the
     * default fold -- the specific failure this option had */
    ck_assert_str_ne(s_bat[k], s_def[k]);
  }

  for (k = 0; k < NGU_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_def[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    vrna_fold_compound_free(def[k]);
  }
}

#main-pre
    srunner_set_tap(sr, "-");

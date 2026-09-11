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
 * Dangle models on the batch backend: 0 and 2 ACCELERATED, 1 and 3 DECLINED.
 *
 * WHY 0 IS CHEAP AND 1/3 ARE NOT, which is the thing this test exists to pin
 * down so nobody re-scopes it from scratch. d0 and d2 share the RECURSION --
 * mfe_multibranch.c:686 dispatches ml_pair_d0 or ml_pair_d2 and BOTH read only
 * dmli1 -- so no new DP state was needed.
 *
 * AND ONLY ONE ENERGY TERM ACTUALLY DIFFERED ON THE DEVICE. Upstream zeroes
 * P->mismatchM when dangles == 0 (params.c:644-646), so E_MLstem() already
 * returns a bare stem at d0 and both multibranch sites were correct before a
 * line was written -- measured, 0 nonzero mismatchM entries at d0 against 175
 * at d2, and red-teaming both sites changes nothing. The term that genuinely
 * differed is vrna_mfe_gquad_internal_loop()'s mismatchI (mfe_gquad.c:306,
 * `if (dangles)`), which is NOT zeroed: 158 nonzero entries at both models.
 * So `-d0` alone was already right and merely refused; `-d0 -g` was the one
 * combination that would have answered wrongly.
 *
 * Nothing else in the single-sequence MFE recursion is dangle-dependent --
 * vrna_E_internal() has no dangle branch at all, and neither does the hairpin
 * energy.
 *
 * ml_pair_d1() reads dmli2 as WELL as dmli1. That is a second DMLi generation
 * the sweep does not carry, so d1 (and d3, which adds coaxial stacking on top)
 * would need new DP state rather than new arithmetic. Hence: declined, and
 * asserted below so a future widening of the guard has to face this test.
 *
 * The exterior loop and fM1 needed nothing at all: vrna_mfe_exterior_f5() and
 * vrna_mfe_multibranch_m1() are upstream's own and run on the host, so they are
 * dangle-correct for every model by construction.
 */

/*
 * Multibranch-rich by construction -- the dangle model only bites where a stem
 * closes or joins a multiloop, so a fixture of hairpins would exercise almost
 * none of it.
 *
 * MEASURED, after a first draft asserted it and this test caught the lie. On
 * the frozen tests/noclosinggu/ set, pristine 2.7.2 under -d0 against -d2
 * changes the ENERGY of 20 of 20 records but the STRUCTURE of only 17 -- three
 * folds are unchanged while costing a different amount. The six below are the
 * short records whose structure is among the 17; one earlier pick (a 60 nt
 * record) was one of the three and had to go.
 *
 * Which is why the bite test below is on ENERGY, with structure as the
 * secondary: energy is the strictly more sensitive detector of "the model was
 * silently ignored", and picking the weaker one is how a fixture ends up
 * passing whether or not the feature works.
 */
static const char *dang_seqs[] = {
  "CUGUGGGAUUUUGCUAGUUGCAGAUGCAUUCUUGUGGUGCUGUGG",
  "GUUUUGCUUCCGCUAGGGGGAAUCGAACCCUUUUAAACCGAAACAACCGGGAUGUCGCCCGUACGUAUACUUCGUCAACGCUAUUGAUGG",
  "UGUUUUGGGAUGGGUUGGUUGCGUGUAUUGUUUUAGUUGGGUAGUAGGGGGAUCGGCCUGGAUGUGUAGUUCUCG",
  "GAGUUUUUUGGGUGAGUUGGUUAGAUCUGUGGGGUCUAUGUUCUGAUCUGCUUCAGGUUUGUUUUGAGUGCCUGUUGGGCGUUUAUAUUGGGUGUUUGUGUGUUAUGGUUUUAUGGGGGG",
  "AGCUACUCAAACGCACCUGUGUGGUUUACCGAGCUAACGUUCGCUAGCAAAGUGUGUGGCCCGCAUCGUGCUUCUAACUCGGGAGAAUUCUGGUGCAACGCUGCAGUCACGCAGGACGUCAAUCUGACUGAUUCACGAGUGGAUCGGCUU",
  "GGUUUGUUAAUUAGCUUUUGGUGGUUAUUGCGAACGGUGUUAGCGCGGGUUGUGUUUUUGUUUUGUUUGGGUGUUUGGGGCACUUCUGCGCGGUGUCGCAUGGUGUUGGUUGGUGUGUUGUGGUCUCUGUAGUGCGUGGAGGAGUUGUGUUGUUUUGCUUAAUAGUGUCUGUGUUGUAGG"
};

#define DANG_N (sizeof(dang_seqs) / sizeof(dang_seqs[0]))


static vrna_fold_compound_t *
dang_fc(const char *seq, int dangles)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  md.dangles = dangles;

  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);

  return fc;
}


#suite mfe_cuda_dangles

#tcase Recursion

#test test_batch_honours_dangles0
{
  vrna_fold_compound_t  *ref[DANG_N], *bat[DANG_N], *def[DANG_N];
  char                  *s_ref[DANG_N], *s_bat[DANG_N], *s_def[DANG_N];
  float                 e_bat[DANG_N], e_ref[DANG_N], e_def[DANG_N];
  size_t                k, bites_e = 0, bites_s = 0;
  unsigned int          devices, registered;

  for (k = 0; k < DANG_N; k++) {
    size_t n = strlen(dang_seqs[k]);

    ref[k]    = dang_fc(dang_seqs[k], 0);
    bat[k]    = dang_fc(dang_seqs[k], 0);
    def[k]    = dang_fc(dang_seqs[k], 2);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_def[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    ck_assert(ref[k]->params->model_details.dangles == 0);
    ck_assert(def[k]->params->model_details.dangles == 2);
  }

  for (k = 0; k < DANG_N; k++) {
    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
    e_def[k] = vrna_mfe(def[k], s_def[k]);
    if (fabs(e_ref[k] - e_def[k]) > 0.01)
      bites_e++;
    if (strcmp(s_ref[k], s_def[k]) != 0)
      bites_s++;
  }

  /*
   * d0 must change the ENERGY of every record, and the structure of all six of
   * these. A path that silently applied d2 anyway returns the default fold, and
   * on a record where the two coincide that is indistinguishable from success --
   * the specific failure this feature could have, since d2 was the only model
   * for the life of the fork and every call site was written for it.
   *
   * Energy is the primary assertion because it is the more sensitive one: d0
   * changes what a fold COSTS far more often than it changes what the fold IS.
   */
  ck_assert(bites_e == DANG_N);
  ck_assert(bites_s == DANG_N);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* the guard must now ACCEPT d0 */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, DANG_N, s_bat, e_bat) != 0);

  for (k = 0; k < DANG_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);

    /* and it is the d0 answer, not the d2 one -- on both axes */
    ck_assert(fabs(e_bat[k] - e_def[k]) > 0.01);
    ck_assert_str_ne(s_bat[k], s_def[k]);
  }

  for (k = 0; k < DANG_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_def[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    vrna_fold_compound_free(def[k]);
  }
}

#test test_guard_still_declines_dangles_1_and_3
{
  /*
   * NOT a formality. d1 and d3 need a DMLi generation the sweep does not carry,
   * so accepting them would not be "slightly wrong" -- it would silently return
   * the d0/d2 answer under a different model. If a future change widens the
   * guard, this test is what makes that a decision rather than an accident.
   */
  const int models[] = { 1, 3 };
  size_t    i;

  if (vrna_cuda_devices() == 0)
    return;

  ck_assert(vrna_cuda_register_batch_backend() != 0);

  for (i = 0; i < sizeof(models) / sizeof(models[0]); i++) {
    vrna_fold_compound_t  *fc     = dang_fc(dang_seqs[0], models[i]);
    const char            *reason = NULL;

    ck_assert(fc->params->model_details.dangles == models[i]);
    ck_assert(vrna_cuda_engine_supports(fc, &reason) == 0);
    ck_assert(reason != NULL);

    vrna_fold_compound_free(fc);
  }
}

#main-pre
    srunner_set_tap(sr, "-");

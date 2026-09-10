#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/utils/structures.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/eval/structures.h>
#include <ViennaRNA/params/basic.h>
#include <ViennaRNA/mfe/cuda/engine.h>

/*
 * --nsp (non-standard base pairs) through the batch backend.
 *
 * This replaces tests/mfe_cuda_guard.ts::test_guard_declines_nonstandard_pairs,
 * which asserted the opposite. The guard was lifted on 2026-09-10 because the
 * CAUSE was fixed: int_loop.cu's Energy() used to take the raw pair value for
 * `type` (no 0 -> 7 promotion) and obtain `type_2` by SWAPPING the index order
 * instead of applying rtype[]. Both are exact identities at default settings,
 * which is why the port was byte-identical for a year with them; --nsp breaks
 * the second, because model.c:1104 FORCES rtype[7] = 7 after the derivation
 * loop. See PORT_NSP_PARAMFILE_SCOPE.md 1.
 *
 * THE SPEC MUST BE ASYMMETRIC, AND THAT IS THE WHOLE POINT OF THIS FILE.
 *
 * --nsp="-GA" (leading '-') sets BOTH directions of the pair to 7, which makes
 * rtype[pair[p][q]] == pair[q][p] hold again and MASKS the divergence
 * completely. Measured on the broken binary: --nsp="-GA" gave 0 differing
 * lines while --nsp=GA gave 20 and --nsp=AC gave 22. A symmetric-only
 * regression test is a FALSE GREEN -- it passes against code with the bug.
 *
 * So the asymmetric case is asserted first and separately, and a symmetric case
 * is kept only to show it is not accidentally broken.
 */

/*
 * THESE SEQUENCES ARE NOT ARBITRARY, AND THE FIRST SET WAS A FALSE GREEN.
 *
 * The first version of this test used five hand-picked 80-92 nt sequences. It
 * PASSED against the broken Energy() -- all three cases -- because `--nsp`
 * changing the answer and the GPU/CPU divergence being EXERCISED are two
 * different things, and the fixture only established the first. `bites > 0`
 * compares nsp against plain on the CPU; it says nothing about whether the
 * device ever reaches an interior loop whose type_2 row differs.
 *
 * These four were selected by folding 30 random sequences of 45-1200 nt with
 * --nsp=GA on a DELIBERATELY BROKEN build (Energy() fix reverted) and keeping
 * the shortest that disagreed GPU vs CPU. 26 of the 30 disagreed; the four
 * kept are the smallest, so the test stays quick while still failing against
 * the bug. Verified RED before this green was believed.
 *
 * Lengths are mixed on purpose: the --noLP + RNA_SLOT_FLOW defect was invisible
 * to uniform-length fixtures (PORT_NOLP_SPEC.md).
 */
static const char *nsp_seqs[] = {
  /* 88 nt */
  "CCAUCUGAGCAUACAGGGGACAGCUCGAUGCGGAAGAUAUGGCCACAGUCCUUCGAGAUACGGACAUCCGCCCUGCAAUCAACGCACA",
  /* 97 nt */
  "AGACUAAUUGAUCGGGAUUGGCUCAUAGUCACAUUGGCAACUCUAUGUUCUCGGCGGACUAACUAACCUGACUUAUCUUAGAAGCCUCGUGACGGAG",
  /* 150 nt */
  "CUAUUACUAUGGCAAUAAGGCCCCGCGGCAUGAGCUUGCAGACGACACUCGAUGGAUUAAAAUACAAGGUGAGUUAGACGAAUUCCUCUCCAGAAAGUAGCCUUGGACAUCAACGCGUGUGCCCUUGGCCUUCCAACGCUUCUCGGGUGA",
  /* 166 nt */
  "GAGAACCGAGGCAAGCGGAUGUGUCUUAUCGCGUGAUUCGCAAUUUGGCCGGCGGAUCACUCACGGUUUCGGCGGUUCACCACCGCAUGGUCGCGCUGUGUUUGCACUUUGAUCGUGAACGCGACCUGUGGACGUUUCACUAGUCCGUGUCGUCGUCGGGGAGGUG"
};

#define NSP_N (sizeof(nsp_seqs) / sizeof(nsp_seqs[0]))


static vrna_fold_compound_t *
nsp_fc(const char *seq, const char *spec)
{
  vrna_md_t md;

  vrna_md_set_default(&md);
  if (spec)
    vrna_md_set_nonstandards(&md, spec);

  return vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
}


/* Count the 7s the guard used to look for. If a spec does not put any in the
 * pair table then it is not enabling non-standard pairs at all, and every
 * comparison below would be a fold compared against an identical fold. */
static unsigned int
sevens(const vrna_md_t *md)
{
  unsigned int i, j, n = 0;

  for (i = 0; i <= MAXALPHA; i++)
    for (j = 0; j <= MAXALPHA; j++)
      if (md->pair[i][j] == 7)
        n++;

  return n;
}


/* The shared body: fold every sequence one at a time entirely by upstream, then
 * the same compounds as a batch through the device, and require both the
 * structure and the energy to match. */
static void
compare_batch_against_upstream(const char *spec, int require_asymmetric)
{
  vrna_fold_compound_t  *ref[NSP_N], *bat[NSP_N], *plain[NSP_N];
  char                  *s_ref[NSP_N], *s_bat[NSP_N], *s_plain[NSP_N];
  float                 e_bat[NSP_N];
  size_t                k;
  unsigned int          devices, registered, bites = 0;

  for (k = 0; k < NSP_N; k++) {
    size_t n = strlen(nsp_seqs[k]);

    ref[k]      = nsp_fc(nsp_seqs[k], spec);
    bat[k]      = nsp_fc(nsp_seqs[k], spec);
    plain[k]    = nsp_fc(nsp_seqs[k], NULL);
    s_ref[k]    = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]    = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_plain[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    /* the spec must have reached the pair table, or nothing here is a test */
    ck_assert(sevens(&(ref[k]->params->model_details)) > 0);
  }

  /*
   * An ASYMMETRIC spec must leave the pair table asymmetric. This is the
   * property the whole bug depends on, so assert it rather than trusting the
   * spec string: if vrna_md_set_nonstandards() ever symmetrised "GA", this test
   * would keep passing while testing nothing.
   */
  if (require_asymmetric) {
    const vrna_md_t *md     = &(ref[0]->params->model_details);
    unsigned int    i, j, asym = 0;

    for (i = 0; i <= MAXALPHA; i++)
      for (j = 0; j <= MAXALPHA; j++)
        if ((md->pair[i][j] == 7) != (md->pair[j][i] == 7))
          asym++;

    ck_assert(asym > 0);
  }

  for (k = 0; k < NSP_N; k++) {
    (void)vrna_mfe(ref[k], s_ref[k]);
    (void)vrna_mfe(plain[k], s_plain[k]);
    if (strcmp(s_ref[k], s_plain[k]) != 0)
      bites++;
  }

  /* The option must CHANGE THE ANSWER on this input. Two identical folds
   * compared to each other is the failure shape that let --helical-rise sit in
   * PORT_OPTION_STATUS.md for a week as "identical, but did not bite". */
  ck_assert(bites > 0);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    /* No device: assert the real behaviour of this configuration rather than
     * comparing upstream against itself and calling that a pass. */
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* The guard must now ACCEPT --nsp. Without this the test silently
     * degenerates into the no-device case and folds on the CPU twice --
     * which is exactly how it would look if the guard were reinstated. */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, NSP_N, s_bat, e_bat) != 0);

  for (k = 0; k < NSP_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);

    /* and the structure is worth the energy reported beside it */
    {
      float actual = vrna_eval_structure(bat[k], s_bat[k]);
      ck_assert(fabs(e_bat[k] - actual) < 0.01);
    }
  }

  for (k = 0; k < NSP_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_plain[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    vrna_fold_compound_free(plain[k]);
  }

  /* vrna_md_set_nonstandards() also writes the deprecated process-wide
   * `nonstandards` global (model.c:322-326). Clear it, or a later test in this
   * same process inherits it. */
  {
    vrna_md_t md;
    vrna_md_set_default(&md);
    vrna_md_set_nonstandards(&md, NULL);
  }
}


#suite mfe_cuda_nsp

#tcase Asymmetric

/* THE test. This is the case that fails against the pre-2026-09-10 Energy(). */
#test test_batch_matches_upstream_asymmetric_GA
{
  compare_batch_against_upstream("GA", 1);
}

#test test_batch_matches_upstream_asymmetric_AC
{
  compare_batch_against_upstream("AC", 1);
}

#tcase Symmetric

/*
 * Kept, but NOT relied on: a symmetric spec restores the identity the bug
 * depends on, so this passed even on the broken binary. It is here to catch a
 * regression that breaks the symmetric case too, not to guard --nsp.
 */
#test test_batch_matches_upstream_symmetric
{
  compare_batch_against_upstream("-GA", 0);
}

#main-pre
    srunner_set_tap(sr, "-");

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
 * --noLP through the batch backend.
 *
 * WHY THIS TEST IS NOT AN ENERGY COMPARISON.
 *
 * Before this feature was implemented, the sweep filtered ptype for noLP but
 * never applied the RECURSION constraint (mfe/mfe.c:4413 stores
 * cc1[j-1]+stackEnergy into c[ij], not new_c). The result was that the
 * REPORTED energy agreed with upstream on 45 of 60 records, while the
 * STRUCTURES returned were inconsistent with their own reported energy by 87
 * to 300 kcal/mol -- the matrix fill and the backtrack disagreeing with each
 * other. An energy comparison is structurally blind to that: the number it
 * checks is the one that happens to be nearly right.
 *
 * So this asserts three things an energy check cannot:
 *   1. the structure re-evaluates to the energy reported beside it;
 *   2. the structure contains ZERO lonely pairs, which is what the option means;
 *   3. the structure and energy match a compound upstream folded on its own.
 *
 * See PORT_NOLP_SPEC.md.
 */

/* These are NOT arbitrary. The first version of this test used five 66-70 nt
 * sequences chosen by eye, and test_noLP_actually_changes_the_answer FAILED on
 * `plain_lonely > 0`: their PLAIN folds contained no lonely pairs at all, so
 * --noLP had nothing to remove and the discriminator asserted nothing. These
 * were selected by folding 300 random 120 nt sequences both ways and keeping
 * the ones with >= 2 lonely pairs plain and 0 with --noLP (30 of 300 qualify),
 * so the option demonstrably bites on every one. */
static const char *nolp_seqs[] = {
  "UGAGAUUGCUGCCGCCCAGUUGCAUGUGUCAGUCUGGUAACCCCCCAUUUCUGGACAUCGAUUACUUCGGGGACGAGAAAAUUGGAUCUCCGGCUAGUAAGAUGGAGUACAAGGUACUAG",  /* 4 lonely pairs unconstrained */
  "UUGUUCUCGCACAUGACUCGUUCGGUACUGAAUAGUGGCGUUGUACACCCUUCGAAUACUCGAGCAAUCCCCGGAACCACGUUGCCAGAAGGAAAUUCAUGUGUUGUGCCUCAAGACCUA",  /* 3 */
  "UAUGCCGAGACAAUGGAUCUUUACGAGCAGACUAGUCGCCGCCAACAGUAAAUACCACAGAUGGAGCAUGAGCAAUGGAAGAUUUCAACUAUAACGUAAUUCGAAGGGGCCUCAUAGCUU",  /* 3 */
  "UAUACUGAUGAUCAGUCUGCACGUCGGGAUCGUCGGGUCAAUCCCCACGCCGCCGCAAGAAAUGACAGCACAAGUCUGGAGAGAACACGAGAACCAGGCACAUGACCAGUUGGGAAAAGA",  /* 3 */
  "GCUUUGUCCGGGGGCCCUGAGGGACACGAGAGCGGCAAUUACCACAACCGAAUUAACGUAAUUCAGCGAGGUUAAUGGCCGACACGACGGCUGUUCCGGUGCUUAGCGACACAGAGGAAA"   /* 3 */
};

#define NOLP_N (sizeof(nolp_seqs) / sizeof(nolp_seqs[0]))

static vrna_fold_compound_t *
nolp_fc(const char *seq, int noLP)
{
  vrna_md_t md;

  vrna_md_set_default(&md);
  md.noLP = noLP;

  return vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
}

/* A lonely pair: paired, but neither (i-1,j+1) nor (i+1,j-1) is also paired.
 * Counting them needs no reference implementation -- it reads the definition
 * of the option straight off the structure. */
static size_t
count_lonely(const char *structure)
{
  short         *pt = vrna_ptable(structure);
  unsigned int  n   = (unsigned int)pt[0];
  unsigned int  i;
  size_t        lonely = 0;

  for (i = 1; i <= n; i++) {
    unsigned int j = (unsigned int)pt[i];
    int up, dn;

    if ((j == 0) || (j < i))
      continue;                 /* unpaired, or the closing half of a pair */

    up = (i > 1) && (j < n) && (pt[i - 1] == (short)(j + 1));
    dn = (i + 1 < j) && (pt[i + 1] == (short)(j - 1));

    if (!up && !dn)
      lonely++;
  }

  free(pt);
  return lonely;
}

#suite  MFE_CUDA_noLP

#tcase  Recursion

#test test_batch_honours_noLP
{
  vrna_fold_compound_t  *ref[NOLP_N], *bat[NOLP_N];
  char                  *s_ref[NOLP_N], *s_bat[NOLP_N];
  float                 e_bat[NOLP_N];
  size_t                k;
  unsigned int          devices, registered;

  for (k = 0; k < NOLP_N; k++) {
    size_t n = strlen(nolp_seqs[k]);

    ref[k]    = nolp_fc(nolp_seqs[k], 1);
    bat[k]    = nolp_fc(nolp_seqs[k], 1);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    /* noLP must actually be set, or this test asserts nothing about it */
    ck_assert(ref[k]->params->model_details.noLP == 1);
  }

  /* the reference: folded one at a time, entirely by upstream */
  for (k = 0; k < NOLP_N; k++)
    (void)vrna_mfe(ref[k], s_ref[k]);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    /* No device: assert the real behaviour of this configuration rather than
     * comparing upstream against itself and calling that a pass. */
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* the guard must now ACCEPT noLP, or this silently becomes the
     * no-device case above and tests the CPU path twice */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, NOLP_N, s_bat, e_bat) != 0);

  for (k = 0; k < NOLP_N; k++) {
    /* (3) same answer as upstream, structure AND energy */
    ck_assert_str_eq(s_ref[k], s_bat[k]);

    /* (1) the structure returned is worth the energy reported beside it. This
     * is the check that catches a matrix/backtrack disagreement, and it needs
     * no reference at all. */
    {
      float reported = e_bat[k];
      float actual   = vrna_eval_structure(bat[k], s_bat[k]);
      ck_assert(fabs(reported - actual) < 0.01);
    }

    /* (2) and it contains no lonely pairs, which is what --noLP means */
    ck_assert(count_lonely(s_bat[k]) == 0);
  }

  for (k = 0; k < NOLP_N; k++) {
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    free(s_ref[k]);
    free(s_bat[k]);
  }
}

#test test_noLP_actually_changes_the_answer
{
  /*
   * A --noLP run that agrees with the unconstrained fold has not applied the
   * option, and every assertion above would still pass. So assert the option
   * BITES on this sequence set: the plain fold must contain lonely pairs that
   * the noLP fold does not.
   *
   * This is the discriminator the old CLI bar lacked -- it compared --noLP
   * output against a reference without ever establishing that --noLP changed
   * anything, which is how a sweep that ignored the flag stayed green.
   */
  size_t k, plain_lonely = 0, nolp_lonely = 0;

  for (k = 0; k < NOLP_N; k++) {
    size_t                n   = strlen(nolp_seqs[k]);
    vrna_fold_compound_t  *p  = nolp_fc(nolp_seqs[k], 0);
    vrna_fold_compound_t  *q  = nolp_fc(nolp_seqs[k], 1);
    char                  *sp = (char *)vrna_alloc(sizeof(char) * (n + 1));
    char                  *sq = (char *)vrna_alloc(sizeof(char) * (n + 1));

    (void)vrna_mfe(p, sp);
    (void)vrna_mfe(q, sq);

    plain_lonely  += count_lonely(sp);
    nolp_lonely   += count_lonely(sq);

    free(sp); free(sq);
    vrna_fold_compound_free(p);
    vrna_fold_compound_free(q);
  }

  /* the option must have something to remove on this set ... */
  ck_assert(plain_lonely > 0);
  /* ... and must remove all of it */
  ck_assert(nolp_lonely == 0);
}

#main-pre
    srunner_set_tap(sr, "-");

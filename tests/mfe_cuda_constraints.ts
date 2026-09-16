#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/constraints/hard.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/params/constants.h>
#include <ViennaRNA/mfe/cuda/engine.h>

/*
 * HARD STRUCTURE CONSTRAINTS (-C) on the batch backend.
 *
 * A hard constraint is TWO statements, and the port carried one of them for a
 * year:
 *
 *   hc->mx      may this PAIR form?
 *   hc->up_hp / up_int / up_ml / up_ext
 *               may this BASE be left unpaired, in each loop context?
 *
 * The device packed hc->mx all along and carried only up_ml. That is invisible
 * for a constraint that forbids pairing -- 'x' clears bits in the matrix, and
 * the matrix is what the sweep reads -- and silently wrong for one that FORCES
 * pairing: nothing in the four masks moves, and the fill goes on admitting
 * hairpins and interior loops whose unpaired span covers the base that must
 * pair. Measured before the fix, on six records of 80-320 nt with '|' blocks
 * under --enforceConstraint: the device answered BETTER THAN LEGAL on all six,
 * by 1.3 to 17.4 kcal/mol, and its structures re-evaluated to a different
 * energy than it reported.
 *
 * Three things landed together (2026-09-16):
 *   1. the masks are packed AFTER vrna_fold_compound_prepare() materialises
 *      hc->depot into hc->mx -- they were packed five lines before it, which
 *      also made RNA_HC_VERIFY compare two UNCONSTRAINED matrices and agree;
 *   2. up_hp reaches new_c_kernel, gating the hairpin term on
 *      up_hp[i+1] >= j-i-1 (wrap_hairpin_hc.inc:42-52). NOT fill_arrays_loop.c,
 *      which looks like the combine site and is dead code under the default
 *      GPU-resident sweep;
 *   3. up_int reaches Energy(), refusing a candidate whose two unpaired runs do
 *      not fit (wrap_internal_hc.inc:57-67).
 *
 * up_ext needs nothing: the exterior loop is vrna_mfe_exterior_f5(), which is
 * upstream's own host code.
 *
 * THE TWO TESTS BELOW ARE DELIBERATELY DIFFERENT SHAPES. The forbid-pairing one
 * passed before any of this work and is here as the control; the force-pairing
 * one is the case that was wrong. A fixture with only the first would have
 * reported success for the same reason the first -C bar did.
 */

static const char *hc_seqs[] = {
  "ACCCAGACUCUCAGGCCUGGCUGAUAGCCUAGUUGGCACGGACUGACGACUAGACUAAGC",
  "GUUUUGCUUCCGCUAGGGGGAAUCGAACCCUUUUAAACCGAAACAACCGGGAUGUCGCCCGUACGUAUACUUCGUCAACGCUAUUGAUGG",
  "UGUUUUGGGAUGGGUUGGUUGCGUGUAUUGUUUUAGUUGGGUAGUAGGGGGAUCGGCCUGGAUGUGUAGUUCUCG",
  "GAGUUUUUUGGGUGAGUUGGUUAGAUCUGUGGGGUCUAUGUUCUGAUCUGCUUCAGGUUUGUUUUGAGUGCCUGUUGGGCGUUUAUAUUGGGUGUUUGUGUGUUAUGGUUUUAUGGGGGG",
  "AGCUACUCAAACGCACCUGUGUGGUUUACCGAGCUAACGUUCGCUAGCAAAGUGUGUGGCCCGCAUCGUGCUUCUAACUCGGGAGAAUUCUGGUGCAACGCUGCAGUCACGCAGGACGUCAAUCUGACUGAUUCACGAGUGGAUCGGCUU",
  "GGUUUGUUAAUUAGCUUUUGGUGGUUAUUGCGAACGGUGUUAGCGCGGGUUGUGUUUUUGUUUUGUUUGGGUGUUUGGGGCACUUCUGCGCGGUGUCGCAUGGUGUUGGUUGGUGUGUUGUGGUCUCUGUAGUGCGUGGAGGAGUUGUGUUGUUUUGCUUAAUAGUGUCUGUGUUGUAGG"
};

#define HC_N (sizeof(hc_seqs) / sizeof(hc_seqs[0]))


static vrna_fold_compound_t *
hc_fc(const char *seq, const char *constraint, unsigned int opts)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);

  if ((fc) && (constraint))
    vrna_constraints_add(fc, constraint, opts);

  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);

  return fc;
}


/* The free fold, so the constraints can be derived FROM it. A fixed pattern
 * does not bite: the first version of tools/verify_constraint_parity.sh forced
 * a pair the MFE already contained, compared two identical outputs, and
 * passed. */
static char *
hc_free_fold(const char *seq)
{
  vrna_fold_compound_t  *fc = hc_fc(seq, NULL, 0);
  char                  *s  = (char *)vrna_alloc(sizeof(char) * (strlen(seq) + 1));

  (void)vrna_mfe(fc, s);
  vrna_fold_compound_free(fc);

  return s;
}


#suite mfe_cuda_constraints

#tcase Recursion

#test test_batch_honours_forbidden_pairs
{
  /* 'x' on every base the free MFE pairs: maximal bite, and it lands entirely
   * in hc->mx, which the device has always packed. The control. */
  vrna_fold_compound_t  *ref[HC_N], *bat[HC_N];
  char                  *s_ref[HC_N], *s_bat[HC_N], *s_free[HC_N], *con[HC_N];
  float                 e_ref[HC_N], e_bat[HC_N];
  size_t                k, i, n, bites = 0;

  for (k = 0; k < HC_N; k++) {
    n         = strlen(hc_seqs[k]);
    s_free[k] = hc_free_fold(hc_seqs[k]);
    con[k]    = (char *)vrna_alloc(sizeof(char) * (n + 1));

    for (i = 0; i < n; i++)
      con[k][i] = (s_free[k][i] == '.') ? '.' : 'x';

    ref[k]   = hc_fc(hc_seqs[k], con[k], VRNA_CONSTRAINT_DB_DEFAULT);
    bat[k]   = hc_fc(hc_seqs[k], con[k], VRNA_CONSTRAINT_DB_DEFAULT);
    s_ref[k] = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k] = (char *)vrna_alloc(sizeof(char) * (n + 1));

    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
    if (strcmp(s_ref[k], s_free[k]) != 0)
      bites++;
  }

  ck_assert(bites == HC_N);       /* or nothing below is being tested */

  if (vrna_cuda_devices() == 0)
    return;

  ck_assert(vrna_cuda_register_batch_backend() != 0);
  ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);   /* accepted now */
  ck_assert(vrna_mfe_batch(bat, HC_N, s_bat, e_bat) != 0);

  for (k = 0; k < HC_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);
  }

  for (k = 0; k < HC_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_free[k]); free(con[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
  }
}

#test test_batch_honours_enforced_pairing
{
  /*
   * THE CASE THAT WAS SILENTLY WRONG. '|' forces a base to pair with something,
   * and -- measured -- it is NOT enforced without VRNA_CONSTRAINT_DB_ENFORCE_BP
   * (the same constraint gives the free answer without it and a different one
   * with it), so the flag is part of the test rather than an embellishment.
   *
   * Forcing a base to PAIR moves no bit in hc->mx. It moves hc->up_hp and
   * hc->up_int, which is exactly the half the device did not carry.
   */
  vrna_fold_compound_t  *ref[HC_N], *bat[HC_N];
  char                  *s_ref[HC_N], *s_bat[HC_N], *s_free[HC_N], *con[HC_N];
  float                 e_ref[HC_N], e_bat[HC_N];
  size_t                k, i, n, forced, want, bites = 0;
  const unsigned int    opts = VRNA_CONSTRAINT_DB_DEFAULT | VRNA_CONSTRAINT_DB_ENFORCE_BP;

  for (k = 0; k < HC_N; k++) {
    n         = strlen(hc_seqs[k]);
    s_free[k] = hc_free_fold(hc_seqs[k]);
    con[k]    = (char *)vrna_alloc(sizeof(char) * (n + 1));
    want      = (n / 12 > 6) ? n / 12 : 6;

    for (i = 0, forced = 0; i < n; i++) {
      if ((s_free[k][i] == '.') && (forced < want)) {
        con[k][i] = '|';          /* force PAIRED what the free fold left open */
        forced++;
      } else {
        con[k][i] = '.';
      }
    }
    ck_assert(forced == want);

    ref[k]   = hc_fc(hc_seqs[k], con[k], opts);
    bat[k]   = hc_fc(hc_seqs[k], con[k], opts);
    s_ref[k] = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k] = (char *)vrna_alloc(sizeof(char) * (n + 1));

    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
    if (strcmp(s_ref[k], s_free[k]) != 0)
      bites++;
  }

  ck_assert(bites == HC_N);

  if (vrna_cuda_devices() == 0)
    return;

  ck_assert(vrna_cuda_register_batch_backend() != 0);
  ck_assert(vrna_mfe_batch(bat, HC_N, s_bat, e_bat) != 0);

  for (k = 0; k < HC_N; k++) {
    /* the device must not be BETTER than legal: that is the shape the defect
     * took, and an equality check alone would not say which way it failed */
    ck_assert(e_bat[k] > e_ref[k] - 0.01);
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);
  }

  for (k = 0; k < HC_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_free[k]); free(con[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
  }
}

#test test_unconstrained_batch_is_untouched
{
  /* The other half of the gate: both arrays are NULL when no record carries a
   * depot, so an unconstrained fold reads neither. This cannot see the pointer,
   * but it can see that the answer and the routing are unchanged -- and int_loop
   * is latency-bound, so two extra loads per candidate would land in the worst
   * possible place if the gate ever inverted. */
  vrna_fold_compound_t  *ref[HC_N], *bat[HC_N];
  char                  *s_ref[HC_N], *s_bat[HC_N];
  float                 e_ref[HC_N], e_bat[HC_N];
  size_t                k, n;

  for (k = 0; k < HC_N; k++) {
    n        = strlen(hc_seqs[k]);
    ref[k]   = hc_fc(hc_seqs[k], NULL, 0);
    bat[k]   = hc_fc(hc_seqs[k], NULL, 0);
    s_ref[k] = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k] = (char *)vrna_alloc(sizeof(char) * (n + 1));
    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
  }

  if (vrna_cuda_devices() == 0)
    return;

  ck_assert(vrna_cuda_register_batch_backend() != 0);
  ck_assert(vrna_mfe_batch(bat, HC_N, s_bat, e_bat) != 0);

  for (k = 0; k < HC_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);
    free(s_ref[k]); free(s_bat[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
  }
}

#main-pre
    srunner_set_tap(sr, "-");

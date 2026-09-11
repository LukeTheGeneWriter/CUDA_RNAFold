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
 * --maxBPspan (md.max_bp_span) on the batch backend.
 *
 * The span is a pure HARD CONSTRAINT -- constraints/hard.c:778 keeps a pair only
 * when `(j - i) < md->max_bp_span` -- and rnafold_hc_opt(), the device replica of
 * hc_reset_to_default(), has carried that test all along. Restricting the span
 * changes no extent in the sweep: the triangles, row offsets and chunk sizing
 * are all derived from LENGTHS, so it only makes more cells INF.
 *
 * WHAT WAS ACTUALLY WRONG was that the span reached the device as ONE SCALAR for
 * a batch in which it is PER RECORD. vrna_fold_compound() sets max_bp_span to
 * each compound's own length when the user gives no --maxBPspan
 * (fold_compound.c:598-601), so even the default model has a different span per
 * record in a mixed-length batch. d_span_H is now a table, built beside d_len_H.
 * A second defect went with it: the device clamped `span < 5` up to the record
 * length, which would have turned --maxBPspan=3 into an UNRESTRICTED fold.
 *
 * AND NEITHER DEFECT COULD EVER HAVE CHANGED AN ANSWER, which is worth writing
 * down so the next person does not go looking for the wrong-answer window.
 * The span constraint is NESTED-MONOTONE: if (i,j) satisfies j - i < span, then
 * every pair (k,l) with i < k < l < j satisfies l - k < span automatically. So
 * an over-permissive device mask can only admit extra OUTERMOST pairs -- and
 * vrna_mfe_exterior_f5() runs on the HOST, from upstream's own hc->mx, and
 * filters exactly those. Measured both ways: restoring either defect leaves the
 * fold byte-identical while RNA_HC_VERIFY reports 139 269 mismatching words,
 * and forbidding every pair in the same replica opens every structure and moves
 * 80 lines -- which is the control proving the path is live and the probe can
 * reach it.
 *
 * So the bar that matters here is RNA_HC_VERIFY=1, not the fold: it rebuilds all
 * four masks the host way, straight out of VC[H]->hc->mx, and compares them word
 * for word. That check has NO ORACLE in it, and "does the device's
 * hard-constraint replica agree with upstream's" is the whole feature.
 * This test covers what a .ts can: routing, and the answer.
 */

static const char *span_seqs[] = {
  "ACCCAGACUCUCAGGCCUGGCUGAUAGCCUAGUUGGCACGGACUGACGACUAGACUAAGC",
  "GUUUUGCUUCCGCUAGGGGGAAUCGAACCCUUUUAAACCGAAACAACCGGGAUGUCGCCCGUACGUAUACUUCGUCAACGCUAUUGAUGG",
  "UGUUUUGGGAUGGGUUGGUUGCGUGUAUUGUUUUAGUUGGGUAGUAGGGGGAUCGGCCUGGAUGUGUAGUUCUCG",
  "GAGUUUUUUGGGUGAGUUGGUUAGAUCUGUGGGGUCUAUGUUCUGAUCUGCUUCAGGUUUGUUUUGAGUGCCUGUUGGGCGUUUAUAUUGGGUGUUUGUGUGUUAUGGUUUUAUGGGGGG",
  "AGCUACUCAAACGCACCUGUGUGGUUUACCGAGCUAACGUUCGCUAGCAAAGUGUGUGGCCCGCAUCGUGCUUCUAACUCGGGAGAAUUCUGGUGCAACGCUGCAGUCACGCAGGACGUCAAUCUGACUGAUUCACGAGUGGAUCGGCUU",
  "GGUUUGUUAAUUAGCUUUUGGUGGUUAUUGCGAACGGUGUUAGCGCGGGUUGUGUUUUUGUUUUGUUUGGGUGUUUGGGGCACUUCUGCGCGGUGUCGCAUGGUGUUGGUUGGUGUGUUGUGGUCUCUGUAGUGCGUGGAGGAGUUGUGUUGUUUUGCUUAAUAGUGUCUGUGUUGUAGG"
};

#define SPAN_N (sizeof(span_seqs) / sizeof(span_seqs[0]))
#define SPAN   30

/*
 * SPAN 30 AND THESE SIX RECORDS ARE BOTH MEASURED, not chosen. "Shorter than
 * the record" is NOT enough for the restriction to bind: a 45 nt record whose
 * MFE fold happens to contain no pair spanning more than 20 is unaffected by a
 * span of 30, and the first draft of this fixture included exactly that record.
 * Checked against pristine 2.7.2 at spans 20, 30 and 40 -- all six below change
 * energy at 20 and 30 (one is unaffected at 40, which is why SPAN is 30).
 */


static vrna_fold_compound_t *
span_fc(const char *seq, int span)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  if (span > 0)
    md.max_bp_span = span;

  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);

  return fc;
}


#suite mfe_cuda_span

#tcase Recursion

#test test_batch_honours_max_bp_span
{
  vrna_fold_compound_t  *ref[SPAN_N], *bat[SPAN_N], *def[SPAN_N];
  char                  *s_ref[SPAN_N], *s_bat[SPAN_N], *s_def[SPAN_N];
  float                 e_bat[SPAN_N], e_ref[SPAN_N], e_def[SPAN_N];
  size_t                k, bites = 0;
  unsigned int          devices, registered;

  for (k = 0; k < SPAN_N; k++) {
    size_t n = strlen(span_seqs[k]);

    ck_assert(n > SPAN);          /* or the restriction would not bind */
    ref[k]    = span_fc(span_seqs[k], SPAN);
    bat[k]    = span_fc(span_seqs[k], SPAN);
    def[k]    = span_fc(span_seqs[k], 0);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_def[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    ck_assert(ref[k]->params->model_details.max_bp_span == SPAN);
    /* the default really is per-record, which is why one scalar was wrong */
    ck_assert((unsigned int)def[k]->params->model_details.max_bp_span == n);
  }

  for (k = 0; k < SPAN_N; k++) {
    e_ref[k] = vrna_mfe(ref[k], s_ref[k]);
    e_def[k] = vrna_mfe(def[k], s_def[k]);
    if (fabs(e_ref[k] - e_def[k]) > 0.01)
      bites++;
  }

  /* the restriction must change the answer on every record, or the comparison
   * below is testing nothing -- the same trap the dangles fixture fell into */
  ck_assert(bites == SPAN_N);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* the guard must now ACCEPT a genuinely restricted span */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, SPAN_N, s_bat, e_bat) != 0);

  for (k = 0; k < SPAN_N; k++) {
    ck_assert_str_eq(s_ref[k], s_bat[k]);
    ck_assert(fabs(e_ref[k] - e_bat[k]) < 0.01);
    ck_assert(fabs(e_bat[k] - e_def[k]) > 0.01);
  }

  for (k = 0; k < SPAN_N; k++) {
    free(s_ref[k]); free(s_bat[k]); free(s_def[k]);
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    vrna_fold_compound_free(def[k]);
  }
}

#test test_span_smaller_than_turn_forbids_everything
{
  /*
   * --maxBPspan=3 against a min_loop_size of 3 leaves NO legal pair at all, and
   * the fold is entirely open. It is here because it is the case the removed
   * `span < 5` clamp would have got most wrong -- that line substituted the
   * record's LENGTH for any span below 5, turning the tightest possible
   * restriction into no restriction whatsoever.
   */
  vrna_fold_compound_t  *bat[SPAN_N];
  char                  *s[SPAN_N];
  float                 e[SPAN_N];
  size_t                k;

  if (vrna_cuda_devices() == 0)
    return;

  ck_assert(vrna_cuda_register_batch_backend() != 0);

  for (k = 0; k < SPAN_N; k++) {
    bat[k] = span_fc(span_seqs[k], 3);
    s[k]   = (char *)vrna_alloc(sizeof(char) * (strlen(span_seqs[k]) + 1));
  }

  ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  ck_assert(vrna_mfe_batch(bat, SPAN_N, s, e) != 0);

  for (k = 0; k < SPAN_N; k++) {
    ck_assert(strchr(s[k], '(') == NULL);     /* no pair anywhere */
    ck_assert(fabs(e[k]) < 0.01);             /* and it costs nothing */
    free(s[k]);
    vrna_fold_compound_free(bat[k]);
  }
}

#main-pre
    srunner_set_tap(sr, "-");

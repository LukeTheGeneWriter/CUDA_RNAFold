#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/params/basic.h>
#include <ViennaRNA/mfe/cuda/engine.h>

/*
 * fM1 under uniq_ML, through the batch backend.
 *
 * The MFE recursion never reads fM1, so a wrong fM1 cannot show up in an
 * energy or a structure -- every existing bar in this project would stay green
 * with the matrix left entirely INF, which is exactly what the sweep used to
 * do. vrna_subopt() reads it, though, and vrna_mfe_batch() is public. So the
 * matrix itself has to be compared, cell by cell, against a compound that
 * upstream folded on its own.
 */

static const char *fm1_seqs[] = {
  "GGGAAACCCAUAGCUAGCUAGCGGCCAAUUGGCCAUAUAUCGCGCAUAUAGGCUUAAGGCCUUAAGG",
  "AUGGCCAUUGCAUUGGCCAUAGGCUAUAGCAUCGAUCGAUUAGCUAGCUAGCAUCGAUCGGGCCCAAAUUU",
  "CGCGCGAUAUAUCGCGAUAUCGCGCAUAUAUCGCGGGCCCAAAUUUGGGCCCAAAUUUGGGCCCAAA",
  "UUUAAAGGGCCCUUUAAAGGGCCCAUAUAUGCGCGCAUAUAUGCGCGCUUUAAAGGGCCCUUUAAAG"
};

#define FM1_N (sizeof(fm1_seqs) / sizeof(fm1_seqs[0]))

static vrna_fold_compound_t *
fm1_fc(const char *seq)
{
  vrna_md_t md;

  vrna_md_set_default(&md);
  md.uniq_ML = 1;               /* the whole point: allocates and fills fM1 */

  return vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
}

#suite  MFE_CUDA_fM1

#tcase  Reconstruction

#test test_batch_fills_fM1_exactly_as_upstream
{
  vrna_fold_compound_t  *ref[FM1_N], *bat[FM1_N];
  char                  *s_ref[FM1_N], *s_bat[FM1_N];
  float                 e_bat[FM1_N];
  size_t                k;
  unsigned int          devices, registered;

  for (k = 0; k < FM1_N; k++) {
    size_t n = strlen(fm1_seqs[k]);

    ref[k]    = fm1_fc(fm1_seqs[k]);
    bat[k]    = fm1_fc(fm1_seqs[k]);
    s_ref[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));
    s_bat[k]  = (char *)vrna_alloc(sizeof(char) * (n + 1));

    /* uniq_ML must actually have produced the matrix, or this test is empty */
    ck_assert(ref[k]->params->model_details.uniq_ML == 1);
  }

  /* the reference: folded one at a time, entirely by upstream */
  for (k = 0; k < FM1_N; k++)
    (void)vrna_mfe(ref[k], s_ref[k]);

  for (k = 0; k < FM1_N; k++)
    ck_assert(ref[k]->matrices->fM1 != NULL);

  devices     = vrna_cuda_devices();
  registered  = vrna_cuda_register_batch_backend();

  if (devices == 0) {
    /*
     * No device. Rather than compare upstream against itself and call that a
     * pass -- the shape of several checks that have already fooled this
     * project -- assert the real behaviour of this configuration: nothing is
     * registered, and vrna_mfe_batch() therefore falls through to its own loop
     * over vrna_mfe(), which fills fM1 itself.
     */
    ck_assert(registered == 0);
  } else {
    ck_assert(registered != 0);
    /* and the guard must now ACCEPT uniq_ML, or the batch would decline and
     * this would silently become the no-device case above */
    ck_assert(vrna_cuda_engine_supports(bat[0], NULL) == 1);
  }

  ck_assert(vrna_mfe_batch(bat, FM1_N, s_bat, e_bat) != 0);

  for (k = 0; k < FM1_N; k++) {
    unsigned int  n     = bat[k]->length;
    const int     *indx = bat[k]->jindx;
    unsigned int  i, j;
    size_t        compared = 0;

    /* same answer first: if the structures differ, an fM1 comparison is moot */
    ck_assert_str_eq(s_ref[k], s_bat[k]);

    ck_assert(bat[k]->matrices->fM1 != NULL);

    for (j = 1; j <= n; j++) {
      for (i = 1; i <= j; i++) {
        int a = ref[k]->matrices->fM1[indx[j] + i];
        int b = bat[k]->matrices->fM1[indx[j] + i];
        ck_assert_int_eq(a, b);
        compared++;
      }
    }

    /* a comparison that compared nothing is not a passing comparison */
    ck_assert(compared == (size_t)n * (n + 1) / 2);
  }

  for (k = 0; k < FM1_N; k++) {
    vrna_fold_compound_free(ref[k]);
    vrna_fold_compound_free(bat[k]);
    free(s_ref[k]);
    free(s_bat[k]);
  }
}

#test test_fM1_is_not_trivially_infinite
{
  /*
   * The failure this whole feature is about is an fM1 left at INF everywhere.
   * That would satisfy a cell-by-cell comparison against another all-INF
   * matrix, so assert independently that a real fold produces finite cells.
   */
  vrna_fold_compound_t  *fc = fm1_fc(fm1_seqs[0]);
  unsigned int          n   = fc->length;
  const int             *indx;
  unsigned int          i, j;
  size_t                finite = 0;
  char                  *s = (char *)vrna_alloc(sizeof(char) * (n + 1));

  (void)vrna_mfe(fc, s);

  indx = fc->jindx;
  for (j = 1; j <= n; j++)
    for (i = 1; i <= j; i++)
      if (fc->matrices->fM1[indx[j] + i] < INF / 2)
        finite++;

  ck_assert(finite > 0);

  free(s);
  vrna_fold_compound_free(fc);
}

#main-pre
    srunner_set_tap(sr, "-");

#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <string.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/grammar/mfe.h>

static const char *engine_seq =
  "GGGAAACCCAUAGCUAGCUAGCGGCCAAUUGGCCAUAUAUCGCGCAUAUAGGCUUAAGGCCUUAAGG";

struct engine_probe {
  unsigned int  calls;
  int           handle;   /* 0 = decline, 1 = handle */
  int           energy;   /* energy to report when handling, in dekacal/mol */
};

static int
probe_engine(vrna_fold_compound_t *fc,
             int                  *energy,
             void                 *data)
{
  struct engine_probe *p = (struct engine_probe *)data;

  (void)fc;
  p->calls++;

  if (p->handle) {
    *energy = p->energy;
    return 1;
  }

  return 0;
}

#suite  MFE_Inside_Engine

#tcase  Dispatch

#test test_engine_declines
{
  /*
   * An engine that declines every fold compound must leave the prediction
   * exactly as it was -- same energy, same structure -- and must be reached
   * exactly once per vrna_mfe() call, or the dispatch is dead code.
   */
  unsigned int          n     = (unsigned int)strlen(engine_seq);
  char                  *s_a  = (char *)vrna_alloc(sizeof(char) * (n + 1));
  char                  *s_b  = (char *)vrna_alloc(sizeof(char) * (n + 1));
  struct engine_probe   p     = { 0, 0, 0 };
  vrna_fold_compound_t  *fc_a = vrna_fold_compound(engine_seq, NULL, VRNA_OPTION_DEFAULT);
  vrna_fold_compound_t  *fc_b = vrna_fold_compound(engine_seq, NULL, VRNA_OPTION_DEFAULT);
  float                 mfe_a, mfe_b;

  mfe_a = vrna_mfe(fc_a, s_a);

  ck_assert(vrna_gr_set_inside_engine(fc_b, &probe_engine, (void *)&p, NULL, NULL) != 0);
  mfe_b = vrna_mfe(fc_b, s_b);

  ck_assert_uint_eq(p.calls, 1);
  ck_assert(mfe_a == mfe_b);
  ck_assert_str_eq(s_a, s_b);

  free(s_a);
  free(s_b);
  vrna_fold_compound_free(fc_a);
  vrna_fold_compound_free(fc_b);
}

#test test_engine_handles
{
  /*
   * An engine that handles a fold compound must have its reported energy used.
   * The value is deliberately absurd: what is under test is that it travels,
   * not that it is correct.
   */
  unsigned int          n   = (unsigned int)strlen(engine_seq);
  char                  *ss = (char *)vrna_alloc(sizeof(char) * (n + 1));
  struct engine_probe   p   = { 0, 1, -424242 };
  vrna_fold_compound_t  *fc = vrna_fold_compound(engine_seq, NULL, VRNA_OPTION_DEFAULT);
  float                 mfe;

  ck_assert(vrna_gr_set_inside_engine(fc, &probe_engine, (void *)&p, NULL, NULL) != 0);
  mfe = vrna_mfe(fc, ss);

  ck_assert_uint_eq(p.calls, 1);
  ck_assert(mfe == (float)(-424242 / 100.));

  free(ss);
  vrna_fold_compound_free(fc);
}

#test test_engine_is_exclusive
{
  /* At most one inside engine per fold compound; the first one wins. */
  struct engine_probe   first   = { 0, 0, 0 };
  struct engine_probe   second  = { 0, 0, 0 };
  vrna_fold_compound_t  *fc     = vrna_fold_compound(engine_seq, NULL, VRNA_OPTION_DEFAULT);

  ck_assert(vrna_gr_set_inside_engine(fc, &probe_engine, (void *)&first, NULL, NULL) != 0);
  ck_assert(vrna_gr_set_inside_engine(fc, &probe_engine, (void *)&second, NULL, NULL) == 0);

  vrna_fold_compound_free(fc);
}

#test test_engine_rejects_null
{
  vrna_fold_compound_t *fc = vrna_fold_compound(engine_seq, NULL, VRNA_OPTION_DEFAULT);

  ck_assert(vrna_gr_set_inside_engine(NULL, &probe_engine, NULL, NULL, NULL) == 0);
  ck_assert(vrna_gr_set_inside_engine(fc, NULL, NULL, NULL, NULL) == 0);

  vrna_fold_compound_free(fc);
}

#main-pre
    srunner_set_tap(sr, "-");

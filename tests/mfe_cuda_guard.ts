#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/grammar/mfe.h>
#include <ViennaRNA/constraints/soft.h>
#include <ViennaRNA/mfe/cuda/engine.h>

static const char *guard_seq =
  "GGGAAACCCAUAGCUAGCUAGCGGCCAAUUGGCCAUAUAUCGCGCAUAUAGGCUUAAGGCCUUAAGG";

/* build a fold compound with one model detail changed from the default */
static vrna_fold_compound_t *
fc_with(void (*tweak)(vrna_md_t *))
{
  vrna_md_t md;

  vrna_md_set_default(&md);
  if (tweak)
    tweak(&md);

  return vrna_fold_compound(guard_seq, &md, VRNA_OPTION_DEFAULT);
}

static void md_dangles0(vrna_md_t *md)  { md->dangles = 0;      }
static void md_gquad(vrna_md_t *md)     { md->gquad = 1;        }
static void md_circ(vrna_md_t *md)      { md->circ = 1;         }
static void md_nolp(vrna_md_t *md)      { md->noLP = 1;         }
static void md_noguclose(vrna_md_t *md) { md->noGUclosure = 1;  }
static void md_uniqml(vrna_md_t *md)    { md->uniq_ML = 1;      }
static void md_salt(vrna_md_t *md)      { md->salt = 0.2;       }

static int
declines(vrna_fold_compound_t *fc)
{
  const char *reason = NULL;
  int         ok     = vrna_cuda_engine_supports(fc, &reason) ? 0 : 1;

  /* a decline must always come with a reason, or the log is useless */
  if (ok && (reason == NULL))
    return 0;

  return ok;
}

#suite  MFE_CUDA_Guard

#tcase  Routing

#test test_guard_accepts_the_default_model
{
  /*
   * The guard is only meaningful if it says yes to something. If this fails,
   * every other case in this suite passes vacuously.
   */
  vrna_fold_compound_t  *fc     = fc_with(NULL);
  const char            *reason = "not set";

  ck_assert(vrna_cuda_engine_supports(fc, &reason) != 0);
  ck_assert(reason == NULL);

  vrna_fold_compound_free(fc);
}

#test test_guard_declines_unsupported_models
{
  /*
   * Each of these changes the recursion or the energies. Every one was either
   * silently wrong or half-applied on the 2.3.0 GPU path before it was guarded.
   */
  void (*tweaks[])(vrna_md_t *) = {
    md_dangles0, md_gquad, md_circ, md_nolp, md_noguclose, md_uniqml, md_salt
  };
  size_t i;

  for (i = 0; i < sizeof(tweaks) / sizeof(tweaks[0]); i++) {
    vrna_fold_compound_t *fc = fc_with(tweaks[i]);
    ck_assert(declines(fc));
    vrna_fold_compound_free(fc);
  }
}

#test test_guard_declines_soft_constraints
{
  /* soft constraints reach the recursion as host callbacks */
  vrna_fold_compound_t *fc = fc_with(NULL);

  vrna_sc_init(fc);
  ck_assert(declines(fc));

  vrna_fold_compound_free(fc);
}

#test test_guard_declines_a_windowed_fold_compound
{
  /*
   * Regression: the first version of the guard tested md->window_size > 0,
   * which looks right and is not -- vrna_fold_compound() sets window_size AND
   * max_bp_span to the sequence length for an ordinary global fold, so that
   * test declined every fold compound including the default one. What actually
   * separates a local fold is the hard constraint layout, so a real windowed
   * fold compound is built here rather than a model detail being poked.
   */
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  md.window_size = 30;
  md.max_bp_span = 30;

  fc = vrna_fold_compound(guard_seq, &md, VRNA_OPTION_WINDOW);

  ck_assert(declines(fc));

  vrna_fold_compound_free(fc);
}

#test test_guard_declines_a_restricted_bp_span
{
  /* a span genuinely shorter than the sequence, unlike the default span == n */
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;

  vrna_md_set_default(&md);
  md.max_bp_span = 10;

  fc = vrna_fold_compound(guard_seq, &md, VRNA_OPTION_DEFAULT);

  ck_assert(declines(fc));

  vrna_fold_compound_free(fc);
}

#test test_guard_declines_null
{
  ck_assert(vrna_cuda_engine_supports(NULL, NULL) == 0);
}

#test test_attaching_the_backend_does_not_change_the_answer
{
  /*
   * The end-to-end property that matters while the device fill does not exist
   * yet, and the one to keep asserting after it does for anything the backend
   * declines: attaching the backend must leave energy AND structure untouched.
   *
   * Deliberately indifferent to whether a GPU is present. With no device the
   * attach simply fails; either way the answer must be upstream's own.
   */
  unsigned int          n     = (unsigned int)strlen(guard_seq);
  char                  *s_a  = (char *)vrna_alloc(sizeof(char) * (n + 1));
  char                  *s_b  = (char *)vrna_alloc(sizeof(char) * (n + 1));
  vrna_fold_compound_t  *fc_a = fc_with(NULL);
  vrna_fold_compound_t  *fc_b = fc_with(NULL);
  float                 mfe_a, mfe_b;

  mfe_a = vrna_mfe(fc_a, s_a);

  (void)vrna_cuda_attach(fc_b);
  mfe_b = vrna_mfe(fc_b, s_b);

  ck_assert(mfe_a == mfe_b);
  ck_assert_str_eq(s_a, s_b);

  free(s_a);
  free(s_b);
  vrna_fold_compound_free(fc_a);
  vrna_fold_compound_free(fc_b);
}

#main-pre
    srunner_set_tap(sr, "-");

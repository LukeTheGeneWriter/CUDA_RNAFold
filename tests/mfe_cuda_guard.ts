#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/grammar/mfe.h>
#include <ViennaRNA/constraints/soft.h>
#include <ViennaRNA/constraints/hard.h>
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
static void md_dangles1(vrna_md_t *md)  { md->dangles = 1;      }
static void md_dangles3(vrna_md_t *md)  { md->dangles = 3;      }
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
  /* md_uniqml, md_salt, md_nolp, md_gquad, md_circ, md_noguclose and
   * md_dangles0 are deliberately NOT here any more -- all are supported, and
   * each has its own accepts- test so that a guard silently re-tightening shows
   * up as a failure rather than as a quiet loss of acceleration. md_gquad moved
   * out on 2026-09-10 (G3); md_circ, md_noguclose and md_dangles0 on
   * 2026-09-11.
   *
   * WHAT IS LEFT IS DANGLE MODELS 1 AND 3, and they are a different kind of
   * "no" from everything that has left this list. The others were unimplemented
   * arithmetic. These need new DP STATE: ml_pair_d1() reads dmli2 as well as
   * dmli1, a second DMLi generation the sweep does not carry, and d3 adds
   * coaxial stacking on top. tests/mfe_cuda_dangles.ts says the same thing from
   * the other side and explains why d0 was cheap. */
  void (*tweaks[])(vrna_md_t *) = {
    md_dangles1, md_dangles3
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

#test test_guard_accepts_dangles0
{
  /* Dangle model 0, accepted 2026-09-11. d0 and d2 share the recursion --
   * mfe_multibranch.c:686 dispatches ml_pair_d0 or ml_pair_d2 and both read
   * only dmli1 -- so it cost two energy terms and no new DP state.
   * tests/mfe_cuda_dangles.ts checks the ANSWER; this only asserts routing. */
  vrna_fold_compound_t *fc = fc_with(md_dangles0);

  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);

  vrna_fold_compound_free(fc);
}

#test test_guard_accepts_noGUclosure
{
  /* --noClosingGU was the LAST option behind all three gates. It was declined
   * because it was HALF implemented: rnafold_hc_opt() dropped HP_LOOP and
   * MB_LOOP for a GU/UG pair, but nothing applied the interior-loop half, so c
   * was internally inconsistent rather than merely suboptimal. Energy() now
   * skips GU-closed and GU-enclosed bulges and interior loops (stacks exempt,
   * as in upstream's mfe_stacks()). tests/mfe_cuda_noclosinggu.ts checks the
   * ANSWER; this only asserts the routing decision. */
  vrna_fold_compound_t *fc = fc_with(md_noguclose);

  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);

  vrna_fold_compound_free(fc);
}

#test test_guard_accepts_uniq_ML
{
  /* The counterpart to the removal above. uniq_ML was declined because the
   * sweep left fM1 entirely INF; it is accepted now that backtrack_one_slot()
   * reconstructs it. tests/mfe_cuda_fm1.ts checks the matrix itself -- this
   * only asserts the routing decision, so that a guard silently re-tightening
   * shows up as a failure here rather than as a quiet loss of acceleration. */
  vrna_fold_compound_t *fc = fc_with(md_uniqml);

  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);

  vrna_fold_compound_free(fc);
}

#test test_guard_accepts_circ
{
  /* -c was declined because the sweep never filled fM2_real, which
   * postprocess_circular() reads in 13 places -- so the GPU returned the LINEAR
   * answer for a circular fold. Accepted as of 2026-09-11: DMLi, which the
   * modular decomposition already reduces every row and the fork discarded, IS
   * fM2_real, and it is now persisted into a triangle.
   *
   * tests/mfe_cuda_circ.ts checks the STRUCTURES; this only asserts the routing
   * decision, so a guard silently re-tightening shows up here rather than as a
   * quiet loss of acceleration. */
  vrna_fold_compound_t *fc = fc_with(md_circ);

  ck_assert(fc->params->model_details.circ == 1);
  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);
  vrna_fold_compound_free(fc);
}

#test test_guard_accepts_gquad
{
  /* -g was declined because the sweep scored no quadruplex contribution into
   * c/fML at all and returned a valid, self-consistent structure 15-31
   * kcal/mol above the true MFE -- the worst failure shape available. It is
   * accepted as of 2026-09-10: c_gq is carried to the device (G0),
   * extend_fm_3p()'s term is in fml_scan_kernel (G1), and
   * vrna_mfe_gquad_internal_loop()'s three sweeps are in gq_internal_kernel
   * (G2). The last blocker was the BACKTRACK, not the recursion -- see
   * VRNA-PATCH(bps-backtrack).
   *
   * tests/mfe_cuda_gquad.ts checks the STRUCTURES; this only asserts the
   * routing decision, so a guard silently re-tightening shows up here rather
   * than as a quiet loss of acceleration. */
  vrna_fold_compound_t *fc = fc_with(md_gquad);

  ck_assert(fc->params->model_details.gquad == 1);
  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);
  vrna_fold_compound_free(fc);
}

#test test_guard_accepts_noLP
{
  /* noLP was declined because the sweep filtered ptype for it but never
   * applied the RECURSION constraint, leaving the matrix fill and the
   * backtrack disagreeing with each other by 87-300 kcal/mol -- while the
   * reported ENERGY still agreed with upstream on 45 of 60 records, which is
   * why an energy comparison mis-scored it as "3 of 12". new_c_kernel now
   * writes cc1[j-1]+stackEnergy into c and carries the unconstrained value in
   * cc, per mfe/mfe.c:4413.
   *
   * tests/mfe_cuda_nolp.ts checks the STRUCTURES; this only asserts the
   * routing decision, so a guard silently re-tightening shows up here rather
   * than as a quiet loss of acceleration. */
  vrna_fold_compound_t *fc = fc_with(md_nolp);

  ck_assert(fc->params->model_details.noLP == 1);
  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);

  vrna_fold_compound_free(fc);
}

#test test_guard_accepts_salt
{
  /* Salt reaches the device three ways, and only one of them is code: the
   * multibranch parameters carry it already (params.c:640-645), while the
   * hairpin and internal-loop kernels add a term from a host-built table.
   * tools/verify_salt_parity.sh checks the ENERGIES; this checks only that the
   * routing decision still says yes. */
  vrna_fold_compound_t *fc = fc_with(md_salt);

  ck_assert(fc->params->model_details.salt != VRNA_MODEL_DEFAULT_SALT);
  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);

  vrna_fold_compound_free(fc);
}

/*
 * test_guard_declines_nonstandard_pairs lived here from 2026-09-09 to
 * 2026-09-10. THE GUARD WAS LIFTED, so an assertion that it declines would
 * now be asserting the bug back into existence.
 *
 * It was not deleted for being wrong -- it closed a live wrong answer, and it
 * tested md->pair[i][j] == 7 (the EFFECT) rather than md->nonstandards (the
 * request field), which is still the right shape for any future guard. It was
 * replaced because int_loop.cu's Energy() was fixed: it now promotes ptype
 * 0 -> 7 and applies rtype[] instead of swapping the index order, so --nsp is
 * byte-identical GPU vs CPU.
 *
 * The replacement is tests/mfe_cuda_nsp.ts, which compares the batch backend
 * against upstream under an ASYMMETRIC spec. That distinction is not
 * cosmetic: a symmetric spec (--nsp="-GA") masks the divergence completely
 * and passes against the BROKEN binary. See PORT_NSP_PARAMFILE_SCOPE.md 1.
 */

#test test_guard_declines_hard_structure_constraints
{
  /*
   * Regression, and it was a live wrong-answer bug rather than a hypothetical.
   * A -C style dot-bracket constraint leaves hc->type at VRNA_HC_DEFAULT and
   * hc->f at NULL, so before the depot check every other test in this guard
   * passed a constrained compound straight through to the device -- which
   * returned -14.30 for a structure worth -5.40.
   *
   * The compound is deliberately NOT prepared here, because that is the state
   * the batch callback sees: vrna_constraints_add() only queues, and hc->mx,
   * ptype and up_* are all still byte-identical to an unconstrained compound
   * until vrna_fold_compound_prepare() runs inside par_mfe(). A guard that
   * inspected hc->mx at this point would compare two identical matrices and
   * accept everything.
   */
  vrna_fold_compound_t  *fc = fc_with(NULL);
  char                  *con;
  size_t                n = strlen(guard_seq), i;

  con = (char *)vrna_alloc(sizeof(char) * (n + 1));
  for (i = 0; i < n; i++)
    con[i] = (i > n / 4 && i < n / 2) ? 'x' : '.';
  con[n] = '\0';

  /* the guard must accept it before the constraint is added ... */
  ck_assert(vrna_cuda_engine_supports(fc, NULL) == 1);

  vrna_constraints_add(fc, con, VRNA_CONSTRAINT_DB_DEFAULT);

  /* ... and decline it after, with nothing prepared in between */
  ck_assert(fc->hc->depot != NULL);
  ck_assert(declines(fc));

  free(con);
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

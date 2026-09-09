/* What is inside `build`? -- the stage that is 19% of wall at 400 x 5601.
 *
 * stage_build_s wraps vrna_fold_compound() per record, and that call does three
 * O(n^2)-ish things: the sequence encodings, vrna_ptypes(), and vrna_hc_init().
 * Guessing which one dominates is how the gpuinit "deferred cudaMalloc"
 * hypothesis got written down and stayed wrong for two weeks, so measure it.
 *
 * Isolation without adding timers to upstream:
 *   vrna_ptypes()                    is PUBLIC, so it can be timed directly.
 *   VRNA_OPTION_EVAL_ONLY            skips vrna_hc_init() (fold_compound.c:265)
 *                                    but still builds ptype, so
 *                                    DEFAULT - EVAL_ONLY == vrna_hc_init().
 *
 * Build (against a configured tree):
 *   gcc -O2 -o build_split tools/build_split.c -I<tree>/src \
 *       <tree>/src/ViennaRNA/.libs/libRNA.a -lm -lstdc++ -lgomp -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/sequences/alphabet.h>
#include <ViennaRNA/constraints/hard.h>
#include <ViennaRNA/utils/basic.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static char *rnd_seq(int n, unsigned *seed) {
  static const char b[] = "ACGU";
  char *s = (char *)malloc(n + 1);
  int i;
  for (i = 0; i < n; i++) {
    *seed = *seed * 1103515245u + 12345u;
    s[i] = b[(*seed >> 16) & 3];
  }
  s[n] = '\0';
  return s;
}

int main(int argc, char **argv) {
  const int n    = (argc > 1) ? atoi(argv[1]) : 5601;
  const int reps = (argc > 2) ? atoi(argv[2]) : 5;
  unsigned seed  = 12345;
  vrna_md_t md;
  int r;
  double t_full = 0, t_eval = 0, t_pt = 0, t_free = 0, t_prep = 0;

  vrna_md_set_default(&md);

  printf("n=%d  reps=%d\n", n, reps);

  for (r = 0; r < reps; r++) {
    char *s = rnd_seq(n, &seed);
    double t0, t1;
    vrna_fold_compound_t *fc;

    /* 1. the real thing, exactly as RNAfold.c's chunk loop calls it */
    t0 = now();
    fc = vrna_fold_compound(s, &md, VRNA_OPTION_DEFAULT);
    t1 = now(); t_full += t1 - t0;

    t0 = now();
    vrna_fold_compound_free(fc);
    t1 = now(); t_free += t1 - t0;

    /* 2. What did DEFAULT actually build, and what does prepare() add?
     *
     * The first version of this probe assumed VRNA_OPTION_EVAL_ONLY isolated
     * vrna_hc_init(), and reported a NEGATIVE residual -- which is how it
     * announced that the assumption was wrong. vrna_ptypes_prepare() is gated on
     * `options & VRNA_OPTION_MFE` (alphabet.c:166) and VRNA_OPTION_DEFAULT is 0,
     * so DEFAULT builds NO ptype at all; RNAfold's chunk loop gets it later from
     * vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE), which is stage_prepare.
     * Measure both, and print what is actually allocated rather than assuming. */
    t0 = now();
    fc = vrna_fold_compound(s, &md, VRNA_OPTION_DEFAULT);
    t1 = now(); t_eval += t1 - t0;
    if (r == 0)
      printf("  after DEFAULT:  ptype=%s  hc=%s  hc->mx=%s\n",
             fc->ptype ? "built" : "NULL",
             fc->hc ? "built" : "NULL",
             (fc->hc && fc->hc->mx) ? "built" : "NULL");
    t0 = now();
    vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);
    t1 = now(); t_prep += t1 - t0;
    if (r == 0)
      printf("  after prepare:  ptype=%s\n", fc->ptype ? "built" : "NULL");
    vrna_fold_compound_free(fc);

    /* 3. vrna_ptypes() alone, on the same encoding vrna_ptypes_prepare uses */
    {
      short *S2 = vrna_seq_encode_simple(s, &md);
      char  *pt;
      t0 = now();
      pt = vrna_ptypes(S2, &md);
      t1 = now(); t_pt += t1 - t0;
      free(pt);
      free(S2);
    }
    free(s);
  }

#define P(lbl, v) printf("  %-36s %8.3f s   %6.1f%%\n", lbl, (v)/reps, 100.0*(v)/t_full)
  printf("\nper record, averaged over %d (%% of vrna_fold_compound):\n", reps);
  P("vrna_fold_compound(DEFAULT)  [=build]", t_full);
  P("   run again, as a repeatability check", t_eval);
  P("vrna_fold_compound_prepare(MFE)", t_prep);
  P("   of which vrna_ptypes(), standalone", t_pt);
  P("vrna_fold_compound_free()", t_free);
#undef P
  printf("\nscaled to 400 records:  build %.1f s   prepare %.1f s   free %.1f s\n",
         400 * t_full / reps, 400 * t_prep / reps, 400 * t_free / reps);
  printf("(compare the stress run's stage line: build ~122 s, prepare ~0.2 s)\n");
  return 0;
}

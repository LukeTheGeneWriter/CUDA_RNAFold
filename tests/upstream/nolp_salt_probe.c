/* Does --noLP's stacking term miss the salt correction that every other stack
 * in the MFE recursion receives?
 *
 * THE CLAIM
 *
 * Two functions in 2.7.2 compute the energy of the same physical thing -- pair
 * (i,j) stacked directly on (i+1,j-1) -- and they disagree under a non-default
 * salt concentration:
 *
 *   vrna_E_internal(0, 0, type, type_2, ..., P)      eval/eval_internal.c
 *       returns  P->stack[type][type_2] + P->SaltStack
 *
 *   vrna_eval_stack(fc, i, j, VRNA_EVAL_LOOP_DEFAULT)
 *       -> eval_stack()                              eval/eval_internal.c:473
 *       returns  P->stack[type][type_2]              (+ soft constraints only)
 *
 * The second one has no salt term. And the second one is what the --noLP
 * recursion calls:
 *
 *   mfe/mfe.c:4415   stackEnergy = vrna_eval_stack(fc, i, j, ...);
 *                    new_c       = MIN2(new_c, cc1[j-1] + stackEnergy);
 *                    e           = cc1[j-1] + stackEnergy;   -> c[i][j]
 *
 * So with --noLP AND --salt, the stacking term that decides whether a pair may
 * enter c at all is uncorrected, while the identical stack reached through the
 * interior-loop recursion is corrected. The two paths price the same stack
 * differently within a single fold.
 *
 * WHY IT MATTERS BEYOND THE ARITHMETIC
 *
 * --noLP is not an optional extra term: it is what noLP *replaces* c[i][j]
 * with. Every pair in the final structure passes through it. So a missing
 * correction here is not a rounding difference on one loop, it shifts the
 * criterion for admitting pairs across the whole matrix.
 *
 * WHAT THIS PROBE DOES
 *
 * It needs no oracle and no reference implementation. For each sequence it
 * folds once to get a valid pair (i,j) whose (i+1,j-1) is also paired -- a real
 * stack from a real MFE structure -- and asks BOTH functions for its energy, at
 * default salt and at several non-default concentrations. At default salt they
 * must agree (SaltStack is 0). At any other salt they should still agree, and
 * the claim is that they differ by exactly P->SaltStack.
 *
 * Public RNAlib API only, so it can be handed to the maintainers as-is.
 *
 * Reported by the CUDA_RNAFold port, 2026-09-07. Found while implementing
 * --noLP on the GPU path: reusing the interior-loop stack energy would have
 * been wrong by exactly this term, which is what sent us to read both.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/params/basic.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/eval/internal.h>
#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/utils/structures.h>

#ifndef INF
#define INF 10000000
#endif

static const char *seqs[] = {
  "GGGGAAAACCCCGGGGAAAACCCCGGGGAAAACCCCAUAUAUAUGCGCGCGC",
  "GCGCGCGCAAAAGCGCGCGCUUUUGCGCGCGCAAAAGCGCGCGCAUAUAUAU",
  "GGGAAACCCAUAGCUAGCUAGCGGCCAAUUGGCCAUAUAUCGCGCAUAUAGGCUUAAGGCCUUAAGG",
  NULL
};

/* Find a stacked pair in the MFE structure: (i,j) paired AND (i+1,j-1) paired.
 * Returns 1 and sets *oi/*oj, or 0 if the structure has no stack at all. */
static int
find_stack(const char *structure, unsigned int *oi, unsigned int *oj)
{
  short         *pt = vrna_ptable(structure);
  unsigned int  n   = (unsigned int)pt[0];
  int            found = 0;

  for (unsigned int i = 1; i + 2 <= n; i++) {
    unsigned int j = (unsigned int)pt[i];
    if ((j > i) && (pt[i + 1] == (short)(j - 1)) && (j >= i + 3)) {
      *oi = i; *oj = j; found = 1; break;
    }
  }
  free(pt);
  return found;
}

static int
check_one(const char *seq, double salt, int verbose)
{
  vrna_md_t md;
  vrna_md_set_default(&md);
  if (salt != VRNA_MODEL_DEFAULT_SALT)
    md.salt = salt;

  vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
  char  *structure = (char *)vrna_alloc(sizeof(char) * (strlen(seq) + 1));
  (void)vrna_mfe(fc, structure);

  unsigned int i, j;
  if (!find_stack(structure, &i, &j)) {
    if (verbose)
      printf("    salt %-6.3f  no stacked pair in the MFE structure, skipped\n", salt);
    free(structure); vrna_fold_compound_free(fc);
    return 0;
  }

  vrna_param_t *P = fc->params;

  /* Route A: what --noLP calls (mfe/mfe.c:4415). */
  int a = vrna_eval_stack(fc, i, j, VRNA_EVAL_LOOP_DEFAULT);

  /* Route B: the same stack as a degree-2 loop with n1 = n2 = 0, which is what
   * the interior-loop recursion reaches for the identical pair of pairs. */
  int b = vrna_eval_internal(fc, i, j, i + 1, j - 1, VRNA_EVAL_LOOP_DEFAULT);

  const int delta = b - a;
  const int expect = P->SaltStack;

  if (verbose)
    printf("    salt %-6.3f  (i,j)=(%u,%u)  noLP-route %6d   int-loop-route %6d"
           "   diff %5d   P->SaltStack %5d   %s\n",
           salt, i, j, a, b, delta, expect,
           (delta == expect) ? ((expect == 0) ? "agree (salt term is 0)"
                                              : "*** DIFFER BY SaltStack ***")
                             : "differ by something else");

  free(structure);
  vrna_fold_compound_free(fc);
  return (expect != 0) && (delta == expect);
}

int
main(int argc, char **argv)
{
  const int verbose = (argc > 1) && (strcmp(argv[1], "-q") != 0);
  const double salts[] = { VRNA_MODEL_DEFAULT_SALT, 0.05, 0.2, 0.5, 1.0, 5.0 };
  const int    ns = (int)(sizeof(salts) / sizeof(salts[0]));
  int confirmed = 0, tested = 0;

  printf("nolp_salt_probe: does --noLP's stack term miss the salt correction?\n");
  printf("  vrna_eval_stack()   is what mfe.c:4415 calls under noLP\n");
  printf("  vrna_eval_internal() is the same stack via the interior-loop path\n\n");

  for (int s = 0; seqs[s]; s++) {
    printf("  sequence %d (%zu nt)\n", s, strlen(seqs[s]));
    for (int k = 0; k < ns; k++) {
      tested++;
      confirmed += check_one(seqs[s], salts[k], verbose);
    }
    printf("\n");
  }

  printf("VERDICT: %d of %d non-default-salt cases show the two routes pricing\n"
         "         the SAME stack differently, by exactly P->SaltStack.\n",
         confirmed, tested);
  printf("\nIf confirmed, --noLP combined with --salt applies the salt\n"
         "correction to interior-loop stacks but NOT to the stacking term that\n"
         "decides whether a pair may enter c at all.\n");
  return confirmed ? 1 : 0;
}

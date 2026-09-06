/* Probe: does the MFE auxiliary-grammar inside callback receive the segment's
 * 5' delimiter i, as documented (vrna_gr_inside_f, grammar/mfe.h:37), or does it
 * receive the rule index because of the shadowing `size_t i` at mfe/mfe.c:502?
 *
 * Two rules are registered so the two hypotheses give different answers:
 *   documented behaviour -> i ranges over 1..n, same values for both rules
 *   shadowing bug        -> rule 0 always sees i==0, rule 1 always sees i==1
 * The partition-function twin (partfunc/partfunc.c:410) uses `c` for the rule
 * index and passes the real i, which is the intended shape.
 */
#include <stdio.h>
#include <stdlib.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/grammar/mfe.h>
#include <ViennaRNA/utils/basic.h>

#define INF_E 10000000

struct probe {
  int tag;
  unsigned int min_i, max_i, min_j, max_j, calls;
};

static int
probe_cb(vrna_fold_compound_t *fc, unsigned int i, unsigned int j, void *data)
{
  struct probe *p = (struct probe *)data;
  (void)fc;
  if (p->calls == 0) {
    p->min_i = p->max_i = i;
    p->min_j = p->max_j = j;
  } else {
    if (i < p->min_i) p->min_i = i;
    if (i > p->max_i) p->max_i = i;
    if (j < p->min_j) p->min_j = j;
    if (j > p->max_j) p->max_j = j;
  }
  p->calls++;
  return INF_E;              /* contribute nothing: INF, so the MFE cannot change */
}

int
main(void)
{
  const char *seq = "GGGAAACCCAUAGCUAGCUAGCGGCCAAUU";
  unsigned int n = 30;
  struct probe p0 = { 0, 0, 0, 0, 0, 0 };
  struct probe p1 = { 1, 0, 0, 0, 0, 0 };
  char *structure = (char *)vrna_alloc(sizeof(char) * (n + 1));
  vrna_fold_compound_t *fc = vrna_fold_compound(seq, NULL, VRNA_OPTION_DEFAULT);

  if (!vrna_gr_add_aux(fc, &probe_cb, NULL, (void *)&p0, NULL, NULL)) {
    fprintf(stderr, "failed to register rule 0\n");
    return 1;
  }
  if (!vrna_gr_add_aux(fc, &probe_cb, NULL, (void *)&p1, NULL, NULL)) {
    fprintf(stderr, "failed to register rule 1\n");
    return 1;
  }

  float mfe = vrna_mfe(fc, structure);

  printf("sequence length n = %u, mfe = %.2f\n", n, mfe);
  printf("rule 0: %u calls, i in [%u, %u], j in [%u, %u]\n",
         p0.calls, p0.min_i, p0.max_i, p0.min_j, p0.max_j);
  printf("rule 1: %u calls, i in [%u, %u], j in [%u, %u]\n",
         p1.calls, p1.min_i, p1.max_i, p1.min_j, p1.max_j);

  if (p0.calls && p0.min_i == 0 && p0.max_i == 0 && p1.min_i == 1 && p1.max_i == 1)
    printf("VERDICT: BUG CONFIRMED -- callbacks receive the RULE INDEX as i\n");
  else if (p0.calls && p0.max_i > 1 && p0.min_i == p1.min_i && p0.max_i == p1.max_i)
    printf("VERDICT: documented behaviour -- callbacks receive the segment 5' end\n");
  else
    printf("VERDICT: inconclusive -- inspect the ranges above\n");

  free(structure);
  vrna_fold_compound_free(fc);
  return 0;
}

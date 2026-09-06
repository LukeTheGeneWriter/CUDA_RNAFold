/* Is fM2_real -- the extra matrix that circular folding needs -- the SAME
 * quantity the multibranch modular decomposition already computes for fML?
 *
 * fill_arrays() fills it per (i,j) with vrna_mfe_multibranch_m2_fast(), whose
 * body (mfe/mfe_multibranch.c:1143) is the ML_ML_ML decomposition
 *
 *     min over k of  fML[i,k] + fML[k+1,j]
 *
 * which is exactly what get_aux_arrays() documents DMLi[j] as holding while
 * fML itself is being computed. If the identity holds, a GPU sweep that already
 * computes that minimum for fML can supply fM2_real by STORING a value it
 * currently discards -- no new arithmetic in the hot kernel, only VRAM for one
 * more triangular matrix.
 *
 * This checks the identity directly on filled matrices, for every (i,j).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/datastructures/dp_matrices.h>

#ifndef INF
#define INF 10000000
#endif

static const char *seqs[] = {
  "GGGAAACCCAUAGCUAGCUAGCGGCCAAUUGGCCAUAUAUCGCGCAUAUAGGCUUAAGGCCUUAAGG",
  "AUGCAUGCAUGCAUGCGGGGCCCCAAAUUUGGGCCCCAUAUAUGCGCGCAUAUAUAGCUAGCUAGCUAGCAUCGAUCGAUCG",
  "GCGCGCAUAUAUCGCGCGAAAAGGGGCCCCUUUUAAAACCCCGGGGUUUUAAAUUUGCGCGCAUAUAUCGCGCGCAUAUAUA",
  NULL
};

static int
check_one(const char *seq, int verbose)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;
  char                  *structure;
  unsigned int          n, i, j, k;
  int                   *indx, *fML, *fM2_real, mismatches = 0, checked = 0;

  vrna_md_set_default(&md);
  md.circ = 1;                       /* this is what allocates fM2_real */

  n         = (unsigned int)strlen(seq);
  structure = (char *)vrna_alloc(sizeof(char) * (n + 1));
  fc        = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);

  (void)vrna_mfe(fc, structure);

  indx      = fc->jindx;
  fML       = fc->matrices->fML;
  fM2_real  = fc->matrices->fM2_real;

  if (!fM2_real) {
    printf("  n=%3u  fM2_real is NULL -- md.circ did not allocate it\n", n);
    free(structure);
    vrna_fold_compound_free(fc);
    return -1;
  }

  for (i = 1; i <= n; i++) {
    for (j = i + 1; j <= n; j++) {
      int expect = INF;

      /* min over k of fML[i,k] + fML[k+1,j], the ML_ML_ML decomposition */
      for (k = i + 1; k + 1 <= j && k <= j - 2; k++) {
        int a = fML[indx[k] + i];
        int b = fML[indx[j] + k + 1];
        if ((a != INF) && (b != INF) && (a + b < expect))
          expect = a + b;
      }

      checked++;
      if (fM2_real[indx[j] + i] != expect) {
        if (mismatches < 5 && verbose)
          printf("    (%u,%u): fM2_real=%d  min_k(fML+fML)=%d\n",
                 i, j, fM2_real[indx[j] + i], expect);
        mismatches++;
      }
    }
  }

  printf("  n=%3u  cells checked %6d   mismatches %6d   %s\n",
         n, checked, mismatches, mismatches ? "DIFFER" : "IDENTICAL");

  free(structure);
  vrna_fold_compound_free(fc);
  return mismatches;
}

int
main(void)
{
  int total = 0, i;

  printf("fM2_real  vs  min_k( fML[i,k] + fML[k+1,j] )   (md.circ = 1)\n\n");

  for (i = 0; seqs[i]; i++) {
    int m = check_one(seqs[i], 1);
    if (m < 0)
      return 2;
    total += m;
  }

  printf("\n%s\n", total == 0
         ? "VERDICT: fM2_real IS the modular decomposition the sweep already computes"
         : "VERDICT: fM2_real is NOT that quantity -- it needs its own evaluation");
  return total ? 1 : 0;
}

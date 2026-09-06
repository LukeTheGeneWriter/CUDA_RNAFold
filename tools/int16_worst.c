/*
 * Size the encoding against the WORST case, not a mid-length default.
 *
 * Two effects compound: the blocked spread grows slowly with n, and it grows
 * with lower temperature (stronger pairing -> larger energy magnitudes). 10 C
 * at 3000 nt already cost 46% over the 37 C default. The design margin has to
 * come from the corner, so this walks length x temperature x salt.
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/mfe/global.h"

#define IDX(i, j) ((j) * ((j) - 1) / 2 + (i))
#define INF_HALF  5000000

static unsigned long rs = 12345;
static int rnd4(void) {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return (int)((rs >> 33) & 3);
}

static unsigned int
spread(unsigned int n, double T, double salt, int B, unsigned int *worst_n)
{
  rs = 12345;
  char *seq = malloc(n + 1);
  static const char *A = "ACGU";
  for (unsigned int i = 0; i < n; i++) seq[i] = A[rnd4()];
  seq[n] = '\0';

  vrna_md_t md;
  vrna_md_set_default(&md);
  md.temperature = T;
  md.salt = salt;

  vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
  char *st = malloc(n + 1);
  (void)vrna_mfe(fc, st);
  const int *fML = fc->matrices->fML;

  unsigned int worst = 0;
  for (unsigned int j = 1; j <= n; j++)
    for (unsigned int base = 1; base <= j; base += B) {
      long cmin = LONG_MAX, cmax = LONG_MIN;
      unsigned int hi = base + B - 1; if (hi > j) hi = j;
      for (unsigned int i = base; i <= hi; i++) {
        long v = fML[IDX(i, j)];
        if (v >= INF_HALF) continue;
        if (v < cmin) cmin = v;
        if (v > cmax) cmax = v;
      }
      if (cmin != LONG_MAX && (unsigned long)(cmax - cmin) > worst)
        worst = (unsigned int)(cmax - cmin);
    }

  free(st); free(seq);
  vrna_fold_compound_free(fc);
  if (worst > *worst_n) *worst_n = worst;
  return worst;
}

int
main(void)
{
  const unsigned int lens[] = { 2000, 5601, 8001 };
  const double temps[] = { 37.0, 20.0, 10.0, 0.0 };
  const double salts[] = { 1.021, 0.05, 5.0 };
  unsigned int overall = 0;

  printf("worst blocked spread, B=128, over length x temperature x salt\n");
  printf("(int16 signed offset ceiling is 32766 with one value reserved for INF)\n\n");
  printf("%6s %7s %7s %10s %9s\n", "n", "T(C)", "salt", "spread", "headroom");

  for (unsigned li = 0; li < 3; li++)
    for (unsigned ti = 0; ti < 4; ti++)
      for (unsigned si = 0; si < 3; si++) {
        unsigned int w = spread(lens[li], temps[ti], salts[si], 128, &overall);
        printf("%6u %7.1f %7.3f %10u %8.1fx\n",
               lens[li], temps[ti], salts[si], w, 32766.0 / (w ? w : 1));
      }

  printf("\nWORST OVER ALL: %u  -> %.1fx headroom on a signed int16 offset\n",
         overall, 32766.0 / overall);
  printf("B=64 would roughly halve this again if more margin is ever wanted.\n");
  return 0;
}

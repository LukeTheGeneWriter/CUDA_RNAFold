/*
 * Two holes in the int16 scoping, both prompted by the existing value-range
 * analysis in hp_mb_loop.cu:873-903 -- the same class of reasoning that the
 * earlier int-widening fixes came out of.
 *
 * HOLE 1 -- "non-finite" is not one value.
 *   That comment records that the host's two INF guards are NOT symmetric: when
 *   fml_prev[j] is a real energy and en_i is INF, the host computes
 *   fml_prev[j] + 10000000, "a large positive number, NOT INF", and the INF test
 *   is `== INF` and never `>= INF` precisely so those survive as distinct
 *   values. My first probe lumped everything >= INF/2 into "INF" and dropped it,
 *   then concluded a single sentinel would do. If fML actually carries a SPREAD
 *   of near-INF values, one sentinel is wrong and the encoding needs more.
 *
 * HOLE 2 -- I measured the default energy model only.
 *   The same comment bounds the tropical sums by length*|MLbase| "with a
 *   non-default parameter file", i.e. it already identifies non-default
 *   parameters as the case that stretches ranges. Temperature and salt both
 *   change every energy; salt was only just enabled on the GPU path.
 *
 * Reports, for each model: the distinct values at or above INF/2, and the
 * worst-case blocked spread at B=128 among the strictly-finite ones.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/mfe/global.h"

#define IDX(i, j) ((j) * ((j) - 1) / 2 + (i))
#define INF_VAL   10000000
#define INF_HALF  5000000
#define BLK       128

static unsigned long rng_state = 12345;
static int rnd4(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (int)((rng_state >> 33) & 3);
}
static char *make_seq(unsigned int n) {
  static const char *A = "ACGU";
  char *s = malloc(n + 1);
  for (unsigned int i = 0; i < n; i++) s[i] = A[rnd4()];
  s[n] = '\0';
  return s;
}

static int cmpl(const void *a, const void *b) {
  long x = *(const long *)a, y = *(const long *)b;
  return (x > y) - (x < y);
}

static void
probe(const char *label, unsigned int n, double temperature, double salt)
{
  rng_state = 12345;                    /* same sequence for every model */
  char      *seq = make_seq(n);
  vrna_md_t md;
  vrna_md_set_default(&md);
  md.temperature = temperature;
  md.salt        = salt;

  vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
  char *st = malloc(n + 1);
  (void)vrna_mfe(fc, st);
  const int *fML = fc->matrices->fML;

  /* --- hole 1: how many DISTINCT values live at or above INF/2? --- */
  long  *big = malloc(sizeof(long) * ((size_t)n * (n + 1) / 2 + 2));
  size_t nbig = 0;
  long   bigmin = LONG_MAX, bigmax = LONG_MIN;
  size_t exact_inf = 0;

  /* --- hole 2: worst blocked spread among the strictly-finite --- */
  unsigned int worst = 0;

  for (unsigned int j = 1; j <= n; j++) {
    for (unsigned int base = 1; base <= j; base += BLK) {
      long cmin = LONG_MAX, cmax = LONG_MIN;
      unsigned int hi = base + BLK - 1; if (hi > j) hi = j;
      for (unsigned int i = base; i <= hi; i++) {
        long v = fML[IDX(i, j)];
        if (v >= INF_HALF) {
          if (v == INF_VAL) exact_inf++;
          else { big[nbig++] = v;
                 if (v < bigmin) bigmin = v;
                 if (v > bigmax) bigmax = v; }
          continue;
        }
        if (v < cmin) cmin = v;
        if (v > cmax) cmax = v;
      }
      if (cmin != LONG_MAX && (unsigned long)(cmax - cmin) > worst)
        worst = (unsigned int)(cmax - cmin);
    }
  }

  /* distinct count among the non-exact big values */
  size_t distinct = 0;
  if (nbig) {
    qsort(big, nbig, sizeof(long), cmpl);
    distinct = 1;
    for (size_t k = 1; k < nbig; k++)
      if (big[k] != big[k - 1]) distinct++;
  }

  printf("%-22s n=%-5u  exact INF %8zu | near-INF cells %8zu distinct %6zu",
         label, n, exact_inf, nbig, distinct);
  if (nbig)
    printf(" range [%ld, %ld]", bigmin, bigmax);
  printf("\n%-22s %-7s worst blocked spread (B=%d, finite only): %u  %s\n",
         "", "", BLK, worst, (worst <= 32766u) ? "fits int16" : "DOES NOT FIT");

  free(big); free(st); free(seq);
  vrna_fold_compound_free(fc);
}

int
main(void)
{
  const unsigned int n = 3000;

  printf("INF = %d.  A single sentinel is only sufficient if EVERY non-finite\n"
         "cell is exactly INF -- i.e. 'near-INF cells' is 0 below.\n\n", INF_VAL);

  probe("default",            n, 37.0, VRNA_MODEL_DEFAULT_SALT);
  probe("temperature 10 C",   n, 10.0, VRNA_MODEL_DEFAULT_SALT);
  probe("temperature 60 C",   n, 60.0, VRNA_MODEL_DEFAULT_SALT);
  probe("salt 0.05 M",        n, 37.0, 0.05);
  probe("salt 0.2 M",         n, 37.0, 0.2);
  probe("salt 5.0 M",         n, 37.0, 5.0);
  probe("10 C + salt 0.05",   n, 10.0, 0.05);
  return 0;
}

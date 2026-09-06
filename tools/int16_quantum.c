/*
 * Luke's idea: represent fML as small discrete symbols rather than raw energies,
 * with a table to recover full precision.
 *
 * Three separable questions, in increasing order of ambition:
 *
 *  Q1 THE QUANTUM. Every entry in stack37 is a multiple of 10 (energies are
 *     stored in 0.01 kcal/mol but the tables are quoted to 0.1). If the whole
 *     of fML shares a common divisor g, then dividing by g is free range --
 *     g=10 buys 3.3 bits, and at that point even an ABSOLUTE int16 might hold
 *     the full matrix with no blocks, no baselines and no offsets at all.
 *     This is the cheapest possible version of the idea and it needs no table.
 *
 *  Q2 THE DICTIONARY. If the number of DISTINCT values in fML is under 65536,
 *     a global symbol table indexes them exactly, at any range.
 *
 *  Q3 DEFERRED EVALUATION -- reported here only as arithmetic that must hold:
 *     how often does the value produced by a MIN2 come from adding two cells?
 *     A min-plus DP has to COMPARE numerically at every step, so symbols that
 *     cannot be added are not usable mid-sweep however compact they are.
 *
 * Measured under models that break the tidy cases: temperature rescaling and
 * the lxc log-extrapolation both produce truncated, non-round integers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static long gcd(long a, long b) { while (b) { long t = a % b; a = b; b = t; } return a < 0 ? -a : a; }
static int cmpl(const void *a, const void *b) {
  long x = *(const long *)a, y = *(const long *)b;
  return (x > y) - (x < y);
}

static void
probe(const char *label, unsigned int n, double T, double salt)
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

  size_t ncell = (size_t)n * (n + 1) / 2 + 2;
  long *vals = malloc(sizeof(long) * ncell);
  size_t nv = 0;
  long g = 0, vmin = LONG_MAX, vmax = LONG_MIN;

  for (unsigned int j = 1; j <= n; j++)
    for (unsigned int i = 1; i <= j; i++) {
      long v = fML[IDX(i, j)];
      if (v >= INF_HALF) continue;
      vals[nv++] = v;
      if (v) g = gcd(g, v);
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }

  qsort(vals, nv, sizeof(long), cmpl);
  size_t distinct = nv ? 1 : 0;
  for (size_t k = 1; k < nv; k++)
    if (vals[k] != vals[k - 1]) distinct++;

  long span = vmax - vmin;
  printf("%-20s n=%-5u  gcd=%-4ld  distinct=%-8zu  range [%ld, %ld] span %ld\n",
         label, n, g, distinct, vmin, vmax, span);
  printf("%-20s   absolute int16 after /gcd: span/g = %ld  -> %s\n", "",
         g ? span / g : span,
         (g && span / g <= 65534) ? "FITS" : "does not fit");
  printf("%-20s   dictionary of distinct values: %s (needs <= 65536)\n", "",
         (distinct <= 65536) ? "FITS" : "does not fit");

  free(vals); free(st); free(seq);
  vrna_fold_compound_free(fc);
}

int
main(void)
{
  printf("Q1 = is there a common divisor?   Q2 = is the value vocabulary small?\n\n");
  probe("default 37C",       2000, 37.0, VRNA_MODEL_DEFAULT_SALT);
  probe("default 37C",       5601, 37.0, VRNA_MODEL_DEFAULT_SALT);
  probe("T=10C (rescaled)",  2000, 10.0, VRNA_MODEL_DEFAULT_SALT);
  probe("T=23.5C (odd)",     2000, 23.5, VRNA_MODEL_DEFAULT_SALT);
  probe("salt 0.05",         2000, 37.0, 0.05);
  probe("T=10C + salt 0.05", 2000, 10.0, 0.05);
  return 0;
}

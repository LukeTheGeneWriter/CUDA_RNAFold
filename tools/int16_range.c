/*
 * The decisive first experiment for the int16 fml_j idea.
 *
 * modular_decomposition_kernel's dominant DRAM stream is fml_j: for an output
 * cell (i,j) the inner loop walks my_fML[indx[j] + k + 1] for k ascending, i.e.
 * CONTIGUOUSLY DOWN COLUMN j of the triangular matrix (see the host reference
 * modular_decomposition_ij in modular_decomposition.cu -- k1j increments while
 * indx[j] stays fixed).
 *
 * A naive int16 cannot hold these: INF is 10000000 and a 5601 nt MFE is around
 * -171840 in units of 0.01 kcal/mol, against an int16 range of +/-32767. The
 * only bit-exact route is an OFFSET encoding -- store each value as a 16-bit
 * delta from a per-column baseline, with a reserved sentinel for INF.
 *
 * That route lives or dies on one number: the SPREAD (max - min) of the finite
 * values within a single column. This measures it, at several lengths, with no
 * kernel change and no GPU -- fML is the same matrix on the CPU path.
 *
 * Reports what the encoding actually needs to survive:
 *   - the worst-case column spread, which must fit in 16 bits with room for a
 *     sentinel (so <= 32766 for unsigned-offset, <= 65534 if the baseline is
 *     the column min and the offset is unsigned)
 *   - how that scales with n, since 5601 nt is the hard case
 *   - the global range, to show why an ABSOLUTE int16 cannot work
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

static unsigned long
rng_state = 12345;

static int
rnd4(void)
{
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (int)((rng_state >> 33) & 3);
}

static char *
make_seq(unsigned int n)
{
  static const char *A = "ACGU";
  char *s = malloc(n + 1);
  for (unsigned int i = 0; i < n; i++)
    s[i] = A[rnd4()];
  s[n] = '\0';
  return s;
}

int
main(int argc, char **argv)
{
  const unsigned int lens[] = { 500, 1000, 2000, 3000, 5601 };
  const int nlen = (int)(sizeof(lens) / sizeof(lens[0]));
  const int reps = (argc > 1) ? atoi(argv[1]) : 2;

  printf("%6s %5s %12s %12s %10s %10s %10s %8s\n",
         "n", "rep", "global_min", "global_max", "worst_col", "p99_col",
         "median_col", "fits16?");

  for (int li = 0; li < nlen; li++) {
    unsigned int n = lens[li];

    for (int r = 0; r < reps; r++) {
      char      *seq = make_seq(n);
      vrna_md_t md;

      vrna_md_set_default(&md);

      vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
      char *st = malloc(n + 1);
      (void)vrna_mfe(fc, st);

      const int *fML = fc->matrices->fML;

      long gmin = LONG_MAX, gmax = LONG_MIN;
      unsigned int *spreads = calloc(n + 2, sizeof(unsigned int));
      unsigned int ns = 0;
      unsigned long finite_cells = 0, inf_cells = 0;

      for (unsigned int j = 1; j <= n; j++) {
        long cmin = LONG_MAX, cmax = LONG_MIN;
        for (unsigned int i = 1; i <= j; i++) {
          int v = fML[IDX(i, j)];
          if (v >= INF_HALF) { inf_cells++; continue; }
          finite_cells++;
          if (v < cmin) cmin = v;
          if (v > cmax) cmax = v;
          if (v < gmin) gmin = v;
          if (v > gmax) gmax = v;
        }
        if (cmin != LONG_MAX)
          spreads[ns++] = (unsigned int)(cmax - cmin);
      }

      /* order statistics over the per-column spreads */
      int cmp(const void *a, const void *b) {
        unsigned int x = *(const unsigned int *)a, y = *(const unsigned int *)b;
        return (x > y) - (x < y);
      }
      qsort(spreads, ns, sizeof(unsigned int), cmp);

      unsigned int worst  = ns ? spreads[ns - 1] : 0;
      unsigned int p99    = ns ? spreads[(ns * 99) / 100] : 0;
      unsigned int median = ns ? spreads[ns / 2] : 0;

      printf("%6u %5d %12ld %12ld %10u %10u %10u %8s\n",
             n, r, gmin, gmax, worst, p99, median,
             (worst <= 65534u) ? "yes" : "NO");

      if (li == nlen - 1 && r == 0)
        printf("       (finite cells %lu, INF cells %lu, %.1f%% finite)\n",
               finite_cells, inf_cells,
               100.0 * finite_cells / (finite_cells + inf_cells));

      free(spreads);
      free(st);
      free(seq);
      vrna_fold_compound_free(fc);
    }
  }

  printf("\nint16 absolute range is [-32768, 32767].\n");
  printf("A per-column OFFSET (value - column_min) needs the WORST COLUMN SPREAD\n");
  printf("to fit, leaving one value for the INF sentinel: <= 65534 as uint16,\n");
  printf("or <= 32766 if a signed offset from a mid-point baseline is preferred.\n");
  return 0;
}

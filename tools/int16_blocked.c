/*
 * Follow-up to int16_range.c, which showed a PER-COLUMN offset cannot work:
 * the worst column spread is 187270 at 5601 nt, against a uint16 ceiling of
 * 65534. That is unsurprising in hindsight -- column j holds fML[i][j] for
 * every i, i.e. segments of every length, so its spread is essentially the
 * whole energy range of the fold.
 *
 * But the kernel reads that column CONTIGUOUSLY, and consecutive entries differ
 * only by the cost of extending a segment by one base. So the question that
 * actually decides the idea is not the column spread but the spread within a
 * BLOCK of consecutive entries -- because the baseline can be per block.
 *
 * Cost of a blocked baseline: one extra 4-byte load per B entries (0.8% of the
 * stream at B=128) and an add in the inner loop. Decode is base[blk] + off[k].
 *
 * This measures the worst-case block spread for several B, at several lengths.
 * The encoding is bit-exact iff that worst case fits with a sentinel to spare.
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

int
main(int argc, char **argv)
{
  const unsigned int lens[] = { 1000, 2000, 3000, 5601, 8001 };
  const int BS[] = { 32, 64, 128, 256, 1024 };
  const int nB = (int)(sizeof(BS) / sizeof(BS[0]));
  const int nlen = (int)(sizeof(lens) / sizeof(lens[0]));

  printf("worst-case spread within a block of B consecutive fML entries down a column\n");
  printf("(INF cells are excluded -- they take the sentinel, not an offset)\n\n");
  printf("%6s", "n");
  for (int b = 0; b < nB; b++) printf(" %10s%-4d", "B=", BS[b]);
  printf("   %s\n", "verdict at B=128");

  for (int li = 0; li < nlen; li++) {
    unsigned int n = lens[li];
    char *seq = make_seq(n);
    vrna_md_t md;
    vrna_md_set_default(&md);
    vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);
    char *st = malloc(n + 1);
    (void)vrna_mfe(fc, st);
    const int *fML = fc->matrices->fML;

    unsigned int worst[16];
    memset(worst, 0, sizeof(worst));

    for (int b = 0; b < nB; b++) {
      const unsigned int B = (unsigned int)BS[b];
      unsigned int w = 0;
      for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int base = 1; base <= j; base += B) {
          long cmin = LONG_MAX, cmax = LONG_MIN;
          unsigned int hi = base + B - 1; if (hi > j) hi = j;
          for (unsigned int i = base; i <= hi; i++) {
            int v = fML[IDX(i, j)];
            if (v >= INF_HALF) continue;
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
          }
          if (cmin != LONG_MAX && (unsigned long)(cmax - cmin) > w)
            w = (unsigned int)(cmax - cmin);
        }
      }
      worst[b] = w;
    }

    printf("%6u", n);
    for (int b = 0; b < nB; b++) printf(" %14u", worst[b]);
    printf("   %s\n", (worst[2] <= 65534u) ? "FITS uint16" : "does not fit");

    free(st); free(seq);
    vrna_fold_compound_free(fc);
  }

  printf("\nOverhead of a blocked baseline: one int32 per B entries.\n");
  printf("  B=64  -> 4/64  =  6.2%% added, net stream 0.50+0.06 = 0.56x\n");
  printf("  B=128 -> 4/128 =  3.1%% added, net stream 0.50+0.03 = 0.53x\n");
  printf("  B=256 -> 4/256 =  1.6%% added, net stream 0.50+0.02 = 0.52x\n");
  return 0;
}

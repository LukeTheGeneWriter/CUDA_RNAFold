/*
 * fml_decode_equiv.c -- the bar for fetch_fML_one_H()'s host decode.
 *
 * The int16 fML stream is stored as a per-block baseline plus a short offset,
 * so the host has to widen it back to int32 at the device boundary. The decode
 * loop in modular_decomposition.cu did one *dependent* baseline load per cell:
 *
 *     dst[t] = hb[colb[j] + (i-1)/FML_BLK] + o;
 *
 * The baseline is constant across FML_BLK consecutive i, so that load can be
 * hoisted to once per block. This file holds both loops and proves they agree
 * cell for cell, then times them.
 *
 * WHY IT MATTERS: measured at 400 x 5601 on a T4, int16 wins 62.4 s in
 * modular_decomp and hands 34.8 s back, of which fetch_mx is 7.7-12.5 s
 * (STRESS272_RESULTS.md 15.3, 16.2). This loop is that component.
 *
 *   gcc -O2 -o fml_decode_equiv tools/fml_decode_equiv.c && ./fml_decode_equiv
 *
 * Exit 0 = identical on every shape. Non-zero = the decode changed an answer.
 *
 * The equality check goes RED: widening the block end by one
 * (`end + 1`) gives 1151 mismatches of 1500 shapes. Verified 2026-09-10 before
 * the green was believed.
 *
 * NOTE ON TERMINATION: the blocked loop advances `i` only inside the inner
 * for, so it terminates only while `end >= i`. That holds because `end` is
 * derived from `i`'s own block: ((i-1)/BLK + 1)*BLK > i-1, hence >= i. Do not
 * change how `end` is computed without re-checking that -- a red-team mutation
 * that shifted the block index by 7 instead of 6 turned this into an infinite
 * loop rather than a wrong answer.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FML_BLK    64
#define FML_INF16  ((short) 32767)
#define INF        10000000

static double
now(void)
{
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}


/* The loop as it stood before 2026-09-10. Kept verbatim: it is the reference,
 * and any future rewrite is judged against it, not against its successor. */
static void
decode_reference(int          *dst,
                 const short  *h16,
                 const int    *hb,
                 const size_t *colb,
                 size_t        cells,
                 int           n)
{
  for (int j = 1; j <= n; j++)
    for (int i = 1; i <= j; i++) {
      const size_t t = (size_t) j * (j - 1) / 2 + i;

      if (t >= cells)
        continue;

      const short o = h16[t];

      dst[t] = (o == FML_INF16) ? INF
                                : hb[colb[j] + (size_t) ((i - 1) / FML_BLK)] + (int) o;
    }
}


/* One baseline load per FML_BLK cells instead of one per cell. */
static void
decode_blocked(int          *dst,
               const short  *h16,
               const int    *hb,
               const size_t *colb,
               size_t        cells,
               int           n)
{
  for (int j = 1; j <= n; j++) {
    const size_t row = (size_t) j * (j - 1) / 2;

    if (row >= cells)
      break;                      /* rows only grow: nothing after this is ours */

    int hi = j;
    if (row + (size_t) hi >= cells)
      hi = (int) (cells - row) - 1;

    const int   *b   = hb + colb[j];
    const short *src = h16 + row;
    int         *out = dst + row;

    for (int i = 1; i <= hi; ) {
      const int blk  = (i - 1) / FML_BLK;
      const int base = b[blk];
      int       end  = (blk + 1) * FML_BLK;   /* last i sharing this baseline */

      if (end > hi)
        end = hi;

      for (; i <= end; i++) {
        const short o = src[i];

        out[i] = (o == FML_INF16) ? INF : base + (int) o;
      }
    }
  }
}


static size_t *
make_colb(int n)
{
  size_t *colb = malloc((size_t) (n + 2) * sizeof(size_t));

  colb[0] = colb[1] = 0;
  for (int j = 1; j <= n; j++)
    colb[j + 1] = colb[j] + (size_t) ((j + FML_BLK - 1) / FML_BLK);

  return colb;
}


/* Every shape, including the ragged ones where `cells` cuts the final row --
 * that truncation is the only reason the reference has a per-cell bound test,
 * so it is exactly where a rewrite is most likely to differ. */
static int
check(void)
{
  int bad = 0, shapes = 0;

  for (int n = 1; n <= 300; n++) {
    const size_t full = ((size_t) (n + 1) * (n + 2)) / 2;
    const size_t trials[5] = {
      full,
      full > 2 ? full - 1 : full,
      full > 3 ? full - 2 : full,
      full / 2 + 1,
      full > (size_t) n ? full - n : full
    };

    for (int k = 0; k < 5; k++) {
      const size_t cells = trials[k];

      if (cells == 0)
        continue;

      shapes++;

      size_t      *colb = make_colb(n);
      const size_t bn   = colb[n + 1] ? colb[n + 1] : 1;
      short       *h    = malloc(full * sizeof(short));
      int         *hb   = malloc(bn * sizeof(int));
      int         *a    = calloc(full, sizeof(int));
      int         *b    = calloc(full, sizeof(int));

      srand(n * 17 + k);
      for (size_t t = 0; t < full; t++)
        h[t] = (rand() % 10 < 3) ? FML_INF16 : (short) (rand() % 20000 - 10000);
      for (size_t t = 0; t < bn; t++)
        hb[t] = -(rand() % 30000);

      decode_reference(a, h, hb, colb, cells, n);
      decode_blocked(b, h, hb, colb, cells, n);

      if (memcmp(a, b, full * sizeof(int)) != 0) {
        if (bad < 5)
          printf("  MISMATCH n=%d cells=%zu (full=%zu)\n", n, cells, full);
        bad++;
      }

      free(colb); free(h); free(hb); free(a); free(b);
    }
  }

  printf("%d shapes compared, %d mismatches -> %s\n",
         shapes, bad, bad ? "FAIL" : "PASS");

  return bad;
}


static void
bench(int n, int records)
{
  const size_t cells = ((size_t) (n + 1) * (n + 2)) / 2;
  size_t      *colb  = make_colb(n);
  const size_t bn    = colb[n + 1] ? colb[n + 1] : 1;
  short       *h16   = malloc(cells * sizeof(short));
  int         *hb    = malloc(bn * sizeof(int));
  int         *dst   = malloc(cells * sizeof(int));

  if (!h16 || !hb || !dst) {
    printf("(skipping bench: needs ~%zu MB)\n", cells * 6 / (1 << 20));
    return;
  }

  srand(1);
  for (size_t t = 0; t < cells; t++)
    h16[t] = (rand() % 10 < 3) ? FML_INF16 : (short) (rand() % 20000 - 10000);
  for (size_t t = 0; t < bn; t++)
    hb[t] = -(rand() % 30000);

  printf("\nn=%d, cells=%zu, %d records\n", n, cells, records);

  double ref = 0.0;

  for (int k = 0; k < 2; k++) {
    double best = 1e30;

    for (int rep = 0; rep < 3; rep++) {
      const double t0 = now();

      if (k == 0)
        decode_reference(dst, h16, hb, colb, cells, n);
      else
        decode_blocked(dst, h16, hb, colb, cells, n);

      const double dt = now() - t0;
      if (dt < best)
        best = dt;
    }

    if (k == 0)
      ref = best;

    printf("  %-10s %6.2f ns/cell  %6.1f worker-s over %d records  (%.2fx)\n",
           k ? "blocked" : "reference",
           best * 1e9 / cells, best * records, records, ref / best);
  }

  /* int16 already saves half the PCIe bytes; the decode is what it spends the
   * saving on. Quote both or the comparison is not honest. */
  printf("  int16 moves %zu MB/record instead of %zu MB: ~%.1f s saved over\n"
         "  %d records at ~6 GB/s, against the decode cost above.\n",
         cells * 2 / (1 << 20), cells * 4 / (1 << 20),
         (double) cells * 2.0 * records / 6.0e9, records);

  free(colb); free(h16); free(hb); free(dst);
}


int
main(int argc, char **argv)
{
  const int rc = check();

  if (argc > 1 && strcmp(argv[1], "--bench") == 0)
    bench(argc > 2 ? atoi(argv[2]) : 5601, argc > 3 ? atoi(argv[3]) : 400);

  return rc != 0;
}

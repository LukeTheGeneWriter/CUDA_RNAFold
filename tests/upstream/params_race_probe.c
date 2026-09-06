/* Probe: is the SPEEDUP_PARAMS cache in params/params.c (p_pre / p_pre_init,
 * lines 100-112, 155-180) a live data race that returns WRONG energy parameters
 * to a caller, not merely a benign one?
 *
 * SPEEDUP_PARAMS is #define'd to 1 unconditionally at params/params.c:23 -- there
 * is no configure switch to turn it off -- and vrna_params() is reached from
 * vrna_fold_compound(), so upstream's own threaded RNAfold -j hits this path.
 *
 * Method: build a single-threaded reference parameter table for each of four
 * temperatures, then hammer vrna_params() from N threads with those same four
 * model details and compare every returned table against its reference,
 * byte-for-byte, skipping only the leading `id` field (documented to differ).
 * Any mismatch is a silently wrong energy model handed to a caller.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/params/basic.h>

#ifndef NTEMP
#define NTEMP    4
#endif
#define NTHREAD  8
#define NITER    20000

static const double temps[4] = { 20.0, 30.0, 37.0, 50.0 };
static vrna_param_t *ref[4];

/* everything after the `id` field */
static const size_t off = offsetof(struct vrna_param_s, stack);
static size_t cmp_len;

static void
md_for(vrna_md_t *md, int k)
{
  vrna_md_set_default(md);
  md->temperature = temps[k];
}

struct result {
  unsigned long checked, mismatched, first_bad_temp;
};

static void *
worker(void *arg)
{
  struct result *r = (struct result *)arg;
  unsigned int seed = (unsigned int)(uintptr_t)arg;

  r->checked = r->mismatched = 0;
  r->first_bad_temp = (unsigned long)-1;

  for (int it = 0; it < NITER; it++) {
    int k = rand_r(&seed) % NTEMP;
    vrna_md_t md;
    md_for(&md, k);

    vrna_param_t *p = vrna_params(&md);
    if (!p)
      continue;

    r->checked++;
    if (memcmp((char *)p + off, (char *)ref[k] + off, cmp_len) != 0) {
      r->mismatched++;
      if (r->first_bad_temp == (unsigned long)-1)
        r->first_bad_temp = (unsigned long)temps[k];
    }
    free(p);
  }
  return NULL;
}

int
main(void)
{
  pthread_t th[NTHREAD];
  struct result res[NTHREAD];
  unsigned long checked = 0, bad = 0;

  cmp_len = sizeof(struct vrna_param_s) - off;

  /* single-threaded references, built before any thread exists */
  for (int k = 0; k < NTEMP; k++) {
    vrna_md_t md;
    md_for(&md, k);
    ref[k] = vrna_params(&md);
  }
  /* sanity: the four references must differ from one another */
  for (int k = 1; k < NTEMP; k++) {
    if (memcmp((char *)ref[k] + off, (char *)ref[0] + off, cmp_len) == 0) {
      fprintf(stderr, "setup error: T=%.0f table equals T=%.0f table\n",
              temps[k], temps[0]);
      return 2;
    }
  }

  for (int t = 0; t < NTHREAD; t++)
    pthread_create(&th[t], NULL, worker, &res[t]);
  for (int t = 0; t < NTHREAD; t++)
    pthread_join(th[t], NULL);

  for (int t = 0; t < NTHREAD; t++) {
    checked += res[t].checked;
    bad     += res[t].mismatched;
  }

  printf("threads=%d iterations/thread=%d temperatures=%d\n",
         NTHREAD, NITER, NTEMP);
  printf("parameter tables checked : %lu\n", checked);
  printf("tables WRONG vs reference: %lu (%.4f%%)\n",
         bad, checked ? 100.0 * (double)bad / (double)checked : 0.0);
  if (bad)
    printf("VERDICT: RACE CONFIRMED -- vrna_params() returned a parameter table "
           "that does not match the model details it was asked for\n");
  else
    printf("VERDICT: no mismatch observed in this run (absence of evidence only; "
           "the unsynchronised memcpy at params.c:170 is still a data race)\n");

  return bad ? 1 : 0;
}

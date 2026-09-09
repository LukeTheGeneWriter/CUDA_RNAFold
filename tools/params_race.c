/* Does vrna_params() still corrupt tables when threads disagree about the model?
 *
 * This is the bar for the Defect B fix. RNAfold cannot exercise it -- it gives
 * every record one model, which is exactly the case the race leaves LOOKING
 * fine (measured 0 of 160 000 tables wrong with identical model details, 25 140
 * of 160 000 = 15.7 % when they differ). So the failing case needs a program of
 * its own, and without one "we fixed the race" is an unverified claim.
 *
 * Method: T threads each repeatedly ask for parameters at their OWN temperature
 * and compare what comes back against a table computed serially for that same
 * temperature. Any mismatch is the cache handing one thread another thread's
 * model.
 *
 * Build:
 *   gcc -O2 -pthread -o params_race tools/params_race.c -I<tree>/src \
 *       <tree>/src/ViennaRNA/.libs/libRNA.a -lm -lstdc++ -lgomp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <ViennaRNA/model.h>
#include <ViennaRNA/params/basic.h>

#define NTHREAD 8
#define NITER   2000

static double temps[NTHREAD] = { 20.0, 25.0, 30.0, 37.0, 42.0, 50.0, 55.0, 60.0 };

struct ref {
  vrna_param_t *p;
  long          checked, wrong;
};

static struct ref refs[NTHREAD];

static void *
worker(void *arg)
{
  const long     idx = (long)arg;
  vrna_md_t      md;
  int            it;

  for (it = 0; it < NITER; it++) {
    vrna_param_t *p;

    vrna_md_set_default(&md);
    md.temperature = temps[idx];

    p = vrna_params(&md);

    /* stack[][] is the table the original measurement counted, and it is
     * temperature-scaled, so a swapped model shows up in it immediately. */
    refs[idx].checked++;
    if (memcmp(p->stack, refs[idx].p->stack, sizeof(p->stack)) != 0)
      refs[idx].wrong++;

    /* the model that came back must be the one we asked for */
    if (p->model_details.temperature != temps[idx])
      refs[idx].wrong++;

    free(p);
  }

  return NULL;
}

int
main(void)
{
  pthread_t th[NTHREAD];
  long      i;
  long      total = 0, bad = 0;

  /* Serial references first, one per temperature, with no threads running. */
  for (i = 0; i < NTHREAD; i++) {
    vrna_md_t md;
    vrna_md_set_default(&md);
    md.temperature = temps[i];
    refs[i].p      = vrna_params(&md);
    refs[i].checked = refs[i].wrong = 0;
  }

  for (i = 0; i < NTHREAD; i++)
    if (pthread_create(&th[i], NULL, worker, (void *)i) != 0) {
      fprintf(stderr, "pthread_create failed\n");
      return 2;
    }

  for (i = 0; i < NTHREAD; i++)
    pthread_join(th[i], NULL);

  for (i = 0; i < NTHREAD; i++) {
    printf("  %5.1f C   %6ld checked   %6ld wrong\n",
           temps[i], refs[i].checked, refs[i].wrong);
    total += refs[i].checked;
    bad   += refs[i].wrong;
    free(refs[i].p);
  }

  printf("\n%ld of %ld wrong (%.2f%%)\n", bad, total, 100.0 * bad / total);
  printf("%s\n", bad ? "*** RACE PRESENT ***" : "clean");
  return bad ? 1 : 0;
}

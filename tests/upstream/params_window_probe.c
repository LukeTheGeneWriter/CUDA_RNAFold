/* Probe C: the SPEEDUP_PARAMS cache-HIT path at params/params.c:157-160 WRITES
 * three fields into the shared static `p_pre` before comparing:
 *
 *     p_pre.model_details.window_size   = md_p->window_size;
 *     p_pre.model_details.min_loop_size = md_p->min_loop_size;
 *     p_pre.model_details.max_bp_span   = md_p->max_bp_span;
 *
 * so two threads that want the SAME energy model but different window settings
 * (RNAfold vs RNALfold/RNAplfold in one process, or any windowed sweep) write
 * over each other and can be handed the other thread's values inside their own
 * parameter set. This costs nothing at the level of the energy tables, so probe
 * B (identical md) cannot see it -- it shows up only in model_details.
 *
 * Each thread asks for its own window_size / max_bp_span and checks that the
 * table it gets back carries the values it asked for.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/params/basic.h>

#ifndef NTHREAD
#define NTHREAD  8
#endif
#define NITER    20000

struct result {
  int           window;
  unsigned long checked, wrong_window, wrong_span;
};

static void *
worker(void *arg)
{
  struct result *r = (struct result *)arg;

  for (int it = 0; it < NITER; it++) {
    vrna_md_t md;
    vrna_md_set_default(&md);
    md.window_size = r->window;
    md.max_bp_span = r->window;

    vrna_param_t *p = vrna_params(&md);
    if (!p)
      continue;

    r->checked++;
    if (p->model_details.window_size != r->window)
      r->wrong_window++;
    if (p->model_details.max_bp_span != r->window)
      r->wrong_span++;

    free(p);
  }
  return NULL;
}

int
main(void)
{
  pthread_t th[NTHREAD];
  struct result res[NTHREAD];
  unsigned long checked = 0, bw = 0, bs = 0;

  /* prime the cache single-threaded, as any real program would */
  vrna_md_t md;
  vrna_md_set_default(&md);
  free(vrna_params(&md));

  for (int t = 0; t < NTHREAD; t++) {
    memset(&res[t], 0, sizeof(res[t]));
    res[t].window = 100 + 10 * t;   /* every thread wants a different window */
    pthread_create(&th[t], NULL, worker, &res[t]);
  }
  for (int t = 0; t < NTHREAD; t++)
    pthread_join(th[t], NULL);

  for (int t = 0; t < NTHREAD; t++) {
    checked += res[t].checked;
    bw      += res[t].wrong_window;
    bs      += res[t].wrong_span;
  }

  printf("threads=%d iterations/thread=%d, each thread asks for its own window\n",
         NTHREAD, NITER);
  printf("parameter sets checked        : %lu\n", checked);
  printf("wrong model_details.window_size: %lu (%.4f%%)\n",
         bw, checked ? 100.0 * (double)bw / (double)checked : 0.0);
  printf("wrong model_details.max_bp_span: %lu (%.4f%%)\n",
         bs, checked ? 100.0 * (double)bs / (double)checked : 0.0);
  printf("%s\n", (bw || bs)
         ? "VERDICT: CONFIRMED -- a caller received another thread's window settings"
         : "VERDICT: no mismatch observed in this run");
  return (bw || bs) ? 1 : 0;
}

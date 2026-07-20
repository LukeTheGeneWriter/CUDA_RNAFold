/* New Jul 2026: see RNAfold_cpu_queue.h for design notes. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "ViennaRNA/data_structures.h"
#include "ViennaRNA/params.h"
#include "ViennaRNA/mfe.h"
#include "ViennaRNA/utils.h"

#include "RNAfold_cpu_queue.h"

/* static (internal linkage) -- safe to duplicate across translation units,
 * same pattern src/bin/RNAfold.c itself already uses for this .inc file. */
#include "ViennaRNA/color_output.inc"

#define QUEUE_CAPACITY 64

typedef struct {
  char *seq_id;
  char *orig_sequence;
  char *rec_sequence;
} rnafold_cpu_job_t;

static rnafold_cpu_job_t queue_jobs[QUEUE_CAPACITY];
static int    queue_head       = 0;
static int    queue_tail       = 0;
static int    queue_count      = 0;
static int    queue_shutdown   = 0;
static int    queue_n_threads  = 0;
static long   queue_n_folded   = 0;

static pthread_mutex_t queue_mutex     = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  queue_not_empty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  queue_not_full  = PTHREAD_COND_INITIALIZER;
/* Serializes worker output so concurrent threads don't interleave/garble
 * each other's print_fasta_header()/print_structure() calls. */
static pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

static vrna_md_t  queue_md;     /* snapshot taken at init, read-only after */
static FILE       *queue_output = NULL;
static pthread_t  *queue_threads = NULL;

static void
fold_and_print_one(const rnafold_cpu_job_t *job) {
  vrna_fold_compound_t *vc = vrna_fold_compound(job->rec_sequence, &queue_md, VRNA_OPTION_MFE);
  char *structure = (char *) vrna_alloc(sizeof(char) * (vc->length + 1));

  const float min_en = vrna_mfe_cpu(vc, structure);

  if(queue_output) {
    pthread_mutex_lock(&print_mutex);
    print_fasta_header(queue_output, job->seq_id);
    fprintf(queue_output, "%s\n", job->orig_sequence);
    char *msg = vrna_strdup_printf(" (%6.2f)", min_en);
    print_structure(queue_output, structure, msg);
    free(msg);
    (void) fflush(queue_output);
    pthread_mutex_unlock(&print_mutex);
  }

  free(structure);
  vrna_fold_compound_free(vc);
}

static void *
worker_main(void *arg) {
  (void) arg;
  for(;;) {
    pthread_mutex_lock(&queue_mutex);
    while(queue_count == 0 && !queue_shutdown)
      pthread_cond_wait(&queue_not_empty, &queue_mutex);

    if(queue_count == 0 && queue_shutdown) {
      pthread_mutex_unlock(&queue_mutex);
      break;
    }

    rnafold_cpu_job_t job = queue_jobs[queue_head];
    queue_head = (queue_head + 1) % QUEUE_CAPACITY;
    queue_count--;
    pthread_cond_signal(&queue_not_full);
    pthread_mutex_unlock(&queue_mutex);

    fold_and_print_one(&job);
    __sync_add_and_fetch(&queue_n_folded, 1);

    free(job.seq_id);
    free(job.orig_sequence);
    free(job.rec_sequence);
  }
  return NULL;
}

void
rnafold_cpu_queue_init(int n_threads, const vrna_md_t *md, FILE *output) {
  queue_n_threads = (n_threads > 0) ? n_threads : 0;
  if(queue_n_threads == 0) return;

  queue_md     = *md; /* snapshot -- workers only ever read this copy */
  queue_output = output;

  queue_threads = (pthread_t *) vrna_alloc(sizeof(pthread_t) * queue_n_threads);
  for(int t = 0; t < queue_n_threads; t++) {
    if(pthread_create(&queue_threads[t], NULL, worker_main, NULL) != 0) {
      fprintf(stderr, "RNAfold_cpu_queue: pthread_create failed for worker %d, continuing with fewer threads\n", t);
      queue_n_threads = t; /* only this many actually started */
      break;
    }
  }
}

int
rnafold_cpu_queue_submit(const char *seq_id, const char *orig_sequence, const char *rec_sequence) {
  if(queue_n_threads == 0) return 0;

  rnafold_cpu_job_t job;
  job.seq_id        = seq_id        ? strdup(seq_id)        : NULL;
  job.orig_sequence = orig_sequence ? strdup(orig_sequence) : strdup("");
  job.rec_sequence  = strdup(rec_sequence);

  pthread_mutex_lock(&queue_mutex);
  while(queue_count == QUEUE_CAPACITY)
    pthread_cond_wait(&queue_not_full, &queue_mutex);

  queue_jobs[queue_tail] = job;
  queue_tail = (queue_tail + 1) % QUEUE_CAPACITY;
  queue_count++;
  pthread_cond_signal(&queue_not_empty);
  pthread_mutex_unlock(&queue_mutex);

  return 1;
}

void
rnafold_cpu_queue_shutdown(void) {
  if(queue_n_threads == 0) return;

  pthread_mutex_lock(&queue_mutex);
  queue_shutdown = 1;
  pthread_cond_broadcast(&queue_not_empty);
  pthread_mutex_unlock(&queue_mutex);

  for(int t = 0; t < queue_n_threads; t++)
    pthread_join(queue_threads[t], NULL);

  free(queue_threads);
  queue_threads = NULL;

  fprintf(stderr, "RNAfold_cpu_queue: %ld sequence(s) folded on %d CPU thread(s)\n",
          queue_n_folded, queue_n_threads);
}

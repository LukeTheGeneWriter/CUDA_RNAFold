/*
 *  Resource budgets: how much host RAM, device memory and how many threads a run
 *  may take (PORT_BUDGETS.md, signed off 2026-10-13).
 *
 *  No CUDA here: the file is built into the CUDA library and the CPU-only one
 *  alike, because a GPU-less laptop is exactly where a RAM and thread budget
 *  matters most.
 *
 *  A budget is a number (MB for RAM and VRAM, a count for threads) or a fraction
 *  of the machine's TOTAL: quarter, half, threequarter, full. It is an UPPER
 *  LIMIT. Every consumer keeps the soft limit it has always had (half of
 *  MemAvailable for host builds, 0.85 x free VRAM), so a run still adapts to what
 *  is free right now; the budget only lowers the ceiling. `full` adds no ceiling
 *  at all, which is what makes it today's behaviour byte for byte.
 *
 *  Precedence, per resource: vrna_budget_set() (the Python API), then the
 *  environment (RNA_RAM_BUDGET, RNA_VRAM_BUDGET, RNA_THREAD_BUDGET, read at every
 *  query so a script that changes os.environ is obeyed from its next call), then
 *  the program's default (vrna_budget_default(): RNAfold `full`, Python `half`).
 *  An explicit per-consumer setting (`-j`, RNA_BACKTRACK_THREADS, ...) wins over
 *  any budget, with a warning; that is the consumer's job, not this file's.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <unistd.h>

#include "ViennaRNA/mfe/cuda/engine.h"

typedef struct {
  int     kind;    /* 0 unset, 1 fraction, 2 absolute (MB, or a count for threads) */
  double  value;
} budget_spec_t;

static budget_spec_t  g_set[3];                                  /* vrna_budget_set() */
static budget_spec_t  g_default[3] = { { 1, 1.0 }, { 1, 1.0 }, { 1, 1.0 } };   /* full */
static const char     *g_env[3]  = { "RNA_RAM_BUDGET", "RNA_VRAM_BUDGET", "RNA_THREAD_BUDGET" };
static const char     *g_name[3] = { "RAM", "VRAM", "threads" };


/* "quarter" | "half" | "threequarter" | "full" (any case, also "three-quarter",
 * "three_quarter"), or a positive number. 1 on success. */
static int
budget_parse(const char     *s,
             budget_spec_t  *out)
{
  char    *end;
  double  v;

  if ((s == NULL) || (s[0] == '\0'))
    return 0;

  if (!strcasecmp(s, "quarter"))
    *out = (budget_spec_t){ 1, 0.25 };
  else if (!strcasecmp(s, "half"))
    *out = (budget_spec_t){ 1, 0.5 };
  else if ((!strcasecmp(s, "threequarter")) || (!strcasecmp(s, "three-quarter")) ||
           (!strcasecmp(s, "three_quarter")))
    *out = (budget_spec_t){ 1, 0.75 };
  else if (!strcasecmp(s, "full"))
    *out = (budget_spec_t){ 1, 1.0 };
  else {
    v = strtod(s, &end);
    if ((end == s) || (*end != '\0') || !(v > 0.0))
      return 0;

    *out = (budget_spec_t){ 2, v };
  }

  return 1;
}


static int
budget_valid(int which)
{
  return (which >= VRNA_BUDGET_RAM) && (which <= VRNA_BUDGET_THREADS);
}


/* The spec in force for one resource: set > environment > default. An
 * unparseable environment value is reported once and ignored. */
static budget_spec_t
budget_spec(int which)
{
  static int    warned[3];
  budget_spec_t s;
  const char    *e;

  if (g_set[which].kind)
    return g_set[which];

  e = getenv(g_env[which]);
  if ((e != NULL) && (e[0] != '\0')) {
    if (budget_parse(e, &s))
      return s;

    if (!warned[which]) {
      warned[which] = 1;
      fprintf(stderr,
              "%-24s ignoring %s=%s (want quarter, half, threequarter, full or a "
              "positive number)\n", "mfe/cuda/budget.c", g_env[which], e);
    }
  }

  return g_default[which];
}


int
vrna_budget_set(int         which,
                const char  *spec)
{
  budget_spec_t s;

  if (!budget_valid(which))
    return 0;

  if ((spec == NULL) || (spec[0] == '\0')) {
    g_set[which].kind = 0;      /* back to the environment / the default */
    return 1;
  }

  if (!budget_parse(spec, &s))
    return 0;

  g_set[which] = s;
  return 1;
}


int
vrna_budget_default(const char *spec)
{
  budget_spec_t s;
  int           k;

  if (!budget_parse(spec, &s))
    return 0;

  for (k = 0; k < 3; k++)
    g_default[k] = s;

  return 1;
}


/* /proc/meminfo, in bytes; 0 when it cannot be read. */
static size_t
budget_meminfo(const char *key)
{
  FILE          *f;
  char          line[256];
  unsigned long v;
  size_t        kb  = 0;
  size_t        len = strlen(key);

  if ((f = fopen("/proc/meminfo", "r")) == NULL)
    return 0;

  while (fgets(line, sizeof(line), f))
    if ((!strncmp(line, key, len)) && (line[len] == ':') &&
        (sscanf(line + len + 1, "%lu", &v) == 1)) {
      kb = (size_t)v;
      break;
    }

  fclose(f);
  return kb * (size_t)1024;
}


size_t
vrna_budget_host_total(void)
{
  size_t t = budget_meminfo("MemTotal");

#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
  if (t == 0) {
    long p = sysconf(_SC_PHYS_PAGES), s = sysconf(_SC_PAGESIZE);
    if ((p > 0) && (s > 0))
      t = (size_t)p * (size_t)s;
  }
#endif

  return t;
}


size_t
vrna_budget_host_available(void)
{
  const char  *e = getenv("RNA_HOST_AVAIL_MB");   /* the existing test hook */

  if ((e != NULL) && (e[0] != '\0') && (atol(e) >= 0))
    return (size_t)atol(e) * (size_t)1048576;

  return budget_meminfo("MemAvailable");
}


int
vrna_budget_hw_threads(void)
{
  long n = sysconf(_SC_NPROCESSORS_ONLN);

  return (n > 0) ? (int)n : 1;
}


/* A byte ceiling from a spec against a total: SIZE_MAX means "no ceiling" (full, or an
 * unknown total under a fraction -- a budget must not invent a limit it cannot see). */
static size_t
budget_bytes(budget_spec_t  s,
             size_t         total)
{
  if (s.kind == 2)
    return (size_t)(s.value * 1048576.0);

  if (s.value >= 1.0)
    return SIZE_MAX;

  if (total == 0)
    return SIZE_MAX;

  return (size_t)(s.value * (double)total);
}


size_t
vrna_budget_ram(void)
{
  return budget_bytes(budget_spec(VRNA_BUDGET_RAM), vrna_budget_host_total());
}


size_t
vrna_budget_vram(size_t device_total)
{
  return budget_bytes(budget_spec(VRNA_BUDGET_VRAM), device_total);
}


int
vrna_budget_threads(void)
{
  const budget_spec_t s   = budget_spec(VRNA_BUDGET_THREADS);
  const int           hw  = vrna_budget_hw_threads();
  int                 n;

  if (s.kind == 2)
    n = (int)s.value;
  else
    n = (int)(s.value * (double)hw);

  if (n < 1)
    n = 1;

  return n;
}


int
vrna_budget_is_full(int which)
{
  budget_spec_t s;

  if (!budget_valid(which))
    return 1;

  s = budget_spec(which);
  return (s.kind == 1) && (s.value >= 1.0);
}


static void
budget_describe(char          *buf,
                size_t        n,
                int           which,
                size_t        total)
{
  const budget_spec_t s = budget_spec(which);
  const char          *word = (s.value == 0.25) ? "quarter" : (s.value == 0.5) ? "half" :
                              (s.value == 0.75) ? "threequarter" : "full";

  if (which == VRNA_BUDGET_THREADS) {
    if (s.kind == 2)
      snprintf(buf, n, "threads %d (set)", vrna_budget_threads());
    else
      snprintf(buf, n, "threads %d (%s of %d)", vrna_budget_threads(), word,
               vrna_budget_hw_threads());

    return;
  }

  if ((s.kind == 1) && (s.value >= 1.0))
    snprintf(buf, n, "%s full", g_name[which]);
  else if (s.kind == 2)
    snprintf(buf, n, "%s %.0f MB (set)", g_name[which], s.value);
  else if (total == 0)
    snprintf(buf, n, "%s %s (total unknown: no ceiling)", g_name[which], word);
  else
    snprintf(buf, n, "%s %.0f MB (%s of %.0f)", g_name[which],
             s.value * (double)total / 1048576.0, word, (double)total / 1048576.0);
}


/* ---------------------------------------------------------------- the host model (B3)
 *
 * FITTED, not asserted (PORT_BUDGETS.md 8.4, laptop, 2026-10-13): peak RSS at uniform
 * lengths 600 / 1200 / 2800 / 5601 with 1 / 4 / 12 backtrack workers and with half the
 * records, plus four-chunk runs with each pipeline toggled. These bounds sit above every
 * measured arm, by 0-13 %; a budget's model must never predict under. */
#define BUDGET_MIB ((size_t)1048576)

size_t
vrna_budget_compound_bytes(size_t length)
{
  const size_t L = length + 1;

  return (3 * L * L) / 2 + 192 * L + BUDGET_MIB / 2;
}


size_t
vrna_budget_worker_bytes(size_t  length,
                         int     circ)
{
  const size_t L = length + 1;

  /* a c/fML pair (and fM2 with -c) sized to the chunk's longest record, its trace
   * stack, and the thread's own stack and allocator arena (the 8 MiB) */
  return 8 * BUDGET_MIB + (circ ? (15 * L * L) / 2 : 5 * L * L);
}


size_t
vrna_budget_hostfold_bytes(size_t length)
{
  const size_t L = length + 1;

  return 5 * L * L + 1024 * L;
}


size_t
vrna_budget_base_bytes(int gpu)
{
  return (gpu ? 100 : 16) * BUDGET_MIB;
}


/* Host memory the DRIVER holds beside the chunk being backtracked: the next chunk a build
 * pipeline is building, a chunk retained for the backtrack pipeline. Set and cleared by
 * whoever holds it; the worker decision subtracts the sum. */
static size_t g_held[3];

void
vrna_budget_held(int     slot,
                 size_t  bytes)
{
  if ((slot >= 0) && (slot < 3))
    g_held[slot] = bytes;
}


size_t
vrna_budget_held_total(void)
{
  return g_held[0] + g_held[1] + g_held[2];
}


/* The process's resident set right now, in bytes; 0 when it cannot be read. MEASURED rather
 * than modelled, like the VRAM budget's context: a fitted base was right for RNAfold (100
 * MiB) and 30 MB short for a Python process, which carries the interpreter and a 56 MB
 * extension module; RSS also includes memory the allocator keeps after a free, which a
 * model would have to guess. */
size_t
vrna_budget_rss(void)
{
  FILE          *f;
  unsigned long pages = 0, rss = 0;

  if ((f = fopen("/proc/self/statm", "r")) == NULL)
    return 0;

  if (fscanf(f, "%lu %lu", &pages, &rss) != 2)
    rss = 0;

  fclose(f);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
}


/* SHED BACKTRACK WORKERS FIRST (decision 2): they cost 8 MiB + 5 L^2 each and, on the
 * laptop, nothing in wall time to give up. Returns how many to run: `want` when no RAM
 * budget is in force or it fits, otherwise as many as fit beside `need_now` bytes (what
 * the process holds before the workers' scratch, plus what the driver is about to add),
 * at least one. An explicit RNA_BACKTRACK_THREADS wins over the budget (decision 1), with
 * a warning. Each distinct decision is announced once. */
int
vrna_budget_workers(int     want,
                    size_t  need_now,
                    size_t  max_length,
                    int     circ,
                    int     explicit_setting)
{
  static int    said_want = -1, said_got = -1;
  const size_t  cap = vrna_budget_ram();
  size_t        need, room, per;
  int           got;

  if ((cap == SIZE_MAX) || (want <= 1))
    return want;

  per   = vrna_budget_worker_bytes(max_length, circ);
  need  = need_now;
  /* 10 % of the room for what arrives with the workers beyond their scratch -- the pinned
   * transfer stage each gets, per-record backtrack allocations: without it a Python fold
   * at a 700 MB budget peaked at 718 with 4 workers allowed beside 511 MB held */
  room  = (cap > need) ? (size_t)((double)(cap - need) * 0.9) : 0;
  got   = (int)(room / per);
  if (got < 1)
    got = 1;

  if (got >= want)
    return want;

  if (explicit_setting) {
    if (said_want != want) {
      said_want = want;
      fprintf(stderr, "%-24s WARNING: RNA_BACKTRACK_THREADS=%d exceeds the RAM budget, which "
                      "fits %d (%.0f MB each at %zu nt); the explicit setting wins\n",
              "mfe/cuda/budget.c", want, got, (double)per / 1048576.0, max_length);
    }

    return want;
  }

  if ((said_want != want) || (said_got != got)) {
    said_want = want;
    said_got  = got;
    if (!vrna_cuda_quiet())
      fprintf(stderr, "%-24s RAM budget: backtrack workers %d -> %d (%.0f MB each at %zu nt; "
                      "%.0f MB of the %.0f MB budget already held)\n", "mfe/cuda/budget.c",
              want, got, (double)per / 1048576.0, max_length, (double)need / 1048576.0,
              (double)cap / 1048576.0);
  }

  return got;
}


void
vrna_budget_report(size_t device_total)
{
  static int  said = 0;
  char        r[96], v[96], t[96];

  if (said)
    return;

  /* Only when a budget is in force: `full` everywhere is today's behaviour and says
   * nothing, which keeps the CPU-only build silent as stock ViennaRNA is. */
  if (vrna_budget_is_full(VRNA_BUDGET_RAM) && vrna_budget_is_full(VRNA_BUDGET_VRAM) &&
      vrna_budget_is_full(VRNA_BUDGET_THREADS))
    return;

  said = 1;
  if (vrna_cuda_quiet())
    return;

  budget_describe(r, sizeof(r), VRNA_BUDGET_RAM, vrna_budget_host_total());
  budget_describe(v, sizeof(v), VRNA_BUDGET_VRAM, device_total);
  budget_describe(t, sizeof(t), VRNA_BUDGET_THREADS, 0);
  fprintf(stderr, "%-24s budget: %s | %s | %s\n", "mfe/cuda/budget.c", r, v, t);
}


/* ---------------------------------------------------------------- threads (B4)
 *
 * Two steps. vrna_budget_cap_threads() caps what a pool ASKS for at the budget, once,
 * announced; an explicit setting (-j N, RNA_BUILD_THREADS, RNA_BACKTRACK_THREADS) wins,
 * with a warning (decision 1). Then every pool DRAWS its workers from one shared count
 * (vrna_budget_threads_acquire / _release), because pools overlap: with the build
 * pipeline the next chunk's builders run beside this chunk's backtrack, and with the
 * backtrack pipeline a trace runs beside the next sweep. Capping each pool alone would
 * let two of them run 2 x the budget.
 *
 * A pool of n workers counts n whether it spawns them or runs serially on its caller; a
 * caller that only joins its workers is not counted. An auto-sized pool waits while the
 * budget is spent and then takes what is free (at least one); an explicit pool never
 * waits and is counted. No holder waits on anything while it holds workers (builders
 * build, backtrack workers trace), so the wait cannot deadlock.
 *
 * At `full` none of this runs: the count is not kept, nothing waits, today's behaviour. */
#if VRNA_WITH_PTHREADS
# include <pthread.h>
static pthread_mutex_t  g_thr_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   g_thr_cv  = PTHREAD_COND_INITIALIZER;
#endif
static int              g_thr_in_use, g_thr_peak, g_thr_budget_seen, g_thr_waits,
                        g_thr_short, g_thr_reported;


static void
budget_threads_report(void)
{
  if (vrna_cuda_quiet())
    return;

  fprintf(stderr, "%-24s thread budget: peak %d concurrent workers of a budget of %d "
                  "(%d draws waited, %d took fewer than asked)\n", "mfe/cuda/budget.c",
          g_thr_peak, g_thr_budget_seen, g_thr_waits, g_thr_short);
}


int
vrna_budget_cap_threads(const char  *who,
                        int         want,
                        int         explicit_setting)
{
  static char said[8][32];
  static int  nsaid;
  int         cap, k;

  if ((want <= 1) || vrna_budget_is_full(VRNA_BUDGET_THREADS))
    return want;

  cap = vrna_budget_threads();
  if (want <= cap)
    return want;

  /* each pool says it once (builders and a trace thread may get here together) */
#if VRNA_WITH_PTHREADS
  pthread_mutex_lock(&g_thr_mtx);
#endif
  for (k = 0; k < nsaid; k++)
    if (!strcmp(said[k], who))
      break;

  if ((k == nsaid) && (nsaid < 8)) {
    snprintf(said[nsaid++], sizeof(said[0]), "%s", who);
    if (explicit_setting)
      fprintf(stderr, "%-24s WARNING: %s %d exceeds the thread budget of %d; the explicit "
                      "setting wins\n", "mfe/cuda/budget.c", who, want, cap);
    else if (!vrna_cuda_quiet())
      fprintf(stderr, "%-24s thread budget: %s %d -> %d\n", "mfe/cuda/budget.c", who,
              want, cap);
  }

#if VRNA_WITH_PTHREADS
  pthread_mutex_unlock(&g_thr_mtx);
#endif

  return explicit_setting ? want : cap;
}


/* Returns how many workers to run; *held is what to hand back to _release() (0 when
 * nothing was counted, so a budget changed between the two cannot unbalance it). */
int
vrna_budget_threads_acquire(int want,
                            int explicit_setting,
                            int *held)
{
  int b, got;

  *held = 0;
  if (want < 1)
    want = 1;

  if (vrna_budget_is_full(VRNA_BUDGET_THREADS))
    return want;

  b = vrna_budget_threads();

#if VRNA_WITH_PTHREADS
  pthread_mutex_lock(&g_thr_mtx);
  if (!explicit_setting && (g_thr_in_use >= b)) {
    g_thr_waits++;
    while (g_thr_in_use >= b)
      pthread_cond_wait(&g_thr_cv, &g_thr_mtx);
  }
#endif

  got = want;
  if (!explicit_setting && (got > b - g_thr_in_use))
    got = b - g_thr_in_use;

  if (got < 1)                  /* only without pthreads, where nothing overlaps */
    got = 1;

  if (got < want)
    g_thr_short++;

  g_thr_in_use += got;
  if (g_thr_in_use > g_thr_peak)
    g_thr_peak = g_thr_in_use;

  if (b > g_thr_budget_seen)
    g_thr_budget_seen = b;

  if (!g_thr_reported) {
    g_thr_reported = 1;
    atexit(budget_threads_report);
  }

#if VRNA_WITH_PTHREADS
  pthread_mutex_unlock(&g_thr_mtx);
#endif

  *held = got;
  return got;
}


void
vrna_budget_threads_release(int held)
{
  if (held <= 0)
    return;

#if VRNA_WITH_PTHREADS
  pthread_mutex_lock(&g_thr_mtx);
#endif
  g_thr_in_use -= held;

#if VRNA_WITH_PTHREADS
  pthread_cond_broadcast(&g_thr_cv);
  pthread_mutex_unlock(&g_thr_mtx);
#endif
}

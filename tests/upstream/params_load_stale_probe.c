/*
 * Does a parameter set loaded AFTER the first fold reach the next fold?
 *
 * vrna_params() keeps a one-entry cache (SPEEDUP_PARAMS, params/params.c) keyed
 * on the model details alone. p_pre_init is set the first time it is filled and
 * never cleared, and nothing on the load path (vrna_params_load*() ->
 * set_parameters_from_string(), params/io.c) invalidates it. So once any fold has
 * filled the cache, a fold compound with unchanged model details gets the OLD
 * table back -- even though the load returned success.
 *
 * RNAfold -P cannot see this: it loads before it folds, so the cache is still
 * empty. A library caller that folds, loads another parameter set, and folds
 * again can -- the Python binding is the natural case (found through it on
 * 2026-10-02, confirmed on a pristine v2.7.2 build on 2026-10-03).
 *
 * The reference is a CHILD process that loads first and then folds, i.e. the
 * same sequence and the same parameter set with the cache still empty. The
 * parent folds first (priming the cache), loads, and folds again.
 *
 *   STALE CONFIRMED   the parent's second fold equals its FIRST (old table),
 *                     not the child's
 *   no defect         the parent's second fold equals the child's
 *   INCONCLUSIVE      the two parameter sets give the same energy here
 *
 * Usage: params_load_stale_probe [parameter-file]
 *   with no argument the built-in Andronescu 2007 set is loaded; with one, that
 *   file is loaded with vrna_params_load().
 *
 * Public API only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/params/io.h>
#include <ViennaRNA/utils/basic.h>

static const char *SEQ =
  "GGGGCCCCAAAAGGGGCCCCAAAAGGGGCCCCAAAACCCCGGGGAUAUCGCGAUAUCGCGUAUAUGCGC";

static float
fold(void)
{
  vrna_fold_compound_t  *fc = vrna_fold_compound(SEQ, NULL, VRNA_OPTION_DEFAULT);
  char                  *s  = vrna_alloc(strlen(SEQ) + 1);
  float                 e   = vrna_mfe(fc, s);

  free(s);
  vrna_fold_compound_free(fc);
  return e;
}


static int
load(const char *file)
{
  return file ? vrna_params_load(file, VRNA_PARAMETER_FORMAT_DEFAULT)
              : vrna_params_load_RNA_Andronescu2007();
}


int
main(int argc, char **argv)
{
  const char  *file = (argc > 1) ? argv[1] : NULL;
  const char  *what = file ? file : "built-in Andronescu 2007";
  int         fd[2];
  float       e_ref, e_first, e_after;
  pid_t       pid;

  /* reference: a fresh process, load BEFORE any fold */
  if (pipe(fd) != 0)
    return 2;

  pid = fork();
  if (pid == 0) {
    float e = load(file) ? fold() : 1e9f;

    if (write(fd[1], &e, sizeof(e)) != (ssize_t)sizeof(e))
      _exit(2);

    _exit(0);
  }

  if ((pid < 0) || (read(fd[0], &e_ref, sizeof(e_ref)) != (ssize_t)sizeof(e_ref)))
    return 2;

  waitpid(pid, NULL, 0);
  if (e_ref > 1e8f) {
    fprintf(stderr, "could not load %s\n", what);
    return 2;
  }

  /* this process: fold FIRST (fills the cache), then load, then fold again */
  e_first = fold();
  if (!load(file)) {
    fprintf(stderr, "could not load %s\n", what);
    return 2;
  }

  e_after = fold();

  printf("parameter set                 %s\n", what);
  printf("fold, default table           %8.2f\n", e_first);
  printf("load first, then fold (child) %8.2f   <- the loaded table\n", e_ref);
  printf("fold, load, fold again        %8.2f\n", e_after);

  if (e_ref == e_first) {
    printf("INCONCLUSIVE: both tables give %.2f for this sequence\n", e_ref);
    return 2;
  }

  if (e_after == e_first) {
    printf("STALE CONFIRMED: the second fold used the table from before the load\n");
    return 1;
  }

  printf("no defect: the second fold used the loaded table\n");
  return 0;
}

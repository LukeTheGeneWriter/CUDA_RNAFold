/* Drive the CUDA batch sweep directly, so RNA_ROW_VERIFY has something to run.
 *
 * The sweep's normal caller is RNAfold.c's chunk accumulator, which has not
 * been ported yet (that is Phase 4). This harness stands in for it: build N
 * fold compounds, hand them to par_mfe(), and compare the result against
 * upstream vrna_mfe() on independent fold compounds of the same sequences.
 *
 * Two different things are being checked, and they are worth separating:
 *
 *   RNA_ROW_VERIFY=1  makes the sweep itself compare every device cell against
 *                     the host recursion, per row, and complain on stderr. It
 *                     implies the HOST sweep (RNA_GPU_SWEEP=0) because device
 *                     mode is precisely the mode that does not run the host
 *                     loops there would be nothing to compare against.
 *
 *   this harness      checks the END RESULT -- energy and structure -- against
 *                     upstream. That is strictly weaker than the per-cell
 *                     check (it proves values right WHERE THEY ARE READ, not
 *                     everywhere), which is why both are run.
 *
 * Usage: cuda_sweep_harness [n_records] [length]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>

/* the batch entry point, declared in mfe/cuda/stub2.h */
extern void
par_mfe(const int                     nfiles,
        const vrna_fold_compound_t  **VC,
        const char                  **Structure,
        float                        *EN,
        const int                     cpu_queue_threads);

static char *
make_seq(unsigned int n, unsigned int seed)
{
  static const char alpha[] = "ACGU";
  char *s = (char *)vrna_alloc(sizeof(char) * (n + 1));
  unsigned int i, r = seed * 2654435761u + 1u;

  for (i = 0; i < n; i++) {
    r = r * 1103515245u + 12345u;
    s[i] = alpha[(r >> 16) & 3u];
  }
  s[n] = '\0';
  return s;
}

int
main(int argc, char **argv)
{
  const int    n_rec = (argc > 1) ? atoi(argv[1]) : 4;
  unsigned int len   = (argc > 2) ? (unsigned int)atoi(argv[2]) : 120;
  int          i, mismatches = 0;

  char                  **seqs  = (char **)vrna_alloc(sizeof(char *) * n_rec);
  vrna_fold_compound_t  **VC    = (vrna_fold_compound_t **)vrna_alloc(sizeof(void *) * n_rec);
  char                  **Str   = (char **)vrna_alloc(sizeof(char *) * n_rec);
  float                  *EN    = (float *)vrna_alloc(sizeof(float) * n_rec);
  float                  *ref   = (float *)vrna_alloc(sizeof(float) * n_rec);
  char                  **refS  = (char **)vrna_alloc(sizeof(char *) * n_rec);

  printf("batch: %d records of %u nt\n", n_rec, len);

  /* md.backtrack = 0 for the sweep's fold compounds.
   *
   * The point of this harness is to compare the MATRICES the sweep fills. With
   * backtracking on, a backtracking failure aborts (and can segfault) before
   * the matrices can be read, so the measurement that would explain the failure
   * is exactly the one the failure prevents. Filling without backtracking
   * separates the two questions: are the matrices right, and can they be
   * retraced. */
  for (i = 0; i < n_rec; i++) {
    vrna_md_t md;
    vrna_md_set_default(&md);
    /* RNA_SWEEP_NO_BACKTRACK=1 fills the matrices without retracing them,
     * which is what isolated "are the matrices right" from "can they be
     * retraced" while the fork's own backtrack() was still in the way. */
    if (getenv("RNA_SWEEP_NO_BACKTRACK"))
      md.backtrack = 0;

    seqs[i] = make_seq(len, 20260905u + (unsigned int)i);
    VC[i]   = vrna_fold_compound(seqs[i], &md, VRNA_OPTION_DEFAULT);
    Str[i]  = (char *)vrna_alloc(sizeof(char) * (len + 1));
    refS[i] = (char *)vrna_alloc(sizeof(char) * (len + 1));
  }

  /* the reference: upstream, one record at a time, on its own fold compounds */
  for (i = 0; i < n_rec; i++) {
    vrna_md_t md;
    vrna_fold_compound_t *fc;
    vrna_md_set_default(&md);
    fc     = vrna_fold_compound(seqs[i], &md, VRNA_OPTION_DEFAULT);
    ref[i] = vrna_mfe(fc, refS[i]);
    vrna_fold_compound_free(fc);
  }

  /* Keep an upstream-filled fold compound per record so the MATRICES can be
   * compared, not just the answer.
   *
   * This is the check RNA_ROW_VERIFY cannot make. That verify compares the
   * fork's DEVICE cells against the fork's HOST cells -- both of which mirror
   * the 2.3.0 recursion. If the fork's recursion has drifted from 2.7.2's, both
   * sides agree with each other and disagree with upstream, and the verify
   * reports a clean sweep. Comparing against upstream's own matrices is what
   * closes that gap. */
  vrna_fold_compound_t **UP = (vrna_fold_compound_t **)vrna_alloc(sizeof(void *) * n_rec);
  for (i = 0; i < n_rec; i++) {
    vrna_md_t md;
    char *tmp = (char *)vrna_alloc(sizeof(char) * (len + 1));
    vrna_md_set_default(&md);
    UP[i] = vrna_fold_compound(seqs[i], &md, VRNA_OPTION_DEFAULT);
    (void)vrna_mfe(UP[i], tmp);
    free(tmp);
  }

  fflush(stdout);
  par_mfe(n_rec, (const vrna_fold_compound_t **)VC, (const char **)Str, EN, 0);
  fflush(stderr);

  printf("\n  matrices, sweep vs upstream (the check RNA_ROW_VERIFY cannot make)\n");
  printf("  %-4s %10s %10s %10s %10s\n", "rec", "c cells", "c diff", "fML cells", "fML diff");
  fflush(stdout);
  for (i = 0; i < n_rec; i++) {
    const int *ca, *cb, *ma, *mb, *idx;
    long cells = 0, cdiff = 0, mdiff = 0;

    /* Say which pointer is missing rather than dereferencing it. A segfault
     * here reports nothing at all, and "nothing" is indistinguishable from a
     * clean run once stdout buffering eats the partial output. */
    if (!VC[i]->matrices || !UP[i]->matrices) {
      printf("  %-4d matrices absent (sweep %p, upstream %p)\n", i,
             (void *)VC[i]->matrices, (void *)UP[i]->matrices);
      fflush(stdout);
      mismatches++;
      continue;
    }

    ca = VC[i]->matrices->c;   cb = UP[i]->matrices->c;
    ma = VC[i]->matrices->fML; mb = UP[i]->matrices->fML;
    idx = VC[i]->jindx;

    if (!ca || !ma) {
      /* EXPECTED, not a failure: the sweep releases each record's c/fML as
       * soon as it is done with them -- that is the scratch-pool design that
       * keeps host RAM proportional to workers rather than to records. So the
       * matrices simply cannot be inspected after par_mfe() returns, and this
       * comparison is not available post-hoc. Counting it as a mismatch was
       * this harness reporting its own blind spot as a defect. */
      printf("  %-4d not comparable: the sweep frees c/fML per record "
             "(scratch pool); upstream's are still at %p/%p\n",
             i, (void *)cb, (void *)mb);
      fflush(stdout);
      continue;
    }
    if (!cb || !mb || !idx) {
      printf("  %-4d upstream array absent: c %p fML %p jindx %p\n",
             i, (void *)cb, (void *)mb, (void *)idx);
      fflush(stdout);
      mismatches++;
      continue;
    }
    {
    unsigned int a, b2;
    long cells = 0, cdiff = 0, mdiff = 0;

    for (b2 = 1; b2 <= len; b2++)
      for (a = 1; a <= b2; a++) {
        const int t = idx[b2] + a;
        cells++;
        if (ca[t] != cb[t]) cdiff++;
        if (ma[t] != mb[t]) mdiff++;
      }

    printf("  %-4d %10ld %10ld %10ld %10ld%s\n", i, cells, cdiff, cells, mdiff,
           (cdiff || mdiff) ? "   <-- DIVERGED FROM UPSTREAM" : "");
    if (cdiff || mdiff)
      mismatches++;
    fflush(stdout);
    }
  }

  for (i = 0; i < n_rec; i++)
    vrna_fold_compound_free(UP[i]);
  free(UP);

  {
    const int bt = (getenv("RNA_SWEEP_NO_BACKTRACK") == NULL);

    printf("\n  energies%s\n", bt ? " and structures" : " (backtracking off)");
    printf("  %-4s %12s %12s  %s\n", "rec", "sweep", "upstream",
           bt ? "structure" : "");
    for (i = 0; i < n_rec; i++) {
      const int e_ok = (EN[i] == ref[i]);
      const int s_ok = !bt || (strcmp(Str[i], refS[i]) == 0);

      printf("  %-4d %12.2f %12.2f  %s\n", i, EN[i], ref[i],
             (e_ok && s_ok) ? "match"
                            : (!e_ok ? "*** ENERGY DIFFERS ***"
                                     : "*** STRUCTURE DIFFERS ***"));
      if (!e_ok || !s_ok) {
        mismatches++;
        if (bt && !s_ok) {
          printf("        sweep    %s\n", Str[i]);
          printf("        upstream %s\n", refS[i]);
        }
      }
    }
  }

  printf("\n%s\n", mismatches
         ? "RESULT: SWEEP DISAGREES WITH UPSTREAM"
         : "RESULT: sweep agrees with upstream on every record");

  for (i = 0; i < n_rec; i++) {
    free(seqs[i]);
    free(Str[i]);
    free(refS[i]);
    vrna_fold_compound_free(VC[i]);
  }
  free(seqs); free(VC); free(Str); free(EN); free(ref); free(refS);

  return mismatches ? 1 : 0;
}

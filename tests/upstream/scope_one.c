/* One model detail per PROCESS.
 *
 * par_mfe() carries energy-parameter state across calls -- temperature 25
 * scores 12/12 as the first call and 0/12 after a 37 C batch, and
 * teardown_gpu/2/3() does not reset it. So a probe that loops over models in
 * one process measures the caching bug, not the models. One per process is the
 * only way to get a clean answer until that is fixed.
 *
 * Usage: scope_one <model>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>

extern void par_mfe(const int, const vrna_fold_compound_t **, const char **,
                    float *, const int);

#define NREC 12
#define LEN  160

static char *
seq_of(unsigned int n, unsigned int seed)
{
  static const char a[] = "ACGU";
  char *s = (char *)vrna_alloc(n + 1);
  unsigned int i, r = seed * 2654435761u + 1u;
  for (i = 0; i < n; i++) { r = r * 1103515245u + 12345u; s[i] = a[(r >> 16) & 3u]; }
  s[n] = 0;
  return s;
}

static void
apply(const char *m, vrna_md_t *md)
{
  if (!strcmp(m, "baseline"))         return;
  else if (!strcmp(m, "temp25"))      md->temperature = 25.0;
  else if (!strcmp(m, "noLP"))        md->noLP = 1;
  else if (!strcmp(m, "noGUclosure")) md->noGUclosure = 1;
  else if (!strcmp(m, "noGU"))        md->noGU = 1;
  else if (!strcmp(m, "uniqML"))      md->uniq_ML = 1;
  else if (!strcmp(m, "dangles0"))    md->dangles = 0;
  else if (!strcmp(m, "dangles1"))    md->dangles = 1;
  else if (!strcmp(m, "dangles3"))    md->dangles = 3;
  else if (!strcmp(m, "logML"))       md->logML = 1;
  else { fprintf(stderr, "unknown model %s\n", m); exit(2); }
}

int
main(int argc, char **argv)
{
  const char *m = (argc > 1) ? argv[1] : "baseline";
  vrna_fold_compound_t **VC = (vrna_fold_compound_t **)vrna_alloc(sizeof(void *) * NREC);
  char **Str  = (char **)vrna_alloc(sizeof(char *) * NREC);
  char **seqs = (char **)vrna_alloc(sizeof(char *) * NREC);
  float *EN   = (float *)vrna_alloc(sizeof(float) * NREC);
  int i, bad = 0;

  for (i = 0; i < NREC; i++) {
    vrna_md_t md;
    vrna_md_set_default(&md);
    apply(m, &md);
    seqs[i] = seq_of(LEN, 4242u + i);
    VC[i]   = vrna_fold_compound(seqs[i], &md, VRNA_OPTION_DEFAULT);
    Str[i]  = (char *)vrna_alloc(LEN + 1);
  }

  par_mfe(NREC, (const vrna_fold_compound_t **)VC, (const char **)Str, EN, 0);

  for (i = 0; i < NREC; i++) {
    vrna_md_t md;
    char *ref = (char *)vrna_alloc(LEN + 1);
    vrna_fold_compound_t *fc;
    float e;
    vrna_md_set_default(&md);
    apply(m, &md);
    fc = vrna_fold_compound(seqs[i], &md, VRNA_OPTION_DEFAULT);
    e  = vrna_mfe(fc, ref);
    if ((e != EN[i]) || (strcmp(ref, Str[i]) != 0))
      bad++;
    vrna_fold_compound_free(fc);
    free(ref);
  }

  printf("  %-14s %2d/%d match%s\n", m, NREC - bad, NREC,
         bad ? "" : "   <-- ALREADY WORKS");
  return bad ? 1 : 0;
}

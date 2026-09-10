/* c_gq's ROW structure, which is what the device representation turns on.
 *
 * gquad_sparsity_probe.c established that c_gq is under 5% full and that the
 * entry count grows roughly LINEARLY in n, so uploading it dense would cost
 * another whole triangle per record to carry ~1% useful data. That settled
 * "upload sparse". It did NOT settle how the kernel should LOOK THINGS UP,
 * and that is where the cost actually lands: vrna_smx_csr_int_get() is a
 * binary search, run per cell, in the innermost loop of the kernel already
 * measured to sit at its DRAM floor.
 *
 * The two device call sites want different things:
 *
 *   extend_fm_3p()  (mfe_multibranch.c:986)  needs c_gq(i,j) for the cell it
 *                                            is already computing -- a point
 *                                            lookup keyed by the cell itself.
 *   vrna_mfe_gquad_internal_loop()           needs c_gq(p,q) over a bounded
 *                                            (p,q) window near (i,j), three
 *                                            such sweeps.
 *
 * Both are cheap IF a row with no entries can be rejected in one comparison.
 * So the numbers that decide the design are not fill % but:
 *
 *   1. what fraction of rows i hold ZERO entries  -> is per-row skip effective?
 *   2. the MAX entries in any single row          -> can a row be expanded into
 *                                                    shared memory at row start?
 *   3. bytes for CSR vs dense, per record         -> the upload budget
 *
 * Build: see run_probes.sh. Takes an optional FASTA; with none it uses the
 * same synthetic G-rich generator as the sparsity probe so the two are
 * comparable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/datastructures/dp_matrices.h>
#include <ViennaRNA/datastructures/sparse_mx.h>

static char *
gq_sequence(unsigned int n, unsigned int seed)
{
  static const char *blocks[] = {
    "GGGG", "A", "GGG", "UUA", "GGGG", "CU", "GGG", "AAA",
    "AUGCAUGC", "GGGG", "U", "GGG", "AC", "GGGG", "AU", "GGG"
  };
  const unsigned int nb = sizeof(blocks) / sizeof(blocks[0]);
  char *s = (char *)vrna_alloc(sizeof(char) * (n + 1));
  unsigned int len = 0, r = seed;

  while (len < n) {
    r = r * 1103515245u + 12345u;
    const char *b = blocks[(r >> 16) % nb];
    unsigned int bl = (unsigned int)strlen(b);
    if (len + bl > n)
      bl = n - len;
    memcpy(s + len, b, bl);
    len += bl;
  }
  s[n] = '\0';
  return s;
}


static void
report(const char *tag, const char *seq)
{
  vrna_md_t             md;
  vrna_fold_compound_t  *fc;
  unsigned int          n, i, j;
  unsigned long         entries = 0, cells;
  unsigned int          rows_used = 0, row_max = 0;
  unsigned int          *per_row;

  vrna_md_set_default(&md);
  md.gquad = 1;

  /* VRNA_OPTION_MFE, not VRNA_OPTION_DEFAULT. DEFAULT is literally 0
   * (fold_compound.h:398), so vrna_mx_add() adds NO matrices and c_gq comes
   * back NULL -- which looks exactly like "gquad is off". The MFE matrices,
   * c_gq included, are added by vrna_fold_compound_prepare(fc,
   * VRNA_OPTION_MFE), which upstream runs inside vrna_mfe() and the port runs
   * inside par_mfe() (mfe_cuda.c:1090). Cost the first version of this probe
   * one wrong conclusion. */
  fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
  if (fc)
    (void)vrna_fold_compound_prepare(fc, VRNA_OPTION_MFE);
  if ((fc == NULL) || (fc->matrices == NULL) || (fc->matrices->c_gq == NULL)) {
    printf("%-10s  NO c_gq -- is md.gquad set and the matrices allocated?\n", tag);
    if (fc)
      vrna_fold_compound_free(fc);
    return;
  }

  n       = fc->length;
  cells   = ((unsigned long)n * (n + 1)) / 2;
  per_row = (unsigned int *)vrna_alloc(sizeof(unsigned int) * (n + 2));

  /* Walk every (i,j). This is O(n^2) GETS and therefore slow, but it is the
   * only representation-independent way to count: reading the CSR internals
   * would bake in assumptions about a layout we are deciding whether to keep. */
  for (i = 1; i <= n; i++) {
    for (j = i + 1; j <= n; j++) {
#ifndef VRNA_DISABLE_C11_FEATURES
      int e = vrna_smx_csr_get(fc->matrices->c_gq, i, j, INF);
#else
      int e = vrna_smx_csr_int_get(fc->matrices->c_gq, i, j, INF);
#endif
      if (e != INF) {
        entries++;
        per_row[i]++;
      }
    }
  }

  for (i = 1; i <= n; i++) {
    if (per_row[i]) {
      rows_used++;
      if (per_row[i] > row_max)
        row_max = per_row[i];
    }
  }

  {
    /* CSR: one int energy + one int column per entry, plus a per-row offset.
     * Dense: one int per triangular cell. */
    double csr_kib   = (entries * 2.0 * sizeof(int) + (n + 2.0) * sizeof(int)) / 1024.0;
    double dense_kib = (cells * (double)sizeof(int)) / 1024.0;

    printf("%-10s n=%-6u entries=%-7lu fill=%5.2f%%  rows_with_entries=%u/%u (%.1f%%)"
           "  max_row=%-5u  CSR=%.1f KiB  dense=%.1f KiB  (%.0fx)\n",
           tag, n, entries, 100.0 * entries / (double)cells,
           rows_used, n, 100.0 * rows_used / (double)n,
           row_max, csr_kib, dense_kib,
           csr_kib > 0 ? dense_kib / csr_kib : 0.0);
  }

  free(per_row);
  vrna_fold_compound_free(fc);
}


int
main(int argc, char **argv)
{
  printf("c_gq row structure. 'rows_with_entries' is the number of i for which\n"
         "ANY j has a quadruplex; a per-row skip is worth having only if that\n"
         "fraction is small. 'max_row' bounds a shared-memory row expansion.\n\n");

  if (argc > 1) {
    /* a FASTA, so the frozen bar's own records can be measured */
    FILE *f = fopen(argv[1], "r");
    char line[1 << 16], *seq = NULL, tag[64] = "";
    size_t seqlen = 0;

    if (!f) {
      fprintf(stderr, "cannot open %s\n", argv[1]);
      return 1;
    }

    while (fgets(line, sizeof(line), f)) {
      char *nl = strchr(line, '\n');
      if (nl)
        *nl = '\0';
      if (line[0] == '>') {
        if (seq) {
          report(tag, seq);
          free(seq);
          seq = NULL;
          seqlen = 0;
        }
        snprintf(tag, sizeof(tag), "%.60s", line + 1);
      } else if (line[0]) {
        size_t l = strlen(line);
        seq = (char *)realloc(seq, seqlen + l + 1);
        memcpy(seq + seqlen, line, l + 1);
        seqlen += l;
      }
    }
    if (seq) {
      report(tag, seq);
      free(seq);
    }
    fclose(f);
  } else {
    const unsigned int lens[] = { 100, 300, 600, 1200, 2400 };
    unsigned int k;

    for (k = 0; k < sizeof(lens) / sizeof(lens[0]); k++) {
      char *s = gq_sequence(lens[k], 20260905u + lens[k]);
      char  tag[32];

      snprintf(tag, sizeof(tag), "synth%u", lens[k]);
      report(tag, s);
      free(s);
    }
  }

  return 0;
}

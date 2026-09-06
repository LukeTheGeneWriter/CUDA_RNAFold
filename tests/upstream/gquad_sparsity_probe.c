/* How sparse is c_gq, the G-quadruplex energy matrix?
 *
 * It is built ONCE, up front, at matrix-allocation time
 * (datastructures/dp_matrices.c:547 -> vrna_mfe_gquad_mx()), independently of
 * the MFE recursion; every later use in mfe/mfe.c is a read via
 * vrna_smx_csr_int_get(). So on the GPU it is a read-only lookup table uploaded
 * before the sweep -- and the question is what representation to upload.
 *
 * A dense triangular int matrix costs n(n+1)/2 ints per record, the same as
 * another c/fML, which is exactly the memory the chunker is short of. A sparse
 * form is only worth its lookup cost if the fill fraction is genuinely tiny.
 * This measures the fill fraction on G-rich sequence at several lengths.
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

/* deterministic G-rich sequence: G-runs separated by short linkers, which is
 * what a quadruplex needs, interleaved with ordinary structured RNA */
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

int
main(void)
{
  const unsigned int lens[] = { 100, 300, 600, 1200, 2400 };
  const unsigned int nl = sizeof(lens) / sizeof(lens[0]);
  unsigned int i;

  printf("c_gq fill, G-rich sequence, md.gquad = 1\n\n");
  printf("  %6s  %10s  %12s  %9s  %14s\n",
         "n", "entries", "cells n(n+1)/2", "fill %", "dense KiB/record");

  for (i = 0; i < nl; i++) {
    unsigned int n = lens[i];
    char *seq = gq_sequence(n, 20260905u + n);
    vrna_md_t md;
    vrna_fold_compound_t *fc;
    unsigned long cells, entries;
    char *structure;

    vrna_md_set_default(&md);
    md.gquad = 1;

    structure = (char *)vrna_alloc(sizeof(char) * (n + 1));
    fc = vrna_fold_compound(seq, &md, VRNA_OPTION_DEFAULT);

    /* fold so the matrices are certainly built and populated */
    (void)vrna_mfe(fc, structure);

    if (!fc->matrices->c_gq) {
      printf("  %6u  c_gq is NULL -- md.gquad did not build it\n", n);
      free(seq); free(structure); vrna_fold_compound_free(fc);
      continue;
    }

    entries = (unsigned long)vrna_smx_csr_int_get_size(fc->matrices->c_gq);
    cells   = (unsigned long)n * (n + 1) / 2;

    printf("  %6u  %10lu  %12lu  %8.4f%%  %14.1f\n",
           n, entries, cells,
           100.0 * (double)entries / (double)cells,
           (double)cells * sizeof(int) / 1024.0);

    free(seq);
    free(structure);
    vrna_fold_compound_free(fc);
  }

  printf("\nA dense upload costs the right-hand column per record, on top of\n"
         "c and fML. A sparse upload costs ~8 bytes per entry plus a lookup.\n");
  return 0;
}

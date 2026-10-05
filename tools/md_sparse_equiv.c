/*
 * md_sparse_equiv.c -- the S0 bar for sparse multiloop decomposition (PORT_SPARSE_MD.md).
 *
 * Claim. For every cell (i,j), with A any left operand that satisfies
 *     A(i,k) <= A(i,k-1) + MLbase   wherever up_ml(k)                          (chain)
 * the dense split
 *     D(i,j) = min over k in (i, j] of  A(i,k-1) + fML(k,j)
 * equals the sparse one
 *     S(i,j) = min( up_ml(j) ? S(i,j-1) + MLbase : INF ,
 *                   min over CANDIDATES k of A(i,k-1) + fML(k,j) )
 * where k is NOT a candidate iff
 *     [up_ml(k) and fML(k,j) >= fML(k+1,j) + MLbase]  or  [up_ml(j) and fML(k,j) >= fML(k,j-1) + MLbase].
 *
 * fML is upstream's own matrix, after vrna_mfe(), so the options under test shape it exactly as RNAfold
 * does. The left operand is either upstream's full fML row (--left=full) or a synthetic row that satisfies
 * (chain) and nothing else (--left=chain) -- the second is what makes the check speak for the GPU, whose
 * left operand is the row's fML WITHOUT the split term, which (chain) is the only property used of.
 *
 * Negative controls, each must report mismatches on a fixture that reaches it:
 *   --negctl=scan   drop the S(i,j-1) term
 *   --negctl=hc     ignore up_ml in the candidate test and the scan (needs -C with forced-paired '|')
 *
 * Input: FASTA; a line after the sequence that starts with one of .()|x<>[]{} is its constraint (-C).
 * Options: -g --circ -d0 -d2 --noLP --salt=M -T=C --maxBPspan=N -C --enforce --left=full|chain --negctl=...
 * Exit 0 iff no mismatch (and, under --negctl, iff there WERE mismatches).
 *
 * Build against any libRNA (see tools/md_sparse_equiv.sh).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/constraints/basic.h>
#include <ViennaRNA/constraints/hard.h>
#include <ViennaRNA/utils/basic.h>

#define ADD(a, b) (((a) >= INF || (b) >= INF) ? INF : (a) + (b))

static unsigned int rng = 12345u;
static int
rnd(void)
{
  rng = rng * 1103515245u + 12345u;
  return (int)((rng >> 8) & 0x7fff);
}


int
main(int argc, char **argv)
{
  vrna_md_t md;
  int       use_C = 0, enforce = 0, left_chain = 0, neg_scan = 0, neg_hc = 0;
  const char *path = NULL;

  vrna_md_set_default(&md);
  for (int a = 1; a < argc; a++) {
    const char *s = argv[a];
    if (!strcmp(s, "-g")) md.gquad = 1;
    else if (!strcmp(s, "--circ")) md.circ = 1;
    else if (!strcmp(s, "-d0")) md.dangles = 0;
    else if (!strcmp(s, "-d2")) md.dangles = 2;
    else if (!strcmp(s, "--noLP")) md.noLP = 1;
    else if (!strncmp(s, "--salt=", 7)) md.salt = atof(s + 7);
    else if (!strncmp(s, "-T=", 3)) md.temperature = atof(s + 3);
    else if (!strncmp(s, "--maxBPspan=", 12)) md.max_bp_span = atoi(s + 12);
    else if (!strcmp(s, "-C")) use_C = 1;
    else if (!strcmp(s, "--enforce")) enforce = 1;
    else if (!strcmp(s, "--left=chain")) left_chain = 1;
    else if (!strcmp(s, "--left=full")) left_chain = 0;
    else if (!strcmp(s, "--negctl=scan")) neg_scan = 1;
    else if (!strcmp(s, "--negctl=hc")) neg_hc = 1;
    else if (s[0] != '-') path = s;
    else { fprintf(stderr, "unknown option %s\n", s); return 2; }
  }
  if (!path) { fprintf(stderr, "usage: md_sparse_equiv [options] file.fa\n"); return 2; }

  FILE *f = fopen(path, "r");
  if (!f) { perror(path); return 2; }

  static char line[1 << 20];
  char        *seq = NULL, *con = NULL, name[256] = "";
  long long   cells = 0, bad = 0, td = 0, ts = 0, nrec = 0;
  int         more = 1;

  while (more) {
    more = (fgets(line, sizeof(line), f) != NULL);
    line[strcspn(line, "\r\n")] = 0;
    int is_hdr = more && line[0] == '>';
    if ((!more || is_hdr) && seq) {
      /* ---- one record ---- */
      vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
      if (use_C && con)
        vrna_constraints_add(fc, con, VRNA_CONSTRAINT_DB_DEFAULT |
                             (enforce ? VRNA_CONSTRAINT_DB_ENFORCE_BP : 0));
      char *st = malloc(strlen(seq) + 1);
      vrna_mfe(fc, st);
      const int     n   = (int)fc->length;
      const int     *F  = fc->matrices->fML, *jx = fc->jindx, mb = fc->params->MLbase;
      unsigned int  *up = fc->hc->up_ml;
#define FML(a, b) (((a) >= 1 && (b) <= n && (a) <= (b)) ? F[jx[b] + (a)] : INF)
#define UP(k)     (neg_hc ? 1 : (up[k] > 0))
      char *cand = calloc((size_t)(n + 2) * (n + 2), 1);
      for (int j = 1; j <= n; j++)
        for (int k = 1; k <= j; k++) {
          const int v = FML(k, j);
          if (v >= INF) continue;
          const int l = FML(k + 1, j), r = FML(k, j - 1);
          const int dom_l = UP(k) && l < INF && v >= l + mb;
          const int dom_r = UP(j) && r < INF && v >= r + mb;
          cand[(size_t)k * (n + 2) + j] = !(dom_l || dom_r);
        }
      int *A = malloc(sizeof(int) * (n + 2)), *S = malloc(sizeof(int) * (n + 2));
      for (int i = 1; i <= n; i++) {
        /* the left operand row: upstream's fML(i,.) or a synthetic row with (chain) and nothing else */
        for (int k = 0; k <= n; k++) {
          if (!left_chain) { A[k] = (k >= i) ? FML(i, k) : INF; continue; }
          int v = (k >= i && (rnd() % 3)) ? (rnd() % 4000) - 3000 : INF;
          if (k > i && up[k] > 0 && A[k - 1] < INF && A[k - 1] + mb < v) v = A[k - 1] + mb;
          A[k] = v;
        }
        S[i] = INF;
        for (int j = i + 1; j <= n; j++) {
          int d = INF;
          int s = (!neg_scan && UP(j)) ? ADD(S[j - 1], mb) : INF;
          for (int k = i + 1; k <= j; k++) {
            const int t = ADD(A[k - 1], FML(k, j));
            td++;
            if (t < d) d = t;
            if (cand[(size_t)k * (n + 2) + j]) { ts++; if (t < s) s = t; }
          }
          S[j] = s;
          cells++;
          if (s != d) {
            if (bad < 5 && !neg_scan && !neg_hc)
              fprintf(stderr, "  MISMATCH %s i=%d j=%d dense %d sparse %d\n", name, i, j, d, s);
            bad++;
          }
        }
      }
      free(A); free(S); free(cand); free(st);
      vrna_fold_compound_free(fc);
      free(seq); seq = NULL;
      free(con); con = NULL;
      nrec++;
    }
    if (!more) break;
    if (is_hdr) { snprintf(name, sizeof(name), "%s", line + 1); continue; }
    if (!line[0]) continue;
    if (seq && strchr(".()|x<>[]{}", line[0])) { con = strdup(line); continue; }
    if (!seq) seq = strdup(line);
  }
  fclose(f);

  const int negctl = neg_scan || neg_hc;
  printf("%s%s: %lld records, %lld cells, %lld differ | terms dense %lld, sparse %lld (%.1fx fewer)\n",
         negctl ? (neg_scan ? "NEGCTL scan " : "NEGCTL hc ") : "",
         left_chain ? "left=chain" : "left=full", nrec, cells, bad, td, ts, ts ? (double)td / ts : 0.0);
  if (negctl) return bad > 0 ? 0 : 1;
  return bad == 0 ? 0 : 1;
}

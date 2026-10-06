/* tools/lyngso_equiv.c -- S0 bar for the Lyngsø int_loop (roadmap N3). Is a Lyngsø carry EXACT against upstream's own interior-loop MFE, under the options the
 * 10-05 check skipped -- hard constraints (-C, --enforceConstraint, 'x'), --noClosingGU, -g, --noLP?
 *
 * Reference:  R(i,j) = vrna_mfe_internal(fc, i, j)  on upstream's finished c matrix (stacks, bulges,
 *             generic loops, special loops, enclosed G-quads, every hard-constraint check).
 * Carry form: X(i,j) = min( stack , bulges , generic DIRECT , generic CARRY , gquad term )
 *   generic DIRECT: u1,u2 >= 1 with short side 1, or 2x2 / 2x3 / 3x2, upstream's per-loop checks
 *   generic CARRY : G(i,j,u) = min( G(i+1,j-1,u-2) if i+1 and j-1 may be unpaired in an interior loop,
 *                                   fresh entries at (i,j): short side 2 (not 2x2/2x3/3x2), 3x3, 3x4, 4x3 )
 *                   entry  = c(k,l) + mismatchI(inner) + MIN2(MAX_NINIO, |u1-u2| ninio), kept only if the
 *                            inner pair is allowed as an enclosed pair and is not GU under --noClosingGU,
 *                            and its own unpaired runs fit hc->up_int
 *                   use    = G(i,j,u) + internal_loop[u] + mismatchI(outer) + salt(u), outer pair gated once
 *   gquad term: vrna_mfe_gquad_internal_loop(fc,i,j) -- SHARED with the reference, so -g checks only that
 *               the carry is unaffected by G-quads in c, not the G-quad term itself.
 * The hc mask is the new part: a carried loop gains unpaired i+1 and j-1, and hc->up_int counts consecutive
 * unpairable positions, so the carry is valid iff up_int[i+1] >= 1 and up_int[j-1] >= 1.
 *
 * usage: lyngso_hc L nseq seed mode [negctl]
 *   mode: plain | C (random '|', 'x' and enforced pairs, --enforceConstraint) | noGU | g (G-rich) | noLP | salt
 *   negctl 1: ignore the hc mask on the carry (must go red under C);  negctl 2: drop the carry from (i+1,j-1) (red always) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/mfe/internal.h>
#include <ViennaRNA/mfe/gquad.h>
#include <ViennaRNA/eval/internal.h>
#include <ViennaRNA/constraints/hard.h>
#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/params/salt.h>
#define ADD(a,b) (((a) >= INF || (b) >= INF) ? INF : (a) + (b))
#define MINI(a,b) ((a) < (b) ? (a) : (b))

static int generic_carried(int u1, int u2) {      /* belongs to G: short side >= 2, not 2x2/2x3/3x2 */
  int s = MINI(u1, u2), l = u1 > u2 ? u1 : u2;
  return s >= 2 && !(s == 2 && l <= 3);
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: L nseq seed mode [negctl]\n"); return 2; }
  int L = atoi(argv[1]), nseq = atoi(argv[2]), negctl = argc > 5 ? atoi(argv[5]) : 0;
  const char *mode = argv[4];
  srand(atoi(argv[3]));
  long long cells = 0, bad = 0, finite = 0;
  for (int sq = 0; sq < nseq; sq++) {
    char *seq = malloc(L + 1);
    for (int i = 0; i < L; i++) {
      if (!strcmp(mode, "g") && rand() % 9 == 0 && i + 7 < L) { memcpy(seq + i, "GGGAGGG", 7); i += 6; continue; }
      seq[i] = "ACGU"[rand() % 4];
    }
    seq[L] = 0;
    vrna_md_t md; vrna_md_set_default(&md);
    if (!strcmp(mode, "noGU")) md.noGUclosure = 1;
    if (!strcmp(mode, "g"))    md.gquad = 1;
    if (!strcmp(mode, "noLP")) md.noLP = 1;
    if (!strcmp(mode, "salt")) md.salt = 0.2;
    vrna_fold_compound_t *fc = vrna_fold_compound(seq, &md, VRNA_OPTION_MFE);
    if (!strcmp(mode, "C")) {
      /* a dot-bracket constraint: some 'x' (unpaired), some '|' (paired), a few enforced nested pairs */
      char *db = malloc(L + 1); memset(db, '.', L); db[L] = 0;
      for (int t = 0; t < L / 25; t++) db[rand() % L] = 'x';
      for (int t = 0; t < L / 25; t++) db[rand() % L] = '|';
      for (int t = 0, a = L / 10; t < 3 && a + 40 < L - L / 10; t++, a += L / 6) {
        int b = L - 1 - (a - L / 10) - t * 7;
        if (b - a > 10 && db[a] == '.' && db[b] == '.') { db[a] = '('; db[b] = ')'; }
      }
      vrna_hc_add_from_db(fc, db, VRNA_CONSTRAINT_DB_DEFAULT | VRNA_CONSTRAINT_DB_ENFORCE_BP);
      free(db);
    }
    char *st = malloc(L + 1);
    vrna_mfe(fc, st);

    vrna_param_t *P = fc->params;
    vrna_hc_t *hc = fc->hc;
    int *c = fc->matrices->c, *jx = fc->jindx, n = L;
    short *S = fc->sequence_encoding;
    unsigned int *up = hc->up_int;
    unsigned char *mx = hc->mx;
    int *rt = P->model_details.rtype;
    /* upstream: vrna_get_ptype() -- ptype 0 on a pair the constraints allow means 7, non-standard */
    #define TYPE(a,b) (fc->ptype[jx[b] + (a)] ? (int)fc->ptype[jx[b] + (a)] : 7)
    #define CC(a,b) c[jx[b] + (a)]
    #define ENC(k,l) (mx[n * (k) + (l)] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP_ENC)
    #define GU(t) ((t) == 3 || (t) == 4)
    int W = MAXLOOP + 1;
    int *Gp = malloc(sizeof(int) * (L + 2) * W), *Gc = malloc(sizeof(int) * (L + 2) * W);
    for (int t = 0; t < (L + 2) * W; t++) Gp[t] = Gc[t] = INF;

    for (int i = L; i >= 1; i--) {
      for (int t = 0; t < (L + 2) * W; t++) Gc[t] = INF;
      for (int j = i + 1; j <= L; j++) {
        int *G = &Gc[j * W];
        /* the carry, masked by hc: the carried loops gain unpaired i+1 and j-1 */
        if (negctl != 2 && j - 1 > i + 1 && (negctl == 1 || (up[i + 1] >= 1 && up[j - 1] >= 1)))
          for (int u = 2; u <= MAXLOOP; u++) G[u] = Gp[(j - 1) * W + u - 2];
        /* fresh entries at (i,j) */
        for (int u1 = 2; u1 <= MAXLOOP; u1++)
          for (int u2 = 2; u1 + u2 <= MAXLOOP; u2++) {
            if (!generic_carried(u1, u2)) continue;
            int sh = MINI(u1, u2), lo = u1 > u2 ? u1 : u2;
            int fresh = (sh == 2) || (sh == 3 && lo <= 4);
            if (!fresh) continue;
            int k = i + u1 + 1, l = j - u2 - 1;
            if (l - k < 1) continue;
            if ((int)up[i + 1] < u1 || (int)up[l + 1] < u2) continue;
            int t2 = TYPE(k, l);
            if (!ENC(k, l) || CC(k, l) >= INF) continue;
            int tt2 = rt[t2];
            if (md.noGUclosure && GU(tt2)) continue;
            int v = CC(k, l) + P->mismatchI[tt2][S[l + 1]][S[k - 1]] + MINI(MAX_NINIO, abs(u1 - u2) * P->ninio[2]);
            if (v < G[u1 + u2]) G[u1 + u2] = v;
          }

        int R = vrna_mfe_internal(fc, i, j);
        int X = INF;
        if (mx[n * i + j] & VRNA_CONSTRAINT_CONTEXT_INT_LOOP) {
          int type = TYPE(i, j);
          int noclose = md.noGUclosure && GU(type);
          /* stack */
          if (i + 1 < j - 1 && hc->eval_int(i, j, i + 1, j - 1, hc) && CC(i + 1, j - 1) < INF)
            X = MINI(X, CC(i + 1, j - 1) + vrna_E_internal(0, 0, type, rt[TYPE(i + 1, j - 1)], S[i + 1], S[j - 1], S[i], S[j], P));
          if (!noclose) {
            /* bulges and generic DIRECT: (u1,u2) with u1 or u2 == 0 (not both), short side 1, or the specials */
            for (int u1 = 0; u1 <= MAXLOOP; u1++)
              for (int u2 = 0; u1 + u2 <= MAXLOOP; u2++) {
                if (u1 == 0 && u2 == 0) continue;
                if (generic_carried(u1, u2)) continue;
                int k = i + u1 + 1, l = j - u2 - 1;
                if (l - k < 1) continue;
                if (u1 && (int)up[i + 1] < u1) continue;
                if (u2 && (int)up[l + 1] < u2) continue;
                if (!hc->eval_int(i, j, k, l, hc)) continue;
                int t2 = TYPE(k, l);
                if (CC(k, l) >= INF) continue;
                int tt2 = rt[t2];
                if (md.noGUclosure && GU(tt2)) continue;
                X = MINI(X, CC(k, l) + vrna_E_internal(u1, u2, type, tt2, S[i + 1], S[j - 1], S[k - 1], S[l + 1], P));
              }
            /* generic CARRY */
              for (int u = 4; u <= MAXLOOP; u++) {
                if (G[u] >= INF) continue;
                int saltc = 0;
                if (P->model_details.salt != VRNA_MODEL_DEFAULT_SALT)
                  saltc = (u + 2 <= MAXLOOP + 1) ? P->SaltLoop[u + 2] :
                          vrna_salt_loop_int(u + 2, P->model_details.salt, P->temperature + K0, P->model_details.backbone_length);
                X = MINI(X, G[u] + P->internal_loop[u] + P->mismatchI[type][S[i + 1]][S[j - 1]] + saltc);
              }
            if (md.gquad) X = MINI(X, vrna_mfe_gquad_internal_loop(fc, i, j));
          }
        }
        cells++;
        if (R < INF) finite++;
        if (!((R >= INF && X >= INF) || R == X)) {
          if (bad < 3 && getenv("LY_VERBOSE")) printf("  diff i=%d j=%d R %d X %d\n", i, j, R, X);
          bad++;
        }
      }
      int *t = Gp; Gp = Gc; Gc = t;
    }
    free(Gp); free(Gc); vrna_fold_compound_free(fc); free(seq); free(st);
  }
  printf("mode %-5s L=%d x%d%s: %lld cells (%lld finite), %lld differ\n", mode, L, nseq,
         negctl == 1 ? " NEGCTL(hc mask off)" : negctl == 2 ? " NEGCTL2(carry from i+1,j-1 dropped)" : "",
         cells, finite, bad);
  return 0;
}

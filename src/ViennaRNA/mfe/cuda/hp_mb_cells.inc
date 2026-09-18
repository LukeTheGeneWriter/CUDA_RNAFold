/* hp_mb_cells.inc -- per-cell bodies lifted out of the kernels in hp_mb_loop.cu.
 *
 * The fused per-record megakernel and the standalone kernels must run the
 * SAME arithmetic; device code cannot cross a translation unit in this build
 * (no -rdc), so a header is the only way to share it. The bodies are copied
 * verbatim -- only the thread-index prologue became a parameter.
 */

__device__ inline void
hp_mb_3p_cell(const int nfiles, const int i_row, const int turn, const int length,
                 const short* __restrict__ S,
                 const char*  __restrict__ seq,
                 const char*  __restrict__ pair,
                 const unsigned int* __restrict__ hccc_mb,
                 const unsigned int* __restrict__ hccc_mbenc,
                 const unsigned int* __restrict__ hccc_any,
                 const unsigned int* __restrict__ hccc_gu,
                 const cuda_param2_t* __restrict__ P,
                 const int* __restrict__ salt_loop,   //in  d_salt_loop, by backbone count
                       int* __restrict__ energy_hp_row,
                       int* __restrict__ energy_mb_row,
                       int* __restrict__ energy_3p00_row,
                      char* __restrict__ gate_row,
                 const size_t* __restrict__ row_off_H,
                 const size_t* __restrict__ hc2_off_H,
                 const size_t* __restrict__ seq_off_H,
                 const int* __restrict__ len_H,   //continuous flow phase C1
                 const size_t* __restrict__ size_off_H, const size_t total,
                const int* __restrict__ i_H,
               const long long m) {
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  // PHASE A (continuous flow): the row index is PER-RECORD now. It equals
  // i_row today, and the assert proves that at RUNTIME rather than by
  // argument -- .cu files never see -DNDEBUG, so a table that ever
  // disagrees traps instead of folding silently wrong.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int j = mj + i+turn+1;

  const short* S_H   = &S[seq_off_H[H]];
  const char*  seq_H = &seq[seq_off_H[H]];
  const int ij = Indx2(i,j);
  const unsigned int* Hccc_mb    = &hccc_mb[hc2_off_H[H]];
  const unsigned int* Hccc_mbenc = &hccc_mbenc[hc2_off_H[H]];

  // The gate new_c_host used to build from Ptype(H,ij)/Hard_constraints(H,ij).
  // Same two values, same ij (Indx2 here == jindx[j]+i on the host), just read
  // out of bitmasks by a GPU thread instead of out of two triangles by a host
  // loop walking them at stride ~j.
  gate_row[row_off_H[H]+j] = (char)(Hc2(ij,&hccc_any[hc2_off_H[H]])
                                 | (Hc2(ij,&hccc_gu [hc2_off_H[H]]) << 1));

  //raw_type: the 0->7 fixup must NOT be applied before the rtype[] lookup in
  //energy_mb below -- mb_loop_fast.c:74,105 uses the raw (possibly-0) type as
  //the rtype[] index and only fixes up the *result*. energy_hp/energy_3p_00
  //DO use the fixed-up type directly (hairpin_loops.c:252-255,
  //fill_arrays.c's old type_ local) -- see `type` below.
  const int raw_type = (int)Ptype2(S_H,pair,i,j);
  int type = raw_type;
  if(type == 0) type = 7;

  //energy_hp: vrna_E_hp_loop()/vrna_eval_hp_loop() do no hc check of their
  //own (hairpin_loops.c:226-293) -- fill_arrays_loop.c's read site already
  //gates on hc_decompose/no_close identically to how fill_arrays.c used to
  //gate the write, so compute unconditionally here (see file header comment).
  {
    const int u = j-i-1;
    energy_hp_row[row_off_H[H]+j] =
      E_Hairpin_device(u, type, S_H[i+1], S_H[j-1], &seq_H[i-1], P, salt_loop);
  }

  //energy_mb: mb_loop_fast.c:92-148 (dangle_model==2, cp==-1, sc==NULL path)
  //-- decomp starts at 0 (NOT INF), only overwritten if the MB_LOOP bit is set.
  {
    int decomp = 0;
    if(Hc2(ij,Hccc_mb)){
      int tt = P->rtype[raw_type]; //raw_type, NOT type -- see comment above
      if(tt == 0) tt = 7;
      // mfe_multibranch.c:686 dispatches ml_pair_d0 / ml_pair_d2 here: d0 closes
      // the multiloop with a bare stem, d2 with the two flanking bases. Both
      // read only dmli1, which is why d0 needs no new DP state -- d1/d3 would,
      // and are declined.
      //
      // Belt and braces: mismatchM is already all-zero at d0 (see
      // E_MLstem_device above), so this selection is redundant TODAY.
      const int md2 = (P->dangles == 2);
      decomp = E_MLstem_device(tt, md2 ? S_H[j-1] : -1, md2 ? S_H[i+1] : -1, P)
               + P->MLclosing;
    }
    energy_mb_row[row_off_H[H]+j] = decomp;
  }

  //energy_3p_00: inlined extend_fm_3p() fragment, fill_arrays.c:551-565
  //(cp==-1 forced, so ON_SAME_STRAND(...) is always true) -- default INF,
  //overwritten only if the MB_LOOP_ENC bit is set.
  {
    int e00 = INF;
    if(Hc2(ij,Hccc_mbenc)){
      // Staggered_Row_Batching: the i==1 wrap must land on THIS record's last
      // base, not the batch's. `length` is max(VC[H]->length) over the chunk,
      // so S_H[length] indexed a per-H block of only VC[H]->length+2 shorts:
      // for any H that isn't the longest it silently read the *next* record's
      // bases, and for the last H in a chunk it ran off the end of d_S2
      // entirely (illegal access, reproduced with descending-length input at
      // RNA_GPU_VRAM_BUDGET_MB=8/16 -- the tail chunk there is 13 records with
      // batch max 560 whose last record is only 80nt). Recovered from the
      // offset table rather than a new parameter -- SUPERSEDED by continuous
      // flow phase C1, which took it from d_len_H instead: seq_off_H's stride is
      // the SLOT's capacity+2 now, and this read needs the RECORD's length.
      // Getting that wrong here is exactly the 00d1e07 bug.
      const int length_H = len_H[H];
      // This kernel carried no asserts at all, which is the reason the bug
      // above survived: every S_H read stayed inside the *whole* d_S2
      // allocation for all but the last H of a chunk, so nothing trapped. The
      // two reads below are the only ones in this kernel indexed by anything
      // other than i or j, so bound them against H's own block. j <= length_H
      // because size_off_H is built from VC[H]->length - i - turn per H (see
      // fill_arrays_loop.c), making j < length_H + 1.
      assert(length_H >= 0 && j <= length_H);
      const short s_i1 = (i==1) ? S_H[length_H] : S_H[i-1];
      // extend_fm_3p(), mfe_multibranch.c:956 -- `if (dangle_model == 2)` takes
      // the flanking bases, every other model takes a bare stem. The i==1 wrap
      // above is still evaluated under d0 so its bounds logic stays exercised;
      // only the value is discarded. Redundant today for the same reason as the
      // closing site above.
      const int md2 = (P->dangles == 2);
      e00 = E_MLstem_device(type, md2 ? s_i1 : -1, md2 ? S_H[j+1] : -1, P);
    }
    energy_3p00_row[row_off_H[H]+j] = e00;
  }
}

__device__ inline void
new_c_cell(const int nfiles, const int i_row, const int turn, const int noGUclosure,
             const int*  __restrict__ energy_min2,    //in  d_energy_min2
             const int*  __restrict__ energy_hp_row,  //in
             const int*  __restrict__ energy_mb_row,  //in
             const char* __restrict__ gate_row,       //in
             const int*  __restrict__ dml1,           //in  d_dml1
             const int*  __restrict__ up_hp,          //in  d_up_hp -- NULL when unconstrained
             const size_t* __restrict__ seq_off_H,    //in  up_hp's per-record base
                   int*  __restrict__ new_e,          //out d_new_e
             const int*  __restrict__ stack_row,      //in  noLP: NULL when off
             const int*  __restrict__ cc1,            //in  noLP: previous row
                   int*  __restrict__ cc,             //out noLP: this row
             const size_t* __restrict__ row_off_H,    //in
             const size_t* __restrict__ size_off_H, const size_t total,
                const int* __restrict__ i_H,
            const long long m) {
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  // PHASE A (continuous flow): the row index is PER-RECORD now. It equals
  // i_row today, and the assert proves that at RUNTIME rather than by
  // argument -- .cu files never see -DNDEBUG, so a table that ever
  // disagrees traps instead of folding silently wrong.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int j = mj + i+turn+1;
  const size_t o = row_off_H[H];

  const unsigned char gate = (unsigned char)gate_row[o+j];
  // Pair not evaluated. cc[j] is left at the INF the rotation put there, which
  // is exactly what upstream does -- its noLP block sits inside
  // `if (hc_decompose)` and rotate_aux_arrays() refills cc with INF.
  if(!(gate & 1)) { new_e[o+j] = INF; return; }   // pair not evaluated

  int new_c = energy_min2[o+j];
  if(!(((gate & 2) != 0) && noGUclosure)) {
    // THE HAIRPIN HALF OF A HARD CONSTRAINT. gate bit 0 carries hc->mx, which
    // says whether the PAIR (i,j) may close anything; whether its LOOP may be
    // left unpaired is hc->up_hp, and upstream admits the hairpin only when
    // up_hp[i+1] >= j-i-1 (wrap_hairpin_hc.inc:42-52). Without this a '|'
    // constraint under --enforceConstraint was answered with a hairpin whose
    // loop covered the base it forced to pair -- measured, and better than
    // legal by 1.3 to 17.4 kcal/mol.
    const int hp = (up_hp && (up_hp[seq_off_H[H] + i + 1] < (j - i - 1)))
                   ? INF : energy_hp_row[o+j];
    if(hp < new_c) new_c = hp;
    const int d1 = dml1[o+(j-1)];
    if(d1 != INF) {
      const int e_mb = d1 + energy_mb_row[o+j];
      if(e_mb < new_c) new_c = e_mb;
    }
  }

  // noLP (mfe/mfe.c:4413). The asymmetry is the whole feature: c[ij] receives
  // the STACKED value, while the unconstrained new_c survives only in cc[j]
  // for the next row to stack onto. Writing new_c into c[ij] -- which is what
  // this sweep did before -- lets a lonely pair through, and worse, leaves
  // upstream's backtrack (mfe.c:4289, vrna_bt_stacked_pairs) walking a matrix
  // built under a different convention than the one it assumes.
  if(stack_row) {
    const int se = stack_row[o+j];
    const int p  = cc1[o+(j-1)];
    // INF is a sentinel, not a number: adding to it would wrap into a
    // plausible finite energy. Both operands have to be finite.
    const int stacked = ((p != INF) && (se != INF)) ? (p + se) : INF;
    if(stacked < new_c) new_c = stacked;   // new_c = MIN2(new_c, cc1[j-1]+se)
    cc[o+j]   = new_c;                     // carried sideways to row i-1
    new_e[o+j] = stacked;                  // ... and c[ij] gets the stack only
    return;
  }

  new_e[o+j] = new_c;
}

__device__ inline void
fml_prev_cell(const int nfiles, const int i_row, const int turn,
                const int* __restrict__ energy_min,   //in  d_energy_min
                const int* __restrict__ dml,          //in  d_dml
                      int* __restrict__ fml_prev,     //out d_fml_prev
                const size_t* __restrict__ row_off_H, //in
                const size_t* __restrict__ size_off_H, const size_t total,
                const int* __restrict__ i_H,
               const long long m) {
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  // PHASE A (continuous flow): the row index is PER-RECORD now. It equals
  // i_row today, and the assert proves that at RUNTIME rather than by
  // argument -- .cu files never see -DNDEBUG, so a table that ever
  // disagrees traps instead of folding silently wrong.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int j = mj + i+turn+1;
  const size_t o = row_off_H[H];

  if(mj == 0) fml_prev[o + (i+turn)] = INF;   // the diagonal-band cell, per H

  const int a = energy_min[o+j];
  const int b = dml[o+j];
  fml_prev[o+j] = (a < b) ? a : b;            // MIN2, without the host macro
}

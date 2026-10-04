/* md_chain_cells.inc -- per-cell bodies lifted out of the kernels in modular_decomposition.cu.
 *
 * The fused per-record megakernel and the standalone kernels must run the
 * SAME arithmetic; device code cannot cross a translation unit in this build
 * (no -rdc), so a header is the only way to share it. The bodies are copied
 * verbatim -- only the thread-index prologue became a parameter.
 */

/* Store one final fML value v at (H,i,j) of the int16 triangle. Split out of
 * pack_fml_cell so md_close_row_cell (RNA_MD_TAIL) packs with the SAME code
 * rather than a copy of it; pack_fml_cell reads v from d_fml_row, the close cell
 * computes it in registers. */
__device__ inline void
pack_fml_value(const int v, const int H, const int i, const int j,
                      short* __restrict__ fml_j16,
                      int* __restrict__ fml_b,
                const size_t* __restrict__ tri_off_H,
                const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off) {
  const size_t t  = tri_off_H[H] + Indx(i,j);

  if(v == INF) { fml_j16[t] = FML_INF16; return; }

  const size_t bidx = fml_bidx(base_off_H, colb_off, H, j, i);
  int b = fml_b[bidx];
  if(b == FML_BASE_UNSET) {
    // First non-INF entry of this block. Race-free: rows are separate kernel
    // launches and, within a row, each thread owns a distinct (column, block).
    fml_b[bidx] = b = v;
  }
  const long long d = (long long)v - (long long)b;
  // TRAP, never wrap -- AND IT NOW ACTUALLY TRAPS.
  //
  // This said "TRAP, never wrap" while calling assert(0), which is a NO-OP in
  // every release build because they define NDEBUG. Measured 2026-09-10 with a
  // -P file whose stack entries were all -2000: it printed 48 781 lines, wrapped,
  // and returned a wrong answer on 8 of 12 records (up to 31.8 kcal/mol). Worse,
  // device printf goes to the process STDOUT, so the diagnostics landed inside
  // the fold output.
  //
  // __trap() is not assert(): it is unconditional, survives NDEBUG, and kills
  // the context, so the run dies instead of emitting a plausible wrong answer.
  //
  // Reaching this at all is now a BUG rather than a user error --
  // rnafold_fml_int16_vet_params() declines the encoding on the host before any
  // fold begins. This stays because that bound counts stacking only (tetraloop
  // bonuses and dangles also contribute), so it is a backstop for the gap
  // between "provable" and "proved", not the primary check.
  if(d > 32766 || d < -32766) {
    printf("RNA_FML_INT16 range: H=%d (i=%d,j=%d) value %d baseline %d delta %lld "
           "exceeds int16 -- ABORTING. The host-side vet should have declined "
           "int16 for this parameter table; that it did not is a bug. See "
           "INT16_FML_SCOPE.md.\n", H, i, j, v, b, d);
    __trap();
  }
  fml_j16[t] = (short)d;
}

__device__ inline void
pack_fml_cell(const int nfiles, const int i_row, const int turn, const int length,
                const int* __restrict__ fml_row,
                      short* __restrict__ fml_j16,
                      int* __restrict__ fml_b,
                const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                const size_t* __restrict__ size_off_H, const size_t total,
                const int* __restrict__ i_H,
               const long long m) {
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = mj + i+turn+1 + (int)size_off_H[nfiles + 1];

  pack_fml_value(fml_row[row_off_H[H]+j], H, i, j, fml_j16, fml_b, tri_off_H,
                 base_off_H, colb_off);
}

/* RNA_MD_TAIL: close row i in ONE pass, replacing load_fML + load_min_fML +
 * pack_fml + fml_prev (and fmli, which md no longer needs -- it reads its row
 * operand straight out of energy_min). Over the SIZE range, one thread per cell.
 *
 * Byte-identity with the four kernels it replaces rests on reproducing each old
 * writer's rule exactly, and they are NOT the same rule:
 *
 *   triangle  load_fML wrote energy_min over the whole size range, then
 *             load_min_fML overwrote the SIDE range (mj >= turn+1) with
 *             min(energy_min, dml). Below the side range it stays RAW.
 *   fml_prev  min(energy_min, dml) over the whole size range, plus INF on the
 *             diagonal-band cell (i, i+turn).
 *
 * Below the side range dml is INF (md never writes there), so the two agree
 * unless energy_min exceeds INF -- which the asymmetric INF guard can produce
 * (INT16_FML_SCOPE.md). The triangle keeps that value and fml_prev clamps it,
 * so writing one shared value would change the answer on exactly the inputs
 * that are hardest to test. Hence the `side` test below. */
__device__ inline void
md_close_row_cell(const int nfiles, const int i_row, const int turn,
                  const int* __restrict__ energy_min,
                  const int* __restrict__ dml,
                        int* __restrict__ fml_prev,
                        int* __restrict__ fml_j,       //int32 path; unused when fml_j16
                        short* __restrict__ fml_j16,   //int16 path, NULL when off
                        int* __restrict__ fml_b,
                  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
                  const size_t* __restrict__ base_off_H, const size_t* __restrict__ colb_off,
                  const size_t* __restrict__ size_off_H, const size_t total,
                  const int* __restrict__ i_H,
                  const long long m,
                  /* RNA_ROW_BATCH: this row's slot of the row-major stage, or NULL.
                   * When set, the triangle value goes HERE -- one coalesced store --
                   * and md_flush_rows_kernel writes K staged rows into the triangle
                   * at once, contiguous per column. NULL: the triangle, as before. */
                        int* __restrict__ stage = NULL) {
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  /* ROW-local (3a.2): the triangle's rule below and the band cell key on the position in
   * the ROW, so the column shift goes into mj itself, not only into j */
  const long long mj = (long long)m - (long long)size_off_H[H] + (long long)size_off_H[nfiles + 1];
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);
  const int j = mj + i+turn+1;
  const size_t o = row_off_H[H];

  const int e  = energy_min[o+j];
  const int d  = dml[o+j];
  const int mn = (e < d) ? e : d;

  /* No diagonal-band write here any more (blocked md 3b): the scan treats (i+1, i+turn+1)
   * as INF itself. Written from row i's FIRST step, this landed in the column block to
   * the left -- which, in column-block-major order, lower rows had already written with
   * real values -- and clobbered the lowest row's fML there (failed whenever CB < RB). */
  fml_prev[o+j] = mn;

  const int v = (mj >= turn+1) ? mn : e;   // the triangle's rule -- see above
  if(stage)        stage[o+j] = v;
  else if(fml_j16) pack_fml_value(v, H, i, j, fml_j16, fml_b, tri_off_H, base_off_H, colb_off);
  else             fml_j[tri_off_H[H] + Indx(i,j)] = v;
}

__device__ inline void
load_fML_cell(const int nfiles, const int i_row, const int turn, const int length,
		const int* __restrict__ energy_min,
	              int* __restrict__ fml_j,
	              int* __restrict__ fml_row,   //int16 path: NULL when off
		const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
		const size_t* __restrict__ size_off_H, const size_t total,
		const int* __restrict__ i_H,
               const long long m,
               /* RNA_MD_PRUNE: block minima of the triangle, one int per 32
                * consecutive cells. NULL when off. Maintained HERE, in the
                * kernel that already writes the cell, so it costs no extra
                * launch -- which is the mistake band_fill made. */
               int* __restrict__ fml_jmin = NULL) {
  // Staggered_Row_Batching Phase 4: flat index -> (H, position) via
  // flatten_index_to_H() over this row's real per-H active width
  // (size_off_H), replacing the old uniform m/nfiles split. j is guaranteed
  // <= length by size_off_H's own construction (built from
  // length_H[H]-i-turn, clamped >=0) -- no bound check needed here anymore.
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, size_off_H, nfiles);
  const long long mj = (long long)m - (long long)size_off_H[H];
  // Continuous flow phase A2: this record's own row index. Identical to the old
  // shared scalar i_row today, and the assert checks that at RUNTIME, because it
  // is precisely the property phase B stops holding.
  const int i = i_H[H];
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  const long long j  = mj + i+turn+1 + (int)size_off_H[nfiles + 1];

  assert(H >= 0 && H < nfiles);
  assert(j>=0 && j<=length);
  const long long ij = Indx(i,j);
  assert(ij>=0 && ij<Hoff(1,length));
  if(fml_row) {
    // int16 path: the row's cells stay in full precision until pack_fml_kernel
    // closes the row. `ij` is unused here then, but the asserts above still
    // check it.
    (void)ij;
    fml_row[row_off_H[H]+j] = energy_min[row_off_H[H]+j];
  } else {
    assert(fml_j[tri_off_H[H]+ij] == INF);
           fml_j[tri_off_H[H]+ij] = energy_min[row_off_H[H]+j];
    if (fml_jmin) {
      const size_t idx = tri_off_H[H]+ij;
      atomicMin(&fml_jmin[idx >> 5], energy_min[row_off_H[H]+j]);
    }
  }
}

__device__ inline void
load_min_fML_cell(const int nfiles, const int i_row, const int turn, const int length,
		    const int* __restrict__ energy_min,
		    const int* __restrict__ dml,     //in  d_dml   DMLi
		          int* __restrict__ fml_j,     //out d_fml_j my_fML
		          int* __restrict__ fml_row,   //int16 path: NULL when off
		    const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
		    const size_t* __restrict__ side_off_H, const size_t total,
		    const int* __restrict__ i_H,
                   const long long m) {
  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, side_off_H, nfiles);
  const long long mj = (long long)m - (long long)side_off_H[H];
  const int i = i_H[H];   // continuous flow phase A2 -- see load_fML_kernel
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows

  const int j  = mj + (i + 2*(turn+1)) + 1 + (int)side_off_H[nfiles + 1];
  const long long ij = Indx(i,j);

  assert(H >= 0 && H < nfiles);
  assert(j >=0 && j<=length);
  assert(ij>=0 && ij<Hoff(1,length));

  if(fml_row) fml_row[row_off_H[H]+j] = MIN2(energy_min[row_off_H[H]+j],dml[row_off_H[H]+j]);
  else        fml_j[tri_off_H[H]+ij]   = MIN2(energy_min[row_off_H[H]+j],dml[row_off_H[H]+j]);
}

__device__ inline void
fmli_cell(
  const int nfiles, const int i_row, const int turn, const int length,
        int* __restrict__ fml_i,   //out
  const int* __restrict__ fml_j,   //In  d_fml_j
  const int* __restrict__ fml_row, //In  d_fml_row -- int16 path, NULL when off
  const size_t* __restrict__ tri_off_H, const size_t* __restrict__ row_off_H,
  const size_t* __restrict__ side_off_H, const size_t total,
  const int* __restrict__ i_H,
           const long long m) {

  if((size_t)m >= total) return;
  const int H = flatten_index_to_H((size_t)m, side_off_H, nfiles);
  const long long mj = (long long)m - (long long)side_off_H[H];
  const int i = i_H[H];   // continuous flow phase A2 -- see load_fML_kernel
  assert(i_row < 0 || i == i_row);   // i_row<0: continuous flow, records are on different rows
  // `start` moved below the flatten: it depends on i, which is per-record now
  // and so is not known until H is.
  const int start = i+turn+1;

  const int k  = start + mj;
  const long long ik = Indx(i,k);
  assert(H >= 0 && H < nfiles);
  // Staggered_Row_Batching Phase 2d: fml_i is now table-driven (row_off_H),
  // so it can no longer be written via the flat H-tightest index m -- mj is
  // fml_i's within-row position (0-based, same value the old H+mj*nfiles
  // convention used), so row_off_H[H]+mj is the equivalent table-driven cell.
  // int16 path: row i is exactly what fml_row holds, and it is CONTIGUOUS there
  // rather than strided across columns at Indx(i,k) -- so this read gets
  // cheaper, not harder. It must not read the packed triangle: row i is not
  // final yet (load_min_fML_kernel has not run), which is the whole reason the
  // staging buffer exists.
  if(fml_row) { (void)ik; fml_i[row_off_H[H]+mj] = fml_row[row_off_H[H]+k]; }
  else                    fml_i[row_off_H[H]+mj] = fml_j[tri_off_H[H]+ik]; //ith column
}

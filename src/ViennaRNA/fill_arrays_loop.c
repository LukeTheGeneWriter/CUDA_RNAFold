//WBL 10 Dec 2017 $Revision: 1.24 $ GGGP ViennaRNA-2.3.0 rf/rf/
//Helper for fill_arrays.c -> mfe.c for eventual CUDA version

//WBL 27 Jan 2018 Add loop H for nfiles different structures

// Staggered_Row_Batching Phase 6d: the host-side join mask.
//
// `length` (fill_arrays.c) is now max(VC[H]->length) over the batch rather
// than VC[0]'s, so the shared sweep covers the longest sequence and every
// shorter H *joins late*: it is inactive for rows above its own length and
// active from i == VC[H]->length-turn-1 down to 1, alongside everyone else.
// (This is the shared-i + join-mask design, deliberately not independent
// per-H sweep positions with early retirement -- see the branch notes.)
//
// The GPU side already gets this for free: the per-row width tables built
// below (length_H[H]-i-turn and length_H[H]-i-2*turn-2, both clamped >=0)
// go to zero for a not-yet-joined H, so flatten_index_to_H() simply never
// hands a thread to it. The host loops need the check spelled out, and not
// only for their j bounds -- several of them index by `i` per H before the
// j-loop even starts (VC[H]->hc->up_ml[i] in the fml_host loop), which reads
// off the end of a short H's arrays. So skip the whole H, don't just clamp j.
//
// Degenerates to "always true" while chunks are uniform-length, which is what
// keeps this phase a no-op until Phase 6c actually admits mixed lengths.
#define HAS_JOINED(H) ((i) + (turn) + 1 <= (int)VC[(H)]->length)

 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    // Staggered_Row_Batching Phase 2d: table-driven per-H row offset,
    // replacing the H-tightest H+j*nfiles convention.
    // Phase 6d: `length` is now max(VC[H]->length) across the batch, so every
    // host loop in this row body has to bound itself by its OWN H's length and
    // skip H entirely on rows above it -- HAS_JOINED() below. The shared bound
    // would spill past a short H's row into the next H's.
    // GPU-resident sweep (RNA_GPU_SWEEP): in device mode nothing on the host
    // reads energy_min this row -- the three host loops below are skipped and
    // int_loop_i's D2H into it is gated off -- and int_loop_kernel writes every
    // cell it later reads unconditionally (store is outside all conditionals,
    // grid range == read range), so the reset is dead. See the plan's
    // "no INF fill needed in device mode".
    if(!rnafold_gpu_sweep())
    for (int H=0;H<nfiles; H++) {
    if(!HAS_JOINED(H)) continue;
    for (j = i+turn+1; j <= (int)VC[H]->length; j++) energy_min[row_off_H[H]+j] = INF;
    }

    // Staggered_Row_Batching Phase 5: this row's "size" active-width table
    // (length_H[H]-i-turn, clamped >=0) -- built once here, up front, since
    // it's now shared by int_loop_i()/hp_mb_3p_i()/load_my_c() (the 3
    // "rectangular" kernels, Phase 5) as well as load_fML() (Phase 4, still
    // computed separately below for side_off_H, which nothing here needs).
    size_t size_off_H[nfiles+1];
    {
      size_t size_H[nfiles];
      for(int H=0;H<nfiles;H++) {
        const int size_raw = (int)VC[H]->length - i - turn;
        size_H[H] = (size_raw>0) ? (size_t)size_raw : 0;
      }
      compute_flatten_offsets(nfiles, size_H, size_off_H);
    }

    // Continuous flow, PHASE A: the per-record row index. Every entry is the
    // shared i today, so the kernels are unchanged in effect -- phase A exists
    // to move the row index onto a per-record table WITHOUT changing behaviour,
    // so that phase B can let records retire at their own rows. The kernels
    // assert(i_H[H] == i_row), which turns any future divergence into a trap
    // rather than a silently wrong fold.
    int i_H[nfiles];
    for(int H=0;H<nfiles;H++) i_H[H] = i;

    {
      const double t0 = now_seconds();
      int_loop_i(nfiles,VC,i,turn,length,/*indx,ijsize,
		 hard_constraints, my_c,*/
		 energy_min, //replaces vrna_E_int_loop(vc, i, j);
		 size_off_H);
      phase_int_loop_s += now_seconds() - t0;
    }

    //hairpin-loop / multibranch-loop / 3'-extension energies for this row,
    //computed fresh on GPU each i rather than precomputed as a full
    //nfiles*ijsize array (see fill_arrays.c) -- same row-buffer pattern as
    //energy_min/int_loop_i above.
    {
      const double t0 = now_seconds();
      hp_mb_3p_i(nfiles,VC,i,turn,length,energy_hp_row,energy_mb_row,energy_3p00_row,gate_row,size_off_H,i_H);
      phase_hp_mb_s += now_seconds() - t0;
    }

    //could pack new_C more tightly for load_my_c_kernel but expect modest savings
    const double new_c_host_t0 = now_seconds();
    // GPU-resident sweep (RNA_GPU_SWEEP): new_c_kernel (new_c_i, below) already
    // wrote d_new_e directly from device buffers, and load_my_c's H2D of new_C
    // over it is gated off, so this loop has no consumer in device mode.
    if(!rnafold_gpu_sweep())
    for (int H=0;H<nfiles; H++) {
    if(!HAS_JOINED(H)) continue; // Phase 6d
    for (j = i+turn+1; j <= (int)VC[H]->length; j++) {
      new_C[row_off_H[H]+j] = INF;
      // Both of these used to be triangle reads -- Ptype(H,ij) and
      // Hard_constraints(H,ij), each a stride-~j cache miss per (H,j), and
      // together 3.2 s of workload A. They index static per-(i,j) data, so
      // unlike the my_c/fML mirrors there was no row buffer already holding
      // them; hp_mb_3p_kernel now packs both into gate_row as it sweeps the
      // same j range a moment earlier. Bit 0 is hc->matrix[ij] != 0, bit 1 is
      // "ptype[ij] is a GU/UG closing pair", both taken from the host's own
      // arrays at init_gpu3() time, not re-derived.
      const unsigned char gate = (unsigned char)gate_row[row_off_H[H]+j];
      hc_decompose  = gate & 1;
      no_close      = ((gate & 2) != 0) && noGUclosure;

      //fprintf(stderr,"i %2d, j %2d, hard_constraints[%3d] %2d, ptype[%3d] %d, no_close %d ",
      //      i,j,ij,hard_constraints[ij],ij,ptype[ij],no_close);
      //fflush(stderr);
      /*moved to int_loop_i **
      if (hc_decompose) {   ** we evaluate this pair **
        new_c = INF;

        ** check for interior loops **
        energy = vrna_E_int_loop(vc, i, j);
	//fprintf(stderr,"vrna_E_int_loop(vc, %d, %d)returned %d ",
	//	i,j,energy);
	//fflush(stderr);
        new_c = MIN2(new_c, energy);
	energy_min[j] = new_c;
      } ** end >> if (pair) << */

      if (hc_decompose) {   /* we evaluate this pair */
	new_c = energy_min[row_off_H[H]+j];

        if(!no_close){
          /* check for hairpin loop */
          /*energy_hp[ij] = energy = vrna_E_hp_loop(vc, i, j); */
          new_c = MIN2(new_c, energy_hp_row[row_off_H[H]+j]);

          /* check for multibranch loops */
          //energy  = vrna_E_mb_loop_fast(vc, i, j, DMLi1, DMLi2);
	  const int e_mb = (DMLi1[row_off_H[H]+(j-1)] != INF)? DMLi1[row_off_H[H]+(j-1)] + energy_mb_row[row_off_H[H]+j] : INF;
          new_c   = MIN2(new_c, e_mb);
        }

        /*gov says not used if(dangle_model == 3){ ** coaxial stacking * E_mb_loop_stack(i, j, vc);*/

        /* gcov says not used  remember stack energy for --noLP option * if(noLP) vrna_E_stack(vc, i, j) cc[j] = new_c */
          // My_c(H,ij) = new_c dropped here: d_my_c already receives exactly
          // this value from load_my_c_kernel (new_C is uploaded and written to
          // d_my_c[tri_off_H[H]+Indx(i,j)] a few lines below), and the host
          // triangle has no reader until E_ext_loop_5()/backtrack() after the
          // sweep. fetch_my_c() fills it in one contiguous copy per record.
	  new_C[row_off_H[H]+j]    = new_c;
      } /* end >> if (pair) << */

      else {
        // Nothing to do: new_C[..j] was set to INF at the top of this
        // iteration, d_my_c is INF from init_my_c(), and the host triangle is
        // filled after the sweep by fetch_my_c(). The My_c(H,ij) = INF store
        // that used to be here was writing INF over INF, at stride ~j.
      }

    } /* end of j-loop */
    }//endfor H
    phase_new_c_host_s += now_seconds() - new_c_host_t0;

    // GPU-resident sweep, step 3: the same row, computed on the device from
    // five buffers the GPU already had -- int_loop's energies, hp_mb_3p's three
    // outputs, and the previous row's DMLi. Deliberately between the host loop
    // (so RNA_ROW_VERIFY has something to compare) and load_my_c (which uploads
    // the host's new_C over d_new_e, so the readback must precede it and the
    // sweep still consumes the host's values either way).
    new_c_i(nfiles, i, turn, noGUclosure,
            rnafold_gpu_sweep() ? NULL : new_C,  // no host result to verify against in device mode
            row_off_H, size_off_H, i_H);

    {
      const double t0 = now_seconds();
      load_my_c(nfiles,i,turn,length,new_C,size_off_H,i_H); //keep my_c on GPU instep with my_c
      phase_load_my_c_s += now_seconds() - t0;
    }

    const double fml_host_t0 = now_seconds();
    // GPU-resident sweep (RNA_GPU_SWEEP): fml_scan_kernel (fml_scan_i, below)
    // already wrote d_energy_min with this recurrence, and the graph trio's H2D
    // of energy_min over it is gated off, so this loop has no consumer in
    // device mode.
    if(!rnafold_gpu_sweep())
    for (int H=0;H<nfiles; H++) {
    // Phase 6d: must precede the en_i computation below, not just guard the
    // j-loop -- VC[H]->hc->up_ml[i] reads past a not-yet-joined H's array.
    if(!HAS_JOINED(H)) continue;
      /*  extension with one unpaired nucleotide at 5' site
	  and all other variants which are needed for odd
	  dangle models -- per-H (was incorrectly computed once from
	  VC[0] only and reused for every H, see energy_3p_en_j below
	  for the already-correct per-H sibling of this check)
      */
      const int cp = -1;
      const int en_i = (ON_SAME_STRAND(i - 1, i, cp) &&
                         ON_SAME_STRAND(i, i + 1, cp) &&
                         VC[H]->hc->up_ml[i] > 0) ? P->MLbase : INF;
    for (j = i+turn+1; j <= (int)VC[H]->length; j++) {
      // No `ij = Indx(H,i,j)` here any more: this loop's last two triangle
      // accesses (My_fML(H,ij+1) and My_c(H,ij)) are gone, so the index was
      // feeding nothing but an assert -- and asserts in this file are
      // compiled out anyway (-DNDEBUG reaches mfe_cuda.c via the conda
      // CPPFLAGS; the .cu files escape it). Everything below is row-indexed.
      /* done with c[i,j], now compute fML[i,j] and fM1[i,j] */

      //my_fML[ij] = vrna_E_ml_stems_fast(vc, i, j, Fmi, DMLi);

      /*  extension with one unpaired nucleotide at the right (3' site)
	  or full branch of (i,j)
      */
      //from extend_fm_3p()...
      //const int cp = -1;
      int  e00           = INF;
      int  en0           = INF;

  // c(i,j) out of the row buffer rather than the triangle. new_c_host, a few
  // dozen lines up, set new_C[..j] to exactly what it set My_c(H,ij) to --
  // new_c when the pair is evaluated, INF otherwise -- so the two are equal by
  // construction, and this read is sequential where My_c(H,ij) was stride ~j.
  // Worth 2.2 s of workload A on its own.
  e00 = (energy_3p00_row[row_off_H[H]+j] != INF)? new_C[row_off_H[H]+j] + energy_3p00_row[row_off_H[H]+j] : INF;
  //energy_3p_en is just P->MLbase behind a hard-constraint check on
  //already-host-resident data -- not worth a GPU kernel, computed inline
  const int energy_3p_en_j = (VC[H]->hc->up_ml[j] > 0) ? P->MLbase : INF;
  // fML(i,j-1), read out of a row buffer instead of the fML triangle. This
  // loop set My_fML(H,Indx(H,i,j-1)) = energy_min[..j-1] one iteration ago
  // (the write that used to sit at the bottom of this loop), so the row
  // buffer already holds it -- and reading it here is sequential where the
  // triangle walk was stride ~j. j == i+turn+1 is the exception: j-1 is then
  // i+turn, inside the diagonal band, a cell no row ever computes and which
  // the fML prefill leaves at INF.
  const int fml_i_jm1 = (j == i+turn+1) ? INF : energy_min[row_off_H[H]+(j-1)];
  en0 = ((fml_i_jm1 != INF) && (energy_3p_en_j != INF))? fml_i_jm1 + energy_3p_en_j : INF;
  e00 = MIN2(e00, en0);
      //end from extend_fm_3p()...

      //const int e0 = extend_fm_3p(i, j, my_fML, vc);

      // fML(i+1,j) -- ij+1 == Indx(H,i+1,j) -- out of the previous row's
      // final-fML row cache instead of the triangle. NB this deliberately is
      // NOT reconstructed from energy_min[..j] + DMLi1[..j]: energy_min is
      // reused twice per row (reset at the top of the i-loop, filled with
      // interior-loop energies by int_loop_i, and only overwritten with the
      // fML extension value at the bottom of this loop), so at this point it
      // holds row i's int_loop energies, not row i+1's fML. fml_prev is
      // written once per row from the values that ARE right, just below.
      const int fml_i1_j = fml_prev[row_off_H[H]+j];
      const int e3 = (fml_i1_j != INF)? fml_i1_j + en_i : INF;


      //energy_mls (multiloop-stems-fast) deleted: under dangle_model==2
      //(enforced in fill_arrays.c) it always evaluated to INF, so
      //MIN2(e3,energy_mls[...]) always reduced to e3 -- see fill_arrays.c.
      energy_min[row_off_H[H]+j] = MIN2(e00,e3); //e1 e31
//    } /* end of j-loop */
//
      // The provisional `My_fML(H,ij) = energy_min[..j]` that used to be here
      // is gone. Its only intra-sweep reader was the fml_i_jm1 read above,
      // now served from the row buffer; the host fML triangle itself is
      // filled once after the sweep by fetch_fML() from d_fml_j, which the
      // GPU has been maintaining all along. Dropping this strided store is
      // worth more than the store itself: it also stops evicting the
      // neighbouring loops' working set.
    } /* end of j-loop */
    }//endfor H
    phase_fml_host_s += now_seconds() - fml_host_t0;

    // GPU-resident sweep, step 4: the same recurrence, run on the device as an
    // inclusive scan over affine min-plus maps. Between the host loop (so
    // RNA_ROW_VERIFY has something to compare) and the graph trio below, which
    // uploads the host's energy_min over d_energy_min -- so the readback must
    // precede it and the sweep still consumes the host's values either way.
    fml_scan_i(nfiles, i, turn,
               rnafold_gpu_sweep() ? NULL : energy_min,  // no host result to verify against in device mode
               row_off_H, size_off_H, i_H);

    //load_fML + modular_decomposition_i + load_min_fML fused into one CUDA
    //graph capture/replay (no host CPU logic runs between these three calls,
    //which is what makes that legal) -- updates my_fML GPU, then
    //my_fML GPU = MIN2(energy_min[j], DMLi[j])
    {
      const double t0 = now_seconds();
      // Staggered_Row_Batching Phase 4: side_off_H (shared by fmli/
      // modular_decomposition/load_min_fML) is the only table still built
      // here -- size_off_H was already built at the top of this row's loop
      // body (Phase 5), where int_loop_i()/hp_mb_3p_i()/load_my_c() now need
      // it too.
      size_t side_off_H[nfiles+1];
      {
        size_t side_H[nfiles];
        for(int H=0;H<nfiles;H++) {
          const int side_raw = (int)VC[H]->length - i - 2*turn - 2;
          side_H[H] = (side_raw>0) ? (size_t)side_raw : 0;
        }
        compute_flatten_offsets(nfiles, side_H, side_off_H);
      }

      load_fML_modular_decomposition_load_min_fML(nfiles,i,turn,length,energy_min,DMLi,row_off_H,size_off_H,side_off_H,i_H);
      phase_modular_decomp_s += now_seconds() - t0;
    }

    // Was my_fml_update_host, which wrote MIN2(energy_min[..j], DMLi[..j])
    // into the fML *triangle* at stride ~j. load_min_fML_kernel had already
    // computed exactly that into d_fml_j a moment earlier on the GPU, and the
    // host triangle has no reader until backtrack(), so it is now filled once
    // after the sweep by fetch_fML(). What survives here is only the part the
    // sweep itself still needs: row i's final fML, cached row-shaped for the
    // fml_i1_j read one row later. Same arithmetic, contiguous destination
    // instead of a strided one -- which is what the cost was.
    const double fml_prev_host_t0 = now_seconds();
    // GPU-resident sweep (RNA_GPU_SWEEP): fml_prev_kernel (fml_prev_i, below)
    // already wrote d_fml_prev, and DMLi's D2H is gated off, so this loop has
    // no consumer in device mode.
    if(!rnafold_gpu_sweep())
    for (int H=0;H<nfiles; H++) {
      if(!HAS_JOINED(H)) continue;
      // The cell one below this row's first, (i, i+turn), is inside the
      // diagonal band -- never computed, INF in the fML prefill. Row i-1 will
      // read it as its own j == i+turn, so write it explicitly rather than
      // leaving row i+1's value there.
      fml_prev[row_off_H[H]+(i+turn)] = INF;
      for (int jj = i+turn+1; jj <= (int)VC[H]->length; jj++)
        fml_prev[row_off_H[H]+jj] = MIN2(energy_min[row_off_H[H]+jj],
                                         DMLi[row_off_H[H]+jj]);
    }
    phase_fml_prev_host_s += now_seconds() - fml_prev_host_t0;

    // GPU-resident sweep, step 2: the same row, computed on the device from
    // d_energy_min and d_dml -- both of which the GPU already had, which is why
    // this loop should never have been on the host. Runs AFTER the host loop so
    // RNA_ROW_VERIFY has something to compare against; nothing reads d_fml_prev
    // yet, so the sweep's behaviour is unchanged either way.
    fml_prev_i(nfiles, i, turn,
               rnafold_gpu_sweep() ? NULL : fml_prev,  // no host result to verify against in device mode
               row_off_H, size_off_H, i_H);

    // GPU-resident sweep, step 1: the device twin of the DMLi1 rotation below.
    // Publishes row i's DMLi as "the previous row's" for row i-1, which is what
    // new_c_kernel will read as DMLi1[j-1] once new_c_host moves to the GPU.
    // Placed here, at exactly the host's rotate point, so the two representations
    // cannot drift. Nothing reads d_dml1 yet -- this is behaviour-neutral, and
    // costs one 3.3 MB device-to-device copy per row (~0.18 s over a whole run).
    md_snapshot_dml();

    {
      int *FF; /* rotate the auxilliary arrays */
      FF = DMLi2; DMLi2 = DMLi1; DMLi1 = DMLi; DMLi = FF;
    }
  } /* end of i-loop */



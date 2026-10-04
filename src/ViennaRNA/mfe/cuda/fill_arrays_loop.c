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
// Continuous flow PHASE B replaces the join mask with an ACTIVE mask. A record
// is active on this iteration if it has a row of its own to compute:
// i_H[H] >= 1 (it has not retired) and that row is inside its own triangle.
//
// Off the flow path i_H[H] == i for every H and i >= 1 by the loop bound, so
// this is exactly the old HAS_JOINED() test, character for character in effect.
#define IS_ACTIVE(H) (i_H[(H)] >= 1 && i_H[(H)] + (turn) + 1 <= (int)VC[(H)]->length)

// One row's two width tables, from that row's per-record row index IH:
//   SO  "size", length_H[H]-i-turn        (int_loop, hp_mb_3p, load_my_c, load_fML)
//   SD  "side", length_H[H]-i-2*turn-2    (fmli, modular_decomposition, load_min_fML)
// both clamped >= 0, and 0 for a RETIRED record (IH[H] < 1) -- without that
// guard the width would be computed from the sentinel and come out positive.
// A macro because this file is included into the middle of a function body.
#define ROW_WIDTH_TABLES(IH, SO, SD) do {                                              size_t w_so_[nfiles], w_sd_[nfiles];                                              for(int h_=0; h_<nfiles; h_++) {                                                    const int so_ = ((IH)[h_] >= 1) ? ((int)VC[h_]->length - (IH)[h_] - turn) : 0;             const int sd_ = ((IH)[h_] >= 1) ? ((int)VC[h_]->length - (IH)[h_] - 2*turn - 2) : 0;       w_so_[h_] = (so_ > 0) ? (size_t)so_ : 0;                                          w_sd_[h_] = (sd_ > 0) ? (size_t)sd_ : 0;                                        }                                                                                 compute_flatten_offsets(nfiles, w_so_, (SO));                                     compute_flatten_offsets(nfiles, w_sd_, (SD));                                   } while(0)

 // Continuous flow PHASE B: the per-record row index, now hoisted out of the
 // loop because under flow it CARRIES from one iteration to the next.
 //
 // OFF (default): rewritten to the shared i at the top of every iteration, so
 // every record sits on the same row and the kernels' assert(i == i_row) holds.
 //
 // ON: each record starts at its OWN top row -- so it works from iteration 0
 // instead of idling until the shared counter comes down to it -- and stops
 // when it reaches row 1, i.e. i_H[H] == 0 means retired. Every record still
 // advances exactly ONE row per iteration, which is what keeps every piece of
 // per-row machinery valid: the DMLi/DMLi1 rotation, md_snapshot_dml(), the
 // graph capture, and all the per-row offset tables.
 //
 // The iteration count is unchanged either way (length-turn-1, set by the
 // LONGEST record), and each record still computes exactly its own rows in the
 // same order -- so this moves WHEN work happens, not how much. A record
 // shorter than turn+2 has no rows at all and starts retired.
 int i_H[nfiles];
 // Continuous flow phase C3: a schedule IMPLIES continuous flow -- slots take
 // their next record as soon as the current one finishes, which only means
 // anything if each record is on its own row.
 const int continuous_flow = rnafold_continuous_flow() || (sched != NULL);
 // How far each slot has got through its queue. 0 == the occupant installed by
 // par_mfe() before the sweep started.
 int q_pos[nfiles];
 for(int H=0;H<nfiles;H++) q_pos[H] = 0;
 // Phase B instrumentation. The plan expects continuous flow to cut the row
 // count, so count what the sweep actually launches rather than arguing about
 // it: iterations, (record, iteration) pairs that had a row to compute, and the
 // summed per-row widths. Clock-independent, so it is meaningful even on a
 // throttling box. One extra pass over nfiles per row, next to the several this
 // loop body already does.
 long long cf_iters = 0, cf_rows = 0, cf_cells = 0;
 // The totals above are conserved by phase B (it recomputes the same rows in a
 // different order), so they alone cannot show what it did. The PEAKS can: what
 // continuous flow changes is the per-iteration shape -- every live record is at
 // the same width, instead of the batch piling its widest rows into the last
 // iterations. A lower peak at the same total is a flatter sweep.
 long long cf_peak_cells = 0, cf_peak_rows = 0;
 for(int H=0;H<nfiles;H++) {
   const int top = (int)VC[H]->length - turn - 1;
   i_H[H] = (continuous_flow && top >= 1) ? top : 0;
 }

 // The sweep runs until the busiest slot has finished its whole queue. Without
 // a schedule that is just the longest record, exactly as before.
 int sweep_iters = length-turn-1;
 if(sched) {
   sweep_iters = 0;
   for(int s=0;s<nfiles;s++) {
     int tot = 0;
     for(int q=sched->qoff[s]; q<sched->qoff[s+1]; q++) {
       const int rl = (int)sched->VC_all[sched->queue[q]]->length - turn - 1;
       if(rl > 0) tot += rl;
     }
     if(tot > sweep_iters) sweep_iters = tot;
   }
 }

 // Luke's Flow Batching, fix 3: THE CHUNK'S ROW TABLES (device.cu). One slot
 // per iteration, written once. In lock-step every row's tables depend only on
 // the record lengths, so they are all built here and go to the device in ONE
 // pinned upload -- the sweep then issues no per-row table traffic at all, and
 // no queued kernel on any stream can see a table change under it. Under
 // continuous flow a row is only known when its iteration arrives (slots turn
 // over mid-sweep), so each slot is built and uploaded as its row comes up.
 rnafold_rowtab_begin(nfiles, sweep_iters);
 if(!continuous_flow) {
   for(int r = sweep_iters; r >= 1; r--) {
     int* ih = rnafold_rowtab_ih_host(r);
     for(int H=0;H<nfiles;H++) ih[H] = r;
     ROW_WIDTH_TABLES(ih, rnafold_rowtab_size_host(r), rnafold_rowtab_side_host(r));
   }
   rnafold_rowtab_upload_all();
 }

 // ---- RNA_MEGAKERNEL: the whole row chain as one cooperative kernel per
 // record (megakernel.cu). SAME arithmetic -- every phase calls the same
 // *_cell() the per-phase kernels do, out of the shared headers -- so this is a
 // different SCHEDULE, not a different recurrence, and the sha must not move.
 //
 // Why it lives here rather than in engine.c's routing guard: that guard
 // answers "can the CUDA backend do this at all", and there is no way there to
 // say "supported, but not by the fused kernel". So the decision is taken at
 // the sweep, and anything refused simply runs the loop below.
 int mk_done = 0;
 if(rnafold_megakernel()) {
   int depot = 0;
   const char *why;
   for(int H=0; H<nfiles; H++)
     if(VC[H]->hc && VC[H]->hc->depot) { depot = 1; break; }
   why = rnafold_megakernel_refuse(nfiles,
                                   P->model_details.circ, P->model_details.gquad,
                                   noLP, uniq_ML, depot, P->model_details.dangles,
                                   continuous_flow);
   if(why) {
     fprintf(stderr,"%-24s RNA_MEGAKERNEL declined: %s -- running the per-phase sweep\n",
             __FILE__, why);
   } else {
     int *slots = (int*)malloc((size_t)nfiles*sizeof(int));
     int *lens  = (int*)malloc((size_t)nfiles*sizeof(int));
     double t0;
     int rc;
     for(int H=0; H<nfiles; H++) { slots[H] = H; lens[H] = (int)VC[H]->length; }
     t0 = now_seconds();
     rc = rnafold_megakernel_sweep(nfiles, slots, nfiles, turn, length, lens,
                                   noGUclosure, P->TerminalAU, P->ninio[2],
                                   (float)P->lxc);
     free(slots); free(lens);
     if(rc == 0) {
       mk_done = 1;
       fprintf(stderr,"%-24s megakernel wall=%.3f s for %d records\n",
               __FILE__, now_seconds() - t0, nfiles);
     } else {
       fprintf(stderr,"%-24s megakernel could not launch -- running the per-phase sweep\n",
               __FILE__);
     }
   }
 }

 // mk_done starts the loop at 0 so its body never runs, while everything AFTER
 // it -- slot retirement, the row-table release, the shape line -- still happens
 // exactly once. Under the fused path the per-phase timers stay at zero by
 // construction: there are no phases on the host to time, and the device-side
 // clock shares megakernel.cu prints are the measurement instead.
 const int md3_probe_n = rnafold_md3_launch_probe();

 /* BLOCKED MD STAGES 3a'/3b (PORT_MD_BLOCKING_DRIVER.md 2, 3.2): the sweep as a sequence
  * of STEPS, each one (row i, column block from tile_jlo).
  *
  *   no tiles (the default)   one step per row, rows descending, the whole row: exactly
  *                            the old `for (i = sweep_iters; i >= 1; i--)`.
  *   RNA_MD_TILE_CB=N, RB=1   (3a') each row as N-column blocks, left to right.
  *   RNA_MD_TILE_RB=RB > 1    (3b) block-rows of RB rows; for each column block, left to
  *                            right, the block-row's rows top down -- COLUMN-BLOCK MAJOR.
  *
  * A row's first step does its per-row head (stats, the launch probe, noLP's cc refill)
  * and its last step its per-row tail. The rotating rows are SELECTED by row at every
  * step (rnafold_md_ring_select / rnafold_cc_ring_select) rather than advanced, since
  * rows interleave. RNA_MD_TILE_REVERSE runs the column blocks right to left: 3b's
  * negative control, which must give wrong answers. */
 const int tile_cb  = rnafold_md_tile_cb();
 const int tile_rb  = tile_cb ? rnafold_md_tile_rb() : 1;
 const int tile_rev = tile_cb ? rnafold_md_tile_reverse() : 0;
 const int tile_fuse = tile_cb ? rnafold_md_tile_fuse() : 0;   /* 3e-fuse: c chain + fML scan, one launch */
 const int md_blk   = tile_cb ? rnafold_md_block() : 0;       /* 3c: blocked md, bulk + corners */
 int       blk_cur  = -1;                                     /* 3c: the block-row ACC holds */
 long           n_st   = 0;
 int           *st_i   = NULL, *st_jlo = NULL;
 unsigned char *st_fl  = NULL;                 /* bit 0: row's first step; bit 1: its last */
 if(!mk_done && sweep_iters >= 1) {
   if(!tile_cb) {
     n_st = sweep_iters;
   } else {
     const long cap = (long)sweep_iters * (long)(length / tile_cb + 2);
     st_i   = (int *)malloc(sizeof(int) * (size_t)cap);
     st_jlo = (int *)malloc(sizeof(int) * (size_t)cap);
     st_fl  = (unsigned char *)calloc((size_t)cap, 1);
     unsigned char *seen = (unsigned char *)calloc((size_t)sweep_iters + 2, 1);
     if(!st_i || !st_jlo || !st_fl || !seen) {
       fprintf(stderr, "%-24s tile steps: allocation of %ld steps failed\n", __FILE__, cap);
       exit(EXIT_FAILURE);
     }
     for(int top = sweep_iters; top >= 1; top -= tile_rb) {
       const int bot = (top - tile_rb + 1 > 1) ? top - tile_rb + 1 : 1;
       const int j0  = bot + turn + 1;           /* the block-row's leftmost column */
       const int nJ  = (length >= j0) ? (length - j0) / tile_cb + 1 : 0;
       for(int q = 0; q < nJ; q++) {
         const int jlo = j0 + (tile_rev ? (nJ - 1 - q) : q) * tile_cb;
         for(int r = top; r >= bot; r--) {
           if(jlo + tile_cb - 1 < r + turn + 1) continue;   /* wholly left of row r */
           st_i[n_st] = r; st_jlo[n_st] = jlo; n_st++;
         }
       }
     }
     for(long s = 0; s < n_st; s++)  if(!(seen[st_i[s]] & 1)) { seen[st_i[s]] |= 1; st_fl[s] |= 1; }
     for(long s = n_st - 1; s >= 0; s--) if(!(seen[st_i[s]] & 2)) { seen[st_i[s]] |= 2; st_fl[s] |= 2; }
     free(seen);
   }
 }

 for (long st = 0; st < n_st; st++) {
    const int tile_jlo = tile_cb ? st_jlo[st] : 0;
    const int st_first = tile_cb ? (st_fl[st] & 1) : 1;
    const int st_last  = tile_cb ? ((st_fl[st] >> 1) & 1) : 1;
    i = tile_cb ? st_i[st] : sweep_iters - (int)st;   /* i,j in [1..length] */

    if(!continuous_flow) for(int H=0;H<nfiles;H++) i_H[H] = i;

    // Staggered_Row_Batching Phase 2d: table-driven per-H row offset,
    // replacing the H-tightest H+j*nfiles convention.
    // Phase 6d: `length` is now max(VC[H]->length) across the batch, so every
    // host loop in this row body has to bound itself by its OWN H's length and
    // skip H entirely on rows above it -- IS_ACTIVE() below. The shared bound
    // would spill past a short H's row into the next H's.
    // GPU-resident sweep (RNA_GPU_SWEEP): in device mode nothing on the host
    // reads energy_min this row -- the three host loops below are skipped and
    // int_loop_i's D2H into it is gated off -- and int_loop_kernel writes every
    // cell it later reads unconditionally (store is outside all conditionals,
    // grid range == read range), so the reset is dead. See the plan's
    // "no INF fill needed in device mode".
    if(!rnafold_gpu_sweep())
    for (int H=0;H<nfiles; H++) {
    if(!IS_ACTIVE(H)) continue;
    for (j = i_H[H]+turn+1; j <= (int)VC[H]->length; j++) energy_min[row_off_H[H]+j] = INF;
    }

    // This row's width tables: its slot in the chunk's row tables, which is
    // also the host's source of truth -- the device reads the same bytes.
    // Staggered_Row_Batching Phase 5 introduced "size" (int_loop_i/hp_mb_3p_i/
    // load_my_c/load_fML), Phase 4 "side" (fmli/modular_decomposition/
    // load_min_fML); see ROW_WIDTH_TABLES above.
    const size_t* size_off_H = rnafold_rowtab_size_host(i);
    const size_t* side_off_H = rnafold_rowtab_side_host(i);
    if(continuous_flow) {
      int* ih = rnafold_rowtab_ih_host(i);
      for(int H=0;H<nfiles;H++) ih[H] = i_H[H];
      ROW_WIDTH_TABLES(ih, rnafold_rowtab_size_host(i), rnafold_rowtab_side_host(i));
      rnafold_rowtab_upload_row(i);
    }

    if(st_first) {   /* per ROW, once, whatever the step order */
    cf_iters++;
    cf_cells += (long long)size_off_H[nfiles];
    if((long long)size_off_H[nfiles] > cf_peak_cells) cf_peak_cells = (long long)size_off_H[nfiles];
    {
      long long active = 0;
      for(int H=0;H<nfiles;H++) if(IS_ACTIVE(H)) active++;
      cf_rows += active;
      if(active > cf_peak_rows) cf_peak_rows = active;
    }

    /* RNA_MD3_LAUNCH_PROBE: price blocked-Zuker stage 3's launch multiple before
     * building the driver that would pay it. Extra no-op launches per row, same
     * stream as the phases; the fold is unchanged and only the wall moves. See
     * device.cu. */
    if(md3_probe_n) rnafold_md3_launch_probe_fire(md3_probe_n, i);
    }

    /* BLOCKED MD STAGES 3a'/3b (see the step sequence above): this step's column block.
     * The rotating rows are selected for row i first -- rows interleave in 3b, so they
     * cannot be advanced -- then the block gets tables of the row's layout with every
     * record's width clipped to [jlo, jhi] and the column shift in slot nfiles+1, and
     * rnafold_tile_begin points row i's lookups at them, so the phases below run
     * unchanged. Without tiles none of this runs: today's path. */
    if(tile_cb) {
      rnafold_md_ring_select(i);
      if(noLP) rnafold_cc_ring_select(i, st_first);
      const int jhi = tile_jlo + tile_cb - 1;
      size_t w_so[nfiles], w_sd[nfiles];
      for(int H=0; H<nfiles; H++) {
        const int L   = (int)VC[H]->length;
        const int top = (L < jhi) ? L : jhi;
        const int lso = (tile_jlo > i + turn + 1)       ? tile_jlo : i + turn + 1;
        const int lsd = (tile_jlo > i + 2*turn + 3)     ? tile_jlo : i + 2*turn + 3;
        /* the row tables' own rule: a record with no row here has no width */
        const int act = (i_H[H] >= 1) && (L - i_H[H] - turn > 0);
        w_so[H] = (act && top >= lso) ? (size_t)(top - lso + 1) : 0;
        w_sd[H] = (act && top >= lsd) ? (size_t)(top - lsd + 1) : 0;
      }
      compute_flatten_offsets(nfiles, w_so, rnafold_tile_size_host());
      compute_flatten_offsets(nfiles, w_sd, rnafold_tile_side_host());
      rnafold_tile_size_host()[nfiles+1] = (size_t)((tile_jlo > i + turn + 1)   ? tile_jlo - (i + turn + 1)   : 0);
      rnafold_tile_side_host()[nfiles+1] = (size_t)((tile_jlo > i + 2*turn + 3) ? tile_jlo - (i + 2*turn + 3) : 0);
      rnafold_tile_begin(i);
      size_off_H = rnafold_rowtab_size_host(i);   // now the tile's, by the override
      side_off_H = rnafold_rowtab_side_host(i);
    }

    {
      const double t0 = now_seconds();
      int_loop_i(nfiles,VC,i,turn,length,/*indx,ijsize,
		 hard_constraints, my_c,*/
		 energy_min, //replaces vrna_E_int_loop(vc, i, j);
		 size_off_H,i_H);
      /* G-quadruplex G2: the interior-loop term, MIN2 into the same
       * d_energy_min2 int_loop_kernel just wrote. Inside this phase's
       * timer because it IS interior-loop work, and before the sync so it
       * is charged honestly rather than draining into the next phase --
       * which is the mistake this file spent a session unpicking. */
      gq_internal_i(nfiles, i, turn, size_off_H, i_H);
      rnafold_phase_sync();   // RNA_PHASE_SYNC: charge this phase its OWN GPU time
      phase_int_loop_s += now_seconds() - t0;
    }

    //hairpin-loop / multibranch-loop / 3'-extension energies for this row,
    //computed fresh on GPU each i rather than precomputed as a full
    //nfiles*ijsize array (see fill_arrays.c) -- same row-buffer pattern as
    //energy_min/int_loop_i above.
    //
    // RNA_ROW_FUSE (hp_mb_loop.cu): these three phases -- hp_mb_3p, new_c and
    // load_my_c -- chain within ONE cell, so one kernel can run all three. The
    // launcher returns 0 when it cannot (noLP, RNA_STREAM_OVERLAP) and the three
    // separate phases below run instead. Charged to the hp_mb timer, which is the
    // largest of the three, so the phase split stays readable rather than
    // pretending the work moved somewhere new.
    // 3e-fuse (hp_mb_loop.cu, tile_front_kernel): on the tile path the step's hp_mb_3p,
    // stack row, new_c, c store AND fML scan are one launch, one block per record. The
    // -g row expansion the scan reads goes first. row_fused then skips the separate
    // phases below exactly as RNA_ROW_FUSE does, and the same hp_mb timer is charged.
    int tile_fused = 0;
    if(tile_fuse) {
      const double t0 = now_seconds();
      rnafold_gq_fill_row(nfiles, turn,
                          rnafold_i_H_device(i), rnafold_row_off_device(),
                          rnafold_size_off_device(i),
                          size_off_H[nfiles]);
      tile_fused = tile_front_i(nfiles, i, turn, length, noGUclosure, noLP, size_off_H);
      rnafold_phase_sync();
      phase_hp_mb_s += now_seconds() - t0;
    }
    int row_fused = tile_fused;
    if(!tile_fused) {
      const double t0 = now_seconds();
      row_fused = row_cells_i(nfiles,VC,i,turn,length,noGUclosure,noLP,size_off_H,i_H);
      if(row_fused) {
        rnafold_phase_sync();
        phase_hp_mb_s += now_seconds() - t0;
      }
    }

    if(!row_fused) {
      const double t0 = now_seconds();
      hp_mb_3p_i(nfiles,VC,i,turn,length,energy_hp_row,energy_mb_row,energy_3p00_row,gate_row,size_off_H,i_H);
      rnafold_phase_sync();   // RNA_PHASE_SYNC: charge this phase its OWN GPU time
      phase_hp_mb_s += now_seconds() - t0;
    }

    //could pack new_C more tightly for load_my_c_kernel but expect modest savings
    const double new_c_host_t0 = now_seconds();
    // GPU-resident sweep (RNA_GPU_SWEEP): new_c_kernel (new_c_i, below) already
    // wrote d_new_e directly from device buffers, and load_my_c's H2D of new_C
    // over it is gated off, so this loop has no consumer in device mode.
    if(!rnafold_gpu_sweep())
    for (int H=0;H<nfiles; H++) {
    if(!IS_ACTIVE(H)) continue; // Phase 6d; continuous flow phase B
    for (j = i_H[H]+turn+1; j <= (int)VC[H]->length; j++) {
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
          //
          // THE OTHER HALF OF A HARD CONSTRAINT, and it belongs here rather
          // than in the kernel. wrap_hairpin_hc.inc:42-52 admits a hairpin only
          // when hc->mx[ij] carries HP_LOOP *and* hc->up_hp[i+1] >= j-i-1 --
          // the mask says whether the PAIR is legal, up_hp whether the loop may
          // be left unpaired. The mask half already arrives through gate_row;
          // this is the half that was missing, and it only ever fires under -C:
          // for an unconstrained compound up_hp[i+1] is the whole remaining
          // sequence, so the test is true by construction.
          //
          // Host-side because the term is applied host-side: the kernel
          // computes energy_hp_row unconditionally and this loop is what gates
          // it (see the kernel's own comment). One array read per cell, in a
          // loop whose phase timer reads 0.000 s.
          const unsigned int u_hp = (unsigned int)(j - i - 1);
          if(VC[H]->hc->up_hp[i+1] >= u_hp)
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
    if(!row_fused)
    new_c_i(nfiles, i, turn, noGUclosure, noLP,
            rnafold_gpu_sweep() ? NULL : new_C,  // no host result to verify against in device mode
            row_off_H, size_off_H, i_H);

    if(!row_fused) {
      const double t0 = now_seconds();
      load_my_c(nfiles,i,turn,length,new_C,size_off_H,i_H); //keep my_c on GPU instep with my_c
      rnafold_phase_sync();   // RNA_PHASE_SYNC: charge this phase its OWN GPU time
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
    if(!IS_ACTIVE(H)) continue;
    const int i_h = i_H[H];   // continuous flow phase B: this record's own row
      /*  extension with one unpaired nucleotide at 5' site
	  and all other variants which are needed for odd
	  dangle models -- per-H (was incorrectly computed once from
	  VC[0] only and reused for every H, see energy_3p_en_j below
	  for the already-correct per-H sibling of this check)
      */
      const int cp = -1;
      const int en_i = (ON_SAME_STRAND(i_h - 1, i_h, cp) &&
                         ON_SAME_STRAND(i_h, i_h + 1, cp) &&
                         VC[H]->hc->up_ml[i_h] > 0) ? P->MLbase : INF;
    for (j = i_h+turn+1; j <= (int)VC[H]->length; j++) {
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
  const int fml_i_jm1 = (j == i_h+turn+1) ? INF : energy_min[row_off_H[H]+(j-1)];
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
    /* G-quadruplex G1: expand this row's c_gq into the dense row buffer that
     * fml_scan_kernel indexes. MUST precede fml_scan_i(), which reads it, and
     * it uses the device offset tables hp_mb_loop.cu already owns rather than
     * re-uploading them. Returns immediately unless a c_gq was uploaded. */
    if(!tile_fused) {   /* 3e-fuse ran both already */
    rnafold_gq_fill_row(nfiles, turn,
                        rnafold_i_H_device(i), rnafold_row_off_device(),
                        rnafold_size_off_device(i),
                        size_off_H[nfiles]);

    fml_scan_i(nfiles, i, turn,
               rnafold_gpu_sweep() ? NULL : energy_min,  // no host result to verify against in device mode
               row_off_H, size_off_H, i_H);
    }

    //load_fML + modular_decomposition_i + load_min_fML fused into one CUDA
    //graph capture/replay (no host CPU logic runs between these three calls,
    //which is what makes that legal) -- updates my_fML GPU, then
    //my_fML GPU = MIN2(energy_min[j], DMLi[j])
    {
      const double t0 = now_seconds();
      // side_off_H: built with this row's slot at the top of the iteration.
      // 3c (RNA_MD_BLOCK): md is min(ACC, corners). A block-row's first step resets its
      // ACC rows; every step tells md its block-row top and column block.
      const int blk_top = md_blk ? sweep_iters - ((sweep_iters - i) / tile_rb) * tile_rb : i;
      const int blk_bot = (blk_top - tile_rb + 1 > 1) ? blk_top - tile_rb + 1 : 1;
      if(md_blk) {
        if(blk_top != blk_cur) { rnafold_md_blk_rowstart(blk_top, blk_bot); blk_cur = blk_top; }
        rnafold_md_blk_set_step(blk_top, tile_jlo);
      }
      load_fML_modular_decomposition_load_min_fML(nfiles,i,turn,length,energy_min,DMLi,row_off_H,size_off_H,side_off_H,i_H);
      // 3c: row blk_bot is the LAST row of this column block in its block-row (rows run
      // top down and no column block is wholly left of bot), so every row's E for this
      // block now exists: fold it into ACC for every later column.
      if(md_blk && i == blk_bot)
        rnafold_md_blk_update(nfiles, turn, length, blk_top, blk_bot, tile_jlo, tile_jlo + tile_cb - 1);
      rnafold_phase_sync();   // RNA_PHASE_SYNC: charge this phase its OWN GPU time
      phase_modular_decomp_s += now_seconds() - t0;
    }

    /* Stages 3a'/3b: this step's block is done. The row's tail runs once, after its LAST
     * step -- in 3b that is after every column block of the whole block-row. */
    if(tile_cb) {
      rnafold_tile_end();
      size_off_H = rnafold_rowtab_size_host(i);   // the row's own tables again
      side_off_H = rnafold_rowtab_side_host(i);
    }
    if(!st_last) continue;
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
      if(!IS_ACTIVE(H)) continue;
      const int i_h = i_H[H];   // continuous flow phase B: this record's own row
      // The cell one below this row's first, (i, i+turn), is inside the
      // diagonal band -- never computed, INF in the fML prefill. Row i-1 will
      // read it as its own j == i+turn, so write it explicitly rather than
      // leaving row i+1's value there.
      fml_prev[row_off_H[H]+(i_h+turn)] = INF;
      for (int jj = i_h+turn+1; jj <= (int)VC[H]->length; jj++)
        fml_prev[row_off_H[H]+jj] = MIN2(energy_min[row_off_H[H]+jj],
                                         DMLi[row_off_H[H]+jj]);
    }
    phase_fml_prev_host_s += now_seconds() - fml_prev_host_t0;

    // GPU-resident sweep, step 2: the same row, computed on the device from
    // d_energy_min and d_dml -- both of which the GPU already had, which is why
    // this loop should never have been on the host. Runs AFTER the host loop so
    // RNA_ROW_VERIFY has something to compare against; nothing reads d_fml_prev
    // yet, so the sweep's behaviour is unchanged either way.
    /* tiles: md_close wrote fml_prev for every block already, and in 3b the "last row the
     * tail closed" is not row i, so this must not run (it would rewrite fml_prev whole). */
    if(!tile_cb)
    fml_prev_i(nfiles, i, turn,
               rnafold_gpu_sweep() ? NULL : fml_prev,  // no host result to verify against in device mode
               row_off_H, size_off_H, i_H);

    // GPU-resident sweep, step 1: the device twin of the DMLi1 rotation below.
    // Publishes row i's DMLi as "the previous row's" for row i-1, which is what
    // new_c_kernel will read as DMLi1[j-1] once new_c_host moves to the GPU.
    // Placed here, at exactly the host's rotate point, so the two representations
    // cannot drift. Nothing reads d_dml1 yet -- this is behaviour-neutral, and
    // costs one 3.3 MB device-to-device copy per row (~0.18 s over a whole run).
    if(!tile_cb) md_snapshot_dml();   /* tiles: the rings are selected by row, not advanced */

    /* RNA_SYNC_PROBE: k extra device syncs per row, to price the per-row
     * barriers this row already carries before building the machinery to
     * remove them. See PORT_STREAM_OVERLAP_SCOPE.md stage 0. */
    rnafold_sync_probe_tick();

    {
      int *FF; /* rotate the auxilliary arrays */
      FF = DMLi2; DMLi2 = DMLi1; DMLi1 = DMLi; DMLi = FF;
    }

    // noLP: rotate cc/cc1 at exactly the same point, because upstream does
    // (rotate_aux_arrays() rotates the multibranch helpers and cc together,
    // mfe/mfe.c:4460). Placed AFTER the DMLi rotate for the same reason
    // md_snapshot_dml() sits where it does: the two representations of "the
    // previous row" must be published at one point, or they drift.
    if(noLP && !tile_cb)   /* tiles: cc is selected by row (rnafold_cc_ring_select) */
      nolp_rotate_cc();

    // Continuous flow phase B: every active record advances one row; a record
    // that was on row 1 lands on 0 and is retired from here on. Off the flow
    // path this does nothing -- i_H is rewritten from i at the top of the next
    // iteration.
    if(continuous_flow)
      for(int H=0;H<nfiles;H++) if(i_H[H] >= 1) i_H[H]--;

    // Continuous flow phase C3: SLOT TURNOVER, mid-sweep. A slot whose row index
    // has just reached 0 has finished its occupant's last row -- and that row is
    // complete on the device, because the graph trio at the end of this
    // iteration synced. So the record can be handed over for fetching and
    // backtracking, and the slot handed to the next record in its queue.
    //
    // Everything the incoming record needs is put back to the state a chunk
    // starts in, for THIS SLOT ONLY: the neighbours are mid-recursion and their
    // buffers must not be touched.
    if(sched) {
      for(int s=0;s<nfiles;s++) {
        if(i_H[s] >= 1) continue;                                  // still working
        const int qn = sched->qoff[s+1] - sched->qoff[s];
        if(q_pos[s] >= qn) continue;                               // already all retired
        // The retire fetch reads this record's triangles on a copy stream that
        // is not ordered against the sweep's non-blocking streams. md's per-row
        // host sync used to cover that; RNA_MD_ROW_SYNC=0 removes it, so drain
        // here instead. Once per handover, never per row.
        rnafold_device_drain();
        sched->on_retire(sched->ctx, s, sched->queue[sched->qoff[s] + q_pos[s]]);
        q_pos[s]++;
        if(q_pos[s] >= qn) continue;                               // queue empty: slot idle
        const int rec = sched->queue[sched->qoff[s] + q_pos[s]];
        ((const vrna_fold_compound_t **)VC)[s] = sched->VC_all[rec];

        // host: the three DMLi generations and fml_prev. All three, because they
        // rotate -- whichever becomes DMLi1 next row must read INF for a record
        // that has no previous row yet.
        for(size_t k=row_off_H[s]; k<row_off_H[s+1]; k++)
          DMLi[k] = DMLi1[k] = DMLi2[k] = fml_prev[k] = INF;

        // device: this slot's sweep state, then its sequence-derived content.
        reset_slot_md(tri_off_H[s], tri_off_H[s+1]-tri_off_H[s],
                      row_off_H[s], row_off_H[s+1]-row_off_H[s]);
        // noLP's cc/cc1 are row-shaped state that rotates, exactly like the
        // DMLi generations reset just above, and reset_slot_md() predates them.
        // Without this the incoming record reads the outgoing record's cc1 and
        // returns a self-consistent but SUBOPTIMAL structure.
        if(noLP)
          reset_slot_nolp(row_off_H[s], row_off_H[s+1]-row_off_H[s]);
        refill_slot2(nfiles, VC, turn, length, 512, tri_off_H, row_off_H, cap_H, s);
        // ONE slot: only s's occupant changed. This was a whole-batch repack
        // until 2026-09-27, and it was the entire measured cost of slot flow.
        refill_gpu3 (nfiles, VC, turn, length, 512, row_off_H, cap_H, s);
        // -g: the c_gq table is laid out by SLOT (mfe_cuda.c uploads VCsl), so a
        // new occupant needs its own. Whole-table re-upload: c_gq is sparse and
        // this is once per handover, on a path that is off by default. Safe here
        // because the device was drained at the top of this handover, and the
        // kernels read the table's pointers at launch, never from a stale capture.
        if(rnafold_gq_active())
          (void)rnafold_gq_upload(nfiles, VC);

        const int top = (int)VC[s]->length - turn - 1;
        i_H[s] = (top >= 1) ? top : 0;
      }
    }
  } /* end of the step loop (one step per row without tiles: the old i-loop) */
 free(st_i); free(st_jlo); free(st_fl);

 // Anything still unretired -- a record with no rows of its own, or a slot that
 // emptied on the final iteration -- is retired here, so on_retire() is called
 // exactly once for every record in the schedule.
 if(sched)
   rnafold_device_drain();   // before rnafold_rowtab_end()'s drain -- see the handover above
 if(sched)
   for(int s=0;s<nfiles;s++)
     while(q_pos[s] < sched->qoff[s+1] - sched->qoff[s]) {
       sched->on_retire(sched->ctx, s, sched->queue[sched->qoff[s] + q_pos[s]]);
       q_pos[s]++;
     }

 // The sweep is over; drain and release the row tables.
 rnafold_c_ring_end(nfiles);   // RNA_C_RING: rows still only in the ring, before the tables go
 rnafold_rowtab_end();
 rnafold_tri_checksum();   // RNA_TRI_CHECKSUM=1: hash both triangles, cell for cell

 if (!vrna_cuda_quiet())
 fprintf(stderr,"%-24s sweep shape: %lld iterations, %lld active record-rows, "
                "%lld cells; peak/iteration %lld records %lld cells; "
                "continuous flow %s\n",
         __FILE__, cf_iters, cf_rows, cf_cells, cf_peak_rows, cf_peak_cells,
         continuous_flow ? "ON" : "OFF");



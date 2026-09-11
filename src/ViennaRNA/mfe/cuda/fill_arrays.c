//WBL Dec 2017 include file for mfe.c $Revision: 1.20 $

//WBL 27 Jan 2018 Add par_fill_arrays

//try and help compiler by inlining
//#include "modular_decomposition.c"


void min_fml(const int i, const int j, const int* my_fML, const int* DMLi, const char* name, const int turn, const int* indx,
	     const int length, const int ijsize) {
      {// does DMLi[j] holds  MIN(fML[i,k]+fML[k+1,j])
	//Cf. modular_decomposition() in multibranch_loops.c
	int min=INF;
	const int start = i+turn+1; const int stop = j - 2 - turn;
	//const int start = 1; const int stop = j;
	int k   = start;
	//fprintf(stderr,"%s (",name);
	//fflush(stderr);
	for(;k<=stop;k++){
	  const int ik  = indx[k]+i;   //to get fML[i,k]
	  const int k1j = indx[j]+k+1; //to get fML[k+1,j]
//	 {const int k1j_= indx[j] + i + turn + 1 + 1 + k - start; //to get fML[i+k+1,j] cf modular_decomposition
//	  assert(k1j == k1j_);}
	  assert(i  >0 &&  i <=j);      //starts at 1 not 0
	  assert(j  >0 &&  j <=length); //starts at 1 not 0
	  assert(k  >0 &&  k <=j     ); //starts at 1 not 0
	  assert(k+1>0 && k+1<=j     ); //starts at 1 not 0
	  assert(ik >0 && ik < ijsize); //starts at 1 not 0
	  assert(k1j>0 && k1j< ijsize); //starts at 1 not 0
	  const int fML_i = my_fML[ik];
	  const int fML_j = my_fML[k1j];
	  const int add = (fML_i != INF && fML_j != INF)? fML_i + fML_j : INF;
	  if(/*k>=start && k<=stop && */add<min) min = add;
	  //fprintf(stderr,"[%d %d,%d]%d [%d %d,%d]%d =%d\n",ik,i,k,fML_i,k1j,k+1,j,fML_j,add);
	  //fprintf(stderr,"my_fML[%d] %d, ",ik,my_fML[ik]);
	}
	//fprintf(stderr,"start %2d stop %2d min %d\n",start,stop,min);
	//int u;
	//for(u=1;u<=length;u++) fprintf(stderr,"%s[%2d] %d ",name,u,DMLi[u]);
	//fflush(stderr);
	//fprintf(stderr,"\n");
	assert(DMLi[j] == min);
      }
}//end min_fml

/**
*** fill "c", "fML" and "f5" arrays and return  optimal energy
**/
#define Hard_constraints(H,ij) (VC[H]->hc->matrix[ij])
#define Ptype(H,ij)            (VC[H]->ptype[ij])
#define My_c(H,ij)              VC[H]->matrices->c[ij]
#define My_fML(H,ij)            VC[H]->matrices->fML[ij]
#define My_fM1(H,ij)            VC[H]->matrices->fM1[ij]
//might actually make sense to comput ij, as is done on GPU, but stick with minimal change
#define Indx(H,i,j)            (VC[H]->jindx[j]+i)

PRIVATE void
par_fill_arrays(const int nfiles, const vrna_fold_compound_t **VC, int* Energy,
                const rnafold_schedule_t *sched) {

  {
    static int phase_timing_registered = 0;
    if(!phase_timing_registered) {
      atexit(print_phase_timing_stats);
      atexit(print_stage_timing_stats);
      phase_timing_registered = 1;
    }
  }

  unsigned char     type;
//char              *ptype, *hard_constraints;
  int               i, j, ij, length, /*energy,*/ new_c, /*stackEnergy,*/ no_close, turn,
                    noGUclosure, noLP, uniq_ML, /*dangle_model, *indx, *my_f5,
                    my_c, *my_fML, *my_fM1,*/ hc_decompose, /* *cc, *cc1, *Fmi,*/ *DMLi,
                    *DMLi1, *DMLi2,
                    // Staggered_Row_Batching 2026-08-22: row-shaped cache of the
                    // PREVIOUS row's final fML, the one thing the sweep still
                    // needed from the host fML triangle once my_fml_update_host's
                    // per-row triangular mirroring was retired in favour of
                    // fetch_fML(). Same rotate-free lifetime as DMLi* above.
                    *fml_prev;
  // New Jul 2026: hoisted out of fill_arrays_loop.c's per-row loop (were
  // malloc()'d/free()'d every row there -- see that file's history) and
  // page-locked (cuda_host_alloc_ints(), modular_decomposition.cu) since
  // all 5 are cudaMemcpy source/destination every row too. Safe to reuse
  // across rows unchanged: every element actually read each row is
  // unconditionally overwritten that same row before use, same as
  // DMLi/DMLi1/DMLi2's existing rotate-in-place pattern below.
  int               *energy_min, *energy_hp_row, *energy_mb_row,
                    *energy_3p00_row, *new_C;
  // Staggered_Row_Batching 2026-08-22: per-row flags hp_mb_3p_kernel now emits
  // so new_c_host no longer reads the ptype/hc triangles. char, not int -- two
  // bits per cell, and it is copied back every sweep row.
  char              *gate_row;
  vrna_param_t      *P;
//vrna_mx_mfe_t     *matrices;
//vrna_hc_t         *hc;
  vrna_ud_t         *domains_up;

  // Staggered_Row_Batching Phase 6d: the shared sweep bound is now the LONGEST
  // sequence in the batch, not VC[0]'s length. With mixed lengths (Phase 6c)
  // VC[0] is an arbitrary member, and any H longer than it would simply never
  // have its top rows swept -- silently truncated folding, not a crash.
  // Taking the max is what makes the sweep cover every H; each H then joins
  // the sweep on its own row via the per-H width tables built every row in
  // fill_arrays_loop.c (length_H[H]-i-turn, clamped >=0), which go to zero for
  // rows above that H's own length and so mask it out of every kernel launch
  // until it joins. Degenerates to exactly VC[0]->length while chunks are
  // uniform-length, which is what keeps this phase regression-testable.
  length            = 0;
  for(int H=0; H<nfiles; H++)
    if((int)VC[H]->length > length) length = (int)VC[H]->length;
  // Continuous flow phase C3: with a schedule, a slot's later occupants can be
  // longer than its first, and `length` sizes buffers and bounds asserts for the
  // WHOLE chunk -- so take the schedule's own maximum, not this pass's.
  if(sched && sched->length > length) length = sched->length;
//ptype             = vc->ptype;
//indx              = vc->jindx;
  P                 = VC[0]->params;
  // Staggered_Row_Batching Phase 2c: recomputed here (redundant with
  // par_mfe()'s own copy, mfe_cuda.c) rather than threaded down as
  // parameters -- keeps par_fill_arrays()'s signature/stub2.h declaration
  // unchanged, and the cost is negligible (O(nfiles)). fill_arrays_loop.c
  // (#include'd below) uses these directly, same function scope.
  size_t row_off_H[nfiles+1], tri_off_H[nfiles+1];
  // Continuous flow phase C1: same capacity rule as par_mfe(), from the same
  // function -- these tables address the device buffers par_mfe() allocated, so
  // the two must not be able to disagree.
  // Phase C1/C2: the chunk's capacity table comes from par_mfe(), which
  // allocated the device buffers from it. Recomputing it here would be wrong
  // under slot turnover, where a slot's capacity covers both of its occupants
  // and so cannot be derived from this pass's records alone.
  const size_t* cap_H = rnafold_chunk_capacities(nfiles);
  compute_batch_offsets(nfiles, cap_H, row_off_H, tri_off_H);
  // The GPU-accelerated hairpin/multibranch energy precompute (hp_mb_3p_i(),
  // GATE 3 IS NOT EMPTY AGAIN, and deliberately so.
  //
  // Dangle models 0 and 2 are both implemented (2026-09-11): they share the
  // recursion and differ in two energy terms, E_MLstem_device()'s two callers
  // in hp_mb_loop.cu plus gq_internal_kernel's mismatchI. Models 1 and 3 do
  // NOT share it -- ml_pair_d1() reads dmli2 as well as dmli1, a second DMLi
  // generation the sweep never carries -- so reaching the sweep with one would
  // silently produce the d0/d2 answer under a d1/d3 request.
  //
  // E_MLstem_device() is the specific reason this must be enforced rather than
  // asserted: it implements only upstream's two-sided and no-sided cases,
  // because d0 and d2 are the only models that can ask for the others. A d1
  // single-sided call would return 0 where upstream returns a dangle5/dangle3
  // term. assert() is compiled out here (NVCC_ASSERT_FLAGS is -DNDEBUG by
  // design -- PORT_INVESTIGATIONS.md item 4), so this check is the enforcement.
  if ((P->model_details.dangles != 0) && (P->model_details.dangles != 2)) {
    fprintf(stderr,
            "par_fill_arrays: this CUDA build implements dangle models 0 and 2 "
            "(got %d); 1 and 3 need a second DMLi generation the sweep does not "
            "carry. vrna_cuda_engine_supports() should have declined this.\n",
            P->model_details.dangles);
    exit(EXIT_FAILURE);
  }
  // Three model settings the GPU sweep cannot honour, each measured against
  // pristine ViennaRNA 2.7.2 on 2026-09-04 (PORT_FEATURE_AUDIT.md). Without
  // these checks the fork answers ANYWAY, which is the one failure mode worse
  // than not supporting them at all.
  /* PORT TO 2.7.2: these are now a BACKSTOP, and should be unreachable.
   *
   * On the 2.3.0 base they were the whole defence, and each message ended with
   * a workaround involving RNA_CPU_THREADS/RNA_CPU_THRESHOLD -- the fork's own
   * worker queue, running stock 2.3.0 recursions that themselves disagreed
   * with 2.7.2 on some of these options.
   *
   * None of that is true any more. The queue is retired in favour of upstream's
   * per-record parallel path, which IS ViennaRNA 2.7.2, and RNAfold.c decides
   * per run (gpu_path_usable()) whether the device may be used at all --
   * routing anything unsupported to that path automatically. A user asking for
   * --gquad gets their answer; they never see these messages.
   *
   * So reaching one of these now means the driver's gate has a hole, and the
   * message says that instead of offering advice that no longer applies. They
   * are kept precisely so "unreachable" is enforced rather than assumed.
   */
#define VRNA_CUDA_BACKSTOP(cond, opt, detail)                                  \
  if (cond) {                                                                  \
    fprintf(stderr,                                                            \
            "par_fill_arrays: %s reached the GPU sweep, which cannot reproduce "\
            "it (%s).\n"                                                        \
            "This is a BUG in the routing guard, not a user error: RNAfold "    \
            "should have folded this run on the CPU path without the device. "  \
            "Refusing rather than returning a plausible wrong answer.\n",       \
            opt, detail);                                                      \
    exit(EXIT_FAILURE);                                                        \
  }

  /* The circular backstop is GONE (2026-09-11). It was the first of the three
   * gates listed in PORT_OPTION_STATUS.md, and it is removed rather than
   * loosened because the sweep now persists fM2_real and the answer is
   * byte-identical to upstream. See PORT_CIRC_SPEC.md.
 */

  /* The G-quadruplex backstop is GONE (G3, 2026-09-10). It was the third of
   * three gates -- after RNAfold.c's gpu_path_usable() and
   * vrna_cuda_engine_supports() -- and it is removed rather than loosened
   * because the sweep now scores quadruplexes into c and fML and the answer is
   * byte-identical to upstream. See PORT_GQUAD_SPEC.md.
   *
   * This file's whole point is that "unreachable" is ENFORCED rather than
   * assumed; all three of its backstops have now been retired by
   * implementation, not by argument. */

  /* The --noClosingGU backstop is GONE (2026-09-11). It was the LAST of the
   * three gates listed in PORT_OPTION_STATUS.md. Removed rather than loosened:
   * Energy() now implements the interior-loop half the backstop existed to
   * describe, and the answer is byte-identical to upstream. See
   * PORT_NOCLOSINGGU_SPEC.md.
   *
   * NO BACKSTOP REMAINS IN THIS FILE. That is not the same as "nothing needs
   * one": the macro above stays defined so the next half-implemented option
   * can be gated the same way. */

#undef VRNA_CUDA_BACKSTOP
  noGUclosure       = P->model_details.noGUclosure;
  // Re-enabled 2026-09-07. This was commented out alongside the cc/cc1
  // machinery in fill_arrays_loop.c, which gcov reported as unused -- because
  // noLP was never exercised, not because it was unreachable. See
  // PORT_NOLP_SPEC.md.
  noLP              = P->model_details.noLP;

  // noLP + RNA_FML_INT16 is REFUSED, same shape and same reason as
  // RNA_FML_INT16 + RNA_SLOT_FLOW (modular_decomposition.cu): an invariant the
  // encoding depends on does not hold, so the pairing is declined rather than
  // approximated.
  //
  // Measured 2026-09-07, and the existing range guard is what found it:
  //   RNA_FML_INT16 range: H=56 (i=894,j=900) value 220 baseline 9999810
  //   delta -9999590 exceeds int16
  // The baseline is ~1e7 -- the ASYMMETRIC INF guard INT16_FML_SCOPE.md
  // documents, where fml_prev[j] + 10000000 is a large positive number and NOT
  // INF, and survives as a distinct value because the test is == INF, never
  // >= INF. That doc measured "fML carries ZERO near-INF cells" on the DEFAULT
  // model. Under noLP, c[ij] receives cc1[j-1]+stackEnergy, which is INF at the
  // top of the triangle because cc1 starts INF, so many more cells take the
  // +1e7 path -- and once one of them is the first-written entry of a block it
  // becomes that block's baseline, putting ordinary energies 1e7 away from it.
  //
  // So the provable B/2*340 bound is intact; its PREMISE (that fML holds no
  // near-INF finite values) is what noLP breaks. Fixing that means changing
  // what the guard writes, which is a change to the DEFAULT path's semantics,
  // and is not worth doing to make two optional features compose.
  // --noLP + RNA_ROW_VERIFY is REFUSED, and unlike the pairing below this one
  // is a defect in the INSTRUMENT, not a limit of an encoding.
  //
  // RNA_ROW_VERIFY checks the device kernels against the host loops, and turns
  // the GPU-resident sweep OFF so those loops run (mfe_cuda.c:132). The host
  // new_c loop has never implemented noLP -- its branch is still the
  // commented-out original at fill_arrays_loop.c:215, deleted on the same
  // "gcov says not used" evidence that PORT_NOLP_SPEC.md is about. So with
  // --noLP the verifier compares a device path that applies noLP against a
  // host path that does not, and measured on 30 x 80-1240 nt it reports
  // 922897 of 8262880 cells "mismatching" -- every one of them a FALSE alarm
  // against a device result that is byte-identical to upstream.
  //
  // Worse, it is not only noisy. In verify mode load_my_c uploads the host's
  // new_C over the device's, so the host's non-noLP values feed the rest of
  // the sweep and upstream's backtrack then walks them under the noLP
  // convention (mfe/mfe.c:4289). The fold that comes back is neither the noLP
  // answer nor the plain one. A debug flag must not change the answer.
  //
  // The real repair is to implement noLP in the host loop; until then refusing
  // is the honest option, because a bar that cries wolf gets bypassed and a
  // bypassed bar is how a false pass gets through (see PORT_FEATURE_AUDIT.md).
  if (noLP && getenv("RNA_ROW_VERIFY")) {
    fprintf(stderr,
            "%-24s --noLP and RNA_ROW_VERIFY cannot be combined: the host new_c "
            "loop does not implement noLP, so the verifier would report ~11%% of "
            "cells as false mismatches AND its host values would overwrite the "
            "device's, returning a fold that is neither answer. Unset one. See "
            "PORT_NOLP_SPEC.md.\n",
            __FILE__);
    exit(EXIT_FAILURE);
  }

  if (noLP && rnafold_fml_int16()) {
    fprintf(stderr,
            "%-24s --noLP and RNA_FML_INT16 cannot be combined: noLP puts "
            "near-INF finite values into fML, which the 16-bit per-block "
            "offsets cannot represent. Unset one. See PORT_NOLP_SPEC.md.\n",
            __FILE__);
    exit(EXIT_FAILURE);
  }
  uniq_ML           = P->model_details.uniq_ML;
//dangle_model      = P->model_details.dangles;
  turn              = P->model_details.min_loop_size;
//hc                = vc->hc;
//hard_constraints  = hc->matrix;
//matrices          = vc->matrices;
//my_f5             = matrices->f5;
//my_c              = matrices->c;
//my_fML            = matrices->fML;
//my_fM1            = matrices->fM1;
//domains_up        = vc->domains_up;


  /* allocate memory for all helper arrays */
  //cc    = (int *) vrna_alloc(sizeof(int)*(length + 2)); /* auxilary arrays for canonical structures     */
  //cc1   = (int *) vrna_alloc(sizeof(int)*(length + 2)); /* auxilary arrays for canonical structures     */
  //Fmi   = (int *) vrna_alloc(sizeof(int)*(length + 1)); /* holds row i of fML (avoids jumps in memory)  */
  DMLi  = cuda_host_alloc_ints(row_off_H[nfiles]); /* DMLi[j] holds  MIN(fML[i,k]+fML[k+1,j])      */
  DMLi1 = cuda_host_alloc_ints(row_off_H[nfiles]); /*                MIN(fML[i+1,k]+fML[k+1,j])    */
  DMLi2 = cuda_host_alloc_ints(row_off_H[nfiles]); /*                MIN(fML[i+2,k]+fML[k+1,j])    */
  fml_prev = cuda_host_alloc_ints(row_off_H[nfiles]); /* previous row's final fML  */

  if((turn < 0) || (turn > length))
    turn = length; /* does this make any sense? */

 for(int H=1;H<nfiles;H++) sanity(VC[0],VC[H]);

 for(int H=0;H<nfiles;H++) {
   domains_up        = VC[H]->domains_up;
  /* pre-processing ligand binding production rule(s) */
  if(domains_up && domains_up->prod_cb)
    domains_up->prod_cb(VC[H], domains_up->data);

  /* prefill helper arrays */
  // Staggered_Row_Batching Phase 6d: per-H bound, not the shared (now maximum)
  // length -- H's row slot is exactly VC[H]->length+1 entries, so the shared
  // bound would spill into the next H's row. This prefill is also what makes
  // the join mask correct: an H that has not joined the sweep yet is never
  // written by any kernel, so its DMLi1/DMLi2 still hold INF when it finally
  // does join -- exactly the state a single-sequence fold starts from.
  for(j = 0; j <= (int)VC[H]->length; j++){
    //Fmi[j] =
    // Staggered_Row_Batching Phase 2d: table-driven per-H row offset,
    // replacing the H-tightest H+j*nfiles convention (see the coalescing
    // finding in harmonic-swimming-hare.md for why that convention doesn't
    // survive staggering/mixed lengths).
    DMLi[row_off_H[H]+j] = DMLi1[row_off_H[H]+j] = DMLi2[row_off_H[H]+j] = INF;
    // Matches the fML prefill this replaces reading: before an H joins the
    // sweep nothing writes its fml_prev, and the first row it does join reads
    // cells that no row ever computed. Both must look like INF.
    fml_prev[row_off_H[H]+j] = INF;
  }
 }//endfor H


  /* prefill matrices with init contributions */
 const double t_prefill = rnafold_now_seconds();
 for(int H=0;H<nfiles;H++) {
  // Staggered_Row_Batching Phase 6d: per-H bound -- Indx(H,i,j) resolves
  // through VC[H]->jindx into VC[H]'s own matrices, so a j past this H's
  // length indexes outside them.
  for(j = 1; j <= (int)VC[H]->length; j++)
    //for(i = (j > turn ? (j - turn) : 1); i <= j; i++){
    // Staggered_Row_Batching 2026-08-22: the My_c/My_fML halves of this
    // prefill are gone. Nothing reads either triangle between here and the
    // end of the sweep any more -- new_c_host stopped writing My_c in
    // a1430bd and fml_host stopped writing My_fML in 89e5721, and both now
    // read row buffers instead -- and fetch_my_c()/fetch_fML() then overwrite
    // each record's triangle in FULL (tri_off_H[H+1]-tri_off_H[H] is exactly
    // the allocation size dp_matrices.c used). So every value written here was
    // read by nobody. It measured 2.4 s of E600's 32.7 s wall, 1.1 s of
    // workload A's 18.2 s -- 5-7%, for nothing.
    //
    // fM1 is different and stays: no kernel computes it and nothing fetches
    // it, so under uniq_ML it genuinely needs to start at INF. The whole loop
    // therefore only runs when uniq_ML is set, which for MFE folding it is not.
    if(uniq_ML)
      for(i = 1; i <= j; i++)
        My_fM1(H,Indx(H,i,j)) = INF;
 }//endfor H
  stage_prefill_s += rnafold_now_seconds() - t_prefill;
  init_fML(nfiles,length,tri_off_H[nfiles],row_off_H[nfiles]);//on GPU

  /* G-quadruplex G1: the per-row expansion buffer, same extent as
   * energy_3p00_row and the other row scratch. A no-op returning 0 unless
   * rnafold_gq_upload() actually uploaded a c_gq for this batch, so with -g off
   * nothing is allocated. */
  (void)rnafold_gq_row_alloc(row_off_H[nfiles]);

  /* CIRCULAR: the persistent fM2_real triangle, same extent as fML. Allocated
   * only under md->circ, so a linear fold pays nothing. Must precede the sweep,
   * which writes into it row by row. */
  if (rnafold_circ_alloc(P->model_details.circ, tri_off_H[nfiles]) < 0) {
    vrna_message_warning("par_fill_arrays: could not allocate fM2_real for "
                         "circular folding");
    exit(EXIT_FAILURE);
  }

  /* start recursion */

  if (length <= turn){
    // No sweep and no fetch_my_c()/fetch_fML() on this path, so the triangles
    // the prefill above no longer touches are still uninitialised here, and
    // backtrack() is about to read them. Fill them now -- free, because this
    // branch only triggers when every record is at most `turn` (3) bases.
    for(int H=0;H<nfiles;H++)
      for(j = 1; j <= (int)VC[H]->length; j++)
        for(i = 1; i <= j; i++)
          My_c(H,Indx(H,i,j)) = My_fML(H,Indx(H,i,j)) = INF;
    /* clean up memory */
    //free(cc);
    //free(cc1);
    //free(Fmi);
    cuda_host_free(DMLi);
    cuda_host_free(DMLi1);
    cuda_host_free(DMLi2);
    cuda_host_free(fml_prev);
    /* return free energy of unfolded chain */
    for(int H=0;H<nfiles;H++) {
      Energy[H] = 0;
    }//endfor H
    return;
  }

  const int ijsize = (length+1)*(length+2)/2;
  //fprintf(stderr,"fill_arrays(vc) length %d, ijsize %d\n",length,ijsize);

  // energy_hp/energy_mb/energy_3p_00/energy_3p_en (hairpin loop, multibranch
  // loop, and 3' multibranch-stem-extension energies) used to be precomputed
  // here into 4 full nfiles*ijsize host arrays -- at nfiles=80,length=5601
  // that's ~20GB of host RAM. None of the 4 actually need the whole triangle
  // precomputed: every read site in fill_arrays_loop.c consumes a value
  // during the SAME outer i iteration that produces it, and none of the 4
  // source computations depend on any other row's DP state. So they're now
  // computed fresh, per row, inside the main i-loop by hp_mb_3p_i()
  // (hp_mb_loop.cu), mirroring how int_loop_i()/load_my_c() already handle
  // interior-loop energies -- see fill_arrays_loop.c.
  // energy_mls (multiloop-stems-fast) was a 5th such array; it's deleted
  // outright rather than ported: under dangle_model==2 (enforced above),
  // vrna_E_ml_stems_fast2() always returned INF, contributing nothing at its
  // one read site.

  // New Jul 2026: allocated once here (page-locked, see DMLi above) rather
  // than malloc()'d/free()'d every row inside fill_arrays_loop.c's loop --
  // see the field declarations up top for why reuse across rows is safe.
  // 32-bit signed integer overflow bug fix: (size_t) cast must apply to
  // nfiles, the first operand -- casting the product after the fact would
  // already have overflowed in 32-bit int arithmetic by then.
  energy_min       = cuda_host_alloc_ints(row_off_H[nfiles]);
  energy_hp_row    = cuda_host_alloc_ints(row_off_H[nfiles]);
  energy_mb_row    = cuda_host_alloc_ints(row_off_H[nfiles]);
  energy_3p00_row  = cuda_host_alloc_ints(row_off_H[nfiles]);
  new_C            = cuda_host_alloc_ints(row_off_H[nfiles]);
  gate_row         = cuda_host_alloc_bytes(row_off_H[nfiles]);

#include "fill_arrays_loop.c"

  // The bulk fetch that used to sit here is gone, and so is the E_ext_loop_5()
  // + Energy[] loop that followed it. Both now happen one record at a time in
  // par_mfe()'s post-processing loop, against a small pool of reusable scratch
  // matrices, so the host never holds more than a few records' c/fML triangles
  // at once instead of all nfiles of them. At 5601nt that is 125.6 MB per
  // record -- about 1:1 with the record's VRAM cost -- so holding a whole
  // chunk's worth made host RAM as binding a constraint as the GPU's.
  //
  // This is only safe because nothing between here and there reads either
  // triangle: new_c_host stopped writing My_c in a1430bd, fml_host stopped
  // writing My_fML in 89e5721, and the dead prefill went in 4b2a18b.

  /* clean up memory */
  //free(cc);
  //free(cc1);
  //free(Fmi);
  cuda_host_free(DMLi);
  cuda_host_free(DMLi1);
  cuda_host_free(DMLi2);
  cuda_host_free(fml_prev);
  cuda_host_free(energy_min);
  cuda_host_free(energy_hp_row);
  cuda_host_free(energy_mb_row);
  cuda_host_free(energy_3p00_row);
  cuda_host_free(new_C);
  cuda_host_free(gate_row);

}

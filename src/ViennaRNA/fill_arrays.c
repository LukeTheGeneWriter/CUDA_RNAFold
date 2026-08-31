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
par_fill_arrays(const int nfiles, const vrna_fold_compound_t **VC, int* Energy) {

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
                    noGUclosure, /*noLP,*/ uniq_ML, /*dangle_model, *indx, *my_f5,
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
  // hp_mb_loop.cu) hardcodes the dangle_model==2 simplifications already
  // assumed throughout this CUDA fork's fill_arrays_loop.c/mb_loop_fast.c
  // (see the assert(dangle_model==2) calls there). Those are compiled out
  // under NDEBUG, and RNAfold_cmdl still technically accepts other --dangles
  // values at the CLI, so check for real here rather than relying on asserts.
  if (P->model_details.dangles != 2) {
    fprintf(stderr,
            "par_fill_arrays: this CUDA build requires --dangles=2 (got %d); "
            "the GPU hairpin/multibranch energy path assumes dangle_model==2.\n",
            P->model_details.dangles);
    exit(EXIT_FAILURE);
  }
  noGUclosure       = P->model_details.noGUclosure;
//noLP              = P->model_details.noLP;
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

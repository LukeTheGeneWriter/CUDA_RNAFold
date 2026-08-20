//WBL 10 Dec 2017 $Revision: 1.24 $ GGGP ViennaRNA-2.3.0 rf/rf/
//Helper for fill_arrays.c -> mfe.c for eventual CUDA version

//WBL 27 Jan 2018 Add loop H for nfiles different structures

 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    // Staggered_Row_Batching Phase 2d: table-driven per-H row offset,
    // replacing the H-tightest H+j*nfiles convention.
    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) energy_min[row_off_H[H]+j] = INF;
    }

    {
      const double t0 = now_seconds();
      int_loop_i(nfiles,VC,i,turn,length,/*indx,ijsize,
		 hard_constraints, my_c,*/
		 energy_min); //replaces vrna_E_int_loop(vc, i, j);
      phase_int_loop_s += now_seconds() - t0;
    }

    //hairpin-loop / multibranch-loop / 3'-extension energies for this row,
    //computed fresh on GPU each i rather than precomputed as a full
    //nfiles*ijsize array (see fill_arrays.c) -- same row-buffer pattern as
    //energy_min/int_loop_i above.
    {
      const double t0 = now_seconds();
      hp_mb_3p_i(nfiles,VC,i,turn,length,energy_hp_row,energy_mb_row,energy_3p00_row);
      phase_hp_mb_s += now_seconds() - t0;
    }

    //could pack new_C more tightly for load_my_c_kernel but expect modest savings
    const double new_c_host_t0 = now_seconds();
    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) {
      new_C[row_off_H[H]+j] = INF;
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      type          = (unsigned char)Ptype(H,ij);
      hc_decompose  = Hard_constraints(H,ij);
      //energy      = INF;

      no_close = (((type==3)||(type==4))&&noGUclosure);

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
	assert(My_c(H,ij) == INF);
          My_c(H,ij)    = new_c;
	  new_C[row_off_H[H]+j]    = new_c;
      } /* end >> if (pair) << */

      else {
	//fprintf(stderr,"\nmy_c[%3d] %d <= %d\n",ij,my_c[ij],INF);
	assert(My_c(H,ij) == INF);
	My_c(H,ij) = INF;
      }

    } /* end of j-loop */
    }//endfor H
    phase_new_c_host_s += now_seconds() - new_c_host_t0;

    {
      const double t0 = now_seconds();
      load_my_c(nfiles,i,turn,length,new_C); //keep my_c on GPU instep with my_c
      phase_load_my_c_s += now_seconds() - t0;
    }

    const double fml_host_t0 = now_seconds();
    for (int H=0;H<nfiles; H++) {
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
    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      /* done with c[i,j], now compute fML[i,j] and fM1[i,j] */

      //my_fML[ij] = vrna_E_ml_stems_fast(vc, i, j, Fmi, DMLi);

      /*  extension with one unpaired nucleotide at the right (3' site)
	  or full branch of (i,j)
      */
      //from extend_fm_3p()...
      //const int cp = -1;
      int  e00           = INF;
      int  en0           = INF;

  e00 = (energy_3p00_row[row_off_H[H]+j] != INF)? My_c(H,ij) + energy_3p00_row[row_off_H[H]+j] : INF;
  //energy_3p_en is just P->MLbase behind a hard-constraint check on
  //already-host-resident data -- not worth a GPU kernel, computed inline
  const int energy_3p_en_j = (VC[H]->hc->up_ml[j] > 0) ? P->MLbase : INF;
  en0 = ((My_fML(H,Indx(H,i,j - 1)) != INF) && (energy_3p_en_j != INF))? My_fML(H,Indx(H,i,j - 1)) + energy_3p_en_j : INF;
  e00 = MIN2(e00, en0);
      //end from extend_fm_3p()...

      //const int e0 = extend_fm_3p(i, j, my_fML, vc);

      const int e3 = (My_fML(H,ij + 1) != INF)? My_fML(H,ij + 1) + en_i : INF;


      //energy_mls (multiloop-stems-fast) deleted: under dangle_model==2
      //(enforced in fill_arrays.c) it always evaluated to INF, so
      //MIN2(e3,energy_mls[...]) always reduced to e3 -- see fill_arrays.c.
      energy_min[row_off_H[H]+j] = MIN2(e00,e3); //e1 e31
//    } /* end of j-loop */
//
      assert(My_fML(H,ij) == INF);
      My_fML(H,ij) = energy_min[row_off_H[H]+j];
    } /* end of j-loop */
    }//endfor H
    phase_fml_host_s += now_seconds() - fml_host_t0;

    //load_fML + modular_decomposition_i + load_min_fML fused into one CUDA
    //graph capture/replay (no host CPU logic runs between these three calls,
    //which is what makes that legal) -- updates my_fML GPU, then
    //my_fML GPU = MIN2(energy_min[j], DMLi[j])
    {
      const double t0 = now_seconds();
      // Staggered_Row_Batching Phase 4: per-H "active width" tables for
      // load_fML/fmli/modular_decomposition/load_min_fML's flat grids,
      // rebuilt every row since they depend on i. Only two distinct shapes
      // needed: size_H (load_fML's own) and side_H (shared by the other
      // three -- their pre-Phase-4 bound-check formulas are algebraically
      // identical, verified by hand against each kernel).
      size_t size_H[nfiles], side_H[nfiles];
      for(int H=0;H<nfiles;H++) {
        const int size_raw = (int)VC[H]->length - i - turn;
        const int side_raw = (int)VC[H]->length - i - 2*turn - 2;
        size_H[H] = (size_raw>0) ? (size_t)size_raw : 0;
        side_H[H] = (side_raw>0) ? (size_t)side_raw : 0;
      }
      size_t size_off_H[nfiles+1], side_off_H[nfiles+1];
      compute_flatten_offsets(nfiles, size_H, size_off_H);
      compute_flatten_offsets(nfiles, side_H, side_off_H);

      load_fML_modular_decomposition_load_min_fML(nfiles,i,turn,length,energy_min,DMLi,row_off_H,size_off_H,side_off_H);
      phase_modular_decomp_s += now_seconds() - t0;
    }

    const double my_fml_update_host_t0 = now_seconds();
    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      My_fML(H,ij) = MIN2(energy_min[row_off_H[H]+j], DMLi[row_off_H[H]+j]);

      /* gcov says not used
      if(uniq_ML){  ** compute fM1 for unique decomposition **
        my_fM1[ij] = E_ml_rightmost_stem(i, j, vc);
      }*/

      //fprintf(stderr,"\n");

      /* does 
	 DMLi[j] holds  MIN(fML[i,k]+fML[k+1,j])      **
	 DMLi1          MIN(fML[i+1,k]+fML[k+1,j])    **
	 DMLi2          MIN(fML[i+2,k]+fML[k+1,j])    **
      min_fml(i,  j,my_fML,DMLi, " DMLi", turn, indx, length, ijsize);
      min_fml(i+1,j,my_fML,DMLi1,"DMLi1", turn, indx, length, ijsize);
      min_fml(i+2,j,my_fML,DMLi2,"DMLi2", turn, indx, length, ijsize);
      */
    } /* end of j-loop */
    }//endfor H
    phase_my_fml_update_host_s += now_seconds() - my_fml_update_host_t0;

    {
      int *FF; /* rotate the auxilliary arrays */
      FF = DMLi2; DMLi2 = DMLi1; DMLi1 = DMLi; DMLi = FF;
    }
  } /* end of i-loop */



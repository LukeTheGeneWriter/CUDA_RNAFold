//WBL 10 Dec 2017 $Revision: 1.24 $ GGGP ViennaRNA-2.3.0 rf/rf/
//Helper for fill_arrays.c -> mfe.c for eventual CUDA version

//WBL 27 Jan 2018 Add loop H for nfiles different structures

 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    //Identical calculation for all of H, so do only once
    //int en0;
    int en = INF;
    //for (int H=0;H<nfiles; H++) {
      /*  extension with one unpaired nucleotide at 5' site
	  and all other variants which are needed for odd
	  dangle models
      */
      const int cp = -1;
      //const int cp = -1;
      if(ON_SAME_STRAND(i - 1, i, cp)){
	if(ON_SAME_STRAND(i, i + 1, cp)){
	  if(VC[0]->hc->up_ml[i] > 0){ //eval_loop = () ? (char)1 : (char)0;
	      en = P->MLbase;
	  }
	}
      }
      //if(H==0) en0 = en;
      //else assert(en0==en);
    //}endfor H


    int* energy_min = (int*)malloc(nfiles*(length+1)*sizeof(int));
    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) energy_min[H*(length+1)+j] = INF;
    }

    int_loop_i(nfiles,VC,i,turn,length,/*indx,ijsize,
	       hard_constraints, my_c,*/
	       energy_min); //replaces vrna_E_int_loop(vc, i, j);

    //could pack new_C more tightly for load_my_c_kernel but expect modest savings
    int* new_C = malloc(nfiles*(length+1)*sizeof(int)); //for GPU
    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) {
      new_C[H*(length+1)+j] = INF;
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
	new_c = energy_min[H*(length+1)+j];

        if(!no_close){
          /* check for hairpin loop */
          /*energy_hp[ij] = energy = vrna_E_hp_loop(vc, i, j); */
          new_c = MIN2(new_c, energy_hp[H*ijsize+ij]);

          /* check for multibranch loops */
          //energy  = vrna_E_mb_loop_fast(vc, i, j, DMLi1, DMLi2);
	  const int e_mb = (DMLi1[H*(length+1)+j-1] != INF)? DMLi1[H*(length+1)+j-1] + energy_mb[H*ijsize+ij] : INF;
          new_c   = MIN2(new_c, e_mb);
        }

        /*gov says not used if(dangle_model == 3){ ** coaxial stacking * E_mb_loop_stack(i, j, vc);*/

        /* gcov says not used  remember stack energy for --noLP option * if(noLP) vrna_E_stack(vc, i, j) cc[j] = new_c */
	assert(My_c(H,ij) == INF);
          My_c(H,ij)    = new_c;
	  new_C[H*(length+1)+j]    = new_c;
      } /* end >> if (pair) << */

      else {
	//fprintf(stderr,"\nmy_c[%3d] %d <= %d\n",ij,my_c[ij],INF);
	assert(My_c(H,ij) == INF);
	My_c(H,ij) = INF;
      }

    } /* end of j-loop */
    }//endfor H

    load_my_c(nfiles,i,turn,length,new_C); //keep my_c on GPU instep with my_c

    for (int H=0;H<nfiles; H++) {
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

  e00 = (energy_3p_00[H*ijsize+ij] != INF)? My_c(H,ij) + energy_3p_00[H*ijsize+ij] : INF;
  en0 = ((My_fML(H,Indx(H,i,j - 1)) != INF) && (energy_3p_en[H*ijsize+ij] != INF))? My_fML(H,Indx(H,i,j - 1)) + energy_3p_en[H*ijsize+ij] : INF;
  e00 = MIN2(e00, en0);
      //end from extend_fm_3p()...

      //const int e0 = extend_fm_3p(i, j, my_fML, vc);

      const int e3 = (My_fML(H,ij + 1) != INF)? My_fML(H,ij + 1) + en : INF;


//    const int e1 = MIN2(e3,energy_mls[H*ijsize+ij]); //e31
      energy_min[H*(length+1)+j] = MIN2(e00,MIN2(e3,energy_mls[H*ijsize+ij])); //e1 e31
//    } /* end of j-loop */
//
      assert(My_fML(H,ij) == INF);
      My_fML(H,ij) = energy_min[H*(length+1)+j];
    } /* end of j-loop */
    }//endfor H

    load_fML(nfiles,i,turn,length,energy_min); //update my_fML GPU

    modular_decomposition_i(nfiles,i,turn,length,/*indx,ijsize,my_fML,*/ DMLi);

    load_min_fML(nfiles,i,turn,length); //update my_fML GPU = MIN2(energy_min[j], DMLi[j])

    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      My_fML(H,ij) = MIN2(energy_min[H*(length+1)+j], DMLi[H*(length+1)+j]);

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

    {
      int *FF; /* rotate the auxilliary arrays */
      FF = DMLi2; DMLi2 = DMLi1; DMLi1 = DMLi; DMLi = FF;
    }

    free(new_C);
    free(energy_min);//optimise malloc and free later
  } /* end of i-loop */



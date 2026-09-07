//WBL 10 Dec 2017 $Revision: 1.48 $ GGGP ViennaRNA-2.3.0 rf/rf/
//Helper for fill_arrays.c -> mfe.c for eventual CUDA version

//WBL  7 Sep 2026 move load_my_c code into int_loop_mb
//WBL  2 Sep 2026 move code to new function int_loop_mb
//WBL 14 Aug 2026 Clean debug for GitHub
//WBL  8 Aug 2026 move code to new function int_loop_mls
//WBL  1 Aug 2026 make H tightest index DMLi,DMLi1
//WBL 31 Jul 2026 make H tightest index energy_min
//WBL 20 Jul 2026 make H tightest index
//WBL 27 Jan 2018 Add loop H for nfiles different structures

//now new_C packed tightly for load_my_c_kernel
int* new_C = malloc(nfiles*(length-(turn+1))*sizeof(int)); //for GPU
int* energy_min = (int*)malloc(nfiles*(length+1)*sizeof(int));

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


    for (int H=0;H<nfiles; H++) {
    for (j = i+turn+1; j <= length; j++) energy_min[H*(length+1)+j] = INF;
    }

    int_loop_i(nfiles,VC,i,turn,length,/*indx,ijsize,
	       hard_constraints, my_c,*/
	       energy_min); //replaces vrna_E_int_loop(vc, i, j);
    print_energy_min("int_loop_i",nfiles,length,i+turn+1,energy_min);

    int_loop_mb(nfiles,i,/*turn,*/length,ijsize,noGUclosure,
		energy_min,energy_hp,energy_mb,DMLi1,
		VC, //hard_constraints, My_C
		new_C);
    print_energy_min("int_loop_mb",nfiles,length,i+turn+1,energy_min);

    int_loop_mls(nfiles,VC,//out
		 i, /*turn,*/ length, ijsize,
		 new_C,          //contents of My_c(H,ij)
		 en,
		 energies,
		 energy_min); //out
    print_energy_min("int_loop_mls",nfiles,length,i+turn+1,energy_min);

    //Aug 2026 load_fML now done as part of int_loop_mls

    modular_decomposition_i(nfiles,i,turn,length,/*indx,ijsize,my_fML,*/ DMLi);
    print_energy_min("modular_decomposition_i",nfiles,length,i+turn+1,energy_min);

    load_min_fML(nfiles,i,turn,length); //update my_fML GPU = MIN2(energy_min[j], DMLi[j])
    print_energy_min("load_min_fML",nfiles,length,i+turn+1,energy_min);

    int_loop_DMLi(nfiles,
		  i, /*turn,*/ length, ijsize,
		  energy_min, DMLi,
		  VC); //out
    print_energy_min("int_loop_DMLi",nfiles,length,i+turn+1,energy_min);

    {
      int *FF; /* rotate the auxilliary arrays */
      FF = DMLi2; DMLi2 = DMLi1; DMLi1 = DMLi; DMLi = FF;
    }

  } /* end of i-loop */
 free(energy_min);
 free(new_C);



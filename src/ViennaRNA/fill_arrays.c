//WBL Dec 2017 include file for mfe.c $Revision: 1.20 $

//WBL 27 Jan 2018 Add par_fill_arrays

//try and help compiler by inlining
//#include "modular_decomposition.c"

/*taken from multibranch_loops.c r1.37 */
PRIVATE int
E_ml_stems_fast(const vrna_fold_compound_t *vc,
                const int i,
                const int j){

  char          eval_loop;
  int           /*k,*/ en, /*decomp,*/ mm5, mm3, /*type_2, k1j, stop,*/
                type, e/*, u,
                cnt*/;
  
#ifdef WITH_GEN_HC
  vrna_callback_hc_evaluate *f;
#endif

  const int length        = (int)vc->length;
  const char* ptype       = vc->ptype;
  const short* S          = vc->sequence_encoding;
  const int* indx         = vc->jindx;
  const char* hc          = vc->hc->matrix;
  const int* hc_up        = vc->hc->up_ml;
  const vrna_sc_t* sc     = vc->sc;
  const int *c            = vc->matrices->c;
//const int* fm           = vc->matrices->fML;
  const vrna_param_t* P   = vc->params;
  const int ij            = indx[j] + i;
  const int dangle_model  = P->model_details.dangles;
//const int turn          = P->model_details.min_loop_size;
  type          = ptype[ij];
//const int* rtype        = &(P->model_details.rtype[0]);
  const int circular      = P->model_details.circ;
  const int cp            = vc->cutpoint;
  const vrna_ud_t* domains_up    = vc->domains_up;
  const int with_ud       = (domains_up && domains_up->energy_cb) ? 1 : 0;
  e             = INF;
  assert(with_ud==0); //domains_up not supported
  assert(cp == -1);
  assert(dangle_model==2); //md->dangles

#ifdef WITH_GEN_HC
  f = vc->hc->f;
#endif

  /*  extension with one unpaired nucleotide at the right (3' site)
      or full branch of (i,j)

  e = extend_fm_3p(i, j, fm, vc);
  */

  /*  extension with one unpaired nucleotide at 5' site
      and all other variants which are needed for odd
      dangle models
  */
  if(ON_SAME_STRAND(i - 1, i, cp)){
/*
    if(ON_SAME_STRAND(i, i + 1, cp)){
      eval_loop = (hc_up[i] > 0) ? (char)1 : (char)0;

#ifdef WITH_GEN_HC
      if(f)
        eval_loop = (f(i, j, i+1, j, VRNA_DECOMP_ML_ML, vc->hc->data)) ? eval_loop : (char)0;
#endif

      if(eval_loop){
        if(fm[ij + 1] != INF){
          en = fm[ij + 1] + P->MLbase;
	  assert(!sc); **ensure we agree always agree with gcov
          if(sc){
            if(sc->energy_up)
              en += sc->energy_up[i][1];
            if(sc->f)
              en += sc->f(i, j, i+1, j, VRNA_DECOMP_ML_ML, sc->data);
          }**
          e = MIN2(e, en);
        }

      }
    }
*/
    /* extension with bound ligand on 5'site **
    if(with_ud){
      for(cnt = 0; cnt < domains_up->uniq_motif_count; cnt++){
        u = domains_up->uniq_motif_size[cnt];
        k = i + u - 1;
        if((k < j) && ON_SAME_STRAND(i, k + 1, cp)){
          eval_loop = (hc_up[i] >= u) ? (char)1 : (char)0;

#ifdef WITH_GEN_HC
          if(f)
            eval_loop = (f(i, j, k + 1, j, VRNA_DECOMP_ML_ML, vc->hc->data)) ? eval_loop : (char)0;
#endif

          if(eval_loop){
            if(fm[ij + u] != INF){
              en = domains_up->energy_cb( vc,
                                          i, k,
                                          VRNA_UNSTRUCTURED_DOMAIN_MB_LOOP | VRNA_UNSTRUCTURED_DOMAIN_MOTIF,
                                          domains_up->data);
              if(en != INF){
                en +=   fm[ij + u]
                      + u * P->MLbase;

                if(sc){
                  if(sc->energy_up)
                    en += sc->energy_up[i][u];
                  if(sc->f)
                    en += sc->f(i, j, k + 1, j, VRNA_DECOMP_ML_ML, sc->data);
                }
                e = MIN2(e, en);
              }
            }
          }
        }
      }
    }
    */

    if(dangle_model % 2){ /* dangle_model = 1 || 3 */

      mm5 = ((i>1) || circular) ? S[i] : -1;
      mm3 = ((j<length) || circular) ? S[j] : -1;

      if(ON_SAME_STRAND(i, i + 1, cp)){
        eval_loop = hc[ij+1] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC;
        eval_loop = (hc_up[i] > 0) ? eval_loop : (char)0;

#ifdef WITH_GEN_HC
        if(f)
          eval_loop = (f(i, j, i+1, j, VRNA_DECOMP_ML_STEM, vc->hc->data)) ? eval_loop : (char)0;
#endif

        if(eval_loop){
          if(c[ij+1] != INF){
            type = ptype[ij+1];

            if(type == 0)
              type = 7;

            en = c[ij+1] + E_MLstem(type, mm5, -1, P) + P->MLbase;
            if(sc){
              if(sc->energy_up)
                en += sc->energy_up[i][1];
              if(sc->f)
                en += sc->f(i, j, i+1, j, VRNA_DECOMP_ML_STEM, sc->data);
            }
            e = MIN2(e, en);
          }
        }
      }

      if(ON_SAME_STRAND(j - 1, j, cp)){
        eval_loop = hc[indx[j-1]+i] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC;
        eval_loop = (hc_up[j] > 0) ? eval_loop : (char)0;

#ifdef WITH_GEN_HC
        if(f)
          eval_loop = (f(i, j, i, j-1, VRNA_DECOMP_ML_STEM, vc->hc->data)) ? eval_loop : (char)0;
#endif

        if(eval_loop){
          if(c[indx[j-1]+i] != INF){
            type = ptype[indx[j-1]+i];

            if(type == 0)
              type = 7;

            en = c[indx[j-1]+i] + E_MLstem(type, -1, mm3, P) + P->MLbase;
            if(sc){
              if(sc->energy_up)
                en += sc->energy_up[j][1];
              if(sc->f)
                en += sc->f(i, j, i, j-1, VRNA_DECOMP_ML_STEM, sc->data);
            }
            e = MIN2(e, en);
          }
        }
      }

      if(ON_SAME_STRAND(j - 1, j, cp) && ON_SAME_STRAND(i, i + 1, cp)){
        eval_loop = hc[indx[j-1]+i+1] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC;
        eval_loop = (hc_up[i] && hc_up[j]) ? eval_loop : (char)0;

#ifdef WITH_GEN_HC
        if(f)
          eval_loop = (f(i, j, i+1, j-1, VRNA_DECOMP_ML_STEM, vc->hc->data)) ? eval_loop : (char)0;
#endif

        if(eval_loop){
          if(c[indx[j-1]+i+1] != INF){
            type = ptype[indx[j-1]+i+1];

            if(type == 0)
              type = 7;

            en = c[indx[j-1]+i+1] + E_MLstem(type, mm5, mm3, P) + 2*P->MLbase;
            if(sc){
              if(sc->energy_up)
                en += sc->energy_up[j][1] + sc->energy_up[i][1];
              if(sc->f)
                en += sc->f(i, j, i+1, j-1, VRNA_DECOMP_ML_STEM, sc->data);
            }
            e = MIN2(e, en);
          }
        }
      }
    } /* end special cases for dangles == 1 || dangles == 3 */
  }
  return e;

  /* modular decomposition -------------------------------**
  assert((sc && sc->f)==0);
  decomp = modular_decomposition(i,ij,j,turn,fmi,vc->matrices->fML);
  /* End modular decomposition -------------------------------**


  //not true assert(e == decomp);
  dmli[j] = decomp;               /* store for use in fast ML decompositon **
  e = MIN2(e, decomp);
  fmi[j] = MIN2(e, dmli[j]);
  //not true assert(fmi[j] == dmli[j]);
  return e;

  /* coaxial stacking ** not supported yet
  if (dangle_model==3) {
    /* additional ML decomposition as two coaxially stacked helices **
    int ik;
    k1j = indx[j]+i+turn+2;
    for (decomp = INF, k = i + 1 + turn; k <= stop; k++, k1j++){
      ik = indx[k]+i;
      if((hc[ik] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC) && (hc[k1j] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC)){
        type    = rtype[(unsigned char)ptype[ik]];
        type_2  = rtype[(unsigned char)ptype[k1j]];

        if(type == 0)
          type = 7;
        if(type_2 == 0)
          type_2 = 7;

        en      = c[ik] + c[k1j] + P->stack[type][type_2];
        if(sc){
          if(sc->f)
            en += sc->f(i, k, k+1, j, VRNA_DECOMP_ML_COAXIAL, sc->data);
        }
        decomp  = MIN2(decomp, en);
      }
    }
    k++; k1j++;
    for (; k <= j-2-turn; k++, k1j++){
      ik = indx[k]+i;
      if((hc[ik] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC) && (hc[k1j] & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC)){
        type    = rtype[(unsigned char)ptype[ik]];
        type_2  = rtype[(unsigned char)ptype[k1j]];

        if(type == 0)
          type = 7;
        if(type_2 == 0)
          type_2 = 7;

        en      = c[ik] + c[k1j] + P->stack[type][type_2];
        if(sc){
          if(sc->f)
            en += sc->f(i, k, k+1, j, VRNA_DECOMP_ML_COAXIAL, sc->data);
        }
        decomp  = MIN2(decomp, en);
      }
    }

    decomp += 2*P->MLintern[1];        /* no TermAU penalty if coax stack **
#if 0
        /* This is needed for Y shaped ML loops with coax stacking of
           interior pairts, but backtracking will fail if activated **
        DMLi[j] = MIN2(DMLi[j], decomp);
        DMLi[j] = MIN2(DMLi[j], DMLi[j-1]+P->MLbase);
        DMLi[j] = MIN2(DMLi[j], DMLi1[j]+P->MLbase);
        new_fML = MIN2(new_fML, DMLi[j]);
#endif
    e = MIN2(e, decomp);
  }
**
  fmi[j] = e;

  return e;
  */
}


/*taken from multibranch_loops.c r1.37 */
/*renamed to avoid name conflict*/
PUBLIC int
vrna_E_ml_stems_fast2( const vrna_fold_compound_t *vc,
                      const int i,
                      const int j){

  int e = INF;

  if(vc){
    switch(vc->type){
      case VRNA_FC_TYPE_SINGLE:
        e = E_ml_stems_fast(vc, i, j);//, fmi, dmli);
        break;
	/*
      case VRNA_FC_TYPE_COMPARATIVE:
        e = E_ml_stems_fast_comparative(vc, i, j, fmi, dmli);
        break;
	*/
    default: assert(0==1);
    }
  }

  return e;
}


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

  unsigned char     type;
//char              *ptype, *hard_constraints;
  int               i, j, ij, length, /*energy,*/ new_c, /*stackEnergy,*/ no_close, turn,
                    noGUclosure, /*noLP,*/ uniq_ML, /*dangle_model, *indx, *my_f5,
                    my_c, *my_fML, *my_fM1,*/ hc_decompose, /* *cc, *cc1, *Fmi,*/ *DMLi,
                    *DMLi1, *DMLi2;
  vrna_param_t      *P;
//vrna_mx_mfe_t     *matrices;
//vrna_hc_t         *hc;
  vrna_ud_t         *domains_up;

  length            = (int)VC[0]->length;
//ptype             = vc->ptype;
//indx              = vc->jindx;
  P                 = VC[0]->params;
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
  DMLi  = (int *) vrna_alloc(nfiles*sizeof(int)*(length + 1)); /* DMLi[j] holds  MIN(fML[i,k]+fML[k+1,j])      */
  DMLi1 = (int *) vrna_alloc(nfiles*sizeof(int)*(length + 1)); /*                MIN(fML[i+1,k]+fML[k+1,j])    */
  DMLi2 = (int *) vrna_alloc(nfiles*sizeof(int)*(length + 1)); /*                MIN(fML[i+2,k]+fML[k+1,j])    */

  if((turn < 0) || (turn > length))
    turn = length; /* does this make any sense? */

 for(int H=1;H<nfiles;H++) sanity(VC[0],VC[H]);

 for(int H=0;H<nfiles;H++) {
   domains_up        = VC[H]->domains_up;
  /* pre-processing ligand binding production rule(s) */
  if(domains_up && domains_up->prod_cb)
    domains_up->prod_cb(VC[H], domains_up->data);

  /* prefill helper arrays */
  for(j = 0; j <= length; j++){
    //Fmi[j] = 
    DMLi[H*(length+1)+j] = DMLi1[H*(length+1)+j] = DMLi2[H*(length+1)+j] = INF;
  }
 }//endfor H


  /* prefill matrices with init contributions */
 for(int H=0;H<nfiles;H++) {
  for(j = 1; j <= length; j++)
    //for(i = (j > turn ? (j - turn) : 1); i <= j; i++){
    for(i = 1; i <= j; i++){
      My_c(H,Indx(H,i,j)) = My_fML(H,Indx(H,i,j)) = INF;
      if(uniq_ML)
        My_fM1(H,Indx(H,i,j)) = INF;
    }
 }//endfor H
  init_fML(nfiles,length);//on GPU

  /* start recursion */

  if (length <= turn){
    /* clean up memory */
    //free(cc);
    //free(cc1);
    //free(Fmi);
    free(DMLi);
    free(DMLi1);
    free(DMLi2);
    /* return free energy of unfolded chain */
    for(int H=0;H<nfiles;H++) {
      Energy[H] = 0;
    }//endfor H
    return;
  }

  const int ijsize = (length+1)*(length+2)/2;
  //fprintf(stderr,"fill_arrays(vc) length %d, ijsize %d\n",length,ijsize);

  int* energy_hp    = calloc(nfiles*ijsize,sizeof(int));
  int* energy_mb    = calloc(nfiles*ijsize,sizeof(int));
  int* energy_mls   = calloc(nfiles*ijsize,sizeof(int));
  int* energy_3p_00 = calloc(nfiles*ijsize,sizeof(int));
  int* energy_3p_en = calloc(nfiles*ijsize,sizeof(int));

  /*We can move energy_hp out of loop */
 for(int H=0;H<nfiles;H++) {
 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      type          = (unsigned char)Ptype(H,ij);
      hc_decompose  = Hard_constraints(H,ij);

      no_close = (((type==3)||(type==4))&&noGUclosure);

      if (hc_decompose) {   /* we evaluate this pair */
        if(!no_close){
          /* check for hairpin loop */
          energy_hp[H*ijsize+ij] = vrna_E_hp_loop(VC[H], i, j);
	}
      }
    }
 }
 }//endfor H

  /*We can move energy_mb out of loop if we calculate change in energy */
 for(int H=0;H<nfiles;H++) {
 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      type          = (unsigned char)Ptype(H,ij);
      hc_decompose  = Hard_constraints(H,ij);

      no_close = (((type==3)||(type==4))&&noGUclosure);

      if (hc_decompose) {   /* we evaluate this pair */
        if(!no_close){
          /* check for multibranch loops, return change in energy relative to DMLi1 */
          energy_mb[H*ijsize+ij] = mb_loop_fast(VC[H], i, j);
	}
      }
    } /* end of j-loop */

  } /* end of i-loop */
 }//endfor H

 /* take union of above c to fake my_c for vrna_E_int_loop() */
 /* fails because DMLi1 is not set up */

 /*can we move energy_int out of loop ? NO */
 /* fails due to E_int_loop() line 317 c_pq in inner p,q loop */
        /* check for interior loops **
        energy_int[ij] = vrna_E_int_loop(vc, i, j);	*/


  /*We can now move (modified) vrna_E_ml_stems_fast out of main loop */

 for(int H=0;H<nfiles;H++) {
 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      energy_mls[H*ijsize+ij] = vrna_E_ml_stems_fast2(VC[H], i, j);
    }
 }
 }//endfor H


  /* Can we move extend_fm_3p()... out of loop? */
 for(int H=0;H<nfiles;H++) {
 for (i = length-turn-1; i >= 1; i--) { /* i,j in [1..length] */

    for (j = i+turn+1; j <= length; j++) {
      ij            = Indx(H,i,j);
      assert(ij>=0 && ij<ijsize);
      type          = (unsigned char)Ptype(H,ij);
      hc_decompose  = Hard_constraints(H,ij);

      /*  extension with one unpaired nucleotide at the right (3' site)
	  or full branch of (i,j)
      */
      //from extend_fm_3p()...
      const int cp = -1;

      energy_3p_00[H*ijsize+ij] = INF;
      energy_3p_en[H*ijsize+ij] = INF;

  if(ON_SAME_STRAND(i - 1, i, cp)){
    const short* S   = VC[H]->sequence_encoding;
    const int* hc_up = VC[H]->hc->up_ml;
    if(ON_SAME_STRAND(j, j + 1, cp)){
      if(hc_decompose & VRNA_CONSTRAINT_CONTEXT_MB_LOOP_ENC){ //eval_loop

        const int type_ = (type == 0)? 7 : type;

        //e00 = my_c[ij]; //<<<<<<<<<<<<<
	//fprintf(stderr,"extend_fm_3p(%d,%d,*fm,*vc) c[%d] %d\n",i,j,ij,c[ij]);
            energy_3p_00[H*ijsize+ij] = E_MLstem(type_, (i==1) ? S[length] : S[i-1], S[j+1], P);
      }
    }

    if(ON_SAME_STRAND(j - 1, j, cp)){
      if(hc_up[j] > 0){ //eval_loop = () ? (char)1 : (char)0;
          energy_3p_en[H*ijsize+ij] = P->MLbase;
      }
    }
  }
      //end from extend_fm_3p()...
    }
 }
 }//endfor H


#include "fill_arrays_loop.c"

  /* calculate energies of 5' fragments */
 for(int H=0;H<nfiles;H++) {
   E_ext_loop_5(VC[H]);
   Energy[H] = VC[H]->matrices->f5[length];
   //printf("Energy[%d]%d\n",H,Energy[H]);
 }//endfor H

  /* clean up memory */
  free(energy_3p_en);
  free(energy_3p_00);
  free(energy_mls);
  free(energy_mb);
  free(energy_hp);

  //free(cc);
  //free(cc1);
  //free(Fmi);
  free(DMLi);
  free(DMLi1);
  free(DMLi2);

}

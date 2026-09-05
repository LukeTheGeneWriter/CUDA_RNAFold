//WBL 12 Jan 2018 $Revision: 1.16 $ CUDA GGGP ViennaRNA-2.3.0 rf/rf_cuda2

//Modifications:
//WBL 18 Jan 2018 Replace doube 30. by float 30.0f
//WBL 12 Jan 2018 Remove unused functions

#ifndef VIENNA_RNA_PACKAGE_INTERIOR_LOOPS_HX
#define VIENNA_RNA_PACKAGE_INTERIOR_LOOPS_HX

#include <ViennaRNA/utils.h>
#include "ViennaRNA/energy_par.h"
#include <ViennaRNA/data_structures.h>
#include <ViennaRNA/params.h>
#include <ViennaRNA/constraints.h>

#ifdef __GNUC__
# define INLINE inline
#else
# define INLINE
#endif

#ifdef ON_SAME_STRAND
#undef ON_SAME_STRAND
#endif

#define ON_SAME_STRAND(I,J,C)  (((I)>=(C))||((J)<(C)))

/**
 *  @file     interior_loops.h
 *  @ingroup  loops
 *  @brief    Energy evaluation of interior loops for MFE and partition function calculations
 */

/**
 *  @{
 *  @ingroup   loops
 */

/**
 *  <H2>Compute the Energy of an interior-loop</H2>
 *  This function computes the free energy @f$\Delta G@f$ of an interior-loop with the
 *  following structure: <BR>
 *  <PRE>
 *        3'  5'
 *        |   |
 *        U - V
 *    a_n       b_1
 *     .        .
 *     .        .
 *     .        .
 *    a_1       b_m
 *        X - Y
 *        |   |
 *        5'  3'
 *  </PRE>
 *  This general structure depicts an interior-loop that is closed by the base pair (X,Y).
 *  The enclosed base pair is (V,U) which leaves the unpaired bases a_1-a_n and b_1-b_n
 *  that constitute the loop. In this example, the length of the interior-loop is @f$(n+m)@f$
 *  where n or m may be 0 resulting in a bulge-loop or base pair stack.
 *  The mismatching nucleotides for the closing pair (X,Y) are:<BR>
 *  5'-mismatch: a_1<BR>
 *  3'-mismatch: b_m<BR>
 *  and for the enclosed base pair (V,U):<BR>
 *  5'-mismatch: b_1<BR>
 *  3'-mismatch: a_n<BR>
 *  @note Base pairs are always denoted in 5'->3' direction. Thus the enclosed base pair
 *  must be 'turned arround' when evaluating the free energy of the interior-loop
 *  @see scale_parameters()
 *  @see vrna_param_t
 *  @note This function is threadsafe
 * 
 *  @param  n1      The size of the 'left'-loop (number of unpaired nucleotides)
 *  @param  n2      The size of the 'right'-loop (number of unpaired nucleotides)

 *  @param  ns      smaller of n1 and n2
 *  @param  nl      larger of n1 and n2

 *  @param  type    The pair type of the base pair closing the interior loop
 *  @param  type_2  The pair type of the enclosed base pair
 *  @param  si1     The 5'-mismatching nucleotide of the closing pair
 *  @param  sj1     The 3'-mismatching nucleotide of the closing pair
 *  @param  sp1     The 3'-mismatching nucleotide of the enclosed pair
 *  @param  sq1     The 5'-mismatching nucleotide of the enclosed pair
 *  @param  P       The datastructure containing scaled energy parameters
 *  @return The Free energy of the Interior-loop in dcal/mol
 */

//PRIVATE INLINE int
__device__ inline int
IntLoop_X(const int n1,
          const int ns,
          const int nl,
          const int type,
          const int type_2,
          const int si1,
          const int sj1,
          const int sp1,
          const int sq1,
	  //Remainder are in const vrna_param_t *P,
	  //approx in order of how much gcov says they are used
	  const int TerminalAU,
	  const int ninio2,
	  const int bulge[MAXLOOP+1],
	  const int internal_loop[MAXLOOP+1],
	  const float lxc,
	  const int mismatchI[NBPAIRS+1][5][5],
	  const int mismatch1nI[NBPAIRS+1][5][5],
	  const int mismatch23I[NBPAIRS+1][5][5],
	  //gcov says p->stack,P->int11,P->int21,P->int22 seldom used
	  const int stack[NBPAIRS+1][NBPAIRS+1],
	  const int int11[NBPAIRS+1][NBPAIRS+1][5][5],
	  const int int21[NBPAIRS+1][NBPAIRS+1][5][5][5],
	  const int int22[NBPAIRS+1][NBPAIRS+1][5][5][5][5] /*,
	  const int v*/){ //verbose
//printf("E_IntLoop(%d,%d,%d,%d,%d,%d,%d,P)\n",
//	 n1,n2,type,type_2,si1,sj1,sp1,sq1);

  /* compute energy of degree 2 loop (stack bulge or interior) */
  int u, energy;
  energy = INF;

  if (nl == 0) {
    //if(v)printf("n1==0 stack[%d][%d]",type,type_2);
    return stack[type][type_2];  /* stack */
  }
  if (ns==0) {                      /* bulge */
    energy = (nl<=MAXLOOP)? bulge[nl]:
      (bulge[30]+(int)(lxc*log(nl/30.0f)));
    if (nl==1) {
      //if(v)printf("nl==1 stack[%d][%d] ",type,type_2);
      //if(v)printf("ns==0 bulge[%d] %d %d ",nl,type,type_2);
      energy += stack[type][type_2];
    }
    else {
      //if(v)printf("ns==0 bulge[%d] %d %d ",nl,type,type_2);
      //if(v)printf("nl!=1 %d %d TerminalAU?? ",type,type_2);
      if (type>2) energy += TerminalAU;
      if (type_2>2) energy += TerminalAU;
    }
    return energy;
  }
  else {                            /* interior loop */
    if (ns==1) {
      if (nl==1) {                    /* 1x1 loop */
	//if(v)printf("ns==1 nl==1 int11[%d][%d][%d][%d]",type,type_2,si1,sj1);
        return int11[type][type_2][si1][sj1];
      }
      if (nl==2) {                  /* 2x1 loop */
        if (n1==1) {
	  //if(v)printf("ns==1 nl==2 n1==1 int21[%d][%d][%d][%d][%d]",type,type_2,si1,sq1,sj1);
          energy = int21[type][type_2][si1][sq1][sj1];
	}
	else {
	  //if(v)printf("ns==1 nl==2 n1!=1 int21[%d][%d][%d][%d][%d]",type_2,type,sq1,si1,sp1);
          energy = int21[type_2][type][sq1][si1][sp1];
	}
        return energy;
      }
      else {  /* 1xn loop */
	//if(v)printf("ns==1 nl!=1 && nl!=2 internal_loop[%d] ",nl+1);
	//if(v)printf("mismatch1nI[%d][%d][%d] + mismatch1nI[%d][%d][%d]",type,si1,sj1, type_2,sq1,sp1);
        energy = (nl+1<=MAXLOOP)?(internal_loop[nl+1]) : (internal_loop[30]+(int)(lxc*log((nl+1)/30.0f)));
        energy += MIN2(MAX_NINIO, (nl-ns)*ninio2);
        energy += mismatch1nI[type][si1][sj1] + mismatch1nI[type_2][sq1][sp1];
        return energy;
      }
    }
    else if (ns==2) {
      if(nl==2)      {              /* 2x2 loop */
	//if(v)printf("ns!=1 ns==2 nl==2 int22[%d][%d][%d][%d][%d][%d]",type,type_2,si1,sp1,sq1,sj1);
        return int22[type][type_2][si1][sp1][sq1][sj1];}
      else if (nl==3){              /* 2x3 loop */
	//if(v)printf("ns!=1 ns==2 nl==3 internal_loop_5 ");
	//if(v)printf("mismatch23I[%d][%d][%d] + mismatch23I[%d][%d][%d]",type,si1,sj1, type_2,sq1,sp1);
        energy = internal_loop[5]+ninio2;
        energy += mismatch23I[type][si1][sj1] + mismatch23I[type_2][sq1][sp1];
        return energy;
      }

    }
    { /* generic interior loop (no else here!)*/
      u = nl + ns;
      //if(v)printf("other internal_loop[%d] xninio2 ",u);
      //if(v)printf("mismatchI[%d][%d][%d] + mismatchI[%d][%d][%d]",type,si1,sj1, type_2,sq1,sp1);
      energy = (u <= MAXLOOP) ? (internal_loop[u]) : (internal_loop[30]+(int)(lxc*log((u)/30.0f)));

      energy += MIN2(MAX_NINIO, (nl-ns)*ninio2);

      energy += mismatchI[type][si1][sj1] + mismatchI[type_2][sq1][sp1];
    }
  }
  return energy;
}

#endif

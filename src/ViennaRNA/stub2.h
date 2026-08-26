/* Modifications for eventual CUDA version $Revision: 1.30 $
WBL 24 Aug 2026 add struct energy_3p etc (for commit)
WBL  7 Aug 2026 did not (yet) allow user to select GPU (ie take default)
WBL 19 Jul 2026 Allow arrays to exceed two billion elements
WBL  8 Jan 2018 Extend linkage for CUDA interface in modular_decomposition.cu
WBL  3 Dec 2017 investigate data dependence in E_mb_loop_fast
  split off multibranch_loops.c r1.10 for time being
*/

struct energy_3p {
  //int c may be reorder my_c later, may be remove energy_3p_en later
  int energy_3p_00;
  int energy_3p_en;
  int energy_mls;
};
#ifdef __cplusplus
extern "C" void
#else
PUBLIC void
#endif
choose_gpu(const int argc, const char **argv); //not yet! updates argc and argv

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_fML(const int nfiles,const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_fML(const int nfiles,
	 const int i, const int turn, const int length,
	 const int* energy_min);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_min_fML(const int nfiles, const int i, const int turn, const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
modular_decomposition_i(const int nfiles,
			const int i, const int turn, const int length,
		      //const int* indx,
		      //const int ijsize,  //for sanity checks
		      //const int* my_fML,
			int* DMLi);

PUBLIC int
extend_fm_3p( const int i,
              const int j,
              const int *fm,
              const vrna_fold_compound_t *vc);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu(const int nfiles, const int length);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
init_gpu2(const int nfiles, const vrna_fold_compound_t **VC, const int turn_, const int length, const int block_size);

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_my_c(const int nfiles,
	  const int i, const int turn, const int length,
	  const int* min_e); //in

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
int_loop_i(const int nfiles,
	   const vrna_fold_compound_t **VC,
	   const int i, const int turn, const int length,
	   /*const int* indx, const int ijsize,
	   const char* hard_constraints, const int* my_c,*/
	   int* energy_min ); //out


extern int* d_energy_min;
extern int* d_fml_j;  //my_fML
extern int* d_dml;  //DMLi

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
int_loop_mls(const int nfiles,
	     const vrna_fold_compound_t **VC,
	     const int i, /*const int turn,*/ const int length,
	     const long long ijsize,
	     const int* my_c,          //contents of My_c(H,ij)
	     const int en,
	     const struct energy_3p*,
	     int* energy_min); //out

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
int_loop_DMLi(const int nfiles,
	      const int i, /*const int turn,*/ const int length,
	      const long long ijsize,
	      const int* energy_min,
	      const int* DMLi,
	      const vrna_fold_compound_t **VC); //out My_fML

PUBLIC void
par_mfe(const int nfiles,
	const vrna_fold_compound_t** VC,
	const char** Structure,
	float* EN); //out

#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
sanity(const vrna_fold_compound_t* vc0, const vrna_fold_compound_t* vc);

PRIVATE void
par_fill_arrays(const int nfiles, const vrna_fold_compound_t **VC, int* Energy);

int
mb_loop_fast( vrna_fold_compound_t *vc,
                int i,
                int j);

#ifdef __CUDACC__
extern "C"
__host__ __device__
#else
extern
#endif
long long Hindx(const int H, const int nfiles,
		const int i, const int j, const int length);

//inline functions needed by both int_loop.cu and modular_decomposition.cu
#ifdef __CUDACC__
__host__ __device__
inline long long Indx(const int i, const int j) { //j*(j-1)/2+i
  const long long j_1 = j-1; //force 64 bit calculation for my_c
  return j*j_1/2+i;
}

__host__ __device__
inline long long
Hoff(const int H, const int length){ //H*((length+1)*(length+2)/2);
  const long long l1 = length+1;
  const long long l2 = length+2;
  return H*(l1*l2)/2;
}
#endif /*__CUDACC__*/

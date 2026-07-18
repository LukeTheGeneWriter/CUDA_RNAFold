/* Modifications for eventual CUDA version $Revision: 1.13 $
WBL  8 Jan 2018 Extend linkage for CUDA interface in modular_decomposition.cu
WBL  3 Dec 2017 investigate data dependence in E_mb_loop_fast
  split off multibranch_loops.c r1.10 for time being
*/

#ifdef __cplusplus
extern "C" void
#else
PUBLIC void
#endif
choose_gpu(int argc, char **argv); //updates argc and argv

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

// CUDA-graph-captured fusion of load_fML() + modular_decomposition_i() +
// load_min_fML() -- see definition in modular_decomposition.cu for why these
// three (and only these three) can be captured as a single graph.
#ifdef __cplusplus
extern "C" /*PUBLIC*/ void
#else
PUBLIC void
#endif
load_fML_modular_decomposition_load_min_fML(const int nfiles,
					     const int i, const int turn, const int length,
					     const int* energy_min, //in
					     int* DMLi);            //out

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

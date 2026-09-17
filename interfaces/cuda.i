/**********************************************/
/* BEGIN interface for the CUDA batch backend */
/**********************************************/

/*
 * WHY THIS FILE EXISTS.
 *
 * Users reach ViennaRNA through `import RNA`, not through the CLI. Until the
 * accelerated path is reachable there, its adoption is whatever the command
 * line gets -- which, for a tool whose whole advantage is folding a BATCH at
 * once, is close to nothing.
 *
 * And it is not reachable by accident. RNA.i:243 ignores every function whose
 * name starts with `vrna_`:
 *
 *     %rename("$ignore", %$isfunction, regextarget=1) "^vrna_";
 *
 * and each one is then explicitly renamed back into the API. So
 * vrna_mfe_batch(), vrna_cuda_devices() and vrna_cuda_register_batch_backend()
 * are not partially wrapped or awkwardly wrapped -- they are **absent**, and
 * adding `%include <ViennaRNA/mfe/cuda/engine.h>` would not change that.
 *
 * Nor would wrapping the C signature help if it were:
 *
 *     int vrna_mfe_batch(vrna_fold_compound_t **fcs, size_t n,
 *                        char **structures, float *energies);
 *
 * which in Python means "hand me a pointer to an array of pointers, plus two
 * output buffers you allocated yourself".
 *
 * Hence three helpers, in the `my_*` style this interface already uses for
 * exactly this reason (see my_alifold in mfe.i):
 *
 *     RNA.cuda_devices()    how many usable CUDA devices there are. 0 when the
 *                           library was built without CUDA, which is not an
 *                           error -- folding still works
 *     RNA.cuda_enable()     register the batch backend. Idempotent, and 0 here
 *                           simply means the batch below runs on the CPU
 *     RNA.cuda_fold([...])  fold a LIST of sequences, get a list of
 *                           (structure, energy) tuples
 *
 * cuda_fold() works with or without a GPU, deliberately: a script should not
 * have to branch on hardware to be correct, and the answer is the same either
 * way -- byte-identical is the bar this whole backend is held to.
 */

%{
/* extern "C", like RNA.i:16 does for every other header. The wrapper is
 * compiled as C++ and none of ViennaRNA's headers carry their own linkage
 * guard, so without this the generated code calls _Z17vrna_cuda_devicesv and
 * the module fails to load with an undefined symbol -- at import time, not at
 * link time, which is the expensive way to find out. */
extern "C" {
#include <ViennaRNA/mfe/cuda/engine.h>
#include <ViennaRNA/mfe/global.h>
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/model.h>
#include <ViennaRNA/utils/basic.h>
}
%}

%template(StructureEnergy)  std::pair<std::string, float>;
%template(CudaFoldResult)   std::vector<std::pair<std::string, float> >;

%rename (cuda_devices) my_cuda_devices;
%rename (cuda_enable)  my_cuda_enable;
%rename (cuda_fold)    my_cuda_fold;

%{
  unsigned int
  my_cuda_devices(void)
  {
    return vrna_cuda_devices();
  }


  unsigned int
  my_cuda_enable(void)
  {
    return vrna_cuda_register_batch_backend();
  }


  std::vector<std::pair<std::string, float> >
  my_cuda_fold(std::vector<std::string>  sequences,
               vrna_md_t                *md)
  {
    std::vector<std::pair<std::string, float> > out;
    size_t                                      n = sequences.size();

    if (n == 0)
      return out;

    /* Registration is idempotent, and 0 is not a failure: vrna_mfe_batch()
     * then folds every compound on the host through upstream's own vrna_mfe().
     * Same answers, no GPU. */
    (void)vrna_cuda_register_batch_backend();

    vrna_fold_compound_t  **fcs        = (vrna_fold_compound_t **)vrna_alloc(sizeof(vrna_fold_compound_t *) * n);
    char                  **structures = (char **)vrna_alloc(sizeof(char *) * n);
    float                  *energies   = (float *)vrna_alloc(sizeof(float) * n);

    for (size_t i = 0; i < n; i++) {
      fcs[i]        = vrna_fold_compound(sequences[i].c_str(), md, VRNA_OPTION_MFE);
      structures[i] = (char *)vrna_alloc(sizeof(char) * (sequences[i].size() + 1));
    }

    int ok = vrna_mfe_batch(fcs, n, structures, energies);

    for (size_t i = 0; i < n; i++) {
      if (ok)
        out.push_back(std::make_pair(std::string(structures[i]), energies[i]));

      free(structures[i]);
      vrna_fold_compound_free(fcs[i]);
    }

    free(structures);
    free(energies);
    free(fcs);

    return out;
  }


  std::vector<std::pair<std::string, float> >
  my_cuda_fold(std::vector<std::string> sequences)
  {
    return my_cuda_fold(sequences, (vrna_md_t *)NULL);
  }
%}

unsigned int my_cuda_devices(void);
unsigned int my_cuda_enable(void);

std::vector<std::pair<std::string, float> >
my_cuda_fold(std::vector<std::string>  sequences,
             vrna_md_t                *md);

std::vector<std::pair<std::string, float> >
my_cuda_fold(std::vector<std::string> sequences);

/**********************************************/
/* END interface for the CUDA batch backend   */
/**********************************************/

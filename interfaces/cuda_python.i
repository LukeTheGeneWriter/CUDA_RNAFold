/**********************************************/
/* BEGIN: the GPU by default in RNA.fold and  */
/* fold_compound.mfe (Python only)            */
/**********************************************/

/* Included LAST in RNA.i, after every proxy class exists: it rebinds two names that
 * mfe.i and fold_compound.i define. There is no separate call to learn -- the normal
 * ones use the device, and cpu_only=True calls upstream's exact functions instead.
 *
 *   RNA.fold(seq)                     -> [structure, mfe]       on the device
 *   RNA.fold(seq, constraint)         -> [structure, mfe]       on the device
 *   RNA.fold([seq, ...])              -> [(structure, mfe), ...] one device batch
 *   RNA.fold(..., cpu_only=True)      -> upstream's own RNA.fold
 *   fc.mfe()                          -> [structure, mfe]       on the device
 *   fc.mfe(cpu_only=True)             -> upstream's own fc.mfe()
 *
 * "On the device" means: through vrna_mfe_batch(), which uses the GPU when this build
 * has one, a device is present and the model is supported, and otherwise folds with
 * upstream's vrna_mfe(). So the default never changes an answer and never fails on a
 * machine without a GPU; cpu_only=True only guarantees the device is not touched.
 * After fc.mfe() the fold compound's MFE matrices are populated either way, so
 * fc.backtrack() and friends work as they do upstream.
 *
 * Two environment variables, read at every call:
 *   RNA_GPU=0          the host folds everything, as for RNAfold (the batch backend
 *                      declines; RNA.cuda_batches() stops counting)
 *   RNA_GPU_VERBOSE=1  print the backend's diagnostics on stderr, which RNAfold always
 *                      prints and the binding leaves out by default */

#ifdef SWIGPYTHON
%pythoncode %{

_fold_upstream = fold


def fold(sequence, *args, cpu_only=False):
    """fold(sequence[, constraints], cpu_only=False) -> [structure, mfe]
fold([sequence, ...], cpu_only=False) -> [(structure, mfe), ...]

Minimum free energy structure. Uses the GPU by default when this build has one and a
device is present (otherwise, and for any model the device does not support, the host
folds it -- the answer is identical). A list of sequences is folded as one GPU batch,
which is where the device pays. cpu_only=True never touches the device and calls
ViennaRNA's own fold(). RNA_GPU=0 in the environment also keeps the fold on the host,
and RNA_GPU_VERBOSE=1 prints the GPU backend's diagnostics on stderr.
"""
    if isinstance(sequence, (list, tuple)):
        if args:
            raise TypeError("fold(): a list of sequences takes no constraints")
        seqs = [str(s) for s in sequence]
        if cpu_only:
            return [tuple(_fold_upstream(s)) for s in seqs]
        return [(s, e) for s, e in cuda_fold(seqs)]
    if cpu_only:
        return _fold_upstream(sequence, *args)
    return fold_device(sequence, *args)


_fc_mfe_upstream = fold_compound.mfe


def _fc_mfe(self, *args, cpu_only=False, **kwargs):
    """mfe(cpu_only=False) -> [structure, energy]

Minimum free energy structure of this fold compound. Uses the GPU by default when this
build has one and a device is present (otherwise, and for any model the device does not
support, the host folds it -- the answer is identical). Afterwards the MFE matrices are
populated either way, so backtracking works as upstream. cpu_only=True never touches
the device and calls ViennaRNA's own mfe(). RNA_GPU=0 in the environment also keeps
the fold on the host, and RNA_GPU_VERBOSE=1 prints the GPU backend's diagnostics.
"""
    if cpu_only or args or kwargs:
        return _fc_mfe_upstream(self, *args, **kwargs)
    return self._mfe_device()


fold_compound.mfe = _fc_mfe

%}
#endif

/**********************************************/
/* END: the GPU by default in RNA.fold and    */
/* fold_compound.mfe                          */
/**********************************************/

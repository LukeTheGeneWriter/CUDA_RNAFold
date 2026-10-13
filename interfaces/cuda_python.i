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
 *                      prints and the binding leaves out by default
 *
 * multiprocessing: a CUDA context does not survive fork(), so a worker FORKED after
 * this process used the GPU folds on the host (the engine's cuda_process_ok()) --
 * the same answer, without the device, and never a hang. Workers started with
 * multiprocessing.get_context("spawn") (or "forkserver") have their own context and
 * use the GPU; so do fork workers whose parent never touched it. */

#ifdef SWIGPYTHON
%pythoncode %{

# ---- resource budgets (PORT_BUDGETS.md) ----------------------------------------------
# Python defaults to HALF of the machine's RAM, VRAM and threads, so a fold in a script
# or a notebook leaves the rest of the laptop usable; RNAfold defaults to full. The
# environment (RNA_RAM_BUDGET / RNA_VRAM_BUDGET / RNA_THREAD_BUDGET) overrides this
# default, and set_budget() overrides both.
_budget_default("half")
_BUDGET_WHICH = {"ram": 0, "vram": 1, "threads": 2}
_budget_mine = {}


def set_budget(ram=None, vram=None, threads=None, reset=False):
    """set_budget(ram=None, vram=None, threads=None, reset=False)

How much of this machine a fold may take. Each is a number -- MB for ram and vram,
a count for threads -- or a fraction of the machine's total: "quarter", "half",
"threequarter", "full". None leaves a resource as it is; reset=True returns all
three to the environment or the default first.

A budget only lowers the ceiling: the fold still adapts to what is free right now,
so "full" means as much as is safely available, as before. Python's default is
"half". A single sequence too large for its budget is folded anyway, with a warning.
"""
    if reset:
        for k in _BUDGET_WHICH:
            _budget_set(_BUDGET_WHICH[k], "")
        _budget_mine.clear()
    before = dict(_budget_mine)
    for k, v in (("ram", ram), ("vram", vram), ("threads", threads)):
        if v is None:
            continue
        # "" would reach the C layer as "back to the default": that is reset=True's job,
        # and a blank value from a config file must not silently lift a budget
        if (not str(v).strip()) or (not _budget_set(_BUDGET_WHICH[k], str(v))):
            # all or nothing: undo what this call already applied
            for j in _BUDGET_WHICH:
                _budget_set(_BUDGET_WHICH[j], before.get(j, ""))
            _budget_mine.clear()
            _budget_mine.update(before)
            raise ValueError("set_budget(): %s=%r -- want quarter, half, threequarter, "
                             "full or a positive number" % (k, v))
        _budget_mine[k] = str(v)


def budget():
    """budget() -> {"ram": MB or None, "vram": MB or None, "threads": n}

The budgets in force now, resolved against this machine (None: no ceiling beyond what
is free). RAM is re-read from the host at every fold, so this is a snapshot."""
    out = {}
    for k, w in _BUDGET_WHICH.items():
        v = _budget_mb(w)
        out[k] = int(v) if k == "threads" else (None if v < 0 else round(v, 1))
    return out


class _budget_for_call(object):
    """ram=/vram=/threads= on one call: set for the call, then restored."""
    def __init__(self, ram, vram, threads):
        self.req = {k: v for k, v in (("ram", ram), ("vram", vram), ("threads", threads))
                    if v is not None}

    def __enter__(self):
        if self.req:
            self.saved = dict(_budget_mine)
            set_budget(**self.req)

    def __exit__(self, *exc):
        if self.req:
            for k in self.req:
                _budget_set(_BUDGET_WHICH[k], self.saved.get(k, ""))
                if k in self.saved:
                    _budget_mine[k] = self.saved[k]
                else:
                    _budget_mine.pop(k, None)
        return False


_fold_upstream = fold


def fold(sequence, *args, cpu_only=False, ram=None, vram=None, threads=None):
    """fold(sequence[, constraints], cpu_only=False) -> [structure, mfe]
fold([sequence, ...], cpu_only=False) -> [(structure, mfe), ...]

Minimum free energy structure. Uses the GPU by default when this build has one and a
device is present (otherwise, and for any model the device does not support, the host
folds it -- the answer is identical). A list of sequences is folded as one GPU batch,
which is where the device pays. cpu_only=True never touches the device and calls
ViennaRNA's own fold(). RNA_GPU=0 in the environment also keeps the fold on the host,
and RNA_GPU_VERBOSE=1 prints the GPU backend's diagnostics on stderr.

With multiprocessing, workers FORKED after this process used the GPU fold on the host
(a CUDA context does not survive fork); start them with
multiprocessing.get_context("spawn") to give each worker the GPU.

ram=, vram=, threads= set a budget for this call only (see set_budget()).
"""
    with _budget_for_call(ram, vram, threads):
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


def _fc_mfe(self, *args, cpu_only=False, ram=None, vram=None, threads=None, **kwargs):
    """mfe(cpu_only=False, ram=None, vram=None, threads=None) -> [structure, energy]

Minimum free energy structure of this fold compound. Uses the GPU by default when this
build has one and a device is present (otherwise, and for any model the device does not
support, the host folds it -- the answer is identical). Afterwards the MFE matrices are
populated either way, so backtracking works as upstream. cpu_only=True never touches
the device and calls ViennaRNA's own mfe(). RNA_GPU=0 in the environment also keeps
the fold on the host, and RNA_GPU_VERBOSE=1 prints the GPU backend's diagnostics.
ram=, vram=, threads= set a budget for this call only (see RNA.set_budget()).
"""
    with _budget_for_call(ram, vram, threads):
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

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest

if __name__ == '__main__':
    from py_include import taprunner
    import RNApath
    RNApath.addSwigInterfacePath()

import RNA


def rand_seq(n, rng):
    return "".join(rng.choice("ACGU") for _ in range(n))


class env(object):
    """Set (value) or unset (None) one environment variable for a with-block"""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __enter__(self):
        self.saved = os.environ.get(self.name)
        if self.value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.value

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.saved


def stderr_of(call):
    """What call() writes to file descriptor 2 -- the backend writes there from C,
    below sys.stderr, so only a dup2 can see it"""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 2)
        try:
            call()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        f.seek(0)
        return f.read().decode("utf-8", "replace")


def cpu(seq, md=None):
    fc = RNA.fold_compound(seq, md) if md is not None else RNA.fold_compound(seq)
    # cpu_only=True is LOAD-BEARING: fc.mfe() uses the device by default since
    # 2026-09-30, and without it every parity check below compares the GPU to itself.
    return fc.mfe(cpu_only=True)


class cuda_batchTest(unittest.TestCase):
    """The CUDA batch API, which is correct with or without a GPU.

    Every check here is a PARITY check against RNA.fold_compound().mfe() --
    the same bar the backend is held to everywhere else -- so the suite is
    meaningful on a machine with no CUDA at all, where cuda_fold() simply
    folds through upstream's own vrna_mfe(). RNA.cuda_devices() reports which
    of the two was exercised; 0 is not a failure.
    """

    DATADIR = ""

    def test_cuda_devices(self):
        """Device count is queryable and never negative"""
        self.assertTrue(RNA.cuda_devices() >= 0)

    def test_cuda_enable_is_idempotent(self):
        """Registering the backend twice is not an error"""
        RNA.cuda_enable()
        RNA.cuda_enable()
        self.assertTrue(RNA.cuda_devices() >= 0)

    def test_empty_batch(self):
        """An empty batch is an empty result, not a crash"""
        self.assertEqual(len(RNA.cuda_fold([])), 0)

    def test_records_too_short_to_pair(self):
        """A batch whose records are all <= 3 nt folds instead of crashing

        Such a batch never sweeps, and the no-sweep branch of par_fill_arrays()
        used to fill the host c/fML triangles -- which par_mfe() had already freed
        and NULLed, so RNA.cuda_fold(["A"]) SEGFAULTED the interpreter. RNAfold
        never hands the device a chunk like this, so no CLI bar could see it."""
        for seqs in (["A"], ["AC", "ACG"], ["A", "C", "GGG"], ["", "A"]):
            got = [(s, round(e, 2)) for s, e in RNA.cuda_fold(seqs)]
            ref = [(RNA.fold(q, cpu_only=True)[0], round(RNA.fold(q, cpu_only=True)[1], 2))
                   for q in seqs]
            self.assertEqual(got, ref, seqs)

    # ---- the device inside the NORMAL calls (2026-09-30) ---------------------------
    # RNA.fold and fold_compound.mfe use the GPU by default; cpu_only=True calls
    # upstream's own. RNA.cuda_batches() is the positive evidence: a host fallback is
    # byte-identical, so only the counter can say which path ran.

    def test_fold_uses_the_device_by_default(self):
        """RNA.fold(seq) folds on the device, and agrees with cpu_only=True"""
        rng = random.Random(11)
        seq = rand_seq(300, rng)
        b0 = RNA.cuda_batches()
        got = RNA.fold(seq)
        if RNA.cuda_devices() > 0:
            self.assertEqual(RNA.cuda_batches(), b0 + 1, "the default did not use the device")
        else:
            self.assertEqual(RNA.cuda_batches(), b0)
        ref = RNA.fold(seq, cpu_only=True)
        self.assertEqual(got[0], ref[0])
        self.assertAlmostEqual(got[1], ref[1], 2)

    def test_cpu_only_never_touches_the_device(self):
        """cpu_only=True leaves the device alone in every entry point"""
        rng = random.Random(12)
        seq = rand_seq(200, rng)
        b0 = RNA.cuda_batches()
        RNA.fold(seq, cpu_only=True)
        RNA.fold([seq, seq], cpu_only=True)
        RNA.fold_compound(seq).mfe(cpu_only=True)
        self.assertEqual(RNA.cuda_batches(), b0)

    def test_fold_accepts_a_list(self):
        """RNA.fold([...]) is one device batch of (structure, mfe) tuples"""
        rng = random.Random(13)
        seqs = [rand_seq(n, rng) for n in (90, 210, 333)]
        b0 = RNA.cuda_batches()
        got = RNA.fold(seqs)
        if RNA.cuda_devices() > 0:
            self.assertEqual(RNA.cuda_batches(), b0 + 1, "a list should be ONE batch")
        self.assertIsInstance(got, list)
        for (structure, energy), seq in zip(got, seqs):
            ref = RNA.fold(seq, cpu_only=True)
            self.assertEqual(structure, ref[0])
            self.assertAlmostEqual(energy, ref[1], 2)

    def test_rna_gpu_0_keeps_every_entry_point_on_the_host(self):
        """RNA_GPU=0 is obeyed by the library, not only by RNAfold

        RNAfold.c's admission gate was the switch's only reader, so the binding used
        the device with RNA_GPU=0 set and RNA.cuda_batches() counted it. The second
        half is the control: the same calls with the switch cleared must reach the
        device again, or the first half proves nothing on a machine with a GPU."""
        rng = random.Random(14)
        seqs = [rand_seq(n, rng) for n in (120, 240)]

        def every_entry_point():
            return (RNA.fold(seqs[0]), RNA.fold(seqs), RNA.cuda_fold(seqs),
                    RNA.fold_compound(seqs[1]).mfe())

        with env("RNA_GPU", "0"):
            b0 = RNA.cuda_batches()
            one, many, batch, fc = every_entry_point()
            self.assertEqual(RNA.cuda_batches(), b0, "RNA_GPU=0 reached the device")
        self.assertEqual(one[0], cpu(seqs[0])[0])
        for (s, e), (t, f), q in zip(many, batch, seqs):
            self.assertEqual(s, cpu(q)[0])
            self.assertEqual(t, cpu(q)[0])
        self.assertEqual(fc[0], cpu(seqs[1])[0])

        if RNA.cuda_devices() > 0:
            for value in (None, "1"):
                with env("RNA_GPU", value):
                    b0 = RNA.cuda_batches()
                    every_entry_point()
                    self.assertEqual(RNA.cuda_batches(), b0 + 4,
                                     "RNA_GPU=%s should use the device" % value)

    def test_diagnostics_are_quiet_unless_asked_for(self):
        """The backend's stderr diagnostics are off in the binding by default

        RNAfold prints them on every run, and the harnesses grep them, so they stay
        on there. Through the binding they were ~20 lines per process and six per
        batch on every call. RNA_GPU_VERBOSE=1 brings them back -- which is also the
        control that this capture can see the lines at all."""
        rng = random.Random(15)
        seqs = [rand_seq(n, rng) for n in (150, 260)]

        def calls():
            RNA.fold(seqs[0])
            RNA.fold(seqs)
            RNA.fold_compound(seqs[1]).mfe()

        with env("RNA_GPU_VERBOSE", None):
            quiet = stderr_of(calls)
        self.assertEqual(quiet, "", "diagnostics printed by default:\n" + quiet)

        if RNA.cuda_devices() > 0:
            with env("RNA_GPU_VERBOSE", "1"):
                loud = stderr_of(calls)
            self.assertEqual(loud.count("sweep shape:"), 3, loud)

    def test_fc_mfe_on_the_device_keeps_the_matrices(self):
        """After a device fc.mfe(), backtracking works exactly as upstream"""
        rng = random.Random(14)
        seq = rand_seq(400, rng)
        fc_d, fc_h = RNA.fold_compound(seq), RNA.fold_compound(seq)
        b0 = RNA.cuda_batches()
        d, h = fc_d.mfe(), fc_h.mfe(cpu_only=True)
        if RNA.cuda_devices() > 0:
            self.assertEqual(RNA.cuda_batches(), b0 + 1)
        self.assertEqual(d[0], h[0])
        self.assertAlmostEqual(d[1], h[1], 2)
        # c/fML must be populated afterwards; before vrna_cuda_keep_matrices() they
        # were NULL after a device fold and this would crash
        for length in (None, 150):
            bd = fc_d.backtrack() if length is None else fc_d.backtrack(length)
            bh = fc_h.backtrack() if length is None else fc_h.backtrack(length)
            self.assertEqual(bd[0], bh[0])
            self.assertAlmostEqual(bd[1], bh[1], 2)

    def test_batch_matches_the_cpu(self):
        """A batch agrees with vrna_mfe() record for record"""
        rng = random.Random(4242)
        seqs = [rand_seq(n, rng) for n in (80, 137, 240, 61)]
        for seq, (structure, energy) in zip(seqs, RNA.cuda_fold(seqs)):
            ref_structure, ref_energy = cpu(seq)
            self.assertEqual(structure, ref_structure)
            self.assertAlmostEqual(energy, ref_energy, 2)

    def test_second_batch_may_be_longer(self):
        """A later batch is sized for itself, not for the first one

        Regression. init_gpu/2/3 each open with `if(!first) return;`, so the
        device buffers a batch allocates are the batch's own -- and they were
        released only by RNAfold.c, which was the one caller. A second,
        LONGER batch in the same process then read and wrote buffers sized
        for the first: CUDA error 719, or an invalid-argument copy. The
        teardown now happens in the batch callback, where it belongs.

        RNAfold could never see it (records are sorted descending, so no
        chunk outgrows the first), which is exactly why the API needs the
        test and the CLI's parity runs do not supply it.
        """
        rng = random.Random(11)
        short, long_ = [rand_seq(90, rng)], [rand_seq(320, rng)]
        for batch in (short, long_, short):
            for seq, (structure, energy) in zip(batch, RNA.cuda_fold(batch)):
                ref_structure, ref_energy = cpu(seq)
                self.assertEqual(structure, ref_structure)
                self.assertAlmostEqual(energy, ref_energy, 2)

    def test_second_batch_may_use_a_different_model(self):
        """A later batch folds under its OWN model

        The same regression seen from the other side, and the nastier half:
        the per-batch energy-parameter upload sits below that early return,
        so a second batch at a different temperature was folded with the
        FIRST batch's tables -- no error, and perfectly plausible output.
        """
        seq = "GGGAAACCCGGGAAACCCGGGAAACCC"
        md = RNA.md()
        md.temperature = 25.0

        for model in (None, md, None):
            structure, energy = RNA.cuda_fold([seq], model)[0] if model is not None \
                                else RNA.cuda_fold([seq])[0]
            ref_structure, ref_energy = cpu(seq, model)
            self.assertEqual(structure, ref_structure)
            self.assertAlmostEqual(energy, ref_energy, 2)

    def test_int16_may_stand_down_between_batches(self):
        """A batch after an int16 stand-down still folds correctly

        Regression, and a SHIPPED one: int16 fML became the default on
        2026-09-27 and this broke every batch after the first in a process.

        rnafold_fml_int16_stand_down() turns int16 off for the rest of the
        process when a model turns out not to support it (--noLP, a
        non-default salt). teardown_gpu() decided WHICH buffers to free by
        asking rnafold_fml_int16() -- a question whose answer had just
        changed -- so it freed d_fml_j16/d_fml_b and left the pointers set.
        md_cell selects its path on `if(fml_j16)`, so the next batch read
        FREED DEVICE MEMORY through a dangling pointer:

            plain, then noLP    wrong structure, no error, and every later
                                call in the process stays wrong
            noLP first          fine -- int16 is off before anything is
                                allocated, so nothing dangles

        The order below is the failing one on purpose. Reversing it passes
        even with the defect present, which is why this test asserts the
        transition rather than just "two batches work".

        RNAfold cannot reach this: one batch shape and one model per run,
        then exit. Every parity bar in this project runs through that CLI,
        so all of them were blind to it -- the same reason the teardown
        defect in test_second_batch_may_be_longer needed this API to be
        found at all.
        """
        rng = random.Random(1616)
        nolp = RNA.md()
        nolp.noLP = 1

        # plain FIRST, so int16 is on and allocates its buffers
        for model in (None, nolp, None):
            batch = [rand_seq(150, rng)]
            got = RNA.cuda_fold(batch, model) if model is not None \
                  else RNA.cuda_fold(batch)
            for seq, (structure, energy) in zip(batch, got):
                ref_structure, ref_energy = cpu(seq, model)
                self.assertEqual(structure, ref_structure,
                                 "wrong structure after an int16 stand-down "
                                 "(model=%s)" % ("noLP" if model else "plain"))
                self.assertAlmostEqual(energy, ref_energy, 2)

    def test_model_details_reach_the_fold(self):
        """A model handed to cuda_fold() is the model that folds"""
        rng = random.Random(7)
        seqs = [rand_seq(150, rng), rand_seq(210, rng)]
        md = RNA.md()
        md.noLP = 1
        for seq, (structure, energy) in zip(seqs, RNA.cuda_fold(seqs, md)):
            ref_structure, ref_energy = cpu(seq, md)
            self.assertEqual(structure, ref_structure)
            self.assertAlmostEqual(energy, ref_energy, 2)


# A fresh interpreter: fold on the device first (or not), then fork a Pool. Run as a
# separate process because the test runner's own process has already used the device,
# and because a regression here is a HANG, which only a timeout from outside can catch.
FORK_PROBE = r'''
import json, multiprocessing as mp, random, sys
sys.path.insert(0, %r)
import RNA

def work(seqs):
    return [tuple(x) for x in RNA.fold(seqs)], RNA.cuda_devices(), RNA.cuda_batches()

if __name__ == "__main__":
    rng = random.Random(7)
    jobs = [["".join(rng.choice("ACGU") for _ in range(200)) for _ in range(3)] for _ in range(4)]
    if %r:
        RNA.fold(jobs[0])
    parent_batches = RNA.cuda_batches()
    with mp.get_context("fork").Pool(2) as pool:
        out = pool.map_async(work, jobs).get(timeout=120)
    ok = all(got == [tuple(RNA.fold(s, cpu_only=True)) for s in job]
             for (got, _, _), job in zip(out, jobs))
    print("RESULT " + json.dumps({"ok": ok, "parent_batches": parent_batches,
                                  "child_devices": [d for _, d, _ in out],
                                  "child_batches": [b for _, _, b in out]}))
'''


def module_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(RNA.__file__)))


def fork_probe(parent_uses_device):
    p = subprocess.run([sys.executable, "-c", FORK_PROBE % (module_dir(), parent_uses_device)],
                       capture_output=True, text=True, timeout=300)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise AssertionError("fork probe produced no result (rc %d): %s"
                         % (p.returncode, (p.stdout + p.stderr)[-800:]))


def device_present():
    # asked in a child, so the answer cannot depend on this process having asked before
    p = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import RNA; "
                        "print(RNA.cuda_devices())" % module_dir()],
                       capture_output=True, text=True, timeout=120)
    return p.stdout.strip().endswith(tuple("123456789"))


class cuda_forkTest(unittest.TestCase):
    """fork() after the device was used: the children fold on the host, they do not hang

    Found 2026-10-04 by the binder notebook's dry run. A CUDA context does not survive
    fork(), so every worker of a fork Pool died at its first fold ("initialization
    error") and Pool.map waited forever -- after an RNA.fold() in the parent, which uses
    the device by default. Fork is Linux's default start method up to Python 3.13, so
    this is what an existing ViennaRNA script does. The engine now lets only the process
    that first reached CUDA use it; a forked child of that process folds on the host.
    """

    def test_fork_after_the_device_was_used(self):
        """Workers forked after a device fold return the CPU's answers, on the host"""
        if not device_present():
            self.skipTest("no CUDA device")
        try:
            r = fork_probe(True)
        except subprocess.TimeoutExpired:
            self.fail("the fork Pool HUNG after the parent used the device")
        self.assertTrue(r["ok"], "a forked worker's answer differs from the CPU's: %s" % r)
        self.assertGreater(r["parent_batches"], 0, "the parent never used the device: %s" % r)
        self.assertEqual(r["child_devices"], [0] * 4, "a forked worker still sees a device: %s" % r)
        self.assertEqual(r["child_batches"], [r["parent_batches"]] * 4,
                         "a forked worker folded on the device: %s" % r)

    def test_fork_before_the_device_was_used(self):
        """Workers forked from a parent that never touched CUDA keep the device"""
        if not device_present():
            self.skipTest("no CUDA device")
        try:
            r = fork_probe(False)
        except subprocess.TimeoutExpired:
            self.fail("the fork Pool HUNG with a parent that never used the device")
        self.assertTrue(r["ok"], "a forked worker's answer differs from the CPU's: %s" % r)
        self.assertEqual(r["parent_batches"], 0)
        self.assertTrue(all(d > 0 for d in r["child_devices"]),
                        "a worker forked before any CUDA use lost the device: %s" % r)
        self.assertTrue(all(b > 0 for b in r["child_batches"]),
                        "a worker forked before any CUDA use did not fold on the device: %s" % r)


# A fresh interpreter with a tiny VRAM budget, so the batch has to be split -- and one
# record cannot fit at all -- on any GPU. RNA_GPU_VERBOSE=1 makes each device chunk print
# its init_gpu() line, which is the positive evidence that splitting happened.
CHUNK_PROBE = r'''
import json, random, sys
sys.path.insert(0, %r)
import RNA

rng = random.Random(2)
seqs = ["".join(rng.choice("ACGU") for _ in range(L)) for L in (400, 1200, 400, 400, 400, 1200, 400)]
got = [tuple(x) for x in RNA.fold(seqs)]
ref = [tuple(RNA.fold(s, cpu_only=True)) for s in seqs]
print("RESULT " + json.dumps({"equal": got == ref, "device_batches": RNA.cuda_batches()}))
'''


# Six records of 400 nt, for the chunk cell cap.
CAP_PROBE = r'''
import json, random, sys
sys.path.insert(0, %r)
import RNA

rng = random.Random(4)
seqs = ["".join(rng.choice("ACGU") for _ in range(400)) for _ in range(6)]
got = [tuple(x) for x in RNA.fold(seqs)]
ref = [tuple(RNA.fold(s, cpu_only=True)) for s in seqs]
print("RESULT " + json.dumps({"equal": got == ref}))
'''


# P3: a list folded with the backtrack pipeline on; the shape is the probe's argument.
PIPE_PROBE = r'''
import json, random, sys
sys.path.insert(0, %r)
import RNA

rng = random.Random(6)
seqs = ["".join(rng.choice("ACGU") for _ in range(L)) for L in %r]
got = [tuple(x) for x in RNA.fold(seqs)]
ref = [tuple(RNA.fold(s, cpu_only=True)) for s in seqs]
print("RESULT " + json.dumps({"equal": got == ref}))
'''


class cuda_pipelineTest(unittest.TestCase):
    """P3 (PORT_BUDGETS.md 9): RNA_BT_PIPELINE=1 inside one vrna_mfe_batch() call

    Each sub-chunk but the last is traced beside the next one's sweep, and joined before
    the call returns. Rule (b): never narrow a sub-chunk for the pipeline, so the device
    chunk count equals the pipeline-off run's."""

    def run_probe(self, lengths, extra):
        e = dict(os.environ, RNA_GPU_VERBOSE="1")
        for n in ("RNA_GPU_VRAM_BUDGET_MB", "RNA_VRAM_BUDGET", "RNA_BT_PIPELINE",
                  "RNA_BT_PIPELINE_NEGCTL", "RNA_GPU_CHUNK_CELLS"):
            e.pop(n, None)
        e.update(extra)
        p = subprocess.run([sys.executable, "-c", PIPE_PROBE % (module_dir(), lengths)],
                           capture_output=True, text=True, timeout=600, env=e)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else None
        inside = re.findall(r"inside vrna_mfe_batch\(\): (\d+) sub-chunk\(s\) traced beside the "
                            r"next, (\d+) joined early", p.stderr)
        return p, r, p.stderr.count("init_gpu("), inside

    def check(self, lengths, extra, want_overlap):
        if not device_present():
            self.skipTest("no CUDA device")
        p0, r0, c0, _ = self.run_probe(lengths, dict(extra, RNA_BT_PIPELINE="0"))
        self.assertTrue(r0 and r0["equal"], "pipeline off is wrong: %s" % (p0.stderr[-600:]))
        p1, r1, c1, inside = self.run_probe(lengths, dict(extra, RNA_BT_PIPELINE="1"))
        self.assertTrue(r1, "the pipeline probe crashed (rc %d): %s" % (p1.returncode, p1.stderr[-800:]))
        self.assertTrue(r1["equal"], "answers differ with the pipeline on")
        self.assertEqual(c1, c0, "rule (b): the pipeline changed the device chunk count")
        self.assertEqual(len(inside), 1, "no P3 report line: %s" % p1.stderr[-600:])
        if want_overlap:
            self.assertGreater(int(inside[0][0]), 0, "no sub-chunk was traced beside the next")
        return c1, inside[0]

    def test_cap_bound_list_overlaps(self):
        """the cell cap sets the width: sub-chunks overlap, no early joins, same chunk count"""
        c, (n, early) = self.check([400] * 8, {"RNA_GPU_CHUNK_CELLS": "170000"}, True)
        self.assertEqual(c, 4)
        self.assertEqual(early, "0", "a cap-bound list should never need an early join")

    def test_vram_bound_list_with_host_fold(self):
        """VRAM sets the width (rule b may join early) and the list ends in a host fold (the drain)"""
        self.check([400, 1200, 400, 400, 400, 1200, 400, 400, 400, 2800],
                   {"RNA_GPU_VRAM_BUDGET_MB": "1", "RNA_GPU_CHUNK_CELLS": "0"}, False)

    def test_negative_control_bites_on_internal_handoffs(self):
        """NEGCTL=2 frees a handed-off set: an internal hand-off must then fail or differ"""
        if not device_present():
            self.skipTest("no CUDA device")
        p, r, _, _ = self.run_probe([400] * 8, {"RNA_GPU_CHUNK_CELLS": "170000",
                                                "RNA_BT_PIPELINE": "1",
                                                "RNA_BT_PIPELINE_NEGCTL": "2"})
        self.assertTrue((r is None) or (not r["equal"]),
                        "the negative control did not bite: internal hand-offs never happened")


# P3 under model options: RNA.cuda_fold(list, md) with the pipeline on, several sub-chunks.
MODEL_PIPE_PROBE = r'''
import json, random, sys
sys.path.insert(0, %r)
import RNA

settings = json.loads(%r)
rng = random.Random(11)
def seq(L):
    s = []
    while len(s) < L:
        s += list("GGGAGGGAGGGAGGG") if settings.get("gquad") and rng.random() < 0.05 else [rng.choice("ACGU")]
    return "".join(s[:L])
seqs = [seq(400) for _ in range(8)]
md = RNA.md()
for k, v in settings.items():
    setattr(md, k, v)
got = [tuple(x) for x in RNA.cuda_fold(seqs, md)]
ref = []
for s in seqs:
    st, e = RNA.fold_compound(s, md).mfe(cpu_only=True)
    ref.append((st, round(e, 2)))
print("RESULT " + json.dumps({"equal": [(a, round(b, 2)) for a, b in got] == ref}))
'''


class cuda_pipelineOptionsTest(unittest.TestCase):
    """P3 with the model options a library caller sets: the same answers as the CPU, with
    sub-chunks really traced beside the next (no option silently turns the pipeline off)"""

    def probe(self, settings):
        if not device_present():
            self.skipTest("no CUDA device")
        e = dict(os.environ, RNA_BT_PIPELINE="1", RNA_GPU_CHUNK_CELLS="170000", RNA_GPU_VERBOSE="1")
        for n in ("RNA_GPU_VRAM_BUDGET_MB", "RNA_VRAM_BUDGET", "RNA_BT_PIPELINE_NEGCTL"):
            e.pop(n, None)
        p = subprocess.run([sys.executable, "-c", MODEL_PIPE_PROBE % (module_dir(), json.dumps(settings))],
                           capture_output=True, text=True, timeout=600, env=e)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, "%s: the probe crashed (rc %d): %s" % (settings, p.returncode, p.stderr[-800:]))
        self.assertTrue(json.loads(line[-1][7:])["equal"], "%s: answers differ from the CPU's" % settings)
        self.assertGreaterEqual(p.stderr.count("init_gpu("), 2, "%s: not folded on the device in sub-chunks" % settings)
        inside = re.findall(r"inside vrna_mfe_batch\(\): (\d+) sub-chunk", p.stderr)
        self.assertTrue(inside and int(inside[0]) > 0, "%s: no sub-chunk traced beside the next" % settings)

    def test_noLP(self):       self.probe({"noLP": 1})
    def test_circ(self):       self.probe({"circ": 1})
    def test_dangles0(self):   self.probe({"dangles": 0})
    def test_salt(self):       self.probe({"salt": 0.2})
    def test_gquad(self):      self.probe({"gquad": 1})
    def test_temperature(self): self.probe({"temperature": 30.0})


class cuda_chunkTest(unittest.TestCase):
    """A batch bigger than the device is split to fit it, not handed over whole

    Found 2026-10-04 by the binder notebook: RNA.fold() on 300 x 3000 nt killed the
    Python process on a 4 GB card ("CUDA error: out of memory", exit 2). RNAfold chunks
    by the VRAM budget before it calls the backend; the library's batch path handed the
    whole list to the device. cuda_batch_cb() now chunks with the same model, and folds
    on the host a record that does not fit even alone.
    """

    def test_batch_is_split_to_fit_the_device(self):
        """A 1 MB budget: several device chunks, an oversized record on the host, CPU answers"""
        if not device_present():
            self.skipTest("no CUDA device")
        e = dict(os.environ, RNA_GPU_VRAM_BUDGET_MB="1", RNA_GPU_VERBOSE="1")
        p = subprocess.run([sys.executable, "-c", CHUNK_PROBE % module_dir()],
                           capture_output=True, text=True, timeout=600, env=e)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, "the chunk probe crashed (rc %d): %s"
                        % (p.returncode, (p.stdout + p.stderr)[-800:]))
        r = json.loads(line[-1][7:])
        self.assertTrue(r["equal"], "a chunked batch's answers differ from the CPU's: %s" % r)
        self.assertEqual(r["device_batches"], 1, "the batch never reached the device: %s" % r)
        chunks = p.stderr.count("init_gpu(")
        self.assertGreaterEqual(chunks, 2, "the batch was not split (%d device chunk(s))" % chunks)

    def test_chunk_cell_cap_applies_to_library_lists(self):
        """PORT_BUDGETS.md 9.6 decision 1: RNA_GPU_CHUNK_CELLS splits a library list as in RNAfold"""
        if not device_present():
            self.skipTest("no CUDA device")
        counts = {}
        for cap in ("170000", "0"):
            # 400 nt = 80 200 cells: two records per chunk under 170 000, so 6 records -> 3
            e = dict(os.environ, RNA_GPU_CHUNK_CELLS=cap, RNA_GPU_VERBOSE="1")
            for n in ("RNA_GPU_VRAM_BUDGET_MB", "RNA_VRAM_BUDGET", "RNA_BT_PIPELINE"):
                e.pop(n, None)
            p = subprocess.run([sys.executable, "-c", CAP_PROBE % module_dir()],
                               capture_output=True, text=True, timeout=600, env=e)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            self.assertTrue(line, "the cap probe crashed (rc %d): %s"
                            % (p.returncode, (p.stdout + p.stderr)[-800:]))
            self.assertTrue(json.loads(line[-1][7:])["equal"], "answers differ at cap %s" % cap)
            self.assertIn("RNA_GPU_CHUNK_CELLS=%s" % cap, p.stderr, "the cap was not announced")
            counts[cap] = p.stderr.count("init_gpu(")
        self.assertEqual(counts, {"170000": 3, "0": 1}, "device chunks per cap: %s" % counts)


# Each host window is its own vrna_mfe_batch() call, so RNA.cuda_batches() counts the
# windows: the positive evidence that the list was windowed, not just answered.
WINDOW_PROBE = r'''
import json, random, sys
sys.path.insert(0, %r)
import RNA

rng = random.Random(5)
seqs = ["".join(rng.choice("ACGU") for _ in range(L)) for L in [400] * 10 + [1200, 400, 1200]]
got = [tuple(x) for x in RNA.fold(seqs)]
ref = [tuple(RNA.fold(s, cpu_only=True)) for s in seqs]
print("RESULT " + json.dumps({"equal": got == ref, "n": len(got), "device_batches": RNA.cuda_batches()}))
'''


class cuda_hostWindowTest(unittest.TestCase):
    """A list bigger than the HOST is built and folded in windows, not all at once

    Found 2026-10-05 by the binder notebook on a Colab T4 (12.7 GB RAM, 15 GB VRAM):
    RNA.fold() on 2014 x 2000 nt was SIGKILLed with nothing on stderr. The device side
    was already chunked; the wrapper built every fold compound (6.2 MB each at 2000 nt)
    before folding any, ~12.5 GB. It now builds, folds and frees in windows that fit in
    half of MemAvailable; RNA_HOST_AVAIL_MB pretends a size, as for RNAfold.
    """

    def probe(self, avail_mb):
        e = dict(os.environ)
        e.pop("RNA_HOST_AVAIL_MB", None)
        if avail_mb is not None:
            e["RNA_HOST_AVAIL_MB"] = str(avail_mb)
        p = subprocess.run([sys.executable, "-c", WINDOW_PROBE % module_dir()],
                           capture_output=True, text=True, timeout=600, env=e)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, "the window probe crashed (rc %d): %s"
                        % (p.returncode, (p.stdout + p.stderr)[-800:]))
        return json.loads(line[-1][7:])

    def test_list_is_windowed_to_fit_the_host(self):
        """4 MB pretended: several windows, each 1200 nt record alone, CPU answers"""
        if not device_present():
            self.skipTest("no CUDA device")
        r = self.probe(4)
        self.assertTrue(r["equal"] and r["n"] == 13, "windowed answers differ from the CPU's: %s" % r)
        # 2 MB budget: ~6 records of 400 nt per window, and a 1200 nt record (2.3 MB) alone
        self.assertGreaterEqual(r["device_batches"], 4, "the list was not windowed: %s" % r)

    def test_roomy_host_is_one_window(self):
        """The negative control: with memory to spare the list is still one batch"""
        if not device_present():
            self.skipTest("no CUDA device")
        r = self.probe(None)
        self.assertTrue(r["equal"], "answers differ from the CPU's: %s" % r)
        self.assertEqual(r["device_batches"], 1, "a list that fits was split: %s" % r)


def meminfo_mb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0
    return None


BUDGET_REPORT_PROBE = r"""
import sys
sys.path.insert(0, %r)
import RNA
RNA.fold("GGGGAAAACCCC")
"""

# B4: enough records that an uncapped backtrack pool would take every core
THREAD_PROBE = r'''
import json, random, sys
sys.path.insert(0, %r)
import RNA

rng = random.Random(9)
seqs = ["".join(rng.choice("ACGU") for _ in range(300)) for _ in range(16)]
got = [tuple(x) for x in RNA.fold(seqs)]
ref = [tuple(RNA.fold(s, cpu_only=True)) for s in seqs]
print("RESULT " + json.dumps({"equal": got == ref}))
'''


class cuda_budgetTest(unittest.TestCase):
    """Resource budgets (PORT_BUDGETS.md B1): parsing, precedence and the report

    Resolution only -- what each consumer does with its budget is B2-B4's bar."""

    NAMES = ("RNA_RAM_BUDGET", "RNA_VRAM_BUDGET", "RNA_THREAD_BUDGET")

    def setUp(self):
        self.saved = {n: os.environ.pop(n, None) for n in self.NAMES}
        RNA.set_budget(reset=True)

    def tearDown(self):
        RNA.set_budget(reset=True)
        for n, v in self.saved.items():
            if v is not None:
                os.environ[n] = v

    def test_python_defaults_to_half(self):
        hw = os.cpu_count()
        b = RNA.budget()
        self.assertEqual(b["threads"], max(1, hw // 2))
        total = meminfo_mb("MemTotal")
        self.assertAlmostEqual(b["ram"], total / 2, delta=2)

    def test_words_and_numbers(self):
        hw, total = os.cpu_count(), meminfo_mb("MemTotal")
        RNA.set_budget(ram="quarter", threads=3)
        self.assertAlmostEqual(RNA.budget()["ram"], total / 4, delta=2)
        self.assertEqual(RNA.budget()["threads"], 3)
        RNA.set_budget(ram=1024, threads="full")
        self.assertEqual(RNA.budget()["ram"], 1024.0)
        self.assertEqual(RNA.budget()["threads"], hw)
        RNA.set_budget(ram="full", threads="ThreeQuarter")
        self.assertIsNone(RNA.budget()["ram"], "full must add no ceiling")
        self.assertEqual(RNA.budget()["threads"], max(1, int(0.75 * hw)))

    def test_junk_is_refused_and_changes_nothing(self):
        RNA.set_budget(ram=2048)
        before = RNA.budget()
        for bad in ("most", "-1", "0", "12MB", ""):
            with self.assertRaises(ValueError, msg="accepted ram=%r" % bad):
                RNA.set_budget(threads=2, ram=bad)
            self.assertEqual(RNA.budget(), before, "a refused call left a change behind")

    def test_precedence_set_over_environment_over_default(self):
        with env("RNA_THREAD_BUDGET", "1"):
            self.assertEqual(RNA.budget()["threads"], 1, "the environment beats the default")
            RNA.set_budget(threads=2)
            self.assertEqual(RNA.budget()["threads"], 2, "set_budget() beats the environment")
            RNA.set_budget(reset=True)
            self.assertEqual(RNA.budget()["threads"], 1, "reset returns to the environment")

    def test_per_call_budget_is_restored(self):
        RNA.set_budget(ram=3000)
        s, e = RNA.fold("GGGGAAAACCCC", ram=512, threads=1)
        self.assertEqual([s, e], list(RNA.fold("GGGGAAAACCCC", cpu_only=True)))
        self.assertEqual(RNA.budget()["ram"], 3000.0)
        RNA.set_budget(reset=True)
        RNA.fold(["GGGGAAAACCCC"], vram="quarter")
        self.assertEqual(RNA.budget(), cuda_budgetTest.default_budget(), "a per-call budget leaked")

    @staticmethod
    def default_budget():
        RNA.set_budget(reset=True)
        return RNA.budget()

    def report(self, extra):
        e = dict(os.environ)
        for n in self.NAMES:
            e.pop(n, None)
        e.update(extra)
        p = subprocess.run([sys.executable, "-c", BUDGET_REPORT_PROBE % module_dir()],
                           capture_output=True, text=True, timeout=300, env=e)
        self.assertEqual(p.returncode, 0, p.stderr[-800:])
        self.last_stderr = p.stderr
        return [l for l in p.stderr.splitlines() if "budget: RAM" in l]

    def test_report_says_what_is_in_force(self):
        lines = self.report({"RNA_GPU_VERBOSE": "1"})
        self.assertEqual(len(lines), 1, "want one budget line: %s" % lines)
        self.assertIn("threads %d (half of %d)" % (max(1, os.cpu_count() // 2), os.cpu_count()), lines[0])
        if device_present():
            # B2: the VRAM budget's positive evidence, the peak this process held
            peak = [l for l in self.last_stderr.splitlines() if "VRAM budget: peak held" in l]
            self.assertEqual(len(peak), 1, "want one VRAM peak line: %s" % self.last_stderr[-600:])
            self.assertIn("within it", peak[0])

    def test_report_is_silent_at_full_and_when_quiet(self):
        full = {n: "full" for n in self.NAMES}
        full["RNA_GPU_VERBOSE"] = "1"
        self.assertEqual(self.report(full), [], "full is today's behaviour and says nothing")
        self.assertEqual(self.report({}), [], "the binding is quiet unless RNA_GPU_VERBOSE")

    def test_thread_budget_binds_backtrack_workers(self):
        """B4: threads=2 caps the backtrack pool and the peak drawn; answers unchanged"""
        if not device_present():
            self.skipTest("no CUDA device (the backtrack pool is the device path's)")
        if (os.cpu_count() or 1) < 3:
            self.skipTest("a budget of 2 binds nothing on %d threads" % os.cpu_count())
        e = dict(os.environ, RNA_THREAD_BUDGET="2", RNA_GPU_VERBOSE="1")
        e.pop("RNA_BACKTRACK_THREADS", None)
        p = subprocess.run([sys.executable, "-c", THREAD_PROBE % module_dir()],
                           capture_output=True, text=True, timeout=600, env=e)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, "the thread probe crashed (rc %d): %s"
                        % (p.returncode, (p.stdout + p.stderr)[-800:]))
        self.assertTrue(json.loads(line[-1][7:])["equal"], "answers differ under a thread budget")
        self.assertIn("thread budget: backtrack workers", p.stderr, "the cap never acted")
        peak = re.findall(r"thread budget: peak (\d+) concurrent workers of a budget of (\d+)",
                          p.stderr)
        self.assertEqual(len(peak), 1, "want one peak line: %s" % p.stderr[-600:])
        self.assertEqual(peak[0][1], "2")
        self.assertLessEqual(int(peak[0][0]), 2, "more workers ran than the budget")


if __name__ == '__main__':
    unittest.main(testRunner=taprunner.TAPTestRunner())

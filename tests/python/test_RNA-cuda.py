import os
import random
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

        ON THIS BRANCH int16 is opt-in, so with the default environment this
        is a plain model switch between batches; the transition the docstring
        describes is reached by test_int16_declined_between_batches below.
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

    def test_int16_declined_between_batches(self):
        """With RNA_FML_INT16=1, a batch after the int16 vet declines still folds

        The same defect as above, reached the way this branch can reach it.
        int16 is opt-in here, and rnafold_fml_int16_vet_params() shuts it for the
        rest of the process the first time a batch arrives with a parameter table
        whose stacks break the 16-bit bound -- which a caller can load between
        batches. teardown_gpu() chose what to free by asking rnafold_fml_int16(),
        so after the decline it freed d_fml_j (never allocated) and left the int16
        buffers set: measured on the laptop, batch 2 died with "an illegal memory
        access was encountered" (code 700) and took the interpreter with it.

        Runs in a CHILD process because RNA_FML_INT16 is read once, at the first
        fold. The DECLINED line is the control: without it the vet never fired and
        nothing here was tested. The table is passed as an explicit RNA.md(),
        because a fold compound made with no model keeps the cached default
        parameters after params_load().
        """
        if RNA.cuda_devices() == 0:
            self.skipTest("needs a device: the defect is in device teardown")
        child = r'''
import os, random, sys, RNA
d = sys.argv[1]
rng = random.Random(21)
seqs = ["".join(rng.choice("ACGU") for _ in range(n)) for n in (180, 260, 340, 420)]
bad = dev = 0
def check(md):
    global bad, dev
    b0 = RNA.cuda_batches()
    got = RNA.cuda_fold(seqs, md) if md is not None else RNA.cuda_fold(seqs)
    dev += RNA.cuda_batches() - b0
    for s, (st, e) in zip(seqs, got):
        fc = RNA.fold_compound(s, md) if md is not None else RNA.fold_compound(s)
        ref = fc.mfe(cpu_only=True)
        bad += (st != ref[0]) or abs(e - ref[1]) > 0.005
check(None)
RNA.params_save(os.path.join(d, "default.par"))
out, sect = [], None
for line in open(os.path.join(d, "default.par")):
    if line.startswith("#"):
        sect = line.strip()
    elif sect == "# stack":
        t = line.split()
        if t and all(x.lstrip("-").isdigit() or x in ("INF", "DEF") for x in t):
            line = "".join("%6s" % (x if x in ("INF", "DEF") or int(x) >= 0
                                    else str(int(x) * 10)) for x in t) + "\n"
    out.append(line)
with open(os.path.join(d, "x10.par"), "w") as f:
    f.writelines(out)
assert RNA.params_load(os.path.join(d, "x10.par")) == 1
check(RNA.md())
check(RNA.md())
print("device_batches %d mismatching %d" % (dev, bad))
'''
        import subprocess
        env = dict(os.environ, RNA_FML_INT16="1", RNA_GPU_VERBOSE="1")
        env.pop("RNA_GPU", None)
        here = os.path.dirname(os.path.dirname(os.path.abspath(RNA.__file__)))
        env["PYTHONPATH"] = os.pathsep.join([here] + [p for p in [os.environ.get("PYTHONPATH")] if p])
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-c", child, d], env=env,
                               capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("RNA_FML_INT16 DECLINED", r.stderr, "the vet never fired")
        self.assertEqual(r.stdout.strip().splitlines()[-1], "device_batches 3 mismatching 0")

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


if __name__ == '__main__':
    unittest.main(testRunner=taprunner.TAPTestRunner())

import random
import unittest

if __name__ == '__main__':
    from py_include import taprunner
    import RNApath
    RNApath.addSwigInterfacePath()

import RNA


def rand_seq(n, rng):
    return "".join(rng.choice("ACGU") for _ in range(n))


def cpu(seq, md=None):
    fc = RNA.fold_compound(seq, md) if md is not None else RNA.fold_compound(seq)
    return fc.mfe()


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

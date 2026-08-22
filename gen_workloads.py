import random, math, sys, os
OUT = os.path.expanduser("~/rnatest")

def emit(name, lengths, seed):
    rng = random.Random(seed)
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for i, L in enumerate(lengths):
            f.write(">%s_%d_L%d\n" % (name.split(".")[0], i, L))
            f.write("".join(rng.choice("ACGU") for _ in range(L)) + "\n")
    n = len(lengths)
    work = sum(L**3 for L in lengths)
    vram = max(4.3*L*L for L in lengths)/1e6
    print("%-12s n=%-5d min=%-5d max=%-5d mean=%-7.1f  max/mean=%.2f  sum_n3=%.3g  biggest_rec=%.0fMB"
          % (name, n, min(lengths), max(lengths), sum(lengths)/n,
             max(lengths)/(sum(lengths)/n), work, vram))

# C400-local: the Colab benchmark shape (uniform, long), scaled to fit 4GB in a few chunks
emit("C400.fa", [5601]*40, 1)

# D: long, all distinct
emit("D_long.fa", list(range(3000, 6000, 100)), 2)

# E: genome-like -- 2000 records, long-tailed, essentially all distinct
rng = random.Random(3)
E = []
for _ in range(2000):
    L = int(round(math.exp(rng.gauss(math.log(700), 0.55))))
    E.append(max(200, min(3000, L)))
emit("E_genome.fa", E, 3)

# F: extreme spread in one input
rng = random.Random(4)
F = []
for L in [50, 80, 120, 200, 320, 500, 800, 1300, 2000, 3200, 5000, 8000]:
    F += [L + rng.randint(-3, 3)] * (6 if L < 3000 else 3)
rng.shuffle(F)
emit("F_extreme.fa", F, 4)

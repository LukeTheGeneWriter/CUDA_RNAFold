#!/usr/bin/env python3
"""Generate the v2 ViennaRNA 2.7.2 GPU benchmark notebook.

v1 answered "how much faster than upstream?" and got 7.98x at 2000 nt, 5.63x at
5601 nt. It also exposed its own biggest limitation: `sweeps: 1` in both CUDA
arms, meaning the whole workload fit in ONE chunk on the L4's 24 GB. So none of
the machinery that makes this a batch accelerator rather than a single-batch one
was exercised -- no VRAM budgeting, no per-chunk teardown/re-init, no slot flow
-- and both workloads were uniform-length, so the mixed-length join-mask work
was untouched too.

v2 adds the two arms that cover it:

  MIXED       a shuffled mixed-length workload, which is what a real FASTA
              looks like and what the join mask exists for.
  MULTI-CHUNK the SAME workload as a single-chunk run, but with the VRAM budget
              forced down so it must span several chunks. Because the work is
              identical, the difference IS the cost of chunking -- and the
              output must stay byte-identical, which makes it a correctness
              result as well as a timing one.

The multi-chunk arms are GPU-only: their reference is the single-chunk GPU run
of the same workload, so they cost seconds rather than another CPU pass. The
warm-up is GPU-only for the same reason -- a CPU arm gains nothing from it, and
at 5601 nt a discarded warm-up costs ten minutes.
"""
import json

C = []


def _lines(s):
    # nbformat wants each entry to KEEP its trailing newline.
    return s.splitlines(keepends=True)


def md(s):
    C.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": _lines(s)})


# --------------------------------------------------------------------------
md(r"""# CUDA_RNAFold on ViennaRNA 2.7.2 — benchmark v2

**What v1 established, and what it could not.** v1 gave this project its first
timing against upstream 2.7.2: **7.98x** at 120×2000 nt, **5.63x** at 60×5601 nt,
with the port costing upstream's own CPU path nothing measurable.

It also showed its own limit. Both CUDA arms reported `sweeps: 1` — the whole
workload fit in a **single chunk** on the L4's 24 GB. Everything that makes this
a *batch* accelerator was therefore untested: VRAM budgeting, per-chunk teardown
and re-init, slot flow, `MIN_GPU_BATCH`. And both workloads were uniform-length,
so the mixed-length join mask never came into play.

## What v2 adds

| arm | what it covers |
|---|---|
| **MIXED** | a shuffled mixed-length workload — what a real FASTA looks like |
| **MULTI-CHUNK** | the *same* work forced across several chunks, so the delta is the cost of chunking itself |

The multi-chunk arm is the interesting one, because it is two results at once:

- **timing** — same records, same total work, more chunks. The difference is
  per-chunk overhead (`init_gpu`/`teardown_gpu` ×3, re-upload, re-budget).
- **correctness** — the output must stay **byte-identical** to the single-chunk
  run. Chunk boundaries must not move an answer. This was confirmed locally
  before the run was designed, on both a uniform workload (40 × 600 nt, 1 → 2
  chunks) and a shuffled mixed one (60 records, 208–1075 nt, 1 → 3 chunks):
  identical output at every chunk count. This arm asserts it at benchmark scale
  and on the L4.

## What still would make the run worthless

Same three gates as v1, plus one:

- different answers between arms → the timings aren't comparable;
- a CUDA arm with **zero** sweeps → it folded on the CPU and the number is a lie;
- **a "multi-chunk" arm with only ONE sweep** → the budget didn't bite and the
  arm tested nothing. This is the new one, and it is the same shape as the trap
  where `RNA_GPU_CHUNK=8` against `MIN_GPU_BATCH=10` produced a green GPU bar
  that had never touched the GPU.
- clock throttling.""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""import subprocess, os, sys, json, time, re, hashlib, random, statistics

def sh(cmd, check=True, quiet=False):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        if p.stdout.strip(): print(p.stdout.strip()[:4000])
        if p.returncode and p.stderr.strip(): print(p.stderr.strip()[:4000])
    if check and p.returncode:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p

print(sh("nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm "
         "--format=csv,noheader", quiet=True).stdout.strip())
print(sh("nvcc --version | tail -2", quiet=True).stdout.strip())
print("cores:", os.cpu_count())""")

code(r"""# Build-from-git needs more than the release tarball: the generated gengetopt
# parsers, man pages and doxygen XML are not in upstream's git, and configure
# hard-errors until dlib/libsvm are unpacked. libtool and `time` were added
# after the v1 run needed them.
sh("apt-get -qq update && apt-get -qq install -y "
   "gengetopt help2man texinfo doxygen check libsubunit-dev bison flex "
   "libtool libtool-bin time > /dev/null 2>&1", check=False)
for t in ["gengetopt", "help2man", "makeinfo", "doxygen", "autoreconf", "libtool", "time"]:
    p = sh(f"command -v {t}", check=False, quiet=True)
    print(f"  {t:12s} {'OK' if p.returncode == 0 else 'MISSING'}")""")

# --------------------------------------------------------------------------
md("## 2. Fetch both trees")

code(r"""REPO = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"

sh("rm -rf /content/ours /content/upstream")
sh(f"git clone -q --branch {BRANCH} {REPO} /content/ours")
print("ours    :", sh("git -C /content/ours log --oneline -1", quiet=True).stdout.strip())

# arm A comes from the same clone's v2.7.2 tag -- same objects, no second
# download, and provably the tag port27 is based on.
sh("git clone -q /content/ours /content/upstream")
sh("git -C /content/upstream checkout -q v2.7.2")
print("upstream:", sh("git -C /content/upstream log --oneline -1", quiet=True).stdout.strip())

# The shape of the diff, split by status: "71 files changed" sounds enormous,
# but the added files are ours to ignore. What matters is how few of UPSTREAM's
# files are touched, and how little is removed from them.
print()
st = sh("git -C /content/ours diff --name-status v2.7.2 port27", quiet=True).stdout
mod = [l.split("\t")[1] for l in st.strip().splitlines() if l.startswith("M")]
add = [l for l in st.strip().splitlines() if l.startswith("A")]
print(f"upstream files MODIFIED: {len(mod)}     new files added: {len(add)}")
for m in mod:
    print("   ", m)
print()
print("deletions -- the only upstream lines this port removes:")
print(sh("git -C /content/ours diff --numstat v2.7.2 port27 "
         "| awk '$2>0{print \"   -\"$2\"  \"$3}'", quiet=True).stdout.rstrip())""")

# --------------------------------------------------------------------------
md("""## 3. Build the three arms

Identical `CFLAGS` across all three — same `configure` invocation, differing
only in `--enable-cuda`.""")

code(r"""COMMON = ("--without-python --without-perl --without-swig --without-doc "
          "--without-rnaxplorer --without-forester --without-kinfold "
          "--without-rnalocmin")

def build(path, tag, extra=""):
    t0 = time.time()
    sh(f"cd {path} && tar -xjf src/dlib-*.tar.bz2 -C src/ 2>/dev/null || true", check=False, quiet=True)
    sh(f"cd {path} && tar -xzf src/libsvm-*.tar.gz -C src/ 2>/dev/null || true", check=False, quiet=True)
    sh(f"chmod +x {path}/doc/man2rst.py 2>/dev/null || true", check=False, quiet=True)

    p = sh(f"cd {path} && ./autogen.sh > ag.log 2>&1", check=False, quiet=True)
    if p.returncode:
        print(sh(f"tail -30 {path}/ag.log", check=False, quiet=True).stdout)
        raise RuntimeError(f"autogen failed for {tag}")
    p = sh(f"cd {path} && ./configure {COMMON} {extra} > cfg.log 2>&1", check=False, quiet=True)
    if p.returncode:
        print(sh(f"grep -m3 'configure: error' {path}/cfg.log", check=False, quiet=True).stdout)
        raise RuntimeError(f"configure failed for {tag}")
    p = sh(f"cd {path} && make -j$(nproc) > mk.log 2>&1", check=False, quiet=True)
    if p.returncode:
        print(sh(f"grep -B 3 -A 3 -inE '\\berror\\b' {path}/mk.log | head -30",
                 check=False, quiet=True).stdout)
        raise RuntimeError(f"make failed for {tag}")
    bin_ = f"{path}/src/bin/RNAfold"
    assert os.path.exists(bin_), bin_
    print(f"  {tag:24s} built in {time.time()-t0:5.0f}s")
    return bin_

BIN = {}
BIN["A upstream 2.7.2"]   = build("/content/upstream", "A upstream 2.7.2")
BIN["B port27 (no cuda)"] = build("/content/ours", "B port27 (no cuda)")

sh("cp /content/ours/src/bin/RNAfold /content/rnafold_nocuda", quiet=True)
BIN["B port27 (no cuda)"] = "/content/rnafold_nocuda"
sh("cd /content/ours && make distclean > /dev/null 2>&1", check=False, quiet=True)
BIN["C port27 (cuda)"]    = build("/content/ours", "C port27 (cuda)", "--enable-cuda")

print()
print("nvcc invocations in arm C:",
      sh("grep -c NVCC /content/ours/mk.log", check=False, quiet=True).stdout.strip())""")

# --------------------------------------------------------------------------
md(r"""## 4. Workloads

**W1** and **W2** are v1's, kept identical so the two runs are comparable.
**W3** is new: a shuffled mixed-length workload, which is what a real input file
looks like and the only one that exercises the join mask.

W3 is deliberately *shuffled* rather than sorted. Sorted input is the easy case
for chunk packing — records of similar size group naturally. Shuffled is the
honest one.""")

code(r"""def make_uniform(path, n, length, seed):
    random.seed(seed)
    with open(path, "w") as f:
        for i in range(n):
            s = "".join(random.choice("ACGU") for _ in range(length))
            f.write(f">rec_{i}\n{s}\n")
    return path

def make_mixed(path, n, lo, hi, seed):
    # Log-uniform lengths, shuffled. Log-uniform rather than uniform because
    # real sequence collections skew toward the short end, and because a linear
    # draw over 300-2500 is dominated by its long tail -- which would quietly
    # turn this into another uniform-long workload wearing a mixed label.
    import math
    random.seed(seed)
    lens = [int(math.exp(random.uniform(math.log(lo), math.log(hi)))) for _ in range(n)]
    random.shuffle(lens)
    with open(path, "w") as f:
        for i, L in enumerate(lens):
            s = "".join(random.choice("ACGU") for _ in range(L))
            f.write(f">rec_{i}_{L}\n{s}\n")
    return path, lens

os.makedirs("/content/wl", exist_ok=True)
W = {}
W["W1 120x2000"]  = make_uniform("/content/wl/w1.fa", 120, 2000, 20260906)
W["W2  60x5601"]  = make_uniform("/content/wl/w2.fa",  60, 5601, 20260906)
p, lens = make_mixed("/content/wl/w3.fa", 150, 300, 2500, 20260906)
W["W3 150 mixed"] = p

print(f"  W3: {len(lens)} records, {min(lens)}-{max(lens)} nt, "
      f"median {sorted(lens)[len(lens)//2]}, total {sum(lens)/1e6:.2f} Mnt")
for k, v in W.items():
    print(f"  {k:16s} {os.path.getsize(v)/1e6:6.1f} MB")""")

# --------------------------------------------------------------------------
md(r"""## 5. Run

Configurations, not just arms. Each CUDA workload runs twice: once letting the
budget take the whole card (one chunk), once with `RNA_GPU_VRAM_BUDGET_MB`
forced down so it must span several.

`RNA_GPU_VRAM_BUDGET_MB` can only *lower* the budget, never raise it, so this
cannot accidentally over-commit the card.

Warm-up is GPU-only. A CPU arm gains nothing from it, and at 5601 nt a discarded
CPU warm-up costs ten minutes.""")

code(r"""REPS = 3
TARGET_CHUNKS = 4

# A single budget cannot force several chunks across workloads whose records
# differ 20-fold in area, so it is computed per workload.
#
# Device memory per record is dominated by the triangular matrices, so it goes
# as n^2. Calibrating against the one measured point this project has -- 125.6 MB
# at 5601 nt -- gives ~8 bytes per cell over n^2/2 cells, i.e. ~4*n^2 bytes:
#
#     5601 nt -> 4 * 5601^2  = 125.5 MB   (measured 125.6)
#     2000 nt -> 4 * 2000^2  =  16.0 MB
#
# This only has to be close enough to land in the right order of magnitude:
# the validity gate below FAILS the run if the N-chunk arm produced <= 1 chunk,
# and run_multichunk() halves the budget and retries a few times first.
BYTES_PER_RECORD = lambda L: 4.0 * L * L

def budget_for(fa, target=TARGET_CHUNKS):
    lens = []
    with open(fa) as f:
        cur = 0
        for line in f:
            if line.startswith(">"):
                if cur: lens.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
        if cur: lens.append(cur)
    total = sum(BYTES_PER_RECORD(L) for L in lens)
    mb = int(total / target / 1e6)
    # The floor is 16, not 64: verified locally that a 40 x 600 nt workload
    # (~58 MB by this estimate) needs a 32 MB cap before it splits at all, so a
    # 64 MB floor would hand the calibration loop a budget already too large and
    # burn rounds halving back down.
    return max(16, mb), len(lens)

def run_once(binary, fa, gpu, budget_mb=None):
    env = dict(os.environ)
    cmd = [binary, "--noPS", "-i", fa]
    if gpu:
        env["RNA_GPU_CHUNK"] = "0"          # budget decides the chunk size
        if budget_mb:
            env["RNA_GPU_VRAM_BUDGET_MB"] = str(budget_mb)
        else:
            env.pop("RNA_GPU_VRAM_BUDGET_MB", None)
    else:
        env.pop("RNA_GPU_CHUNK", None)
        env.pop("RNA_GPU_VRAM_BUDGET_MB", None)
        cmd.insert(1, f"-j{os.cpu_count()}")
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True,
                       text=True, env=env)
    wall = time.time() - t0
    rss = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m: rss = int(m.group(1)) / 1e6
    return dict(wall=wall, rss=rss, rc=p.returncode,
                sha=hashlib.sha256(p.stdout.encode()).hexdigest()[:16],
                sweeps=len(re.findall(r"sweep shape:", p.stderr)),
                stderr_tail=p.stderr[-600:] if p.returncode else "")

# (config label, arm key, gpu?)
CONFIGS = [
    ("A upstream",      "A upstream 2.7.2",   False),
    ("B port, no cuda", "B port27 (no cuda)", False),
    ("C GPU 1-chunk",   "C port27 (cuda)",    True),
    ("D GPU N-chunk",   "C port27 (cuda)",    True),
]

BUDGET = {}
results = {}

# INTERLEAVED, and this is a correction to v2's first run rather than a
# refinement. That run executed A's three reps, then B's, then C's, then D's --
# about twenty minutes per workload, arm by arm. Any drift over that window
# (thermal, noisy neighbour, a migrated VM) lands unevenly on the arms and
# appears as a difference BETWEEN them.
#
# It appears to have done exactly that: A/B moved from 0.994/0.996 in v1 to
# 0.976/0.980 in v2, consistently, against a within-run spread of 0.27-0.33%.
# No change in that diff can reach arm B -- every CUDA source is inside the
# `if VRNA_AM_SWITCH_CUDA` block and does not compile without --enable-cuda --
# so the likeliest cause is that `spread` measures repeatability within ONE run
# on ONE machine and systematically understates variance across a session.
#
# Running one rep of every arm, then the next round, puts all four arms inside
# the same slice of whatever the machine is doing. A/B is the number upstream
# will care about most, so it is the one that must not be an artifact of
# ordering.
for wname, fa in W.items():
    print(f"\n=== {wname}")

    # Calibration and warm-up first, outside the timed rounds.
    budget_for_cfg = {}
    for label, armkey, gpu in CONFIGS:
        b = BIN[armkey]
        if label == "D GPU N-chunk":
            # The estimate is only a starting point; what settles it is the
            # observed chunk count, because a "multi-chunk" arm that produced
            # one chunk has measured nothing at all.
            budget, nrec = budget_for(fa)
            for _ in range(5):
                probe = run_once(b, fa, True, budget)
                if probe["sweeps"] >= 2:
                    break
                print(f"    calibrating: {budget} MB gave {probe['sweeps']} chunk(s), halving")
                budget //= 2
            BUDGET[wname] = budget
            budget_for_cfg[label] = budget
            print(f"    budget {budget} MB over {nrec} records -> {probe['sweeps']} chunks")
        else:
            budget_for_cfg[label] = None
            if gpu:
                run_once(b, fa, gpu, None)      # warm-up, GPU only, discarded

    # Timed rounds, one rep of every arm per round.
    runs = {label: [] for label, _, _ in CONFIGS}
    for rep in range(REPS):
        for label, armkey, gpu in CONFIGS:
            runs[label].append(run_once(BIN[armkey], fa, gpu, budget_for_cfg[label]))
        print(f"    round {rep+1}/{REPS}: " +
              "  ".join(f"{lab.split()[0]}={runs[lab][-1]['wall']:.1f}s"
                        for lab, _, _ in CONFIGS))

    for label, armkey, gpu in CONFIGS:
        rs = runs[label]
        best  = min(r["wall"] for r in rs)
        worst = max(r["wall"] for r in rs)
        r0 = rs[0]
        results[(wname, label)] = dict(
            best=best, spread=100*(worst-best)/best,
            rss=max(r["rss"] for r in rs), sha=r0["sha"],
            sweeps=r0["sweeps"], rc=r0["rc"], tail=r0["stderr_tail"],
            budget_mb=budget_for_cfg[label],
            walls=[round(r["wall"], 3) for r in rs])
        print(f"  {label:16s} {best:8.2f}s  spread {results[(wname,label)]['spread']:4.1f}%"
              f"  rss {results[(wname,label)]['rss']:5.2f}G"
              f"  chunks {r0['sweeps']:3d}  sha {r0['sha']}")""")

# --------------------------------------------------------------------------
md(r"""## 6. Validity gates

Read these before any speedup. Each one can invalidate the numbers, and each has
a real precedent in this project.""")

code(r"""ok = True
for wname in W:
    shas = {lab: results[(wname, lab)]["sha"] for lab, _, _ in CONFIGS}
    if len(set(shas.values())) != 1:
        ok = False
        print(f"  {wname}: ARMS DISAGREE -- timings are NOT comparable")
        for a, s in shas.items(): print(f"      {a:18s} {s}")
    else:
        print(f"  {wname}: all four configs identical  ({list(shas.values())[0]})")
        print(f"      ^ includes 1-chunk vs N-chunk: chunk boundaries do not move the answer")

    c1 = results[(wname, "C GPU 1-chunk")]
    cn = results[(wname, "D GPU N-chunk")]

    if c1["sweeps"] == 0:
        ok = False
        print(f"  {wname}: the 1-chunk GPU arm ran NO sweeps -- it folded on the CPU")
    if cn["sweeps"] <= 1:
        ok = False
        print(f"  {wname}: the N-chunk arm produced {cn['sweeps']} chunk(s). The budget"
              f" did not bite, so this arm tested nothing. Lower BUDGET_MB.")
    for lab in ("A upstream", "B port, no cuda"):
        if results[(wname, lab)]["sweeps"] != 0:
            ok = False
            print(f"  {wname}: {lab} logged GPU sweeps -- it is not a CPU arm")
    for lab, _, _ in CONFIGS:
        r = results[(wname, lab)]
        if r["rc"] != 0:
            ok = False
            print(f"  {wname}: {lab} exited {r['rc']}\n{r['tail']}")

print()
print(sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu "
         "--format=csv,noheader", quiet=True).stdout.strip(), " <- clocks after the run")
print()
print("VALID" if ok else "*** INVALID -- do not quote these numbers ***")""")

# --------------------------------------------------------------------------
md("## 7. The answer")

code(r"""print(f"{'workload':16s} {'A upstream':>11s} {'B port,off':>11s} {'C 1-chunk':>11s}"
      f" {'D N-chunk':>11s} {'A/C':>7s} {'A/B':>7s}")
for wname in W:
    a = results[(wname, "A upstream")]["best"]
    b = results[(wname, "B port, no cuda")]["best"]
    c = results[(wname, "C GPU 1-chunk")]["best"]
    d = results[(wname, "D GPU N-chunk")]["best"]
    print(f"{wname:16s} {a:10.2f}s {b:10.2f}s {c:10.2f}s {d:10.2f}s "
          f"{a/c:6.2f}x {a/b:6.2f}x")

print()
print("--- what chunking costs (same work, more chunks) ---")
print(f"{'workload':16s} {'budget':>8s} {'chunks':>7s} {'1-chunk':>10s} {'N-chunk':>10s}"
      f" {'penalty':>9s} {'per chunk':>11s}")
for wname in W:
    c = results[(wname, "C GPU 1-chunk")]
    d = results[(wname, "D GPU N-chunk")]
    extra = d["best"] - c["best"]
    per = extra / max(1, d["sweeps"] - c["sweeps"])
    print(f"{wname:16s} {d['budget_mb'] or 0:6d}MB {d['sweeps']:6d}  "
          f"{c['best']:9.2f}s {d['best']:9.2f}s "
          f"{100*extra/c['best']:8.1f}% {per:10.2f}s")

print()
print("--- A/B per round, so an ordering artifact would be visible rather than inferred ---")
for wname in W:
    aw = results[(wname, "A upstream")]["walls"]
    bw = results[(wname, "B port, no cuda")]["walls"]
    ratios = "  ".join(f"{a/b:.3f}" for a, b in zip(aw, bw))
    print(f"{wname:16s} A/B by round: {ratios}")
print("A drift that moves both arms together leaves these flat; one that hits")
print("them unevenly makes them trend. v2's first run could not show this.")

print()
print("A/C  the headline: CUDA build vs upstream ViennaRNA 2.7.2")
print("A/B  the port's cost with the GPU OFF. Below 1.00 means our CPU path is")
print("     slower than upstream's, which needs explaining before this goes to")
print("     the maintainers.")
print("The chunk penalty is the price of not fitting on the card at once. It is")
print("the number that decides whether int16 (half the fml_j stream, so roughly")
print("twice the records per chunk) pays for itself on real inputs.")

print()
print(f"{'workload':16s} {'peak RSS A':>12s} {'B':>8s} {'C':>8s} {'D':>8s}")
for wname in W:
    print(f"{wname:16s} "
          f"{results[(wname,'A upstream')]['rss']:11.2f}G "
          f"{results[(wname,'B port, no cuda')]['rss']:7.2f}G "
          f"{results[(wname,'C GPU 1-chunk')]['rss']:7.2f}G "
          f"{results[(wname,'D GPU N-chunk')]['rss']:7.2f}G")""")

code(r"""out = {"date": time.strftime("%Y-%m-%d"),
       "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader", quiet=True).stdout.strip(),
       "commit": sh("git -C /content/ours rev-parse --short HEAD", quiet=True).stdout.strip(),
       "budget_mb": BUDGET,
       "results": {f"{w}|{a}": {k: v for k, v in r.items() if k != "tail"}
                   for (w, a), r in results.items()}}
with open("/content/bench272_v2.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:2000])
try:
    from google.colab import files
    files.download("/content/bench272_v2.json")
except Exception as e:
    print("(download skipped:", e, ")")""")

# --------------------------------------------------------------------------
nb = {"cells": C,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("CUDA_RNAFold_Bench272_v2.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote CUDA_RNAFold_Bench272_v2.ipynb", len(C), "cells")

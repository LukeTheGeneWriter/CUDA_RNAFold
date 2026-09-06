#!/usr/bin/env python3
"""Generate the ViennaRNA 2.7.2 GPU benchmark notebook.

The question this answers is the one nobody has asked yet and everybody will
ask first: **how much faster is the CUDA build than ViennaRNA 2.7.2 itself?**
Every speed number this project owns was measured on the 2.3.0 fork.

Three arms, so the answer separates two different claims:

    A  upstream v2.7.2, stock build              the real baseline
    B  port27, built WITHOUT --enable-cuda       does the port cost anything
                                                 when the accelerator is off?
    C  port27, built WITH --enable-cuda          what the GPU buys

A vs B is the one upstream cares about (our changes must not slow their CPU
path). B vs C isolates the accelerator from every other difference. A vs C is
the headline.

All three are release builds -- NDEBUG on, which is now the default -- because
this is a timing run. The verification bar is the thing that uses
--enable-asserts, and it is not this notebook.
"""
import json

C = []


def _lines(s):
    # nbformat wants each entry to KEEP its trailing newline; splitting on "\n"
    # collapses the cell onto one line and silently breaks the notebook.
    return s.splitlines(keepends=True)


def md(s):
    C.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": _lines(s)})


# --------------------------------------------------------------------------
md(r"""# CUDA_RNAFold on ViennaRNA 2.7.2 — the benchmark

**What this measures, and why it does not exist yet.** The port is finished and
verified: `RNAfold` folds on the GPU on 2.7.2, byte-identical to the reference
set, with a green release bar. But every timing this project owns was taken on
the **2.3.0** fork. The claim we are about to put to the ViennaRNA maintainers
is "this accelerates your code", and we have never timed it against their code.

## Three arms

| | tree | build | what it answers |
|---|---|---|---|
| **A** | upstream `v2.7.2` | stock | the real baseline |
| **B** | `port27` | **no** `--enable-cuda` | does the port cost anything with the accelerator OFF? |
| **C** | `port27` | `--enable-cuda` | what the GPU buys |

**A vs B is the arm upstream cares about most.** A patch that slows their CPU
path is a patch they will not take, however fast the GPU is. **B vs C isolates
the accelerator** from every other difference between the trees. **A vs C** is
the headline number.

All three are **release** builds (`NDEBUG` on, the new default). The
verification bar is what uses `--enable-asserts`; this is not that.

## What would make this run worthless

- **Different answers.** Timings are only comparable if all three arms fold the
  same structures. Every arm's output is hashed and compared; a mismatch
  invalidates the run rather than being noted in passing.
- **A GPU arm where the GPU never ran.** Arm C counts the sweep's own
  `sweep shape:` lines. Zero sweeps is a failed arm, not a fast one. (This
  already happened once locally: a chunk size below `MIN_GPU_BATCH` sent every
  record to the CPU fallback and the bar reported success.)
- **Clock throttling.** GPU clocks are printed before and after.""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""import subprocess, os, sys, json, time, re, hashlib, statistics

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

code(r"""# Build-from-git needs more than the release tarball does: the generated
# gengetopt parsers, man pages and doxygen XML are NOT in upstream's git, and
# configure hard-errors until dlib/libsvm are unpacked. Each of these cost a
# failed build the first time round.
sh("apt-get -qq update && apt-get -qq install -y "
   "gengetopt help2man texinfo doxygen check libsubunit-dev bison flex "
   "> /dev/null 2>&1", check=False)
for t in ["gengetopt", "help2man", "makeinfo", "doxygen", "autoreconf"]:
    p = sh(f"command -v {t}", check=False, quiet=True)
    print(f"  {t:12s} {'OK' if p.returncode == 0 else 'MISSING'}")""")

# --------------------------------------------------------------------------
md(r"""## 2. Fetch both trees

`port27` is branched off the `v2.7.2` tag, which is upstream `master` exactly
(`1ffec79`) — so arm A is not an old release, it is current upstream.""")

code(r"""REPO = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"

sh(f"rm -rf /content/ours /content/upstream")
sh(f"git clone -q --branch {BRANCH} {REPO} /content/ours")
print("ours   :", sh("git -C /content/ours log --oneline -1", quiet=True).stdout.strip())

# arm A comes from the same clone's v2.7.2 tag -- same objects, no second
# download, and provably the tag port27 is based on.
sh("git clone -q /content/ours /content/upstream")
sh("git -C /content/upstream checkout -q v2.7.2")
print("upstream:", sh("git -C /content/upstream log --oneline -1", quiet=True).stdout.strip())

# The shape of the diff being benchmarked. This is the number the maintainers
# look at first, so print it split by status rather than as one total:
# "71 files changed" sounds enormous, but 59 of those are NEW files (the CUDA
# backend, tools, docs) that upstream can ignore entirely. What matters is how
# few of THEIR files were touched, and how little was removed from them.
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
md(r"""## 3. Build the three arms

Identical `CFLAGS` across all three: they come from the same `configure`
invocation, differing only in `--enable-cuda`. That matters — the two local
trees in development were **not** flag-comparable (`-g -O2` versus conda's
tuned flags) and no timing could cross between them.""")

code(r"""COMMON = ("--without-python --without-perl --without-swig --without-doc "
          "--without-rnaxplorer --without-forester --without-kinfold "
          "--without-rnalocmin")

def build(path, tag, extra=""):
    t0 = time.time()
    sh(f"cd {path} && tar -xjf src/dlib-*.tar.bz2 -C src/ 2>/dev/null || true", check=False, quiet=True)
    sh(f"cd {path} && tar -xzf src/libsvm-*.tar.gz -C src/ 2>/dev/null || true", check=False, quiet=True)
    sh(f"cd {path} && ./autogen.sh > ag.log 2>&1", quiet=True)
    p = sh(f"cd {path} && ./configure {COMMON} {extra} > cfg.log 2>&1", check=False, quiet=True)
    if p.returncode:
        print(sh(f"grep -m3 'configure: error' {path}/cfg.log", check=False, quiet=True).stdout)
        raise RuntimeError(f"configure failed for {tag}")
    p = sh(f"cd {path} && make -j$(nproc) > mk.log 2>&1", check=False, quiet=True)
    if p.returncode:
        print(sh(f"grep -inE '\\berror\\b' {path}/mk.log | head -8", check=False, quiet=True).stdout)
        raise RuntimeError(f"make failed for {tag}")
    bin_ = f"{path}/src/bin/RNAfold"
    assert os.path.exists(bin_), bin_
    print(f"  {tag:24s} built in {time.time()-t0:5.0f}s")
    return bin_

BIN = {}
BIN["A upstream 2.7.2"]   = build("/content/upstream", "A upstream 2.7.2")
BIN["B port27 (no cuda)"] = build("/content/ours", "B port27 (no cuda)")

# arm C reuses the same tree; distclean so the only difference is the flag
sh("cd /content/ours && cp src/bin/RNAfold /content/rnafold_nocuda", quiet=True)
BIN["B port27 (no cuda)"] = "/content/rnafold_nocuda"
sh("cd /content/ours && make distclean > /dev/null 2>&1", check=False, quiet=True)
BIN["C port27 (cuda)"]    = build("/content/ours", "C port27 (cuda)", "--enable-cuda")

# confirm the CUDA arm really has the backend compiled in
print()
print("nvcc invocations in arm C:",
      sh("grep -c NVCC /content/ours/mk.log", check=False, quiet=True).stdout.strip())""")

# --------------------------------------------------------------------------
md(r"""## 4. Workloads

Two shapes, because the accelerator's advantage is length-dependent:

- **W1 — 120 × 2000 nt.** The same shape the historical ladder used, so this
  number is comparable with the whole 2.3.0 progress series.
- **W2 — 60 × 5601 nt.** The long-record shape. `MIN_GPU_BATCH`'s break-even is
  ~1 record at ≥1200 nt, so this is where the GPU should be furthest ahead.

Both are uniform-length. Mixed-length batching works (the join mask), but a
uniform workload keeps the comparison about speed rather than about chunking.""")

code(r"""import random

def make_fa(path, n, length, seed):
    random.seed(seed)
    with open(path, "w") as f:
        for i in range(n):
            s = "".join(random.choice("ACGU") for _ in range(length))
            f.write(f">rec_{i}\n{s}\n")
    return path

os.makedirs("/content/wl", exist_ok=True)
W = {
  "W1 120x2000": make_fa("/content/wl/w1.fa", 120, 2000, 20260906),
  "W2  60x5601": make_fa("/content/wl/w2.fa",  60, 5601, 20260906),
}
for k, v in W.items():
    print(f"  {k}  {os.path.getsize(v)/1e6:.1f} MB")""")

# --------------------------------------------------------------------------
md(r"""## 5. Run

Each arm gets a discarded warm-up plus three timed repetitions; the best is
reported with the spread, because a single run on a shared machine is noise.

The CPU arms use `-j` (all cores) — the honest comparison for an accelerator is
against the CPU actually working, not against one core.""")

code(r"""REPS = 3

def run_once(binary, fa, gpu):
    env = dict(os.environ)
    cmd = [binary, "--noPS", "-i", fa]
    if gpu:
        env["RNA_GPU_CHUNK"] = "0"      # budget decides the chunk size
    else:
        env.pop("RNA_GPU_CHUNK", None)   # must not leak into the CPU arms
        cmd.insert(1, f"-j{os.cpu_count()}")
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True,
                       text=True, env=env)
    wall = time.time() - t0
    rss = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m: rss = int(m.group(1)) / 1e6
    sweeps = len(re.findall(r"sweep shape:", p.stderr))
    return dict(wall=wall, rss=rss, rc=p.returncode,
                sha=hashlib.sha256(p.stdout.encode()).hexdigest()[:16],
                lines=p.stdout.count("\n"), sweeps=sweeps)

results = {}
for wname, fa in W.items():
    print(f"\n=== {wname}")
    for aname, b in BIN.items():
        gpu = aname.startswith("C")
        run_once(b, fa, gpu)                      # warm-up, discarded
        rs = [run_once(b, fa, gpu) for _ in range(REPS)]
        best = min(r["wall"] for r in rs)
        worst = max(r["wall"] for r in rs)
        r0 = rs[0]
        results[(wname, aname)] = dict(
            best=best, spread=100*(worst-best)/best, rss=max(r["rss"] for r in rs),
            sha=r0["sha"], lines=r0["lines"], sweeps=r0["sweeps"], rc=r0["rc"])
        print(f"  {aname:24s} {best:8.2f}s  spread {results[(wname,aname)]['spread']:4.1f}%"
              f"  rss {results[(wname,aname)]['rss']:5.2f}G  sweeps {r0['sweeps']}"
              f"  sha {r0['sha']}")""")

# --------------------------------------------------------------------------
md(r"""## 6. Validity gates

Run these **before** reading any speedup. Each one can invalidate the numbers
above, and each has produced a misleading result in this project before.""")

code(r"""ok = True
for wname in W:
    shas = {a: results[(wname, a)]["sha"] for a in BIN}
    if len(set(shas.values())) != 1:
        ok = False
        print(f"  {wname}: ARMS DISAGREE -- timings are NOT comparable")
        for a, s in shas.items(): print(f"      {a:24s} {s}")
    else:
        print(f"  {wname}: all arms identical  ({list(shas.values())[0]})")

    c = results[(wname, "C port27 (cuda)")]
    if c["sweeps"] == 0:
        ok = False
        print(f"  {wname}: arm C ran NO GPU SWEEPS -- it folded on the CPU, "
              f"so its time is not a GPU time")
    if c["rc"] != 0:
        ok = False
        print(f"  {wname}: arm C exited {c['rc']}")

print()
print(sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu "
         "--format=csv,noheader", quiet=True).stdout.strip(), " <- clocks after the run")
print()
print("VALID" if ok else "*** INVALID -- do not quote these numbers ***")""")

# --------------------------------------------------------------------------
md("## 7. The answer")

code(r"""print(f"{'workload':14s} {'A upstream':>11s} {'B port,off':>11s} {'C port,GPU':>11s}"
      f" {'A/C':>7s} {'B/C':>7s} {'A/B':>7s}")
for wname in W:
    a = results[(wname, "A upstream 2.7.2")]["best"]
    b = results[(wname, "B port27 (no cuda)")]["best"]
    c = results[(wname, "C port27 (cuda)")]["best"]
    print(f"{wname:14s} {a:10.2f}s {b:10.2f}s {c:10.2f}s "
          f"{a/c:6.2f}x {b/c:6.2f}x {a/b:6.2f}x")

print()
print("A/C  the headline: CUDA build vs upstream ViennaRNA 2.7.2")
print("B/C  what the accelerator itself buys, same tree, same flags")
print("A/B  the port's cost with the GPU OFF -- should be ~1.00x, and a number")
print("     BELOW 1.00 (our CPU path slower than upstream's) is a finding that")
print("     needs explaining before any of this goes to the maintainers")

print()
print(f"{'workload':14s} {'peak RSS A':>12s} {'B':>8s} {'C':>8s}")
for wname in W:
    print(f"{wname:14s} "
          f"{results[(wname,'A upstream 2.7.2')]['rss']:11.2f}G "
          f"{results[(wname,'B port27 (no cuda)')]['rss']:7.2f}G "
          f"{results[(wname,'C port27 (cuda)')]['rss']:7.2f}G")""")

code(r"""out = {"date": time.strftime("%Y-%m-%d"),
       "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader", quiet=True).stdout.strip(),
       "commit": sh("git -C /content/ours rev-parse --short HEAD", quiet=True).stdout.strip(),
       "results": {f"{w}|{a}": v for (w, a), v in results.items()}}
with open("/content/bench272.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:1500])
try:
    from google.colab import files
    files.download("/content/bench272.json")
except Exception as e:
    print("(download skipped:", e, ")")""")

# --------------------------------------------------------------------------
nb = {"cells": C,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("CUDA_RNAFold_Bench272.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote CUDA_RNAFold_Bench272.ipynb", len(C), "cells")

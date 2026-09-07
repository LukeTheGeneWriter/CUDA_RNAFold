#!/usr/bin/env python3
"""Build the CUDA_RNAFold benchmark v4 Colab notebook.

v4 exists to ask a question v1-v3 could not afford to: what happens at a
workload big enough that the GPU path is doing what it was built for, and the
CPU path is simply not runnable inside a session?

400 x 5601 nt is ~50 GB of int32 triangles -- genuinely multi-chunk on a 24 GB
L4 without any artificial cap. Upstream would need ~1.1 hours per arm for it,
times two CPU arms times reps: most of a day, and the session dies first.

So the CPU arms are EXTRAPOLATED. That is a claim, not a convenience, and the
notebook treats it as one: it fits on small points, VALIDATES the fit against a
larger point it did not fit on, and re-measures that point at the END of the
run to see whether the machine held its throughput. An extrapolation that has
never been checked against a measurement is exactly the class of instrument
this project has been burned by eight times.

usage: python3 tools/make_nb_bench272_v4.py [out.ipynb]
"""
import json
import sys

cells = []


def _lines(s):
    # nbformat wants each entry to KEEP its trailing newline.
    return s.splitlines(True)


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(s.strip("\n"))})


# --------------------------------------------------------------------------
md(r"""# CUDA_RNAFold on ViennaRNA 2.7.2 - benchmark v4

**The workload is the point.** v1 measured 60-120 records and found every CUDA
arm reporting `sweeps: 1` -- the whole input fit one chunk, so none of the batch
machinery was exercised. v2 forced chunking with an artificial VRAM cap and
found it cost 2.3-4.2x. v3 added the int16 arms and measured 1.08-1.12x on the
DRAM stream against a predicted 1.4x.

All three shared a ceiling: the CPU arm had to fold the same input, so the input
had to stay small enough for a CPU to finish it.

v4 removes that ceiling. **400 records x 5601 nt** is ~50 GB of int32 triangles
-- multi-chunk on a 24 GB L4 with no artificial cap at all. This is the first
time the chunking path is measured on a workload that genuinely needs it rather
than one squeezed into needing it.

## The CPU arms are extrapolated, and that is a claim

Records are independent and identical in length, so total time is `a + b*n`:
`a` is process startup plus parameter loading, `b` is one 5601 nt fold. Both
coefficients mean something physical -- this is not a curve fitted to a shape.

**Three guards keep it honest, and any of them fails the run:**

| guard | what it catches |
|---|---|
| **fit on {5,10,20}, validate at 40** | the model is checked against a point it was NOT fitted to. Miss by >5% and the run fails rather than reports. |
| **re-measure 40 at the END** | the extrapolation assumes a shared VM sustains for ~1.1 h the throughput it showed for ~6 min. Two probes hours apart make drift visible instead of assumed. |
| **reach is printed beside every ratio** | the model is read out 10x beyond its largest measured point. That is the dominant uncertainty and it travels with the number. |

None of this makes an extrapolated number as good as a measured one. What it
does is stop an extrapolation being quoted as though it were one.

## The instrument v3 discarded

`fill_arrays_loop.c:480` prints one line per chunk: iterations, active
record-rows, cells, and peak records. **v3 counted those lines to get a chunk
count and threw the contents away.**

That mattered. Measured locally 2026-09-07: `MIN_GPU_BATCH = 10` makes
`flush_gpu_chunk()` fold any chunk of fewer than ten records **entirely on the
CPU**. Chunk capacity comes from the VRAM budget, so the last chunk holds a
quantisation remainder -- and int16's larger chunks leave a *larger* remainder.
On 40 x 2000 nt at a 448 MB budget, int16 left 6 records on the CPU and took
27.5 s where int32 took 8.7 s. It looked exactly like an int16 regression, and
`RNA_MIN_GPU_BATCH=1` collapsed it to a 2% difference.

**v3's gates could not see it.** They tested `sweeps == 0`, which catches a
total fallback and never a partial one. v4 sums the per-chunk record counts and
fails on any unintended CPU record -- and adds `RNA_MIN_GPU_BATCH` arms to
measure what the threshold costs at 5601 nt, where `RNAfold.c`'s own comment
says break-even should be **1 record**.

## What would make this run worthless

- arms disagreeing on the records they both folded;
- a CUDA arm with zero sweeps -- it folded on the CPU and the number is a lie;
- **records folded on the CPU inside a GPU arm** -- the gate v3 lacked;
- an int16 arm where the gate never engaged -- the int32 arm wearing a label;
- the extrapolation failing its validation point, or drifting across the run;
- clocks throttling.
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""import subprocess, os, sys, json, time, re, hashlib, random, statistics

def sh(cmd, check=True, quiet=False):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        if p.stdout.strip(): print(p.stdout.strip()[:4000])
        if p.returncode and p.stderr.strip(): print(p.stderr.strip()[:4000])
    if check and p.returncode:
        raise RuntimeError("failed (%d): %s\n%s" % (p.returncode, cmd, p.stderr[:2000]))
    return p

def clocks():
    q = ("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
         "clocks_throttle_reasons.active --format=csv,noheader")
    return sh(q, quiet=True).stdout.strip()

print(sh("nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm "
         "--format=csv,noheader", quiet=True).stdout.strip())
print(sh("nvcc --version | tail -2", quiet=True).stdout.strip())
print("cores:", os.cpu_count())
print("clocks:", clocks())""")

code(r"""# Build-from-git needs more than the release tarball: the generated gengetopt
# parsers, man pages and doxygen XML are not in upstream's git, and configure
# hard-errors until dlib/libsvm are unpacked.
sh("apt-get -qq update && apt-get -qq install -y gengetopt help2man texinfo "
   "doxygen autoconf automake libtool time > /dev/null 2>&1", check=False)
for t in ["gengetopt", "help2man", "makeinfo", "doxygen", "autoreconf",
          "libtool", "time"]:
    r = sh("which " + t, check=False, quiet=True)
    print("  %-12s %s" % (t, "OK" if r.returncode == 0 else "MISSING"))""")

# --------------------------------------------------------------------------
md("## 2. Fetch both trees")

code(r"""REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/bench"
sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
print("ours    :", sh("cd %s/port27 && git log --oneline -1" % ROOT,
                      quiet=True).stdout.strip())

# Arm A comes from the SAME clone's v2.7.2 tag -- same objects, no second
# download, and provably the tag port27 is based on.
sh("git -C %s/port27 worktree add -q --detach %s/upstream v2.7.2" % (ROOT, ROOT))
print("upstream:", sh("cd %s/upstream && git log --oneline -1" % ROOT,
                      quiet=True).stdout.strip())

sh("git -C %s/port27 worktree add -q --detach %s/port27nocuda %s"
   % (ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT,
            quiet=True).stdout.strip()
print("commit  :", COMMIT)""")

# --------------------------------------------------------------------------
md("""## 3. Build the three arms

Identical `CFLAGS` across all three -- same `configure` invocation, differing
only in `--enable-cuda` and which tree it runs in.""")

code(r"""COMMON = ("--without-python --without-perl --without-swig --without-doc "
          "--without-rnaxplorer --without-forester --without-kinfold "
          "--without-rnalocmin")

def build(path, tag, extra=""):
    t0 = time.time()
    sh("cd %s && ./autogen.sh > /dev/null 2>&1" % path, check=False, quiet=True)
    log = "/tmp/conf_%s.log" % tag.split()[0]
    p = sh("cd %s && ./configure %s %s CFLAGS='-g -O2' CXXFLAGS='-g -O2' > %s 2>&1"
           % (path, COMMON, extra, log), check=False, quiet=True)
    if p.returncode:
        print(sh("tail -30 " + log, quiet=True).stdout)
        raise RuntimeError("configure failed for " + tag)
    mlog = "/tmp/make_%s.log" % tag.split()[0]
    p = sh("cd %s && make -j$(nproc) > %s 2>&1" % (path, mlog),
           check=False, quiet=True)
    if p.returncode:
        print(sh("tail -40 " + mlog, quiet=True).stdout)
        raise RuntimeError("make failed for " + tag)
    b = path + "/src/bin/RNAfold"
    assert os.path.exists(b), b
    print("  %-24s built in %6.0fs" % (tag, time.time() - t0))
    return b

BIN = {}
BIN["A"] = build(ROOT + "/upstream",     "A upstream 2.7.2")
BIN["B"] = build(ROOT + "/port27nocuda", "B port27 (no cuda)")
BIN["C"] = build(ROOT + "/port27",       "C port27 (cuda)", "--enable-cuda")
print("nvcc invocations in arm C:",
      sh("grep -c nvcc /tmp/make_C.log", check=False, quiet=True).stdout.strip())

# The stale-binary trap, which has produced a false PASS in this project
# before: a binary older than its own sources tested something else.
for k, b in BIN.items():
    src = os.path.dirname(os.path.dirname(os.path.dirname(b)))
    newer = sh("find %s/src \\( -name '*.c' -o -name '*.cu' -o -name '*.h' \\) "
               "-newer %s | head -3" % (src, b), check=False, quiet=True).stdout.strip()
    print("  arm %s: %s" % (k, ("STALE -- " + newer) if newer
                            else "binary newer than every source"))""")

# --------------------------------------------------------------------------
md(r"""## 4. Workload

One workload, deliberately: **400 x 5601 nt**. Uniform length, because the
extrapolation's whole basis is that every record costs the same -- a mixed
workload would need a per-length model, and v4 is already spending its risk
budget on the extrapolation.

The calibration files are **prefixes** of the big one, so the records the CPU
folds are literally the records the GPU folds. That is what lets the
correctness gate compare them at all.""")

code(r"""N_BIG   = 400
LEN     = 5601
SEED    = 20260907
CAL_N   = [5, 10, 20]     # points the linear model is FITTED on
VAL_N   = 40              # point it is VALIDATED against, and never fitted to

os.makedirs(ROOT + "/fa", exist_ok=True)

def make_uniform(path, n, length, seed):
    random.seed(seed)
    with open(path, "w") as f:
        for i in range(n):
            f.write(">u%d\n%s\n" % (i, "".join(random.choice("ACGU")
                                              for _ in range(length))))
    return path

BIG = make_uniform(ROOT + "/fa/big.fa", N_BIG, LEN, SEED)

def prefix_fa(src, n, dst):
    # First n records, verbatim: same sequences, same order, same headers.
    out, kept = [], 0
    with open(src) as f:
        for line in f:
            if line.startswith(">"):
                kept += 1
                if kept > n:
                    break
            out.append(line)
    with open(dst, "w") as f:
        f.writelines(out)
    return dst

SUB = {n: prefix_fa(BIG, n, "%s/fa/sub%d.fa" % (ROOT, n))
       for n in CAL_N + [VAL_N]}
for n, p in sorted(SUB.items()):
    assert sh("grep -c '^>' " + p, quiet=True).stdout.strip() == str(n)

print("big: %d x %d nt = %.1f MB" % (N_BIG, LEN, os.path.getsize(BIG) / 1e6))
print("calibration subsets:", sorted(SUB))
print("int32 triangles: ~%.0f GB total" % (N_BIG * 4.0 * LEN * LEN / 1e9))""")

# --------------------------------------------------------------------------
md(r"""## 5. The runner

Every run records more than its wall clock, because v3 proved a wall clock
alone cannot tell a healthy run from one quietly folding records on the CPU:

- **`sweeps`** -- one per GPU chunk;
- **`gpu_records`** -- summed from each chunk's `peak/iteration N records`. Short
  of the record count means the difference folded on the CPU. **This is the
  number v3 discarded**;
- **`chunk_shapes`** -- the full per-chunk line kept rather than counted, so
  composition can be read afterwards instead of inferred from a count;
- **`int16_active`** -- the gate's own announcement, so an int16 arm that ran as
  int32 cannot report 1.00x and look real;
- **clocks before and after** each run.""")

code(r"""SWEEP_RE = re.compile(r"sweep shape: (\d+) iterations, (\d+) active record-rows, "
                      r"(\d+) cells; peak/iteration (\d+) records (\d+) cells")

def run_once(binary, fa, gpu, budget_mb=None, int16=False, min_batch=None):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16", "RNA_GPU_CHUNK", "RNA_GPU_VRAM_BUDGET_MB",
              "RNA_MIN_GPU_BATCH"):
        env.pop(k, None)
    if gpu:
        env["RNA_GPU_CHUNK"] = "0"            # the budget decides chunk size
        if budget_mb:
            env["RNA_GPU_VRAM_BUDGET_MB"] = str(budget_mb)
        if int16:
            env["RNA_FML_INT16"] = "1"
        if min_batch:
            env["RNA_MIN_GPU_BATCH"] = str(min_batch)

    c0 = clocks()
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-v", binary, "--noPS", "-i", fa],
                       capture_output=True, text=True, env=env)
    wall = time.time() - t0
    c1 = clocks()

    rss = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m:
        rss = int(m.group(1)) / 1e6

    shapes = [dict(iters=int(a), rows=int(b), cells=int(c),
                   records=int(d), peak_cells=int(e))
              for a, b, c, d, e in SWEEP_RE.findall(p.stderr)]
    return dict(wall=wall, rss=rss, rc=p.returncode,
                sha=hashlib.sha256(p.stdout.encode()).hexdigest()[:16],
                nrec=p.stdout.count(">"),
                sweeps=len(shapes),
                gpu_records=sum(s["records"] for s in shapes),
                chunk_shapes=shapes,
                int16_active=("RNA_FML_INT16=1" in p.stderr),
                int16_wanted=bool(int16),
                min_batch_ok=(("RNA_MIN_GPU_BATCH=%d" % min_batch) in p.stderr)
                             if min_batch else None,
                clocks_before=c0, clocks_after=c1)

RESULTS = {}

def save():
    # Written after EVERY arm, so a Colab timeout costs the remaining arms and
    # not the finished ones.
    with open("/content/bench272_v4.json", "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d"), "commit": COMMIT,
                   "n_big": N_BIG, "len": LEN,
                   "results": {"%s|%s" % k: v for k, v in RESULTS.items()}},
                  f, indent=2)

print("runner ready")""")

# --------------------------------------------------------------------------
md(r"""## 6. CPU calibration

Arms A and B fold {5, 10, 20} records. The model `t = a + b*n` is fitted on
those three and then asked to predict 40 -- a point it has never seen.

If it misses by more than 5%, the run fails here rather than reporting an
extrapolated speedup built on a model that does not hold.""")

code(r"""TOL = 0.05     # how far the validation point may miss before the run fails

def fit_linear(ns, ts):
    # Least squares on three points. No library, and the arithmetic should be
    # readable at a glance.
    n = len(ns)
    mx, my = sum(ns) / n, sum(ts) / n
    sxx = sum((x - mx) ** 2 for x in ns)
    sxy = sum((x - mx) * (y - my) for x, y in zip(ns, ts))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(ns, ts))
    ss_tot = sum((y - my) ** 2 for y in ts)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2

CPU = {}
for arm in ("A", "B"):
    ns, ts = [], []
    for k in CAL_N:
        r = run_once(BIN[arm], SUB[k], gpu=False)
        assert r["rc"] == 0, "arm %s n=%d rc=%d" % (arm, k, r["rc"])
        RESULTS[("cal%d" % k, arm)] = r
        ns.append(k)
        ts.append(r["wall"])
        print("  arm %s  n=%3d  %8.1fs" % (arm, k, r["wall"]))
    a, b, r2 = fit_linear(ns, ts)

    v = run_once(BIN[arm], SUB[VAL_N], gpu=False)
    RESULTS[("val%d" % VAL_N, arm)] = v
    pred = a + b * VAL_N
    err = (pred - v["wall"]) / v["wall"]
    CPU[arm] = dict(a=a, b=b, r2=r2, val_pred=pred, val_actual=v["wall"],
                    val_err=err, pred_big=a + b * N_BIG)
    print("  arm %s  t = %.1f + %.3f*n   R2=%.5f" % (arm, a, b, r2))
    print("  arm %s  validate n=%d: predicted %.1fs, measured %.1fs (%+.2f%%)  %s"
          % (arm, VAL_N, pred, v["wall"], err * 100,
             "OK" if abs(err) <= TOL else "*** FIT REJECTED ***"))
    print("  arm %s  => extrapolated n=%d: %.0fs (%.2f h)\n"
          % (arm, N_BIG, CPU[arm]["pred_big"], CPU[arm]["pred_big"] / 3600))
    save()""")

# --------------------------------------------------------------------------
md(r"""## 7. The GPU arms

Four configurations on the full 400 records, all at the card's natural budget
-- no artificial VRAM cap, because at this size the workload chunks on its own.
That is the difference from v2 and v3, where chunking had to be forced.

| arm | what it is |
|---|---|
| **C** | int32, stock `MIN_GPU_BATCH` |
| **E** | int16, stock `MIN_GPU_BATCH` |
| **Cm** | int32, `RNA_MIN_GPU_BATCH=1` |
| **Em** | int16, `RNA_MIN_GPU_BATCH=1` |

C vs E is the encoding. C vs Cm and E vs Em are what the fallback threshold
costs at 5601 nt -- where break-even should be one record, so any fallback is
pure loss.""")

code(r"""REPS = 1
GPU_ARMS = [("C",  dict(int16=False, min_batch=None)),
            ("E",  dict(int16=True,  min_batch=None)),
            ("Cm", dict(int16=False, min_batch=1)),
            ("Em", dict(int16=True,  min_batch=1))]
ARM_KW = dict(GPU_ARMS)

# Interleaved, one rep of every arm per round -- v3's correction to v2, which
# ran each arm's reps consecutively and so let VM drift land unevenly on the
# arms and appear as a difference between them.
runs = {tag: [] for tag, _ in GPU_ARMS}
for rep in range(REPS):
    line = []
    for tag, kw in GPU_ARMS:
        r = run_once(BIN["C"], BIG, gpu=True, **kw)
        runs[tag].append(r)
        cpu_rec = r["nrec"] - r["gpu_records"]
        line.append("%s=%.0fs(%dch%s)" % (tag, r["wall"], r["sweeps"],
                                          ",%dcpu" % cpu_rec if cpu_rec else ""))
        RESULTS[("big_r%d" % rep, tag)] = r
        save()
    print("  round %d/%d: %s" % (rep + 1, REPS, "  ".join(line)))

GPU = {}
for tag, _ in GPU_ARMS:
    rs = runs[tag]
    best = min(r["wall"] for r in rs)
    worst = max(r["wall"] for r in rs)
    r0 = min(rs, key=lambda r: r["wall"])
    GPU[tag] = dict(r0, best=best, spread=100 * (worst - best) / best)
    print("  %-3s %8.1fs  spread %4.1f%%  %d chunks  gpu_records %d/%d  "
          "rss %.2fG  sha %s"
          % (tag, best, GPU[tag]["spread"], r0["sweeps"], r0["gpu_records"],
             r0["nrec"], r0["rss"], r0["sha"]))""")

# --------------------------------------------------------------------------
md(r"""## 8. The drift probe

The extrapolation's real risk is not the model -- it is the assumption that a
shared VM sustains for over an hour the throughput it showed for six minutes. The
GPU arms have just spent a long time on this machine. Re-measuring the
validation point now, against the same measurement taken before them, is the
only way to see that assumption fail.""")

code(r"""DRIFT = {}
for arm in ("A", "B"):
    r = run_once(BIN[arm], SUB[VAL_N], gpu=False)
    RESULTS[("drift%d" % VAL_N, arm)] = r
    before = CPU[arm]["val_actual"]
    d = (r["wall"] - before) / before
    DRIFT[arm] = dict(before=before, after=r["wall"], drift=d)
    print("  arm %s  n=%d: %.1fs before the GPU arms, %.1fs after  (%+.2f%%)"
          % (arm, VAL_N, before, r["wall"], d * 100))
save()""")

# --------------------------------------------------------------------------
md(r"""## 9. Validity gates

Read these before any speedup. Each can invalidate the run, and each exists
because something once passed without it.""")

code(r"""ok, notes = True, []

def fail(msg):
    global ok
    ok = False
    notes.append("FAIL  " + msg)
    print("  FAIL  " + msg)

def good(msg):
    notes.append("ok    " + msg)
    print("  ok    " + msg)

# 1. Correctness, on the records both sides actually folded. The CPU never
#    folds all 400, so the bar is the VAL_N-record prefix -- same records, same
#    order. Comparing a 400-record sha against a 40-record one would always
#    differ, which is the shape of check that proves nothing.
sub_shas = {}
for tag, _ in GPU_ARMS:
    sub_shas[tag] = run_once(BIN["C"], SUB[VAL_N], gpu=True, **ARM_KW[tag])["sha"]
sub_shas["A"] = RESULTS[("val%d" % VAL_N, "A")]["sha"]
sub_shas["B"] = RESULTS[("val%d" % VAL_N, "B")]["sha"]
if len(set(sub_shas.values())) == 1:
    good("all arms identical on the %d-record subset (%s)" % (VAL_N, sub_shas["A"]))
else:
    fail("arms disagree on the subset: %s" % sub_shas)

# 2. Every GPU arm used the GPU, and used it for EVERY record. v3 checked only
#    `sweeps == 0`, which catches a total fallback and never a partial one --
#    the defect that made int16 look like a regression.
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    if g["sweeps"] == 0:
        fail("arm %s: zero sweeps -- folded on the CPU" % tag)
    elif g["sweeps"] == 1:
        notes.append("note  arm %s: single chunk -- chunking untested here" % tag)
        print("  note  arm %s: single chunk" % tag)
    else:
        good("arm %s: %d chunks at the natural budget" % (tag, g["sweeps"]))
    short = g["nrec"] - g["gpu_records"]
    if short and not tag.endswith("m"):
        notes.append("note  arm %s: %d of %d records fell to the CPU "
                     "(MIN_GPU_BATCH) -- the effect Cm/Em measure"
                     % (tag, short, g["nrec"]))
        print("  note  arm %s: %d records folded on the CPU" % (tag, short))
    elif short:
        fail("arm %s: %d records on the CPU despite RNA_MIN_GPU_BATCH=1"
             % (tag, short))
    else:
        good("arm %s: all %d records folded on the GPU" % (tag, g["nrec"]))

# 3. The int16 gate engaged where asked for, and nowhere else.
if all(GPU[t]["int16_wanted"] == GPU[t]["int16_active"] for t, _ in GPU_ARMS):
    good("int16 gate state matches intent in every arm")
else:
    for tag, _ in GPU_ARMS:
        g = GPU[tag]
        if g["int16_wanted"] != g["int16_active"]:
            fail("arm %s: int16 wanted=%s active=%s"
                 % (tag, g["int16_wanted"], g["int16_active"]))

# 3b. The threshold override announced itself where it was set. Without this an
#     arm that ignored RNA_MIN_GPU_BATCH is the stock arm wearing a label.
for tag, kw in GPU_ARMS:
    if kw["min_batch"] and not GPU[tag]["min_batch_ok"]:
        fail("arm %s: RNA_MIN_GPU_BATCH never announced -- override not applied" % tag)

# 4. The extrapolation held at a point it was not fitted to.
for arm in ("A", "B"):
    e = abs(CPU[arm]["val_err"])
    if e > TOL:
        fail("arm %s: fit mispredicts n=%d by %.1f%% (> %.0f%%)"
             % (arm, VAL_N, e * 100, TOL * 100))
    else:
        good("arm %s: fit predicts n=%d within %.2f%%, R2=%.5f"
             % (arm, VAL_N, e * 100, CPU[arm]["r2"]))

# 5. The machine did not drift underneath the extrapolation.
for arm in ("A", "B"):
    d = abs(DRIFT[arm]["drift"])
    if d > TOL:
        fail("arm %s: CPU throughput drifted %.1f%% across the run -- the "
             "extrapolation cannot be trusted to %d records" % (arm, d * 100, N_BIG))
    else:
        good("arm %s: CPU throughput stable within %.2f%% across the run"
             % (arm, d * 100))

# 6. Clocks and exit codes.
print("  clocks:", clocks())
for tag, _ in GPU_ARMS:
    if "Not Active" not in GPU[tag]["clocks_after"]:
        notes.append("note  arm %s throttle: %s" % (tag, GPU[tag]["clocks_after"]))
    if GPU[tag]["rc"] != 0:
        fail("arm %s: exit %d" % (tag, GPU[tag]["rc"]))

print()
print("VALID" if ok else "*** INVALID -- do not quote these numbers ***")""")

# --------------------------------------------------------------------------
md("## 10. The answer")

code(r"""EA, EB = CPU["A"]["pred_big"], CPU["B"]["pred_big"]
reach = float(N_BIG) / VAL_N

print("%d x %d nt\n" % (N_BIG, LEN))
print("  A upstream 2.7.2   %9.0fs  (%.2f h)  EXTRAPOLATED from n<=%d"
      % (EA, EA / 3600, VAL_N))
print("  B port, no cuda    %9.0fs  (%.2f h)  EXTRAPOLATED" % (EB, EB / 3600))
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    print("  %-3s                %9.1fs  (%.2f h)  measured, %d chunks"
          % (tag, g["best"], g["best"] / 3600, g["sweeps"]))

print("\n%-4s %10s %8s %7s %9s %7s" % ("arm", "wall", "A/x", "chunks",
                                       "cpu recs", "rss"))
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    print("%-4s %9.1fs %7.2fx %7d %9d %6.2fG"
          % (tag, g["best"], EA / g["best"], g["sweeps"],
             g["nrec"] - g["gpu_records"], g["rss"]))

print("\n  A/B (extrapolated both sides): %.3f" % (EA / EB))
print("  int16, stock threshold    C/E   = %.3fx"
      % (GPU["C"]["best"] / GPU["E"]["best"]))
print("  int16, threshold at 1     Cm/Em = %.3fx"
      % (GPU["Cm"]["best"] / GPU["Em"]["best"]))
print("  threshold cost, int32     C/Cm  = %.3fx"
      % (GPU["C"]["best"] / GPU["Cm"]["best"]))
print("  threshold cost, int16     E/Em  = %.3fx"
      % (GPU["E"]["best"] / GPU["Em"]["best"]))

print("\nEXTRAPOLATION REACH: the CPU arms are modelled from at most %d records "
      "and read out at %d\n-- %.0fx beyond the largest measured point. The fit "
      "predicted n=%d within %.2f%% (A) /\n%.2f%% (B), and CPU throughput moved "
      "%+.2f%% (A) / %+.2f%% (B) across the run.\nEvery A/x above inherits both "
      "uncertainties; none of them is a measured ratio."
      % (VAL_N, N_BIG, reach, VAL_N, abs(CPU["A"]["val_err"]) * 100,
         abs(CPU["B"]["val_err"]) * 100, DRIFT["A"]["drift"] * 100,
         DRIFT["B"]["drift"] * 100))""")

code(r"""# Per-chunk composition, kept rather than counted. v3 printed these lines and
# discarded them, which is why a partial CPU fallback went unseen for a session.
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    print("%s: %d chunks, %d/%d records on the GPU"
          % (tag, g["sweeps"], g["gpu_records"], g["nrec"]))
    for i, s in enumerate(g["chunk_shapes"]):
        print("     chunk %2d: %4d records  %6d iterations  %14s cells"
              % (i, s["records"], s["iters"], "{:,}".format(s["cells"])))
    print()""")

# --------------------------------------------------------------------------
md(r"""## 11. Optional: is `modular_decomposition` still DRAM-bound under int16?

The question v3 left open. int16 halved the `fml_j` stream and bought only
1.08-1.12x against a predicted ~1.4x. Two explanations, calling for opposite
next moves:

- **`fml_j` was never the whole stream.** If `fml_i` also reaches DRAM, halving
  one of two comparable streams gives ~0.75x before overheads -- close to what
  was measured, and no second lever is hiding here.
- **The kernel left the DRAM roof.** It was at 88.3% of peak at int32. If int16
  dropped it well below, something else is now the limiter and there IS a
  further lever.

`lts__t_sector_hit_rate` answers a third question on top. The kernel re-reads
each `fML` column once per sweep row, and a proposed optimisation stages those
re-reads in shared memory across a batch of rows. If L2 already catches them,
that work duplicates what the hardware does for free.

**Sampled, never whole-app.** Kernel replay saves and restores multi-GB buffers
per launch; whole-app profiling on this project has already cost a five-hour
dead end.""")

code(r"""RUN_NCU = False    # set True to spend ~5-10 min answering the DRAM question

if RUN_NCU:
    M = ",".join(["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                  "dram__bytes_read.sum", "dram__bytes_write.sum",
                  "l1tex__t_sector_hit_rate.pct",
                  "lts__t_sector_hit_rate.pct",
                  "gpu__time_duration.sum"])
    for tag, i16 in (("i32", False), ("i16", True)):
        env = "RNA_GPU_CHUNK=0" + (" RNA_FML_INT16=1" if i16 else "")
        sh("%s ncu --target-processes all -k modular_decomposition_kernel "
           "--launch-skip 2000 --launch-count 5 --metrics %s --csv "
           "%s --noPS -i %s > /content/ncu_%s.csv 2> /content/ncu_%s.err"
           % (env, M, BIN["C"], SUB[CAL_N[0]], tag, tag),
           check=False, quiet=True)
        print("--- %s ---" % tag)
        print(sh("tail -8 /content/ncu_%s.csv" % tag, check=False,
                 quiet=True).stdout)
else:
    print("RUN_NCU is False -- set it True to answer the DRAM-roof question.")""")

# --------------------------------------------------------------------------
code(r"""out = {"date": time.strftime("%Y-%m-%d"),
       "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader",
                 quiet=True).stdout.strip(),
       "commit": COMMIT, "valid": ok,
       "n_big": N_BIG, "len": LEN, "cal_n": CAL_N, "val_n": VAL_N,
       "cpu_model": CPU, "drift": DRIFT, "notes": notes,
       "gpu_arms": {t: {k: v for k, v in GPU[t].items() if k != "chunk_shapes"}
                    for t, _ in GPU_ARMS},
       "chunk_shapes": {t: GPU[t]["chunk_shapes"] for t, _ in GPU_ARMS}}
with open("/content/bench272_v4.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "chunk_shapes"},
                 indent=2)[:3000])""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

dst = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Bench272_v4.ipynb"
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (dst, len(cells)))

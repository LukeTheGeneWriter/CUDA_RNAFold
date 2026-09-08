#!/usr/bin/env python3
"""Build the CUDA_RNAFold benchmark v5 Colab notebook.

v5 is v4's question with v4's CPU baseline replaced.

v4 established that 400 x 5601 nt is only reachable if the CPU arms are
EXTRAPOLATED, and it built the discipline that makes an extrapolation quotable:
fit on small points, validate against a point never fitted on, re-measure at the
end to catch VM drift. That discipline is kept here in full.

What is replaced is how the points are BOUGHT. v4 folded four nested prefixes
per arm -- 5, 10, 20 and 40 records -- to learn four points on t(n), and then
re-folded 40 more for the drift probe: 115 record-folds per arm, 230 across the
two CPU arms, and at ~10 s a record that is over half an hour before the GPU
arms start.

Three of those four points were bought twice over. sub5 is a prefix of sub10 is
a prefix of sub20 is a prefix of sub40, so the first five records are folded
four times and the whole exercise learns four points from 75 folds.

It does not have to. `vrna_cstr_fflush()` (datastructures/char_stream.c:127)
does an `fprintf` followed by an explicit `fflush`, and `RNAfold.c:1908` hands
each record to the ostream the moment it is folded. So every record's output
arrives on the pipe as ONE atomic block, at the instant that record finishes. A
reader that timestamps those blocks gets t(1), t(2), ... t(40) from a SINGLE run
of 40 records -- the entire curve v4 sampled at four places, at a third of the
cost and with forty times the resolution.

    v4:  5 + 10 + 20 + 40 + 40  = 115 folds/arm  ->  4 points
    v5:           40 + 6 + tiny =  47 folds/arm  -> 40 points

The saved time is not respent. v5 keeps CAL_N = 40, so the extrapolation reach
(400/40 = 10x beyond the largest measured point) is exactly v4's and the two
runs' headline numbers stay comparable.

What the resolution buys, beyond speed:

- the held-out check becomes twenty points instead of one, and can see
  CURVATURE -- a memory-pressure knee that three points and one check cannot;
- drift becomes visible WITHIN a single run (first-half slope vs second-half),
  not only across the hour;
- a VM stall shows as one outlying delta instead of silently corrupting a fit;
- in the GPU arms the same timestamps make the MIN_GPU_BATCH tail DIRECTLY
  visible as the records that arrive slowly at the end -- v4 could only infer
  it from `gpu_records`.

And because the substitution is itself a claim, it is checked rather than
assumed: the end-of-run drift probe is a genuine standalone process, and the
model built entirely from stream timestamps has to predict its wall clock. If
timestamped arrivals were not equivalent to whole-run costs, that check fails.

usage: python3 tools/make_nb_bench272_v5.py [out.ipynb]
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
md(r"""# CUDA_RNAFold on ViennaRNA 2.7.2 - benchmark v5

Same question as v4, same workload, same extrapolation discipline. **The CPU
baseline is bought differently, and costs a third of what it did.**

## What v4 paid, and why it did not have to

v4 measured the CPU curve `t(n)` by folding four nested prefixes -- 5, 10, 20
and 40 records -- and then re-folded 40 more as a drift probe. That is 115
record-folds per arm, 230 across the two CPU arms, and at ~10 s a record it is
over half an hour before the GPU arms start.

But `sub5` is a prefix of `sub10` is a prefix of `sub20` is a prefix of `sub40`.
The first five records get folded four times. **75 folds buy four points.**

They are already on the curve of a single run. `vrna_cstr_fflush()`
(`datastructures/char_stream.c:127`) does an `fprintf` and then an explicit
`fflush`, and `RNAfold.c:1908` hands each record to the ostream the moment it
is folded -- so every record's output lands on the pipe as one atomic block at
the instant that record finishes. Timestamp the blocks and one 40-record run
yields `t(1) ... t(40)`.

|  | folds per arm | points on `t(n)` |
|---|---|---|
| v4 | 5 + 10 + 20 + 40 + 40 = **115** | 4 |
| v5 | 40 + 6 + tiny = **47** | **40** |

**The saved time is not respent.** `CAL_N` stays 40, so the extrapolation reads
out at 400 from a largest measured point of 40 -- the same 10x reach as v4, and
the two runs stay directly comparable.

## What forty points buy that four could not

- the held-out check is **twenty points, not one**, so it can see *curvature* --
  a memory-pressure knee that three fitted points and one check cannot;
- **drift within a single run** (first-half slope vs second-half), not only
  across the hour;
- a VM **stall** appears as one outlying delta instead of quietly corrupting `b`;
- in the **GPU arms**, the same timestamps make the `MIN_GPU_BATCH` tail
  *directly visible* as the records arriving slowly at the end. v4 could only
  infer it from `gpu_records`.

## The substitution is a claim, so it is checked

Reading `t(k)` off a stream is not self-evidently the same as running `n = k`
and timing the process. So the end-of-run drift probe is a genuine standalone
run, and a model built **entirely from stream timestamps** has to predict its
wall clock. If the substitution were invalid, that check fails.

## What is unchanged from v4

The workload (400 x 5601 nt), the three builds, the four GPU arms, and every
validity gate. A run is INVALID -- not merely noisy -- on:

- arms disagreeing on the records they both folded;
- a CUDA arm with zero sweeps;
- **records folded on the CPU inside a GPU arm**;
- an int16 arm where the gate never engaged;
- the extrapolation failing its held-out points, or drifting across the run;
- **stream timestamps that do not reconcile with a standalone run** (new);
- clocks throttling.
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""import subprocess, os, sys, json, time, re, hashlib, random, statistics, threading

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
md(r"""## 3b. Does this binary actually flush per record?

The whole saving rests on one property of the build in front of us, so it is
verified on that build rather than read out of the source. Fold three short
records and check their output blocks arrive at three *separated* times.

If they arrive together, the stream is being buffered somewhere this notebook
does not control, and the fast path is invalid. `STREAM_OK` records that, and
the validity gates fail the run on it -- an instrument that cannot reach what it
claims to measure must announce that, not report success.""")

code(r"""def stream_probe(binary, fa):
    p = subprocess.Popen([binary, "--noPS", "-i", fa],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    t0, marks, pend = time.time(), [], b""
    while True:
        chunk = os.read(p.stdout.fileno(), 1 << 16)
        if not chunk:
            break
        t = time.time() - t0
        pend += chunk
        parts = pend.split(b"\n")
        pend = parts[-1]
        marks += [t for ln in parts[:-1] if ln.startswith(b">")]
    p.wait()
    return marks

print("stream_probe defined -- run in section 4, once the FASTA files exist")""")

# --------------------------------------------------------------------------
md(r"""## 4. Workload

One workload, deliberately: **400 x 5601 nt**. Uniform length, because the
extrapolation's whole basis is that every record costs the same -- a mixed
workload would need a per-length model, and v5 is still spending its risk budget
on the extrapolation.

The calibration file is a **prefix** of the big one, so the records the CPU
folds are literally the records the GPU folds. In v5 the calibration file *is*
the correctness subset, because there is only one of them now.""")

code(r"""N_BIG   = 400
LEN     = 5601
SEED    = 20260907
CAL_N   = 40    # ONE streamed run per arm. Yields t(1)..t(40).
FIT_N   = CAL_N // 2      # the model sees only n <= FIT_N
DRIFT_N = 6               # standalone end-of-run probe; also validates streaming

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
       for n in (DRIFT_N, CAL_N)}
for n, p in sorted(SUB.items()):
    assert sh("grep -c '^>' " + p, quiet=True).stdout.strip() == str(n)

# A trivial record, to see process startup on its own. It costs nothing to fold
# and it is the only way to read `a` without inferring it from the fit.
TINY = ROOT + "/fa/tiny.fa"
random.seed(1)
with open(TINY, "w") as f:
    f.write(">tiny\n%s\n" % "".join(random.choice("ACGU") for _ in range(60)))

print("big: %d x %d nt = %.1f MB" % (N_BIG, LEN, os.path.getsize(BIG) / 1e6))
print("calibration / correctness subset: %d records" % CAL_N)
print("int32 triangles: ~%.0f GB total" % (N_BIG * 4.0 * LEN * LEN / 1e9))""")

code(r"""# Now run the flush probe from 3b, on short records so it costs a second.
probe_fa = ROOT + "/fa/probe.fa"
random.seed(2)
with open(probe_fa, "w") as f:
    for i in range(3):
        f.write(">p%d\n%s\n" % (i, "".join(random.choice("ACGU")
                                          for _ in range(700))))

pm = stream_probe(BIN["A"], probe_fa)
gaps = [pm[i] - pm[i - 1] for i in range(1, len(pm))]
STREAM_OK = (len(pm) == 3 and all(g > 1e-3 for g in gaps))
print("record arrivals: %s" % ["%.4fs" % m for m in pm])
print("gaps           : %s" % ["%.4fs" % g for g in gaps])
print("per-record flushing: %s"
      % ("CONFIRMED on this build" if STREAM_OK else
         "*** NOT OBSERVED -- the streamed calibration is not valid here ***"))""")

# --------------------------------------------------------------------------
md(r"""## 5. The runner

Every run records more than its wall clock, because v3 proved a wall clock alone
cannot tell a healthy run from one quietly folding records on the CPU:

- **`sweeps`** -- one per GPU chunk;
- **`gpu_records`** -- summed from each chunk's `peak/iteration N records`. Short
  of the record count means the difference folded on the CPU;
- **`chunk_shapes`** -- the full per-chunk line kept rather than counted;
- **`int16_active`** -- the gate's own announcement;
- **`marks`** -- *new in v5*: the arrival time of every record's output block.
  On a CPU arm that is the whole `t(n)` curve. On a GPU arm it is the chunk
  timeline, and the `MIN_GPU_BATCH` tail is visible in it directly;
- clocks before and after each run.

`stderr` is drained on its own thread. Reading one pipe to completion while the
other fills is a deadlock, and at 400 records `stderr` is not small.""")

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
    p = subprocess.Popen(["/usr/bin/time", "-v", binary, "--noPS", "-i", fa],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    err = []
    th = threading.Thread(target=lambda: err.append(p.stderr.read()))
    th.start()

    out, marks, pend = bytearray(), [], b""
    while True:
        chunk = os.read(p.stdout.fileno(), 1 << 16)
        if not chunk:
            break
        t = time.time() - t0
        out += chunk
        pend += chunk
        parts = pend.split(b"\n")
        pend = parts[-1]
        marks += [t for ln in parts[:-1] if ln.startswith(b">")]
    rc = p.wait()
    th.join()
    wall = time.time() - t0
    c1 = clocks()
    stderr = err[0].decode("utf-8", "replace") if err else ""

    rss = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    if m:
        rss = int(m.group(1)) / 1e6

    blob = bytes(out)
    shapes = [dict(iters=int(a), rows=int(b), cells=int(c),
                   records=int(d), peak_cells=int(e))
              for a, b, c, d, e in SWEEP_RE.findall(stderr)]
    return dict(wall=wall, rss=rss, rc=rc,
                sha=hashlib.sha256(blob).hexdigest()[:16],
                nrec=blob.count(b"\n>") + (1 if blob[:1] == b">" else 0),
                marks=marks,
                sweeps=len(shapes),
                gpu_records=sum(s["records"] for s in shapes),
                chunk_shapes=shapes,
                int16_active=("RNA_FML_INT16=1" in stderr),
                int16_wanted=bool(int16),
                min_batch_ok=(("RNA_MIN_GPU_BATCH=%d" % min_batch) in stderr)
                             if min_batch else None,
                clocks_before=c0, clocks_after=c1)

RESULTS = {}

def save():
    # Written after EVERY arm, so a Colab timeout costs the remaining arms and
    # not the finished ones.
    with open("/content/bench272_v5.json", "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d"), "commit": COMMIT,
                   "n_big": N_BIG, "len": LEN,
                   "results": {"%s|%s" % k: v for k, v in RESULTS.items()}},
                  f, indent=2)

print("runner ready")""")

# --------------------------------------------------------------------------
md(r"""## 6. CPU calibration -- one run per arm

Each arm folds `CAL_N = 40` records **once**. The arrival timestamps give
`t(1) ... t(40)` directly, so the model `t = a + b*n` is fitted on `n <= 20` and
then asked to predict every point in `n = 21..40` -- twenty held-out points
where v4 had one, bought with a third of the folds.

`a` is reported but barely matters: at n = 400 it is a fraction of a percent of
the total. **`b` is the entire extrapolation**, which is why forty samples of it
is the thing worth buying, and why the standard error of `b` is printed rather
than assumed.

Three checks the four-point version could not make:

- **curvature** -- the largest held-out residual, not just the one at n = 40;
- **within-run drift** -- mean per-record cost over the first half against the
  second, so an hour-long trend is visible in six minutes;
- **stalls** -- the largest single per-record delta against the median. One VM
  preemption inside a calibration run corrupts `b`, and in v4 it would have
  surfaced only as a slightly worse three-point fit.""")

code(r"""TOL       = 0.05    # held-out miss, and drift, that invalidates the run
STALL_TOL = 3.0     # max/median per-record delta before we call it a stall

def fit_linear(ns, ts):
    # Least squares. No library, and the arithmetic should be readable.
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
    tiny = run_once(BIN[arm], TINY, gpu=False)
    r = run_once(BIN[arm], SUB[CAL_N], gpu=False)
    assert r["rc"] == 0, "arm %s rc=%d" % (arm, r["rc"])
    RESULTS[("cal%d" % CAL_N, arm)] = r
    RESULTS[("tiny", arm)] = tiny

    marks = r["marks"]
    assert len(marks) == CAL_N, ("arm %s: %d arrivals for %d records -- the "
                                 "stream is not per-record" % (arm, len(marks), CAL_N))
    ks = list(range(1, CAL_N + 1))
    deltas = [marks[i] - marks[i - 1] for i in range(1, CAL_N)]

    # Fitted on the first half only. The second half is never shown to the fit.
    a, b, r2 = fit_linear(ks[:FIT_N], marks[:FIT_N])

    held = [(k, marks[k - 1], a + b * k) for k in ks[FIT_N:]]
    errs = [(pred - act) / act for _, act, pred in held]
    val_err = errs[-1]                    # at n=CAL_N -- v4's single comparison
    max_err = max(errs, key=abs)          # curvature, which v4 could not see

    h = len(deltas) // 2
    b_first  = statistics.mean(deltas[:h])
    b_second = statistics.mean(deltas[h:])
    within   = (b_second - b_first) / b_first
    med      = statistics.median(deltas)
    stall    = max(deltas) / med
    se_b     = statistics.stdev(deltas) / (len(deltas) ** 0.5)

    CPU[arm] = dict(a=a, b=b, r2=r2, val_pred=a + b * CAL_N,
                    val_actual=marks[-1], val_err=val_err, max_held_err=max_err,
                    b_first=b_first, b_second=b_second, within_drift=within,
                    delta_med=med, delta_sd=statistics.stdev(deltas),
                    stall_ratio=stall, se_b=se_b,
                    deltas=deltas, marks=marks, cal_wall=r["wall"],
                    tiny_wall=tiny["wall"], pred_big=a + b * N_BIG)
    c = CPU[arm]
    print("  arm %s  folded %d records ONCE, in %.1fs" % (arm, CAL_N, r["wall"]))
    print("           t = %.2f + %.3f*n   R2=%.6f   (fitted on n<=%d only)"
          % (a, b, r2, FIT_N))
    print("           per record %.3fs median, SE(b) %.3fs = %.2f%%; startup "
          "a=%.2fs (tiny run %.2fs)"
          % (med, se_b, 100 * se_b / b, a, tiny["wall"]))
    print("           held out n=%d..%d: worst %+.2f%%, at n=%d %+.2f%%   %s"
          % (FIT_N + 1, CAL_N, max_err * 100, CAL_N, val_err * 100,
             "OK" if abs(max_err) <= TOL else "*** FIT REJECTED ***"))
    print("           within-run drift: %.3fs/rec -> %.3fs/rec (%+.2f%%)"
          % (b_first, b_second, within * 100))
    print("           largest delta / median: %.2fx  %s"
          % (stall, "" if stall <= STALL_TOL else "*** STALL ***"))
    print("           => extrapolated n=%d: %.0fs (%.2f h)\n"
          % (N_BIG, c["pred_big"], c["pred_big"] / 3600))
    save()

print("CPU baseline will cost %d record-folds this run; v4's nested prefixes "
      "cost %d." % (2 * (CAL_N + DRIFT_N + 1), 2 * 115))""")

# --------------------------------------------------------------------------
md(r"""## 7. The GPU arms

Four configurations on the full 400 records, all at the card's natural budget --
no artificial VRAM cap, because at this size the workload chunks on its own.

| arm | what it is |
|---|---|
| **C** | int32, stock `MIN_GPU_BATCH` |
| **E** | int16, stock `MIN_GPU_BATCH` |
| **Cm** | int32, `RNA_MIN_GPU_BATCH=1` |
| **Em** | int16, `RNA_MIN_GPU_BATCH=1` |

C vs E is the encoding. C vs Cm and E vs Em are what the fallback threshold costs
at 5601 nt -- where break-even should be under one record, so any fallback is
pure loss. **v5 prices that loss directly**: the records a chunk leaves behind
arrive one slow block at a time at the end of the run, so the tail is read off
the timeline in seconds instead of inferred from a record count.""")

code(r"""REPS = 1
GPU_ARMS = [("C",  dict(int16=False, min_batch=None)),
            ("E",  dict(int16=True,  min_batch=None)),
            ("Cm", dict(int16=False, min_batch=1)),
            ("Em", dict(int16=True,  min_batch=1))]
ARM_KW = dict(GPU_ARMS)

def cpu_tail_seconds(r):
    # GPU records arrive in chunk-sized bursts; anything MIN_GPU_BATCH left
    # behind is folded serially afterwards. Time from the last GPU record's
    # arrival to the end of the run is that tail, measured rather than modelled.
    short = r["nrec"] - r["gpu_records"]
    if short <= 0 or len(r["marks"]) < r["nrec"]:
        return 0.0
    return r["marks"][-1] - r["marks"][r["nrec"] - short - 1]

# Interleaved, one rep of every arm per round -- v3's correction to v2, which
# ran each arm's reps consecutively and so let VM drift land unevenly.
runs = {tag: [] for tag, _ in GPU_ARMS}
for rep in range(REPS):
    line = []
    for tag, kw in GPU_ARMS:
        r = run_once(BIN["C"], BIG, gpu=True, **kw)
        r["cpu_tail"] = cpu_tail_seconds(r)
        runs[tag].append(r)
        cpu_rec = r["nrec"] - r["gpu_records"]
        line.append("%s=%.0fs(%dch%s)"
                    % (tag, r["wall"], r["sweeps"],
                       ",%dcpu/%.0fs" % (cpu_rec, r["cpu_tail"]) if cpu_rec else ""))
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
          "cpu tail %5.1fs  rss %.2fG  sha %s"
          % (tag, best, GPU[tag]["spread"], r0["sweeps"], r0["gpu_records"],
             r0["nrec"], r0["cpu_tail"], r0["rss"], r0["sha"]))""")

# --------------------------------------------------------------------------
md(r"""## 8. The drift probe -- and the check on v5's own method

Two jobs in one `DRIFT_N`-record run per arm.

**Drift.** The extrapolation's real risk is not the model, it is the assumption
that a shared VM sustains for over an hour the throughput it showed for six
minutes. The GPU arms have just spent a long time on this machine.

**Validating the substitution.** This probe is a genuine standalone process,
timed end to end from outside. The model predicting it was built *entirely from
stream timestamps inside a different run*. If reading `t(k)` off a stream were
not equivalent to running `n = k` and timing the process, this prediction fails
-- and that is the one way v5's cheaper method can be caught being wrong.

`DRIFT_N` is 6 rather than v4's 40 because the calibration run now reports the
per-record dispersion, so the noise floor of a 6-record probe is *known* rather
than assumed. It is printed next to the result, so a "drift" inside the noise
cannot be read as a finding.""")

code(r"""DRIFT = {}
for arm in ("A", "B"):
    r = run_once(BIN[arm], SUB[DRIFT_N], gpu=False)
    RESULTS[("drift%d" % DRIFT_N, arm)] = r
    c = CPU[arm]

    # 1. Does the stream-built model predict a whole process it never saw?
    pred = c["a"] + c["b"] * DRIFT_N
    method_err = (pred - r["wall"]) / r["wall"]

    # 2. Did per-record throughput move? Compared like with like -- the probe's
    #    own per-record deltas against the calibration run's.
    d = [r["marks"][i] - r["marks"][i - 1] for i in range(1, len(r["marks"]))]
    after = statistics.mean(d) if d else float("nan")
    drift = (after - c["delta_med"]) / c["delta_med"]
    noise = c["delta_sd"] / max(len(d), 1) ** 0.5

    DRIFT[arm] = dict(pred=pred, wall=r["wall"], method_err=method_err,
                      before=c["delta_med"], after=after, drift=drift,
                      noise_pct=100 * noise / c["delta_med"])
    print("  arm %s  method check: model says %.1fs for a standalone n=%d run, "
          "it took %.1fs (%+.2f%%)"
          % (arm, pred, DRIFT_N, r["wall"], method_err * 100))
    print("           throughput  : %.3fs/rec before the GPU arms, %.3fs/rec "
          "after (%+.2f%%, noise floor +/-%.2f%%)"
          % (c["delta_med"], after, drift * 100, DRIFT[arm]["noise_pct"]))
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

# 0. v5's OWN method, gated before anything it produced is read. If per-record
#    flushing does not happen on this build, the calibration is not measuring
#    what it claims and nothing downstream of it means anything.
if not STREAM_OK:
    fail("per-record flushing not observed -- the streamed calibration is invalid")
else:
    good("per-record flushing confirmed on this build")
for arm in ("A", "B"):
    m = abs(DRIFT[arm]["method_err"])
    if m > TOL:
        fail("arm %s: the stream-built model mispredicts a standalone n=%d run "
             "by %.1f%% -- t(k) read off a stream is NOT a whole-run cost here"
             % (arm, DRIFT_N, m * 100))
    else:
        good("arm %s: stream-built model predicts a standalone run within %.2f%%"
             % (arm, m * 100))

# 1. Correctness, on the records both sides actually folded. The CPU never folds
#    all 400, so the bar is the CAL_N-record prefix -- same records, same order.
#    Comparing a 400-record sha against a 40-record one would always differ,
#    which is the shape of check that proves nothing.
sub_shas = {}
for tag, _ in GPU_ARMS:
    sub_shas[tag] = run_once(BIN["C"], SUB[CAL_N], gpu=True, **ARM_KW[tag])["sha"]
sub_shas["A"] = RESULTS[("cal%d" % CAL_N, "A")]["sha"]
sub_shas["B"] = RESULTS[("cal%d" % CAL_N, "B")]["sha"]
if len(set(sub_shas.values())) == 1:
    good("all arms identical on the %d-record subset (%s)" % (CAL_N, sub_shas["A"]))
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
        notes.append("note  arm %s: %d of %d records fell to the CPU, costing "
                     "%.1fs of tail -- the effect Cm/Em measure"
                     % (tag, short, g["nrec"], g["cpu_tail"]))
        print("  note  arm %s: %d records on the CPU, %.1fs of measured tail"
              % (tag, short, g["cpu_tail"]))
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

# 4. The extrapolation held on points it was not fitted to -- ALL of them, not
#    just the far one, so curvature cannot hide between the fit and n=CAL_N.
for arm in ("A", "B"):
    e = abs(CPU[arm]["max_held_err"])
    if e > TOL:
        fail("arm %s: fit misses a held-out point by %.1f%% (> %.0f%%) -- t(n) "
             "is not linear over this range" % (arm, e * 100, TOL * 100))
    else:
        good("arm %s: every held-out point (n=%d..%d) within %.2f%%, R2=%.6f"
             % (arm, FIT_N + 1, CAL_N, e * 100, CPU[arm]["r2"]))

# 4b. No stall inside the calibration run. One VM preemption corrupts b, and in
#     v4 it could only have surfaced as a slightly worse three-point fit.
for arm in ("A", "B"):
    s = CPU[arm]["stall_ratio"]
    if s > STALL_TOL:
        fail("arm %s: one record took %.1fx the median -- the machine stalled "
             "inside the calibration and b is corrupted" % (arm, s))
    else:
        good("arm %s: no stall (largest record %.2fx the median)" % (arm, s))

# 5. The machine did not drift underneath the extrapolation -- within the
#    calibration run, and across the hour.
for arm in ("A", "B"):
    w = abs(CPU[arm]["within_drift"])
    d = abs(DRIFT[arm]["drift"])
    if w > TOL:
        fail("arm %s: per-record cost moved %.1f%% WITHIN the calibration run"
             % (arm, w * 100))
    else:
        good("arm %s: stable within the calibration run (%.2f%%)" % (arm, w * 100))
    if d > TOL:
        fail("arm %s: CPU throughput drifted %.1f%% across the run -- the "
             "extrapolation cannot be trusted to %d records" % (arm, d * 100, N_BIG))
    else:
        good("arm %s: CPU throughput stable within %.2f%% across the run "
             "(noise floor %.2f%%)" % (arm, d * 100, DRIFT[arm]["noise_pct"]))

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
reach = float(N_BIG) / CAL_N
spent, v4_spent = 2 * (CAL_N + DRIFT_N + 1), 2 * 115

print("%d x %d nt\n" % (N_BIG, LEN))
print("  A upstream 2.7.2   %9.0fs  (%.2f h)  EXTRAPOLATED from n<=%d"
      % (EA, EA / 3600, CAL_N))
print("  B port, no cuda    %9.0fs  (%.2f h)  EXTRAPOLATED" % (EB, EB / 3600))
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    print("  %-3s                %9.1fs  (%.2f h)  measured, %d chunks"
          % (tag, g["best"], g["best"] / 3600, g["sweeps"]))

print("\n%-4s %10s %8s %7s %9s %9s %7s"
      % ("arm", "wall", "A/x", "chunks", "cpu recs", "cpu tail", "rss"))
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    print("%-4s %9.1fs %7.2fx %7d %9d %8.1fs %6.2fG"
          % (tag, g["best"], EA / g["best"], g["sweeps"],
             g["nrec"] - g["gpu_records"], g["cpu_tail"], g["rss"]))

print("\n  A/B (extrapolated both sides): %.3f" % (EA / EB))
print("  int16, stock threshold    C/E   = %.3fx"
      % (GPU["C"]["best"] / GPU["E"]["best"]))
print("  int16, threshold at 1     Cm/Em = %.3fx"
      % (GPU["Cm"]["best"] / GPU["Em"]["best"]))
print("  threshold cost, int32     C/Cm  = %.3fx  (%.1fs of measured CPU tail)"
      % (GPU["C"]["best"] / GPU["Cm"]["best"], GPU["C"]["cpu_tail"]))
print("  threshold cost, int16     E/Em  = %.3fx  (%.1fs of measured CPU tail)"
      % (GPU["E"]["best"] / GPU["Em"]["best"], GPU["E"]["cpu_tail"]))

print("\nEXTRAPOLATION REACH: the CPU arms are modelled from at most %d records "
      "and read out at %d\n-- %.0fx beyond the largest measured point, the SAME "
      "reach as v4. The fit saw only\nn<=%d, and every held-out point out to n=%d "
      "landed within %.2f%% (A) / %.2f%% (B).\nCPU throughput moved %+.2f%% (A) / "
      "%+.2f%% (B) across the run.\nEvery A/x above inherits both uncertainties; "
      "none of them is a measured ratio."
      % (CAL_N, N_BIG, reach, FIT_N, CAL_N,
         abs(CPU["A"]["max_held_err"]) * 100, abs(CPU["B"]["max_held_err"]) * 100,
         DRIFT["A"]["drift"] * 100, DRIFT["B"]["drift"] * 100))

print("\nCPU BASELINE COST: %d record-folds against v4's %d -- %.1fx less wall "
      "clock for the\nsame reach, and 40 points on t(n) where v4 bought 4."
      % (spent, v4_spent, float(v4_spent) / spent))""")

code(r"""# The calibration curve, printed rather than plotted: every fifth point, plus
# the per-record deltas v4 never had. The fitted region is marked, so the
# held-out half can be read as the held-out half.
for arm in ("A", "B"):
    c = CPU[arm]
    print("arm %s   t = %.2f + %.3f*n   (fitted on n<=%d)" % (arm, c["a"], c["b"], FIT_N))
    print("   %4s %10s %10s %8s %9s   %s" % ("n", "measured", "model", "err%",
                                             "delta", "region"))
    for k in range(1, CAL_N + 1):
        if k != 1 and k % 5:
            continue
        act = c["marks"][k - 1]
        pred = c["a"] + c["b"] * k
        dl = c["deltas"][k - 2] if k >= 2 else float("nan")
        print("   %4d %9.1fs %9.1fs %+7.2f%% %8.2fs   %s"
              % (k, act, pred, 100 * (pred - act) / act, dl,
                 "fitted" if k <= FIT_N else "HELD OUT"))
    print()""")

code(r"""# Per-chunk composition, kept rather than counted. v3 printed these lines and
# discarded them, which is why a partial CPU fallback went unseen for a session.
for tag, _ in GPU_ARMS:
    g = GPU[tag]
    print("%s: %d chunks, %d/%d records on the GPU, %.1fs CPU tail"
          % (tag, g["sweeps"], g["gpu_records"], g["nrec"], g["cpu_tail"]))
    for i, s in enumerate(g["chunk_shapes"]):
        print("     chunk %2d: %4d records  %6d iterations  %14s cells"
              % (i, s["records"], s["iters"], "{:,}".format(s["cells"])))
    print()""")

# --------------------------------------------------------------------------
md(r"""## 11. Optional: is `modular_decomposition` still DRAM-bound under int16?

The question v3 left open and v4 carried. int16 halved the `fml_j` stream and
bought only 1.08-1.12x against a predicted ~1.4x -- and the RTX 3050
measurements since then show the answer is **machine-dependent**: 1.31x when
bandwidth-bound, 0.71x when the SMs are starved. This asks what the L4 is doing.

- **`fml_j` was never the whole stream.** If `fml_i` also reaches DRAM, halving
  one of two comparable streams gives ~0.75x before overheads.
- **The kernel left the DRAM roof.** It was at 88.3% of peak at int32.

`lts__t_sector_hit_rate` answers a third question: the kernel re-reads each
`fML` column once per sweep row, and a proposed optimisation stages those
re-reads in shared memory. If L2 already catches them, that work duplicates what
the hardware does for free.

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
           % (env, M, BIN["C"], SUB[DRIFT_N], tag, tag),
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
       "method": "streamed per-record timestamps", "stream_ok": STREAM_OK,
       "n_big": N_BIG, "len": LEN, "cal_n": CAL_N, "fit_n": FIT_N,
       "drift_n": DRIFT_N,
       "cpu_folds_spent": 2 * (CAL_N + DRIFT_N + 1), "cpu_folds_v4": 230,
       "cpu_model": {a: {k: v for k, v in CPU[a].items()
                         if k not in ("deltas", "marks")} for a in ("A", "B")},
       "cpu_curve": {a: {"marks": CPU[a]["marks"], "deltas": CPU[a]["deltas"]}
                     for a in ("A", "B")},
       "drift": DRIFT, "notes": notes,
       "gpu_arms": {t: {k: v for k, v in GPU[t].items()
                        if k not in ("chunk_shapes", "marks")}
                    for t, _ in GPU_ARMS},
       "chunk_shapes": {t: GPU[t]["chunk_shapes"] for t, _ in GPU_ARMS},
       "gpu_timelines": {t: GPU[t]["marks"] for t, _ in GPU_ARMS}}
with open("/content/bench272_v5.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("chunk_shapes", "cpu_curve", "gpu_timelines")},
                 indent=2)[:3000])""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

dst = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Bench272_v5.ipynb"
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (dst, len(cells)))

#!/usr/bin/env python3
"""Build the DEEP profiling Colab notebook — where does the wall clock go?

The first profile (2026-09-08, L4) established that int16 makes
modular_decomposition_kernel 1.456x faster, that fml_j is ~75% of its DRAM
reads, and that the kernel never leaves the DRAM roof. It did not reconcile
that with benchmark v5's 1.009x END TO END, and that gap is what this notebook
exists to close.

  1.46x kernel + a ~73% wall share  =>  Amdahl predicts ~1.30x overall
  measured on a T4                  =>  1.009x

Two candidate explanations, and they call for opposite next moves:
  - the T4 was starved (585 MHz of 1590) and erased the win. Then int16 should
    default ON for healthy hardware and there is nothing more to find.
  - the ~73% share does not hold at 400 x 5601 nt. Then most of the wall at
    scale is something we have never characterised, and THAT is the lever.

THE KEY REALISATION that shapes this notebook: RNAfold already prints a full
phase breakdown at exit (print_phase_timing_stats, registered via atexit in
fill_arrays.c:64). phase_modular_decomp_s answers the share question directly,
with no profiler attached and no profiling overhead to argue about. So the
decomposition comes first and ncu second, rather than the other way round.

AND A CONFOUND v5 COULD NOT SEE: int16 changed the CHUNK COUNT (5 -> 4). Fewer
chunks means less per-chunk overhead and more batch width, which is worth real
time on its own (see project_chunking_costs_batch_width). So v5's C/E ratio
conflated the encoding with a scheduling change. This notebook measures both a
natural-chunk pair AND a forced-equal-chunk pair, so the encoding can be priced
on its own.

BUILD FIXES this generator was missing, both of them already-known defects that
tools/standup_git_build.sh works around and this notebook did not inherit:
  - src/dlib-*.tar.bz2 and src/libsvm-*.tar.gz must be UNPACKED; configure
    refuses to proceed otherwise. The release tarball ships them unpacked, so
    this only bites a git-tree build.
  - PYTHON3 must be passed to configure. Without it --without-python leaves
    $(PYTHON3) empty while doc/source/man/Makefile.am:38 still expands
    "$(PYTHON3) ../../man2rst.py", so make tries to EXECUTE a mode-644 file and
    dies with "Permission denied" on all 25 man pages. Passing PYTHON3 is the
    root-cause fix; chmod +x is kept as belt-and-braces.

usage: python3 tools/make_nb_profile272_deep.py [out.ipynb]
"""
import json
import sys

cells = []


def _lines(s):
    return s.splitlines(True)


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(s.strip("\n"))})


# --------------------------------------------------------------------------
md(r"""# Where does the wall clock actually go?

The first profile settled what int16 does to **one kernel**: 1.456x faster,
`fml_j` is ~75% of its DRAM reads, and it never leaves the DRAM roof
(89.3% -> 80.9% of peak). It did **not** reconcile that with benchmark v5's
**1.009x end to end**, and closing that gap is the whole point of this notebook.

```
1.46x kernel  x  ~73% wall share   =>  Amdahl predicts ~1.30x overall
measured on a T4                   =>  1.009x
```

| explanation | what it would mean | next move |
|---|---|---|
| the T4 was **starved** (585 MHz of 1590) | int16 is fine; the machine erased it | default int16 on for healthy GPUs, stop here |
| the **~73% share does not hold** at scale | most of the wall at 400 x 5601 is uncharacterised | that share is the real lever, not int16 |

## The measurement that makes this cheap

RNAfold **already prints a full phase breakdown at exit** —
`print_phase_timing_stats()`, registered via `atexit` in `fill_arrays.c:64`:

```
phase timing (s): int_loop=… hp_mb=… load_my_c=… modular_decomp=… fetch_mx=…
                | new_c_host=… fml_host=… fml_prev_host=…
                || GPU+transfer total=… host-combine total=…
```

`phase_modular_decomp_s / wall` **is** the share, with no profiler attached and
no profiling overhead to argue about. So this notebook measures the
decomposition first and reaches for `ncu` only afterwards, to explain whatever
the decomposition points at.

## A confound v5 could not see

int16 changed the **chunk count**, 5 -> 4. Fewer chunks means less per-chunk
overhead *and* more batch width, and batch width is worth real time on its own.
So v5's `C/E` ratio priced *the encoding plus a scheduling change*.

Every size below is therefore run **twice**: once at the natural chunk count,
and once with the chunk count **forced equal** across both encodings. The
difference between those two ratios is what the VRAM saving is worth, separately
from what the narrower stream is worth.

## Build fixes this notebook previously lacked

Both are known defects `tools/standup_git_build.sh` already works around, and
which the earlier notebook did not inherit — a human had to patch them by hand
mid-run:

- `src/dlib-*.tar.bz2` and `src/libsvm-*.tar.gz` must be **unpacked**;
  `configure` refuses to proceed otherwise. Only a git-tree build hits this.
- **`PYTHON3` must be passed to `configure`.** Without it, `--without-python`
  leaves `$(PYTHON3)` empty while `doc/source/man/Makefile.am:38` still expands
  `$(PYTHON3) ../../man2rst.py`, so `make` tries to *execute* a mode-644 file
  and dies with "Permission denied" on all 25 man pages. That is the root cause;
  `chmod +x` is kept below as belt-and-braces.
""")

# --------------------------------------------------------------------------
md("## 1. Environment — and clocks, which is now a first-class variable")

code(r"""import subprocess, os, sys, json, time, re, random, csv, io, statistics

def sh(cmd, check=True, quiet=False):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        if p.stdout.strip(): print(p.stdout.strip()[:3000])
        if p.returncode and p.stderr.strip(): print(p.stderr.strip()[:3000])
    if check and p.returncode:
        raise RuntimeError("failed (%d): %s\n%s" % (p.returncode, cmd, p.stderr[:2000]))
    return p

def clocks():
    return sh("nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,"
              "clocks_throttle_reasons.active --format=csv,noheader",
              quiet=True).stdout.strip()

def clock_ratio():
    q = sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader,nounits",
           quiet=True).stdout.strip().split(",")
    try: return float(q[0]) / float(q[1])
    except Exception: return float("nan")

print(clocks())
print("clock ratio: %.2f of maximum" % clock_ratio())
print(sh("ncu --version | head -3", check=False, quiet=True).stdout.strip())
print("cores:", os.cpu_count())""")

code(r"""sh("apt-get -qq update && apt-get -qq install -y gengetopt help2man texinfo "
   "doxygen autoconf automake libtool > /dev/null 2>&1", check=False)
for t in ["gengetopt", "help2man", "makeinfo", "doxygen", "autoreconf", "libtool"]:
    r = sh("which " + t, check=False, quiet=True)
    print("  %-12s %s" % (t, "OK" if r.returncode == 0 else "MISSING"))""")

# --------------------------------------------------------------------------
md("""## 2. Clone and build — with the two fixes a human had to add by hand last time""")

code(r"""REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/deep"
EXPECT = None       # set to a short sha to REFUSE a stale clone, e.g. "54386f62"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())
if EXPECT and not COMMIT.startswith(EXPECT):
    raise SystemExit("*** clone is %s, expected %s -- push first ***" % (COMMIT, EXPECT))""")

code(r"""t0 = time.time()

# FIX 1: the bundled third-party sources ship as TARBALLS in git and configure
# refuses to proceed until they are unpacked. The release tarball ships them
# already unpacked, which is why this only ever bites a git-tree build.
for pat, flag in (("src/dlib-*.tar.bz2", "-xjf"), ("src/libsvm-*.tar.gz", "-xzf")):
    got = sh("ls %s/port27/%s 2>/dev/null" % (ROOT, pat), check=False, quiet=True).stdout.split()
    for t in got:
        d = re.sub(r"\.tar\.(bz2|gz)$", "", os.path.basename(t))
        if not os.path.isdir("%s/port27/src/%s" % (ROOT, d)):
            print("  unpacking", os.path.basename(t))
            sh("tar %s %s -C %s/port27/src/" % (flag, t, ROOT), quiet=True)

sh("cd %s/port27 && ./autogen.sh > /dev/null 2>&1" % ROOT, check=False, quiet=True)

# FIX 2: PYTHON3 is the ROOT-CAUSE fix for the man2rst.py failure. Without it,
# --without-python leaves $(PYTHON3) empty while doc/source/man/Makefile.am:38
# still expands "$(PYTHON3) ../../man2rst.py", so make tries to EXECUTE a
# mode-644 file: "Permission denied", 25 times. chmod is belt-and-braces.
sh("chmod +x %s/port27/doc/man2rst.py" % ROOT, check=False, quiet=True)
p = sh("cd %s/port27 && ./configure --without-python --without-perl --without-swig "
       "--without-doc --without-rnaxplorer --without-forester --without-kinfold "
       "--without-rnalocmin --enable-cuda CFLAGS='-g -O2' CXXFLAGS='-g -O2' "
       "PYTHON3=\"$(command -v python3)\" > /content/conf.log 2>&1" % ROOT,
       check=False, quiet=True)
if p.returncode:
    print(sh("tail -30 /content/conf.log", quiet=True).stdout); raise SystemExit("configure failed")

p = sh("cd %s/port27 && make -j$(nproc) > /content/make.log 2>&1" % ROOT, check=False, quiet=True)
if p.returncode:
    print(sh("tail -40 /content/make.log", quiet=True).stdout); raise SystemExit("make failed")

BIN = ROOT + "/port27/src/bin/RNAfold"
assert os.path.exists(BIN), BIN
print("built in %.0fs, %s nvcc invocations"
      % (time.time()-t0, sh("grep -c nvcc /content/make.log", check=False, quiet=True).stdout.strip()))""")

# --------------------------------------------------------------------------
md(r"""## 3. The runner — wall clock plus the phase breakdown the binary already emits

Nothing here attaches a profiler. `print_phase_timing_stats()` fires at exit and
the numbers are the program's own, so the decomposition costs nothing and cannot
be accused of being a profiling artifact.

Clocks are sampled before and after **every** run. The T4 result taught that a
throttled machine is not a different measurement, it is a different question.""")

code(r"""PHASE_RE = re.compile(
    r"int_loop=([\d.]+) hp_mb=([\d.]+) load_my_c=([\d.]+) modular_decomp=([\d.]+) "
    r"fetch_mx=([\d.]+) \| new_c_host=([\d.]+) fml_host=([\d.]+) fml_prev_host=([\d.]+) "
    r"\|\| GPU\+transfer total=([\d.]+) host-combine total=([\d.]+)")
SWEEP_RE = re.compile(r"sweep shape: (\d+) iterations, (\d+) active record-rows, "
                      r"(\d+) cells; peak/iteration (\d+) records (\d+) cells")

def run(fa, int16=False, chunk=0, min_batch=1, extra=None):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH",
              "RNA_SLOT_FLOW","RNA_CONTINUOUS_FLOW","RNA_GPU_VRAM_BUDGET_MB"):
        env.pop(k, None)
    env["RNA_GPU_CHUNK"]     = str(chunk)
    env["RNA_MIN_GPU_BATCH"] = str(min_batch)
    if int16: env["RNA_FML_INT16"] = "1"
    if extra: env.update(extra)

    c0 = clock_ratio(); t0 = time.time()
    p = subprocess.run([BIN, "--noPS", "-i", fa], capture_output=True, text=True, env=env)
    wall = time.time() - t0; c1 = clock_ratio()

    m = PHASE_RE.search(p.stderr)
    ph = {}
    if m:
        names = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
                 "new_c_host","fml_host","fml_prev_host","gpu_total","host_total")
        ph = dict(zip(names, [float(x) for x in m.groups()]))
    shapes = SWEEP_RE.findall(p.stderr)
    return dict(wall=wall, rc=p.returncode, phases=ph,
                sweeps=len(shapes), int16_active=("RNA_FML_INT16=1" in p.stderr),
                cells=sum(int(s[2]) for s in shapes),
                clock_before=c0, clock_after=c1,
                sha=__import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12])

def mkfa(path, n, length, seed=20260908):
    random.seed(seed)
    with open(path, "w") as f:
        for i in range(n):
            f.write(">r%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(length))))
    return path

os.makedirs(ROOT + "/fa", exist_ok=True)
print("runner ready")""")

# --------------------------------------------------------------------------
md(r"""## 4. Phase A — does the ~73 % share survive scale?

This is the question. `modular_decomp` was measured at ~73 % of wall on a much
smaller workload; if it is still ~73 % at 400 × 5601 nt then a 1.46× kernel
*must* show up end to end, and the T4's clocks are the only explanation left. If
the share has collapsed, int16 was never going to help at that size and the
uncharacterised remainder is the real target.

Sizes climb until the answer is clear. Trim `SIZES` if the session is short —
the last row is the expensive one.""")

code(r"""SIZES = [(16, 2000), (40, 2000), (40, 5601), (120, 5601)]

rowsA = []
for n, L in SIZES:
    fa = mkfa("%s/fa/%dx%d.fa" % (ROOT, n, L), n, L)
    for tag, i16 in (("i32", False), ("i16", True)):
        r = run(fa, int16=i16)
        assert r["rc"] == 0, (n, L, tag, r["rc"])
        assert r["int16_active"] == i16, "int16 gate disagreed with intent"
        assert r["phases"], "no phase line -- did print_phase_timing_stats run?"
        md_s = r["phases"]["modular_decomp"]
        rowsA.append(dict(n=n, L=L, tag=tag, wall=r["wall"], share=md_s / r["wall"],
                          md=md_s, sweeps=r["sweeps"], sha=r["sha"],
                          gpu_total=r["phases"]["gpu_total"],
                          host_total=r["phases"]["host_total"],
                          phases=r["phases"],      # kept whole: §6 reads it back
                          clock=r["clock_after"]))
        print("  %4dx%-5d %-4s wall %8.1fs  modular_decomp %8.1fs = %5.1f%%  "
              "gpu_total %6.1f host %5.1f  %d ch  clk %.2f"
              % (n, L, tag, r["wall"], md_s, 100*md_s/r["wall"],
                 r["phases"]["gpu_total"], r["phases"]["host_total"],
                 r["sweeps"], r["clock_after"]))""")

code(r"""print("%-12s %8s %8s %10s %10s %8s" % ("size","i32 share","i16 share","i32 wall","i16 wall","i16 gain"))
byk = {}
for r in rowsA: byk[(r["n"], r["L"], r["tag"])] = r
for n, L in SIZES:
    a, b = byk.get((n,L,"i32")), byk.get((n,L,"i16"))
    if not a or not b: continue
    print("%-12s %7.1f%% %8.1f%% %9.1fs %9.1fs %7.3fx"
          % ("%dx%d"%(n,L), 100*a["share"], 100*b["share"], a["wall"], b["wall"],
             a["wall"]/b["wall"]))
print()
print("If the i32 share stays near 73%, int16's kernel win MUST appear end to end")
print("and clocks are the only remaining explanation. If it falls, the remainder")
print("is the real lever and int16 was never going to move this workload.")""")

# --------------------------------------------------------------------------
md(r"""## 5. Phase B — the Amdahl reconciliation, and the chunk-count confound

Two ratios per size, and the difference between them is the point:

- **natural chunks** — what a user gets. int16 packs more records per chunk, so
  this ratio contains the encoding *and* the extra batch width;
- **forced equal chunks** — both encodings pinned to the same `RNA_GPU_CHUNK`,
  which prices the encoding **alone**.

Then the prediction: from the measured kernel speedup `k` and the measured share
`s`, Amdahl says `1 / ((1 − s) + s/k)`. Comparing that against the measured
end-to-end ratio is the test of whether we understand where the time goes at
all — if they disagree, something outside `modular_decomp` is moving too.""")

code(r"""rowsB = []
for n, L in SIZES:
    fa = "%s/fa/%dx%d.fa" % (ROOT, n, L)
    nat = {t: byk[(n,L,t)] for t in ("i32","i16") if (n,L,t) in byk}
    if len(nat) < 2: continue
    # Force BOTH encodings into the same chunk count: pin records-per-chunk to
    # the whole batch, so neither gets a batch-width advantage over the other.
    fx = {}
    for tag, i16 in (("i32", False), ("i16", True)):
        r = run(fa, int16=i16, chunk=n)
        fx[tag] = r
    k_nat = nat["i32"]["md"] / nat["i16"]["md"]          # kernel-phase speedup
    s     = nat["i32"]["share"]
    pred  = 1.0 / ((1.0 - s) + s / k_nat)
    meas  = nat["i32"]["wall"] / nat["i16"]["wall"]
    meas_fx = fx["i32"]["wall"] / fx["i16"]["wall"]
    rowsB.append(dict(n=n, L=L, s=s, k=k_nat, pred=pred, meas=meas,
                      meas_fixed=meas_fx,
                      ch_i32=nat["i32"]["sweeps"], ch_i16=nat["i16"]["sweeps"],
                      ch_fixed=(fx["i32"]["sweeps"], fx["i16"]["sweeps"])))
    print("  %dx%d: share %.3f  kernel %.3fx  =>  Amdahl %.3fx | measured %.3fx "
          "(natural %d/%d ch)  | fixed-chunk %.3fx %s"
          % (n, L, s, k_nat, pred, meas, nat["i32"]["sweeps"], nat["i16"]["sweeps"],
             meas_fx, fx["i32"]["sweeps"] == fx["i16"]["sweeps"] and "(equal ch)" or "(UNEQUAL!)"))""")

code(r"""print("%-12s %7s %8s %9s %9s %11s %10s" %
      ("size","share","kernel","Amdahl","measured","fixed-chunk","pred-meas"))
for r in rowsB:
    print("%-12s %6.1f%% %7.3fx %8.3fx %8.3fx %10.3fx %+9.1f%%"
          % ("%dx%d"%(r["n"],r["L"]), 100*r["s"], r["k"], r["pred"], r["meas"],
             r["meas_fixed"], 100*(r["pred"]-r["meas"])/r["meas"]))
print()
print("READ IT LIKE THIS")
print("  Amdahl ~= measured        -> we understand the wall clock; the share is the story.")
print("  Amdahl >> measured        -> something OUTSIDE modular_decomp got slower under")
print("                               int16, or the share is smaller than the phase timer says.")
print("  natural >> fixed-chunk    -> most of int16's end-to-end win is the VRAM saving")
print("                               buying batch width, NOT the narrower stream.")""")

# --------------------------------------------------------------------------
md(r"""## 6. Phase C — where the rest of the wall goes

If §4 says the share fell, this says what took its place. Every phase the binary
already times, as a fraction of wall, at the largest size — plus the part no
phase timer covers, which is the interesting residual: chunk setup, teardown,
allocation, backtracking and output.""")

code(r"""n, L = SIZES[-1]
for tag in ("i32", "i16"):
    r = byk.get((n, L, tag))
    if not r: continue
    print("=== %dx%d %s  wall %.1fs ===" % (n, L, tag, r["wall"]))
    tot = 0.0
    for k in ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
              "new_c_host","fml_host","fml_prev_host"):
        v = r["phases"].get(k, 0.0); tot += v
        print("   %-16s %8.2fs  %5.1f%%" % (k, v, 100*v/r["wall"]))
    print("   %-16s %8.2fs  %5.1f%%   <- NOT covered by any phase timer"
          % ("unaccounted", r["wall"]-tot, 100*(r["wall"]-tot)/r["wall"]))
    print()""")

# --------------------------------------------------------------------------
md(r"""## 7. Phase D — roofline, now on whatever §6 actually pointed at

`ncu`, sampled from mid-sweep with the skip **derived from a measured launch
count**. Run it on `modular_decomposition_kernel` for continuity with the first
profile, and on `int_loop_kernel` too, since that kernel has led the wall before
and nothing has profiled it since the batch-width work.""")

code(r"""# (display name, ncu -k pattern).
#
# THE SECOND ENTRY USED TO BE PLAIN "int_loop_kernel" AND MATCHED NOTHING.
# int_loop_kernel_body.inc is #included once per candidate BLOCK_SIZE and
# concatenates the size onto the name, so the real symbols are
# int_loop_kernel_32 / _64 / _128 / _256 -- there is no symbol called
# "int_loop_kernel" at all. ncu matched none of them, the run recorded
# "int_loop_kernel/i32": null, and nobody chased the null for three weeks.
# So the ONE kernel that turned out to be 30% of GPU time has never been
# profiled (STRESS272_RESULTS.md 19).
#
# regex: so it keeps working under RNA_INT_LOOP_BLOCK_SIZE, which selects a
# differently-named instantiation.
KERNELS = [("modular_decomposition_kernel", "modular_decomposition_kernel"),
           ("int_loop_kernel",              "regex:^int_loop_kernel_[0-9]+$")]
METRICS = ",".join([
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "gpu__time_duration.sum"])

PN, PL = 16, 2000                      # small enough for replay, > MIN_GPU_BATCH
pfa = mkfa("%s/fa/prof.fa" % ROOT, PN, PL)

# Measure the launch count instead of hardcoding a --launch-skip. The sweep's
# own "N iterations" is very nearly the number of per-row kernel launches, so
# skipping half of it lands mid-sweep and can never land past the end -- which
# is the other way the first profiling attempt could have reported nothing.
p = subprocess.run([BIN, "--noPS", "-i", pfa], capture_output=True, text=True,
                   env=dict(os.environ, RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1"))
shapes = SWEEP_RE.findall(p.stderr)
assert shapes, "no sweep -- the probe folded on the CPU, nothing to profile"
LAUNCH = max(int(m[0]) for m in shapes)
SKIP, COUNT = max(1, LAUNCH // 2), 5
print("~%d launches; sampling %d from %d" % (LAUNCH, COUNT, SKIP))

PROF = {}
for kern, pattern in KERNELS:
    for tag, i16 in (("i32", False), ("i16", True)):
        key, log = "%s/%s" % (kern, tag), "/content/d_%s_%s.csv" % (kern[:12], tag)
        env = "RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1" + (" RNA_FML_INT16=1" if i16 else "")
        sh("%s ncu --target-processes all -k %s --launch-skip %d --launch-count %d "
           "--metrics %s --csv --log-file %s %s --noPS -i %s > /dev/null 2>&1"
           % (env, pattern, SKIP, COUNT, METRICS, log, BIN, pfa), check=False, quiet=True)
        body = open(log).read() if os.path.exists(log) else ""
        if "No kernels were profiled" in body or "dram__bytes_read" not in body:
            # LOUD. A quiet null here is exactly how int_loop_kernel went
            # three weeks unprofiled while being 30% of GPU time.
            print("  %-40s *** NO KERNELS PROFILED -- the -k pattern %r matched\n"
                  "      nothing, or the sample landed past the end of the sweep.\n"
                  "      THIS IS A BROKEN PROBE, NOT A RESULT. Do not record the\n"
                  "      null and move on." % (key, pattern))
            PROF[key] = None; continue
        rows = list(csv.DictReader(io.StringIO(
            "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
        agg = {}
        for r in rows:
            nm = r.get("Metric Name"); v = (r.get("Metric Value") or "").replace(",","")
            try: agg.setdefault(nm, []).append(float(v))
            except ValueError: pass
        PROF[key] = {k: sum(v)/len(v) for k, v in agg.items()}
        print("  %-40s ok (%d rows)" % (key, len(rows)))""")

code(r"""for kern, _pattern in KERNELS:
    a, b = PROF.get("%s/i32"%kern), PROF.get("%s/i16"%kern)
    if not a or not b: continue
    print("=== %s ===" % kern)
    print("  %-56s %12s %12s %8s" % ("metric","int32","int16","ratio"))
    for k in sorted(set(a) & set(b)):
        r = (b[k]/a[k]) if a[k] else float("nan")
        print("  %-56s %12.3f %12.3f %8.3f" % (k[:56], a[k], b[k], r))
    print()""")

# --------------------------------------------------------------------------
md("## 8. Verdict")

code(r"""print("clocks:", clocks(), " ratio %.2f" % clock_ratio())
print()
if rowsB:
    big = rowsB[-1]
    print("At %dx%d:" % (big["n"], big["L"]))
    print("  modular_decomp share      %.1f%%" % (100*big["s"]))
    print("  kernel speedup (phase)    %.3fx" % big["k"])
    print("  Amdahl prediction         %.3fx" % big["pred"])
    print("  measured, natural chunks  %.3fx  (%d vs %d chunks)"
          % (big["meas"], big["ch_i32"], big["ch_i16"]))
    print("  measured, equal chunks    %.3fx" % big["meas_fixed"])
    print()
    if big["s"] < 0.5:
        print("=> THE SHARE COLLAPSED. int16 cannot move this workload much no matter")
        print("   how good the kernel is. The uncharacterised remainder in §6 is the")
        print("   lever, and bench v5's 1.009x needs no appeal to clocks at all.")
    elif clock_ratio() > 0.9 and big["meas"] > 1.2:
        print("=> THE T4 WAS THE ANOMALY. On a healthy clock the win appears end to end.")
        print("   int16 should default ON for this class of hardware.")
    else:
        print("=> NEITHER cleanly. Compare Amdahl against measured above: a large gap")
        print("   means something outside modular_decomp changed under int16.")

out = dict(date=time.strftime("%Y-%m-%d"), commit=COMMIT, clocks=clocks(),
           clock_ratio=clock_ratio(), sizes=SIZES, phaseA=rowsA, phaseB=rowsB,
           ncu={k: v for k, v in PROF.items()})
with open("/content/profile272_deep.json","w") as f: json.dump(out, f, indent=2)
print("\nwrote /content/profile272_deep.json")""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

dst = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Profile272_deep.ipynb"
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (dst, len(cells)))

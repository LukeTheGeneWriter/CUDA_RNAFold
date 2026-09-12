#!/usr/bin/env python3
"""Build the WarpScan notebook: confirm build threading at scale, A/B the
warp-synchronous scan, and run Step 0 of PORT_WARP_SCAN_SCOPE.md.

WHAT IT IS FOR, in priority order:

  A  BUILD THREADING AT SCALE. RNA_BUILD_THREADS became default-`auto` on
     2026-09-12 on the strength of a 5.12x local measurement and an
     EXTRAPOLATION to the A100 (121.4 s -> ~24 s, wall -44%). That
     extrapolation has never been run. It is the single biggest number in the
     project and it is currently a guess.

  B  THE WARP SCAN AT SCALE. int_loop_warp_kernel is -21.3% on sm_86 and has
     never run on anything else.

  C  CELLS PER BLOCK. RNA_INT_LOOP_BLOCK_SIZE means a DIFFERENT QUANTITY for
     the warp kernel -- cells per block, not threads per cell -- so 64 being
     right for the twin says nothing about it.

  D  STEP 0 of PORT_WARP_SCAN_SCOPE.md: profile the new kernel before
     optimising it. Two questions decided in advance: did `barrier` and
     `short_scoreboard` actually go to zero, and what happened to REGISTERS
     (the design moved two arrays into registers on a kernel already at 50-54
     regs/thread; crossing 64 costs occupancy and would silently eat the win).

THREE STARTUP FAILURES FROM THE LAST NOTEBOOK ARE FIXED HERE. They were found
by Gemini on the Colab side, not by me, and all three are the same shape -- a
step that could not report its own failure:

  1. `apt-get` installed gengetopt/help2man/xxd but NOT libtool, texinfo or
     doxygen, so autogen.sh and the doc build failed. My own project memory
     says "install doxygen BEFORE configure"; I had the note and did not apply
     it.
  2. `/usr/bin/time` is not in the Colab image and the runner used `time -v`
     for RSS. Now installed AND probed, with a fallback, so a missing binary
     degrades to "no RSS" instead of killing every arm.
  3. autogen.sh ran with `check=False` and `> /dev/null 2>&1`. A step that
     cannot fail is not a step. Every build stage is now checked and logged.

AND IT ASSERTS THE DEFAULTS, which is new. A default run must report block size
64 and build threads == nproc, and must NOT report the warp banner. Both knobs
announce themselves on stderr now (the build-threads banner was added for this
-- it was silent for three days while its default disagreed with MERGING.md).

RUNTIME. An A100 arm at 400 x 5601 is ~225 s, much cheaper than the T4's ~525.
A: 4 arms, B: 4, C: 4, D: minutes. About an hour after a ~10 min build.

usage: python3 tools/make_nb_warpscan.py [out.ipynb]
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
md(r"""# WarpScan: confirm build threading at scale, A/B the warp scan, and profile it

## What this run decides

| § | question | why it matters |
|---|---|---|
| **A** | does build threading deliver at 400 × 5601? | `RNA_BUILD_THREADS` went default-`auto` on a **5.12× local** result and an **extrapolation** to −44 % of the A100 wall. The extrapolation has never been run. |
| **B** | does the warp-synchronous scan win on this card? | −21.3 % on sm_86, never run anywhere else. |
| **C** | what cells-per-block is right? | `RNA_INT_LOOP_BLOCK_SIZE` means a **different quantity** for the warp kernel. 64 being right for the twin says nothing about it. |
| **D** | Step 0 — profile the new kernel | Two questions fixed in advance: did `barrier`/`short_scoreboard` go to zero, and **what happened to registers**. |

§A is first because it is the biggest number and the least certain.

## What went wrong last time, and is fixed here

All three failures found on the Colab side had the same shape — **a step that
could not report its own failure**:

1. `apt-get` installed `gengetopt help2man xxd` but not **`libtool texinfo
   doxygen`**, so `autogen.sh` and the doc build failed. The project memory says
   *"install doxygen BEFORE configure"*; the note existed and was not applied.
2. **`/usr/bin/time` is not in the Colab image** and the runner used `time -v`
   for RSS, so every arm died. Now installed *and* probed, with a fallback.
3. `autogen.sh` ran `check=False` with output to `/dev/null`. Every build stage
   is now checked and logged.

## And it asserts the DEFAULTS

A default run must report **block size 64**, **build threads = nproc**, and must
**not** report the warp banner. Both knobs announce themselves on stderr — the
build-threads banner was added for this run, because it was silent for three
days while its default disagreed with `MERGING.md` and no harness could have
caught that.

## Rules

* **`sha` must be `7c0b3d633281` in every arm.** A lever that changes the answer
  is a bug and a bigger finding than any timing.
* **`sweep shape:` must be present**, or the run folded on the CPU.
* **Report the control.** `modular_decomp` cannot be reached by
  `RNA_INT_LOOP_WARP`; `int_loop` cannot be reached by `RNA_BUILD_THREADS`. If a
  control moves, the device moved and the measurement is not one.
* **ABBA, never ABAB** — a monotone drift cancels.
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""
import subprocess, os, sys, json, time, re, random, io, collections
import csv as _csvmod

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
              "clocks_throttle_reasons.active,memory.total --format=csv,noheader",
              quiet=True).stdout.strip()

NPROC = int(sh("nproc", quiet=True).stdout.strip())
print(clocks())
print("cores   :", NPROC, "   <-- RNA_BUILD_THREADS=auto resolves to this")
print("host RAM:", sh("free -g | awk '/^Mem/{print $2\" GB\"}'", quiet=True).stdout.strip())
""")

md(r"""### Dependencies, all of them

`libtool`, `texinfo` and `doxygen` are the three that were missing last time.
`time` is probed rather than assumed — a missing `/usr/bin/time` used to kill
every arm.""")

code(r"""
sh("apt-get -qq update > /dev/null 2>&1", check=False, quiet=True)
sh("apt-get -qq install -y gengetopt help2man xxd libtool texinfo doxygen time "
   "> /dev/null 2>&1", check=False, quiet=True)

MISSING = [t for t in ("gengetopt", "help2man", "xxd", "libtoolize", "makeinfo", "doxygen")
           if sh("command -v %s" % t, check=False, quiet=True).returncode != 0]
print("missing build tools:", MISSING or "none")
if MISSING:
    raise SystemExit("install failed for %s -- autogen or make WILL fail" % MISSING)

TIME_BIN = "/usr/bin/time -v " if os.path.exists("/usr/bin/time") else ""
print("/usr/bin/time:", "present" if TIME_BIN else "ABSENT (RSS will read 0, arms still run)")
""")

# --------------------------------------------------------------------------
md(r"""## 2. Build — every stage checked, and the right branch

A feature probe per knob this run depends on. Probing the **source** rather than
a commit hash: a hash goes stale the moment the branch is rebased, and what
matters is whether the code is there.""")

code(r"""
REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/warpscan"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())

SRC = ROOT + "/port27/src/"
NEED = [
    ("int_loop_warp_kernel",        SRC+"ViennaRNA/mfe/cuda/int_loop.cu", "the warp scan (SS B, C, D)"),
    ("RNA_INT_LOOP_WARP",           SRC+"ViennaRNA/mfe/cuda/int_loop.cu", "its gate"),
    ("INT_LOOP_DEFAULT_BLOCK_SIZE", SRC+"ViennaRNA/mfe/cuda/int_loop.cu", "block size 64 promoted"),
    ("RNA_LAUNCH_STATS",            SRC+"ViennaRNA/mfe/cuda/device.cu",   "per-launch device time"),
    ("build threads %d",            SRC+"bin/RNAfold.c",                  "the build-threads BANNER (SS A)"),
    ("d_span_H",                    SRC+"ViennaRNA/mfe/cuda/hp_mb_loop.cu", "--maxBPspan per record"),
]
missing = []
for tok, path, why in NEED:
    ok = os.path.exists(path) and tok in open(path).read()
    print("  %-30s %-34s %s" % (tok, why, "present" if ok else "MISSING"))
    if not ok: missing.append(tok)
if missing:
    raise SystemExit("STALE CLONE: no %s. Push port27, then re-run." % ", ".join(missing))
""")

code(r"""
t0 = time.time()
for pat, flag in (("src/dlib-*.tar.bz2", "-xjf"), ("src/libsvm-*.tar.gz", "-xzf")):
    for t in sh("ls %s/port27/%s 2>/dev/null" % (ROOT, pat), check=False, quiet=True).stdout.split():
        d = re.sub(r"\.tar\.(bz2|gz)$", "", os.path.basename(t))
        if not os.path.isdir("%s/port27/src/%s" % (ROOT, d)):
            sh("tar %s %s -C %s/port27/src/" % (flag, t, ROOT), check=False, quiet=True)

# EVERY STAGE CHECKED. autogen used to run with check=False into /dev/null,
# which is how a failed build reached the arms and read as "no effect".
for name, cmd in (
    ("autogen", "cd %s/port27 && ./autogen.sh > /content/autogen.log 2>&1" % ROOT),
):
    p = sh(cmd, check=False, quiet=True)
    if p.returncode:
        print(sh("tail -40 /content/%s.log" % name, quiet=True).stdout)
        raise SystemExit("%s failed" % name)
    print("%s ok" % name)

sh("chmod +x %s/port27/doc/man2rst.py" % ROOT, check=False, quiet=True)

p = sh("cd %s/port27 && ./configure --without-python --without-perl --without-swig "
       "--without-doc --without-rnaxplorer --without-forester --without-kinfold "
       "--without-rnalocmin --enable-cuda CFLAGS='-g -O2' CXXFLAGS='-g -O2' "
       "PYTHON3=\"$(command -v python3)\" > /content/conf.log 2>&1" % ROOT,
       check=False, quiet=True)
if p.returncode:
    print(sh("tail -40 /content/conf.log", quiet=True).stdout); raise SystemExit("configure failed")
print("configure ok")

p = sh("cd %s/port27 && make -j$(nproc) > /content/make.log 2>&1" % ROOT, check=False, quiet=True)
if p.returncode:
    print(sh("grep -iE 'error' /content/make.log | head -30", quiet=True).stdout)
    raise SystemExit("make failed")
BIN = ROOT + "/port27/src/bin/RNAfold"
assert os.path.exists(BIN), "make reported success but produced no binary"
print("built in %.0fs" % (time.time() - t0))
""")

md(r"""### The defaults must be what the docs say

This is the check that would have caught `RNA_BUILD_THREADS` shipping serial
while `MERGING.md` said `auto`.""")

code(r"""
p = subprocess.run([BIN, "--noPS"], input=">t\nGGGAAACCCUUUGGGAAACCC\n",
                   capture_output=True, text=True,
                   env=dict(os.environ, RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1"))
err = p.stderr
want_bs   = "int_loop_kernel block size 64"
want_bt   = "build threads %d (auto: nproc, the default)" % NPROC
bad_warp  = "RNA_INT_LOOP_WARP=1"
print("  block size 64 by default :", "OK" if want_bs in err else "*** NO: %r" %
      (re.search(r"int_loop_kernel block size \d+.*", err) or "absent"))
print("  build threads = nproc    :", "OK" if want_bt in err else "*** NO: %r" %
      (re.search(r"build threads .*", err) or "absent"))
print("  warp kernel OFF by default:", "OK" if bad_warp not in err else "*** NO, it is ON")
assert want_bs in err and want_bt in err and bad_warp not in err, \
    "defaults are not what this notebook assumes; stop and read the banners above"
""")

# --------------------------------------------------------------------------
md(r"""## 3. Workload — the same 400 × 5601 as every run since §14

Same seed, so `sha` is comparable with everything from `STRESS272_RESULTS.md`
§14 onward.""")

code(r"""
N_BIG, LEN, SEED = 400, 5601, 20260907
os.makedirs(ROOT + "/fa", exist_ok=True)
BIG = ROOT + "/fa/big.fa"
random.seed(SEED)
with open(BIG, "w") as f:
    for i in range(N_BIG):
        f.write(">u%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(LEN))))
print("%d x %d nt = %.1f MB" % (N_BIG, LEN, os.path.getsize(BIG)/1e6))
EXPECT_SHA = "7c0b3d633281"
""")

# --------------------------------------------------------------------------
md(r"""## 4. The runner

Asserts, before any number is believed: the sweep ran, every knob asked for
actually engaged, and the answer did not move.""")

code(r"""
PHASE_RE = re.compile(
    r"int_loop=([\d.]+) hp_mb=([\d.]+) load_my_c=([\d.]+) modular_decomp=([\d.]+) "
    r"fetch_mx=([\d.]+) \| new_c_host=([\d.]+) fml_host=([\d.]+) fml_prev_host=([\d.]+)")
STAGE_RE = re.compile(
    r"build=([\d.]+) prepare=([\d.]+) prefill=([\d.]+) backtrack=([\d.]+) "
    r"output=([\d.]+) gpuinit=([\d.]+) teardown=([\d.]+) free=([\d.]+)")
SWEEP_RE = re.compile(r"sweep shape: (\d+) iterations, (\d+) active record-rows, "
                      r"(\d+) cells; peak/iteration (\d+) records (\d+) cells")
BS_RE    = re.compile(r"int_loop_kernel block size (\d+)")
BT_RE    = re.compile(r"build threads (\d+)")
PHASES = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
          "new_c_host","fml_host","fml_prev_host")
STAGES = ("build","prepare","prefill","backtrack","output","gpuinit","teardown","free")

RESULTS = {}
OUT = "/content/warpscan.json"
os.makedirs("/content/clk", exist_ok=True)

def save():
    with open(OUT, "w") as f:
        json.dump({"commit": COMMIT, "clocks": clocks(), "nproc": NPROC,
                   "n": N_BIG, "len": LEN, "runs": RESULTS}, f, indent=1)

class ClockSampler(object):
    FIELDS = ("clocks.sm","clocks.mem","temperature.gpu","power.draw",
              "utilization.gpu","utilization.memory","clocks_throttle_reasons.active")
    def __init__(self, path): self.path = path; self.p = None
    def __enter__(self):
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=" + ",".join(self.FIELDS),
             "--format=csv,noheader,nounits", "-lms", "250"],
            stdout=self.f, stderr=subprocess.DEVNULL)
        return self
    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            try: self.p.wait(timeout=5)
            except Exception: self.p.kill()
        self.f.close(); return False
    def summary(self):
        rows = []
        for line in open(self.path):
            q = [x.strip() for x in line.split(",")]
            if len(q) != len(self.FIELDS): continue
            try: rows.append((float(q[0]), float(q[2]), float(q[3]), float(q[4]), q[6]))
            except ValueError: continue
        if not rows: return {}
        busy = [r for r in rows if r[3] >= 50.0] or rows
        thr = {}
        for r in busy: thr[r[4]] = thr.get(r[4], 0) + 1
        return dict(n=len(rows), n_busy=len(busy),
                    sm_mean=sum(r[0] for r in busy)/len(busy),
                    temp_max=max(r[1] for r in busy),
                    power_mean=sum(r[2] for r in busy)/len(busy),
                    throttle=sorted(thr.items(), key=lambda kv: -kv[1])[:2])

def run(tag, int16=False, chunk_cap=29, block_size=None, warp=False,
        build_threads=None, phase_sync=True, fa=None, extra_args=""):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH","RNA_PHASE_SYNC",
              "RNA_GPU_VRAM_BUDGET_MB","RNA_INT_LOOP_BLOCK_SIZE","RNA_BUILD_PIPELINE",
              "RNA_LAUNCH_STATS","RNA_INT_LOOP_WARP","RNA_BUILD_THREADS","RNA_MD_SMEM"):
        env.pop(k, None)
    env["RNA_GPU_CHUNK"] = str(chunk_cap); env["RNA_MIN_GPU_BATCH"] = "1"
    if int16:          env["RNA_FML_INT16"] = "1"
    if phase_sync:     env["RNA_PHASE_SYNC"] = "1"
    if block_size:     env["RNA_INT_LOOP_BLOCK_SIZE"] = str(block_size)
    if warp:           env["RNA_INT_LOOP_WARP"] = "1"
    if build_threads is not None: env["RNA_BUILD_THREADS"] = str(build_threads)

    clk = "/content/clk/%s.csv" % tag
    cmd = (TIME_BIN + BIN + " --noPS " + extra_args + " -i " + (fa or BIG)).split()
    t0 = time.time()
    with ClockSampler(clk) as sampler:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.time() - t0
        clkinfo = sampler.summary()
    err = p.stderr

    if p.returncode != 0:
        print(err[-3000:]); raise SystemExit("%s: rc=%d" % (tag, p.returncode))
    if "sweep shape:" not in err:
        raise SystemExit("%s: NO 'sweep shape:' -- folded on the CPU" % tag)
    for want, token, label in ((phase_sync,"RNA_PHASE_SYNC=1","phase sync"),
                               (int16,"RNA_FML_INT16=1","int16"),
                               (warp,"RNA_INT_LOOP_WARP=1","warp kernel")):
        if bool(want) != (token in err):
            raise SystemExit("%s: %s knob did not engage as asked" % (tag, label))
    if block_size:
        g = BS_RE.search(err)
        if (not g) or int(g.group(1)) != block_size:
            raise SystemExit("%s: block size %s asked, %s reported" % (tag, block_size, g and g.group(1)))
    if build_threads is not None:
        g = BT_RE.search(err)
        if (not g) or int(g.group(1)) != int(build_threads):
            raise SystemExit("%s: build threads %s asked, %s reported" % (tag, build_threads, g and g.group(1)))

    ph = dict(zip(PHASES,[float(x) for x in PHASE_RE.search(err).groups()])) if PHASE_RE.search(err) else {}
    st = dict(zip(STAGES,[float(x) for x in STAGE_RE.search(err).groups()])) if STAGE_RE.search(err) else {}
    shapes = SWEEP_RE.findall(err)
    rss = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
    if m: rss = int(m.group(1))/1e6
    sha = __import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12]
    bt = BT_RE.search(err)
    r = dict(wall=wall, phases=ph, stages=st, chunks=len(shapes),
             cells=sum(int(s[2]) for s in shapes), warp=warp, int16=int16,
             chunk_cap=chunk_cap, block_size=int(BS_RE.search(err).group(1)) if BS_RE.search(err) else None,
             build_threads=int(bt.group(1)) if bt else None,
             rss=rss, clock=clkinfo, sha=sha)
    RESULTS[tag] = r; save()
    warn = "" if sha == EXPECT_SHA else "   <<< SHA MOVED"
    print("  %-16s build %7.2f  int_loop %7.2f  md %7.2f  wall %7.1f  %4.0f MHz  sha %s%s"
          % (tag, st.get("build",0), ph.get("int_loop",0), ph.get("modular_decomp",0),
             wall, clkinfo.get("sm_mean",0), sha, warn))
    return r

print("runner ready ->", OUT)
""")

# --------------------------------------------------------------------------
md(r"""## A. Build threading at scale — the number that is currently a guess

`build` was **121.4 s of a 224.9 s A100 wall** (§23.5) and the knob went
default-`auto` on a 5.12× *local* measurement. The extrapolation says ~24 s and
a −44 % wall. This runs it.

**`int_loop` is the control** — `RNA_BUILD_THREADS` cannot reach it.

ABBA: serial, auto, auto, serial.""")

code(r"""
print("A: RNA_BUILD_THREADS at 400 x 5601, ABBA (int_loop is the control)")
for tag, bt in (("A_serial_1", 1), ("A_auto_1", "auto"),
                ("A_auto_2", "auto"), ("A_serial_2", 1)):
    run(tag, build_threads=bt)
""")

code(r"""
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
ser = [v for k,v in RESULTS.items() if k.startswith("A_serial")]
aut = [v for k,v in RESULTS.items() if k.startswith("A_auto")]
if ser and aut:
    print("%-26s %10s %10s %9s" % ("", "serial", "auto(%d)" % NPROC, "delta"))
    def row(lab, f, fmt="%10.2f"):
        x, y = mean([f(v) for v in ser]), mean([f(v) for v in aut])
        print(("%-26s "+fmt+" "+fmt+" %8.1f%%") % (lab, x, y, 100.0*(y-x)/x if x else 0))
    row("build (s)",              lambda v: v["stages"]["build"])
    row("int_loop (CONTROL)",     lambda v: v["phases"]["int_loop"])
    row("modular_decomp",         lambda v: v["phases"]["modular_decomp"])
    row("backtrack",              lambda v: v["stages"]["backtrack"])
    row("WALL (s)",               lambda v: v["wall"], "%10.1f")
    row("peak RSS (GB)",          lambda v: v["rss"], "%10.2f")
    b0, b1 = mean([v["stages"]["build"] for v in ser]), mean([v["stages"]["build"] for v in aut])
    print()
    print("  build speedup            %.2fx on %d cores" % (b0/b1 if b1 else 0, NPROC))
    print("  predicted (local 5.12x)  %.1f s   measured %.1f s" % (b0/5.12, b1))
    print("  shas:", set(v["sha"] for v in ser+aut))
""")

# --------------------------------------------------------------------------
md(r"""## B. The warp-synchronous scan at scale

−21.3 % on sm_86, never run anywhere else. **`modular_decomp` is the control** —
`RNA_INT_LOOP_WARP` cannot reach it.

Both arms keep build threading on (the new default), so this measures the kernel
against the configuration people will actually run.""")

code(r"""
print("B: RNA_INT_LOOP_WARP at 400 x 5601, ABBA (modular_decomp is the control)")
for tag, w in (("B_twin_1", False), ("B_warp_1", True),
               ("B_warp_2", True),  ("B_twin_2", False)):
    run(tag, warp=w)
""")

code(r"""
tw = [v for k,v in RESULTS.items() if k.startswith("B_twin")]
wp = [v for k,v in RESULTS.items() if k.startswith("B_warp")]
if tw and wp:
    for lab, key in (("int_loop (the LEVER)","int_loop"),
                     ("modular_decomp (CONTROL)","modular_decomp"),
                     ("hp_mb","hp_mb")):
        x, y = mean([v["phases"][key] for v in tw]), mean([v["phases"][key] for v in wp])
        print("  %-26s twin %8.2f  warp %8.2f  %+7.2f%%" % (lab, x, y, 100.0*(y-x)/x))
    x, y = mean([v["wall"] for v in tw]), mean([v["wall"] for v in wp])
    print("  %-26s twin %8.1f  warp %8.1f  %+7.2f%%" % ("wall", x, y, 100.0*(y-x)/x))
    print("  clocks: twin %.0f MHz  warp %.0f MHz"
          % (mean([v["clock"].get("sm_mean",0) for v in tw]),
             mean([v["clock"].get("sm_mean",0) for v in wp])))
    shas = set(v["sha"] for v in tw+wp)
    print("  shas:", shas, "" if shas == {EXPECT_SHA} else "  <<< MOVED")
""")

# --------------------------------------------------------------------------
md(r"""## C. Cells per block — a different quantity from the twin's threads per cell

For the twin, `RNA_INT_LOOP_BLOCK_SIZE` is **threads cooperating on one cell**
and 64 is the measured optimum. For the warp kernel it is **32 × cells per
block**, and the cells are independent. 64 being right for one says nothing
about the other, so sweep it.""")

code(r"""
print("C: cells-per-block for the warp kernel (32/64/128/256 threads = 1/2/4/8 cells)")
for bs in (32, 64, 128, 256):
    run("C_warp_bs%d" % bs, warp=True, block_size=bs)
""")

code(r"""
rows = sorted((v["block_size"], k) for k, v in RESULTS.items() if k.startswith("C_warp"))
if rows:
    base = None
    print("  %6s %8s %12s %12s %9s %9s" % ("threads","cells","int_loop","mod_decomp","wall","MHz"))
    for bs, k in rows:
        v = RESULTS[k]
        if base is None: base = v["phases"]["int_loop"]
        print("  %6d %8d %12.2f %12.2f %9.1f %9.0f   int_loop %+.1f%%"
              % (bs, bs//32, v["phases"]["int_loop"], v["phases"]["modular_decomp"],
                 v["wall"], v["clock"].get("sm_mean",0),
                 100.0*(v["phases"]["int_loop"]-base)/base))
    print()
    print("  The twin turns over at 64 because __syncthreads() becomes a real barrier")
    print("  past one warp per block. The warp kernel has NO barrier, so if it keeps")
    print("  improving past 64 that is the mechanism confirming itself.")
""")

# --------------------------------------------------------------------------
md(r"""## D. Step 0 — profile the new kernel before optimising it

`PORT_WARP_SCAN_SCOPE.md` §0. Two questions decided **in advance**:

1. **Did `barrier` and `short_scoreboard` actually go to zero?** If not, the
   −21.3 % came from something other than the intended mechanism, and the design
   rationale is wrong even though the number is right.
2. **What happened to registers?** The design moved `col_mask`/`prefix` *into*
   registers on a kernel already at 50–54 regs/thread. **Crossing 64 costs
   occupancy** and would silently eat part of the win.

Twin and warp profiled at the same block size, so the comparison is like for
like.""")

code(r"""
SECT = "--section WarpStateStats --section SchedulerStats --section Occupancy"
EXTRA = ",".join([
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks","launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem","launch__occupancy_limit_warps",
    "launch__registers_per_thread","launch__shared_mem_per_block_allocated",
    "l1tex__t_sector_hit_rate.pct","lts__t_sector_hit_rate.pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
])
REASONS = ("barrier","branch_resolving","dispatch_stall","drain","imc_miss",
           "lg_throttle","long_scoreboard","math_pipe_throttle","membar",
           "mio_throttle","misc","no_instruction","not_selected","selected",
           "short_scoreboard","sleeping","tex_throttle","wait")
FALLBACK = ",".join("smsp__average_warps_issue_stalled_%s_per_issue_active.ratio" % r
                    for r in REASONS) + "," + EXTRA

PROF_N, PROF_LEN, PROF_CAP, SKIP, COUNT = 60, 1800, 24, 140, 3
PFA = "%s/fa/prof.fa" % ROOT
if not os.path.exists(PFA):
    random.seed(90211)
    with open(PFA, "w") as f:
        for i in range(PROF_N):
            f.write(">s%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(PROF_LEN))))

STALLS = {}
def profile(tag, kernel_regex, bs, warp):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_PHASE_SYNC","RNA_INT_LOOP_WARP","RNA_MD_SMEM"):
        env.pop(k, None)
    env.update(RNA_GPU_CHUNK=str(PROF_CAP), RNA_MIN_GPU_BATCH="1",
               RNA_INT_LOOP_BLOCK_SIZE=str(bs))
    if warp: env["RNA_INT_LOOP_WARP"] = "1"
    out = "/content/ncu_%s.csv" % tag
    for flavour, extra in (("sections", SECT + " --metrics " + EXTRA),
                           ("explicit", "--metrics " + FALLBACK)):
        cmd = ("ncu --target-processes all --csv %s -k regex:'%s' --launch-skip %d "
               "--launch-count %d --log-file %s %s --noPS -i %s > /dev/null 2>&1"
               % (extra, kernel_regex, SKIP, COUNT, out, BIN, PFA))
        subprocess.run(cmd, shell=True, env=env)
        body = open(out).read() if os.path.exists(out) else ""
        if ("No kernels were profiled" in body) or ("Metric Name" not in body):
            continue
        rows = list(_csvmod.DictReader(io.StringIO(
            "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
        want = kernel_regex.strip("^$")
        for ch in "[(.*+?|{": want = want.split(ch)[0]
        names = set(r.get("Kernel Name","") for r in rows)
        if names and not any(want in nm for nm in names):
            print("  %-14s *** WRONG KERNEL: %r ***" % (tag, sorted(names)[:1])); return {}
        agg = collections.defaultdict(list)
        for r in rows:
            try: agg[r["Metric Name"]].append(float((r["Metric Value"] or "").replace(",","")))
            except Exception: pass
        res = {k: sum(v)/len(v) for k, v in agg.items() if v}
        if any("issue_stalled" in k for k in res):
            print("  %-14s ok via %s" % (tag, flavour)); STALLS[tag] = res; return res
    print("  %-14s *** NO STALL METRICS -- broken probe, not a result ***" % tag)
    return {}

print("D: like-for-like profile, twin vs warp")
for tag, rx, bs, w in (("twin_bs64",  "^int_loop_kernel_64$",      64,  False),
                       ("warp_bs64",  "^int_loop_warp_kernel",     64,  True),
                       ("warp_bs128", "^int_loop_warp_kernel",     128, True)):
    profile(tag, rx, bs, w)
with open("/content/warpscan_stalls.json","w") as f: json.dump(STALLS, f, indent=1)
""")

code(r"""
if not STALLS:
    print("No stall data -- nothing below is a result.")
else:
    tags = [t for t in ("twin_bs64","warp_bs64","warp_bs128") if t in STALLS]
    print("--- occupancy, registers, and the binding limiter ---")
    print("%-12s %10s %11s %9s %9s   %s"
          % ("kernel","occupancy","duration us","regs/thr","smem B","blocks/regs/smem/warps"))
    for t in tags:
        r = STALLS[t]
        lims = " / ".join(str(int(r.get("launch__occupancy_limit_"+k,0)))
                          for k in ("blocks","registers","shared_mem","warps"))
        print("%-12s %9.1f%% %11.1f %9d %9d   %s"
              % (t, r.get("sm__warps_active.avg.pct_of_peak_sustained_active",0),
                 r.get("gpu__time_duration.sum",0)/1e3,
                 int(r.get("launch__registers_per_thread",0)),
                 int(r.get("launch__shared_mem_per_block_allocated",0)), lims))
    print()
    print("  QUESTION 2: if regs/thr crossed 64 on the warp kernel, the register")
    print("  limit will be the smallest of the four and part of the win is being")
    print("  paid back in occupancy.")
    print()
    keys = sorted({k for r in STALLS.values() for k in r if "issue_stalled" in k})
    tot = {t: sum(STALLS[t].get(k,0.0) for k in keys) for t in tags}
    def short(k):
        m = re.search(r"issue_stalled_(.+?)_per_issue_active", k); return m.group(1) if m else k
    order = sorted(keys, key=lambda k: -max(STALLS[t].get(k,0.0) for t in tags))
    print("--- warp issue stalls, %% of total stall cycles per issue ---")
    print("%-20s" % "stall reason" + "".join("%13s" % t for t in tags))
    for k in order:
        vals = [STALLS[t].get(k,0.0) for t in tags]
        if max(vals) < 0.005: continue
        print("%-20s" % short(k) + "".join("%12.1f%%" % (100.0*v/tot[t] if tot[t] else 0)
                                           for t, v in zip(tags, vals)))
    print("%-20s" % "(total cycles/issue)" + "".join("%13.2f" % tot[t] for t in tags))
    print()
    print("  QUESTION 1: barrier and short_scoreboard should be ~0 on the warp rows.")
    print("  If they are not, the -21.3% came from somewhere other than the design.")
""")

# --------------------------------------------------------------------------
md("## E. Summary and export")

code(r"""
print("commit", COMMIT, "| cores", NPROC)
print(clocks())
shas = sorted(set(v["sha"] for v in RESULTS.values() if isinstance(v, dict) and "sha" in v))
print("shas:", shas, "->", "OK" if shas == [EXPECT_SHA] else "*** SOMETHING MOVED ***")
print()
print("%-16s %8s %9s %9s %9s %6s %5s" %
      ("arm","build","int_loop","mod_dec","wall","MHz","sha ok"))
for k in sorted(RESULTS):
    v = RESULTS[k]
    if not isinstance(v, dict) or "phases" not in v: continue
    print("%-16s %8.2f %9.2f %9.2f %9.1f %6.0f %5s"
          % (k, v["stages"].get("build",0), v["phases"].get("int_loop",0),
             v["phases"].get("modular_decomp",0), v["wall"],
             v["clock"].get("sm_mean",0), v["sha"] == EXPECT_SHA))
""")

code(r"""
from google.colab import files
sh("cd /content && tar czf warpscan_artifacts.tar.gz warpscan.json warpscan_stalls.json "
   "clk ncu_*.csv 2>/dev/null", check=False, quiet=True)
for f in ("warpscan.json", "warpscan_stalls.json", "warpscan_artifacts.tar.gz"):
    p = "/content/" + f
    if os.path.exists(p):
        print("%-30s %8.1f KB" % (f, os.path.getsize(p)/1024))
        files.download(p)
""")

md(r"""### The decision table, written before the run

| § | reading | means |
|---|---|---|
| **A** | build ≈ 24 s, wall ≈ 127 s | the extrapolation holds; `build` stops being the headline |
| **A** | build speedup ≪ 5× | the A100 host is not this laptop — re-derive `auto` per host |
| **A** | `int_loop` control moved | the device moved; the arm is not a measurement |
| **B** | `int_loop` −15…−25 %, control flat | the warp scan generalises off sm_86 |
| **B** | no win | it is an sm_86 artefact — keep it gated and say so |
| **C** | keeps improving past 64 | no barrier, so occupancy scales — the mechanism confirming itself |
| **C** | turns over at 64 like the twin | something else limits it; profile before tuning |
| **D** | `barrier` ≈ 0, `short_scoreboard` ≈ 0 | the design did what it claimed |
| **D** | regs/thr > 64, register limit binding | part of the win is being paid back in occupancy — **fix before promoting** |
""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_WarpScan.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (out, len(cells)))

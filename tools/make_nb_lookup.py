#!/usr/bin/env python3
"""Build the Lookup notebook: A/B the two cell -> record lookup changes on an
A100, and then hold the card under sustained load to see what it does.

WHAT IT IS FOR, in priority order:

  A  H6 AND H7 ON A UNIFORM WORKLOAD. Both are -7.5% / -3.7% on sm_86 and have
     never run anywhere else. Four arms: today's binary search, H7's 32-ary
     warp search, H6's 2-D grid, and both together.

  B  H6 AND H7 ON A RAGGED WORKLOAD. This is the arm that decides whether H7
     has a reason to exist. H6's waste guard DECLINES a ragged chunk, so if
     ragged chunks are common H7 carries the case alone -- and if the chunker's
     length sort makes every chunk uniform in practice, H7 is redundant and
     should be said so. The notebook does not assume which; it reports the
     decision trace.

  C  MECHANISM. NCU on the three variants: long_scoreboard should fall with the
     chain depth, and registers must not cross the limit. Plus a toolkit
     register census, because the same source reports 48 regs/thread on local
     nvcc 12.4 and 58 on Colab, and nobody has established which toolkit that
     is or why.

  D  STRESS. Four ways to load the card and watch what gives:
       D1  sustained -- the same fold six times back to back, clocks and power
           sampled throughout. The T4 answer was a POWER CAP at 70 W; the A100
           showed 0x0 throttling on a ~2 minute run, which is not the same as
           surviving twenty.
       D2  scale -- record count and sequence length swept, to see whether the
           phase mix holds or something goes superlinear.
       D3  VRAM pressure -- the budget squeezed until the sweep is many chunks.
           `project_chunking_costs_batch_width` says 0.6%, not k x, measured at
           400 x 5601 on a T4. Never checked on a card with 40 GB.
       D4  contention -- two folds at once on one GPU. The question is not
           speed, it is whether the ANSWER survives; a wrong answer under
           contention is a far bigger finding than a slow one.

WHAT THIS NOTEBOOK ASSERTS BEFORE BELIEVING ANYTHING

  * `sweep shape:` present, or the run folded on the CPU.
  * every knob asked for announced itself on stderr.
  * FOR H6, THE DECISION AND NOT THE KNOB. RNA_INT_LOOP_GRIDY=1 only says what
    was asked; the host's waste guard decides. The first bar for H6 came back
    GREEN over ten option arms having never once taken the 2-D path, because
    the fixture was ragged. The kernel prints on every CHANGE of decision and
    this notebook parses that trace.
  * the defaults are what the docs say -- and they CHANGED on 2026-09-12: the
    warp kernel is now the default and its block size is 32, not 64.
  * sha constant within a fixture.

usage: python3 tools/make_nb_lookup.py [out.ipynb]
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
md(r"""# Lookup: two ways to stop searching, and an A100 held under load

## What this run decides

| § | question | why it matters |
|---|---|---|
| **A** | do H6 and H7 win on this card, on a uniform workload? | −7.5 % and −3.7 % on sm_86, never run anywhere else. Both are **gated off** waiting on exactly this. |
| **B** | **does H7 have a reason to exist?** | H6's waste guard declines ragged chunks. If real chunks are ragged, H7 carries that case alone. If the length sort makes them uniform, H7 is redundant — and that should be said. |
| **C** | is the mechanism what we claim? | `long_scoreboard` should fall with chain depth. Plus: the same source reports **48 regs/thread locally and 58 on Colab** and nobody knows why. |
| **D** | what does an A100 do under sustained load? | Every A100 number so far came from ~2-minute runs showing `0x0` throttling. That is not the same as surviving twenty minutes. |

## The change being tested

Every warp used to open by asking *which record is this cell in?* —
`flatten_index_to_H()`, a binary search over `size_off_H[]` that is **12 SASS
instructions carrying one `LDG` per iteration**, `ceil(log2(nfiles))` times,
with every probe address depending on the previous probe's value.

| | how | chain | sm_86 |
|---|---|---|---|
| today | binary search | 9 dependent loads | — |
| **H7** `RNA_INT_LOOP_WSEARCH=1` | 32 lanes probe 32 points | **2** | **−3.7 %** |
| **H6** `RNA_INT_LOOP_GRIDY=1` | `blockIdx.y` *is* the record | **0** | **−7.5 %** |

They are **complements**. H6 needs near-uniform widths and `nfiles <= 65535`;
the host declines it otherwise, and then H7 is the only thing that helps.

## Rules

* **`sha` must be constant within a fixture.** A lookup that changes the answer
  is a bug and a bigger finding than any timing.
* **`sweep shape:` must be present**, or the run folded on the CPU.
* **Assert the DECISION, not the knob.** `RNA_INT_LOOP_GRIDY=1` says what was
  *asked*; the waste guard decides. H6's first bar was GREEN over ten option
  arms having never taken the 2-D path.
* **Report the control.** `modular_decomp` cannot be reached by either lookup
  knob. If it moves, the device moved and the measurement is not one.
* **Palindromic order, never ABAB** — a monotone drift cancels.
* **Warm up first.** The local box needed *four* discarded folds to reach steady
  clocks; one was not enough and produced a 2.84 → 5.25 s ramp that looked like
  a result.
* **Preflight every configuration on 8 × 300 nt before running it long.**
  `RNA_GPU_CHUNK` is the *master switch*, not a cap — unset means **fold on the
  CPU**, which at 400 × 5601 is an hour per arm that looks exactly like a hang.
  The `sweep shape:` guard catches it, but only *after* the fold.
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
print("nvcc    :", sh("nvcc --version | tail -2 | head -1", quiet=True).stdout.strip())
""")

md(r"""### Dependencies, all of them

`libtool`, `texinfo` and `doxygen` are the three that once went missing and took
the whole build with them. `time` is probed rather than assumed.""")

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

A feature probe per knob this run depends on. Probing the **source**, not a
commit hash: a hash goes stale the moment the branch is rebased, and what
matters is whether the code is there.""")

code(r"""
REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/lookup"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())

SRC = ROOT + "/port27/src/"
IL  = SRC + "ViennaRNA/mfe/cuda/int_loop.cu"
NEED = [
    ("int_loop_warp_kernel",              IL, "the warp kernel (now the default)"),
    ("INT_LOOP_WARP_DEFAULT_BLOCK_SIZE",  IL, "its own block size, 32"),
    ("RNA_INT_LOOP_GRIDY",                IL, "H6, the 2-D grid (SS A, B)"),
    ("RNA_INT_LOOP_WSEARCH",              IL, "H7, the 32-ary search (SS A, B)"),
    ("flatten_index_to_H_warp",           IL, "H7's lookup itself"),
    ("waste guard declined",              IL, "the DECISION probe SS B depends on"),
    ("build threads %d",                  SRC+"bin/RNAfold.c", "the build-threads banner"),
]
missing = []
for tok, path, why in NEED:
    ok = os.path.exists(path) and tok in open(path).read()
    print("  %-34s %-38s %s" % (tok, why, "present" if ok else "MISSING"))
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

p = sh("cd %s/port27 && ./autogen.sh > /content/autogen.log 2>&1" % ROOT,
       check=False, quiet=True)
if p.returncode:
    print(sh("tail -40 /content/autogen.log", quiet=True).stdout)
    raise SystemExit("autogen failed")
print("autogen ok")

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

md(r"""### The defaults must be what the docs say — and they CHANGED on 2026-09-12

The warp kernel was promoted to default and **its block size is 32, not 64**.
A notebook that still asserts 64 is asserting last week's tree.""")

code(r"""
p = subprocess.run([BIN, "--noPS"], input=">t\nGGGAAACCCUUUGGGAAACCC\n",
                   capture_output=True, text=True,
                   env=dict(os.environ, RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1"))
err = p.stderr
checks = [
    ("warp kernel IS the default",  "int_loop kernel: warp-per-cell (the measured default)" in err),
    ("its block size is 32",        "int_loop_kernel block size 32" in err),
    ("build threads = nproc",       "build threads %d (auto: nproc, the default)" % NPROC in err),
    ("H6 OFF by default",           "RNA_INT_LOOP_GRIDY=1" not in err),
    ("H7 OFF by default",           "RNA_INT_LOOP_WSEARCH=1" not in err),
]
for label, ok in checks:
    print("  %-28s %s" % (label, "OK" if ok else "*** NO"))
if not all(ok for _, ok in checks):
    print("\n--- banners actually printed ---")
    for ln in err.splitlines():
        if "int_loop" in ln or "build threads" in ln: print("   ", ln)
    raise SystemExit("defaults are not what this notebook assumes; stop and read the banners")
""")

# --------------------------------------------------------------------------
md(r"""## 3. Toolkit register census — an open question with a one-cell answer

`STRESS272_RESULTS.md` §28.3: the same source, same arch, same `-O3 -DNDEBUG`
gives **48 registers/thread locally (nvcc 12.4) and 58 on Colab**. It is not the
assert flag (asserts cost +42). It changes no conclusion — at one warp per block
the 32-blocks-per-SM limit binds either way — but *a register count is a
property of the toolkit, not of the source*, and any future `__launch_bounds__`
work has to be measured on the toolkit that will build it.

This asks ptxas directly, on this machine, for every instantiation.""")

code(r"""
REG = {}
cmd = ("cd %s/port27/src/ViennaRNA && nvcc -ccbin gcc -O3 -DNDEBUG "
       "-gencode arch=compute_80,code=sm_80 -I%s/port27/src -I./mfe/cuda "
       "-Xptxas -v -cubin -o /content/int_loop_sm80.cubin mfe/cuda/int_loop.cu"
       % (ROOT, ROOT))
p = sh(cmd + " 2>&1", check=False, quiet=True)
body = p.stdout + p.stderr
name = None
for ln in body.splitlines():
    m = re.search(r"Compiling entry function '(.+?)'", ln)
    if m: name = m.group(1)
    m = re.search(r"Used (\d+) registers", ln)
    if m and name: REG[name] = int(m.group(1))
if not REG:
    print("ptxas produced no register report:\n", body[-1500:])
else:
    print("nvcc:", sh("nvcc --version | tail -2 | head -1", quiet=True).stdout.strip())
    print("%-58s %s" % ("kernel", "regs/thread"))
    for k in sorted(REG):
        short = re.sub(r"^_Z\d+", "", k)[:56]
        print("  %-56s %3d" % (short, REG[k]))
    warp1 = [v for k, v in REG.items() if "int_loop_warp_kernelILi1" in k]
    print()
    print("  warp kernel, 1 cell/block:", sorted(set(warp1)),
          " (local nvcc 12.4 says 48; the profiled Colab binary reported 58)")
with open("/content/registers.json", "w") as f:
    json.dump({"nvcc": sh("nvcc --version | tail -2 | head -1", quiet=True).stdout.strip(),
               "regs": REG}, f, indent=1)
""")

# --------------------------------------------------------------------------
md(r"""## 4. Workloads — one uniform, one deliberately ragged

**UNIFORM** is the same 400 × 5601, same seed, as every run since §14, so `sha`
is comparable with the whole results file. Equal lengths mean equal widths,
which is the case H6's waste guard accepts.

**RAGGED** is 300 records spread over 400–3000 nt. Whether its *chunks* end up
ragged is not obvious — the chunker sorts by length — and that is precisely what
§B measures rather than assumes.""")

code(r"""
os.makedirs(ROOT + "/fa", exist_ok=True)

N_BIG, LEN, SEED = 400, 5601, 20260907
BIG = ROOT + "/fa/big.fa"
random.seed(SEED)
with open(BIG, "w") as f:
    for i in range(N_BIG):
        f.write(">u%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(LEN))))
EXPECT_SHA = "7c0b3d633281"
print("uniform : %d x %d nt = %.1f MB   expect sha %s"
      % (N_BIG, LEN, os.path.getsize(BIG)/1e6, EXPECT_SHA))

RAG = ROOT + "/fa/ragged.fa"
random.seed(31337)
lens = [random.randint(400, 3000) for _ in range(300)]
with open(RAG, "w") as f:
    for i, L in enumerate(lens):
        f.write(">r%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(L))))
print("ragged  : %d records, %d..%d nt, total %.1f Mnt"
      % (len(lens), min(lens), max(lens), sum(lens)/1e6))
""")

# --------------------------------------------------------------------------
md(r"""## 5. The runner

Asserts, before any number is believed: the sweep ran, every knob engaged, the
answer did not move — and for H6, **which grid was actually chosen**.""")

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
# The pipeline reports what it ACHIEVED, not merely that it was asked for --
# and with one chunk there is nothing to overlap, so 0% is the correct answer
# there rather than a failure. Assert the report exists; read the number.
PIPE_RE  = re.compile(r"build pipeline: (\d+) chunks, builder ([\d.]+) s, "
                      r"OVERLAPPED ([\d.]+) s \((\d+)% of builder time hidden")
GRID_RE  = re.compile(r"int_loop grid: (2-D, blockIdx\.y = record|flat \(waste guard declined\)), "
                      r"lookup: (none|32-ary warp|binary) "
                      r"\(nfiles (\d+), maxw (\d+), blocks (\d+) vs flat (\d+)\)")
PHASES = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
          "new_c_host","fml_host","fml_prev_host")
STAGES = ("build","prepare","prefill","backtrack","output","gpuinit","teardown","free")

import threading
RESULTS = {}
OUT = "/content/lookup.json"
os.makedirs("/content/clk", exist_ok=True)
# D4 runs two folds from two threads, and both land here.
SAVE_LOCK = threading.Lock()

def save():
    with SAVE_LOCK:
        with open(OUT, "w") as f:
            json.dump({"commit": COMMIT, "clocks": clocks(), "nproc": NPROC,
                       "registers": REG, "runs": dict(RESULTS)}, f, indent=1)

class ClockSampler(object):
    FIELDS = ("clocks.sm","clocks.mem","temperature.gpu","power.draw",
              "utilization.gpu","utilization.memory","memory.used",
              "clocks_throttle_reasons.active")
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
            try: rows.append((float(q[0]), float(q[2]), float(q[3]), float(q[4]),
                              float(q[6]), q[7]))
            except ValueError: continue
        if not rows: return {}
        busy = [r for r in rows if r[3] >= 50.0] or rows
        thr = {}
        for r in busy: thr[r[5]] = thr.get(r[5], 0) + 1
        return dict(n=len(rows), n_busy=len(busy),
                    sm_mean=sum(r[0] for r in busy)/len(busy),
                    sm_min=min(r[0] for r in busy),
                    temp_max=max(r[1] for r in busy),
                    power_mean=sum(r[2] for r in busy)/len(busy),
                    power_max=max(r[2] for r in busy),
                    vram_max=max(r[4] for r in busy),
                    throttle=sorted(thr.items(), key=lambda kv: -kv[1])[:2])

def run(tag, fa=None, gridy=False, wsearch=False, int16=False, chunk_cap=29,
        block_size=None, warp=None, build_threads=None, phase_sync=True,
        vram_mb=None, pipeline=False, extra_args="", quiet=False):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH","RNA_PHASE_SYNC",
              "RNA_GPU_VRAM_BUDGET_MB","RNA_INT_LOOP_BLOCK_SIZE","RNA_BUILD_PIPELINE",
              "RNA_LAUNCH_STATS","RNA_INT_LOOP_WARP","RNA_BUILD_THREADS","RNA_MD_SMEM",
              "RNA_INT_LOOP_GRIDY","RNA_INT_LOOP_WSEARCH"):
        env.pop(k, None)
    env["RNA_MIN_GPU_BATCH"] = "1"
    # RNA_GPU_CHUNK IS THE MASTER SWITCH, NOT A CAP. RNAfold.c:2038 gates
    #     gpu_enabled = (vrna_cuda_devices() > 0) && (e) && (e[0]);
    # on the variable being SET and non-empty, and "0" means "no cap, the VRAM
    # budget alone decides". Leaving it unset does not mean "default chunking",
    # it means FOLD ON THE CPU -- which at 400 x 5601 is about an hour per arm
    # and looks exactly like a hang. Never leave it unset.
    env["RNA_GPU_CHUNK"] = "0" if chunk_cap is None else str(chunk_cap)
    if int16:          env["RNA_FML_INT16"] = "1"
    if phase_sync:     env["RNA_PHASE_SYNC"] = "1"
    if block_size:     env["RNA_INT_LOOP_BLOCK_SIZE"] = str(block_size)
    if warp is not None: env["RNA_INT_LOOP_WARP"] = "1" if warp else "0"
    if gridy:          env["RNA_INT_LOOP_GRIDY"] = "1"
    if wsearch:        env["RNA_INT_LOOP_WSEARCH"] = "1"
    if build_threads is not None: env["RNA_BUILD_THREADS"] = str(build_threads)
    if vram_mb is not None: env["RNA_GPU_VRAM_BUDGET_MB"] = str(vram_mb)
    if pipeline:       env["RNA_BUILD_PIPELINE"] = "1"

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
                               (gridy,"RNA_INT_LOOP_GRIDY=1","H6 gridy"),
                               (wsearch,"RNA_INT_LOOP_WSEARCH=1","H7 wsearch")):
        if bool(want) != (token in err):
            raise SystemExit("%s: %s knob did not engage as asked" % (tag, label))
    pm = PIPE_RE.search(err)
    if bool(pipeline) != bool(pm):
        raise SystemExit("%s: build pipeline asked=%s but it %s report"
                         % (tag, pipeline, "did not" if pipeline else "did"))
    if block_size:
        g = BS_RE.search(err)
        if (not g) or int(g.group(1)) != block_size:
            raise SystemExit("%s: block size %s asked, %s reported" % (tag, block_size, g and g.group(1)))

    # THE DECISION, not the knob. Printed on every CHANGE, so the trace is the
    # sequence of grids this run actually used.
    trace = [dict(grid=("2D" if g[0].startswith("2-D") else "flat"), lookup=g[1],
                  nfiles=int(g[2]), maxw=int(g[3]), blocks=int(g[4]), flat=int(g[5]))
             for g in GRID_RE.findall(err)]

    ph = dict(zip(PHASES,[float(x) for x in PHASE_RE.search(err).groups()])) if PHASE_RE.search(err) else {}
    st = dict(zip(STAGES,[float(x) for x in STAGE_RE.search(err).groups()])) if STAGE_RE.search(err) else {}
    shapes = SWEEP_RE.findall(err)
    rss = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
    if m: rss = int(m.group(1))/1e6
    sha = __import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12]
    bt = BT_RE.search(err)
    r = dict(wall=wall, phases=ph, stages=st, chunks=len(shapes),
             cells=sum(int(s[2]) for s in shapes), gridy=gridy, wsearch=wsearch,
             int16=int16, chunk_cap=chunk_cap, vram_mb=vram_mb, pipeline=pipeline,
             phase_sync=phase_sync, fa=os.path.basename(fa or BIG),
             block_size=int(BS_RE.search(err).group(1)) if BS_RE.search(err) else None,
             build_threads=int(bt.group(1)) if bt else None,
             grid_trace=trace, rss=rss, clock=clkinfo, sha=sha,
             pipe_chunks=int(pm.group(1)) if pm else None,
             pipe_builder=float(pm.group(2)) if pm else None,
             pipe_hidden_s=float(pm.group(3)) if pm else None,
             pipe_hidden_pct=int(pm.group(4)) if pm else None)
    RESULTS[tag] = r; save()
    if not quiet:
        g = "-" if not trace else ("2D" if all(t["grid"] == "2D" for t in trace)
                                   else ("flat" if all(t["grid"] == "flat" for t in trace)
                                         else "mixed(%d)" % len(trace)))
        print("  %-18s int_loop %7.2f  md %7.2f  wall %7.1f  %4.0f MHz %5.0f W  grid %-9s sha %s"
              % (tag, ph.get("int_loop",0), ph.get("modular_decomp",0), wall,
                 clkinfo.get("sm_mean",0), clkinfo.get("power_mean",0), g, sha))
    return r

PRE = ROOT + "/fa/preflight.fa"
random.seed(4242)
with open(PRE, "w") as f:
    for i in range(8):
        f.write(">p%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(300))))

# Run the section's configuration on 8 x 300 nt FIRST and assert the GPU
# engaged. run() already refuses a CPU fold -- but only AFTER the fold, and at
# 400 x 5601 that verdict arrives an hour late. Two seconds here instead.
#
# Not hypothetical: D3 passed chunk_cap=None believing it meant "default
# chunking". RNA_GPU_CHUNK went unset, which is the master switch, and four
# arms folded on the CPU while the run looked hung.
def preflight(label, **kw):
    kw.pop("fa", None); kw.pop("quiet", None); kw.pop("tag", None)
    try:
        r = run("PRE_" + label, fa=PRE, quiet=True, **kw)
    except SystemExit as e:
        raise SystemExit("PREFLIGHT FAILED for %s: %s" % (label, e))
    print("  preflight %-22s ok (%d chunk(s), %.1fs, sha %s)"
          % (label, r["chunks"], r["wall"], r["sha"]))
    return r

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs)/len(xs) if xs else float("nan")

print("runner ready ->", OUT)
""")

md(r"""### Warm up before measuring anything

Four discarded folds. On the local box one was not enough: `int_loop` rose
2.84 → 5.25 s across twelve folds with the control tracking it, and the first
six rows of that run were a clock ramp wearing the costume of a result. Whether
the A100 needs this at all is itself reported below.""")

code(r"""
WARM = ROOT + "/fa/warm.fa"
random.seed(777)
with open(WARM, "w") as f:
    for i in range(60):
        f.write(">w%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(2400))))

print("warm-up (discarded) -- watch whether int_loop and the clock settle")
for k in range(4):
    r = run("WARM_%d" % k, fa=WARM, quiet=True)
    print("   %d  int_loop %6.2f  md %6.2f  %4.0f MHz  %5.0f W  %2.0f C"
          % (k, r["phases"].get("int_loop",0), r["phases"].get("modular_decomp",0),
             r["clock"].get("sm_mean",0), r["clock"].get("power_mean",0),
             r["clock"].get("temp_max",0)))
print("\nIf those four are flat, this card does not need warming and the local")
print("ramp was a property of the laptop, not of the measurement method.")
""")

# --------------------------------------------------------------------------
md(r"""## A. Uniform workload — all four lookups

400 × 5601, equal lengths, so every chunk has equal widths and **H6's waste
guard should accept every launch**. The `grid` column proves it did.

`modular_decomp` is the control: neither knob can reach it.

Palindromic order, twice: `base wsearch gridy both both gridy wsearch base`.

**`both` should land on top of `gridy`, and that is not a bug.** Where the 2-D
grid is accepted there is no search left for H7 to speed up, so the `lookup`
field of the decision line reads `none`. H7's knob still announces itself —
`int_loop_cuda()` reads it unconditionally, precisely so that an arm which asks
for it can be *checked* on a workload where it is never reached.""")

code(r"""
ARMS = {"base":    dict(),
        "wsearch": dict(wsearch=True),
        "gridy":   dict(gridy=True),
        "both":    dict(gridy=True, wsearch=True)}

print("A: preflight -- every arm must reach the GPU before any of them runs long")
for a in ARMS: preflight("A_" + a, **ARMS[a])

print("\nA: the four lookups at 400 x 5601 (modular_decomp is the CONTROL)")
order = ["base","wsearch","gridy","both","both","gridy","wsearch","base"]
for n, a in enumerate(order):
    run("A_%s_%d" % (a, n), **ARMS[a])
""")

code(r"""
def arm_rows(prefix, arm): return [v for k, v in RESULTS.items()
                                   if k.startswith(prefix) and k.split("_")[1] == arm]

def compare(prefix, title, base_arm="base"):
    arms = ["base","wsearch","gridy","both"]
    have = {a: arm_rows(prefix, a) for a in arms}
    have = {a: v for a, v in have.items() if v}
    if base_arm not in have: print("no baseline for", title); return
    b_il = mean([v["phases"]["int_loop"] for v in have[base_arm]])
    b_md = mean([v["phases"]["modular_decomp"] for v in have[base_arm]])
    print(title)
    print("  %-9s %9s %9s %9s %9s %9s  %s"
          % ("arm","int_loop","raw","vs ctrl","mod_dec","wall","grid"))
    for a in arms:
        if a not in have: continue
        v = have[a]
        il = mean([x["phases"]["int_loop"] for x in v])
        mdc = mean([x["phases"]["modular_decomp"] for x in v])
        raw = 100.0*(il-b_il)/b_il
        nrm = ((il/mdc)/(b_il/b_md) - 1)*100 if mdc and b_md else float("nan")
        tr = [t["grid"] for x in v for t in x["grid_trace"]]
        g = "-" if not tr else ("2D" if all(t == "2D" for t in tr)
                                else ("flat" if all(t == "flat" for t in tr) else "mixed"))
        sp = [x["phases"]["int_loop"] for x in v]
        print("  %-9s %9.2f %+8.2f%% %+8.2f%% %9.2f %9.1f  %-5s  spread %.2f%%"
              % (a, il, raw, nrm, mdc, mean([x["wall"] for x in v]), g,
                 100.0*(max(sp)-min(sp))/min(sp) if len(sp) > 1 else 0.0))
    shas = set(x["sha"] for v in have.values() for x in v)
    print("  shas:", shas, "" if len(shas) == 1 else "   <<< MOVED")

compare("A_", "uniform 400 x 5601:")
""")

# --------------------------------------------------------------------------
md(r"""## B. Ragged workload — does H7 have a reason to exist?

This is the section that decides something rather than confirming it.

H6's 2-D grid launches `nfiles × ceil(maxw/cpb)` blocks where the flat grid
launches one per cell; the excess exit on a comparison, but they are still
blocks to schedule, so the host declines the 2-D grid when the waste exceeds
25 %. **H7 is the only thing that helps on a declined launch.**

The open question is how often that happens in practice. The chunker sorts by
length, which *pushes toward* uniform chunks — but records within a chunk still
retire at different rows, and a retired record has width 0. Nobody has measured
which effect wins.

| what the `grid` column shows | what it means |
|---|---|
| `gridy` arm is all **2D** | even a 400–3000 nt spread chunks uniformly; H7 is for `nfiles > 65535` and little else |
| `gridy` arm is all **flat** | the guard never accepts here; **H7 carries this workload alone** |
| `mixed` | it flips during the sweep — read the trace, that is retirement doing it |
""")

code(r"""
print("B: the four lookups on 300 ragged records, 400..3000 nt")
order = ["base","wsearch","gridy","both","both","gridy","wsearch","base"]
for n, a in enumerate(order):
    run("B_%s_%d" % (a, n), fa=RAG, **ARMS[a])
""")

code(r"""
compare("B_", "ragged 300 x 400..3000:")
print()
print("--- the decision trace (printed on every CHANGE of grid) ---")
for k in sorted(RESULTS):
    if not k.startswith("B_") or not RESULTS[k]["grid_trace"]: continue
    tr = RESULTS[k]["grid_trace"]
    print("  %-18s %d change(s)" % (k, len(tr)))
    for t in tr[:6]:
        print("      %-5s lookup %-12s nfiles %4d  maxw %6d  blocks %8d vs flat %8d  (waste %.2fx)"
              % (t["grid"], t["lookup"], t["nfiles"], t["maxw"], t["blocks"], t["flat"],
                 t["blocks"]/max(t["flat"],1)))
    if len(tr) > 6: print("      ... %d more" % (len(tr)-6))
""")

# --------------------------------------------------------------------------
md(r"""## C. Mechanism — does the stall move the way the story says?

The claim is specific: these changes shorten a **dependency chain of global
loads**, so `long_scoreboard` should fall from `base` to `wsearch` to `gridy`,
and nothing else should move much. If `long_scoreboard` does not budge and the
time still improves, the story is wrong even though the number is right.

Registers are the second question. H7 costs +2 locally (48 → 50). At one warp
per block that is free — the 32-blocks-per-SM limit binds long before registers
do — but it should be *seen* to be free rather than assumed.""")

code(r"""
SECT = "--section WarpStateStats --section SchedulerStats --section Occupancy"
EXTRA = ",".join([
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks","launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem","launch__occupancy_limit_warps",
    "launch__registers_per_thread","launch__grid_size","launch__block_size",
    "l1tex__t_sector_hit_rate.pct","lts__t_sector_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
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
def profile(tag, gridy=False, wsearch=False):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_PHASE_SYNC","RNA_MD_SMEM",
              "RNA_INT_LOOP_GRIDY","RNA_INT_LOOP_WSEARCH"):
        env.pop(k, None)
    env.update(RNA_GPU_CHUNK=str(PROF_CAP), RNA_MIN_GPU_BATCH="1")
    if gridy:   env["RNA_INT_LOOP_GRIDY"]   = "1"
    if wsearch: env["RNA_INT_LOOP_WSEARCH"] = "1"
    out = "/content/ncu_%s.csv" % tag
    for flavour, extra in (("sections", SECT + " --metrics " + EXTRA),
                           ("explicit", "--metrics " + FALLBACK)):
        cmd = ("ncu --target-processes all --csv %s -k regex:'^int_loop_warp_kernel' "
               "--launch-skip %d --launch-count %d --log-file %s %s --noPS -i %s "
               "> /dev/null 2>&1" % (extra, SKIP, COUNT, out, BIN, PFA))
        subprocess.run(cmd, shell=True, env=env)
        body = open(out).read() if os.path.exists(out) else ""
        if ("No kernels were profiled" in body) or ("Metric Name" not in body):
            continue
        rows = list(_csvmod.DictReader(io.StringIO(
            "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
        names = set(r.get("Kernel Name","") for r in rows)
        if names and not any("int_loop_warp_kernel" in nm for nm in names):
            print("  %-10s *** WRONG KERNEL: %r ***" % (tag, sorted(names)[:1])); return {}
        agg = collections.defaultdict(list)
        for r in rows:
            try: agg[r["Metric Name"]].append(float((r["Metric Value"] or "").replace(",","")))
            except Exception: pass
        res = {k: sum(v)/len(v) for k, v in agg.items() if v}
        if any("issue_stalled" in k for k in res):
            print("  %-10s ok via %s" % (tag, flavour)); STALLS[tag] = res; return res
    print("  %-10s *** NO STALL METRICS -- broken probe, not a result ***" % tag)
    return {}

print("C: NCU on the three lookups (same kernel, same fixture, same block size)")
for tag, kw in (("base", {}), ("wsearch", dict(wsearch=True)), ("gridy", dict(gridy=True))):
    profile(tag, **kw)
with open("/content/lookup_stalls.json","w") as f: json.dump(STALLS, f, indent=1)
""")

code(r"""
if not STALLS:
    print("No stall data -- nothing below is a result.")
else:
    tags = [t for t in ("base","wsearch","gridy") if t in STALLS]
    print("--- occupancy, registers, grid ---")
    print("%-9s %10s %11s %9s %9s %11s   %s"
          % ("arm","occupancy","duration us","regs/thr","grid","sectors/req","blk/reg/smem/warp"))
    for t in tags:
        r = STALLS[t]
        lims = "/".join(str(int(r.get("launch__occupancy_limit_"+k,0)))
                        for k in ("blocks","registers","shared_mem","warps"))
        print("%-9s %8.1f%% %11.1f %9d %9d %11.2f   %s"
              % (t, r.get("sm__warps_active.avg.pct_of_peak_sustained_active",0),
                 r.get("gpu__time_duration.sum",0)/1e3,
                 int(r.get("launch__registers_per_thread",0)),
                 int(r.get("launch__grid_size",0)),
                 r.get("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",0),
                 lims))
    print()
    print("  NOTE the grid column: the 2-D arm launches a DIFFERENT NUMBER OF BLOCKS")
    print("  for the same work, so its achieved occupancy is not comparable with the")
    print("  others' -- that confound is what made 27.5 misread registers (28.2).")
    print()
    keys = sorted({k for r in STALLS.values() for k in r if "issue_stalled" in k})
    tot = {t: sum(STALLS[t].get(k,0.0) for k in keys) for t in tags}
    def short(k):
        m = re.search(r"issue_stalled_(.+?)_per_issue_active", k); return m.group(1) if m else k
    order2 = sorted(keys, key=lambda k: -max(STALLS[t].get(k,0.0) for t in tags))
    print("--- warp issue stalls: ABSOLUTE cycles per issue (not %%) ---")
    print("%-20s" % "stall reason" + "".join("%11s" % t for t in tags))
    for k in order2:
        vals = [STALLS[t].get(k,0.0) for t in tags]
        if max(vals) < 0.005: continue
        print("%-20s" % short(k) + "".join("%11.3f" % v for v in vals))
    print("%-20s" % "(total)" + "".join("%11.3f" % tot[t] for t in tags))
    print()
    print("  THE PREDICTION: long_scoreboard falls base > wsearch > gridy, in that")
    print("  order, because that is the order of dependency-chain depth (9, 2, 0).")
    print("  Absolute cycles, not percentages -- a share can fall because something")
    print("  else rose.")
""")

# --------------------------------------------------------------------------
md(r"""## D. Stress — hold the card down and see what gives

Every A100 number in this project came from runs of about two minutes that
reported `0x0` in `clocks_throttle_reasons`. That is not evidence about twenty
minutes. The T4's answer turned out to be a **power cap** that cost 25 % of the
clock and looked, in the phase timings, exactly like an int16 regression.

Four questions, each with its own failure mode:

| | question | what a bad answer looks like |
|---|---|---|
| **D1** | does it hold its clock under sustained load? | `sm_min` drifts down, `throttle` stops being `0x0`, later repeats slower than earlier ones |
| **D2** | does the phase mix hold as the work grows? | something goes superlinear — the sweep is O(n³) in length and O(n) in records, and a departure from that is a finding |
| **D3** | what does chunking cost on a 40 GB card? | `project_chunking_costs_batch_width` says 0.6 %, measured on a T4 at one size |
| **D4** | does the ANSWER survive contention? | **a wrong sha under contention outranks every timing in this notebook** |

D1 and D4 run with the **shipped defaults**, because what is being stressed is
the product, not the experiment.""")

md(r"""### D1 — sustained load

The same 400 × 5601 fold, six times back to back, with clocks sampled
throughout. Reported per repeat so a drift is visible as a *slope*, not as an
average that hides it.""")

code(r"""
print("D1: six consecutive folds at 400 x 5601, shipped defaults")
for k in range(6):
    run("D1_rep%d" % k)
""")

code(r"""
reps = [(k, RESULTS[k]) for k in sorted(RESULTS) if k.startswith("D1_rep")]
if reps:
    print("%-9s %9s %9s %9s %8s %8s %8s %8s %7s  %s"
          % ("repeat","int_loop","mod_dec","wall","MHz avg","MHz min","W avg","W max","degC","throttle"))
    for k, v in reps:
        c = v["clock"]
        thr = ",".join("%s:%d" % (a,b) for a,b in c.get("throttle",[])[:2])
        print("%-9s %9.2f %9.2f %9.1f %8.0f %8.0f %8.0f %8.0f %7.0f  %s"
              % (k[3:], v["phases"].get("int_loop",0), v["phases"].get("modular_decomp",0),
                 v["wall"], c.get("sm_mean",0), c.get("sm_min",0), c.get("power_mean",0),
                 c.get("power_max",0), c.get("temp_max",0), thr))
    first, last = reps[0][1], reps[-1][1]
    dw = 100.0*(last["wall"]-first["wall"])/first["wall"]
    dc = 100.0*(last["clock"].get("sm_mean",1)-first["clock"].get("sm_mean",1))/first["clock"].get("sm_mean",1)
    print()
    print("  first -> last:  wall %+.2f%%   clock %+.2f%%" % (dw, dc))
    print("  shas:", set(v["sha"] for _, v in reps))
    print()
    print("  If wall rises while the clock falls by a matching amount, that is a cap,")
    print("  not a regression -- the T4 finding. If the clock holds and the wall")
    print("  holds, this card sustains the load and every A100 number so far stands.")
""")

md(r"""### D2 — scale

Records and length swept separately, because they enter differently: the sweep
is roughly O(length³) per record and linear in record count. `cells` is
reported so the per-cell cost can be compared across shapes — that is the
quantity that should be flat.""")

code(r"""
SCALE = [("n100_L5601", 100, 5601), ("n200_L5601", 200, 5601),
         ("n400_L5601", 400, 5601),
         ("n400_L2000", 400, 2000), ("n400_L3500", 400, 3500)]
print("D2: scale sweep, shipped defaults")
for tag, n, L in SCALE:
    fa = "%s/fa/%s.fa" % (ROOT, tag)
    if not os.path.exists(fa):
        random.seed(SEED)
        with open(fa, "w") as f:
            for i in range(n):
                f.write(">u%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(L))))
    run("D2_" + tag, fa=fa)
""")

code(r"""
rows = [(k, RESULTS[k]) for k in sorted(RESULTS) if k.startswith("D2_")]
if rows:
    print("%-14s %8s %9s %12s %9s %9s %9s %11s"
          % ("shape","wall","build","cells","int_loop","mod_dec","GPU s","ns/cell"))
    for k, v in rows:
        gpu = sum(v["phases"].get(p,0) for p in ("int_loop","hp_mb","load_my_c",
                                                 "modular_decomp","fetch_mx"))
        cells = max(v["cells"], 1)
        print("%-14s %8.1f %9.2f %12d %9.2f %9.2f %9.1f %11.2f"
              % (k[3:], v["wall"], v["stages"].get("build",0), v["cells"],
                 v["phases"].get("int_loop",0), v["phases"].get("modular_decomp",0),
                 gpu, 1e9*gpu/cells))
    print()
    print("  ns/cell is the quantity that should be FLAT. If it rises with size the")
    print("  device is saturating; if it falls, the smaller shapes were not filling it")
    print("  -- and that changes what MIN_GPU_BATCH should be.")
""")

md(r"""### D3 — VRAM pressure

Squeeze the budget until the sweep is many chunks.
`project_chunking_costs_batch_width` says chunking costs **0.6 %, not k×** — but
that was measured at one size on a T4. A 40 GB card chunks for entirely
different reasons, and `chunks` is reported so the cost can be read per chunk.""")

code(r"""
# chunk_cap=0 -- "no cap, the VRAM budget alone decides". NOT None, which
# would leave RNA_GPU_CHUNK unset and fold the whole thing on the CPU.
print("D3: preflight")
for mb in (None, 8192, 4096, 2048):
    preflight("D3_%s" % (mb or "full"), vram_mb=mb, chunk_cap=0)

print("\nD3: VRAM budget squeezed (400 x 5601, the budget alone deciding)")
for mb in (None, 8192, 4096, 2048):
    run("D3_vram%s" % (mb or "full"), vram_mb=mb, chunk_cap=0)
""")

code(r"""
rows = [(k, RESULTS[k]) for k in sorted(RESULTS) if k.startswith("D3_")]
if rows:
    base = None
    print("%-12s %8s %8s %9s %9s %10s %9s"
          % ("budget","chunks","wall","int_loop","mod_dec","peak VRAM","vs full"))
    for k, v in rows:
        if base is None: base = v["wall"]
        print("%-12s %8d %8.1f %9.2f %9.2f %9.0f MB %+8.1f%%"
              % (k[3:], v["chunks"], v["wall"], v["phases"].get("int_loop",0),
                 v["phases"].get("modular_decomp",0), v["clock"].get("vram_max",0),
                 100.0*(v["wall"]-base)/base))
    print("  shas:", set(v["sha"] for _, v in rows))
""")

md(r"""### D4 — contention: two folds at once

Not a speed test. Two processes sharing one GPU without MPS time-slice, so each
will be slower and that is expected and uninteresting.

**The question is whether the answers are still right.** Both processes must
produce the sha their fixture produces alone. A wrong answer here would outrank
every timing in this notebook, because it would mean device state leaking
between contexts — and nothing else in this project's test surface would catch
it.""")

code(r"""
solo = RESULTS.get("D1_rep0") or run("D4_solo_ref")
SOLO_SHA, SOLO_WALL = solo["sha"], solo["wall"]
print("solo reference: sha %s, wall %.1f s" % (SOLO_SHA, SOLO_WALL))

res = {}
def worker(name):
    try: res[name] = run("D4_" + name, quiet=True)
    except SystemExit as e: res[name] = {"error": str(e)}

print("\nlaunching two concurrent folds at 400 x 5601 ...")
t0 = time.time()
ts = [threading.Thread(target=worker, args=("par%d" % i,)) for i in (0, 1)]
for t in ts: t.start()
for t in ts: t.join()
both = time.time() - t0

print("\n%-10s %9s %9s %9s  %s" % ("proc","wall","int_loop","mod_dec","sha"))
ok = True
for name in sorted(res):
    v = res[name]
    if "error" in v:
        print("%-10s FAILED: %s" % (name, v["error"])); ok = False; continue
    print("%-10s %9.1f %9.2f %9.2f  %s%s"
          % (name, v["wall"], v["phases"].get("int_loop",0),
             v["phases"].get("modular_decomp",0), v["sha"],
             "" if v["sha"] == SOLO_SHA else "   <<< WRONG ANSWER UNDER CONTENTION"))
    if v["sha"] != SOLO_SHA: ok = False
print()
print("  wall for the PAIR %.1f s vs %.1f s for one alone (%.2fx)"
      % (both, SOLO_WALL, both/SOLO_WALL))
print("  2.00x would be perfect time-slicing; below 2.00x means one fold was not")
print("  filling the card and the second rode along in the gaps.")
print("  ANSWERS:", "correct under contention" if ok else "*** A SHA MOVED ***")
""")

# --------------------------------------------------------------------------
md(r"""## E. The two questions §31 could not answer locally

### E1 — what is the production wall, and what do per-row barriers cost?

**Every arm above carries `RNA_PHASE_SYNC=1`**, which serialises every phase
boundary so the phase timers are true GPU times. `device.cu` says plainly that
such a wall *"is not comparable to a normal run"* — so the project has never
measured this card's production wall, and §30's absolute figures are all
phase-synced quantities.

One pair settles it. It also prices what `PORT_STREAM_OVERLAP_SCOPE.md` stage 0
wanted: `RNA_PHASE_SYNC` adds ~5 device syncs per sweep row, so the difference
**is** the cost of per-row barriers, run-ahead included.

**The `nosync` arm's phase timers are meaningless and must not be reported** —
reading them is the mistake §19 spent a session unpicking.
"""); 
code(r"""
print("E1: production wall vs phase-synced wall, full VRAM budget")
for tag, ps in (("E1_sync_1", True), ("E1_nosync_1", False),
                ("E1_nosync_2", False), ("E1_sync_2", True)):
    run(tag, chunk_cap=0, phase_sync=ps)
"""); 
code(r"""
syn = [v for k, v in RESULTS.items() if k.startswith("E1_sync")]
nos = [v for k, v in RESULTS.items() if k.startswith("E1_nosync")]
if syn and nos:
    a, b = mean([v["wall"] for v in syn]), mean([v["wall"] for v in nos])
    print("  phase-synced wall  %8.1f s" % a)
    print("  production wall    %8.1f s   %+.1f%%" % (b, 100.0*(b-a)/a))
    print()
    print("  ~5 syncs per row x %d rows x %d chunks" % (5601, syn[0]["chunks"]))
    print("  -> per-row barriers cost %.1f s, %.1f%% of the production wall."
          % (a-b, 100.0*(a-b)/b))
    print("  That is the PORT_STREAM_OVERLAP_SCOPE stage 0 number: it bounds what\n"
          "  removing the two synchronous uploads and the graph sync could pay.")
    print("  shas:", set(v["sha"] for v in syn+nos))
    print()
    print("  NOTE: the nosync arms' phase timers are NOT reported and must not be.")
"""); 
md(r"""### E2 — the build pipeline, and where the chunk optimum really is

`RNA_BUILD_PIPELINE` is **off by default**, and `RNAfold.c:1609` says why:
*"the trade is the user's to make **until it is measured at scale**."* It has
since been measured at scale — 16.8–21.4 % at 400 × 5601, sha unchanged across
nine arms, at ~2× host RSS — and the default never moved. Every arm in §30 ran
with it off, so `build`'s 18.35 s was fully exposed in all of them.

**But it pulls against §30.7.** The pipeline hides `(chunks−1)/chunks` of
`build`, so it wants MORE chunks; chunking costs 2.15 s each, which wants
FEWER. §30.7 swept chunk count with the pipeline off, so the optimum is
unmeasured in both directions at once. This is the 2-D sweep.

**Watch host RSS.** Two chunks of 200 × 5601 nt compounds is simultaneously the
best case for chunking and the worst for memory.
"""); 
code(r"""
print("E2: chunk count x build pipeline, production settings (no phase sync)")
for mb, chunks in ((None, "full"), (8192, "7ish"), (4096, "13ish")):
    for pipe in (False, True):
        run("E2_%s_%s" % (chunks, "pipe" if pipe else "nopipe"),
            chunk_cap=0, vram_mb=mb, pipeline=pipe, phase_sync=False)
"""); 
code(r"""
rows = [(k, RESULTS[k]) for k in sorted(RESULTS) if k.startswith("E2_")]
if rows:
    print("  %-16s %7s %9s %11s %9s" % ("arm","chunks","wall","peak RSS","sha"))
    for k, v in rows:
        print("  %-16s %7d %9.1f %8.2f GB  %s"
              % (k[3:], v["chunks"], v["wall"], v["rss"], v["sha"]))
    print()
    for base in sorted(set(k.rsplit("_",1)[0] for k, _ in rows)):
        a, b = RESULTS.get(base+"_nopipe"), RESULTS.get(base+"_pipe")
        if a and b:
            print("  %-14s pipeline %+6.1f%% wall, %+.2f GB RSS, "
                  "%d%% of builder hidden (ceiling %d%% at %d chunks)"
                  % (base[3:], 100.0*(b["wall"]-a["wall"])/a["wall"],
                     b["rss"]-a["rss"], b.get("pipe_hidden_pct") or 0,
                     100*(b["chunks"]-1)//max(b["chunks"],1), b["chunks"]))
    print()
    best = min(rows, key=lambda kv: kv[1]["wall"])
    print("  fastest configuration:", best[0], "%.1f s" % best[1]["wall"])
    print("  shas:", set(v["sha"] for _, v in rows))
"""); 
md("## F. Summary and export")

code(r"""
print("commit", COMMIT, "| cores", NPROC)
print(clocks())
print()
by_fa = collections.defaultdict(set)
for k, v in RESULTS.items():
    if isinstance(v, dict) and "sha" in v: by_fa[v["fa"]].add(v["sha"])
for fa in sorted(by_fa):
    s = by_fa[fa]
    print("  %-16s %d distinct sha %s  %s" % (fa, len(s), sorted(s),
                                              "OK" if len(s) == 1 else "*** MOVED ***"))
print()
print("%-18s %9s %9s %9s %7s %6s %5s"
      % ("arm","int_loop","mod_dec","wall","MHz","W","sha"))
for k in sorted(RESULTS):
    v = RESULTS[k]
    if not isinstance(v, dict) or "phases" not in v: continue
    print("%-18s %9.2f %9.2f %9.1f %7.0f %6.0f %s"
          % (k, v["phases"].get("int_loop",0), v["phases"].get("modular_decomp",0),
             v["wall"], v["clock"].get("sm_mean",0), v["clock"].get("power_mean",0),
             v["sha"]))
""")

code(r"""
from google.colab import files
sh("cd /content && tar czf lookup_artifacts.tar.gz lookup.json lookup_stalls.json "
   "registers.json clk ncu_*.csv 2>/dev/null", check=False, quiet=True)
for f in ("lookup.json", "lookup_stalls.json", "registers.json", "lookup_artifacts.tar.gz"):
    p = "/content/" + f
    if os.path.exists(p):
        print("%-30s %8.1f KB" % (f, os.path.getsize(p)/1024))
        files.download(p)
""")

md(r"""### The decision table, written before the run

| § | reading | means |
|---|---|---|
| **A** | `gridy` −5…−10 %, `wsearch` −2…−5 %, control flat | both generalise off sm_86; **default them** |
| **A** | `both` ≈ `gridy` | expected — where the 2-D grid is taken there is no search left for H7 to speed up |
| **A** | `gridy` grid column is not `2D` | the waste guard declined a *uniform* workload; the guard is wrong, not the kernel |
| **A** | no win | sm_86 artefact — keep both gated and say so plainly |
| **B** | `gridy` all `flat` | **H7 is the load-bearing change for real data**; H6 only helps equal-length batches |
| **B** | `gridy` all `2D` | the length sort already makes chunks uniform; H7 is for `nfiles > 65535` and should be argued on that alone |
| **B** | `mixed` | retirement is making chunks ragged mid-sweep — read the trace and consider a per-launch decision (which is what the code already does) |
| **C** | `long_scoreboard` falls 9 → 2 → 0 in chain order | the mechanism is what we claim |
| **C** | time improves, `long_scoreboard` flat | **the story is wrong**; find the real mechanism before writing it down again |
| **C** | registers ≠ 48/50 | the toolkit disagrees with the local one — §28.3's open question, now with a number |
| **D1** | clock and wall both flat over six repeats | the A100 sustains; every A100 number in the file stands |
| **D1** | wall up, clock down by the same fraction | a cap, exactly like the T4 — and every long-run number needs re-reading |
| **D2** | ns/cell flat across shapes | the device is neither starved nor saturated in this range |
| **D2** | ns/cell falls with size | small shapes were not filling the card — `MIN_GPU_BATCH` is mis-set for this host |
| **D3** | chunk cost ≈ 0.6 % | the T4 finding holds on a 40 GB card |
| **D4** | both sha match solo | the answer is safe under contention |
| **D4** | **a sha moves** | **stop everything.** Nothing else in this notebook matters. |
""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Lookup.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (out, len(cells)))

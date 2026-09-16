#!/usr/bin/env python3
"""Build the Scaling notebook: what changes with LENGTH, what changes with
BLOCK SIZE, and what is stopping either from scaling.

Everything here exists because STRESS272 §33 left it open.

  A  THE RSS MODEL. The build pipeline's AUTO gate charges 2*(L+1)^2 per record
     and declines when that does not fit in half of MemAvailable. Measured at
     200 x 5601 the real extra residency is 6.40 GB against the 15.31 GB it
     charges -- 2.4x conservative, which costs a 32 GB host the 5.6 % the
     pipeline is worth. One coefficient cannot be fitted from one point, so
     this sweeps records x length and fits bytes-per-record(L).

  B  CACHE vs LENGTH. L1 and L2 hit rates are quoted per kernel and never as a
     function of sequence length, yet every reuse pattern in the sweep is a
     function of the row width. If L2 falls off a cliff at some length, that is
     where the working set stops fitting and it should be a known number.

  C  BLOCK SIZE c*32. Both hot kernels have a block-size knob and neither has
     been swept on this card since the warp kernel became the default. The
     question is not only which is fastest but WHAT BINDS -- ncu reports the
     occupancy limiter per resource (blocks, registers, shared memory, warps),
     so the obstacle can be named rather than guessed.

  D  THE THREE THINGS §33 COULD NOT ANSWER.
       D1  waves per SM at the PRODUCTION shape. §33.1 measured 0.66 on a
           60 x 1800 profiling fixture and flagged it as a fixture property.
       D2  imc_miss 3.25 per issue in modular_decomp, 10x int_loop's, with
           __constant__ memory as its named suspect.
       D3  ACTIVE LANES PER INSTRUCTION -- the metric that decides whether
           PORT_SCOUT_COMPACTION_SCOPE.md is worth designing. A latency-bound
           kernel whose warps are already full has no compaction prize.

WHAT THIS NOTEBOOK ASSERTS BEFORE BELIEVING ANYTHING

  * `sweep shape:` present, or the run folded on the CPU (RNA_GPU_CHUNK is the
    master switch, not a cap).
  * every knob asked for announced itself on stderr.
  * sha constant within a fixture, across every arm that shares it.
  * ncu reports the kernel it was asked for, by name.

usage: python3 tools/make_nb_scaling.py [out.ipynb]
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
md(r"""# Scaling: length, block size, and what is standing in the way

## What this run decides

| § | question | why it is open |
|---|---|---|
| **A** | how much host memory does a record of length *L* really cost? | the AUTO pipeline gate charges **2·(L+1)²** and is **2.4× conservative** at 5601 nt — it declines on a 32 GB host what would fit |
| **B** | how do **L1 and L2 hit rates move with sequence length**? | never measured as a function of length, and every reuse pattern in the sweep is a function of row width |
| **C** | how do block sizes of **c·32** scale, and **what binds**? | both hot kernels have the knob; ncu names the limiting resource, so the obstacle can be stated rather than guessed |
| **D1** | waves per SM at the **production** shape | §33.1's 0.66 is a *profiling-fixture* number and was flagged as one |
| **D2** | why is `imc_miss` **3.25 per issue** in `modular_decomp`, 10× `int_loop`'s? | unexplained; `__constant__` memory is the suspect |
| **D3** | **how full are the warps?** | decides whether `PORT_SCOUT_COMPACTION_SCOPE.md` has a prize at all |

## What is already settled, so it is not re-measured here

* Both hot kernels are **latency-bound**: `long_scoreboard` 11.67 per issue in
  `modular_decomp`, 5.89 in `int_loop`, at 7.9 % and 1.6 % of DRAM peak (§33.1).
* **The 32-lane tile buys coalescing**, not just parallelism — `RNA_MD_TILE=1`
  costs 247 % and takes sectors/request 2.04 → 10.99. Nothing here may break
  lane-striding.
* The production default wall is **85.9 s** at 400 × 5601 and the card sustains
  it (§33.2). Any arm far from that number is misconfigured, not interesting.

## Rules

* **`sha` must be constant within a fixture.** A block size that changes the
  answer is a bug and a bigger finding than any timing.
* **`sweep shape:` must be present**, or the run folded on the CPU.
* **Name the limiter, not the winner.** §C's deliverable is the occupancy-limit
  table, not just the fastest block size.
* **Palindromic order, never ABAB.**
* **Warm up before measuring anything.**
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""
import subprocess, os, sys, json, time, re, random, io, csv as _csvmod, collections

def sh(cmd, check=True, quiet=False):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet and p.stdout: print(p.stdout[-2000:])
    if check and p.returncode: print(p.stderr[-2000:]); raise SystemExit(cmd)
    return p

print(sh("nvidia-smi --query-gpu=name,clocks.max.sm,memory.total --format=csv,noheader",
         quiet=True).stdout.strip())
NPROC = int(sh("nproc", quiet=True).stdout.strip())
print("cores:", NPROC)
""")

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
print("/usr/bin/time:", "present" if TIME_BIN else "ABSENT (RSS will read 0 -- SECTION A IS DEAD)")
""")

# --------------------------------------------------------------------------
md(r"""## 2. Build

Probing the **source** for what this run depends on, not a commit hash: a hash
goes stale the moment the branch is rebased.""")

code(r"""
REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/scaling"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())

SRC = ROOT + "/port27/src/"
NEED = [
    ("RNA_INT_LOOP_BLOCK_SIZE", SRC+"ViennaRNA/mfe/cuda/int_loop.cu",    "SS C's int_loop knob"),
    ("RNA_MD_BLOCK_SIZE",       SRC+"ViennaRNA/mfe/cuda/modular_decomposition.cu", "SS C's md knob"),
    ("int_loop_warp_kernel",    SRC+"ViennaRNA/mfe/cuda/int_loop.cu",    "the default kernel"),
    ("build pipeline AUTO",     SRC+"bin/RNAfold.c",                     "SS A's gate"),
    ("RNA_HOST_AVAIL_MB",       SRC+"bin/RNAfold.c",                     "SS A's test hook"),
]
missing = []
for tok, path, why in NEED:
    ok = os.path.exists(path) and tok in open(path).read()
    print("  %-26s %-28s %s" % (tok, why, "present" if ok else "MISSING"))
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

p = sh("cd %s/port27 && ./autogen.sh > /content/autogen.log 2>&1" % ROOT, check=False, quiet=True)
if p.returncode:
    print(sh("tail -40 /content/autogen.log", quiet=True).stdout); raise SystemExit("autogen failed")
sh("chmod +x %s/port27/doc/man2rst.py" % ROOT, check=False, quiet=True)

p = sh("cd %s/port27 && ./configure --without-python --without-perl --without-swig "
       "--without-doc --without-rnaxplorer --without-forester --without-kinfold "
       "--without-rnalocmin --enable-cuda CFLAGS='-g -O2' CXXFLAGS='-g -O2' "
       "PYTHON3=\"$(command -v python3)\" > /content/conf.log 2>&1" % ROOT,
       check=False, quiet=True)
if p.returncode:
    print(sh("tail -40 /content/conf.log", quiet=True).stdout); raise SystemExit("configure failed")

p = sh("cd %s/port27 && make -j%d > /content/make.log 2>&1" % (ROOT, NPROC), check=False, quiet=True)
if p.returncode:
    print(sh("grep -E 'error|Error' /content/make.log | head -30", quiet=True).stdout)
    raise SystemExit("make failed")

BIN = "%s/port27/src/bin/RNAfold" % ROOT
print("built in %.1f min ->" % ((time.time()-t0)/60), BIN)
print(sh("%s --version" % BIN, quiet=True).stdout.strip())
""")

# --------------------------------------------------------------------------
md(r"""## 3. The runner

Leaner than the Lookup notebook's — this run cares about **wall, peak host RSS,
peak VRAM and the phase split**, and about knobs announcing themselves.

One thing carried over verbatim, because it cost an hour the first time:
`RNA_GPU_CHUNK` is the **master switch**, not a cap. Unset means *fold on the
CPU*.""")

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
MDBS_RE  = re.compile(r"modular_decomposition_kernel block size (\d+)")
AUTO_RE  = re.compile(r"build pipeline AUTO: (ON|off) -- next chunk needs "
                      r"~([\d.]+) GB, MemAvailable ([\d.]+) GB")
PHASES = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
          "new_c_host","fml_host","fml_prev_host")
STAGES = ("build","prepare","prefill","backtrack","output","gpuinit","teardown","free")

RESULTS = {}
OUT = "/content/scaling.json"
os.makedirs("/content/clk", exist_ok=True)

def save():
    with open(OUT, "w") as f:
        json.dump({"commit": COMMIT, "nproc": NPROC, "runs": RESULTS}, f, indent=1)

def vram_peak(path):
    hi = 0.0
    try:
        for line in open(path):
            q = [x.strip() for x in line.split(",")]
            if len(q) >= 2:
                try: hi = max(hi, float(q[1]))
                except ValueError: pass
    except OSError: pass
    return hi

def run(tag, fa, chunk_cap=0, vram_mb=None, pipeline=None, phase_sync=True,
        block_size=None, md_block=None, host_avail_mb=None,
        int16=False, extra_args="", quiet=False):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH","RNA_PHASE_SYNC",
              "RNA_GPU_VRAM_BUDGET_MB","RNA_INT_LOOP_BLOCK_SIZE","RNA_BUILD_PIPELINE",
              "RNA_MD_BLOCK_SIZE","RNA_INT_LOOP_WARP","RNA_BUILD_THREADS",
              "RNA_HOST_AVAIL_MB","RNA_MD_TILE"):
        env.pop(k, None)
    env["RNA_MIN_GPU_BATCH"] = "1"
    env["RNA_GPU_CHUNK"] = "0" if chunk_cap is None else str(chunk_cap)
    if phase_sync:        env["RNA_PHASE_SYNC"] = "1"
    if int16:             env["RNA_FML_INT16"] = "1"
    if block_size:        env["RNA_INT_LOOP_BLOCK_SIZE"] = str(block_size)
    if md_block:          env["RNA_MD_BLOCK_SIZE"] = str(md_block)
    if vram_mb is not None:       env["RNA_GPU_VRAM_BUDGET_MB"] = str(vram_mb)
    if host_avail_mb is not None: env["RNA_HOST_AVAIL_MB"] = str(host_avail_mb)
    # unset means AUTO since 32.4, so an arm that wants it OFF must say so
    if pipeline is not None: env["RNA_BUILD_PIPELINE"] = "1" if pipeline else "0"

    clk = "/content/clk/%s.csv" % tag
    smi = subprocess.Popen(["nvidia-smi",
                            "--query-gpu=clocks.sm,memory.used,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits", "-lms", "250"],
                           stdout=open(clk, "w"), stderr=subprocess.DEVNULL)
    t0 = time.time()
    p  = subprocess.run((TIME_BIN + BIN + " --noPS " + extra_args + " -i " + fa).split(),
                        capture_output=True, text=True, env=env)
    wall = time.time() - t0
    smi.terminate()
    try: smi.wait(timeout=5)
    except Exception: smi.kill()
    err = p.stderr

    if p.returncode != 0:
        print(err[-3000:]); raise SystemExit("%s: rc=%d" % (tag, p.returncode))
    if "sweep shape:" not in err:
        raise SystemExit("%s: NO 'sweep shape:' -- folded on the CPU" % tag)
    if block_size:
        g = BS_RE.search(err)
        if (not g) or int(g.group(1)) != block_size:
            raise SystemExit("%s: int_loop block size %s asked, %s reported"
                             % (tag, block_size, g and g.group(1)))
    if md_block:
        g = MDBS_RE.search(err)
        if (not g) or int(g.group(1)) != md_block:
            raise SystemExit("%s: md block size %s asked, %s reported"
                             % (tag, md_block, g and g.group(1)))

    ph = dict(zip(PHASES,[float(x) for x in PHASE_RE.search(err).groups()])) if PHASE_RE.search(err) else {}
    st = dict(zip(STAGES,[float(x) for x in STAGE_RE.search(err).groups()])) if STAGE_RE.search(err) else {}
    shapes = SWEEP_RE.findall(err)
    rss = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
    if m: rss = int(m.group(1))/1e6
    a = AUTO_RE.search(err)
    r = dict(wall=wall, phases=ph, stages=st, chunks=len(shapes),
             cells=sum(int(s[2]) for s in shapes), rss=rss, vram_mb=vram_peak(clk),
             sha=__import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12],
             fa=os.path.basename(fa), chunk_cap=chunk_cap, budget_mb=vram_mb,
             pipeline=pipeline, phase_sync=phase_sync, block_size=block_size,
             md_block=md_block, int16=int16,
             auto_verdict=a.group(1) if a else None,
             auto_need_gb=float(a.group(2)) if a else None,
             auto_avail_gb=float(a.group(3)) if a else None)
    RESULTS[tag] = r; save()
    if not quiet:
        print("  %-20s wall %7.2f  md %7.2f  il %6.2f  RSS %6.2f GB  VRAM %6.0f MB  sha %s"
              % (tag, wall, ph.get("modular_decomp",0), ph.get("int_loop",0),
                 rss, r["vram_mb"], r["sha"]))
    return r

def fasta(name, n, L, seed=None):
    path = "%s/fa/%s.fa" % (ROOT, name)
    os.makedirs(ROOT + "/fa", exist_ok=True)
    if not os.path.exists(path):
        random.seed(seed if seed is not None else (n*100003 + L))
        with open(path, "w") as f:
            for i in range(n):
                f.write(">%s_%d\n%s\n" % (name, i, "".join(random.choice("ACGU") for _ in range(L))))
    return path

print("runner ready ->", OUT)
""")

code(r"""
print("warm-up (discarded): the local box needed FOUR folds to reach steady clocks")
W = fasta("warm", 24, 1200)
for i in range(3):
    run("WARM_%d" % i, W, quiet=True)
print("  warm")
""")

# --------------------------------------------------------------------------
md(r"""## A. What a record of length *L* actually costs the host

`RNAfold.c`'s AUTO gate charges **2·(L+1)² bytes per record** and enables the
build pipeline only when the next chunk fits in half of `MemAvailable`. That
coefficient is the *structural* upper bound — the dense hard-constraint matrix
plus the triangular ptype — and §33.5 measured the real extra residency at
**6.40 GB against the 15.31 GB it charged**, so a 32 GB host is declined a
pipeline that would fit and pays 5.7 % for it.

**One point cannot fit a coefficient.** This sweeps records × length with the
pipeline forced OFF and forced ON, and reports the *difference* — which is what
the second chunk actually costs — per record and divided by L².""")

code(r"""
print("A: host residency vs records x length (pipeline off vs on)")
A_GRID = [(40, 1200), (40, 2400), (40, 4800), (20, 5601), (20, 8000), (10, 12000)]
for n, L in A_GRID:
    fa = fasta("a_%dx%d" % (n, L), n, L)
    for pipe in (False, True):
        run("A_%dx%d_%s" % (n, L, "on" if pipe else "off"), fa,
            pipeline=pipe, phase_sync=False)
""")

code(r"""
rows = []
for n, L in A_GRID:
    a = RESULTS.get("A_%dx%d_off" % (n, L)); b = RESULTS.get("A_%dx%d_on" % (n, L))
    if not (a and b): continue
    per_rec = (b["rss"] - a["rss"]) * 1e9 / max(n, 1)
    rows.append((n, L, a["rss"], b["rss"], per_rec, per_rec / (float(L)+1)**2,
                 b["auto_need_gb"], a["sha"] == b["sha"]))
print("  %5s %7s %9s %9s %12s %10s %10s %s"
      % ("n","L","RSS off","RSS on","extra/record","bytes/L^2","AUTO says","sha"))
for n, L, ro, rn, pr, c, need, same in rows:
    print("  %5d %7d %8.2fG %8.2fG %10.1f MB %10.2f %8.2f GB %s"
          % (n, L, ro, rn, pr/1e6, c, need or 0.0, "same" if same else "*** MOVED ***"))
if rows:
    cs = [c for *_, c, _, _ in [(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]) for r in rows]]
    cs = [r[5] for r in rows]
    print()
    print("  the gate charges 2.00 bytes per L^2 per record.")
    print("  measured: min %.2f, median %.2f, max %.2f"
          % (min(cs), sorted(cs)[len(cs)//2], max(cs)))
    print()
    print("  A COEFFICIENT THAT DRIFTS WITH LENGTH is the finding, not an average:")
    print("  the estimator is a straight line through the origin and the data may not be.")
""")

# --------------------------------------------------------------------------
md(r"""## B. Cache hit rates as a function of sequence length

Quoted per kernel and never per length, yet the sweep's reuse is entirely a
function of row width: `modular_decomp` re-reads a whole `fml_i` row per cell,
`int_loop` walks a bounded window. Somewhere the working set stops fitting in
L2, and that length is worth knowing — it is the length at which "add more
parallelism" stops being free.

Both kernels, five lengths, same record count so the only variable is *L*.""")

code(r"""
SECT  = "--section WarpStateStats --section SchedulerStats --section Occupancy"
BASE_METRICS = ",".join([
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
    # D3: average ACTIVE LANES per issued instruction. 32 is a full warp; this
    # is the number PORT_SCOUT_COMPACTION_SCOPE.md turns on.
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    # D2: the unexplained one.
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
])

NCU = {}
def profile(tag, kernel, fa, skip, count=3, env_extra=None, store=None):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_PHASE_SYNC","RNA_MD_TILE","RNA_INT_LOOP_BLOCK_SIZE",
              "RNA_MD_BLOCK_SIZE","RNA_INT_LOOP_CELLS_PER_BLOCK","RNA_BUILD_PIPELINE"):
        env.pop(k, None)
    env.update(RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1", RNA_BUILD_PIPELINE="0")
    env.update(env_extra or {})
    sink = NCU if store is None else store
    out  = "/content/ncu_%s.csv" % tag
    want = kernel.lstrip("^")
    for flavour, extra in (("sections", SECT + " --metrics " + BASE_METRICS),
                           ("metrics",  "--metrics " + BASE_METRICS)):
        cmd = ("ncu --target-processes all --csv %s -k regex:'%s' --launch-skip %d "
               "--launch-count %d --log-file %s %s --noPS -i %s > /dev/null 2>&1"
               % (extra, kernel, skip, count, out, BIN, fa))
        subprocess.run(cmd, shell=True, env=env)
        body = open(out).read() if os.path.exists(out) else ""
        if ("No kernels were profiled" in body) or ("Metric Name" not in body): continue
        rows = list(_csvmod.DictReader(io.StringIO(
            "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
        names = set(r.get("Kernel Name","") for r in rows)
        if names and not any(want in nm for nm in names):
            print("  %-16s *** WRONG KERNEL: %r" % (tag, sorted(names)[:1])); return {}
        agg = collections.defaultdict(list)
        for r in rows:
            try: agg[r["Metric Name"]].append(float((r["Metric Value"] or "").replace(",","")))
            except Exception: pass
        res = {k: sum(v)/len(v) for k, v in agg.items() if v}
        if res:
            print("  %-16s ok via %s" % (tag, flavour)); sink[tag] = res; return res
    print("  %-16s *** NO METRICS -- a broken probe, not a result" % tag)
    return {}

print("B: cache and geometry vs length, both kernels")
B_LENGTHS = [600, 1200, 2400, 4800, 8000]
for L in B_LENGTHS:
    fa = fasta("b_%d" % L, 24, L)
    # skip into the middle of the sweep: the first rows are degenerate
    profile("md_L%d" % L, "modular_decomposition_kernel", fa, skip=max(20, L//8))
    profile("il_L%d" % L, "int_loop_warp_kernel",         fa, skip=max(20, L//8))
with open("/content/scaling_ncu.json","w") as f: json.dump(NCU, f, indent=1)
""")

code(r"""
def show(prefix, title):
    rows = [(L, NCU["%s_L%d" % (prefix, L)]) for L in B_LENGTHS
            if "%s_L%d" % (prefix, L) in NCU]
    if not rows: print("  no data for", prefix); return
    print(title)
    print("  %6s %8s %8s %9s %8s %8s %9s %8s %9s"
          % ("L","L1 %","L2 %","sect/req","DRAM %","SM %","occup %","waves","lanes/inst"))
    for L, r in rows:
        print("  %6d %7.1f%% %7.1f%% %9.2f %7.2f%% %7.2f%% %8.1f%% %8.2f %9.2f"
              % (L, r.get("l1tex__t_sector_hit_rate.pct",0),
                 r.get("lts__t_sector_hit_rate.pct",0),
                 r.get("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",0),
                 r.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",0),
                 r.get("sm__throughput.avg.pct_of_peak_sustained_elapsed",0),
                 r.get("sm__warps_active.avg.pct_of_peak_sustained_active",0),
                 r.get("launch__waves_per_multiprocessor",0),
                 r.get("smsp__thread_inst_executed_per_inst_executed.ratio",0)))
    print()

show("md", "--- modular_decomposition_kernel ---")
show("il", "--- int_loop_warp_kernel ---")
print("  L2 falling with length = the working set outgrew it, and THAT length is")
print("  the answer to 'when does adding parallelism stop being free'.")
print("  L1 flat while L2 falls = the reuse is intra-cell and survives; the")
print("  cross-cell reuse is what is being lost.")
""")

# --------------------------------------------------------------------------
md(r"""## C. Block size c·32, and what binds

Neither hot kernel's block size has been swept on this card since the warp
kernel became the default, and the interesting output is not the winner but the
**limiter**. `ncu` reports, per resource, how many blocks an SM could hold:
`launch__occupancy_limit_blocks / registers / shared_mem / warps`. The smallest
of those four IS the obstacle, by name.

§33.1 already showed what this looks like when it bites: `modular_decomp` runs
768-thread blocks at 40 registers and its limiters read **32/2/8/2** — two
blocks per SM, capped by registers *and* by the warp budget, which is 48 of a
possible 64 warps before a single instruction issues.

**For `int_loop` the block size IS the c in c·32.** `int_loop.cu:1972` computes
`cpb = block_size / 32` and the warp kernel gives one cell to each warp, so
asking for 32, 64, 128, 256 is asking for 1, 2, 4, 8 cells per block. That is
the sweep the question names, and it is one knob, not two.""")

code(r"""
print("C: block-size sweeps, timing arm (24 x 2400, phase-synced so the phase is the measurement)")
C_FA = fasta("c_2400", 24, 2400)
for bs in (32, 64, 128, 256, 512, 1024):   # c*32 for c = 1,2,4,8,16,32
    try: run("C_il_bs%d" % bs, C_FA, block_size=bs)
    except SystemExit as e: print("  bs %-4d refused: %s" % (bs, e))
for mb in (64, 128, 256, 512, 768, 1024):
    try: run("C_md_bs%d" % mb, C_FA, md_block=mb)
    except SystemExit as e: print("  md bs %-4d refused: %s" % (mb, e))
""")

code(r"""
print("C: the same sweep under ncu -- the LIMITER, which timing cannot show")
for bs in (32, 128, 512):
    profile("Cil_bs%d" % bs, "int_loop", C_FA, skip=200,
            env_extra={"RNA_INT_LOOP_BLOCK_SIZE": str(bs)})
for mb in (128, 512, 1024):
    profile("Cmd_bs%d" % mb, "modular_decomposition_kernel", C_FA, skip=200,
            env_extra={"RNA_MD_BLOCK_SIZE": str(mb)})
""")

code(r"""
def sweep_table(prefix, label, key):
    rows = [(k, v) for k, v in sorted(RESULTS.items()) if k.startswith(prefix)]
    if not rows: return
    base = rows[0][1][key] if rows else 0
    print(label)
    print("  %-12s %9s %9s %9s %s" % ("arm","wall","int_loop","md","sha"))
    for k, v in rows:
        print("  %-12s %9.2f %9.2f %9.2f %s"
              % (k[2:], v["wall"], v["phases"].get("int_loop",0),
                 v["phases"].get("modular_decomp",0), v["sha"]))
    shas = set(v["sha"] for _, v in rows)
    print("  shas:", shas, "" if len(shas) == 1 else "*** A BLOCK SIZE CHANGED THE ANSWER ***")
    print()

sweep_table("C_il_bs",  "--- int_loop, classic block size ---", "block_size")
sweep_table("C_md_bs",  "--- modular_decomp block size ---", "md_block")

print("--- the LIMITER, per arm (blocks/registers/shared/warps; the SMALLEST binds) ---")
print("  %-12s %6s %6s %9s %9s %8s %s"
      % ("arm","blk","regs","occup %","waves","lanes","limits blk/reg/smem/warp"))
for tag in sorted(k for k in NCU if k.startswith(("Cil_","Cmd_"))):
    r = NCU[tag]
    lim = [int(r.get("launch__occupancy_limit_"+k,0)) for k in ("blocks","registers","shared_mem","warps")]
    names = ("blocks","registers","shared_mem","warps")
    binds = names[lim.index(min(lim))] if min(lim) > 0 else "?"
    print("  %-12s %6d %6d %8.1f%% %9.2f %8.2f %-18s <- %s"
          % (tag, int(r.get("launch__block_size",0)), int(r.get("launch__registers_per_thread",0)),
             r.get("sm__warps_active.avg.pct_of_peak_sustained_active",0),
             r.get("launch__waves_per_multiprocessor",0),
             r.get("smsp__thread_inst_executed_per_inst_executed.ratio",0),
             "/".join(str(x) for x in lim), binds))
print()
print("  THE OBSTACLE IS THE SMALLEST COLUMN. registers -> spill or shrink the")
print("  kernel; warps -> the block is too wide to pack; blocks -> the 32-per-SM")
print("  hardware limit, which no code change can move.")
""")

# --------------------------------------------------------------------------
md(r"""## D. The three things §33 could not answer

**D1 — waves at the production shape.** §33.1 measured 0.66 waves/SM on a
60 × 1800 profiling fixture and said in print that it is a fixture property, not
a production claim. This profiles the same kernel at 200 × 5601, which is the
shape the 85.9 s wall comes from.

**D2 — `imc_miss`.** 3.25 stalls per issue in `modular_decomp` against 0.30 in
`int_loop`. Immediate-constant misses: the kernels read their parameter tables
through `__constant__`, and `modular_decomp` reads a different set. The sweep in
§B reports it per length, which distinguishes "a property of the kernel" from
"a property of how much else is in flight".

**D3 — how full are the warps?** `smsp__thread_inst_executed_per_inst_executed`
is the average number of ACTIVE LANES per issued instruction; 32 is a full warp.
`PORT_SCOUT_COMPACTION_SCOPE.md` proposes grouping the cells that need real work
so that a warp is not half-idle behind one sprawling cell. **If this number is
already near 32, that scope has no prize and should be closed.**""")

code(r"""
print("D: production shape, 200 x 5601 -- the shape the 85.9 s wall comes from")
D_FA = fasta("d_prod", 200, 5601)
for tag, kern, skip in (("D_md_prod", "modular_decomposition_kernel", 2000),
                        ("D_il_prod", "int_loop_warp_kernel",         2000)):
    profile(tag, kern, D_FA, skip=skip, count=3)
""")

code(r"""
print("--- D1/D2/D3, production shape against the profiling fixture ---")
print("  %-14s %8s %9s %9s %9s %9s"
      % ("arm","waves","lanes/inst","imc_miss","long_sb","occup %"))
for tag in ("D_md_prod","D_il_prod"):
    r = NCU.get(tag)
    if not r: continue
    print("  %-14s %8.2f %9.2f %9.3f %9.3f %8.1f%%"
          % (tag, r.get("launch__waves_per_multiprocessor",0),
             r.get("smsp__thread_inst_executed_per_inst_executed.ratio",0),
             r.get("smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",0),
             r.get("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",0),
             r.get("sm__warps_active.avg.pct_of_peak_sustained_active",0)))
print()
print("  33.1 measured, on a 60 x 1800 fixture: md waves 0.66, imc_miss 3.247,")
print("  long_scoreboard 11.667, occupancy 46.1%. The comparison IS the result.")
print()
r = NCU.get("D_md_prod", {})
w = r.get("launch__waves_per_multiprocessor", 0)
lanes = r.get("smsp__thread_inst_executed_per_inst_executed.ratio", 0)
if w:
    print("  D1: %.2f waves/SM in production." % w,
          "Under one wave -- the grid cannot fill the card." if w < 1.0
          else "More than one wave: the 0.66 was the fixture, and grid size is NOT the problem.")
if lanes:
    print("  D3: %.1f of 32 lanes active per instruction." % lanes,
          "Warps are already full -- CLOSE the compaction scope." if lanes > 28
          else "Lanes are idling -- the compaction scope has a measurable prize.")
""")

# --------------------------------------------------------------------------
md("## E. Summary and export")

code(r"""
print("commit", COMMIT, "| cores", NPROC)
print(sh("nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,memory.total "
         "--format=csv,noheader", quiet=True).stdout.strip())
print()
byfa = collections.defaultdict(set)
for k, v in RESULTS.items():
    if not k.startswith("WARM"): byfa[v["fa"]].add(v["sha"])
for fa, shas in sorted(byfa.items()):
    print("  %-18s %d distinct sha %s  %s"
          % (fa, len(shas), sorted(shas)[:2], "OK" if len(shas) == 1 else "*** MISMATCH ***"))
save()
with open("/content/scaling_ncu.json","w") as f: json.dump(NCU, f, indent=1)
print()
print("wrote", OUT, "and /content/scaling_ncu.json")
""")

code(r"""
from google.colab import files
for f in (OUT, "/content/scaling_ncu.json"):
    try: files.download(f)
    except Exception as e: print("download failed:", f, e)
""")

md(r"""### The decision table, written before the run

| § | if it comes back… | then |
|---|---|---|
| **A** | bytes/L² flat near 1.0 | re-base the AUTO gate on the measured coefficient; a 32 GB host gets its pipeline back |
| **A** | bytes/L² **drifts with length** | the estimator cannot be one coefficient — model it, do not average it |
| **A** | a sha moves between pipeline off and on | **stop.** The pipeline is not answer-neutral and the default must come back off |
| **B** | L2 falls with length | that length is where cross-cell reuse dies, and it bounds every "more parallelism" idea |
| **B** | L1 and L2 both flat | cache is not the length story; the length story is grid size, and §D says so directly |
| **C** | one limiter dominates every arm | that resource is the obstacle, and it is the only thing worth attacking |
| **C** | a block size changes a sha | **stop.** Nothing else in this notebook matters. |
| **D1** | ≥ 1 wave in production | grid size is not the problem; §33.1's 0.66 was the fixture, exactly as flagged |
| **D3** | lanes/instruction > 28 | **close `PORT_SCOUT_COMPACTION_SCOPE.md`** — the warps are already full |
| **D3** | lanes/instruction < 24 | the compaction scope has a measured prize; design it |
""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Scaling.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (out, len(cells)))

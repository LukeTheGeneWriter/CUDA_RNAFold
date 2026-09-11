#!/usr/bin/env python3
"""Build the IntLoop2 notebook: chase the device-level int16 slowdown, and test
three proposed levers before building any of them.

WHAT THIS IS FOR. STRESS272_RESULTS.md §20 closed the block-size question (64,
promoted) and left exactly one open:

  * NCU says `int_loop_kernel` is IDENTICAL under int16 and int32 -- 717,676.8 ns
    against 717,683.2 ns, every counter matching -- while the `int_loop` PHASE is
    25 s slower under int16 at 400 x 5601.
  * At scale EVERY phase except `modular_decomp` is slower under int16, including
    `hp_mb` (+7.5%) and `load_my_c` (+5.3%), neither of which touches int16 data.

That is a device-level signature, not an attribution artifact -- all four phases
already end in syncs. §20.4 named two probes that separate the candidates, and
neither existed:

  P1  per-launch DEVICE time, so a phase total splits into the kernel and
      everything else.  ->  RNA_LAUNCH_STATS (device.cu), new.
  P2  SM clock sampled DURING the run.  The IntLoop notebook sampled it BEFORE
      and AFTER each arm, i.e. while the GPU was IDLE, so its 0.19-0.78 spread
      measured idle clock states and could not answer the question it looked
      like it answered.  ->  a sampler that runs concurrently, here.

AND THREE LEVERS, EACH WITH ITS PREMISE CHECKED FIRST. This is the part worth
being careful about: two of the three are answers to a question the profile says
the kernel is not asking.

  shared-memory staging   RNA_MD_SMEM, implemented and gated off. Targets
                          `fml_i` (22 KB, read O(n^2) times) on the premise that
                          modular_decomposition_kernel's ~6% L2 hit means the
                          `fml_j` stream is evicting it. §B checks that premise
                          directly (DRAM bytes against bytes requested) before
                          §C spends an hour on the A/B.
  coalescing              MEASURED, not assumed. With TILE=32 adjacent lanes
                          already walk adjacent y in both streams, so the
                          expectation is ~4 sectors/request and nothing to win.
                          §B and §E report it for both kernels.
  int16 `my_c`            NOT built. It is a large change and §D measures its
                          CEILING first, two ways: int_loop_kernel runs at
                          4.2-5.2% of DRAM peak, so halving a buffer it reads
                          cannot speed it up; and its real value -- more records
                          per chunk -- is emulated directly with RNA_GPU_CHUNK.
                          §20.2 already measured chunk 29->37 costing +3.7 s, so
                          this may be worth nothing before a line is written.

RULES, unchanged from the last notebook and re-stated in the notebook itself:
sha must be 7c0b3d633281 in every arm, `sweep shape:` must be present, and
phase-synced numbers are never compared against async ones.

RUNTIME. ~9 min per 400 x 5601 arm, and §A's arms are slower still because
RNA_LAUNCH_STATS syncs on an event every launch. A: 4 arms, B: minutes,
C: 4 arms, D: 4 arms, E: minutes. About 2.5 h after a ~10 min build. EVERY ARM
SAVES AS IT COMPLETES and the sections are ordered by value.

usage: python3 tools/make_nb_intloop2.py [out.ipynb]
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
md(r"""# IntLoop 2: the device-level int16 slowdown, and three levers with their premises checked

`STRESS272_RESULTS.md` §20 settled block size (64, promoted) and left **one**
question open. It is not a kernel question.

## The finding this notebook exists to explain

NCU profiled `int_loop_kernel` for the first time and found it **identical**
under int16 and int32:

| | i32 | i16 |
|---|---|---|
| duration, cap 24, mean of 5 launches | 717 676.8 ns | 717 683.2 ns |
| occupancy / SM% / DRAM% / L1 / L2 | 47.74 / 44.22 / 4.81 / 76.21 / 90.44 | 47.75 / 44.15 / 4.81 / 76.16 / 90.30 |

…while the **phase** is 25 s slower. And at scale it is not alone:

| phase | i32 | i16 | delta |
|---|---|---|---|
| `modular_decomp` | 215.41 | 148.87 | **−30.9 %** |
| `int_loop` | 102.95 | 128.30 | **+24.6 %** |
| `fetch_mx` | 8.59 | 10.99 | +27.9 % *(explained: host-side decode)* |
| `hp_mb` | 14.25 | 15.32 | **+7.5 %** *(touches no int16 data)* |
| `load_my_c` | 10.58 | 11.14 | **+5.3 %** *(touches no int16 data)* |

Every phase already ends in a sync, so this is **not** §19's attribution
artifact. Something about the device differs between the two arms.

## The two probes, and why the last run could not do this

**P1 — per-launch device time.** New: `RNA_LAUNCH_STATS=1` brackets
`int_loop_kernel` alone with CUDA events and records `(grid, device_ms,
host_ms)` for every one of ~78 000 launches. The decisive plot is **device_ms
against grid, i32 over i16**. If the curves coincide the kernel really is
identical in situ and the 25 s is launch overhead or device state;
`sum(device_ms)` against the phase total says how much.

**P2 — SM clock during the run.** The previous notebook sampled
`clocks.sm / clocks.max.sm` *before and after* each arm — **while the GPU was
idle**. Those numbers (0.19–0.78, no pattern) measured idle clock states. Here a
sampler runs **concurrently** at 4 Hz and reports the clock, power, temperature
and throttle reasons the arm actually ran at.

## The three levers — and the premise check each one gets first

| lever | status | premise, and where it is checked |
|---|---|---|
| shared-memory staging of `fml_i` | **built**, `RNA_MD_SMEM=1`, off by default | that `modular_decomposition_kernel`'s ~6 % L2 hit means the `fml_j` stream evicts the 22 KB `fml_i`. **§B** measures DRAM bytes against bytes requested; **§C** runs the A/B. |
| coalescing | **measured, not assumed** | with `TILE=32` adjacent lanes already walk adjacent `y` in *both* streams, so ~4 sectors/request is expected and there is nothing to win. **§B**, **§E**. |
| int16 `my_c` | **not built** | it is a large change. **§D** measures its ceiling first: `int_loop_kernel` runs at **4.2–5.2 % of DRAM peak**, so halving a buffer it reads cannot speed it up, and its real value — more records per chunk — is emulated directly with `RNA_GPU_CHUNK`. §20.2 already measured chunk 29→37 *costing* 3.7 s. |

That ordering is the point. Two of the three are plausible-sounding answers to a
question the profile says this kernel is not asking, and each gets a cheap test
of its premise before an expensive test of itself.

## Rules that make the numbers mean anything

* **`sha` must be `7c0b3d633281` in every arm.** A lever that changes the answer
  is a bug, and a bigger finding than any timing.
* **`sweep shape:` must be present**, or the run folded on the CPU.
* **Phase-synced against phase-synced, never against async.**
* **§A's arms are slower than everything else** — `RNA_LAUNCH_STATS` syncs on an
  event every launch. Compare §A arms only with each other.
* **`modular_decomp` is a free control** whenever the knob cannot reach it, and
  `int_loop` is the control in §C for the same reason. Report the control.
""")

# --------------------------------------------------------------------------
md("## 1. Environment, and the concurrent clock sampler (probe P2)")

code(r"""
import subprocess, os, sys, json, time, re, random, io, collections, csv as _csvmod

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

print(clocks())
print("host RAM:", sh("free -g | awk '/^Mem/{print $2\" GB\"}'", quiet=True).stdout.strip())
print("cores   :", sh("nproc", quiet=True).stdout.strip())
""")

md(r"""### The sampler

**This is probe P2 and the previous notebook's version was not.** It sampled the
clock ratio *before* and *after* each run — while the GPU was idle — and those
numbers looked like a clock measurement without being one. This runs alongside
the arm at 4 Hz and reports what the GPU was actually doing while the work ran.

`clocks_throttle_reasons.active` is a bitmask; the two that matter here are
`0x4` (SW power cap) and `0x8` (HW slowdown / thermal). A T4 is a 70 W card in a
passively-cooled chassis, so a power cap is the leading hypothesis for a
device-wide slowdown in the arm that finishes its `modular_decomp` work sooner
and therefore asks more of the device per unit time.""")

code(r"""
class ClockSampler(object):
    \"\"\"Samples the GPU at 4 Hz for the LIFETIME OF AN ARM, not around it.\"\"\"
    FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
              "utilization.gpu", "utilization.memory", "clocks_throttle_reasons.active")

    def __init__(self, path):
        self.path = path
        self.p = None

    def __enter__(self):
        q = ",".join(self.FIELDS)
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=" + q,
             "--format=csv,noheader,nounits", "-lms", "250"],
            stdout=self.f, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            try: self.p.wait(timeout=5)
            except Exception: self.p.kill()
        self.f.close()
        return False

    def summary(self):
        rows = []
        for line in open(self.path):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != len(self.FIELDS):
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4]), float(parts[5]), parts[6]))
            except ValueError:
                continue
        if not rows:
            return {}
        # Only samples where the GPU was actually busy describe the arm. An
        # idle tail (teardown, output) would drag the mean clock down and
        # manufacture a throttling story -- the exact error the previous
        # notebook made in the other direction.
        busy = [r for r in rows if r[4] >= 50.0] or rows
        sm = [r[0] for r in busy]
        thr = {}
        for r in busy:
            thr[r[6]] = thr.get(r[6], 0) + 1
        return dict(n=len(rows), n_busy=len(busy),
                    sm_mean=sum(sm)/len(sm), sm_min=min(sm), sm_max=max(sm),
                    temp_max=max(r[2] for r in busy),
                    power_mean=sum(r[3] for r in busy)/len(busy),
                    power_max=max(r[3] for r in busy),
                    throttle=sorted(thr.items(), key=lambda kv: -kv[1])[:3])

print("sampler ready")
""".replace('\\"\\"\\"', '"""'))

code(r"""
sh("apt-get -qq update && apt-get -qq install -y gengetopt help2man xxd > /dev/null 2>&1",
   check=False, quiet=True)
print("deps ok")
""")

# --------------------------------------------------------------------------
md(r"""## 2. Build, and refuse a clone that predates what we are measuring

`RNA_LAUNCH_STATS` and `RNA_MD_SMEM` are both new. A tree without them would run
the arms and report *nothing*, which reads as "the effect is absent" rather than
"you cloned a stale tree" — the failure mode this project has hit eleven times.""")

code(r"""
REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/intloop2"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())

# Feature probes, not commit hashes: a hash goes stale the moment the branch is
# rebased, and what matters is whether the knob EXISTS.
SRC = ROOT + "/port27/src/ViennaRNA/mfe/cuda/"
NEED = [("RNA_LAUNCH_STATS", SRC + "device.cu",                 "probe P1"),
        ("RNA_MD_SMEM",      SRC + "modular_decomposition.cu",  "the shared-memory lever"),
        ("INT_LOOP_DEFAULT_BLOCK_SIZE", SRC + "int_loop.cu",    "block size 64 promoted")]
missing = []
for tok, path, why in NEED:
    ok = os.path.exists(path) and tok in open(path).read()
    print("  %-30s %-28s %s" % (tok, why, "present" if ok else "MISSING"))
    if not ok: missing.append(tok)
if missing:
    raise SystemExit("STALE CLONE: this tree has no %s. Push port27, then re-run."
                     % ", ".join(missing))
""")

code(r"""
t0 = time.time()
for pat, flag in (("src/dlib-*.tar.bz2", "-xjf"), ("src/libsvm-*.tar.gz", "-xzf")):
    for t in sh("ls %s/port27/%s 2>/dev/null" % (ROOT, pat), check=False, quiet=True).stdout.split():
        d = re.sub(r"\.tar\.(bz2|gz)$", "", os.path.basename(t))
        if not os.path.isdir("%s/port27/src/%s" % (ROOT, d)):
            sh("tar %s %s -C %s/port27/src/" % (flag, t, ROOT), check=False, quiet=True)
sh("cd %s/port27 && ./autogen.sh > /dev/null 2>&1" % ROOT, check=False, quiet=True)
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
assert os.path.exists(BIN)
print("built in %.0fs" % (time.time() - t0))
""")

# --------------------------------------------------------------------------
md(r"""## 3. The workload — the same 400 × 5601 as every run since §14

Same seed, so `sha` is comparable with `STRESS272_RESULTS.md` §14 onward
(`7c0b3d633281`).""")

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

Every arm asserts, **before any number is believed**, that the sweep ran, that
each knob it asked for actually engaged, and that the answer did not move.
A knob that silently does nothing is the failure this project keeps re-learning:
`RNA_GPU_BLOCKING_SYNC` measured "no effect" for a week while never running at
all, and the first `--noClosingGU` bar reported `diff=0` on every arm because
`RNA_GPU_CHUNK` was unset and the accelerator was never armed.""")

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
LS_RE    = re.compile(r"RNA_LAUNCH_STATS int_loop_kernel: (\d+) launches, device "
                      r"([\d.]+) s, host ([\d.]+) s, overhead ([\d.]+) s \(([\d.]+)%\)")
LSP_RE   = re.compile(r"RNA_LAUNCH_STATS device ms: min ([\d.]+) p50 ([\d.]+) "
                      r"p90 ([\d.]+) p99 ([\d.]+) max ([\d.]+)")

PHASES = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
          "new_c_host","fml_host","fml_prev_host")
STAGES = ("build","prepare","prefill","backtrack","output","gpuinit","teardown","free")

RESULTS = {}
OUT = "/content/intloop2.json"
os.makedirs("/content/ls", exist_ok=True)
os.makedirs("/content/clk", exist_ok=True)

def save():
    with open(OUT, "w") as f:
        json.dump({"commit": COMMIT, "clocks": clocks(), "n": N_BIG, "len": LEN,
                   "runs": RESULTS}, f, indent=1)

def run(tag, int16=False, chunk_cap=0, block_size=None, phase_sync=True,
        launch_stats=False, md_smem=False, fa=None, extra_args=""):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH","RNA_PHASE_SYNC",
              "RNA_GPU_VRAM_BUDGET_MB","RNA_INT_LOOP_BLOCK_SIZE","RNA_BUILD_PIPELINE",
              "RNA_LAUNCH_STATS","RNA_LAUNCH_STATS_CSV","RNA_MD_SMEM"):
        env.pop(k, None)
    env["RNA_GPU_CHUNK"]     = str(chunk_cap)     # 0 = budget alone decides
    env["RNA_MIN_GPU_BATCH"] = "1"
    if int16:       env["RNA_FML_INT16"] = "1"
    if phase_sync:  env["RNA_PHASE_SYNC"] = "1"
    if block_size:  env["RNA_INT_LOOP_BLOCK_SIZE"] = str(block_size)
    if md_smem:     env["RNA_MD_SMEM"] = "1"
    csv = None
    if launch_stats:
        env["RNA_LAUNCH_STATS"] = "1"
        csv = "/content/ls/%s.csv" % tag
        env["RNA_LAUNCH_STATS_CSV"] = csv

    clk = "/content/clk/%s.csv" % tag
    t0 = time.time()
    with ClockSampler(clk) as sampler:
        p = subprocess.run(["/usr/bin/time","-v",BIN,"--noPS"] + extra_args.split() +
                           ["-i", fa or BIG], capture_output=True, text=True, env=env)
        wall = time.time() - t0
        clkinfo = sampler.summary()
    err = p.stderr

    # --- assertions, before any number is believed -------------------------
    if p.returncode != 0:
        print(err[-3000:]); raise SystemExit("%s: rc=%d" % (tag, p.returncode))
    if "sweep shape:" not in err:
        raise SystemExit("%s: NO 'sweep shape:' -- folded on the CPU" % tag)
    for want, token, label in ((phase_sync, "RNA_PHASE_SYNC=1", "phase sync"),
                               (int16,      "RNA_FML_INT16=1",  "int16"),
                               (launch_stats,"RNA_LAUNCH_STATS=1","launch stats"),
                               (md_smem,    "RNA_MD_SMEM=1",    "md smem")):
        if bool(want) != (token in err):
            raise SystemExit("%s: %s knob did not engage as asked (wanted %s)"
                             % (tag, label, bool(want)))
    if block_size:
        got = BS_RE.search(err)
        if (not got) or int(got.group(1)) != block_size:
            raise SystemExit("%s: block size %s requested, %s reported"
                             % (tag, block_size, got and got.group(1)))

    ph = dict(zip(PHASES, [float(x) for x in PHASE_RE.search(err).groups()])) \
         if PHASE_RE.search(err) else {}
    st = dict(zip(STAGES, [float(x) for x in STAGE_RE.search(err).groups()])) \
         if STAGE_RE.search(err) else {}
    shapes = SWEEP_RE.findall(err)
    bs = BS_RE.search(err)
    rss = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
    if m: rss = int(m.group(1))/1e6

    ls = {}
    m = LS_RE.search(err)
    if m:
        ls = dict(launches=int(m.group(1)), device_s=float(m.group(2)),
                  host_s=float(m.group(3)), overhead_s=float(m.group(4)),
                  overhead_pct=float(m.group(5)), csv=csv)
    m = LSP_RE.search(err)
    if m:
        ls.update(dict(zip(("min_ms","p50_ms","p90_ms","p99_ms","max_ms"),
                           [float(x) for x in m.groups()])))

    cells = sum(int(s[2]) for s in shapes)
    peak  = max((int(s[3]) for s in shapes), default=0)
    sha   = __import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12]
    r = dict(wall=wall, phases=ph, stages=st, chunks=len(shapes), cells=cells,
             peak_records=peak, block_size=int(bs.group(1)) if bs else None,
             int16=int16, phase_sync=phase_sync, md_smem=md_smem,
             launch_stats=ls, chunk_cap=chunk_cap, rss=rss, clock=clkinfo, sha=sha)
    RESULTS[tag] = r; save()

    warn = "" if sha == EXPECT_SHA else "   <<< SHA MOVED, STOP AND READ THIS"
    print("  %-20s int_loop %7.2f  md %7.2f  wall %7.1f  chunks %2d  "
          "sm %4.0f MHz  %.0f W  sha %s%s"
          % (tag, ph.get("int_loop",0), ph.get("modular_decomp",0), wall,
             r["chunks"], clkinfo.get("sm_mean",0), clkinfo.get("power_mean",0),
             sha, warn))
    if ls:
        print("      launches %d  device %.1f s  overhead %.1f s (%.1f%%)  "
              "p50 %.3f ms  p99 %.3f ms"
              % (ls["launches"], ls["device_s"], ls["overhead_s"],
                 ls["overhead_pct"], ls.get("p50_ms",0), ls.get("p99_ms",0)))
    if clkinfo.get("throttle"):
        print("      throttle reasons (busy samples): %s  temp max %.0f C"
              % (clkinfo["throttle"], clkinfo.get("temp_max",0)))
    return r

print("runner ready -- results stream to", OUT)
""")

# --------------------------------------------------------------------------
md(r"""## A. The decisive arms: is the int16 penalty the kernel, the launches, or the device?

Four arms, all phase-synced, all with `RNA_LAUNCH_STATS` and the concurrent
clock sampler, at a **fixed chunk cap of 29** so chunk width is not a variable
(§20.2 measured its effect at ~3.5 s and it is not what we are chasing).

**The three outcomes, decided in advance:**

| if | then |
|---|---|
| `sum(device_ms)` differs between i32 and i16 by ~25 s | the kernel IS slower in situ, and NCU's locked-clock isolation is what hid it. Look at the clock trace next. |
| `sum(device_ms)` matches but `overhead_s` differs | it is launch overhead — 78 000 launches, and something about the int16 arm makes each one cost more. |
| both match and the phase still differs | the time is in the two offset uploads that the phase contains and the probe excludes. |

Two repeats per datatype, **ABBA-ordered** (i32, i16, i16, i32) so a monotone
drift over the ~45 minutes cancels rather than being attributed to the
datatype. Do not reorder them.""")

code(r"""
print("A: per-launch device time + concurrent clocks, cap 29, ABBA")
for tag, i16 in (("A_i32_1", False), ("A_i16_1", True),
                 ("A_i16_2", True),  ("A_i32_2", False)):
    run(tag, int16=i16, chunk_cap=29, launch_stats=True)
""")

md(r"""### A.1 Read it""")

code(r"""
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

def grp(pred):
    return [v for k, v in RESULTS.items() if k.startswith("A_") and pred(v)]

a32, a16 = grp(lambda v: not v["int16"]), grp(lambda v: v["int16"])
if a32 and a16:
    print("%-22s %10s %10s %9s" % ("", "i32", "i16", "delta"))
    def row(label, f, fmt="%10.2f"):
        x, y = mean([f(v) for v in a32]), mean([f(v) for v in a16])
        print(("%-22s " + fmt + " " + fmt + " %8.1f%%")
              % (label, x, y, 100.0*(y-x)/x if x else 0.0))
    row("int_loop PHASE (s)",  lambda v: v["phases"]["int_loop"])
    row("  of which KERNEL",   lambda v: v["launch_stats"]["device_s"])
    row("  launch overhead",   lambda v: v["launch_stats"]["overhead_s"])
    row("  phase - host (s)",  lambda v: v["phases"]["int_loop"] - v["launch_stats"]["host_s"])
    row("p50 launch (ms)",     lambda v: v["launch_stats"]["p50_ms"], "%10.4f")
    row("p99 launch (ms)",     lambda v: v["launch_stats"]["p99_ms"], "%10.4f")
    row("modular_decomp (s)",  lambda v: v["phases"]["modular_decomp"])
    row("hp_mb (s)",           lambda v: v["phases"]["hp_mb"])
    row("load_my_c (s)",       lambda v: v["phases"]["load_my_c"])
    row("SM clock (MHz)",      lambda v: v["clock"].get("sm_mean", 0), "%10.0f")
    row("power (W)",           lambda v: v["clock"].get("power_mean", 0), "%10.1f")
    row("temp max (C)",        lambda v: v["clock"].get("temp_max", 0), "%10.0f")
    print()
    for v in a32 + a16:
        print("  throttle %-6s %s" % ("i16" if v["int16"] else "i32", v["clock"].get("throttle")))
""")

md(r"""### A.2 The grid-matched curve — the plot NCU cannot draw

NCU sampled five launches with clocks locked. This is all ~78 000, in situ.
Binning by grid size is what makes the two arms comparable: grid **grows**
through a sweep, so an unbinned average confounds the datatype with the row
geometry.""")

code(r"""
def load_launches(tag):
    path = RESULTS[tag]["launch_stats"].get("csv")
    if not path or not os.path.exists(path): return []
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            s, g, d, h = line.split(",")
            out.append((int(g), float(d), float(h)))
    return out

def binned(tag, nbins=12):
    rows = load_launches(tag)
    if not rows: return {}
    gmax = max(r[0] for r in rows)
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for g, d, h in rows:
        b = min(nbins-1, int(nbins * g / (gmax + 1)))
        acc[b][0] += d; acc[b][1] += g; acc[b][2] += 1
    return {b: (v[1]/v[2], v[0]/v[2], v[2]) for b, v in acc.items()}   # grid, ms, n

b32, b16 = binned("A_i32_1"), binned("A_i16_1")
if b32 and b16:
    print("%6s %12s %10s %10s %8s   %s" % ("bin", "mean grid", "i32 ms", "i16 ms", "ratio", "n"))
    for b in sorted(set(b32) & set(b16)):
        g, m32, n = b32[b]; _, m16, _ = b16[b]
        print("%6d %12.0f %10.4f %10.4f %8.3f   %d"
              % (b, g, m32, m16, (m16/m32 if m32 else 0), n))
    print()
    print("A ratio column flat at 1.00 means the kernel is identical at scale and")
    print("the 25 s is NOT in it. A ratio that climbs with grid means the opposite.")
""")

md(r"""### A.3 The clock trace, plotted

If the i16 arm sits at a lower SM clock for the same work, the device-wide
slowdown is a power cap and **not** an `int_loop` problem at all — it would be a
consequence of `modular_decomp` finishing sooner and asking more of the card per
unit time, i.e. a *cost of the win*.""")

code(r"""
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for tag, style in (("A_i32_1", "-"), ("A_i16_1", "--")):
        path = "/content/clk/%s.csv" % tag
        if not os.path.exists(path): continue
        sm, pw = [], []
        for line in open(path):
            q = [x.strip() for x in line.split(",")]
            if len(q) < 7: continue
            try:
                if float(q[4]) < 50: continue      # busy samples only
                sm.append(float(q[0])); pw.append(float(q[3]))
            except ValueError: continue
        t = [0.25*i for i in range(len(sm))]
        ax[0].plot(t, sm, style, label=tag)
        ax[1].plot(t, pw, style, label=tag)
    ax[0].set_ylabel("SM clock (MHz)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_ylabel("power (W)"); ax[1].set_xlabel("seconds into the arm (busy samples)")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("/content/clocks.png", dpi=110)
    print("wrote /content/clocks.png")
    from IPython.display import Image, display; display(Image("/content/clocks.png"))
except Exception as e:
    print("plot skipped:", e)
""")

# --------------------------------------------------------------------------
md(r"""## B. Premise check: where does `modular_decomposition_kernel`'s DRAM traffic go?

**Before** spending an hour on the shared-memory A/B, one NCU run decides
whether it can win anything.

The inner loop reads two streams per `y`:

```
fml_i[row_off_H[H] + y]          a ROW buffer  -- O(n) per record, ~22 KB at n=5601
fml_j[tri_off_H[H] + y + ij0]    a TRIANGLE    -- O(n^2) per record
```

`fml_j` has **no intra-row reuse**: cell *(i,j)* walks column *j*, and cells with
different *j* walk disjoint columns. It is streamed once per row and it is the
O(n³) DRAM traffic. Nothing can cache it and shared memory cannot help it.

`fml_i` is the opposite — every cell in the row reads a prefix of the same 22 KB
array, O(n²) reads from an O(n) array. It *ought* to be permanently resident.

**The test.** Both streams issue exactly one load per `y`, so:

| DRAM bytes ÷ bytes requested | means |
|---|---|
| ≈ **1.0** | neither stream is being cached — `fml_i` is thrashing, and staging it has up to half the traffic to win |
| ≈ **0.5** | `fml_i` is already resident; the 6 % L2 hit is the `fml_j` stream, and **shared memory has nothing to win** |

Sectors per request answers the coalescing question at the same time: 32 lanes ×
4 B = 128 B = **4 sectors** is perfect, which is what `TILE=32` should already
give.""")

code(r"""
NCU = "ncu --target-processes all --csv --page raw"
METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
])

def ncu(tag, kernel_regex, int16=False, chunk_cap=8, n=60, ln=1800,
        skip=140, count=5, md_smem=False):
    \"\"\"Profile a few mid-sweep launches. NCU does not scale -- sample.\"\"\"
    fa = "%s/fa/ncu_%d_%d.fa" % (ROOT, n, ln)
    if not os.path.exists(fa):
        random.seed(4242)
        with open(fa, "w") as f:
            for i in range(n):
                f.write(">n%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(ln))))
    env = dict(os.environ)
    env["RNA_GPU_CHUNK"] = str(chunk_cap); env["RNA_MIN_GPU_BATCH"] = "1"
    if int16:   env["RNA_FML_INT16"] = "1"
    if md_smem: env["RNA_MD_SMEM"] = "1"
    out = "/content/ncu2_%s.csv" % tag
    cmd = ("%s --metrics %s -k regex:'%s' --launch-skip %d --launch-count %d "
           "%s --noPS -i %s > %s 2>/content/ncu2_%s.err"
           % (NCU, METRICS, kernel_regex, skip, count, BIN, fa, out, tag))
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    rows = []
    try:
        _csv = _csvmod
        with open(out) as f:
            lines = [l for l in f if l.strip()]
        start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
        for rec in _csv.DictReader(lines[start:]):
            rows.append(rec)
    except Exception as e:
        print("  %s: could not parse NCU output (%s)" % (tag, e))
        print(open("/content/ncu2_%s.err" % tag).read()[-1500:])
        return {}
    agg = collections.defaultdict(list)
    for rec in rows:
        try: agg[rec["Metric Name"]].append(float(rec["Metric Value"].replace(",", "")))
        except Exception: pass
    res = {k: sum(v)/len(v) for k, v in agg.items() if v}
    RESULTS.setdefault("_ncu", {})[tag] = res; save()
    return res

print("ncu helper ready")
""".replace('\\"\\"\\"', '"""'))

code(r"""
print("B: modular_decomposition_kernel traffic, i32 vs i16")
B = {}
for tag, kw in (("md_i32", dict(int16=False)), ("md_i16", dict(int16=True))):
    B[tag] = ncu(tag, "modular_decomposition_kernel", **kw)

for tag, r in B.items():
    if not r: continue
    req  = r.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 0)
    dram = r.get("dram__bytes_read.sum", 0)
    print("%-8s requested %8.2f MB   DRAM read %8.2f MB   ratio %5.2f   "
          "sectors/req %5.2f   L1 %5.1f%%  L2 %5.1f%%  DRAM %4.1f%% of peak"
          % (tag, req/1e6, dram/1e6, (dram/req if req else 0),
             r.get("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 0),
             r.get("l1tex__t_sector_hit_rate.pct", 0),
             r.get("lts__t_sector_hit_rate.pct", 0),
             r.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 0)))
print()
print("ratio ~1.0 -> fml_i is thrashing, staging has up to half the traffic to win")
print("ratio ~0.5 -> fml_i is already resident, and section C is expected to lose")
print("sectors/request ~4.0 -> already perfectly coalesced; >4 means there is a problem")
""")

# --------------------------------------------------------------------------
md(r"""## C. The shared-memory lever, A/B'd

`RNA_MD_SMEM=1` routes to `modular_decomposition_smem_kernel`, which stages a
1024-int tile of `fml_i` into shared memory once per block and reads it from
there — turning ~24 global loads per element into one. Blocks that straddle two
records fall back to the unstaged loop (the check is block-uniform, so the
`__syncthreads()` is never divergent).

**`int_loop` is the control here** — `RNA_MD_SMEM` cannot reach it, so if it
moves, the device moved and the `modular_decomp` column is not a measurement.
Same trick that made the block-size result believable in §20.1.

ABBA, and **run this even if §B says the premise is wrong** — a measured null is
worth more than a prediction, and it is the only way the "L2 hit is 6 %, so
staging is a live lever" note in the project memory gets retired rather than
repeated.

### What a laptop already says

RTX 3050, 60 × 2400, same ABBA shape, four arms each:

| | `modular_decomp` (the lever) | `int_loop` (the control) |
|---|---|---|
| `RNA_MD_SMEM=0` | 4.547 / 4.556 / 4.544 / 4.558 → **4.551** | 8.466 – 8.527 |
| `RNA_MD_SMEM=1` | 5.230 / 5.231 / 5.243 / 5.233 → **5.234** | 8.523 – 8.539 |
| | **+15.0 %** | flat to 0.9 % |

Byte-identical in all seven configurations tried — default, int16, chunked,
`RNA_MD_TILE` 1 and 8, int16+chunked, and `-c` (which exercises the `fm2` store
in both kernels).

So on sm_86 it is a clean **15 % regression** with a flat control and a 0.3 %
spread — not noise, and not a near-miss. The T4 has different cache sizes, which
is the only reason this section still runs. §B says whether the mechanism is
what the premise claimed.""")

code(r"""
print("C: RNA_MD_SMEM A/B, i32, cap 29, ABBA (int_loop is the control)")
for tag, on in (("C_off_1", False), ("C_on_1", True),
                ("C_on_2", True),   ("C_off_2", False)):
    run(tag, int16=False, chunk_cap=29, md_smem=on)
""")

code(r"""
off = [v for k, v in RESULTS.items() if k.startswith("C_off")]
on  = [v for k, v in RESULTS.items() if k.startswith("C_on")]
if off and on:
    for label, key in (("modular_decomp (the lever)", "modular_decomp"),
                       ("int_loop (the CONTROL)",     "int_loop"),
                       ("hp_mb",                      "hp_mb")):
        x, y = mean([v["phases"][key] for v in off]), mean([v["phases"][key] for v in on])
        print("%-28s off %8.2f  on %8.2f  %+7.2f%%" % (label, x, y, 100.0*(y-x)/x))
    x, y = mean([v["wall"] for v in off]), mean([v["wall"] for v in on])
    print("%-28s off %8.1f  on %8.1f  %+7.2f%%" % ("wall", x, y, 100.0*(y-x)/x))
    shas = set(v["sha"] for v in off + on)
    print("sha across all four arms:", shas, "" if shas == {EXPECT_SHA} else "  <<< MOVED")
""")

md(r"""### C.1 If it won, profile it; if it lost, say why

Either way the NCU numbers for the staged kernel against §B's baseline say
whether the traffic actually moved — which is the difference between "it is
slower" and "it did not do what it was supposed to do".""")

code(r"""
r = ncu("md_smem_i32", "modular_decomposition_smem_kernel", int16=False, md_smem=True)
base = RESULTS.get("_ncu", {}).get("md_i32", {})
if r and base:
    for label, key, scale in (("requested (MB)", "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 1e6),
                              ("DRAM read (MB)", "dram__bytes_read.sum", 1e6),
                              ("duration (us)",  "gpu__time_duration.sum", 1e3),
                              ("L2 hit (%)",     "lts__t_sector_hit_rate.pct", 1),
                              ("sectors/request","l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 1)):
        a, b = base.get(key, 0)/scale, r.get(key, 0)/scale
        print("%-18s twin %10.2f   staged %10.2f   %+8.1f%%"
              % (label, a, b, 100.0*(b-a)/a if a else 0))
""")

# --------------------------------------------------------------------------
md(r"""## D. int16 `my_c`: measure the ceiling before building it

Two independent ceilings, neither of which needs the feature to exist.

**D.1 — the speed ceiling is ~zero, and §20.3 already implies it.**
`int_loop_kernel` runs at **4.2–5.2 % of DRAM peak**. A kernel that is nowhere
near the memory roof cannot be sped up by halving one of its inputs. This
re-measures it and adds the one number §20.3 did not have: how much of that
traffic is `my_c` at all.

**D.2 — the capacity ceiling, emulated with `RNA_GPU_CHUNK`.**
int16 `my_c` buys VRAM, and VRAM buys records per chunk. Nothing else. So the
real question is whether a wider chunk is worth anything, and that can be
answered **today** by simply asking for one.

| configuration | bytes per cell | cap at today's budget |
|---|---|---|
| i32 `my_c` + i32 fML | 8 | 29 |
| i32 `my_c` + i16 fML | 6 | ~37 *(measured, §20.2)* |
| **i16 `my_c` + i16 fML** | **4** | **~58** |

§20.2 measured chunk 29 → 37 **costing** 3.7 s of `int_loop`. If the curve keeps
rising to 58, int16 `my_c` is worth nothing before a line is written — and that
is the result to hope for, because it is the cheap one.""")

code(r"""
print("D.1: int_loop_kernel traffic -- how much is there to halve?")
for tag, kw in (("il_i32", dict(int16=False)), ("il_i16", dict(int16=True))):
    r = ncu(tag, "int_loop_kernel_.*", **kw)
    if not r: continue
    req  = r.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 0)
    dram = r.get("dram__bytes_read.sum", 0)
    dur  = r.get("gpu__time_duration.sum", 0)
    print("%-8s %8.1f us   requested %7.2f MB   DRAM %7.2f MB (%4.1f%% of peak)   "
          "sectors/req %5.2f   L1 %5.1f%%  L2 %5.1f%%"
          % (tag, dur/1e3, req/1e6, dram/1e6,
             r.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 0),
             r.get("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 0),
             r.get("l1tex__t_sector_hit_rate.pct", 0),
             r.get("lts__t_sector_hit_rate.pct", 0)))
print()
print("This is also section E: sectors/request ~4.0 means int_loop_kernel's loads")
print("are already perfectly coalesced and there is no coalescing work to do here.")
""")

code(r"""
print("D.2: does a wider chunk buy anything? (i16 fML, block size default)")
for cap in (29, 37, 45, 58):
    run("D_cap%d" % cap, int16=True, chunk_cap=cap)
""")

code(r"""
caps = sorted((v["chunk_cap"], k) for k, v in RESULTS.items() if k.startswith("D_cap"))
if caps:
    print("%6s %8s %12s %12s %12s %10s" % ("cap", "chunks", "int_loop", "mod_decomp", "wall", "ns/cell"))
    base = None
    for cap, k in caps:
        v = RESULTS[k]
        nsc = 1e9*v["phases"]["int_loop"]/v["cells"] if v["cells"] else 0
        if base is None: base = v["wall"]
        print("%6d %8d %12.2f %12.2f %12.1f %10.2f   (wall %+.1f%%)"
              % (cap, v["chunks"], v["phases"]["int_loop"], v["phases"]["modular_decomp"],
                 v["wall"], nsc, 100.0*(v["wall"]-base)/base))
    print()
    print("If wall does not FALL from cap 29 to cap 58, int16 my_c buys nothing that")
    print("matters, and the only reason left to build it is fitting a bigger record.")
""")

# --------------------------------------------------------------------------
md(r"""## E. Summary, and what each answer means

Nothing below is a conclusion the notebook draws for you — it prints the numbers
next to the decision rule that was written down **before** the run.""")

code(r"""
print("commit", COMMIT)
print(clocks())
print()
shas = sorted(set(v["sha"] for v in RESULTS.values() if isinstance(v, dict) and "sha" in v))
print("shas across every arm:", shas)
print("EXPECTED:", [EXPECT_SHA], "->", "OK" if shas == [EXPECT_SHA] else "*** SOMETHING MOVED ***")
print()
print("%-14s %8s %8s %9s %9s %7s %7s" %
      ("arm", "int_loop", "mod_dec", "wall", "kernel_s", "MHz", "W"))
for k in sorted(RESULTS):
    v = RESULTS[k]
    if not isinstance(v, dict) or "phases" not in v: continue
    print("%-14s %8.2f %8.2f %9.1f %9s %7.0f %7.1f"
          % (k, v["phases"].get("int_loop",0), v["phases"].get("modular_decomp",0),
             v["wall"],
             ("%.1f" % v["launch_stats"]["device_s"]) if v.get("launch_stats") else "-",
             v["clock"].get("sm_mean",0), v["clock"].get("power_mean",0)))
""")

code(r"""
from google.colab import files
sh("cd /content && tar czf intloop2_artifacts.tar.gz intloop2.json ls clk "
   "ncu2_*.csv clocks.png 2>/dev/null", check=False, quiet=True)
for f in ("intloop2.json", "intloop2_artifacts.tar.gz"):
    p = "/content/" + f
    if os.path.exists(p):
        print("%-30s %8.1f KB" % (f, os.path.getsize(p)/1024))
        files.download(p)
""")

md(r"""### The decision table, restated

| section | reading | what it means |
|---|---|---|
| **A** | `sum(device_ms)` matches, phase does not | the kernel is innocent; look at overhead and clocks |
| **A** | i16 SM clock materially lower | the device-wide slowdown is a power cap — a *cost of `modular_decomp`'s win*, not an `int_loop` defect |
| **A** | grid-binned ratio flat at 1.00 | NCU's isolated result holds at scale |
| **B** | DRAM ÷ requested ≈ 0.5 | `fml_i` is already resident → **retire** the "shared-memory staging is a live lever" note |
| **B/D** | sectors/request ≈ 4.0 | both kernels are already perfectly coalesced → **close** the coalescing question |
| **C** | `modular_decomp` unchanged or worse, `int_loop` control flat | the staged kernel is a measured null; keep it gated and record why |
| **D.1** | int_loop DRAM ≈ 5 % of peak | int16 `my_c` cannot speed this kernel up |
| **D.2** | wall flat or rising with cap | int16 `my_c` buys nothing → **do not build it** |

Any of these coming out the other way is a result worth having, which is why the
levers are run even where the premise check says they will lose.
""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_IntLoop2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (out, len(cells)))

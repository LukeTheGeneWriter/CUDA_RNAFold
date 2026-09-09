#!/usr/bin/env python3
"""Build the 400 x 5601 STRESS profiling notebook — full VRAM, multi-chunk.

The deep profile (2026-09-08, L4) swept sizes up to 120 x 5601 and found the
modular_decomposition share flattening near 31 % with 61 % of wall unmeasured.
Every one of those runs fit in a SINGLE CHUNK, so it could say nothing about the
regime the architecture actually exists for.

This is that regime. 400 x 5601 nt is ~50 GB of int32 triangles against 24 GB of
L4, so it MUST chunk, and chunk count is the variable the whole flatten-and-
offset design exists to minimise (~k x wall for k chunks, because the loss is
batch WIDTH rather than per-chunk overhead).

THREE THINGS THIS DOES THAT THE DEEP NOTEBOOK DID NOT:

1. It parses the STAGE line. RNAfold prints two atexit summaries -- a phase line
   (GPU and transfer work) and a stage line (build, prepare, prefill, backtrack,
   output, gpuinit, teardown, free). The deep notebook parsed only the first and
   threw the second away, which is why its "61 % unaccounted" was larger than it
   needed to be. Four of the stage counters were also dead until b799a820; on a
   binary older than that they will read 0.000, and the notebook says so rather
   than reporting a smaller residual than is real.

2. It sweeps the VRAM BUDGET to vary chunk count on a FIXED workload. That is
   the only way to price chunking without changing anything else, and it is what
   answers "what does the wall look like on a multi-batch run".

3. It accounts for ALL of wall, explicitly, and prints the residual rather than
   letting it hide. The residual is the finding; everything else is bookkeeping.

RUNTIME IS REAL: roughly 7-12 min per arm at 400 x 5601, so the default matrix
is ~1 hour after a ~10 min build. Trim BUDGETS or ARMS if the session is short.

usage: python3 tools/make_nb_stress272.py [out.ipynb]
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
md(r"""# 400 × 5601 stress profile — full VRAM, multi-chunk

The deep profile swept up to 120 × 5601 and found `modular_decomposition`
flattening near **31 %** of wall with **61 %** unmeasured. Every one of those
runs **fit in a single chunk**, so none of them said anything about the regime
the architecture exists for.

This is that regime. 400 × 5601 nt is **~50 GB of int32 triangles** against 24 GB
of L4, so it *must* chunk — and chunk count is the variable the whole
flatten-and-offset design exists to minimise (~*k*× wall for *k* chunks, because
the loss is batch **width**, not per-chunk overhead).

## What this does that the deep notebook did not

**1. It parses the stage line.** RNAfold prints *two* `atexit` summaries:

```
phase timing (s): int_loop=… hp_mb=… load_my_c=… modular_decomp=… fetch_mx=… …
stage timing (s): build=… prepare=… prefill=… backtrack=… output=… gpuinit=… teardown=… free=…
```

The deep notebook parsed only the first and discarded the second — which is why
its 61 % residual was bigger than it needed to be. **The answer was being printed
the whole time.**

> `build`, `output`, `teardown` and `free` were **dead counters** until
> `b799a820` — declared, printed, never incremented. On an older binary they read
> `0.000`, and this notebook checks the commit and says so rather than quietly
> reporting a smaller residual than is real.

**2. It sweeps the VRAM budget** to vary chunk count on a *fixed* workload. That
is the only way to price chunking without changing anything else.

**3. It accounts for all of wall and prints the residual.** The residual is the
finding; everything else is bookkeeping.

## Runtime

~7–12 min per arm, so the default matrix is about an hour after a ~10 min build.
Trim `BUDGETS` or `ARMS` if the session is short — the natural-budget i32/i16
pair is the part that must run.
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""import subprocess, os, sys, json, time, re, random, csv, io

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

def clock_ratio():
    q = sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader,nounits",
           quiet=True).stdout.strip().split(",")
    try: return float(q[0]) / float(q[1])
    except Exception: return float("nan")

print(clocks()); print("clock ratio %.2f" % clock_ratio())
print("host RAM:", sh("free -g | awk '/^Mem/{print $2\" GB\"}'", quiet=True).stdout.strip())""")

code(r"""sh("apt-get -qq update && apt-get -qq install -y gengetopt help2man texinfo "
   "doxygen autoconf automake libtool time > /dev/null 2>&1", check=False)
print("toolchain ready")""")

# --------------------------------------------------------------------------
md("""## 2. Build — and check the commit, because four counters depend on it""")

code(r"""REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/stress"
# b799a820 wired up build/output/teardown/free. Older binaries print 0.000 for
# them and the residual below will be overstated. Not fatal -- flagged, not fixed.
STAGE_COUNTERS_FROM = "b799a820"
# Commits this notebook exists to MEASURE. A clone predating any of them reports
# the pre-fix number for that stage, which reads as "the fix does nothing" rather
# than "you cloned a stale tree" -- bench v5 lost two arms exactly that way and
# scored valid:False for reasons invisible in the numbers. Add a row here
# whenever a run is meant to demonstrate a specific change.
REQUIRED_COMMITS = [
    ("a2d19bd2", "device-derived hard-constraint bitmasks (gpuinit)"),
    ("21c6f644", "prefolded fast path, no second fold compound (output)"),
]

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())
HAVE_STAGE = sh("cd %s/port27 && git merge-base --is-ancestor %s HEAD && echo yes || echo no"
                % (ROOT, STAGE_COUNTERS_FROM), check=False, quiet=True).stdout.strip() == "yes"
print("stage counters live:", HAVE_STAGE,
      "" if HAVE_STAGE else "  <-- build/output/teardown/free will read 0.000")

missing = [(c, w) for c, w in REQUIRED_COMMITS
           if sh("cd %s/port27 && git merge-base --is-ancestor %s HEAD && echo yes || echo no"
                 % (ROOT, c), check=False, quiet=True).stdout.strip() != "yes"]
for c, w in REQUIRED_COMMITS:
    print("  %s  %-52s %s" % (c, w, "MISSING" if (c, w) in missing else "present"))
if missing:
    raise SystemExit(
        "STALE CLONE: this tree predates %s.\n"
        "Those stages would come back at their pre-fix values, which reads as\n"
        "'the fix does nothing' rather than 'you cloned an old tree'.\n"
        "Push port27, then re-run the clone cell."
        % ", ".join(c for c, _ in missing))""")

code(r"""t0 = time.time()
# Git-tree build needs the bundled tarballs unpacked, and PYTHON3 passed or
# --without-python leaves $(PYTHON3) empty and make tries to EXECUTE a mode-644
# man2rst.py. Both are known defects; chmod is belt-and-braces for the second.
for pat, flag in (("src/dlib-*.tar.bz2", "-xjf"), ("src/libsvm-*.tar.gz", "-xzf")):
    for t in sh("ls %s/port27/%s 2>/dev/null" % (ROOT, pat), check=False,
                quiet=True).stdout.split():
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
print("built in %.0fs" % (time.time() - t0))""")

# --------------------------------------------------------------------------
md(r"""## 3. The workload — the standard 400 × 5601

Same seed and shape as benchmark v5, so the numbers are directly comparable to
`BENCH272_V5_RESULTS.md` (which measured 791.9 s / 5 chunks on a **T4**).""")

code(r"""N_BIG, LEN, SEED = 400, 5601, 20260907
os.makedirs(ROOT + "/fa", exist_ok=True)
BIG = ROOT + "/fa/big.fa"
random.seed(SEED)
with open(BIG, "w") as f:
    for i in range(N_BIG):
        f.write(">u%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(LEN))))
tri_gb = N_BIG * 4.0 * LEN * LEN / 1e9
print("%d x %d nt = %.1f MB of sequence" % (N_BIG, LEN, os.path.getsize(BIG)/1e6))
print("int32 triangles if resident all at once: %.0f GB -- this MUST chunk" % tri_gb)""")

# --------------------------------------------------------------------------
md(r"""## 4. The runner — every second of wall, attributed or named as residual

Both `atexit` summaries are parsed. Anything neither covers is printed as
**residual**, because a number that is not shown is a number nobody chases.""")

code(r"""PHASE_RE = re.compile(
    r"int_loop=([\d.]+) hp_mb=([\d.]+) load_my_c=([\d.]+) modular_decomp=([\d.]+) "
    r"fetch_mx=([\d.]+) \| new_c_host=([\d.]+) fml_host=([\d.]+) fml_prev_host=([\d.]+)")
STAGE_RE = re.compile(
    r"build=([\d.]+) prepare=([\d.]+) prefill=([\d.]+) backtrack=([\d.]+) "
    r"output=([\d.]+) gpuinit=([\d.]+) teardown=([\d.]+) free=([\d.]+)")
SWEEP_RE = re.compile(r"sweep shape: (\d+) iterations, (\d+) active record-rows, "
                      r"(\d+) cells; peak/iteration (\d+) records (\d+) cells")
# THE THIRD LINE. RNAfold prints a gpuinit breakdown too, and the 2026-09-09
# stress run parsed the first two and threw this one away -- leaving 22.9% of
# wall as a single unattributed number, the same mistake the deep profile made
# with the stage line. It splits gpuinit three ways and names the two suspects
# (host bitmask packing vs cudaMalloc), which is what settles it.
IG_RE = re.compile(
    r"gpuinit breakdown \(s\): init_gpu=([\d.]+) init_gpu2=([\d.]+) init_gpu3=([\d.]+) "
    r"\|\| of which pack=([\d.]+) cudaMalloc=([\d.]+) other=([\d.]+)")

PHASES = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
          "new_c_host","fml_host","fml_prev_host")
STAGES = ("build","prepare","prefill","backtrack","output","gpuinit","teardown","free")
IGPARTS = ("init_gpu","init_gpu2","init_gpu3","pack","cudaMalloc","other")

def run(fa, int16=False, budget_mb=None, chunk=0, min_batch=1):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH",
              "RNA_GPU_VRAM_BUDGET_MB","RNA_SLOT_FLOW","RNA_CONTINUOUS_FLOW"):
        env.pop(k, None)
    env["RNA_GPU_CHUNK"] = str(chunk); env["RNA_MIN_GPU_BATCH"] = str(min_batch)
    if int16:     env["RNA_FML_INT16"] = "1"
    if budget_mb: env["RNA_GPU_VRAM_BUDGET_MB"] = str(budget_mb)

    c0 = clock_ratio(); t0 = time.time()
    p = subprocess.run(["/usr/bin/time","-v",BIN,"--noPS","-i",fa],
                       capture_output=True, text=True, env=env)
    wall = time.time() - t0; c1 = clock_ratio()
    err = p.stderr

    ph = dict(zip(PHASES, [float(x) for x in PHASE_RE.search(err).groups()])) \
         if PHASE_RE.search(err) else {}
    st = dict(zip(STAGES, [float(x) for x in STAGE_RE.search(err).groups()])) \
         if STAGE_RE.search(err) else {}
    ig = dict(zip(IGPARTS, [float(x) for x in IG_RE.search(err).groups()])) \
         if IG_RE.search(err) else {}
    shapes = SWEEP_RE.findall(err)
    rss = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
    if m: rss = int(m.group(1))/1e6

    acc = sum(ph.values()) + sum(st.values())
    return dict(wall=wall, rc=p.returncode, phases=ph, stages=st, gpuinit=ig,
                chunks=len(shapes),
                cells=sum(int(s[2]) for s in shapes),
                gpu_records=sum(int(s[3]) for s in shapes),
                int16_active=("RNA_FML_INT16=1" in err),
                rss=rss, accounted=acc, residual=wall-acc,
                clock_before=c0, clock_after=c1,
                sha=__import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12])

def report(tag, r):
    print("=== %s ===  wall %.1fs   %d chunks   RSS %.1f GB   clk %.2f   sha %s"
          % (tag, r["wall"], r["chunks"], r["rss"], r["clock_after"], r["sha"]))
    for k in PHASES:
        v = r["phases"].get(k, 0.0)
        if v > 0.005: print("   phase %-16s %8.1fs %5.1f%%" % (k, v, 100*v/r["wall"]))
    for k in STAGES:
        v = r["stages"].get(k, 0.0)
        if v > 0.005: print("   stage %-16s %8.1fs %5.1f%%" % (k, v, 100*v/r["wall"]))
    ig = r.get("gpuinit") or {}
    if ig:
        # The line the 2026-09-09 run discarded. pack is the O(n^2)-per-record
        # HOST bitmask packing; it should be ~0 whenever the GPU derivation is
        # active (everything except --noLP), and dominant when it is not.
        print("     of gpuinit: %s"
              % "  ".join("%s=%.1f" % (k, ig[k]) for k in IGPARTS if ig.get(k, 0) > 0.005))
        if ig.get("gpuinit_total", r["stages"].get("gpuinit", 0)) > 0.005:
            print("     pack is %.0f%% of gpuinit"
                  % (100*ig.get("pack", 0)/max(r["stages"].get("gpuinit", 1e-9), 1e-9)))
    else:
        print("     of gpuinit: BREAKDOWN LINE NOT FOUND -- is the binary older "
              "than b799a820, or was stderr truncated?")
    print("   %-22s %8.1fs %5.1f%%   <- RESIDUAL, nothing measures this"
          % ("", r["residual"], 100*r["residual"]/r["wall"]))
print("runner ready")""")

# --------------------------------------------------------------------------
md(r"""## 5. Preflight — confirm it really chunks

If this comes back as one chunk the whole notebook is measuring the deep
profile's regime again. `RNA_MIN_GPU_BATCH=1` throughout, so no records fall to
the CPU and every second is GPU-path time.""")

code(r"""pre = run(BIG)
print("chunks: %d   gpu_records: %d/%d   wall %.1fs   RSS %.1f GB"
      % (pre["chunks"], pre["gpu_records"], N_BIG, pre["wall"], pre["rss"]))
assert pre["rc"] == 0, pre["rc"]
if pre["chunks"] <= 1:
    print("\n*** ONE CHUNK -- this card swallowed the workload whole. Lower the")
    print("    budget in BUDGETS below or raise N_BIG, or this measures nothing new.")
else:
    print("\nmulti-chunk confirmed: %d chunks at the natural budget" % pre["chunks"])
NATURAL = pre["chunks"]""")

# --------------------------------------------------------------------------
md(r"""## 6. The stress matrix

`i32` and `i16` at the natural budget, then the budget halved and quartered to
force more chunks on the *same* workload. Chunk count is the only variable that
moves between budget rows, which is what makes the comparison a price rather
than a coincidence.""")

code(r"""ARMS    = [("i32", False), ("i16", True)]
BUDGETS = [None, "half", "quarter"]     # None = the card's natural budget

total_mb = int(sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
                  quiet=True).stdout.strip())
BUD_MB = {None: None, "half": total_mb//2, "quarter": total_mb//4}

RES = {}
for b in BUDGETS:
    for tag, i16 in ARMS:
        key = "%s/%s" % (tag, b or "natural")
        r = run(BIG, int16=i16, budget_mb=BUD_MB[b])
        assert r["rc"] == 0, (key, r["rc"])
        assert r["int16_active"] == i16, "int16 gate disagreed with intent"
        RES[key] = r
        report(key, r)
        print()
        with open("/content/stress272.json","w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in RES.items()},
                      f, indent=2)""")

# --------------------------------------------------------------------------
md("## 7. What the chunking costs, and where the wall goes")

code(r"""print("%-18s %7s %9s %9s %9s %9s %9s" %
      ("arm/budget","chunks","wall","md_share","resid","backtrack","build"))
for k, r in RES.items():
    print("%-18s %7d %8.1fs %8.1f%% %8.1f%% %8.1f%% %8.1f%%"
          % (k, r["chunks"], r["wall"],
             100*r["phases"].get("modular_decomp",0)/r["wall"],
             100*r["residual"]/r["wall"],
             100*r["stages"].get("backtrack",0)/r["wall"],
             100*r["stages"].get("build",0)/r["wall"]))

print()
for b in BUDGETS:
    a, c = RES.get("i32/%s" % (b or "natural")), RES.get("i16/%s" % (b or "natural"))
    if a and c:
        print("  %-9s int16 end-to-end %.3fx   (%d vs %d chunks)"
              % (b or "natural", a["wall"]/c["wall"], a["chunks"], c["chunks"]))

base = RES.get("i32/natural")
if base:
    print()
    print("  chunking price, int32, same workload:")
    for b in BUDGETS:
        r = RES.get("i32/%s" % (b or "natural"))
        if r:
            print("    %-9s %d chunks  %8.1fs  %.3fx the natural-budget wall"
                  % (b or "natural", r["chunks"], r["wall"], r["wall"]/base["wall"]))
    print("  (k x wall for k chunks would mean the ratio tracks the chunk ratio;")
    print("   a flatter curve means chunking is cheaper here than at 900 nt.)")""")

code(r"""print("clocks:", clocks(), " ratio %.2f" % clock_ratio())
print()
r = RES.get("i32/natural")
if r:
    md_s   = 100*r["phases"].get("modular_decomp",0)/r["wall"]
    resid  = 100*r["residual"]/r["wall"]
    print("AT 400 x 5601, %d CHUNKS:" % r["chunks"])
    print("  modular_decomp   %.1f%% of wall" % md_s)
    print("  residual         %.1f%% of wall" % resid)
    print()
    if resid > 40:
        print("=> the residual SURVIVES multi-chunk. It is not chunk overhead and not")
        print("   anything the stage counters cover. Next: instrument input parsing and")
        print("   the chunk-accumulation loop, which are what is left.")
    elif md_s > 45:
        print("=> the kernel share RECOVERS at scale, so int16 and the shared-memory")
        print("   staging lever are worth more here than the 120x5601 sweep implied.")
    else:
        print("=> read the table: the stage counters absorbed most of what the deep")
        print("   profile called unaccounted, and the remainder is now small.")
with open("/content/stress272.json","w") as f:
    json.dump({"commit": COMMIT, "clocks": clocks(), "have_stage": HAVE_STAGE,
               "n": N_BIG, "len": LEN, "runs": RES}, f, indent=2)
print("\nwrote /content/stress272.json")""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

dst = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Stress272.ipynb"
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (dst, len(cells)))

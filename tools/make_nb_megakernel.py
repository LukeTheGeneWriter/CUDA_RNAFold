#!/usr/bin/env python3
"""Generate CUDA_RNAFold_Megakernel.ipynb.

The fused per-record sweep (RNA_MEGAKERNEL, megakernel.cu) has never run on a
datacentre card. This notebook asks four questions in order, and the first one
decides whether the rest are worth reading:

  A  does it give the same answer, and does every refusal fall back?
  B  what does the grid barrier cost?
  C  what geometry does it get -- registers, occupancy, resident blocks?
  D  what does it cost in wall clock against the per-phase path?

Stage 0 is EXPECTED to be slower. It buys the structure that stages 1-3 need
(the c window and the fML corner cache in shared memory across rows); the
numbers here decide whether that structure is affordable enough to build on.

Written in the house style: predictions before results, engagement checks that
fail loudly, and one sha compared across every arm.
"""
import json
import sys

cells = []


def _lines(s):
    out = s.split("\n")
    return [l + "\n" for l in out[:-1]] + ([out[-1]] if out[-1] else [])


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _lines(s.strip("\n"))})


# --------------------------------------------------------------------------
md(r"""# The per-record megakernel — does the structure hold, and what does it cost?

`megakernel.cu` runs the whole row chain as **one cooperative kernel per
record**: persistent blocks that own column ranges, the row loop inside the
kernel, a grid barrier between phases. Every phase calls the same `*_cell()` the
per-phase kernels call, out of shared headers, so this is a different
**schedule** for the same arithmetic.

**Why it exists.** Launch overhead is worth ~2 s of an 87 s fold, and that is
not the reason. `modular_decomposition` re-reads a whole fML column on every row
with no reuse inside the row — 46.8 TB at 400 × 5601 — and L2 cannot hold that
across rows because the other phases stream `c` through the same cache between
launches (measured: **40.8 % L2 hit at a shape whose entire live set fits in
40 MB of L2**). Only a resident kernel can keep a block's slice on chip across
rows, and only a fused chain makes T1's row reuse legal at all.

**Stage 0 is expected to be SLOWER.** It carries none of the caching yet. What
this run decides is whether the structure is affordable: the grid barrier, the
single block size every phase must share, and the register count set by the
worst phase.

| § | question | what would kill the design |
|---|---|---|
| **A** | same answer? does every refusal fall back? | any sha moves |
| **B** | grid barrier cost | > ~20 µs per barrier: fusion must stop at one row per launch |
| **C** | geometry | occupancy far below the per-phase kernels' |
| **D** | wall against the per-phase path | slower by more than the caching stages can plausibly return (~25 %) |
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""
import subprocess, os, sys, json, time, re, random, io, csv as _csvmod, collections

def sh(cmd, check=True, quiet=False):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        out = (p.stdout or "") + (p.stderr or "")
        if out.strip(): print(out.strip()[-4000:])
    if check and p.returncode != 0: raise SystemExit("failed: %s" % cmd)
    return p

print(sh("nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,compute_cap "
         "--format=csv,noheader", quiet=True).stdout.strip())
NPROC = int(sh("nproc", quiet=True).stdout.strip())
print("cores:", NPROC)
""")

code(r"""
# The build tools. This notebook shipped WITHOUT them and autogen.sh died on a
# clean Colab image -- gengetopt, libtoolize and help2man are not preinstalled.
# The check is separate from the install because apt succeeding says nothing
# about whether the binaries are on PATH.
sh("apt-get -qq update > /dev/null 2>&1", check=False, quiet=True)
sh("apt-get -qq install -y gengetopt help2man xxd libtool texinfo doxygen time "
   "> /dev/null 2>&1", check=False, quiet=True)
MISSING = [t for t in ("gengetopt", "help2man", "xxd", "libtoolize", "makeinfo", "doxygen")
           if sh("command -v %s" % t, check=False, quiet=True).returncode != 0]
print("missing build tools:", MISSING or "none")
if MISSING:
    raise SystemExit("install failed for %s -- autogen or make WILL fail" % MISSING)
""")

md(r"""## 2. Build

The megakernel is the first thing in this tree to use a **cooperative launch**,
so the build is probed for it rather than assumed: if `cudaLaunchCooperativeKernel`
is missing from the object, nothing below means anything.""")

code(r"""
REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "Lukes_Flow_Batching"
ROOT   = "/content/mk"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/tree && cd %s/tree && git checkout -q %s" % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/tree && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit:", sh("cd %s/tree && git log --oneline -1" % ROOT, quiet=True).stdout.strip())

for probe, what in (("RNA_MEGAKERNEL", "the gate"),
                    ("cudaLaunchCooperativeKernel", "the cooperative launch"),
                    ("grid.sync", "the grid barrier"),
                    ("rnafold_megakernel_refuse", "the decline list")):
    hits = sh("grep -rl '%s' %s/tree/src/ViennaRNA/mfe/cuda/ | wc -l" % (probe, ROOT),
              quiet=True).stdout.strip()
    print("  %-32s %s in %s file(s)" % (probe, what, hits))
""")

code(r"""
t0 = time.time()

def step(name, cmd, log):
    # A build step that hides its own log is a step you cannot debug from a
    # notebook -- which is exactly how this cell failed the first time it ran.
    p = sh("%s > %s 2>&1" % (cmd, log), check=False, quiet=True)
    if p.returncode:
        print("---- %s FAILED, tail of %s ----" % (name, log))
        print(sh("tail -40 %s" % log, quiet=True).stdout)
        raise SystemExit("%s failed" % name)
    return p

sh("cd %s/tree && tar -xjf src/dlib-*.tar.bz2 -C src/ && tar -xzf src/libsvm-*.tar.gz -C src/"
   % ROOT, check=False, quiet=True)

step("autogen", "cd %s/tree && ./autogen.sh" % ROOT, "/content/autogen.log")

# Upstream 2.7.2 ships doc/man2rst.py mode 644 and the man-to-rst rule EXECS it,
# so make dies even under --without-doc. One chmod, every time.
sh("chmod +x %s/tree/doc/man2rst.py" % ROOT, check=False, quiet=True)

step("configure",
     "cd %s/tree && ./configure --enable-cuda --without-python --without-perl "
     "--without-swig --without-doc --without-rnaxplorer --without-forester "
     "--without-kinfold --without-rnalocmin CFLAGS='-g -O2' CXXFLAGS='-g -O2' "
     "PYTHON3=\"$(command -v python3)\"" % ROOT, "/content/conf.log")

# upstream races its own gengetopt headers under -j; a second make settles it
p = sh("cd %s/tree && make -j%d > /content/make.log 2>&1" % (ROOT, NPROC),
       check=False, quiet=True)
if p.returncode:
    p = sh("cd %s/tree && make -j%d > /content/make.log 2>&1" % (ROOT, NPROC),
           check=False, quiet=True)
if p.returncode:
    print("---- make FAILED, tail of /content/make.log ----")
    print(sh("tail -60 /content/make.log", quiet=True).stdout)
    raise SystemExit("make failed")

BIN = "%s/tree/src/bin/RNAfold" % ROOT
if not os.path.exists(BIN):
    print(sh("tail -40 /content/make.log", quiet=True).stdout)
    raise SystemExit("make reported success but %s does not exist" % BIN)
print("built in %.1f min -> %s" % ((time.time()-t0)/60.0, BIN))
sh("%s --version" % BIN)

# megakernel.cu must be IN the binary, not merely in the checkout. A tree that
# builds without it looks identical until every megakernel arm reports
# "declined" and the whole notebook quietly measures the per-phase path.
for needle, what in (("RNA_MEGAKERNEL=1: the row chain runs", "the gate banner"),
                     ("geometry: %d threads per block",       "the variant banner"),
                     ("SKEW:",                                "stage 3")):
    n = sh("strings %s | grep -cF '%s' || true" % (BIN, needle), quiet=True).stdout.strip()
    print("  %-22s %s" % (what, "present" if n and n != "0" else "*** ABSENT ***"))
""")

md("## 3. The runner")

code(r"""
OUT     = "/content/megakernel.json"
RESULTS = {}

WALL_RE  = re.compile(r"megakernel wall=([0-9.]+) s for (\d+) records")
SHARE_RE = re.compile(r"phase share of block 0's cycles: (.+)")
GEOM_RE  = re.compile(r"(\d+) SMs x (\d+) resident blocks of (\d+) threads")
DECL_RE  = re.compile(r"RNA_MEGAKERNEL declined: ([^\n]+)")
ONCHIP_RE = re.compile(r"on-chip: (\d+) columns per block, c-ring (\w+), "
                       r"fML corner K=(\d+), (\d+) KB shared")
GAUTO_RE  = re.compile(r"G=AUTO: (\d+) records in flight, (\d+) blocks each")
GEOM2_RE  = re.compile(r"geometry: (\d+) threads per block, md tile (\d+) lanes")
SKEWB_RE  = re.compile(r"SKEW: (\d+) blocks on the c chain")
PHASE_RE = re.compile(r"phase timing \(s\): int_loop=([0-9.]+) hp_mb=([0-9.]+) "
                      r"load_my_c=([0-9.]+) modular_decomp=([0-9.]+) fetch_mx=([0-9.]+)")

def fasta(name, n, L, seed=None):
    path = "%s/%s.fa" % (ROOT, name)
    if not os.path.exists(path):
        random.seed(seed if seed is not None else (n*100003 + L))
        with open(path, "w") as f:
            for i in range(n):
                f.write(">%s_%d\n%s\n" % (name, i, "".join(random.choice("ACGU") for _ in range(L))))
    return path

def run(tag, fa, mega=False, extra_env=None, args="", timeout=1800, quiet=False):
    env = dict(os.environ)
    for k in ("RNA_MEGAKERNEL","RNA_MK_RECORDS","RNA_FML_INT16","RNA_STREAM_OVERLAP",
              "RNA_CONTINUOUS_FLOW","RNA_SLOT_FLOW","RNA_MD_SMEM","RNA_GPU_SWEEP",
              "RNA_MK_THREADS","RNA_MK_TILE","RNA_MK_SKEW","RNA_MK_CWIN","RNA_MK_CORNER",
              "RNA_MK_CORNER_K","RNA_MK_SMEM_KB","RNA_MK_FORCE_OWNS","RNA_MK_OWN_MASK",
              "RNA_MK_TIMEOUT"):
        env.pop(k, None)
    env.update(RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1")
    if mega: env["RNA_MEGAKERNEL"] = "1"
    env.update(extra_env or {})

    t0 = time.time()
    p  = subprocess.run(("%s --noPS %s -i %s" % (BIN, args, fa)).split(),
                        capture_output=True, text=True, env=env, timeout=timeout)
    wall = time.time() - t0
    err  = p.stderr

    m  = WALL_RE.search(err);  mk_wall = float(m.group(1)) if m else None
    s  = SHARE_RE.search(err); shares  = s.group(1) if s else None
    g  = GEOM_RE.search(err)
    d  = DECL_RE.search(err)
    ph = PHASE_RE.search(err)

    oc = ONCHIP_RE.search(err)
    ga = GAUTO_RE.search(err)

    r = dict(wall=wall, rc=p.returncode, mk_wall=mk_wall, shares=shares,
             cols_per_block=int(oc.group(1)) if oc else None,
             ring=(oc.group(2) == "ON") if oc else None,
             corner_k=int(oc.group(3)) if oc else None,
             smem_kb=int(oc.group(4)) if oc else None,
             auto_g=int(ga.group(1)) if ga else None,
             auto_blocks=int(ga.group(2)) if ga else None,
             threads=int(GEOM2_RE.search(err).group(1)) if GEOM2_RE.search(err) else None,
             tile=int(GEOM2_RE.search(err).group(2)) if GEOM2_RE.search(err) else None,
             skew_blocks=int(SKEWB_RE.search(err).group(1)) if SKEWB_RE.search(err) else None,
             declined=d.group(1) if d else None,
             fused=(mk_wall is not None),
             sms=int(g.group(1)) if g else None,
             blocks_per_sm=int(g.group(2)) if g else None,
             block=int(g.group(3)) if g else None,
             phases=dict(zip(("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx"),
                             [float(x) for x in ph.groups()])) if ph else {},
             sha=__import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12],
             fa=os.path.basename(fa), mega=mega, args=args, env=dict(extra_env or {}))
    RESULTS[tag] = r
    with open(OUT, "w") as f: json.dump({"commit": COMMIT, "runs": RESULTS}, f, indent=1)
    if not quiet:
        print("  %-22s wall %8.2f  %-6s sha %s %s"
              % (tag, wall, "FUSED" if r["fused"] else "per-phase", r["sha"],
                 ("declined: " + r["declined"]) if r["declined"] else ""))
    return r
""")

# --------------------------------------------------------------------------
md(r"""## A. Does it give the same answer?

Nothing else in this notebook matters if this section is not clean. Each pair
folds the SAME input with the same binary, once fused and once per-phase.

**The refusals are tested as carefully as the successes.** `--noLP`, `-c`, `-g`,
int16 and continuous flow must each come back with a *declined* line AND the
per-phase answer — a refusal that silently ran the fused path would be the worst
outcome available, because it would look like a pass.

**This is not hypothetical.** The first fused run on a laptop returned every
energy as 0.00: the cell that joins the row switches into noLP semantics when it
is handed a stack buffer, and the fused path was handing it one because it was
allocated. `c[ij]` then received the stacked value (INF on the first row)
instead of the hairpin, with gate=1 and a correctly computed hairpin sitting
right there. A refusal list is only as good as what it actually passes.""")

code(r"""
print("A: fused vs per-phase, and every refusal")
A_CASES = [
    ("small",     fasta("a_4x600",  4,  600), "",       {}),
    ("medium",    fasta("a_8x1200", 8, 1200), "",       {}),
    ("mixed24",   fasta("a_24x2400", 24, 2400), "",     {}),
    ("noLP",      fasta("a_4x600",  4,  600), "--noLP", {}),
    ("circ",      fasta("a_4x600",  4,  600), "-c",     {}),
    ("gquad",     fasta("a_4x600",  4,  600), "-g",     {}),
    ("int16",     fasta("a_4x600",  4,  600), "",       {"RNA_FML_INT16": "1"}),
    ("flow",      fasta("a_4x600",  4,  600), "",       {"RNA_CONTINUOUS_FLOW": "1"}),
    ("dangles0",  fasta("a_4x600",  4,  600), "-d0",    {}),
]
for name, fa, args, env in A_CASES:
    run("A_%s_mk"  % name, fa, mega=True,  extra_env=env, args=args)
    run("A_%s_ref" % name, fa, mega=False, extra_env=env, args=args)
""")

code(r"""
print("  %-10s %-12s %-12s %-8s %s" % ("case", "fused sha", "per-phase sha", "fused?", "verdict"))
bad = 0
for name, _, _, _ in A_CASES:
    a, b = RESULTS.get("A_%s_mk" % name), RESULTS.get("A_%s_ref" % name)
    if not a or not b: continue
    same = (a["sha"] == b["sha"])
    if not same: bad += 1
    print("  %-10s %-12s %-12s %-8s %s"
          % (name, a["sha"], b["sha"], "yes" if a["fused"] else "declined",
             "same" if same else "*** DIFFERS ***"))

print()
expect_declined = ("noLP", "circ", "gquad", "int16", "flow")
for name in expect_declined:
    a = RESULTS.get("A_%s_mk" % name)
    if a and a["fused"]:
        print("  *** %s was NOT refused -- the fused path ran an option it does not implement" % name)
        bad += 1
    elif a and not a["declined"]:
        print("  *** %s neither ran fused nor announced a refusal" % name)
        bad += 1
for name in ("small", "medium", "mixed24", "dangles0"):
    a = RESULTS.get("A_%s_mk" % name)
    if a and not a["fused"]:
        print("  *** %s did NOT run fused (%s) -- section B measures nothing"
              % (name, a["declined"]))
        bad += 1
print()
print("  VERDICT:", "clean" if bad == 0 else "*** %d problem(s): stop here ***" % bad)
""")

# --------------------------------------------------------------------------
md(r"""## B. What does the grid barrier cost?

Eight `grid.sync()` per row is the price of fusing seven phases that feed each
other. At 5601 nt that is ~45 k barriers for one record.

The kernel keeps **device-side `clock64()` accumulators** per phase, because
`RNA_PHASE_SYNC` and ncu's per-kernel attribution both stop meaning anything
once the phases are one kernel. They are block 0's cycles, so they are a SHARE
of the row, never a wall time — read the `grid.sync` column and nothing else as
an absolute.

**Written before the run.** The barrier share should FALL as records get longer,
because the work between barriers grows while the barrier does not. If it stays
above ~25 % at 2400 nt, fusion is too expensive at this granularity and the
answer is one launch per row rather than per record.""")

code(r"""
print("B: the barrier's share of the row, against length")
for n, L in ((4, 600), (4, 1200), (4, 2400), (8, 4800)):
    r = run("B_%dx%d" % (n, L), fasta("b_%dx%d" % (n, L), n, L), mega=True)
    if r["shares"]: print("      ", r["shares"])
""")

code(r"""
print("  %-10s %10s %10s %10s %10s" % ("shape", "grid.sync", "md", "int_loop", "hp_mb"))
for n, L in ((4, 600), (4, 1200), (4, 2400), (8, 4800)):
    r = RESULTS.get("B_%dx%d" % (n, L))
    if not (r and r["shares"]): continue
    d = dict(kv.split("=") for kv in r["shares"].split())
    print("  %-10s %10s %10s %10s %10s"
          % ("%dx%d" % (n, L), d.get("grid.sync","-"), d.get("md","-"),
             d.get("int_loop","-"), d.get("hp_mb","-")))
print()
print("  A share that falls with length is the barrier being amortised by real work.")
print("  A flat or rising share means the barrier IS the kernel, and stage 0 should")
print("  become one launch per row instead of one per record.")
""")

# --------------------------------------------------------------------------
md(r"""## C. Geometry — what the fused kernel is allowed to have

One block size serves all seven phases (256 threads: `int_loop` and `md` want a
warp per cell, `fml_scan` wants a block-wide tile), and the register count is set
by the worst phase. A cooperative launch also needs every block RESIDENT, so the
grid is what occupancy allows, not what the work wants.

`int_loop_warp_kernel` and `modular_decomposition_kernel` are profiled beside it
as the controls: those are the numbers the fused kernel has to live near.""")

code(r"""
SECT = ("--metrics launch__registers_per_thread,launch__block_size,launch__grid_size,"
        "sm__warps_active.avg.pct_of_peak_sustained_active,"
        "launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,"
        "launch__occupancy_limit_blocks,launch__occupancy_limit_warps,"
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
        "sm__throughput.avg.pct_of_peak_sustained_elapsed")
NCU = {}

def profile(tag, kernel, fa, mega, skip=0, count=1):
    env = dict(os.environ)
    env.update(RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1")
    if mega: env["RNA_MEGAKERNEL"] = "1"
    out = "/content/ncu_%s.csv" % tag
    cmd = ("ncu --target-processes all --csv %s -k regex:'%s' --launch-skip %d "
           "--launch-count %d --log-file %s %s --noPS -i %s > /dev/null 2>&1"
           % (SECT, kernel, skip, count, out, BIN, fa))
    subprocess.run(cmd, shell=True, env=env)
    body = open(out).read() if os.path.exists(out) else ""
    if "Metric Name" not in body:
        print("  %-16s *** no metrics -- a broken probe, not a result" % tag); return {}
    rows = list(_csvmod.DictReader(io.StringIO(
        "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
    agg = collections.defaultdict(list)
    for r in rows:
        try: agg[r["Metric Name"]].append(float((r["Metric Value"] or "").replace(",", "")))
        except Exception: pass
    res = {k: sum(v)/len(v) for k, v in agg.items() if v}
    if res: NCU[tag] = res; print("  %-16s ok" % tag)
    return res

FA_C = fasta("b_4x1200", 4, 1200)
profile("C_mega",     "megakernel_record",             FA_C, mega=True)
profile("C_int_loop", "int_loop_warp_kernel",          FA_C, mega=False, skip=200)
profile("C_md",       "modular_decomposition_kernel",  FA_C, mega=False, skip=200)
""")

code(r"""
print("  %-12s %6s %6s %9s %9s %s" % ("kernel","regs","block","grid","occup%","limiter (blk/reg/smem/warp)"))
for tag, label in (("C_mega","megakernel"), ("C_int_loop","int_loop"), ("C_md","md")):
    k = NCU.get(tag)
    if not k: continue
    lim = [k.get("launch__occupancy_limit_blocks"), k.get("launch__occupancy_limit_registers"),
           k.get("launch__occupancy_limit_shared_mem"), k.get("launch__occupancy_limit_warps")]
    names = ["blocks","registers","shared","warps"]
    worst = names[lim.index(min(x for x in lim if x is not None))] if all(x is not None for x in lim) else "?"
    print("  %-12s %6.0f %6.0f %9.0f %8.1f%% %s  <- %s"
          % (label, k.get("launch__registers_per_thread",0), k.get("launch__block_size",0),
             k.get("launch__grid_size",0),
             k.get("sm__warps_active.avg.pct_of_peak_sustained_active",0),
             "/".join("%.0f" % x if x is not None else "-" for x in lim), worst))
print()
print("  The fused kernel's registers are the MAX over seven phases, so its occupancy")
print("  cannot beat the per-phase kernels'. The question is how far below it lands,")
print("  and whether shared memory (stages 1-2 want ~100 KB per block) has room left.")
""")

# --------------------------------------------------------------------------
md(r"""## D. Wall clock, and the honest verdict

Stage 0 carries none of the caching. It should be **slower**. This section says
by how much, which is what decides whether stages 1–3 are worth building: they
have to return that gap plus the prize.

**Written before the run.** The fused path gives up the per-phase block-size
tuning and runs at whatever occupancy the worst phase allows, so expect
**1.2–2× slower**. Beyond ~2.5× the structure is too expensive to carry and the
fallback is one launch per row (which keeps the row loop on the device but pays
a launch per phase again).

`RNA_MK_RECORDS` is the records-in-flight dial: it divides the grid so that G
records can be resident at once. At stage 0 there is nothing to share between
them, so G>1 should be neutral to slightly better (more parallelism, same work).
It matters from stage 2, when each record wants shared memory of its own.""")

code(r"""
print("D: fused against per-phase, same input, ABBA")
for n, L in ((8, 1200), (24, 2400), (24, 4800)):
    fa = fasta("d_%dx%d" % (n, L), n, L)
    for rep, mega in (("a", False), ("a", True), ("b", True), ("b", False)):
        run("D_%dx%d_%s_%s" % (n, L, "mk" if mega else "ref", rep), fa, mega=mega, quiet=True)
    a = [RESULTS["D_%dx%d_ref_%s" % (n, L, r)]["wall"] for r in ("a","b")]
    b = [RESULTS["D_%dx%d_mk_%s"  % (n, L, r)]["wall"] for r in ("a","b")]
    import statistics
    ra, rb = statistics.mean(a), statistics.mean(b)
    shas = {RESULTS["D_%dx%d_%s_%s" % (n, L, k, r)]["sha"] for k in ("ref","mk") for r in ("a","b")}
    print("  %-10s per-phase %7.2f s   fused %7.2f s   %+6.1f%%   %s"
          % ("%dx%d" % (n, L), ra, rb, 100.0*(rb-ra)/ra,
             "one sha" if len(shas) == 1 else "*** %d shas ***" % len(shas)))
""")

code(r"""
print("D: records in flight (RNA_MK_RECORDS) -- the dial stages 2+ turn")
fa = fasta("d_24x2400", 24, 2400)
for g in (1, 2, 4, 8):
    r = run("D_G%d" % g, fa, mega=True, extra_env={"RNA_MK_RECORDS": str(g)}, quiet=True)
    print("  G=%-2d wall %7.2f s  blocks/SM %s  sha %s"
          % (g, r["wall"], r["blocks_per_sm"], r["sha"]))
shas = {RESULTS["D_G%d" % g]["sha"] for g in (1,2,4,8) if "D_G%d" % g in RESULTS}
print("  shas:", shas, "" if len(shas) == 1 else "  *** G CHANGED AN ANSWER ***")
""")


# --------------------------------------------------------------------------
md(r"""## E. Repeats, because one shot lies

Every number below is a median of repeats, and the arms are interleaved rather
than run in blocks. This is not ceremony. Developing stage 1 locally, a single
cold-first comparison read **1.63 s per-phase vs 1.21 s fused** and was reported
as "fused wins by 26 %". Warm, interleaved, three repeats, the same box and the
same binaries gave **0.756 s vs 0.925 s** -- fused LOSES by 22 %. The first
per-phase run was paying start-up the fused runs did not, and the absolute times
were roughly double what the machine actually does.

A cold baseline has flattered this project before (`build` threading, 2.58x that
was really 5.12x). The fix is cheap, so it is not optional.""")

code(r"""
import statistics

def runm(tag, fa, reps=3, **kw):
    # Median of `reps` warm runs. Returns the run dict of the median.
    rs = []
    for k in range(reps):
        rs.append(run("%s_r%d" % (tag, k), fa, quiet=True, **kw))
    rs.sort(key=lambda r: r["wall"])
    m = rs[len(rs)//2]
    m = dict(m, wall_all=[round(r["wall"], 3) for r in rs],
                wall_spread=round((rs[-1]["wall"] - rs[0]["wall"]) / rs[0]["wall"], 3))
    RESULTS[tag] = m
    with open(OUT, "w") as f: json.dump({"commit": COMMIT, "runs": RESULTS}, f, indent=1)
    return m

def interleave(tags, fa, reps=3):
    # ABBA over the arms: arm order is reversed on alternate passes, so a
    # one-directional drift cannot land on one arm.
    acc = {t: [] for t, _ in tags}
    for k in range(reps):
        order = tags if (k % 2 == 0) else list(reversed(tags))
        for t, kw in order:
            acc[t].append(run("%s_p%d" % (t, k), fa, quiet=True, **kw)["wall"])
    return {t: statistics.median(v) for t, v in acc.items()}, acc

print("E: warming up")
_fa = fasta("e_warm", 8, 1200)
run("E_warm", _fa, quiet=True)
print("  ready")
""")

# --------------------------------------------------------------------------
md(r"""## F. int16 is the fused baseline (stage 1a)

int16 halves the fML bytes and is the better default at production (md −20.7 %,
wall −7.9 % at 400×5601). Stage 1 makes it the baseline for the fused kernel
rather than an arm, which also doubles what will fit on chip for stages 2–3.

Two questions here. **Does it still agree?** -- four arms, one sha. And **what
does it do to the barrier's share?** Locally it went the wrong way: 50 % → 68 %
of block 0's cycles, because int16 makes the work phases cheaper without
touching the barrier. If that holds on an A100 it is an argument for stage 3
(the skew), not against int16.""")

code(r"""
print("F: int16 on both paths")
fa = fasta("f_24x2400", 24, 2400)
arms = {
    "F_ref_i32": dict(mega=False, extra_env={}),
    "F_ref_i16": dict(mega=False, extra_env={"RNA_FML_INT16": "1"}),
    "F_mk_i32" : dict(mega=True,  extra_env={}),
    "F_mk_i16" : dict(mega=True,  extra_env={"RNA_FML_INT16": "1"}),
}
for t, kw in arms.items():
    r = runm(t, fa, **kw)
    print("  %-10s wall %7.2f s  sha %s  %s"
          % (t, r["wall"], r["sha"], r["shares"] or ""))
shas = {RESULTS[t]["sha"] for t in arms}
print()
print("  ONE SHA ACROSS FOUR ARMS:", len(shas) == 1, shas)
if len(shas) != 1:
    print("  *** int16 OR THE FUSED PATH CHANGED AN ANSWER -- stop here ***")
""")

code(r"""
def share_of(tag, phase="grid.sync"):
    sh = RESULTS.get(tag, {}).get("shares")
    if not sh: return None
    return float(dict(kv.split("=") for kv in sh.split())[phase].rstrip("%"))

print("  barrier share, int32 vs int16 (fused):")
for t in ("F_mk_i32", "F_mk_i16"):
    print("    %-10s grid.sync %5.1f%%  int_loop %5.1f%%  md %5.1f%%"
          % (t, share_of(t) or -1, share_of(t, "int_loop") or -1, share_of(t, "md") or -1))
""")

# --------------------------------------------------------------------------
md(r"""## G. G — the dial that decides whether any of this is worth it

**This is the most important section in the notebook.** One record's row is at
most `length` cells; the grid is thousands of threads. At G=1 each record gets
the whole device and the row's eight grid barriers are paid by a grid that is
mostly idle. Locally, G was worth **2.2×**:

| G | wall | grid.sync share |
|---|---|---|
| 1 | 2.068 s | 66.6 % |
| 4 | 1.197 s | |
| 8 | **0.925 s** | 27.4 % |
| 24 | 1.376 s | |

At the good G the top phases become `int_loop` (32 %) and `md` (32 %) -- the
actual work -- which is the precondition for stages 2–3 buying anything.

AUTO gives each record just enough blocks to cover one row of the longest record
and spends the rest of the device on more records. On an A100 (108 SMs, so many
more blocks) AUTO should pick a much larger G than it does locally. **The
question this section answers: does AUTO land on the sweep's optimum there
too?** If it does not, the rule is wrong and the number it should use is here.""")

code(r"""
print("G: records in flight, at two shapes")
for name, n, L in (("g_40x1200", 40, 1200), ("g_24x4800", 24, 4800)):
    fa = fasta(name, n, L)
    print("  %s:" % name)
    best, best_g = None, None
    for g in (1, 2, 4, 8, 16, 32, 64):
        r = runm("G_%s_%d" % (name, g), fa, reps=3, mega=True,
                 extra_env={"RNA_MK_RECORDS": str(g), "RNA_FML_INT16": "1"})
        gs = share_of("G_%s_%d" % (name, g))
        print("    G=%-3d wall %7.2f s  spread %4.1f%%  grid.sync %s"
              % (g, r["wall"], 100*r["wall_spread"], ("%.1f%%" % gs) if gs else "-"))
        if best is None or r["wall"] < best:
            best, best_g = r["wall"], g
    a = runm("G_%s_auto" % name, fa, reps=3, mega=True, extra_env={"RNA_FML_INT16": "1"})
    ref = runm("G_%s_ref" % name, fa, reps=3, mega=False, extra_env={"RNA_FML_INT16": "1"})
    print("    AUTO     wall %7.2f s   (sweep best was G=%d at %.2f s)" % (a["wall"], best_g, best))
    print("    per-phase wall %7.2f s" % ref["wall"])
    print("    AUTO vs sweep best : %+.1f%%" % (100.0*(a["wall"]-best)/best))
    print("    fused vs per-phase : %+.1f%%" % (100.0*(a["wall"]-ref["wall"])/ref["wall"]))
""")

# --------------------------------------------------------------------------
md(r"""## H. Where stage 1 actually stands

The honest bar. Fused-at-AUTO against the per-phase path, at the shapes that
matter, interleaved and repeated.

Stage 0 was a structure and was expected to lose. Stage 1a (int16) does not
change that by itself. What this section decides is **which phase to attack
next**, and the plan and the measurement currently disagree:

- the plan's stage 1b is the `c` window in shared memory, justified as
  "int_loop stops reading DRAM";
- but int_loop is **latency**-bound, not DRAM-bound (4–5 % of DRAM peak, L2 hit
  ~82 %, stalls are `wait` 23 % with memory only 12–16 %). Shared memory buys
  latency there, not traffic.
- `md` is the bandwidth-bound one (82.8 % of DRAM peak at production), and it is
  the phase the whole fusion exists to serve.

So if this section shows `md` at or above `int_loop` in the fused mix, stage 2
(the fML corner cache) should come before stage 1b, and the plan's order should
change. The corner arithmetic says a corner of 9 % of the triangle captures
21.6 % of md's traffic, and at the measured elasticity (41 % of md's time is
bytes) that is worth about −3.9 s of wall at production.""")

code(r"""
print("H: fused (AUTO) vs per-phase, interleaved")
for name, n, L in (("h_40x1200", 40, 1200), ("h_24x4800", 24, 4800), ("h_8x9600", 8, 9600)):
    fa = fasta(name, n, L)
    med, raw = interleave([("H_%s_ref" % name, dict(mega=False, extra_env={"RNA_FML_INT16":"1"})),
                           ("H_%s_mk"  % name, dict(mega=True,  extra_env={"RNA_FML_INT16":"1"}))],
                          fa, reps=3)
    a, b = med["H_%s_ref" % name], med["H_%s_mk" % name]
    print("  %-12s per-phase %7.2f s   fused %7.2f s   %+.1f%%"
          % (name, a, b, 100.0*(b-a)/a))
    sh = RESULTS.get("H_%s_mk_p2" % name, {}).get("shares")
    if sh: print("               %s" % sh)
""")

code(r"""
print()
print("PHASE MIX AT THE BEST G -- what to attack next")
for t in [k for k in RESULTS if k.startswith("H_") and "_mk" in k]:
    sh = RESULTS[t].get("shares")
    if not sh: continue
    d = {kv.split("=")[0]: float(kv.split("=")[1].rstrip("%")) for kv in sh.split()}
    top = sorted(((v, k) for k, v in d.items() if k != "grid.sync"), reverse=True)[:3]
    print("  %-22s barrier %5.1f%%   top work: %s"
          % (t, d.get("grid.sync", -1), ", ".join("%s %.1f%%" % (k, v) for v, k in top)))
    break
print()
print("  IF md >= int_loop HERE, do stage 2 (fML corner cache) BEFORE stage 1b")
print("  (the c window) -- int_loop is latency-bound, so a shared-memory c window")
print("  buys it latency, not the traffic the plan credited it with.")
""")



# --------------------------------------------------------------------------
md(r"""## I. Stages 1b and 2 — the two on-chip caches

Both stages need the same thing from the schedule: **a block must own a fixed
range of absolute columns for the whole sweep**, or nothing it caches survives
to the next row. Stage 0 strided cells instead, which balances every row
perfectly, so this is a genuine trade and I.3 measures what it costs.

**Stage 1b, the `c` ring.** MAXLOOP bounds an interior loop to 30 unpaired
bases, so cell (i,j) reads c(p,q) only for p in [i+1,i+31] and q in [j-31,j-1]:
31 rows of the block's column span hold every `c` value it can ask for. Rows
advance by one per sweep row, so steady state fetches **one** row per row and
the other thirty are already on chip — the part only a resident kernel can do.

**Stage 2, the fML corner.** md walks column j from the diagonal downwards and
the range only grows at the bottom as i falls, so entries nearest the diagonal
are re-read every row while deep ones are read once. Element (r,j) is read about
r times, so the cache holds the top K entries of each owned column and md's
y-loop hits it on its last K iterations. A corner of side K covers (K/L)² of
the triangle and captures 3b − 2b^1.5 of the traffic: 9 % of the bytes for
21.6 %, 25 % for 50 %.

Knobs: `RNA_MK_CWIN`, `RNA_MK_CORNER`, `RNA_MK_CORNER_K`, `RNA_MK_SMEM_KB`.

**The geometry can decline itself.** W = span/blocks grows when a record gets
*fewer* blocks, which is exactly what AUTO G does to maximise records in flight
— so the two dials pull against each other, and the host degrades (corner K
halves, then the corner, then the ring, then column ownership) rather than
refusing the fold. Every cell below prints what actually ran, because a
silently-declined cache and a cache that does not pay look identical.""")

code(r"""
print("I.1 correctness -- six arms, one sha")
fa = fasta("i_24x2400", 24, 2400)
ARMS = {
  "I_ref"        : dict(mega=False, extra_env={"RNA_FML_INT16":"1"}),
  "I_none"       : dict(mega=True,  extra_env={"RNA_FML_INT16":"1","RNA_MK_CWIN":"0","RNA_MK_CORNER":"0"}),
  "I_ring"       : dict(mega=True,  extra_env={"RNA_FML_INT16":"1","RNA_MK_CWIN":"1","RNA_MK_CORNER":"0","RNA_MK_SMEM_KB":"96"}),
  "I_corner"     : dict(mega=True,  extra_env={"RNA_FML_INT16":"1","RNA_MK_CWIN":"0","RNA_MK_CORNER":"1","RNA_MK_SMEM_KB":"96"}),
  "I_both"       : dict(mega=True,  extra_env={"RNA_FML_INT16":"1","RNA_MK_SMEM_KB":"96"}),
  "I_both_i32"   : dict(mega=True,  extra_env={"RNA_MK_SMEM_KB":"96"}),
}
for t, kw in ARMS.items():
    r = run(t, fa, quiet=True, **kw)
    print("  %-12s sha %s  cols/block %-5s ring %-5s K %-4s %s KB"
          % (t, r["sha"], r["cols_per_block"], r["ring"], r["corner_k"], r["smem_kb"]))
shas = {RESULTS[t]["sha"] for t in ARMS}
print()
print("  ONE SHA ACROSS SIX ARMS:", len(shas) == 1)
if len(shas) != 1:
    print("  *** A CACHE CHANGED AN ANSWER -- stop, nothing below means anything ***")
""")

code(r"""
print()
print("I.2 did the caches ENGAGE? (byte-identity cannot tell you)")
for t in ("I_ring", "I_corner", "I_both"):
    r = RESULTS.get(t, {})
    want_ring   = t in ("I_ring", "I_both")
    want_corner = t in ("I_corner", "I_both")
    ok = ((r.get("ring") is True) == want_ring) and ((r.get("corner_k") or 0) > 0) == want_corner
    print("  %-10s ring=%-5s K=%-4s  %s"
          % (t, r.get("ring"), r.get("corner_k"),
             "as asked" if ok else "*** DEGRADED -- the host could not fit it ***"))
print()
print("  A cache that is present but never HIT also passes every check above.")
print("  The build bar for that is in the repo: poisoning each reader to return")
print("  INF must change the answer, and change it differently for corner-only")
print("  than for both. Re-run that locally if these numbers look like nulls.")
""")

code(r"""
print()
print("I.3 what the caches cost and buy")
for name, n, L in (("i_24x2400", 24, 2400), ("i_16x4800", 16, 4800)):
    fa = fasta(name, n, L)
    print("  %s:" % name)
    base = None
    for tag, env in (
        ("stage0 (cells strided)", {"RNA_MK_CWIN":"0","RNA_MK_CORNER":"0"}),
        ("columns, no cache",      {"RNA_MK_CWIN":"0","RNA_MK_CORNER":"1","RNA_MK_CORNER_K":"8","RNA_MK_SMEM_KB":"96"}),
        ("ring only",              {"RNA_MK_CWIN":"1","RNA_MK_CORNER":"0","RNA_MK_SMEM_KB":"96"}),
        ("corner K=32",            {"RNA_MK_CWIN":"0","RNA_MK_CORNER":"1","RNA_MK_CORNER_K":"32","RNA_MK_SMEM_KB":"96"}),
        ("corner K=64",            {"RNA_MK_CWIN":"0","RNA_MK_CORNER":"1","RNA_MK_CORNER_K":"64","RNA_MK_SMEM_KB":"160"}),
        ("both",                   {"RNA_MK_SMEM_KB":"160"}),
    ):
        e = dict(env); e["RNA_FML_INT16"] = "1"
        key = "I3_%s_%s" % (name, tag.replace(" ", "_"))
        r = runm(key, fa, reps=3, mega=True, extra_env=e)
        if base is None: base = r["wall"]
        print("    %-22s %7.2f s  %+6.1f%%  K=%-4s %s KB  md %5.1f%%  int_loop %5.1f%%"
              % (tag, r["wall"], 100.0*(r["wall"]-base)/base, r["corner_k"], r["smem_kb"],
                 share_of(key, "md") or -1, share_of(key, "int_loop") or -1))
    ref = runm("I3_%s_ref" % name, fa, reps=3, mega=False, extra_env={"RNA_FML_INT16":"1"})
    print("    %-22s %7.2f s  (per-phase control)" % ("per-phase", ref["wall"]))
""")

md(r"""**Reading I.3.** The second row isolates the cost of the schedule change
on its own — column ownership with a cache too small to matter — against
stage 0's cell striding. That difference is the load imbalance: a block owning
low columns is idle until the sweep reaches them, and an equal-width split is
not work-balanced (column j carries j−turn−1 cells over the sweep, so balanced
boundaries are at j = L·√(b/B)). If that row is badly negative, the caches are
paying a toll before they buy anything and the split is the first thing to fix.

If `md`'s share falls as K rises, stage 2 is working. If `int_loop`'s share
falls with the ring, stage 1b is — though note int_loop is latency-bound, so
the honest expectation there is small.""")



# --------------------------------------------------------------------------
md(r"""## J. Stage 3 — two co-resident halves, one row apart

The c chain for row i needs only `c` rows >= i+1 and the sequence, so it can run
**while** the fML chain finishes row i+1. The halves meet exactly twice a row:
`new_c(i)` reads dml1, the snapshot of md(i+1), and `fml_scan(i+1)` reads new_e
and e3p00, which the c chain wrote a row earlier. Everything else is internal
to one half.

That is what makes the barrier cheap. The undivided schedule pays **eight full
grid barriers** a row; the skew pays **two**, plus a barrier over one half only
(five for the fML chain, one for the c chain). `RNA_MK_SKEW` is the percentage
of blocks given to the c chain; 0 restores the undivided schedule.

Each half also holds only ONE cache — the c chain needs the ring, the fML chain
the corner — so the shared budget covers the larger rather than the sum.

**What this section is really asking.** Stage 3 does not reduce total work, it
overlaps it. It wins if barrier time falls by more than the halves lose to
running on a fraction of the grid, and that balance is a property of the device
and the record length, not of the code. On a 3050 there are too few blocks for
the split to mean much (AUTO G leaves 4 blocks a record, so a 50 % split is 2
and 2). An A100 is the first machine where the question is real.""")

code(r"""
print("J.1 correctness -- the skew must not move a single answer")
fa = fasta("j_24x2400", 24, 2400)
J = {"J_noskew": {}}
for pct in (25, 50, 75):
    J["J_skew%d" % pct] = {"RNA_MK_SKEW": str(pct)}
for tag, env in J.items():
    e = dict(env); e["RNA_FML_INT16"] = "1"; e["RNA_MK_SMEM_KB"] = "96"
    r = run(tag, fa, mega=True, extra_env=e, quiet=True)
    print("  %-12s sha %s  cols/block %-5s K %-4s" % (tag, r["sha"], r["cols_per_block"], r["corner_k"]))
ref = run("J_ref", fa, mega=False, extra_env={"RNA_FML_INT16": "1"}, quiet=True)
shas = {RESULTS[t]["sha"] for t in J} | {ref["sha"]}
print()
print("  ONE SHA ACROSS THE SPLITS AND THE CONTROL:", len(shas) == 1)
if len(shas) != 1:
    print("  *** THE SKEW CHANGED AN ANSWER ***")
    print("  The first time this fired it was the half-barrier missing its memory")
    print("  fences: __syncthreads() orders memory WITHIN a block, so a block's")
    print("  writes could still be in its L1 when the other half-blocks were let")
    print("  go. It showed up as a fold that changed with the BLOCK COUNT, which")
    print("  is a good tell -- check the release/acquire fences in mk_half_sync().")
""")

code(r"""
print()
print("J.2 what the skew does to the barrier, and to the wall")
for name, n, L in (("j_24x2400", 24, 2400), ("j_16x4800", 16, 4800)):
    fa = fasta(name, n, L)
    print("  %s:" % name)
    base = None
    for pct in (0, 25, 50, 75):
        e = {"RNA_FML_INT16": "1", "RNA_MK_SMEM_KB": "96"}
        if pct: e["RNA_MK_SKEW"] = str(pct)
        key = "J2_%s_%d" % (name, pct)
        r = runm(key, fa, reps=3, mega=True, extra_env=e)
        if base is None: base = r["wall"]
        print("    skew %-3s wall %7.2f s  %+6.1f%%   barrier %5.1f%%  int_loop %5.1f%%  md %5.1f%%"
              % (("off" if not pct else "%d%%" % pct), r["wall"], 100.0*(r["wall"]-base)/base,
                 share_of(key) or -1, share_of(key, "int_loop") or -1, share_of(key, "md") or -1))
    rf = runm("J2_%s_ref" % name, fa, reps=3, mega=False, extra_env={"RNA_FML_INT16": "1"})
    print("    per-phase  wall %7.2f s" % rf["wall"])
""")

md(r"""**Reading J.2.** The barrier share is the number to watch: it should fall
sharply from the skew-off row, because eight grid barriers become two. If it
does *not* fall, the half-barrier's atomic spin is costing what the grid barrier
did and the design has not bought anything.

If the barrier share falls but the wall does not improve, the halves are losing
more to running on a fraction of the grid than the barrier saved — which is a
split-tuning problem, so the best `RNA_MK_SKEW` in this sweep is the answer, not
a verdict on stage 3.

One caveat that applies to every row: an equal-width column split is not
work-balanced, so a block owning low columns idles until the sweep reaches it.
That cost is shared with stages 1b and 2 and is measured on its own in I.3.""")



# --------------------------------------------------------------------------
md(r"""## K. Tuning — blocks, threads, tiles

Six dials, and they are not independent, so this is a **coordinate descent**
rather than a cross product: threads and tile first, then blocks per record,
then the on-chip tiling, then the split. A full grid would be several hundred
folds of A100 time to answer a question the first two stages usually settle.

| dial | knob | what it changes |
|---|---|---|
| threads per block | `RNA_MK_THREADS` | 128 / 256 / 512 — a compiled variant |
| md tile | `RNA_MK_TILE` | 16 / 32 lanes per md cell — a compiled variant |
| blocks per record | `RNA_MK_RECORDS` (G) | grid ÷ G, and therefore **columns per block** |
| corner depth | `RNA_MK_CORNER_K` | fML entries cached per column |
| shared budget | `RNA_MK_SMEM_KB` | caps the ring and the corner together |
| skew split | `RNA_MK_SKEW` | % of blocks on the c chain |

**Two couplings worth knowing before reading the numbers.** Threads per block
sets residency through `__launch_bounds__` (the target is ~768 threads/SM, so
128→6 blocks/SM, 256→3, 512→2) *and* sets how many blocks a record gets, which
sets the column width W, which decides whether the caches fit the budget at all.
And G trades records-in-flight against blocks-per-record: fewer blocks per
record means a wider W and more shared memory per block, so pushing G up can
silently switch the caches off.

Every cell prints the geometry that actually ran and checks the sha, because a
knob that did not engage and a knob that did nothing look identical.""")

code(r"""
TUNE_FA   = fasta("k_24x2400", 24, 2400)
TUNE_BIG  = fasta("k_16x4800", 16, 4800)
TUNE_REF  = runm("K_ref", TUNE_FA, reps=3, mega=False, extra_env={"RNA_FML_INT16": "1"})
REFSHA    = TUNE_REF["sha"]
BEST      = {"RNA_FML_INT16": "1", "RNA_MK_SMEM_KB": "96"}
print("K: control %.2f s  sha %s" % (TUNE_REF["wall"], REFSHA))

def share_of_r(r, phase="grid.sync"):
    sh = r.get("shares")
    if not sh: return -1.0
    try:    return float(dict(kv.split("=") for kv in sh.split())[phase].rstrip("%"))
    except Exception: return -1.0

def tune(tag, fa, env, reps=3):
    e = dict(BEST); e.update(env)
    r = runm(tag, fa, reps=reps, mega=True, extra_env=e)
    r["ok"] = (r["sha"] == REFSHA)
    return r

def show(label, r):
    print("    %-26s %7.2f s  %s  thr %-4s tile %-3s cols %-5s K %-4s %-4s KB  barrier %5.1f%%"
          % (label, r["wall"], "ok " if r["ok"] else "SHA!", r["threads"], r["tile"],
             r["cols_per_block"], r["corner_k"], r["smem_kb"], share_of_r(r)))
""")

code(r"""
print("K.1 threads per block x md tile")
best, bestkey = None, None
for thr in (128, 256, 512):
    for tile in (16, 32):
        r = tune("K1_%d_%d" % (thr, tile), TUNE_FA,
                 {"RNA_MK_THREADS": str(thr), "RNA_MK_TILE": str(tile)})
        show("thr=%d tile=%d" % (thr, tile), r)
        if r["ok"] and (best is None or r["wall"] < best):
            best, bestkey = r["wall"], (thr, tile)
if bestkey:
    BEST["RNA_MK_THREADS"], BEST["RNA_MK_TILE"] = str(bestkey[0]), str(bestkey[1])
    print("  -> threads=%d tile=%d (%.2f s)" % (bestkey[0], bestkey[1], best))
""")

code(r"""
print("K.2 blocks per record (G) at the chosen geometry")
best, bestg = None, None
for g in (1, 2, 4, 8, 16, 32, 64, 0):        # 0 = AUTO
    env = {} if g == 0 else {"RNA_MK_RECORDS": str(g)}
    r = tune("K2_%d" % g, TUNE_FA, env)
    show("G=%s" % ("AUTO" if g == 0 else g), r)
    if r["ok"] and (best is None or r["wall"] < best):
        best, bestg = r["wall"], g
if bestg:
    BEST["RNA_MK_RECORDS"] = str(bestg)
    print("  -> G=%d (%.2f s)" % (bestg, best))
else:
    print("  -> AUTO (it won, or nothing beat it)")
""")

code(r"""
print("K.3 on-chip tiling: corner depth and the shared budget")
best, bestkv = None, None
for kb in (32, 64, 96, 160):
    for ck in (0, 8, 16, 32, 64):
        env = {"RNA_MK_SMEM_KB": str(kb)}
        env["RNA_MK_CORNER"] = "0" if ck == 0 else "1"
        if ck: env["RNA_MK_CORNER_K"] = str(ck)
        r = tune("K3_%d_%d" % (kb, ck), TUNE_FA, env, reps=2)
        show("smem=%dKB K=%d" % (kb, ck), r)
        if r["ok"] and (best is None or r["wall"] < best):
            best, bestkv = r["wall"], (kb, ck)
if bestkv:
    BEST["RNA_MK_SMEM_KB"] = str(bestkv[0])
    BEST["RNA_MK_CORNER"]  = "0" if bestkv[1] == 0 else "1"
    if bestkv[1]: BEST["RNA_MK_CORNER_K"] = str(bestkv[1])
    print("  -> smem=%d KB, corner K=%d (%.2f s)" % (bestkv[0], bestkv[1], best))
print()
print("  NOTE: a row whose 'cols' or 'K' differs from what was asked was DEGRADED")
print("  by the host -- the budget did not fit it. That is a real result (the")
print("  geometry does not fit), not a failed run.")
""")

code(r"""
print("K.4 the c ring, and the skew split")
for cw in (0, 1):
    r = tune("K4_ring%d" % cw, TUNE_FA, {"RNA_MK_CWIN": str(cw)})
    show("c ring %s" % ("ON" if cw else "off"), r)
best, bests = None, None
for pct in (0, 25, 40, 50, 60, 75):
    env = {} if pct == 0 else {"RNA_MK_SKEW": str(pct)}
    r = tune("K4_skew%d" % pct, TUNE_FA, env)
    show("skew %s" % ("off" if not pct else "%d%%" % pct), r)
    if r["ok"] and (best is None or r["wall"] < best):
        best, bests = r["wall"], pct
if bests:
    BEST["RNA_MK_SKEW"] = str(bests)
    print("  -> skew %d%% (%.2f s)" % (bests, best))
else:
    print("  -> skew off")
""")

code(r"""
print("K.5 the tuned configuration, confirmed at a second shape")
print("  BEST =", " ".join("%s=%s" % kv for kv in sorted(BEST.items())))
for name, fa in (("24x2400", TUNE_FA), ("16x4800", TUNE_BIG)):
    ref = runm("K5_%s_ref" % name, fa, reps=3, mega=False, extra_env={"RNA_FML_INT16": "1"})
    r   = runm("K5_%s_mk" % name, fa, reps=3, mega=True, extra_env=dict(BEST))
    ok  = (r["sha"] == ref["sha"])
    print("  %-9s per-phase %7.2f s   tuned %7.2f s   %+6.1f%%   %s"
          % (name, ref["wall"], r["wall"], 100.0*(r["wall"]-ref["wall"])/ref["wall"],
             "sha ok" if ok else "*** SHA DIFFERS -- the tuned config is WRONG ***"))
    print("            %s" % (r["shares"] or ""))
print()
print("  Paste this to reproduce the tuned run:")
print("   ", " ".join("%s=%s" % kv for kv in sorted(BEST.items())), "RNA_MEGAKERNEL=1")
""")

md(r"""**Reading K.** The coordinate descent takes each dial's winner forward, so
a later stage can only be judged against the earlier choices — if K.2 picks a G
that switches the caches off, K.3's budget sweep is exploring a different design
point than K.1 was. The geometry columns are printed on every row for exactly
that reason; when a winner's `cols`/`K` differ from the row above it, the two
are not comparable and the sweep should be re-run with the dials pinned.

If K.5 is still negative, the useful output is not the verdict but the phase
shares beside it: barrier-dominated says the schedule is wrong (G, split),
`md`-dominated says stage 2 wants a deeper corner, `int_loop`-dominated says the
`c` ring is not paying and stage 1b should be reconsidered.""")


code(r"""
print("=" * 72)
print("SUMMARY")
print("=" * 72)
ok = all(RESULTS.get("A_%s_mk" % n, {}).get("sha") == RESULTS.get("A_%s_ref" % n, {}).get("sha")
         for n, _, _, _ in A_CASES if "A_%s_mk" % n in RESULTS)
print("  A  correctness          :", "clean" if ok else "*** FAILED -- nothing else counts ***")
b = RESULTS.get("B_8x4800", {}).get("shares")
print("  B  barrier share @4800  :", (dict(kv.split("=") for kv in b.split()).get("grid.sync") if b else "-"))
k = NCU.get("C_mega", {})
print("  C  fused regs/occupancy : %.0f regs, %.1f%%"
      % (k.get("launch__registers_per_thread", 0),
         k.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)))
import statistics
try:
    ra = statistics.mean([RESULTS["D_24x4800_ref_%s" % r]["wall"] for r in ("a","b")])
    rb = statistics.mean([RESULTS["D_24x4800_mk_%s"  % r]["wall"] for r in ("a","b")])
    print("  D  fused vs per-phase   : %+.1f%% at 24x4800" % (100.0*(rb-ra)/ra))
except Exception:
    print("  D  fused vs per-phase   : -")
print()
try:
    print("  F  int16 four-arm sha    :",
          "clean" if len({RESULTS[t]["sha"] for t in
                          ("F_ref_i32","F_ref_i16","F_mk_i32","F_mk_i16")}) == 1
          else "*** DISAGREES ***")
except Exception:
    print("  F  int16 four-arm sha    : -")
try:
    a = RESULTS["G_g_24x4800_auto"]["wall"]; r = RESULTS["G_g_24x4800_ref"]["wall"]
    print("  G  fused AUTO vs per-ph  : %+.1f%% at 24x4800" % (100.0*(a-r)/r))
except Exception:
    print("  G  fused AUTO vs per-ph  : -")
print()
print("  READ IT AS: stage 0 is a structure, not a speedup. It is worth continuing")
print("  if A is clean, B falls with length, and D is inside ~2.5x -- because stages")
print("  1-3 (the c window and the fML corner cache in shared memory, held ACROSS")
print("  rows) are what the structure exists to make possible, and they have to")
print("  return this gap plus the prize.")

from google.colab import files
files.download(OUT)
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Megakernel.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (out, len(cells)))

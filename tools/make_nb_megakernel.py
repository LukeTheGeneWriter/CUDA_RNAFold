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
sh("cd %s/tree && tar -xjf src/dlib-*.tar.bz2 -C src/ && tar -xzf src/libsvm-*.tar.gz -C src/" % ROOT)
sh("cd %s/tree && ./autogen.sh > /tmp/autogen.log 2>&1" % ROOT)
sh("cd %s/tree && ./configure --enable-cuda --without-python --without-perl --without-swig "
   "--without-doc --without-rnaxplorer --without-forester --without-kinfold "
   "--without-rnalocmin > /tmp/configure.log 2>&1" % ROOT)
# upstream races its own gengetopt headers under -j; a second make settles it
sh("cd %s/tree && (make -j%d > /tmp/make.log 2>&1 || make -j%d > /tmp/make.log 2>&1)"
   % (ROOT, NPROC, NPROC))
BIN = "%s/tree/src/bin/RNAfold" % ROOT
print("built in %.1f min -> %s" % ((time.time()-t0)/60.0, BIN))
sh("%s --version" % BIN)
""")

md("## 3. The runner")

code(r"""
OUT     = "/content/megakernel.json"
RESULTS = {}

WALL_RE  = re.compile(r"megakernel wall=([0-9.]+) s for (\d+) records")
SHARE_RE = re.compile(r"phase share of block 0's cycles: (.+)")
GEOM_RE  = re.compile(r"(\d+) SMs x (\d+) resident blocks of (\d+) threads")
DECL_RE  = re.compile(r"RNA_MEGAKERNEL declined: ([^\n]+)")
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
              "RNA_CONTINUOUS_FLOW","RNA_SLOT_FLOW","RNA_MD_SMEM","RNA_GPU_SWEEP"):
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

    r = dict(wall=wall, rc=p.returncode, mk_wall=mk_wall, shares=shares,
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

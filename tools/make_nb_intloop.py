#!/usr/bin/env python3
"""Build the int_loop notebook: chunk width vs datatype, and block size re-opened.

WHY THIS EXISTS. The phase-synced T4 run (cd2b000b) showed int_loop is 30% of
GPU time -- a phase every profile had recorded as 0.8%, because its host timer
only ever measured a kernel LAUNCH and the work drained into hp_mb's blocking
upload. It is now the second-largest GPU phase and nothing has ever been
optimised in it with a measurement behind it.

THE ONE NUMBER THAT PROMPTED THIS. Both synced arms do IDENTICAL work --
6266401200 cells, 400 records -- and differ only in packing:

    records/chunk   i32 28.6   i16 36.4   ratio 1.273
    int_loop/cell   17.12 ns   21.98 ns   ratio 1.284

The per-cell slowdown matches the records-per-chunk ratio to 0.9%. d_my_c is
int32 unconditionally and int_loop touches no int16 data at all, so the
candidate story is LOCALITY, not the encoding: int16 frees VRAM, the chunker
fits 1.27x more records, and each brings its own full-width my_c triangle into
the working set one sweep row strides over.

That is in direct tension with what int_loop.cu:1168 already concludes -- "real
occupancy gains for this kernel come from more BLOCKS in flight (bigger
batches)". More records per chunk IS more blocks in flight. Both can be true
(occupancy up, locality down) and nobody has measured which.

RUNTIME. ~9 min per 400x5601 arm. Section A is 4 arms (~36 min), B is 6
(~54 min), C is 2, D and E are minutes. The full matrix is ~2 hours after a
~10 min build, so EVERY ARM IS SAVED AS IT COMPLETES and the sections are
ordered by value -- a disconnect after Section A still answers the main
question.

usage: python3 tools/make_nb_intloop.py [out.ipynb]
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
md(r"""# `int_loop_kernel`: chunk width, datatype, and block size

`int_loop` is **30 % of GPU time** at 400 × 5601 and has been recorded as
**0.8 %** for the life of this project — its host timer only ever measured a
kernel *launch*, and the work drained into `hp_mb`'s blocking upload. See
`STRESS272_RESULTS.md` §19 and `PROFILE_INT_LOOP.md`.

## The number this notebook exists to explain

Both phase-synced arms do **identical work** — 6 266 401 200 cells, 400 records
— and differ only in how it is packed:

| | i32 | i16 | ratio |
|---|---|---|---|
| records per chunk | 28.6 | 36.4 | **1.273** |
| `int_loop` per cell | 17.12 ns | 21.98 ns | **1.284** |

**The per-cell slowdown matches the records-per-chunk ratio to within 0.9 %.**

`d_my_c` is `int*` unconditionally and `int_loop.cu` contains no int16 code at
all, so the candidate story is **locality, not the encoding**: int16 halves the
*fML* triangle, which frees VRAM, which lets the chunker fit 1.27× more records
— each bringing its own full-width int32 `my_c` triangle into the working set a
single sweep row strides over.

That sits in tension with what `int_loop.cu:1168` already concludes from the
2026-08-20 NCU sweep: *"real occupancy gains for this kernel come from more
**blocks** in flight (bigger batches)"*. More records per chunk **is** more
blocks in flight, and the per-cell cost rose 28 % when it happened. Both can be
true — occupancy up, locality down — and which wins has never been measured,
because **this kernel has never been profiled at all** (`ncu -k int_loop_kernel`
matched nothing; the symbols are `int_loop_kernel_32/_64/_128/_256`).

## Rules that make the numbers mean anything

* **Everything is `RNA_PHASE_SYNC=1`.** Without it `int_loop`'s timer measures a
  launch, not the work. Compare **synced against synced, never against async.**
* **Never quote a synced WALL.** Compare `int_loop`'s synced phase time.
  (Measured at 400 × 5601: phase-sync costs ~0 %, because the blocking H2Ds
  already serialised every row — but §C re-checks that at a second chunk width
  rather than assuming it.)
* **`sha` must be `7c0b3d633281` in every arm.** A block size or chunk width
  that changes the answer is a bug, and a far bigger finding than any timing.
* **`sweep shape:` must be present**, or the run folded on the CPU and the
  profile describes something else.

## Order, and why

**A** is the decisive experiment and runs first — a disconnect after it still
answers the main question. **B** re-opens block size only on the axes that
changed since 2026-08-20. **C**, **D**, **E** are cheap and independent.
""")

# --------------------------------------------------------------------------
md("## 1. Environment")

code(r"""
import subprocess, os, sys, json, time, re, random, io

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
print("host RAM:", sh("free -g | awk '/^Mem/{print $2\" GB\"}'", quiet=True).stdout.strip())
print("cores   :", sh("nproc", quiet=True).stdout.strip())
""")

code(r"""
sh("apt-get -qq update && apt-get -qq install -y gengetopt help2man xxd > /dev/null 2>&1",
   check=False, quiet=True)
print("deps ok")
""")

# --------------------------------------------------------------------------
md(r"""## 2. Build, and refuse a clone that predates what we are measuring

`RNA_PHASE_SYNC` and the fixed NCU kernel pattern are both recent. A tree
without them would silently produce async timings (which is the whole mistake
this notebook exists to avoid) or profile nothing at all.""")

code(r"""
REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/intloop"

# Commits this notebook exists to MEASURE. A clone predating one of them reports
# the pre-fix behaviour, which reads as "the effect is absent" rather than "you
# cloned a stale tree".
REQUIRED_COMMITS = [
    ("d5a03af0", "RNA_PHASE_SYNC -- without it every int_loop number is a launch"),
    ("30a041ce", "ncu -k pattern fixed -- int_loop_kernel_<bs>, not int_loop_kernel"),
    ("abd14579", "PROFILE_INT_LOOP.md, the plan this notebook executes"),
]

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())

missing = [(c, w) for c, w in REQUIRED_COMMITS
           if sh("cd %s/port27 && git merge-base --is-ancestor %s HEAD && echo yes || echo no"
                 % (ROOT, c), check=False, quiet=True).stdout.strip() != "yes"]
for c, w in REQUIRED_COMMITS:
    print("  %s  %-58s %s" % (c, w, "MISSING" if (c, w) in missing else "present"))
if missing:
    raise SystemExit("STALE CLONE: this tree predates %s. Push port27, then re-run."
                     % ", ".join(c for c, _ in missing))
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
md(r"""## 3. The workload — the same 400 × 5601 as every other run

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
""")

# --------------------------------------------------------------------------
md(r"""## 4. The runner

`RNA_GPU_CHUNK=N` is a **hard cap on records per chunk** (`RNAfold.c:2078`), so
it controls the variable directly instead of going through the VRAM budget —
which is what lets §A put *both* datatypes at *both* chunk widths.

Every arm asserts the three things that make it meaningful: the sweep ran, the
phase-sync banner appeared, and the datatype gate agreed with intent.""")

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

PHASES = ("int_loop","hp_mb","load_my_c","modular_decomp","fetch_mx",
          "new_c_host","fml_host","fml_prev_host")
STAGES = ("build","prepare","prefill","backtrack","output","gpuinit","teardown","free")

RESULTS = {}
OUT = "/content/intloop.json"

def save():
    with open(OUT, "w") as f:
        json.dump({"commit": COMMIT, "clocks": clocks(), "n": N_BIG, "len": LEN,
                   "runs": RESULTS}, f, indent=1)

def run(tag, int16=False, chunk_cap=0, block_size=None, phase_sync=True,
        budget_mb=None, fa=None, extra_args=""):
    env = dict(os.environ)
    for k in ("RNA_FML_INT16","RNA_GPU_CHUNK","RNA_MIN_GPU_BATCH","RNA_PHASE_SYNC",
              "RNA_GPU_VRAM_BUDGET_MB","RNA_INT_LOOP_BLOCK_SIZE","RNA_BUILD_PIPELINE"):
        env.pop(k, None)
    env["RNA_GPU_CHUNK"]     = str(chunk_cap)     # 0 = budget alone decides
    env["RNA_MIN_GPU_BATCH"] = "1"
    if int16:       env["RNA_FML_INT16"] = "1"
    if phase_sync:  env["RNA_PHASE_SYNC"] = "1"
    if block_size:  env["RNA_INT_LOOP_BLOCK_SIZE"] = str(block_size)
    if budget_mb:   env["RNA_GPU_VRAM_BUDGET_MB"] = str(budget_mb)

    c0 = clock_ratio(); t0 = time.time()
    p = subprocess.run(["/usr/bin/time","-v",BIN,"--noPS"] + extra_args.split() +
                       ["-i", fa or BIG], capture_output=True, text=True, env=env)
    wall = time.time() - t0; c1 = clock_ratio()
    err = p.stderr

    # --- the assertions, before any number is believed ---
    if p.returncode != 0:
        print(err[-3000:]); raise SystemExit("%s: rc=%d" % (tag, p.returncode))
    if "sweep shape:" not in err:
        raise SystemExit("%s: NO 'sweep shape:' -- folded on the CPU, so this "
                         "profile describes something else" % tag)
    banner = "RNA_PHASE_SYNC=1" in err
    if banner != bool(phase_sync):
        raise SystemExit("%s: phase-sync banner=%s but asked for %s -- knob did "
                         "not engage (binary older than d5a03af0?)" % (tag, banner, phase_sync))
    i16_on = "RNA_FML_INT16=1" in err
    if i16_on != bool(int16):
        raise SystemExit("%s: int16 gate disagreed with intent" % tag)

    ph = dict(zip(PHASES, [float(x) for x in PHASE_RE.search(err).groups()])) \
         if PHASE_RE.search(err) else {}
    st = dict(zip(STAGES, [float(x) for x in STAGE_RE.search(err).groups()])) \
         if STAGE_RE.search(err) else {}
    shapes = SWEEP_RE.findall(err)
    bs = BS_RE.search(err)
    rss = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
    if m: rss = int(m.group(1))/1e6

    cells = sum(int(s[2]) for s in shapes)
    peak  = max((int(s[3]) for s in shapes), default=0)
    r = dict(wall=wall, phases=ph, stages=st, chunks=len(shapes), cells=cells,
             peak_records=peak, block_size=int(bs.group(1)) if bs else None,
             int16=i16_on, phase_sync=banner, chunk_cap=chunk_cap, rss=rss,
             clock_before=c0, clock_after=c1,
             ns_per_cell=(1e9*ph.get("int_loop",0)/cells) if cells else None,
             sha=__import__("hashlib").sha256(p.stdout.encode()).hexdigest()[:12])
    RESULTS[tag] = r; save()
    print("  %-22s int_loop %7.2f s  %6.2f ns/cell  chunks %2d  peak/chunk %3d  "
          "bs %-4s sha %s" % (tag, ph.get("int_loop",0), r["ns_per_cell"] or 0,
                              r["chunks"], peak, r["block_size"], r["sha"]))
    return r

print("runner ready -- results stream to", OUT)
""")

# --------------------------------------------------------------------------
md(r"""## 4b. What a laptop already says — and why it may invert at scale

Dry-run on an RTX 3050 at **24 × 1200** (a ~14 000-block grid), same runner,
phase-synced, `sha` identical in every arm:

| block size | ns/cell | vs 32 |
|---|---|---|
| 32 | 67.06 | — |
| **64** | **54.02** | **0.81×** |
| 128 | 62.75 | 0.94× |
| 256 | 69.62 | 1.04× |

| records/chunk | ns/cell |
|---|---|
| 4 | 108.79 |
| 8 | 67.06 |
| 12 | 51.81 |

**Two things, and the second is why this notebook exists.**

1. **64 was fastest** — 19 % better than the STOPGAP default, on the one size
   never tried. 256 was slowest, reproducing the 2026-08-20 finding.
2. **More records per chunk was FASTER here** (108.8 → 51.8 ns/cell from 4 to
   12), which is the **opposite sign** to the T4 at 400 × 5601, where 28.6 → 36.4
   records/chunk cost 17.12 → 21.98 ns/cell.

A coherent reading: at ~14 000 blocks the device is **not saturated**, so more
blocks in flight genuinely helps — exactly what `int_loop.cu:1168` concluded. At
~2.2 M blocks it is saturated, occupancy is already maxed, and what is left to
lose is **locality** on `my_c`.

If that is right, both statements are true at their own scale and neither
generalises. **So do not read the laptop numbers as a prediction** — read them as
the thing the Colab result has to be compared against. If §A comes back with the
laptop's sign at 400 × 5601, the saturation story is wrong and something simpler
is going on.

### And the first NCU counters this kernel has ever produced

Same dry-run, `-k regex:'^int_loop_kernel_[0-9]+$'` (the pattern fixed in
`30a041ce`; the old one matched nothing), block size 32:

| metric | `int_loop_kernel_32` | for contrast, `modular_decomposition_kernel` |
|---|---|---|
| **L2 hit rate** | **85.97 %** | **6.1 %** |
| achieved occupancy | 27.29 % | — |

**That 86 % is the number to watch, and it makes the locality hypothesis
falsifiable.** `modular_decomposition_kernel` was measured at a 6 % L2 hit rate
and pinned to the DRAM roof; `int_loop_kernel` is nothing like it — at this size
it is getting excellent reuse on `my_c`.

So the hypothesis has a sharp prediction: **widening the chunk at 400 × 5601
should visibly drive that hit rate down.** If §D comes back with L2 still near
86 % at both chunk widths, the locality story is **dead** and the +28 % is
something else.

Occupancy at 27 % is also below the 50 % structural ceiling for one warp per
block, which at ~14 000 blocks most likely means the device simply is not full —
consistent with the small-grid reading above, and worth re-checking at 2.2 M.

*These are one run each on a power-capped laptop; they are a prior, not
evidence.*
""")

# --------------------------------------------------------------------------
md(r"""## A. The decisive experiment: chunk width vs datatype

Four arms. Both datatypes at **both** chunk widths, so the two variables come
apart. `RNA_GPU_CHUNK` caps records per chunk directly; 29 → 14 chunks and
37 → 11 chunks at 400 records.

| arm | datatype | rec/chunk | isolates |
|---|---|---|---|
| A | i32 | 29 | baseline (expect ≈107 s) |
| B | i16 | 37 | baseline (expect ≈138 s) |
| **C** | **i16** | **29** | **int16 at i32's width** |
| **D** | **i32** | **37** | **int32 at i16's width** |

**Prediction, recorded before the run so it can be wrong: C ≈ 107 s, D ≈ 138 s**
— i.e. the cost follows chunk width, not the datatype.

* C ≈ A and D ≈ B → **chunk width**. int16's `int_loop` penalty is an artifact
  of it enabling wider chunks, and §19's give-back needs re-stating.
* C ≈ B and D ≈ A → **the encoding**, by a mechanism inspection has not found —
  which would be a surprising and valuable result, since `int_loop.cu` has no
  int16 in it.
* In between → both, and the split says how much.""")

code(r"""
print("A. chunk width vs datatype -- all phase-synced, natural VRAM budget")
run("A_i32_c29", int16=False, chunk_cap=29)
run("B_i16_c37", int16=True,  chunk_cap=37)
run("C_i16_c29", int16=True,  chunk_cap=29)
run("D_i32_c37", int16=False, chunk_cap=37)
""")

code(r"""
a,b,c,d = (RESULTS.get(k) for k in ("A_i32_c29","B_i16_c37","C_i16_c29","D_i32_c37"))
if all((a,b,c,d)):
    print("%-14s %-6s %-10s %12s %12s" % ("arm","type","rec/chunk","int_loop s","ns/cell"))
    for tag,r in (("A",a),("B",b),("C",c),("D",d)):
        print("%-14s %-6s %-10d %12.2f %12.2f"
              % (tag, "i16" if r["int16"] else "i32", r["peak_records"],
                 r["phases"]["int_loop"], r["ns_per_cell"]))
    print()
    # the two contrasts that answer the question
    w_i32 = d["ns_per_cell"]/a["ns_per_cell"]     # i32: 37 vs 29 records
    w_i16 = b["ns_per_cell"]/c["ns_per_cell"]     # i16: 37 vs 29 records
    t_29  = c["ns_per_cell"]/a["ns_per_cell"]     # at 29: i16 vs i32
    t_37  = b["ns_per_cell"]/d["ns_per_cell"]     # at 37: i16 vs i32
    print("effect of WIDTH  (37 vs 29 rec/chunk):  i32 %.3fx   i16 %.3fx" % (w_i32, w_i16))
    print("effect of TYPE   (i16 vs i32):          @29 %.3fx   @37 %.3fx" % (t_29, t_37))
    print()
    print("records/chunk ratio 37/29 = %.3f;  the §0 correspondence was 1.284" % (37/29))
    print()
    if max(t_29, t_37) < 1.10 and min(w_i32, w_i16) > 1.15:
        print("=> CHUNK WIDTH. The datatype contrast is small at both widths and the")
        print("   width contrast is large for both datatypes. int16's int_loop")
        print("   penalty is a chunking side-effect, not the encoding.")
    elif max(w_i32, w_i16) < 1.10 and min(t_29, t_37) > 1.15:
        print("=> THE ENCODING, by a mechanism inspection has not found. int_loop.cu")
        print("   contains no int16 code, so this needs NCU (section D) before it is")
        print("   believed.")
    else:
        print("=> BOTH, or neither cleanly. Read the four numbers rather than a verdict.")
    shas = {r["sha"] for r in (a,b,c,d)}
    print("\nsha:", shas, "MATCH" if len(shas)==1 else "*** DIFFER -- correctness bug ***")
""")

# --------------------------------------------------------------------------
md(r"""## B. Block size, re-opened only where the axes changed

`int_loop.cu:1152-1173` already records a 2026-08-20 NCU sweep: `BLOCK_SIZE=256`
reached 36–62 % occupancy against 32's 5–33 % and was **slower at every grid
size tried**, 1.07× to 2.35×, never faster. The stated cause is that each
`(i,j)` search is bounded by `MAXLOOP = 30`, so only one warp of work exists per
cell and a bigger block just adds `__syncthreads()` for idle threads.

**Do not repeat that sweep.** Three things have changed since:

| axis | then | now |
|---|---|---|
| grid | ≤ 54 000 blocks | **~2.2 M** — and the gap was *narrowing* monotonically with grid size |
| datatype | int32 only | **int16 too**, which changes the working set underneath it |
| sizes | 32 vs 256 | **64 included** |

**64 is the interesting one.** The "only one warp of work" argument is strongest
against 256 and weakest against 64 — which adds exactly one warp and, on sm_75
(16 blocks/SM, 32 warps/SM, ~256 B shared per block so neither binds first), is
the step that lifts the occupancy ceiling from 50 % to 100 %. It is the single
untested size where the recorded reasoning and the occupancy arithmetic
disagree.

All arms at a **fixed** chunk width (29) so block size is the only variable.""")

code(r"""
print("B. block size at a FIXED chunk width (29 rec/chunk), phase-synced")
for bs in (32, 64, 128, 256):
    run("B_i32_bs%d" % bs, int16=False, chunk_cap=29, block_size=bs)
for bs in (32, 64):
    run("B_i16_bs%d" % bs, int16=True, chunk_cap=29, block_size=bs)
""")

code(r"""
rows = [(k, RESULTS[k]) for k in sorted(RESULTS) if k.startswith("B_") and "bs" in k]
if rows:
    print("%-16s %-6s %-6s %12s %12s %10s" % ("arm","type","bs","int_loop s","ns/cell","vs bs=32"))
    base = {}
    for k, r in rows:
        t = "i16" if r["int16"] else "i32"
        if r["block_size"] == 32: base[t] = r["ns_per_cell"]
    for k, r in rows:
        t = "i16" if r["int16"] else "i32"
        rel = r["ns_per_cell"]/base[t] if base.get(t) else float("nan")
        print("%-16s %-6s %-6d %12.2f %12.2f %9.3fx"
              % (k, t, r["block_size"], r["phases"]["int_loop"], r["ns_per_cell"], rel))
    shas = {r["sha"] for _, r in rows}
    print("\nsha:", shas, "MATCH" if len(shas)==1 else
          "*** DIFFER -- a block size changed the ANSWER, which is a bug, not a tuning result ***")
    print("\nThe 2026-08-20 result was: 256 never faster than 32 at any grid up to")
    print("~54000 blocks. If 64 beats 32 here at ~2.2M, the STOPGAP default is")
    print("worth revisiting -- and the comment at int_loop.cu:1152 needs the new")
    print("measurement written into it, not replaced by it.")
""")

# --------------------------------------------------------------------------
md(r"""## C. Is phase-sync still free at a second chunk width?

§19.4 measured phase-sync at **~0 % of wall** (535.14 synced vs 535.11 async) and
explained it: there was no phase overlap to destroy, because `hp_mb_3p_i()`'s two
synchronous pageable H2Ds already serialised every row.

That was one chunk width. This checks a second one rather than assuming it
generalises — and it is the honest way to keep using "never quote a synced wall"
as a rule with a known cost rather than a superstition.""")

code(r"""
print("C. phase-sync overhead at 37 rec/chunk (§19 measured ~0% at 29)")
s = run("C_i32_c37_sync",  int16=False, chunk_cap=37, phase_sync=True)
a = run("C_i32_c37_async", int16=False, chunk_cap=37, phase_sync=False)
print("\n  synced wall %.1f s   async wall %.1f s   overhead %+.2f%%"
      % (s["wall"], a["wall"], 100*(s["wall"]-a["wall"])/a["wall"]))
print("  async int_loop reads %.2f s; synced reads %.2f s (%.1fx) -- the async"
      % (a["phases"]["int_loop"], s["phases"]["int_loop"],
         s["phases"]["int_loop"]/max(a["phases"]["int_loop"],1e-9)))
print("  number is a LAUNCH, which is the whole reason this notebook syncs.")
""")

# --------------------------------------------------------------------------
md(r"""## D. NCU on `int_loop_kernel` — the first counters this kernel has ever had

`tools/make_nb_profile272_deep.py` ran `ncu -k int_loop_kernel` for weeks and
matched **nothing**: `int_loop_kernel_body.inc` is `#include`d once per candidate
block size and concatenates the size on, so the real symbols are
`int_loop_kernel_32 / _64 / _128 / _256`. It recorded `"int_loop_kernel/i32":
null` beside real data for `modular_decomposition_kernel`, and nobody chased the
null — on what turned out to be 30 % of GPU time. Fixed in `30a041ce`.

Small input on purpose: NCU replays kernels and does not scale. What matters
here is the **L2 hit rate as a function of chunk width** — if §A's locality story
is right, widening the chunk should cost L2 hits on `my_c` while DRAM throughput
stays flat.""")

code(r"""
PN, PL = 24, 2000
pfa = ROOT + "/fa/prof.fa"
random.seed(4242)
with open(pfa, "w") as f:
    for i in range(PN):
        f.write(">p%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(PL))))

METRICS = ",".join([
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "gpu__time_duration.sum"])

# land mid-sweep: the sweep prints its iteration count, and half of it can never
# be past the end -- the other way the first profiling attempt reported nothing.
p = subprocess.run([BIN,"--noPS","-i",pfa], capture_output=True, text=True,
                   env=dict(os.environ, RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1"))
shapes = SWEEP_RE.findall(p.stderr)
assert shapes, "no sweep -- nothing to profile"
LAUNCH = max(int(m[0]) for m in shapes)
SKIP, COUNT = max(1, LAUNCH//2), 5
print("~%d launches; sampling %d from %d" % (LAUNCH, COUNT, SKIP))

NCU = {}
for tag, i16, cap in (("i32/c8", False, 8), ("i32/c24", False, 24),
                      ("i16/c8", True, 8),  ("i16/c24", True, 24)):
    log = "/content/ncu_%s.csv" % tag.replace("/","_")
    env = ("RNA_GPU_CHUNK=%d RNA_MIN_GPU_BATCH=1 RNA_PHASE_SYNC=1%s"
           % (cap, " RNA_FML_INT16=1" if i16 else ""))
    sh("%s ncu --target-processes all -k regex:'^int_loop_kernel_[0-9]+$' "
       "--launch-skip %d --launch-count %d --metrics %s --csv --log-file %s "
       "%s --noPS -i %s > /dev/null 2>&1"
       % (env, SKIP, COUNT, METRICS, log, BIN, pfa), check=False, quiet=True)
    body = open(log).read() if os.path.exists(log) else ""
    if "No kernels were profiled" in body or "lts__t_sector_hit_rate" not in body:
        print("  %-10s *** NO KERNELS PROFILED -- broken probe, not a result ***" % tag)
        NCU[tag] = None; continue
    import csv
    rows = list(csv.DictReader(io.StringIO(
        "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
    agg = {}
    for r in rows:
        nm = r.get("Metric Name"); v = (r.get("Metric Value") or "").replace(",","")
        try: agg.setdefault(nm, []).append(float(v))
        except ValueError: pass
    NCU[tag] = {k: sum(v)/len(v) for k, v in agg.items()}
    print("  %-10s ok (%d rows)" % (tag, len(rows)))

with open("/content/intloop_ncu.json","w") as f: json.dump(NCU, f, indent=1)
""")

code(r"""
good = {k:v for k,v in NCU.items() if v}
if good:
    keys = sorted({m for v in good.values() for m in v})
    print("%-56s %s" % ("metric", "  ".join("%12s" % k for k in good)))
    for m in keys:
        print("%-56s %s" % (m[:56], "  ".join("%12.2f" % good[k].get(m, float('nan'))
                                              for k in good)))
    print()
    print("READ L2 FIRST, ACROSS CHUNK WIDTHS. The laptop measured 85.97% at")
    print("block size 32 -- nothing like modular_decomposition_kernel's 6.1%.")
    print("The locality hypothesis predicts that hit rate FALLS as the chunk")
    print("widens, while DRAM throughput stays flat. If it stays near 86% at")
    print("both widths, the hypothesis is DEAD and the +28% is something else.")
    print("Occupancy: 27.29% on the laptop at ~14000 blocks, below the 50%")
    print("structural ceiling for one warp per block -- i.e. the device was not")
    print("full. At 2.2M blocks it should be at the ceiling, or the ceiling is")
    print("not what is binding.")
""")

# --------------------------------------------------------------------------
md(r"""## E. `-g` at scale — a correctness datapoint, not a timing one

G-quadruplexes became accelerated on 2026-09-10 (`e4d30772`) and have never run
on anything larger than 20 records. This is cheap and it is the kind of thing
that is much better to discover here than in a user's hands.

It compares the GPU against **this same binary with the accelerator off**, so
the accelerator is the only variable.""")

code(r"""
GN, GL = 60, 1200
gfa = ROOT + "/fa/gq.fa"
random.seed(20260911)
blocks = ["GGGG","A","GGG","UUA","GGGG","CU","GGG","AAA","AUGCAUGC",
          "GGGG","U","GGG","AC","GGGG","AU","GGG","CUAGCUAGCUAGC"]
with open(gfa, "w") as f:
    for i in range(GN):
        s = ""
        while len(s) < GL: s += random.choice(blocks)
        f.write(">g%d\n%s\n" % (i, s[:GL]))

env0 = dict(os.environ); [env0.pop(k, None) for k in
            ("RNA_GPU_CHUNK","RNA_FML_INT16","RNA_PHASE_SYNC","RNA_MIN_GPU_BATCH")]
cpu = subprocess.run([BIN,"--noPS","-g","-i",gfa], capture_output=True, text=True, env=env0)

for tag, extra in (("gpu", {}), ("gpu+i16", {"RNA_FML_INT16":"1"})):
    env = dict(env0, RNA_GPU_CHUNK="0", RNA_MIN_GPU_BATCH="1", **extra)
    t0 = time.time()
    gpu = subprocess.run([BIN,"--noPS","-g","-i",gfa], capture_output=True, text=True, env=env)
    dt = time.time() - t0
    swept = gpu.stderr.count("sweep shape:")
    same  = (cpu.stdout == gpu.stdout)
    plus  = sum(l.count("+") for l in gpu.stdout.splitlines() if set(l) <= set(".()+~ -0123456789."))
    print("  %-9s sweep=%d  identical=%s  wall %.1fs  '+' chars %d"
          % (tag, swept, same, dt, plus))
    if not same:
        for a, b in zip(cpu.stdout.splitlines(), gpu.stdout.splitlines()):
            if a != b: print("    cpu: %s\n    gpu: %s" % (a[:90], b[:90])); break
print("\n60 x 1200 G-rich. 'identical=True' with sweep>0 is the result; a single")
print("'+' per quadruplex instead of an expanded box would mean the bps backtrack")
print("patch regressed (see PORT_GQUAD_SPEC.md G2).")
""")

# --------------------------------------------------------------------------
md("## F. Save")

code(r"""
save()
print("wrote", OUT, "and /content/intloop_ncu.json")
print("\narms:")
for k in sorted(RESULTS):
    r = RESULTS[k]
    print("  %-22s int_loop %7.2f s  %6.2f ns/cell  chunks %2d  bs %-4s sha %s"
          % (k, r["phases"].get("int_loop",0), r["ns_per_cell"] or 0,
             r["chunks"], r["block_size"], r["sha"]))
try:
    from google.colab import files
    files.download(OUT)
    files.download("/content/intloop_ncu.json")
except Exception as e:
    print("(download by hand:", e, ")")
""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

dst = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_IntLoop.ipynb"
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (dst, len(cells)))

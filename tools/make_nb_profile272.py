#!/usr/bin/env python3
"""Build the CUDA_RNAFold PROFILING Colab notebook.

Split out of benchmark v5 on 2026-09-08. v5 measures wall clock and defends an
extrapolation, and it does that well -- but it builds three trees and folds a
CPU baseline before it can profile anything, and none of that is needed to read
a hardware counter. This notebook builds ONE tree (port27 + CUDA) and goes
straight at the question v3, v4 and v5 all left open:

  int16 halved the fml_j stream and bought 1.009x of wall on a T4. Is
  modular_decomposition_kernel still DRAM-bound there, or did it leave the
  roof and become limited by something else?

Two explanations call for opposite next moves:
  - fml_j was never the whole stream. If fml_i also reaches DRAM, halving one
    of two comparable streams gives ~0.75x before overheads, and there is no
    second lever hiding here.
  - the kernel left the DRAM roof. It was at 88.3% of peak at int32. If int16
    dropped it well below, something else is the limiter and there IS a lever.

A third question rides along: the kernel re-reads each fML column once per
sweep row, and a proposed optimisation stages those re-reads in shared memory.
If lts__t_sector_hit_rate is already high, L2 does that for free and the work
would duplicate the hardware.

WHAT THE FIRST ATTEMPT GOT WRONG, all of it silent (see PORT_FEATURE_AUDIT.md):
  - the probe held 6 records against a MIN_GPU_BATCH of 10, so the fold went
    entirely to the CPU and NOT ONE KERNEL LAUNCHED. ncu said "No kernels were
    profiled" under a screenful of perfectly normal folded structures.
  - ncu's --csv report and RNAfold's structures BOTH go to stdout, so the .csv
    held dot-bracket strings and zero metric rows.
  - that warning is on stdout, so a gate grepping stderr never fires.
  - --launch-skip was a hardcoded 2000 with nothing checking the kernel is
    launched that many times.

Every one of those is now measured or asserted rather than assumed.

usage: python3 tools/make_nb_profile272.py [out.ipynb]
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
md(r"""# CUDA_RNAFold - `modular_decomposition_kernel` profile

**One tree, one question.** Benchmark v5 measures wall clock and defends an
extrapolation; it needs three builds and a CPU baseline to do it. Reading a
hardware counter needs none of that, so this notebook builds **only** port27
with CUDA and profiles.

## The question

int16 halves the `fml_j` DRAM stream. It bought **1.31x** on a bandwidth-bound
RTX 3050, **1.12x** in bench v3, and **1.009x** on the T4 in bench v5 -- while
genuinely cutting the chunk count from 5 to 4. So the saving is real and the
time is not. Why?

| explanation | what it predicts | what to do next |
|---|---|---|
| `fml_j` was never the whole stream | `fml_i` also reaches DRAM; halving one of two comparable streams gives ~0.75x before overheads | nothing -- no lever here |
| the kernel left the DRAM roof | int32 sat at 88.3% of peak; int16 well below | there IS a second lever |

And one rider: the kernel re-reads each `fML` column once per sweep row, and a
proposed optimisation stages those re-reads in shared memory. If
`lts__t_sector_hit_rate` is already high, **L2 is doing it for free** and that
work would duplicate the hardware.

## What the first attempt got wrong, and how this one refuses to

The profiling cell in v4/v5 reported nothing, four times over, and every failure
was silent:

- the probe held **6 records against a `MIN_GPU_BATCH` of 10**, so the fold went
  entirely to the CPU and **not one kernel launched**. `==WARNING== No kernels
  were profiled`, under a screenful of perfectly normal folded structures;
- `ncu --csv` writes to **stdout and so does RNAfold**, so the `.csv` held
  dot-bracket strings and zero metric rows;
- that warning is on **stdout**, so a gate grepping `stderr` never fires;
- `--launch-skip 2000` was hardcoded with nothing checking the kernel is
  launched 2000 times.

Here the launch count is **measured first and the skip derived from it**, the
report goes to `--log-file`, the gate reads the log, and the run fails loudly
rather than printing an empty table.
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
              "clocks_throttle_reasons.active --format=csv,noheader",
              quiet=True).stdout.strip()

print(clocks())
print(sh("ncu --version | head -3", check=False, quiet=True).stdout.strip())""")

code(r"""sh("apt-get -qq update && apt-get -qq install -y gengetopt help2man texinfo "
   "doxygen autoconf automake libtool > /dev/null 2>&1", check=False)
print("toolchain ready")""")

# --------------------------------------------------------------------------
md("""## 2. One tree, one build

Only `port27` with `--enable-cuda`. No upstream worktree, no no-CUDA arm --
nothing here compares against the CPU, so building them would be pure wall
clock.

**`git log` is printed and checked.** Bench v5 ran against `10acb583` because
Colab clones `origin/port27` and origin was three commits behind the work; its
`RNA_MIN_GPU_BATCH` arms failed for that reason alone. A profile of the wrong
commit is worth as little as a benchmark of one.""")

code(r"""REPO   = "https://github.com/LukeTheGeneWriter/CUDA_RNAFold.git"
BRANCH = "port27"
ROOT   = "/content/prof"
EXPECT = None      # set to a short sha to ASSERT the commit, e.g. "364f92f6"

sh("rm -rf %s && mkdir -p %s" % (ROOT, ROOT))
sh("git clone -q %s %s/port27 && cd %s/port27 && git checkout -q %s"
   % (REPO, ROOT, ROOT, BRANCH))
COMMIT = sh("cd %s/port27 && git rev-parse --short HEAD" % ROOT, quiet=True).stdout.strip()
print("commit :", sh("cd %s/port27 && git log --oneline -1" % ROOT, quiet=True).stdout.strip())
if EXPECT and not COMMIT.startswith(EXPECT):
    raise SystemExit("*** commit is %s, expected %s -- push before running ***"
                     % (COMMIT, EXPECT))""")

code(r"""t0 = time.time()
# Two fixes a human had to add by hand on the 2026-09-08 run, both of them
# already-known defects that tools/standup_git_build.sh works around:
#  1. the bundled dlib/libsvm ship as TARBALLS in git and configure refuses to
#     proceed until they are unpacked (the release tarball ships them unpacked,
#     so only a git-tree build hits it);
#  2. PYTHON3 must reach configure, or --without-python leaves $(PYTHON3) empty
#     while doc/source/man/Makefile.am:38 still expands
#     "$(PYTHON3) ../../man2rst.py" -- make then tries to EXECUTE a mode-644
#     file and dies with "Permission denied" on all 25 man pages. chmod is
#     belt-and-braces for the same defect.
for pat, flag in (("src/dlib-*.tar.bz2", "-xjf"), ("src/libsvm-*.tar.gz", "-xzf")):
    for t in sh("ls %s/port27/%s 2>/dev/null" % (ROOT, pat), check=False,
                quiet=True).stdout.split():
        d = re.sub(r"\.tar\.(bz2|gz)$", "", os.path.basename(t))
        if not os.path.isdir("%s/port27/src/%s" % (ROOT, d)):
            print("  unpacking", os.path.basename(t))
            sh("tar %s %s -C %s/port27/src/" % (flag, t, ROOT), check=False, quiet=True)
sh("cd %s/port27 && ./autogen.sh > /dev/null 2>&1" % ROOT, check=False, quiet=True)
sh("chmod +x %s/port27/doc/man2rst.py" % ROOT, check=False, quiet=True)
p = sh("cd %s/port27 && ./configure --without-python --without-perl --without-swig "
       "--without-doc --without-rnaxplorer --without-forester --without-kinfold "
       "--without-rnalocmin --enable-cuda CFLAGS='-g -O2' CXXFLAGS='-g -O2' "
       "PYTHON3=\"$(command -v python3)\" > /content/conf.log 2>&1" % ROOT,
       check=False, quiet=True)
if p.returncode:
    print(sh("tail -25 /content/conf.log", quiet=True).stdout); raise SystemExit("configure failed")
p = sh("cd %s/port27 && make -j$(nproc) > /content/make.log 2>&1" % ROOT, check=False, quiet=True)
if p.returncode:
    print(sh("tail -30 /content/make.log", quiet=True).stdout); raise SystemExit("make failed")
BIN = ROOT + "/port27/src/bin/RNAfold"
assert os.path.exists(BIN)
print("built in %.0fs, %s nvcc invocations"
      % (time.time()-t0, sh("grep -c nvcc /content/make.log", check=False, quiet=True).stdout.strip()))""")

# --------------------------------------------------------------------------
md(r"""## 3. Workload, sized by the two constraints that broke the last attempt

`NREC` must exceed `MIN_GPU_BATCH` (10, and a compile-time constant before
`130e2c8a`) or the whole chunk folds on the CPU and there is nothing to profile.
`RNA_MIN_GPU_BATCH=1` is also set, but as belt-and-braces -- it does not exist
on older commits, which is exactly how the last attempt failed.

`LEN` sets how many times the kernel is launched: roughly one launch per sweep
row, so ~`LEN` launches. That is what makes `--launch-skip` safe, and the next
cell measures it instead of trusting this paragraph.""")

code(r"""NREC = 16          # > MIN_GPU_BATCH(10) so the batch reaches the GPU at all
LEN  = 2000        # ~LEN sweep rows => ~LEN kernel launches
SEED = 20260908

os.makedirs(ROOT + "/fa", exist_ok=True)
FA = ROOT + "/fa/probe.fa"
random.seed(SEED)
with open(FA, "w") as f:
    for i in range(NREC):
        f.write(">p%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(LEN))))
print("%d x %d nt" % (NREC, LEN))""")

# --------------------------------------------------------------------------
md(r"""## 4. Preflight: does this workload reach the GPU, and how many launches?

Run once WITHOUT ncu. Two numbers come out, and both were assumed last time:

- **sweeps** -- zero means the batch folded on the CPU and no profile is
  possible. This is the check whose absence produced "No kernels were profiled";
- **iterations** -- the sweep's own row count, which is very nearly the number
  of `modular_decomposition_kernel` launches. `--launch-skip` is derived from
  it rather than hardcoded, so it cannot land past the end.""")

code(r"""def run_plain(int16):
    env = dict(os.environ); env["RNA_GPU_CHUNK"] = "0"; env["RNA_MIN_GPU_BATCH"] = "1"
    env.pop("RNA_FML_INT16", None)
    if int16: env["RNA_FML_INT16"] = "1"
    t = time.time()
    p = subprocess.run([BIN, "--noPS", "-i", FA], capture_output=True, text=True, env=env)
    return p, time.time() - t

SWEEP_RE = re.compile(r"sweep shape: (\d+) iterations")
p, wall = run_plain(False)
sweeps = p.stderr.count("sweep shape:")
iters  = [int(x) for x in SWEEP_RE.findall(p.stderr)]
print("sweeps        :", sweeps)
print("iterations    :", iters)
print("wall          : %.1fs" % wall)
if sweeps == 0:
    raise SystemExit("*** batch folded on the CPU -- raise NREC above MIN_GPU_BATCH ***")
LAUNCHES = max(iters)
SKIP = max(1, LAUNCHES // 2)          # sample mid-sweep, never past the end
COUNT = 5
print("\n=> profiling %d launches starting at %d of ~%d" % (COUNT, SKIP, LAUNCHES))

# The int16 gate must actually engage, or the "i16" arm is the i32 arm relabelled.
p16, _ = run_plain(True)
assert "RNA_FML_INT16=1" in p16.stderr, "int16 gate did not engage"
print("int16 gate engages:", "RNA_FML_INT16=1" in p16.stderr)
print("i32/i16 identical answers:", p.stdout == p16.stdout)""")

# --------------------------------------------------------------------------
md(r"""## 5. Profile

`--log-file` is not cosmetic: `ncu --csv` writes its report to **stdout, and so
does RNAfold**, so without it the file fills with dot-bracket structures and no
metrics. The app's own stdout goes to `/dev/null`.

Sampled, never whole-app: kernel replay saves and restores multi-GB buffers per
launch, and whole-app profiling on this project has already cost a five-hour
dead end.""")

code(r"""METRICS = ",".join([
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "gpu__time_duration.sum",
])

def profile(tag, int16):
    log = "/content/ncu_%s.csv" % tag
    env = "RNA_GPU_CHUNK=0 RNA_MIN_GPU_BATCH=1" + (" RNA_FML_INT16=1" if int16 else "")
    sh("%s ncu --target-processes all -k modular_decomposition_kernel "
       "--launch-skip %d --launch-count %d --metrics %s --csv --log-file %s "
       "%s --noPS -i %s > /dev/null 2> /content/ncu_%s.err"
       % (env, SKIP, COUNT, METRICS, log, BIN, FA, tag), check=False, quiet=True)
    body = open(log).read() if os.path.exists(log) else ""
    if "No kernels were profiled" in body or "dram__bytes_read" not in body:
        print("*** %s: NO KERNELS PROFILED ***" % tag)
        print(body[:800] or open("/content/ncu_%s.err" % tag).read()[:800])
        return None
    rows = [r for r in csv.DictReader(io.StringIO(
                "\n".join(l for l in body.splitlines() if not l.startswith("==")))) ]
    print("  %s: %d metric rows" % (tag, len(rows)))
    return rows

R = {t: profile(t, i) for t, i in (("i32", False), ("i16", True))}
if R["i32"] is None or R["i16"] is None:
    raise SystemExit("*** profiling produced no kernels -- see above ***")""")

# --------------------------------------------------------------------------
md("## 6. The answer")

code(r"""def agg(rows):
    out = {}
    for r in rows:
        name = r.get("Metric Name") or r.get("Metric")
        val  = (r.get("Metric Value") or "").replace(",", "")
        if not name: continue
        try: v = float(val)
        except ValueError: continue
        out.setdefault(name, []).append(v)
    return {k: sum(v)/len(v) for k, v in out.items()}

A32, A16 = agg(R["i32"]), agg(R["i16"])
keys = sorted(set(A32) | set(A16))
print("%-58s %14s %14s %8s" % ("metric", "int32", "int16", "i16/i32"))
for k in keys:
    a, b = A32.get(k), A16.get(k)
    if a is None or b is None: continue
    ratio = (b/a) if a else float("nan")
    print("%-58s %14.3f %14.3f %8.3f" % (k[:58], a, b, ratio))

def find(d, frag):
    for k, v in d.items():
        if frag in k: return v
    return None

print()
dram32, dram16 = find(A32, "dram_throughput"), find(A16, "dram_throughput")
sm32,   sm16   = find(A32, "sm__throughput"), find(A16, "sm__throughput")
by32,   by16   = find(A32, "dram__bytes_read"), find(A16, "dram__bytes_read")
l2_32,  l2_16  = find(A32, "lts__t_sector_hit_rate"), find(A16, "lts__t_sector_hit_rate")

if None not in (dram32, dram16):
    print("DRAM %% of peak : int32 %.1f%% -> int16 %.1f%%" % (dram32, dram16))
    print("SM   %% of peak : int32 %.1f%% -> int16 %.1f%%" % (sm32, sm16))
    print()
    if dram16 >= 70:
        print("VERDICT: still DRAM-bound under int16. fml_j was NOT the whole")
        print("         stream -- fml_i and friends still reach DRAM, so halving")
        print("         one of several comparable streams was always going to")
        print("         give a fraction of 2x. No second lever here.")
    elif sm16 > dram16:
        print("VERDICT: the kernel LEFT the DRAM roof and is now SM/latency")
        print("         bound. There IS a further lever, and int16's poor")
        print("         showing is the decode cost dominating once the bytes")
        print("         stopped being the constraint -- the starved-SM regime.")
    else:
        print("VERDICT: neither roof. Latency- or occupancy-bound; look at")
        print("         launch shape before touching the data type again.")
if None not in (by32, by16):
    print("\nDRAM bytes read: %.3f the int32 figure (2.0x saving would be 0.500)"
          % (by16/by32))
if l2_32 is not None:
    print("L2 sector hit rate: int32 %.1f%%, int16 %.1f%%" % (l2_32, l2_16))
    print("  (high => L2 already catches the per-row fML column re-reads, and")
    print("   staging them in shared memory would duplicate the hardware.)")
print("\nclocks now:", clocks())""")

code(r"""out = {"date": time.strftime("%Y-%m-%d"), "commit": COMMIT, "clocks": clocks(),
       "nrec": NREC, "len": LEN, "launches": LAUNCHES, "skip": SKIP, "count": COUNT,
       "i32": A32, "i16": A16}
with open("/content/profile272.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:2500])""")

# --------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

dst = sys.argv[1] if len(sys.argv) > 1 else "CUDA_RNAFold_Profile272.ipynb"
with open(dst, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote %s (%d cells)" % (dst, len(cells)))

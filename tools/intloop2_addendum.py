# =============================================================================
# ADDENDUM -- why is int_loop_kernel latency-bound, and what does block size
# actually do to it?
#
# RUN THIS AFTER the notebook's own sections. It PROFILES, it does not TIME, so
# a warm device is fine -- but it must not overlap a timed arm, and NCU serialises
# the device while it runs.
#
# WHAT IS MISSING WITHOUT IT. STRESS272_RESULTS.md 20.3 established that
# int_loop_kernel sits at 44-48% occupancy, 40-44% SM throughput and 4.2-5.2% of
# DRAM peak -- so it is latency-bound rather than bandwidth- or compute-bound.
# It did NOT establish WHICH latency, and for a latency-bound kernel the stall
# reason is the decisive metric. Nor did it profile block size 64 at all: the
# 10.8% win was measured end-to-end and the occupancy mechanism behind it is
# still inferred.
#
# Two open readings this settles:
#
#   1. Doubling the occupancy ceiling (32 -> 64) bought only 10.8%. Either the
#      achieved occupancy did not follow the ceiling, or extra warps do not hide
#      this kernel's stall. Part 1 reports achieved occupancy AND the binding
#      limiter at 32/64/128, so those separate.
#   2. The suspected latency source is the interior-loop energy tables --
#      P->int22 is ~202 KB and P->int21 ~40 KB, far past a 64 KB L1 but
#      comfortably L2-resident, which is exactly the L1 ~75% / L2 ~90% pattern
#      measured. That is a HYPOTHESIS. `long_scoreboard` dominating would support
#      it; `wait` or `barrier` dominating would point at the cooperative
#      decode/prefix-sum chain instead, and would explain why more warps helped
#      so little.
#
# Part 2 is opt-in and names the exact source line. It needs -lineinfo, which
# this build does not carry (NVCC_FLAGS is "-O3 -gencode..."), so it rebuilds
# ONE object, profiles, and PUTS THE ORIGINAL BINARY BACK.
# =============================================================================

WANT_SOURCE_LEVEL = False       # Part 2: rebuild int_loop.lo with -lineinfo
STALL_BLOCK_SIZES = (32, 64, 128)
STALL_N, STALL_LEN, STALL_CAP = 60, 1800, 24
STALL_SKIP, STALL_COUNT = 140, 3

import os, re, io, json, random, subprocess, collections, shutil
import csv as _csv_add

ROOT_A = globals().get("ROOT", "/content/intloop2")
BIN_A  = globals().get("BIN",  ROOT_A + "/port27/src/bin/RNAfold")
assert os.path.exists(BIN_A), "no RNAfold at %s -- run the build cell first" % BIN_A
STALLS = {}

# A realistic grid. At cap 24 and launch ~140 this is ~40 000 blocks, the same
# order as the 24 072 that section 20.3 profiled -- NOT a toy, because occupancy
# and stall mix both move with grid size.
FA_A = "%s/fa/stall_%d_%d.fa" % (ROOT_A, STALL_N, STALL_LEN)
os.makedirs(os.path.dirname(FA_A), exist_ok=True)
if not os.path.exists(FA_A):
    random.seed(90211)
    with open(FA_A, "w") as f:
        for i in range(STALL_N):
            f.write(">s%d\n%s\n" % (i, "".join(random.choice("ACGU") for _ in range(STALL_LEN))))

SECTIONS = "--section WarpStateStats --section SchedulerStats --section Occupancy"
EXTRA = ",".join([
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
])
# Fallback if a --section name is rejected: name every stall reason explicitly.
STALL_REASONS = ("barrier", "branch_resolving", "dispatch_stall", "drain",
                 "imc_miss", "lg_throttle", "long_scoreboard",
                 "math_pipe_throttle", "membar", "mio_throttle", "misc",
                 "no_instruction", "not_selected", "selected",
                 "short_scoreboard", "sleeping", "tex_throttle", "wait")
FALLBACK = ",".join(
    "smsp__average_warps_issue_stalled_%s_per_issue_active.ratio" % r
    for r in STALL_REASONS) + "," + EXTRA


def _parse(path, want_prefix):
    body = open(path).read() if os.path.exists(path) else ""
    if ("No kernels were profiled" in body) or ("Metric Name" not in body):
        return None, body
    rows = list(_csv_add.DictReader(io.StringIO(
        "\n".join(l for l in body.splitlines() if not l.startswith("==")))))
    names = set(r.get("Kernel Name", "") for r in rows)
    if names and not any(want_prefix in nm for nm in names):
        print("   *** WRONG KERNEL: wanted %r, profiled %r ***"
              % (want_prefix, sorted(names)[:1]))
        return None, body
    agg = collections.defaultdict(list)
    for r in rows:
        try:
            agg[r["Metric Name"]].append(
                float((r["Metric Value"] or "").replace(",", "")))
        except Exception:
            pass
    return {k: sum(v) / len(v) for k, v in agg.items() if v}, body


def profile_stalls(bs):
    """Warp-state + occupancy for int_loop_kernel_<bs>. Returns {} on a broken probe."""
    env = dict(os.environ)
    for k in ("RNA_FML_INT16", "RNA_MD_SMEM", "RNA_PHASE_SYNC",
              "RNA_LAUNCH_STATS", "RNA_LAUNCH_STATS_CSV"):
        env.pop(k, None)
    env["RNA_GPU_CHUNK"] = str(STALL_CAP)
    env["RNA_MIN_GPU_BATCH"] = "1"
    env["RNA_INT_LOOP_BLOCK_SIZE"] = str(bs)

    out = "/content/ncu_stall_bs%d.csv" % bs
    base = ("ncu --target-processes all --csv -k regex:'^int_loop_kernel_%d$' "
            "--launch-skip %d --launch-count %d --log-file %s %s --noPS -i %s "
            "> /dev/null 2>&1" % (bs, STALL_SKIP, STALL_COUNT, out, BIN_A, FA_A))

    for flavour, extra in (("sections", SECTIONS + " --metrics " + EXTRA),
                           ("explicit", "--metrics " + FALLBACK)):
        cmd = base.replace("ncu --target-processes all --csv ",
                           "ncu --target-processes all --csv " + extra + " ")
        subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        res, body = _parse(out, "int_loop_kernel_%d" % bs)
        if res and any("issue_stalled" in k for k in res):
            print("  bs %-4d ok via %s (%d metrics)" % (bs, flavour, len(res)))
            return res
        print("  bs %-4d %s form gave no stall metrics%s"
              % (bs, flavour, "" if body else " (no output at all)"))
    return {}


print("Part 1: warp-state + occupancy for int_loop_kernel at %s"
      % (STALL_BLOCK_SIZES,))
print("  fixture %d x %d, chunk cap %d, launches %d..%d\n"
      % (STALL_N, STALL_LEN, STALL_CAP, STALL_SKIP, STALL_SKIP + STALL_COUNT - 1))
for bs in STALL_BLOCK_SIZES:
    r = profile_stalls(bs)
    if r:
        STALLS[bs] = r
with open("/content/intloop2_stalls.json", "w") as f:
    json.dump({str(k): v for k, v in STALLS.items()}, f, indent=1)

# ----------------------------------------------------------------- report ---
if not STALLS:
    print("\nNo stall data. Nothing below is a result -- check the NCU output in")
    print("/content/ncu_stall_bs*.csv before concluding anything about this kernel.")
else:
    LIMIT = ("blocks", "registers", "shared_mem", "warps")
    print("\n--- occupancy, and WHICH limiter binds "
          "(this is the half section 20.3 never measured) ---")
    print("%6s %10s %12s %10s %8s   %s"
          % ("bs", "achieved", "duration us", "regs/thr", "smem B", "limit: blocks/regs/smem/warps"))
    for bs in sorted(STALLS):
        r = STALLS[bs]
        lims = " / ".join(str(int(r.get("launch__occupancy_limit_" + k, 0))) for k in LIMIT)
        print("%6d %9.1f%% %12.1f %10d %8d   %s"
              % (bs, r.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0),
                 r.get("gpu__time_duration.sum", 0) / 1e3,
                 int(r.get("launch__registers_per_thread", 0)),
                 int(r.get("launch__shared_mem_per_block_allocated", 0)), lims))
    print("  The SMALLEST of the four limits is the binding one. At bs 32 it should")
    print("  be blocks (16 on sm_75) -- which is the whole reason 64 was worth trying.")

    print("\n--- warp issue stalls, as a share of total stall cycles per issue ---")
    keys = sorted({k for r in STALLS.values() for k in r if "issue_stalled" in k})
    tot = {bs: sum(STALLS[bs].get(k, 0.0) for k in keys) for bs in STALLS}
    def short(k):
        m = re.search(r"issue_stalled_(.+?)_per_issue_active", k)
        return m.group(1) if m else k
    rows = sorted(keys, key=lambda k: -max(STALLS[bs].get(k, 0.0) for bs in STALLS))
    hdr = "".join("%14s" % ("bs %d" % bs) for bs in sorted(STALLS))
    print("%-22s%s" % ("stall reason", hdr))
    for k in rows:
        vals = [STALLS[bs].get(k, 0.0) for bs in sorted(STALLS)]
        if max(vals) < 0.005:
            continue
        cells = "".join("%13.1f%%" % (100.0 * v / tot[bs] if tot[bs] else 0)
                        for bs, v in zip(sorted(STALLS), vals))
        print("%-22s%s" % (short(k), cells))
    print("%-22s%s" % ("(total cycles/issue)",
                       "".join("%14.2f" % tot[bs] for bs in sorted(STALLS))))

    print("""
--- how to read it, decided before the numbers arrived -------------------------

DOMINANT REASON                WHAT IT MEANS FOR THIS KERNEL
  long_scoreboard              global-memory latency. Supports the energy-table
                               hypothesis (int22 ~202 KB + int21 ~40 KB thrash a
                               64 KB L1 while staying L2-resident, which is the
                               measured L1 ~75% / L2 ~90%). Levers: shrink or
                               split the tables, __constant__ for the small ones,
                               or a layout that stops int22 evicting the rest.
  wait                         fixed-latency ALU dependency -- the cooperative
                               decode/prefix-sum chain. MORE WARPS WILL NOT FIX
                               IT, which would explain why 64 bought only 10.8%.
                               Lever: shorten the chain, not raise occupancy.
  barrier                      the __syncthreads() between the scan and the
                               search. Lever: a warp-synchronous design.
  short_scoreboard             shared memory -- col_mask[]/prefix[]. Same region
                               as barrier; treat them together.
  not_selected / selected      plenty of eligible warps, the scheduler is simply
                               choosing. Occupancy IS the lever, and 128/256
                               should not have regressed -- if this dominates,
                               re-examine why they did.
  no_instruction               I-cache. Four kernel instantiations plus a heavily
                               unrolled IntLoop_X. Lever: fewer instantiations.

ACROSS BLOCK SIZES
  If long_scoreboard's SHARE falls from 32 to 64 while duration falls, the extra
  warps hid memory latency and that is the mechanism of the 10.8% win.
  If the mix barely moves, the win came from somewhere else and the stall that
  dominates is the thing to attack -- occupancy is already at its ceiling.

  WATCH `barrier` SPECIFICALLY. At BLOCK_SIZE=32 this kernel is one warp per
  block, so its two __syncthreads() are warp-synchronous and cost almost
  nothing. At 64+ they become real barriers across warps that are doing
  DIFFERENT amounts of work (each (H,j) cell's candidate count varies, and the
  scan's `while` search is data-dependent). A barrier share that jumps from
  ~0% at 32 to several % at 64 is the toll on the occupancy win -- and it is
  the most likely reason 128 and 256 regress rather than plateau. If that is
  what the numbers say, the lever is a warp-synchronous redesign of the
  decode/prefix-sum, not a different block size.

  A LOCAL DRY-RUN ALREADY HINTS AT THIS (RTX 3050, 16 x 1200, small grid, so
  directional only): `wait` 36.5% -> 28.9%, `long_scoreboard` 16.8% -> 15.2%,
  `short_scoreboard` 14.3% -> 10.7%, and `barrier` 0.1% -> 9.4% going from 32
  to 64. If the T4 agrees, the dominant stall is the dependent ALU chain and
  NOT memory -- which would refute the energy-table hypothesis above and
  explain why doubling the occupancy ceiling bought only 10.8%.
""")

# =============================================================================
# Part 2 (opt-in) -- name the source LINE.
# =============================================================================
if not WANT_SOURCE_LEVEL:
    print("Part 2 skipped. Set WANT_SOURCE_LEVEL = True to rebuild int_loop.lo")
    print("with -lineinfo and attribute stalls per source line. It restores the")
    print("original binary afterwards, but it does relink -- so run it LAST.")
else:
    SRCDIR = ROOT_A + "/port27/src/ViennaRNA"
    BACKUP = BIN_A + ".prelineinfo"
    print("Part 2: rebuilding int_loop.lo with -lineinfo")
    mk = open(SRCDIR + "/Makefile").read()
    cur = re.search(r"^NVCC_FLAGS = (.*)$", mk, re.M)
    assert cur, "no NVCC_FLAGS in the generated Makefile"
    flags = cur.group(1).strip()
    print("  NVCC_FLAGS was: %s" % flags)
    if "-lineinfo" not in flags:
        flags += " -lineinfo"

    shutil.copy2(BIN_A, BACKUP)
    try:
        subprocess.run("cd %s && rm -f mfe/cuda/int_loop.lo && "
                       "make mfe/cuda/int_loop.lo NVCC_FLAGS='%s' > /content/li_obj.log 2>&1"
                       % (SRCDIR, flags), shell=True, check=True)
        subprocess.run("cd %s && make > /content/li_lib.log 2>&1" % SRCDIR,
                       shell=True, check=True)
        subprocess.run("cd %s/port27/src/bin && touch RNAfold.c && "
                       "make RNAfold > /content/li_bin.log 2>&1" % ROOT_A,
                       shell=True, check=True)
        print("  relinked")

        rep = "/content/int_loop_src"
        env = dict(os.environ, RNA_GPU_CHUNK=str(STALL_CAP), RNA_MIN_GPU_BATCH="1",
                   RNA_INT_LOOP_BLOCK_SIZE=str(STALL_BLOCK_SIZES[0]))
        subprocess.run(
            "ncu --target-processes all --set full --import-source yes "
            "-k regex:'^int_loop_kernel_%d$' --launch-skip %d --launch-count 1 "
            "-f -o %s %s --noPS -i %s > /content/li_ncu.log 2>&1"
            % (STALL_BLOCK_SIZES[0], STALL_SKIP, rep, BIN_A, FA_A),
            shell=True, env=env)

        if not os.path.exists(rep + ".ncu-rep"):
            print("  no report produced; see /content/li_ncu.log")
        else:
            p = subprocess.run(
                "ncu --import %s.ncu-rep --page source --csv > /content/int_loop_src.csv 2>&1"
                % rep, shell=True, capture_output=True, text=True)
            body = open("/content/int_loop_src.csv").read()
            head = body.splitlines()[0] if body else ""
            if "Source" in head:
                rows = list(_csv_add.DictReader(io.StringIO(body)))
                key = next((c for c in rows[0]
                            if "stalled_long_scoreboard" in c or "Stall" in c), None)
                def _f(x):
                    try: return float((x or "0").replace(",", ""))
                    except Exception: return 0.0
                rows.sort(key=lambda r: -_f(r.get(key, 0)))
                print("\n  top source lines by %s" % key)
                for r in rows[:15]:
                    print("    %-52s %8s" % ((r.get("Source File Name", "") or
                                              r.get("Source", ""))[-50:] + ":" +
                                             str(r.get("Source Line", "")),
                                             r.get(key, "")))
            else:
                print("  --page source --csv is not supported by this NCU build.")
                print("  The report is still complete: download %s.ncu-rep and open" % rep)
                print("  it in the Nsight Compute UI, Source page -- that is the same data.")
    finally:
        shutil.copy2(BACKUP, BIN_A)
        print("  original binary restored (%s)" % BACKUP)

try:
    from google.colab import files
    for f in ("/content/intloop2_stalls.json",):
        if os.path.exists(f):
            files.download(f)
except Exception:
    pass

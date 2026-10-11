#!/usr/bin/env python3
# Group gcov function entries by source-level name: template instances, device stubs and
# host wrappers fold together. A stem is COLD only if every entry of it is cold in BOTH
# arms. Cold is necessary, not sufficient, for deletion (PORT_CLEANUP_PLAN.md §1.5).
#
# usage: stems.py gpu.csv [cpu.csv] obj1,obj2,...     (CSVs from summarize.py)
#
# Group by the mangled name's LENGTH PREFIX, never by a regex on the name: a regex
# first-pass called int_loop_warp_kernel, fml_scan_kernel and both flush kernels cold
# when all four are hot. Deletion commits re-check targets against the raw CSV.
import re, sys, collections

def _lp(s):
    m = re.match(r"_Z[L]?(\d+)", s)
    if not m: return None
    n = int(m.group(1)); st = m.end(); return s[st:st + n]

def stem(n):
    s = _lp(n) or n
    if s.startswith("__wrapper__device_stub_"): s = s[len("__wrapper__device_stub_"):]
    if s.startswith("__device_stub__"):
        m = re.match(r"__device_stub__Z(\d+)", s); s = s[m.end():m.end() + int(m.group(1))]
    return s

def load(p):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(p).read().splitlines()[1:]:
        f = l.split(",")
        d[f[0]][stem(f[1])].append(float(f[-1]))
    return d

if __name__ == "__main__":
    if len(sys.argv) == 3: sys.argv.insert(2, "")
    g = load(sys.argv[1]); c = load(sys.argv[2]) if sys.argv[2] else {}
    for o in sys.argv[3].split(","):
        cold = sorted(s for s, v in g[o].items()
                      if max(v) == 0 and max(c.get(o, {}).get(s, [0])) == 0
                      and not s.startswith("__") and "cudaRegister" not in s
                      and "cudaUnregister" not in s)
        hot = [s for s, v in g[o].items() if max(v) > 0]
        print(f"== {o}: {len(g[o])} stems, {len(hot)} hot, {len(cold)} cold")
        print("   " + " ".join(cold))

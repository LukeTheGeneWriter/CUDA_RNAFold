#!/usr/bin/env python3
"""Sample /proc to see which thread burns the core, and whether it is RUNNING.

nsys on this box collects data but Ubuntu's nsight-systems package ships without
the importer, so the .qdstrm cannot be read. This answers the question that
actually matters with no tooling at all:

  state R  = runnable/running  -> the thread is EXECUTING on a core (spinning or
                                  computing); a CPU folder would contend with it
  state S  = interruptible sleep -> parked; the core is genuinely free
  state D  = uninterruptible sleep (usually I/O or a driver ioctl)

Per-thread utime/stime says how much CPU each one actually consumed, so the
thread that owns the busy core is identified rather than guessed.

usage: thread_states.py <command...>
"""
import os
import subprocess
import sys
import time
from collections import defaultdict

HZ = os.sysconf("SC_CLK_TCK")


def snapshot(pid):
    """(tid -> (name, state, utime, stime)) or None once the process is gone."""
    out = {}
    try:
        tids = os.listdir("/proc/%d/task" % pid)
    except OSError:
        return None
    for t in tids:
        try:
            with open("/proc/%d/task/%s/stat" % (pid, t)) as f:
                s = f.read()
        except OSError:
            continue
        # comm is parenthesised and may contain spaces: split on the last ')'
        rp = s.rfind(")")
        name = s[s.find("(") + 1:rp]
        rest = s[rp + 2:].split()
        out[int(t)] = (name, rest[0], int(rest[11]), int(rest[12]))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    p = subprocess.Popen(sys.argv[1:], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    states = defaultdict(lambda: defaultdict(int))
    names = {}
    last = {}
    n = 0
    t0 = time.time()

    while p.poll() is None:
        snap = snapshot(p.pid)
        if snap is None:
            break
        for tid, (name, st, ut, stime) in snap.items():
            names[tid] = name
            states[tid][st] += 1
            last[tid] = ut + stime
        n += 1
        time.sleep(0.004)

    wall = time.time() - t0
    p.wait()

    print("  wall %.2f s, %d samples" % (wall, n))
    print()
    print("  %-8s %-16s %8s %8s   %s" % ("tid", "name", "cpu_s", "%of wall", "states"))
    for tid in sorted(last, key=lambda k: -last[k]):
        cpu = last[tid] / float(HZ)
        if cpu < 0.01 and sum(states[tid].values()) < n * 0.02:
            continue
        tot = sum(states[tid].values())
        mix = "  ".join("%s=%.0f%%" % (k, 100.0 * v / tot)
                        for k, v in sorted(states[tid].items(), key=lambda x: -x[1]))
        print("  %-8d %-16s %8.2f %7.0f%%   %s" % (tid, names[tid], cpu,
                                                   100.0 * cpu / wall, mix))
    print()
    print("  R = executing on a core.  S = parked, core is free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

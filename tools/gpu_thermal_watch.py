#!/usr/bin/env python3
"""Sample the GPU while a command runs, and say whether the clock actually fell.

    python3 tools/gpu_thermal_watch.py <command...>

Wrap any timed run in this before believing a local A/B. It reports the clock
range, the temperature range, the DRIFT between the first and last third of the
run -- which is what biases an A/B -- and WHICH throttle reason fired.

Written 2026-09-10 after the standing note "this laptop cannot hold a GPU clock
under sustained load (1057 -> 712 MHz)" turned out to be a cooling problem
rather than a property of the machine. On a cool flat surface with airflow the
same run drifts -3.9%, and the dominant limiter is the 65 W POWER CAP
(SwPowerCap 66.8% of samples) not heat (SwThermal 9.3%, max 74 C). Cooling fixes
the drift; nothing fixes the power cap, so the card will not reach its 2100 MHz
max no matter what.

Rule of thumb it earns: local A/B is usable for effects above ~5% if the run is
ABBA and the drift is reported next to the result. Below that, or when many
cores matter, use the T4.

The laptop's clock behaviour has been quoted as "1057 -> 712 MHz under sustained
load" and used to dismiss every local measurement below ~15%. That was one
observation. This turns it into a time series, names WHICH throttle reason fired,
and reports the drift between the first and last third of the run -- which is the
thing that biases an A/B.
"""
import re, subprocess, sys, threading, time

# nvidia-smi clocks_event_reasons bitmask
BITS = [(0x0001, "GpuIdle"), (0x0002, "AppClocksSetting"), (0x0004, "SwPowerCap"),
        (0x0008, "HwSlowdown"), (0x0010, "SyncBoost"), (0x0020, "SwThermal"),
        (0x0040, "HwThermal"), (0x0080, "HwPowerBrake"), (0x0100, "DisplayClock")]

Q = ("--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
     "clocks_throttle_reasons.active", "--format=csv,noheader,nounits")

samples = []
stop = threading.Event()


def sampler(period=0.25):
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi"] + list(Q), capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            f = [x.strip() for x in out.split(",")]
            samples.append((time.time(), float(f[0]), float(f[1]), float(f[2]),
                            float(f[3]) if f[3] not in ("[N/A]", "N/A") else float("nan"),
                            int(f[4], 16)))
        except Exception:
            pass
        stop.wait(period)


def summarise():
    if not samples:
        print("no samples"); return
    t0 = samples[0][0]
    sm = [s[1] for s in samples]
    mx = samples[0][2]
    tp = [s[3] for s in samples]
    dur = samples[-1][0] - t0

    print("  duration %.1f s, %d samples, max SM clock %.0f MHz" % (dur, len(samples), mx))
    print("  SM clock   min %.0f  median %.0f  max %.0f MHz  (%.0f%%-%.0f%% of max)"
          % (min(sm), sorted(sm)[len(sm)//2], max(sm),
             100*min(sm)/mx, 100*max(sm)/mx))
    print("  temp       min %.0f  max %.0f C" % (min(tp), max(tp)))

    # Drift: first third vs last third. This is what biases an A/B run.
    n = len(sm)//3
    if n >= 2:
        a = sum(sm[:n])/n
        b = sum(sm[-n:])/n
        print("  DRIFT      first third %.0f MHz -> last third %.0f MHz  (%+.1f%%)"
              % (a, b, 100*(b-a)/a))

    # Throttle reasons, excluding GpuIdle which is not throttling.
    tot = len(samples)
    any_real = False
    for bit, name in BITS:
        c = sum(1 for s in samples if s[5] & bit)
        if c and name != "GpuIdle":
            print("  THROTTLE   %-16s %5.1f%% of samples" % (name, 100.0*c/tot))
            any_real = True
    if not any_real:
        print("  THROTTLE   none active (other than GpuIdle)")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: thermal.py <command...>")
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time.sleep(0.5)
    t0 = time.time()
    rc = subprocess.run(sys.argv[1:]).returncode
    wall = time.time() - t0
    time.sleep(0.5)
    stop.set(); th.join(timeout=3)
    print("\n=== GPU during that run (wall %.1f s, rc %d) ===" % (wall, rc))
    summarise()


main()

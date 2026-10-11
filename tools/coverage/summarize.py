#!/usr/bin/env python3
# gcov function summaries for the objects the port added or changed.
# usage: summarize.py out.csv [tree]   (tree default ~/covq)
# Writes CSV object,function,lines,pct -- one row per gcov function entry, i.e. per
# template instance / device stub / host wrapper. Group them with stems.py.
import os, re, subprocess, sys, glob
B = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/covq") + "/src"
gcnos = set(g[:-5] for g in glob.glob(B + "/ViennaRNA/mfe/cuda/.libs/*.gcno") +
            glob.glob(B + "/ViennaRNA/mfe/.libs/*.gcno") +
            glob.glob(B + "/ViennaRNA/grammar/.libs/*.gcno") +
            glob.glob(B + "/ViennaRNA/params/.libs/*params*.gcno") +
            glob.glob(B + "/bin/*RNAfold*.gcno"))
out = open(sys.argv[1], "w"); out.write("object,function,lines,pct\n")
for base in sorted(gcnos):
    d = os.path.dirname(base)
    p = subprocess.run(["gcov-13", "-f", "-n", "-o", d, base + ".gcno"],
                       capture_output=True, text=True, cwd=d)
    fn = None
    for l in p.stdout.splitlines():
        m = re.match(r"Function '(.+)'", l)
        if m: fn = m.group(1); continue
        m = re.match(r"Lines executed:([\d.]+)% of (\d+)", l)
        if m and fn:
            out.write("%s,%s,%s,%s\n" % (os.path.basename(base), fn.replace(",", ";"),
                                         m.group(2), m.group(1)))
            fn = None
        elif l.startswith("File "): fn = None
out.close(); print("wrote", sys.argv[1])

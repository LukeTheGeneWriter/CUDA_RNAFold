import re, sys, subprocess, os, time
# Runs RNAfold on a fasta, then reconstructs each GPU chunk from the input order
# (records enter chunks consecutively; with RNA_CPU_THREADS=0 none are diverted)
# and reports the max/mean length ratio that decides what streaming is worth.
fa  = sys.argv[1]
bin = os.path.expanduser("~/rnafold_build/src/bin/RNAfold")
lens, cur = [], 0
for line in open(fa):
    if line.startswith(">"):
        if cur: lens.append(cur)
        cur = 0
    else: cur += len(line.strip())
if cur: lens.append(cur)

env = dict(os.environ); env["RNA_CPU_THREADS"] = "0"
for k in sys.argv[3:]:
    a, b = k.split("="); env[a] = b
t0 = time.time()
p = subprocess.run([bin, "--noPS", "-i", fa], capture_output=True, text=True, env=env)
wall = time.time() - t0

chunks = [(int(m.group(1)), int(m.group(2)))
          for m in re.finditer(r"init_gpu2\((\d+),VC,\d+,(\d+),", p.stderr)]
print("== %s ==  wall=%.1fs rc=%d  records=%d  chunks=%d" % (sys.argv[2], wall, p.returncode, len(lens), len(chunks)))
pos, rows_now, rows_stream, tot = 0, 0, 0, 0
turn = 3
for ci, (nf, mx) in enumerate(chunks):
    seg = lens[pos:pos+nf]; pos += nf
    if not seg: break
    mean = sum(seg)/len(seg)
    r = max(seg)/mean
    rows_now += max(seg) - turn - 1
    tot += sum(L - turn - 1 for L in seg)
    if ci < 6 or len(chunks) <= 8:
        print("   chunk %-3d n=%-5d max=%-5d mean=%-7.1f max/mean=%.2f" % (ci, nf, max(seg), mean, r))
if len(chunks) > 8: print("   ... %d chunks total" % len(chunks))
W = (sum(c[0] for c in chunks)/len(chunks)) if chunks else 1
rows_stream = tot / W if W else 0
print("   ROWS today=%d   streaming(ideal)=%.0f   ratio=%.2fx   [avg width %.0f]"
      % (rows_now, rows_stream, rows_now/rows_stream if rows_stream else 0, W))
for line in p.stderr.splitlines():
    if "phase timing" in line: print("   " + line.split("phase timing (s):")[1].strip())

"""Dry-run what can be dry-run: every code cell compiles, and every regex and
literal the notebook asserts on is checked against REAL stderr from the local
binary. Colab-only bits (google.colab, ncu, A100) cannot run here; everything
else can, and the last three notebook failures were all startup, not science.

THE REGEXES ARE TAKEN FROM THE NOTEBOOK, not restated here. The first version of
this checker hard-coded a copy, which went stale the moment the kernel's banner
gained a field -- a checker that duplicates what it checks is one more thing
that can silently disagree."""
import json
import re
import subprocess
import os

NB = '/mnt/c/Users/lukef/CUDA_RNAFold/CUDA_RNAFold_Lookup.ipynb'
BIN = os.path.expanduser('~/port27fml/src/bin/RNAfold')
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']
code = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
print('cells: %d (%d code)' % (len(cells), len(code)))

bad = 0
for i, src in enumerate(code):
    try:
        compile(src, '<cell %d>' % i, 'exec')
    except SyntaxError as e:
        bad += 1
        print('  SYNTAX ERROR cell %d line %s: %s' % (i, e.lineno, e.msg))
print('syntax: %s' % ('all cells compile' if not bad else '%d BROKEN' % bad))

# ---- lift the notebook's own regexes -------------------------------------
runner = [c for c in code if 'PHASE_RE = re.compile' in c]
assert len(runner) == 1, 'expected exactly one runner cell, found %d' % len(runner)
body = runner[0]
start = body.index('PHASE_RE = re.compile')
end = body.index('PHASES = (')
ns = {'re': re}
exec(body[start:end], ns)
names = [k for k in ns if k.endswith('_RE')]
print('lifted from the notebook:', sorted(names))

# ---- real stderr, four ways ---------------------------------------------
def stderr(env_extra, fa):
    env = dict(os.environ, RNA_GPU_CHUNK='0', RNA_MIN_GPU_BATCH='1',
               RNA_PHASE_SYNC='1')
    env.update(env_extra)
    p = subprocess.run([BIN, '--noPS', '-i', fa],
                       capture_output=True, text=True, env=env)
    return p.stderr

UNI = os.path.expanduser('~/h6/uniform_small.fa')
RAG = os.path.expanduser('~/ngubar/ngu.fa')

e_def = stderr({}, UNI)
e_g = stderr({'RNA_INT_LOOP_GRIDY': '1'}, UNI)
e_gr = stderr({'RNA_INT_LOOP_GRIDY': '1'}, RAG)
e_w = stderr({'RNA_INT_LOOP_WSEARCH': '1'}, UNI)
e_b = stderr({'RNA_INT_LOOP_GRIDY': '1', 'RNA_INT_LOOP_WSEARCH': '1'}, UNI)

# ---- the literals the defaults cell asserts -----------------------------
print('\n--- the defaults cell ---')
for label, lit, want_absent in (
        ('warp is the default', 'int_loop kernel: warp-per-cell (the measured default)', False),
        ('block size is 32', 'int_loop_kernel block size 32', False),
        ('build-threads banner', 'build threads ', False),
        ('H6 off by default', 'RNA_INT_LOOP_GRIDY=1', True),
        ('H7 off by default', 'RNA_INT_LOOP_WSEARCH=1', True)):
    present = lit in e_def
    ok = (not present) if want_absent else present
    print('  %-22s %s' % (label, 'OK' if ok else '*** %r %s' %
                          (lit, 'present but should not be' if present else 'ABSENT')))

# ---- every regex the runner depends on ---------------------------------
print('\n--- the runner regexes, against real stderr ---')
for name in ('PHASE_RE', 'STAGE_RE', 'SWEEP_RE', 'BS_RE', 'BT_RE'):
    print('  %-10s %s' % (name, 'matches' if ns[name].search(e_def) else '*** NO MATCH'))

for label, hay, expect in (('uniform (should accept)', e_g, '2-D'),
                           ('ragged  (should decline)', e_gr, 'flat'),
                           ('both knobs, uniform', e_b, '2-D')):
    hits = ns['GRID_RE'].findall(hay)
    print('  GRID_RE   %-24s %d hit(s) %s  %s'
          % (label, len(hits), [(h[0][:4], h[1]) for h in hits[:2]],
             'OK' if hits and hits[0][0].startswith(expect) else '*** expected ' + expect))

print('\n--- the knob-engaged assertion the runner makes ---')
for label, hay, gridy, wsearch in (('gridy only', e_g, True, False),
                                   ('wsearch only', e_w, False, True),
                                   ('BOTH on a uniform fixture', e_b, True, True)):
    ok = True
    for want, token in ((gridy, 'RNA_INT_LOOP_GRIDY=1'), (wsearch, 'RNA_INT_LOOP_WSEARCH=1')):
        if bool(want) != (token in hay):
            ok = False
    print('  %-26s %s' % (label, 'OK' if ok else
                          '*** a knob that was asked for did not announce itself'))

print('\n--- not checkable locally (Colab-only) ---')
for s in ('google.colab files.download', 'ncu availability and -k matching',
          'nvcc -Xptxas on the Colab toolkit', 'A100 clocks/power/throttle'):
    print('  -', s)

# PIPE_RE, against a real pipelined run -- the pipeline reports what it
# ACHIEVED, and asserting that beats asserting the knob (the H6 lesson).
e_p = stderr({'RNA_BUILD_PIPELINE': '1'}, UNI)
m = ns['PIPE_RE'].search(e_p) if 'PIPE_RE' in ns else None
print('\n--- the build-pipeline report ---')
print('  PIPE_RE   %s' % ('matches: %s chunks, %s%% of builder hidden'
                          % (m.group(1), m.group(4)) if m else '*** NO MATCH'))
# NOT e_def any more: the default is AUTO since STRESS272 32.4, and on a host
# with memory to spare the default run IS pipelined. "Off" now has to be asked
# for, and this check silently inverted its own meaning the moment it did not.
e_noP = stderr({'RNA_BUILD_PIPELINE': '0'}, UNI)
print('  absent when off: %s' % ('OK' if not ns['PIPE_RE'].search(e_noP) else '*** present'))

# AUTO_RE, against the shipped default -- RNA_BUILD_PIPELINE unset means
# "decide from MemAvailable" since STRESS272 32.4, and the verdict line is the
# only evidence of which way it went. A regex that does not match the C format
# string turns every AUTO arm into a SystemExit an hour into a run.
e_a = stderr({}, UNI)
a = ns['AUTO_RE'].search(e_a) if 'AUTO_RE' in ns else None
print('  AUTO_RE   %s'
      % ('matches: %s, needs %s GB of %s GB' % (a.group(1), a.group(2), a.group(3))
         if a else '*** NO MATCH -- every AUTO arm would abort'))
if a:
    print('  verdict agrees with the report: %s'
          % ('OK' if (a.group(1) == 'ON') == bool(ns['PIPE_RE'].search(e_a))
             else '*** AUTO said %s but the overlap report disagrees' % a.group(1)))

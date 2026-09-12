#!/usr/bin/env python3
"""Disassemble one kernel from int_loop.o and characterise its inner loop.

H1 asks: does nvcc already hoist the cell-invariant loads out of the work loop?
The way to tell is to find the loop (a backward branch), then count the memory
instructions INSIDE it.
"""
import re
import subprocess
import sys
import collections

OBJ = '/home/lukefpwd/port27fml/src/ViennaRNA/mfe/cuda/.libs/int_loop.o'
WANT = sys.argv[1] if len(sys.argv) > 1 else 'int_loop_warp_kernel'
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'sm_86'

txt = subprocess.run(['cuobjdump', '-sass', '-arch', ARCH, OBJ],
                     capture_output=True, text=True).stdout

# split into per-function blocks
blocks, cur, name = {}, [], None
for line in txt.splitlines():
    m = re.match(r'\s*Function : (.+)$', line)
    if m:
        if name:
            blocks[name] = cur
        name, cur = m.group(1).strip(), []
        continue
    if name is not None:
        cur.append(line)
if name:
    blocks[name] = cur

hits = [k for k in blocks if WANT in k]
if not hits:
    print('no function matching %r in %s' % (WANT, ARCH))
    print('available:', [k[:70] for k in list(blocks)[:12]])
    raise SystemExit(1)

for fn in sorted(hits):
    body = blocks[fn]
    # instruction lines look like:  /*0a30*/   LDG.E R4, [R2.64] ;
    insts = []
    for line in body:
        m = re.match(r'\s*/\*([0-9a-f]{4})\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9._]*)', line)
        if m:
            insts.append((int(m.group(1), 16), m.group(2), line.strip()))
    if not insts:
        continue

    # backward branches mark loop bodies: BRA to an address <= its own
    back = []
    for addr, op, raw in insts:
        if op.startswith('BRA') or op.startswith('BRX'):
            t = re.search(r'0x([0-9a-f]+)', raw)
            if t and int(t.group(1), 16) < addr:
                back.append((int(t.group(1), 16), addr))
    # the largest backward branch span that is not the whole kernel = the work loop
    back.sort(key=lambda p: p[1] - p[0])

    print('=' * 74)
    print('%s   [%s]' % (fn[:70], ARCH))
    print('=' * 74)
    print('  total instructions      %d' % len(insts))
    tally = collections.Counter(op.split('.')[0] for _, op, _ in insts)
    for op in ('LDG', 'LDS', 'LDL', 'STG', 'STS', 'SHFL', 'BAR', 'IMAD', 'LOP3'):
        if tally.get(op):
            print('  %-22s %d' % (op, tally[op]))
    print('  backward branches       %d' % len(back))
    for lo, hi in back:
        span = [i for i in insts if lo <= i[0] <= hi]
        t = collections.Counter(op.split('.')[0] for _, op, _ in span)
        print('    loop [0x%04x,0x%04x]  %4d inst  LDG %-3d LDS %-3d SHFL %-3d BAR %-3d'
              % (lo, hi, len(span), t.get('LDG', 0), t.get('LDS', 0),
                 t.get('SHFL', 0), t.get('BAR', 0)))

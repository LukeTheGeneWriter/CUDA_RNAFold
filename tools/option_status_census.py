#!/usr/bin/env python3
"""Count the option surface from the SOURCE OF TRUTH, not by hand.

PORT_OPTION_STATUS.md carries a summary table -- how many options are
accelerated, declined, neutral, unreachable. It was maintained by hand, and
after three features moved between categories in one day the numbers no longer
added up: the table said 31/9/19/1 while the rows said something else. A count
that can drift is a count nobody can cite.

So this reads `src/bin/RNAfold.ggo` for the real list of options, finds each one
in the doc's table, and prints the census. **It exits non-zero if any option is
not classified at all**, which is the part that makes it a bar rather than a
report: a new option added upstream shows up here as a hole instead of being
silently absent.

Two options are deliberately special-cased, because they are genuinely split
rather than ambiguous:

  --dangles   d0 and d2 are ACCELERATED, d1 and d3 are DECLINED. One ggo
              option, two verdicts, and collapsing that to one would be a lie
              in whichever direction it went.
  --bm-*      the benchmark family is covered by a single grouped row.

usage: python3 tools/option_status_census.py [--quiet]
"""
import re
import sys

GGO = 'src/bin/RNAfold.ggo'
DOC = 'PORT_OPTION_STATUS.md'
CLASSES = ('ACCEL', 'DECLINED', 'NEUTRAL', 'UNREACHABLE')


def classify(cell):
    up = cell.upper()
    for k in CLASSES:
        if k in up:
            return k
    return None


def main():
    quiet = '--quiet' in sys.argv

    ggo = open(GGO, encoding='utf-8', errors='replace').read()
    opts = re.findall(r'^option\s+"([^"]+)"\s+(\S)', ggo, re.M)

    rows = []
    for line in open(DOC, encoding='utf-8'):
        if not line.startswith('| '):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) >= 2:
            k = classify(cells[1])
            if k:
                rows.append((cells[0], k))

    verdict = {}
    for name, short in opts:
        if name == 'dangles':
            verdict[name] = 'SPLIT'          # see the module docstring
            continue
        hit = None
        for label, k in rows:
            if ('`--%s`' % name) in label or ('`--%s=' % name) in label:
                hit = k
                break
            if short != '-' and ('`-%s`' % short) in label:
                hit = k
                break
            if name.startswith('bm-') and '`--bm-*`' in label:
                hit = k
                break
        verdict[name] = hit

    counts = {k: 0 for k in CLASSES}
    counts['SPLIT'] = 0
    missing = []
    for name, k in verdict.items():
        if k is None:
            missing.append(name)
        else:
            counts[k] += 1

    if not quiet:
        print('%s: %d options' % (GGO, len(opts)))
        for k in CLASSES + ('SPLIT',):
            print('  %-12s %3d' % (k, counts[k]))
        print('  %-12s %3d' % ('unclassified', len(missing)))
        print()
        print('  --dangles is SPLIT: d0/d2 accelerated, d1/d3 declined.')
        print()
        for name in sorted(n for n, k in verdict.items() if k == 'DECLINED'):
            print('  declined: --%s' % name)

    if missing:
        print()
        print('*** %d option(s) classified NOWHERE in %s:' % (len(missing), DOC))
        for m in missing:
            print('      --%s' % m)
        print('    Every option must have a verdict. Add a row.')
        return 1

    total = sum(counts.values())
    if total != len(opts):
        print('*** counted %d of %d -- an option matched two rows' % (total, len(opts)))
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Type sequences at RNAfold through a pseudo-terminal, as a user at the prompt does.

PORT_SHORT_SEQUENCES.md S1. W. B. Langdon (2026-10-18): with interactive input the
accelerated build printed the next prompt and held the answer until Ctrl-D, where
stock ViennaRNA 2.7.2 answers each sequence as it is typed. A pipe cannot show this
-- isatty() is false there, so this drives a pty, on stdin AND stdout as a terminal
has them.

    tty_answer.py RNAfold SEQ [SEQ ...]   (environment passed through)

For each sequence, prints one line:
    answered-before-eof <yes|no> <seconds> <energy line or ->
then, after Ctrl-D, everything the process wrote, prefixed "| ".
Exit status 0 only if every sequence was answered before end of input.
"""
import os
import pty
import re
import select
import sys
import time

WAIT = 30.0   # a long record on a cold device still answers well inside this


def read_until(fd, pat, secs):
    out, end = b"", time.time() + secs
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                d = os.read(fd, 65536)
            except OSError:
                break
            if not d:
                break
            out += d
            if pat is not None and re.search(pat, out):
                break
    return out


def main():
    exe, seqs = sys.argv[1], sys.argv[2:]
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(exe, [exe, "--noPS"])
    log = read_until(fd, rb"@ to quit", 10.0)
    ok = True
    for s in seqs:
        t0 = time.time()
        os.write(fd, s.encode() + b"\n")
        got = read_until(fd, rb"minimum free energy = +-?[0-9.]+ kcal/mol", WAIT)
        dt = time.time() - t0
        m = re.search(rb"minimum free energy = +-?[0-9.]+ kcal/mol", got)
        print("answered-before-eof %s %.3f %s" % ("yes" if m else "no", dt,
              m.group(0).decode() if m else "-"))
        ok = ok and bool(m)
        log += got
        log += read_until(fd, rb"@ to quit", 5.0)   # the next prompt
    os.write(fd, b"\x04")
    log += read_until(fd, None, 10.0)
    os.waitpid(pid, 0)
    clean = re.sub(rb"\x1b\[[0-9;]*m", b"", log).decode(errors="replace")
    for line in clean.splitlines():
        print("| " + line)
    sys.exit(0 if ok else 1)


main()

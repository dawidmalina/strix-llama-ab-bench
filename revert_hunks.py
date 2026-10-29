#!/usr/bin/env python3
"""Reverse selected hunks of the 013044e12 ggml-cuda.cu diff in the current worktree.
Usage: revert_hunks.py "36-48,5" (1-based hunk indices, comma/range list)."""
import subprocess, sys, re

spec = sys.argv[1]
want = set()
for part in spec.split(","):
    if "-" in part:
        a, b = part.split("-")
        want.update(range(int(a), int(b) + 1))
    else:
        want.add(int(part))

F = "ggml/src/ggml-cuda/ggml-cuda.cu"
G = ["git", "-c", "safe.directory=*"]
diff = subprocess.run(G + ["diff", "ff64766de", "013044e12", "--", F], capture_output=True, text=True, check=True).stdout
lines = diff.split("\n")
first = next(i for i, l in enumerate(lines) if l.startswith("@@"))
header = lines[:first]
hunks, cur = [], None
for l in lines[first:]:
    if l.startswith("@@"):
        cur = [l]
        hunks.append(cur)
    elif cur is not None:
        cur.append(l)
print(f"{len(hunks)} hunks, reversing {sorted(want)}")
sel = list(header)
for i, h in enumerate(hunks, 1):
    if i in want:
        sel += h
open("/tmp/sel.patch", "w").write("\n".join(sel).rstrip("\n") + "\n")
r = subprocess.run(["patch", "-R", "-p1", "--no-backup-if-mismatch", "-i", "/tmp/sel.patch"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-400:])
sys.exit(r.returncode)

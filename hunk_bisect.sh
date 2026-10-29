#!/bin/bash
# Hunk-revert bisect on 013044e12: args are hunk specs; each: rebuild + 1 server round. Writes /tmp/hunk_map.txt.
cd /tmp || exit 1
export TWIN_MAX=3
: > /tmp/hunk_map.txt
for spec in "$@"; do
  PATCH_CMD="python3 /tmp/revert_hunks.py '$spec'" bash /tmp/bisect_step.sh 013044e12
  echo "$spec $(ls -dt /tmp/srvbisect_013044e12_* | head -1)" >> /tmp/hunk_map.txt
done
touch /tmp/hunk.DONE

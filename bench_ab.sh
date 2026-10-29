#!/bin/bash
# A/B/A/B llama-bench: baseline (rocm) vs candidate (beta), same flags as llama-server.
umask 022
OUT=/tmp/regression_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
exec > "$OUT/run.log" 2>&1

S=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6
M=$S/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
BASE=/var/lib/lemonade/lab/rocm-strix-llama.cpp
NEW=/var/lib/lemonade/lab/beta-strix-llama.cpp

# Abort if a build's binary reports another commit than its git HEAD or loads libs from another dir.
verify() {
  local d=$1 head ver lib
  head=$(git -c safe.directory=$d -C $d rev-parse --short=9 HEAD)
  ver=$(LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench --version 2>&1 | grep -o 'commit [0-9a-f]*' | awk '{print $2}')
  lib=$(LD_LIBRARY_PATH=$d/build/bin ldd $d/build/bin/llama-bench | awk '/libllama.so/{print $3}')
  echo "verify $(basename $d): head=$head version=$ver lib=$lib"
  case "$head" in "$ver"*) ;; *) echo "ABORT: commit mismatch"; exit 1;; esac
  case "$lib" in "$d"/*) ;; *) echo "ABORT: libs from another dir"; exit 1;; esac
}
verify $BASE; verify $NEW

for d in "$BASE" "$NEW"; do
  echo "$(basename $d): $(git -c safe.directory=$d -C $d log -1 --oneline)"
done > "$OUT/commits.txt"
cat "$OUT/commits.txt"
rocm-smi --showdriverversion 2>&1 | head -5 > "$OUT/env.txt"

for round in $(seq 1 ${BENCH_ROUNDS:-2}); do
  for tag in base new; do
    D=$BASE; [ "$tag" = new ] && D=$NEW
    echo "=== round $round $tag $(date +%T)"
    LD_LIBRARY_PATH=$D/build/bin $D/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
      -p 512,4096,16384 -n 128 -d 0,32768 -r 3 -o json \
      > "$OUT/${tag}_r${round}.json" 2> "$OUT/${tag}_r${round}.err"
    echo "exit=$?"
  done
done
echo "DONE $(date +%T)"
touch "$OUT/DONE"

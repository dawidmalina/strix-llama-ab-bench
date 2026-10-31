#!/bin/bash
# Host-side (as lemonade): llama-bench pp1-4 at depth 40000/100000, baseline vs worktree (V1 default), interleaved. Output: /tmp/deep_ab
set -u
LAB=/var/lib/lemonade/lab
O=/tmp/deep_ab
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
mkdir -p $O
exec > $O/run.log 2>&1
cd /tmp || exit 1
for dep in 40000 100000; do
  for t in base v1; do
    d=$LAB/rocm-strix-llama.cpp; [ $t = v1 ] && d=$LAB/bisect-strix-llama.cpp
    echo "=== ${t}_d$dep $(date +%T)"
    LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
      -p 1,2,3,4 -n 0 -d $dep -r 3 -o json > $O/${t}_d$dep.json 2>$O/${t}_d$dep.err || echo "FAILED ${t}_d$dep"
  done
done
echo "DONE $(date +%T)"
touch $O/DONE

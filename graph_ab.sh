#!/bin/bash
# Host-side (as lemonade): llama-bench pp2-4 at depth 0/4096, HIP graphs on/off, baseline vs worktree (V1). Output: /tmp/graph_ab
set -u
LAB=/var/lib/lemonade/lab
O=/tmp/graph_ab
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
mkdir -p $O
exec > $O/run.log 2>&1
cd /tmp || exit 1
for i in 1 2; do
  for t in base v1; do
    d=$LAB/rocm-strix-llama.cpp; [ $t = v1 ] && d=$LAB/bisect-strix-llama.cpp
    for g in 0 1; do
      echo "=== ${t}_g${g}_$i $(date +%T)"
      env GGML_CUDA_DISABLE_GRAPHS=$g LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 \
        -lzm on-direct -p 1,2,3,4 -n 0 -d 0,4096 -r 5 -o json > $O/${t}_g${g}_$i.json 2>/dev/null || echo "FAILED ${t}_g${g}_$i"
    done
  done
done
echo "DONE $(date +%T)"
touch $O/DONE

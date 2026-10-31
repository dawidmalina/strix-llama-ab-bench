#!/bin/bash
# Host-side (as lemonade): llama-bench small-batch pp (MTP/ngram verification sizes), baseline vs worktree, alternated.
set -u
LAB=/var/lib/lemonade/lab
B=$LAB/rocm-strix-llama.cpp
W=$LAB/bisect-strix-llama.cpp
O=/tmp/kprof
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
exec > $O/small.log 2>&1
for i in 1 2; do
  for t in base new; do
    d=$B; [ $t = new ] && d=$W
    echo "=== small $t $i $(date +%T)"
    LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
      -p 1,2,3,4,5,6,8,9,12,16,32 -n 0 -d 0,4096 -r 5 -o json > $O/small_${t}_$i.json 2>$O/small_${t}_$i.err || echo "FAILED $t"
  done
done
echo "DONE $(date +%T)"
touch $O/SMALL_DONE

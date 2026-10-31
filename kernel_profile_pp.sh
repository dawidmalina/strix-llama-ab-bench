#!/bin/bash
# Host-side (as lemonade): rocprofv3 kernel stats of llama-bench pp3 (MTP verify size), baseline vs worktree.
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
O=/tmp/kprof
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
exec > $O/pp3prof.log 2>&1
cd /tmp || exit 1
for t in base new; do
  d=$LAB/rocm-strix-llama.cpp; [ $t = new ] && d=$LAB/bisect-strix-llama.cpp
  for n in 1 3; do
    echo "=== $t pp$n $(date +%T)"
    rm -rf $O/pp${n}_$t
    LD_LIBRARY_PATH=$d/build/bin rocprofv3 --kernel-trace --stats --output-format csv -d $O/pp${n}_$t -o run -- \
      $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct -p $n -n 0 -r 50 -o json > $O/pp${n}_$t.json || echo "FAILED $t pp$n"
  done
done
echo "DONE $(date +%T)"
touch $O/PP3_DONE

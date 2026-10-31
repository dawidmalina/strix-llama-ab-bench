#!/bin/bash
# Host-side (as lemonade): rocprofv3 kernel stats of llama-bench pp2/pp3 at depth 0/4096, baseline vs worktree (V1 default). Output: /tmp/fa_prof
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
O=/tmp/fa_prof
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
mkdir -p $O
exec > $O/run.log 2>&1
cd /tmp || exit 1
for t in base v1; do
  d=$LAB/rocm-strix-llama.cpp; [ $t = v1 ] && d=$LAB/bisect-strix-llama.cpp
  for cfg in "3 0" "3 4096" "2 4096"; do
    set -- $cfg; n=$1; dep=$2; tag=${t}_p${n}_d${dep}
    echo "=== $tag $(date +%T)"
    rm -rf $O/$tag
    LD_LIBRARY_PATH=$d/build/bin rocprofv3 --kernel-trace --stats --output-format csv -d $O/$tag -o run -- \
      $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct -p $n -n 0 -d $dep -r 20 -o json > $O/$tag.json || echo "FAILED $tag"
  done
done
echo "DONE $(date +%T)"
touch $O/DONE

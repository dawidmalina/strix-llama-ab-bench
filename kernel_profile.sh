#!/bin/bash
# Host-side (as lemonade): llama-bench tg (no MTP) + rocprofv3 kernel stats of the server code test, baseline vs worktree.
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
B=$LAB/rocm-strix-llama.cpp
W=$LAB/bisect-strix-llama.cpp
O=/tmp/kprof
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
mkdir -p $O
exec > $O/run.log 2>&1
echo "start $(date +%T) wt=$(git -c safe.directory=* -C $W log --oneline -1)"
cmake --build $W/build --config Release --target llama-bench -j"$(nproc)" >$O/build_bench.log 2>&1 || { echo BENCH_BUILD_FAILED; exit 1; }
for d in $B $W; do
  LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench --version 2>&1 | grep -o 'commit [0-9a-f]*'
  LD_LIBRARY_PATH=$d/build/bin ldd $d/build/bin/llama-bench | awk '/libllama.so/{print $3}'
done
export LLAMA_MTP_RESIDENT=0
for i in 1 2; do
  for t in base new; do
    d=$B; [ $t = new ] && d=$W
    echo "=== bench $t $i $(date +%T)"
    LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
      -p 0 -n 128 -d 0,4096 -r 3 -o json > $O/bench_${t}_$i.json 2>$O/bench_${t}_$i.err || echo "BENCH_FAILED $t"
  done
done
cd /tmp
NEW_ROOT=$W TAGS=base,new ROUNDS=1 ONLY=code \
  WRAP="rocprofv3 --kernel-trace --stats --output-format csv -d $O/prof_{tag} -o run --" \
  python3 /tmp/server_ab.py $O/srv
echo "DONE $(date +%T)"
touch $O/DONE

#!/bin/bash
# Host-side (as lemonade): d3f9fb467 + dq_gemv_patch.py in the bisect worktree; check run, llama-bench, server A/B. Output: /tmp/p2test
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
B=$LAB/rocm-strix-llama.cpp
W=$LAB/bisect-strix-llama.cpp
O=${OUT:-/tmp/p2test}
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
G="git -c safe.directory=*"
mkdir -p $O
exec > $O/run.log 2>&1
cd $W || exit 1
echo "start $(date +%T)"
$G checkout -- . && $G checkout --detach d3f9fb467 || { echo CHECKOUT_FAILED; exit 1; }
python3 /tmp/dq_gemv_patch.py || { echo PATCH_FAILED; exit 1; }
$G diff --stat
$G diff > $O/patch.diff
cmake --build build --config Release --target llama-server llama-bench -j"$(nproc)" >$O/build.log 2>&1 || { echo BUILD_FAILED; grep -E "error" $O/build.log | head -20; exit 1; }
echo "BUILD_OK $(date +%T)"
cd /tmp
echo "=== check $(date +%T)"
GGML_CUDA_DQ_CHECK=1 GGML_CUDA_DISABLE_GRAPHS=1 NEW_ROOT=$W TAGS=new ROUNDS=1 python3 /tmp/server_ab.py $O/check > $O/check.out 2>&1
L=$O/check/new_r1.server.log
BAD=$(grep -c ' BAD' $L)
echo "checks=$(grep -c DQ_GEMV_CHECK $L) bad=$BAD"
[ "$BAD" = 0 ] || { echo CHECK_FAILED; grep ' BAD' $L | head -5; touch $O/DONE; exit 1; }
bench() {  # tag dir gemv_max
  LD_LIBRARY_PATH=$2/build/bin GGML_CUDA_DQ_GEMV_MAX=$3 $2/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
    -p 1,2,3,4,5,6,8,9,16,512 -n 0 -d 0,4096 -r 5 -o json > $O/bench_$1.json 2>$O/bench_$1.err || echo "BENCH_FAILED $1"
}
for i in 1 2; do
  echo "=== bench round $i $(date +%T)"
  bench base_$i $B 8
  bench t0_$i $W 0
  bench t8_$i $W 8
done
echo "=== server $(date +%T)"
NEW_ROOT=$W TAGS=base,new ROUNDS=3 python3 /tmp/server_ab.py $O/srv
echo "DONE $(date +%T)"
touch $O/DONE

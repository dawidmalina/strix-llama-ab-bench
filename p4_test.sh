#!/bin/bash
# Host-side (as lemonade): d3f9fb467 + dq_gemv_patch.py + dq_gemv2_patch.py; V1 (current), A (prefetch kernel), B (no dense R2)
# vs baseline: bit-exact check of A, llama-bench, 3 interleaved server rounds. Output: /tmp/p4test
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
B=$LAB/rocm-strix-llama.cpp
W=$LAB/bisect-strix-llama.cpp
O=/tmp/p4test
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
G="git -c safe.directory=*"
mkdir -p $O
exec > $O/run.log 2>&1
cd $W || exit 1
echo "start $(date +%T)"
$G checkout -- . && $G checkout --detach d3f9fb467 || { echo CHECKOUT_FAILED; exit 1; }
python3 /tmp/dq_gemv_patch.py && python3 /tmp/dq_gemv2_patch.py || { echo PATCH_FAILED; exit 1; }
$G diff --stat
$G diff > $O/patch.diff
cmake --build build --config Release --target llama-server llama-bench -j"$(nproc)" >$O/build.log 2>&1 || { echo BUILD_FAILED; grep -E "error" $O/build.log | head -20; exit 1; }
echo "BUILD_OK $(date +%T)"
cd /tmp
echo "=== check A $(date +%T)"
GGML_CUDA_DQ_GEMV_VARIANT=2 GGML_CUDA_DQ_CHECK=1 GGML_CUDA_DISABLE_GRAPHS=1 NEW_ROOT=$W TAGS=new ROUNDS=1 python3 /tmp/server_ab.py $O/checkA > $O/checkA.out 2>&1
L=$O/checkA/new_r1.server.log
BAD=$(grep -c ' BAD' $L)
echo "checks=$(grep -c DQ_GEMV_CHECK $L) bad=$BAD"
[ "$BAD" = 0 ] || { echo CHECK_FAILED; grep ' BAD' $L | head -5; touch $O/DONE; exit 1; }
bench() {  # tag dir env...
  local t=$1 d=$2; shift 2
  env "$@" LD_LIBRARY_PATH=$d/build/bin $d/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
    -p 1,2,3,4,5,6,8,9,16,512,4096 -n 0 -d 0,4096 -r 5 -o json > $O/bench_$t.json 2>$O/bench_$t.err || echo "BENCH_FAILED $t"
}
for i in 1 2; do
  echo "=== bench round $i $(date +%T)"
  bench base_$i $B X=1
  bench v1_$i $W GGML_CUDA_DQ_GEMV_VARIANT=1
  bench a_$i $W GGML_CUDA_DQ_GEMV_VARIANT=2
  bench b_$i $W GGML_CUDA_MMB_DENSE_R2=0
done
srv() {  # name tag env...
  local n=$1 tag=$2; shift 2
  env "$@" NEW_ROOT=$W TAGS=$tag ROUNDS=1 python3 /tmp/server_ab.py $O/srv_${n}_r$r > $O/srv_${n}_r$r.out 2>&1
}
for r in 1 2 3; do
  echo "=== server round $r $(date +%T)"
  srv base base X=1
  srv v1 new GGML_CUDA_DQ_GEMV_VARIANT=1
  srv a new GGML_CUDA_DQ_GEMV_VARIANT=2
  srv b new GGML_CUDA_MMB_DENSE_R2=0
done
echo "DONE $(date +%T)"
touch $O/DONE

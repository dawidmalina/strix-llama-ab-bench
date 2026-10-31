#!/bin/bash
# Host-side (as lemonade): build d3f9fb467 + dq_mmvq_patch.py in the bisect worktree, then llama-bench (threshold 0/4/8 vs base),
# rocprofv3 pp3, and server_ab 3 interleaved rounds. Output: /tmp/p1test
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
B=$LAB/rocm-strix-llama.cpp
W=$LAB/bisect-strix-llama.cpp
O=/tmp/p1test
C=d3f9fb467
M=/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
G="git -c safe.directory=*"
mkdir -p $O
exec > $O/run.log 2>&1
cd $W || exit 1
echo "start $(date +%T)"
$G checkout -- . && $G checkout --detach $C || { echo CHECKOUT_FAILED; exit 1; }
python3 /tmp/dq_mmvq_patch.py || { echo PATCH_FAILED; exit 1; }
$G diff --stat
$G diff > $O/patch.diff
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH='$ORIGIN' >$O/cmake.log 2>&1 || { echo CONFIGURE_FAILED; exit 1; }
cmake --build build --config Release --target llama-server llama-bench -j"$(nproc)" >$O/build.log 2>&1 || { echo BUILD_FAILED; tail -40 $O/build.log; exit 1; }
echo "BUILD_OK $(date +%T)"
cd /tmp
bench() {  # tag dir threshold
  LD_LIBRARY_PATH=$2/build/bin GGML_CUDA_DQ_MMVQ_MAX=$3 $2/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct \
    -p 1,2,3,4,5,6,8,9,12,16,512,4096 -n 0 -d 0,4096 -r 5 -o json > $O/bench_$1.json 2>$O/bench_$1.err || echo "BENCH_FAILED $1"
}
for i in 1 2; do
  echo "=== bench round $i $(date +%T)"
  bench base_$i $B 8
  bench t0_$i $W 0
  bench t8_$i $W 8
  bench t4_$i $W 4
done
echo "=== prof $(date +%T)"
LD_LIBRARY_PATH=$W/build/bin rocprofv3 --kernel-trace --stats --output-format csv -d $O/pp3_t8 -o run -- \
  $W/build/bin/llama-bench -m "$M" -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct -p 3 -n 0 -r 50 -o json > $O/pp3_t8.json || echo PROF_FAILED
echo "=== server $(date +%T)"
NEW_ROOT=$W TAGS=base,new ROUNDS=3 python3 /tmp/server_ab.py $O/srv
echo "DONE $(date +%T)"
touch $O/DONE

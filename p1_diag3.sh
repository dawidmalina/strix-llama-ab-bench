#!/bin/bash
# Host-side (as lemonade): rebuild d3f9fb467 + dq_mmvq_patch.py + dq_check_patch.py, then server diagnostics. Output: /tmp/p1diag3
set -u
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
W=/var/lib/lemonade/lab/bisect-strix-llama.cpp
O=/tmp/p1diag3
G="git -c safe.directory=*"
mkdir -p $O
exec > $O/run.log 2>&1
cd $W || exit 1
$G checkout -- . && $G checkout --detach d3f9fb467 || { echo CHECKOUT_FAILED; exit 1; }
python3 /tmp/dq_mmvq_patch.py && python3 /tmp/dq_check_patch.py || { echo PATCH_FAILED; exit 1; }
$G diff --stat
cmake --build build --config Release --target llama-server -j"$(nproc)" >$O/build.log 2>&1 || { echo BUILD_FAILED; grep -E "error" $O/build.log | head -20; exit 1; }
echo "BUILD_OK $(date +%T)"
cd /tmp
run() {  # name, tests, env assignments...
  local n=$1 t=$2; shift 2
  echo "=== $n $(date +%T)"
  env "$@" NEW_ROOT=$W TAGS=new ROUNDS=1 ONLY=$t python3 /tmp/server_ab.py $O/$n > $O/$n.out 2>&1
}
run chk8 chat,long32k GGML_CUDA_DQ_MMVQ_MAX=8 GGML_CUDA_DQ_CHECK=1 GGML_CUDA_DISABLE_GRAPHS=1
run t8_nog code,chat,long32k GGML_CUDA_DQ_MMVQ_MAX=8 GGML_CUDA_DISABLE_GRAPHS=1
run t0_nog code,chat,long32k GGML_CUDA_DQ_MMVQ_MAX=0 GGML_CUDA_DISABLE_GRAPHS=1
echo "DONE $(date +%T)"
touch $O/DONE

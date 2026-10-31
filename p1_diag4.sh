#!/bin/bash
# Host-side (as lemonade): OVL bit tests on the patched worktree (already built). Output: /tmp/p1diag4
set -u
W=/var/lib/lemonade/lab/bisect-strix-llama.cpp
O=/tmp/p1diag4
mkdir -p $O
cd /tmp || exit 1
while [ ! -f /tmp/p1diag3/DONE ]; do sleep 10; done   # queue behind p1_diag3.sh
run() {  # name, tests, env assignments...
  local n=$1 t=$2; shift 2
  env "$@" NEW_ROOT=$W TAGS=new ROUNDS=1 ONLY=$t python3 /tmp/server_ab.py $O/$n > $O/$n.out 2>&1
}
run t8_ovl7 code,chat,long32k GGML_CUDA_DQ_MMVQ_MAX=8 GGML_CUDA_OVL=7
run t8_ovl0 code,chat,long32k GGML_CUDA_DQ_MMVQ_MAX=8 GGML_CUDA_OVL=0
run t0_ovl0 code,chat,long32k GGML_CUDA_DQ_MMVQ_MAX=0 GGML_CUDA_OVL=0
run t0_ovl7 code,chat,long32k GGML_CUDA_DQ_MMVQ_MAX=0 GGML_CUDA_OVL=7
touch $O/DONE

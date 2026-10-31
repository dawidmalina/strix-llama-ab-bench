#!/bin/bash
# Host-side (as lemonade): proposal-1 follow-up on the patched worktree. Output: /tmp/p1diag2
set -u
W=/var/lib/lemonade/lab/bisect-strix-llama.cpp
O=/tmp/p1diag2
mkdir -p $O
cd /tmp || exit 1
run() {  # name, env assignments...
  local n=$1; shift
  env "$@" NEW_ROOT=$W TAGS=new ROUNDS=1 ONLY=code,long32k python3 /tmp/server_ab.py $O/$n > $O/$n.out 2>&1
}
run t4 GGML_CUDA_DQ_MMVQ_MAX=4
run t5 GGML_CUDA_DQ_MMVQ_MAX=5
run t4_ovl0 GGML_CUDA_DQ_MMVQ_MAX=4 GGML_CUDA_OVL=0
GGML_CUDA_DQ_MMVQ_MAX=3 NEW_ROOT=$W TAGS=base,new ROUNDS=3 python3 /tmp/server_ab.py $O/ab_t3 > $O/ab_t3.out 2>&1
touch $O/DONE

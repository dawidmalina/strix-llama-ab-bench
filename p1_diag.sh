#!/bin/bash
# Host-side (as lemonade): server code-test diagnostics on the patched worktree, 1 round each. Output: /tmp/p1diag/<variant>
set -u
W=/var/lib/lemonade/lab/bisect-strix-llama.cpp
cd /tmp || exit 1
while [ ! -f /tmp/p1test/DONE ]; do sleep 10; done   # queue behind p1_test.sh
run() {  # name, env assignments...
  local n=$1; shift
  env "$@" NEW_ROOT=$W TAGS=new ROUNDS=1 ONLY=code,long32k python3 /tmp/server_ab.py /tmp/p1diag/$n > /tmp/p1diag/$n.out 2>&1
}
mkdir -p /tmp/p1diag
run t0 GGML_CUDA_DQ_MMVQ_MAX=0
run t8 GGML_CUDA_DQ_MMVQ_MAX=8
run t8_ovl0 GGML_CUDA_DQ_MMVQ_MAX=8 GGML_CUDA_OVL=0
run t2 GGML_CUDA_DQ_MMVQ_MAX=2
run t3 GGML_CUDA_DQ_MMVQ_MAX=3
touch /tmp/p1diag/DONE

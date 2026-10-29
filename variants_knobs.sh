#!/bin/bash
# Knob sweep on the bisect worktree build (8f9915925), twins off, 1 round each. Run as lemonade, detached.
cd /tmp || exit 1
T=$(date +%Y%m%d_%H%M%S)
W=/var/lib/lemonade/lab/bisect-strix-llama.cpp
export LLAMA_BF16_TWINS=0
for v in \
  "GDNCONV GGML_CUDA_DISABLE_GDN_CONV=1" \
  "GDNGATE GGML_CUDA_DISABLE_GDN_GATE=1" \
  "GDNNORMGATE GGML_CUDA_DISABLE_GDN_NORMGATE=1" \
  "GDNOUTNORM GGML_CUDA_DISABLE_GDN_OUTNORM=1" \
  "GDNQKNORM GGML_CUDA_DISABLE_GDN_QKNORM=1" \
  "FA7OFF GGML_FA7=0" \
  "FUSEDGDNCH LLAMA_FUSED_GDN_CH=0" \
  "WDOWN GGML_CUDA_DISABLE_WEIGHTED_DOWN=1" \
  "HCPRE LLAMA_FUSED_DSV4_HC_PRE=0"; do
  set -- $v
  tag=$1; shift
  env "$@" NEW_ROOT=$W TAGS=new ROUNDS=1 python3 /tmp/server_ab.py /tmp/knob_${T}_${tag}
done
touch /tmp/knob_${T}.DONE
echo "$T" > /tmp/knob_last

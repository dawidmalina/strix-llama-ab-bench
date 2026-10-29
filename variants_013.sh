#!/bin/bash
# Knob sweep on the bisect worktree build at 013044e12 (TWIN_MAX=3 + dense_r2 off patches already built), 1 round each.
cd /tmp || exit 1
T=$(date +%Y%m%d_%H%M%S)
W=/var/lib/lemonade/lab/bisect-strix-llama.cpp
for v in \
  "GATE16OFF GGML_CUDA_ATTN_GATE16=0" \
  "GDNGATEFUSEOFF GGML_CUDA_GDN_GATE_FUSE=0" \
  "DOWN16ANYOFF MMB_DOWN16_ANY=0" \
  "ZFORKON GGML_GDN_ZFORK=1" \
  "OVLOFF GGML_CUDA_OVL=0"; do
  set -- $v
  tag=$1; shift
  env "$@" NEW_ROOT=$W TAGS=new ROUNDS=1 python3 /tmp/server_ab.py /tmp/k013_${T}_${tag}
done
touch /tmp/k013_${T}.DONE
echo "$T" > /tmp/k013_last

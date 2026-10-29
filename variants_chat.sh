#!/bin/bash
# Beta-only env knob sweep with bf16 twins off (chat residual): 1 round each. Run as lemonade, detached.
cd /tmp || exit 1
T=$(date +%Y%m%d_%H%M%S)
export LLAMA_BF16_TWINS=0
for v in "ATTN0 LLAMA_FN_LAST_ATTN_TRIM=0" "MIX0 LLAMA_HC_MIX_F16=0" "BOTH0 LLAMA_FN_LAST_ATTN_TRIM=0 LLAMA_HC_MIX_F16=0"; do
  set -- $v
  tag=$1; shift
  env "$@" TAGS=new ROUNDS=1 python3 /tmp/server_ab.py /tmp/varchat_${T}_${tag}
done
touch /tmp/varchat_${T}.DONE
echo "$T" > /tmp/varchat_last

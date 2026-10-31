#!/bin/bash
# Host-side (as lemonade): known-cause fixes (twins MMF cutoff, gfx1151 MMA override) on clean master, 3 rounds each. Map: /tmp/master_known_map.txt
cd /tmp || exit 1
C=e38359599
FA=ggml/src/ggml-cuda/fattn-mma-f16.cuh
P_MMA="sed -i 's/if (rdna3_5 \&\& DV == DKQ \&\&/if (false \&\& rdna3_5 \&\& DV == DKQ \&\&/' $FA && git -c safe.directory=* diff --stat | grep -q fattn-mma"
: > /tmp/master_known_map.txt
run() {
  PATCH_CMD="$2" TWIN_MAX="$3" ROUNDS=3 TAGS=new bash /tmp/bisect_step.sh $C
  echo "$1 $(ls -dt /tmp/srvbisect_${C}_* | head -1)" >> /tmp/master_known_map.txt
}
run F1_twin_max3 "true" 3
run F2_mma_override_off "$P_MMA" ""
run F3_both "$P_MMA" 3
touch /tmp/master_known.DONE

#!/bin/bash
# Beta-only knob sweep (1 round each): MTP draft length and lazy mode. Run as lemonade, detached.
cd /tmp || exit 1
T=$(date +%Y%m%d_%H%M%S)
for v in "3 on-direct" "4 on-direct" "2 off"; do
  set -- $v
  SPEC_N=$1 LZM=$2 TAGS=new ROUNDS=1 python3 /tmp/server_ab.py /tmp/var_${T}_n$1_$2
done
touch /tmp/var_${T}.DONE

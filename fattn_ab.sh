#!/bin/bash
# Usage: sudo -n -u lemonade bash /tmp/fattn_ab.sh <parent> <commit>: build test-backend-ops at each and run FLASH_ATTN_EXT test+perf
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
W=$LAB/bisect-strix-llama.cpp
G="git -c safe.directory=*"
OUT=/tmp/fattn_ab
mkdir -p $OUT
exec > $OUT/driver.log 2>&1
for C in "$@"; do
  $G -C "$W" checkout -- src/models/qwen4exp.cpp
  $G -C "$W" checkout --detach "$C" || { echo CHECKOUT_FAILED $C; exit 1; }
  cd "$W" || exit 1
  cmake --build build --config Release --target test-backend-ops -j"$(nproc)" || { echo BUILD_FAILED $C; exit 1; }
  echo "BUILD_OK $C $(date +%T)"
  ./build/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT > $OUT/test_$C.txt 2>&1
  ./build/bin/test-backend-ops perf -b ROCm0 -o FLASH_ATTN_EXT > $OUT/perf_$C.txt 2>&1
  echo "RUN_DONE $C $(date +%T)"
done
echo ALL_DONE

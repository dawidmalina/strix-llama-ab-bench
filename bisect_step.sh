#!/bin/bash
# Usage: ssh host 'sudo -n -u lemonade bash -s -- <commit>' < bisect_step.sh
# Builds <commit> in a separate worktree (incremental) and runs one server_ab round on it only.
# Needs /tmp/server_ab.py with NEW_ROOT support. Result: /tmp/srvbisect_<commit>_*/new_r1.json
set -u
C=${1:?commit}
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
SRC=$LAB/beta-strix-llama.cpp
W=$LAB/bisect-strix-llama.cpp
mkdir -p "$LAB/build_logs"
exec > "$LAB/build_logs/bisect_${C}_$(date +%Y%m%d_%H%M%S).log" 2>&1
G="git -c safe.directory=*"
echo "start $(date +%T) commit=$C"
if [ ! -d "$W" ]; then
  $G -C "$SRC" worktree add --detach "$W" "$C" || { echo WORKTREE_FAILED; exit 1; }
else
  $G -C "$W" checkout -- .
  $G -C "$W" checkout --detach "$C" || { echo CHECKOUT_FAILED; exit 1; }
fi
cd "$W" || exit 1
$G checkout -- .
# TWIN_MAX=<n>: patch the bf16-twin token threshold (default 8) before building
if [ -n "${TWIN_MAX:-}" ]; then sed -i "s/n_tokens > 8 || !no_lora/n_tokens > ${TWIN_MAX} || !no_lora/" src/models/qwen4exp.cpp; fi
# PATCH_FILE + PATCH_SED: arbitrary sed patch of one file before building
if [ -n "${PATCH_FILE:-}" ]; then sed -i "${PATCH_SED}" "$PATCH_FILE" && $G diff --stat; fi
if [ -n "${PATCH_FILE2:-}" ]; then sed -i "${PATCH_SED2}" "$PATCH_FILE2" && $G diff --stat; fi
# PATCH_CMD: arbitrary command run in the worktree after checkout (e.g. python3 /tmp/revert_hunks.py 1-24)
if [ -n "${PATCH_CMD:-}" ]; then eval "$PATCH_CMD" || { echo PATCH_FAILED; exit 1; }; fi
$G log -1 --oneline
grep -n "n_tokens > .* || !no_lora" src/models/qwen4exp.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH='$ORIGIN' || { echo CONFIGURE_FAILED; exit 1; }
cmake --build build --config Release --target llama-server -j"$(nproc)" || { echo BUILD_FAILED; exit 1; }
echo "BUILD_OK $(date +%T)"
if pgrep -f "[l]lama-server|[l]lama-bench|[s]erver_ab|[b]ench_ab" >/dev/null; then echo "GPU_BUSY"; exit 1; fi
cd /tmp
NEW_ROOT=$W TAGS=new ROUNDS=1 python3 /tmp/server_ab.py "/tmp/srvbisect_${C}_$(date +%Y%m%d_%H%M%S)"
echo "STEP_DONE $(date +%T)"

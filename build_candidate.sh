#!/bin/bash
# Usage: ssh host 'sudo -n -u lemonade bash -s -- <beta|rocm> [pull]' < build_candidate.sh
# Fresh build of <name>-strix-llama.cpp; old build dir is moved aside, not deleted.
# "pull" = git pull --rebase first (beta only; baseline stays on its commit).
set -u
NAME=${1:?beta|rocm}
PULL=${2:-}
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
LAB=/var/lib/lemonade/lab
D=$LAB/$NAME-strix-llama.cpp
mkdir -p "$LAB/build_logs"
exec > "$LAB/build_logs/${NAME}_$(date +%Y%m%d_%H%M%S).log" 2>&1
cd "$D" || exit 1
G="git -c safe.directory=$D"
echo "start $(date +%T) rocm=$(hipconfig --version) $(readlink -f /opt/rocm/core)"
if [ "$PULL" = pull ]; then
  $G pull --rebase || { echo PULL_FAILED; exit 1; }
fi
$G log -1 --oneline
if [ -d build ]; then
  mv build "build.old.$(date +%Y%m%d_%H%M%S)"
fi
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH='$ORIGIN' || { echo CONFIGURE_FAILED; exit 1; }
cmake --build build --config Release -j"$(nproc)" || { echo BUILD_FAILED; exit 1; }
echo "BUILD_OK $(date +%T)"
LD_LIBRARY_PATH=$D/build/bin ./build/bin/llama-server --version 2>&1 | grep -iE "commit|version"
ldd build/bin/llama-server | grep -E "libllama|libggml-hip|libamdhip"

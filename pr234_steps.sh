#!/bin/bash
# Host-side (as lemonade): A = 013044e12 w/o #213 effects (hunks 45,47,48 reverted, NORM_ROPE_MULTI=0); B = A + PR #234 source patch.
cd /tmp || exit 1
export GGML_CUDA_NORM_ROPE_MULTI=0
git -c safe.directory=* -C /var/lib/lemonade/lab/beta-strix-llama.cpp show 1ee46afeb -- ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/mmb.cu ggml/src/ggml-cuda/mmb.cuh src > /tmp/pr234.patch
: > /tmp/pr234_map.txt
[ -n "$SKIP_A" ] || { PATCH_CMD="python3 /tmp/revert_hunks.py 45,47,48" bash /tmp/bisect_step.sh 013044e12; }
[ -n "$SKIP_A" ] || echo "A $(ls -dt /tmp/srvbisect_013044e12_* | head -1)" >> /tmp/pr234_map.txt
PATCH_CMD="python3 /tmp/revert_hunks.py 45,47,48 && patch -p1 --no-backup-if-mismatch -i /tmp/pr234.patch; true" bash /tmp/bisect_step.sh 013044e12
echo "B $(ls -dt /tmp/srvbisect_013044e12_* | head -1)" >> /tmp/pr234_map.txt
touch /tmp/pr234_steps.DONE

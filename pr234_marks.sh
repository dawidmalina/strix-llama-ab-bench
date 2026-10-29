#!/bin/bash
# Host-side (as lemonade): variants on PR #234 head (1ee46afeb): marks off / +R2 prepare off / +fusion on. Map: /tmp/pr234_marks_map.txt
cd /tmp || exit 1
MM=ggml/src/ggml-cuda/mmb.cu
GC=ggml/src/ggml-cuda/ggml-cuda.cu
P_MARKS="sed -i -e 's/{ mmb_state(ctx).f16_marks\[t\] |= bits; }/{ (void) ctx; (void) t; (void) bits; }/' -e 's/if (t->flags & GGML_TENSOR_FLAG_OUTPUT) return;/return;/' $MM && git -c safe.directory=* diff --stat"
P_R2="sed -i -e 's/ggml_cuda_mmb_down_r2_prepare(\*cuda_ctx, t->src\[0\])/(void) 0/' -e 's/ggml_cuda_mmb_dense_r2_prepare(\*cuda_ctx, t->src\[0\])/(void) 0/' -e 's/ggml_cuda_mmb_glu_r2_prepare(\*cuda_ctx, glu->src\[0\]->src\[0\], glu->src\[1\]->src\[0\])/(void) 0/' $GC && git -c safe.directory=* diff --stat"
: > /tmp/pr234_marks_map.txt
run() {
  PATCH_CMD="$2" bash /tmp/bisect_step.sh 1ee46afeb
  echo "$1 $(ls -dt /tmp/srvbisect_1ee46afeb_* | head -1)" >> /tmp/pr234_marks_map.txt
}
export GGML_CUDA_NORM_ROPE_MULTI=0
run V1_marks_off "$P_MARKS"
run V2_marks_and_r2_off "$P_MARKS; $P_R2"
unset GGML_CUDA_NORM_ROPE_MULTI
run V3_V2_fusion_on "$P_MARKS; $P_R2"
touch /tmp/pr234_marks.DONE

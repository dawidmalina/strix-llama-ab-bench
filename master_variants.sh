#!/bin/bash
# Host-side (as lemonade): variants on clean master e38359599, 3 rounds each, new only. Map: /tmp/master_variants_map.txt
cd /tmp || exit 1
C=e38359599
MM=ggml/src/ggml-cuda/mmb.cu
GC=ggml/src/ggml-cuda/ggml-cuda.cu
P_MARKS="sed -i -e 's/{ mmb_state(ctx).f16_marks\[t\] |= bits; }/{ (void) ctx; (void) t; (void) bits; }/' -e 's/if (t->flags & GGML_TENSOR_FLAG_OUTPUT) return;/return;/' $MM && git -c safe.directory=* diff --stat"
P_R2="sed -i -e 's/ggml_cuda_mmb_down_r2_prepare(\*cuda_ctx, t->src\[0\])/(void) 0/' -e 's/ggml_cuda_mmb_dense_r2_prepare(\*cuda_ctx, t->src\[0\])/(void) 0/' -e 's/ggml_cuda_mmb_glu_r2_prepare(\*cuda_ctx, glu->src\[0\]->src\[0\], glu->src\[1\]->src\[0\])/(void) 0/' $GC && git -c safe.directory=* diff --stat"
: > /tmp/master_variants_map.txt
run() {
  PATCH_CMD="$2" ROUNDS=3 TAGS=new bash /tmp/bisect_step.sh $C
  echo "$1 $(ls -dt /tmp/srvbisect_${C}_* | head -1)" >> /tmp/master_variants_map.txt
}
export GGML_CUDA_NORM_ROPE_MULTI=0
run M0_clean_norm_rope_multi_off "true"
unset GGML_CUDA_NORM_ROPE_MULTI
run M1_marks_and_r2_off "$P_MARKS; $P_R2"
touch /tmp/master_variants.DONE

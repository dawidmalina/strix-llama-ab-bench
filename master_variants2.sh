#!/bin/bash
# Host-side (as lemonade): overnight batch on master e38359599, 3 rounds each, new only. Map: /tmp/master_variants2_map.txt
cd /tmp || exit 1
C=e38359599
MM=ggml/src/ggml-cuda/mmb.cu
GC=ggml/src/ggml-cuda/ggml-cuda.cu
S_F16="-e 's/{ mmb_state(ctx).f16_marks\[t\] |= bits; }/{ (void) ctx; (void) t; (void) bits; }/'"
S_BF16="-e 's/if (t->flags & GGML_TENSOR_FLAG_OUTPUT) return;/return;/'"
P_F16="sed -i $S_F16 $MM && git -c safe.directory=* diff --stat"
P_BF16="sed -i $S_BF16 $MM && git -c safe.directory=* diff --stat"
P_MARKS="sed -i $S_F16 $S_BF16 $MM && git -c safe.directory=* diff --stat"
P_R2="sed -i -e 's/ggml_cuda_mmb_down_r2_prepare(\*cuda_ctx, t->src\[0\])/(void) 0/' -e 's/ggml_cuda_mmb_dense_r2_prepare(\*cuda_ctx, t->src\[0\])/(void) 0/' -e 's/ggml_cuda_mmb_glu_r2_prepare(\*cuda_ctx, glu->src\[0\]->src\[0\], glu->src\[1\]->src\[0\])/(void) 0/' $GC && git -c safe.directory=* diff --stat"
: > /tmp/master_variants2_map.txt
run() {
  PATCH_CMD="$3" ROUNDS=3 TAGS=new bash /tmp/bisect_step.sh "$2"
  echo "$1 $(ls -dt /tmp/srvbisect_$2_* | head -1)" >> /tmp/master_variants2_map.txt
}
run M2_marks_off_only $C "$P_MARKS"
run M3_r2_prepare_off_only $C "$P_R2"
run M4_mark_f16_off_only $C "$P_F16"
run M5_mark_bf16_only_off_only $C "$P_BF16"
run M6_pre235_54e9dccd4_clean 54e9dccd4 "true"
touch /tmp/master_variants2.DONE

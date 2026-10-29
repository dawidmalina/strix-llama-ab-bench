#!/bin/bash
# Patch sweep on 8f9915925 (twins off): one mmb.cu patch per variant, rebuild + 1 server round each. Run as lemonade, detached.
cd /tmp || exit 1
export LLAMA_BF16_TWINS=0
F=ggml/src/ggml-cuda/mmb.cu
: > /tmp/patch_map.txt
run() {
  tag=$1; sed_expr=$2
  PATCH_FILE=$F PATCH_SED="$sed_expr" bash /tmp/bisect_step.sh 8f9915925
  echo "$tag $(ls -dt /tmp/srvbisect_8f9915925_* | head -1)" >> /tmp/patch_map.txt
}
run DOWNR2  's/^\(void ggml_cuda_mmb_down_r2_prepare(.*) {\)$/\1 return;/'
run DENSER2 's/^\(void ggml_cuda_mmb_dense_r2_prepare(.*) {\)$/\1 return;/'
run GLUR2   's/^\(void ggml_cuda_mmb_glu_r2_prepare(.*) {\)$/\1 return;/'
run BF16ONLY 's/^\(void ggml_cuda_mmb_mark_bf16_only(.*) {\)$/\1 return;/'
run F16MARK 's/^\(void ggml_cuda_mmb_mark_f16(.*) {\)/\1 return;/'
touch /tmp/patch_sweep.DONE

#!/bin/bash
# Combined mmb.cu patch sweep on 013044e12 with twin threshold 3: rebuild + 1 server round each.
cd /tmp || exit 1
export TWIN_MAX=3
F=ggml/src/ggml-cuda/mmb.cu
D='s/^\(void ggml_cuda_mmb_dense_r2_prepare(.*) {\)$/\1 return;/'
DN='s/^\(void ggml_cuda_mmb_down_r2_prepare(.*) {\)$/\1 return;/'
G='s/^\(void ggml_cuda_mmb_glu_r2_prepare(.*) {\)$/\1 return;/'
B='s/^\(void ggml_cuda_mmb_mark_bf16_only(.*) {\)$/\1 return;/'
M='s/^\(void ggml_cuda_mmb_mark_f16(.*) {\)/\1 return;/'
: > /tmp/combo_map.txt
run() {
  tag=$1; expr=$2
  PATCH_FILE=$F PATCH_SED="$expr" bash /tmp/bisect_step.sh 013044e12
  echo "$tag $(ls -dt /tmp/srvbisect_013044e12_* | head -1)" >> /tmp/combo_map.txt
}
run DENSE_BF16_F16 "$D;$B;$M"
run ALL "$D;$DN;$G;$B;$M"
touch /tmp/combo.DONE

#!/usr/bin/env python3
# Proposal 1: MMVQ reads MMB_DENSE_R2 packed Q8_0 tiles (x_pk) for 2..GGML_CUDA_DQ_MMVQ_MAX tokens instead of the prefill tile GEMM.
# Run in the llama.cpp worktree root. Fails loudly if an anchor is missing or ambiguous.
import sys

def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"PATCH_ANCHOR {path}: expected {count}, found {n}: {old[:80]!r}")
    open(path, "w").write(s.replace(old, new))

MMVQ = "ggml/src/ggml-cuda/mmvq.cu"
GC = "ggml/src/ggml-cuda/ggml-cuda.cu"

sub(MMVQ,
"""template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(""",
"""// vec_dot_q8_0_q8_1 on a weight in the MMB_DENSE_R2 tile layout (same values, see mmvq_fq_q8_0_load)
static __device__ __forceinline__ float vec_dot_q8_0_q8_1_pk(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ bq8_1, const int row, const int blocks_per_row_x,
        const int kbx, const int iqs) {
    const uint8_t * ch = (const uint8_t *) vx + ((size_t) (row >> 7) * (blocks_per_row_x >> 1) + (kbx >> 1)) * 8704;
    const int r = row & 127;
    const int * q = (const int *) (ch + r*64 + (kbx & 1)*32);
    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = q[iqs + i];
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }
    const half d = *(const half *) (ch + 8192 + (r*2 + (kbx & 1))*2);
    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, d, __low2half(bq8_1->ds));
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(""")

sub(MMVQ,
"""            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (type == GGML_TYPE_IQ3_S && has_fusion) {""",
"""            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (type == GGML_TYPE_Q8_0) {
                    if (fusion.x_pk) {
                        tmp[j][i] += vec_dot_q8_0_q8_1_pk(vx, &y[j*stride_col_y + kby], row0 + i, blocks_per_row_x, kbx, kqs);
                        continue;
                    }
                }
                if constexpr (type == GGML_TYPE_IQ3_S && has_fusion) {""")

sub(MMVQ,
"""            if constexpr (type == GGML_TYPE_Q8_0) {
                const bool no_fusion = fusion.gate == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr;
                if (table_id == MMVQ_PARAMETERS_RDNA3_5 && ids == nullptr && no_fusion) {""",
"""            if constexpr (type == GGML_TYPE_Q8_0) {
                const bool no_fusion = fusion.gate == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr;
                if (table_id == MMVQ_PARAMETERS_RDNA3_5 && ids == nullptr && no_fusion && !fusion.x_pk) {""")

sub(MMVQ,
"""    // the generic kernels read Q8_0 rows: an MMB_DENSE_R2 packed weight must never get here
    GGML_ASSERT(!ggml_cuda_mmb_dq_packed(src0) && !(fusion && fusion->gate && ggml_cuda_mmb_dq_packed(fusion->gate)));
""",
"""    // MMB_DENSE_R2 packed weight: only the plain Q8_0 mul_mat_vec_q reads the tile layout (x_pk)
    const bool x_pk = ggml_cuda_mmb_dq_packed(src0);
    GGML_ASSERT(!x_pk || (src0->type == GGML_TYPE_Q8_0 && !ids && !fusion && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        ne01 % 128 == 0 && ne00 % 64 == 0));
    GGML_ASSERT(!(fusion && fusion->gate && ggml_cuda_mmb_dq_packed(fusion->gate)));
    fusion_local.x_pk = x_pk ? 1 : 0;
""")

sub(GC,
"""static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
""",
"""// largest batch an MMB_DENSE_R2 packed weight sends to MMVQ (x_pk) instead of the tile GEMM; 0 = always the tile GEMM
static int ggml_cuda_dq_mmvq_max() {
    static const int v = getenv("GGML_CUDA_DQ_MMVQ_MAX") ? atoi(getenv("GGML_CUDA_DQ_MMVQ_MAX")) : MMVQ_MAX_BATCH_SIZE;
    return std::min(v, MMVQ_MAX_BATCH_SIZE);
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
""")

sub(GC,
"""        } else {
            ggml_cuda_mul_mat_dq(ctx, src0, src1, dst);
        }
        return;
    }
""",
"""        } else if (src1->ne[1] <= ggml_cuda_dq_mmvq_max() && ggml_nrows(src1) == src1->ne[1] &&
                src0->ne[1] % 128 == 0 && src0->ne[0] % 64 == 0) {
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        } else {
            ggml_cuda_mul_mat_dq(ctx, src0, src1, dst);
        }
        return;
    }
""")
print("PATCH_OK")

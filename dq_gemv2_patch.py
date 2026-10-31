#!/usr/bin/env python3
# Applied after dq_gemv_patch.py. Adds:
#   GGML_CUDA_DQ_GEMV_VARIANT=2: same per-output WMMA chain (bit-exact), one wave per block, next 64-K chunk loaded ahead.
#   GGML_CUDA_MMB_DENSE_R2=0: skip the in-place dense Q8_0 repack entirely (stock Q8_0 layout, stock kernels).
import sys

def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"PATCH_ANCHOR {path}: expected {count}, found {n}: {old[:80]!r}")
    open(path, "w").write(s.replace(old, new))

GC = "ggml/src/ggml-cuda/ggml-cuda.cu"
MM = "ggml/src/ggml-cuda/mmb.cu"

sub(GC,
"""static int ggml_cuda_dq_gemv_max() {
""",
"""template <int T>
static __global__ void __launch_bounds__(32, 1) dq_gemv_f16_pf_kernel(
        const uint8_t * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K) {
#if !defined(RDNA3)
    GGML_UNUSED_VARS(W, X, D, M, K);
    NO_DEVICE_CODE;
#else
    const int lane = threadIdx.x, rl = lane & 15;
    const int m = blockIdx.x * 16 + rl, r = m & 127, nks = K / 64;
    const uint8_t * tile = W + (size_t) (m >> 7) * nks * 8704 + r * 64;
    const _Float16 * dtile = (const _Float16 *) (W + (size_t) (m >> 7) * nks * 8704 + 8192 + r * 4);
    const float * xrow = X + (size_t) (rl < T ? rl : 0) * K;
    dqg_v8f acc;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    int4 qc[4], qn[4];
    _Float16 dc[2], dn[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) qc[i] = ((const int4 *) tile)[i];
    dc[0] = dtile[0]; dc[1] = dtile[1];
    for (int ks = 0; ks < nks; ++ks) {
        const int kn = ks + 1 < nks ? ks + 1 : ks;
#pragma unroll
        for (int i = 0; i < 4; ++i) qn[i] = ((const int4 *) (tile + (size_t) kn * 8704))[i];
        dn[0] = dtile[(size_t) kn * 4352]; dn[1] = dtile[(size_t) kn * 4352 + 1];
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            const int8_t * q = (const int8_t *) &qc[kq];
            const _Float16 d = dc[kq >> 1];
            dqg_v16h a, b;
#pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = (_Float16) q[i] * d;
            if (rl < T) {
                const float4 * xp = (const float4 *) (xrow + ks * 64 + kq * 16);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const float4 x4 = xp[v];
                    const float xs[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float c = xs[e] != xs[e] ? xs[e] : fminf(fmaxf(xs[e], -65504.0f), 65504.0f);
                        b[v * 4 + e] = (_Float16) c;
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) b[i] = (_Float16) 0.0f;
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a, acc);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) qc[i] = qn[i];
        dc[0] = dn[0]; dc[1] = dn[1];
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int t = 2 * e + (lane >> 4);
        if (t < T) D[(size_t) t * M + m] = acc[e];
    }
#endif
}

static int ggml_cuda_dq_gemv_max() {
""")

sub(GC,
"""    const dim3 grid(M / 64), block(128);
    cudaStream_t stream = ctx.stream();
    switch (T) {
""",
"""    static const int variant = getenv("GGML_CUDA_DQ_GEMV_VARIANT") ? atoi(getenv("GGML_CUDA_DQ_GEMV_VARIANT")) : 1;
    if (variant == 2) {
        const dim3 grid2(M / 16), block2(32);
        cudaStream_t s2 = ctx.stream();
        switch (T) {
            case 2: dq_gemv_f16_pf_kernel<2><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            case 3: dq_gemv_f16_pf_kernel<3><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            case 4: dq_gemv_f16_pf_kernel<4><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            case 5: dq_gemv_f16_pf_kernel<5><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            case 6: dq_gemv_f16_pf_kernel<6><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            case 7: dq_gemv_f16_pf_kernel<7><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            case 8: dq_gemv_f16_pf_kernel<8><<<grid2, block2, 0, s2>>>(W, X, D, M, K); break;
            default: GGML_ABORT("dq_gemv: T=%d", T);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(M / 64), block(128);
    cudaStream_t stream = ctx.stream();
    switch (T) {
""")

sub(MM,
"""void ggml_cuda_mmb_dense_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!mmb_enabled(ctx)) return;
""",
"""void ggml_cuda_mmb_dense_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!mmb_enabled(ctx)) return;
    static const bool off = getenv("GGML_CUDA_MMB_DENSE_R2") != nullptr && atoi(getenv("GGML_CUDA_MMB_DENSE_R2")) == 0;
    if (off) return;
""")
print("GEMV2_PATCH_OK")

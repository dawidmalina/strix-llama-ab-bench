#!/usr/bin/env python3
# MMB_DENSE_R2 packed Q8_0 weights at 2..GGML_CUDA_DQ_GEMV_MAX tokens: a lean WMMA matvec with the exact per-output
# WMMA chain of the F16 tile GEMM (mmb_tq8_dense_kernel), instead of launching that GEMM with 64/256-token tiles.
# GGML_CUDA_DQ_CHECK=1 (with GGML_CUDA_DISABLE_GRAPHS=1) compares every call with ggml_cuda_mul_mat_dq (expects bit-exact).
# Run in the llama.cpp worktree root on a clean checkout.
import sys

def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"PATCH_ANCHOR {path}: expected {count}, found {n}: {old[:80]!r}")
    open(path, "w").write(s.replace(old, new))

GC = "ggml/src/ggml-cuda/ggml-cuda.cu"

sub(GC,
"""static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
""",
"""// Same WMMA chain per output as mmb_tq8_dense_kernel<..., F16=true> (F16 q*d weights, F16 saturated activations, K in
// 16-steps ascending), so each output is bit-identical to the tile GEMM at any batch size (batch-invariant MTP verify).
// One wave = 16 weight rows x 16 token slots (T real, rest zero). Layout: mmb_tq8_repack_kernel (8704 B per 128 rows x 64 K).
typedef _Float16 dqg_v16h __attribute__((ext_vector_type(16)));
typedef float    dqg_v8f  __attribute__((ext_vector_type(8)));
template <int T>
static __global__ void __launch_bounds__(128, 1) dq_gemv_f16_kernel(
        const uint8_t * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K) {
#if !defined(RDNA3)
    GGML_UNUSED_VARS(W, X, D, M, K);
    NO_DEVICE_CODE;
#else
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, rl = lane & 15;
    const int m = blockIdx.x * 64 + wave * 16 + rl, r = m & 127, nks = K / 64;
    const uint8_t * tile = W + (size_t) (m >> 7) * nks * 8704;
    const float * xrow = X + (size_t) (rl < T ? rl : 0) * K;
    dqg_v8f acc;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    for (int ks = 0; ks < nks; ++ks) {
        const uint8_t * ch = tile + (size_t) ks * 8704;
        const _Float16 d0 = *(const _Float16 *) (ch + 8192 + (r * 2 + 0) * 2);
        const _Float16 d1 = *(const _Float16 *) (ch + 8192 + (r * 2 + 1) * 2);
#pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
            const int4 qv = *(const int4 *) (ch + r * 64 + kk);
            const int8_t * q = (const int8_t *) &qv;
            const _Float16 d = kk < 32 ? d0 : d1;
            dqg_v16h a, b;
#pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = (_Float16) q[i] * d;
            if (rl < T) {
                const float4 * xp = (const float4 *) (xrow + ks * 64 + kk);
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
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int t = 2 * e + (lane >> 4);
        if (t < T) D[(size_t) t * M + m] = acc[e];
    }
#endif
}

static int ggml_cuda_dq_gemv_max() {
    static const int v = getenv("GGML_CUDA_DQ_GEMV_MAX") ? atoi(getenv("GGML_CUDA_DQ_GEMV_MAX")) : 8;
    return std::min(v, 8);
}

static bool ggml_cuda_dq_gemv_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const int64_t T = src1->ne[1];
    return T >= 2 && T <= ggml_cuda_dq_gemv_max() && ggml_nrows(src1) == T && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[1] % 128 == 0 && src0->ne[0] % 64 == 0 && (uintptr_t) src0->data % 16 == 0 && (uintptr_t) src1->data % 16 == 0;
}

static void ggml_cuda_dq_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = (int) src0->ne[1], K = (int) src0->ne[0], T = (int) src1->ne[1];
    const uint8_t * W = (const uint8_t *) src0->data;
    const float * X = (const float *) src1->data;
    float * D = (float *) dst->data;
    const dim3 grid(M / 64), block(128);
    cudaStream_t stream = ctx.stream();
    switch (T) {
        case 2: dq_gemv_f16_kernel<2><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        case 3: dq_gemv_f16_kernel<3><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        case 4: dq_gemv_f16_kernel<4><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        case 5: dq_gemv_f16_kernel<5><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        case 6: dq_gemv_f16_kernel<6><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        case 7: dq_gemv_f16_kernel<7><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        case 8: dq_gemv_f16_kernel<8><<<grid, block, 0, stream>>>(W, X, D, M, K); break;
        default: GGML_ABORT("dq_gemv: T=%d", T);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_dq_gemv_check(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static std::map<std::string, int> seen;
    cudaStream_t stream = ctx.stream();
    const size_t n = (size_t) src0->ne[1] * src1->ne[1];
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), n);
    ggml_tensor d2 = *dst; d2.data = ref.get();
    ggml_cuda_mul_mat_dq(ctx, src0, src1, &d2);
    std::vector<float> a(n), b(n);
    CUDA_CHECK(cudaMemcpyAsync(a.data(), dst->data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.data(), ref.get(), n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float mx = 0.0f, dmax = 0.0f; size_t neq = 0;
    for (size_t i = 0; i < n; ++i) {
        mx = std::max(mx, std::fabs(b[i]));
        dmax = std::max(dmax, std::fabs(a[i] - b[i]));
        neq += memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    const std::string key = std::string(src0->name) + "/T" + std::to_string(src1->ne[1]);
    if (neq || seen[key]++ < 1) {
        fprintf(stderr, "DQ_GEMV_CHECK %s T=%d M=%d K=%d max|ref|=%g maxdiff=%g neq=%zu/%zu%s\\n", src0->name, (int) src1->ne[1],
            (int) src0->ne[1], (int) src0->ne[0], mx, dmax, neq, n, neq ? " BAD" : "");
    }
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
"""        } else if (ggml_cuda_dq_gemv_ok(src0, src1, dst)) {
            ggml_cuda_dq_gemv(ctx, src0, src1, dst);
            static const bool chk = getenv("GGML_CUDA_DQ_CHECK") != nullptr;
            if (chk) ggml_cuda_dq_gemv_check(ctx, src0, src1, dst);
        } else {
            ggml_cuda_mul_mat_dq(ctx, src0, src1, dst);
        }
        return;
    }
""")
print("GEMV_PATCH_OK")

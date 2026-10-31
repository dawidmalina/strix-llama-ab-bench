#!/usr/bin/env python3
# Debug mode on top of dq_mmvq_patch.py: GGML_CUDA_DQ_CHECK=1 compares, per packed-weight MUL_MAT at 2..8 tokens,
# the x_pk MMVQ result with MMVQ on an unpacked copy (expected bit-identical) and with mul_mat_dq (old path).
# Needs GGML_CUDA_DISABLE_GRAPHS=1 (host sync per call).
import sys

def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"PATCH_ANCHOR {path}: expected {count}, found {n}: {old[:80]!r}")
    open(path, "w").write(s.replace(old, new))

GC = "ggml/src/ggml-cuda/ggml-cuda.cu"

sub(GC,
"""// largest batch an MMB_DENSE_R2 packed weight sends to MMVQ (x_pk) instead of the tile GEMM; 0 = always the tile GEMM
""",
"""#include <map>
#include <string>
#include <vector>
#include <cmath>

static __global__ void dq_check_unpack_kernel(const uint8_t * __restrict__ W, uint8_t * __restrict__ out, const int M, const int K) {
    const int nkb = K / 32;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) M * nkb) return;
    const int m = (int) (i / nkb), b = (int) (i % nkb);
    const int mt = m / 128, r = m % 128, ks = b / 2, bb = b % 2;
    const uint8_t * c = W + ((size_t) mt * (K / 64) + ks) * 8704;
    uint8_t * d = out + (size_t) m * nkb * 34 + (size_t) b * 34;
    d[0] = c[8192 + (r * 2 + bb) * 2]; d[1] = c[8192 + (r * 2 + bb) * 2 + 1];
    for (int k = 0; k < 32; ++k) d[2 + k] = c[r * 64 + bb * 32 + k];
}

static void dq_check(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static std::map<std::string, int> seen;
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], T = (int) src1->ne[1];
    const size_t n = (size_t) M * T;
    ggml_cuda_pool_alloc<uint8_t> wu(ctx.pool(), ggml_nbytes(src0));
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), n), old(ctx.pool(), n);
    const size_t nb = (size_t) M * (K / 32);
    dq_check_unpack_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>((const uint8_t *) src0->data, wu.get(), M, K);
    ggml_tensor w2 = *src0; w2.data = wu.get();
    ggml_tensor d2 = *dst;  d2.data = ref.get();
    ggml_cuda_mul_mat_vec_q(ctx, &w2, src1, nullptr, &d2);
    ggml_tensor d3 = *dst;  d3.data = old.get();
    ggml_cuda_mul_mat_dq(ctx, src0, src1, &d3);
    std::vector<float> a(n), r(n), o(n);
    CUDA_CHECK(cudaMemcpyAsync(a.data(), dst->data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(r.data(), ref.get(), n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(o.data(), old.get(), n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float mx = 0, d_pk = 0, d_old = 0; size_t neq = 0, nnan = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) ++nnan;
        mx = std::max(mx, std::fabs(r[i]));
        d_pk = std::max(d_pk, std::fabs(a[i] - r[i]));
        d_old = std::max(d_old, std::fabs(o[i] - r[i]));
        neq += a[i] != r[i];
    }
    const std::string key = std::string(src0->name) + "/T" + std::to_string(T);
    const bool bad = nnan || d_pk > 1e-3f * std::max(mx, 1.0f);
    if (bad || seen[key]++ < 1) {
        fprintf(stderr, "DQ_CHECK %s %s T=%d M=%d K=%d max|ref|=%g pk_vs_ref=%g (neq=%zu nan=%zu) old_vs_ref=%g%s\\n",
            src0->name, dst->name, T, M, K, mx, d_pk, neq, nnan, d_old, bad ? " BAD" : "");
    }
}

// largest batch an MMB_DENSE_R2 packed weight sends to MMVQ (x_pk) instead of the tile GEMM; 0 = always the tile GEMM
""")

sub(GC,
"""            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        } else {
            ggml_cuda_mul_mat_dq(ctx, src0, src1, dst);
        }
        return;
    }
""",
"""            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            static const bool chk = getenv("GGML_CUDA_DQ_CHECK") != nullptr;
            if (chk) dq_check(ctx, src0, src1, dst);
        } else {
            ggml_cuda_mul_mat_dq(ctx, src0, src1, dst);
        }
        return;
    }
""")
print("CHECK_PATCH_OK")

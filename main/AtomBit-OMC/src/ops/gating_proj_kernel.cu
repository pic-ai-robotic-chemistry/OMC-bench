#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

inline void cuda_check_last_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

constexpr int THREADS = 256;

__global__ void fused_gating_proj_fwd_kernel(
    const float* __restrict__ r_hat,     // (E, 3)
    const float* __restrict__ h_src,     // (E, 3, F)
    const float* __restrict__ h_dst,     // (E, 3, F)
    const float* __restrict__ s_basis,   // (E, F)
    float* __restrict__ out,             // (E, 3F)
    int E,
    int F) {

    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) {
        return;
    }

    const float r_x = r_hat[e * 3 + 0];
    const float r_y = r_hat[e * 3 + 1];
    const float r_z = r_hat[e * 3 + 2];

    float* out_ptr = out + e * 3 * F;

    for (int f = tid; f < F; f += blockDim.x) {
        const int h_base = e * 3 * F + f;

        const float s_val = s_basis[e * F + f];

        const float hsx = h_src[h_base];
        const float hsy = h_src[h_base + F];
        const float hsz = h_src[h_base + 2 * F];
        const float p_src = r_x * hsx + r_y * hsy + r_z * hsz;

        const float hdx = h_dst[h_base];
        const float hdy = h_dst[h_base + F];
        const float hdz = h_dst[h_base + 2 * F];
        const float p_dst = r_x * hdx + r_y * hdy + r_z * hdz;

        out_ptr[f]         = s_val;
        out_ptr[f + F]     = p_src;
        out_ptr[f + 2 * F] = p_dst;
    }
}

__global__ void fused_gating_proj_bwd_kernel(
    const float* __restrict__ grad_out,  // (E, 3F)
    const float* __restrict__ r_hat,     // (E, 3)
    const float* __restrict__ h_src,     // (E, 3, F)
    const float* __restrict__ h_dst,     // (E, 3, F)
    float* __restrict__ grad_r_hat,      // (E, 3)
    float* __restrict__ grad_h_src,      // (E, 3, F)
    float* __restrict__ grad_h_dst,      // (E, 3, F)
    float* __restrict__ grad_s_basis,    // (E, F)
    int E,
    int F) {

    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) {
        return;
    }

    const float r_x = r_hat[e * 3 + 0];
    const float r_y = r_hat[e * 3 + 1];
    const float r_z = r_hat[e * 3 + 2];

    const float* g_ptr = grad_out + e * 3 * F;

    float local0 = 0.0f;
    float local1 = 0.0f;
    float local2 = 0.0f;

    for (int f = tid; f < F; f += blockDim.x) {
        const float g_s  = g_ptr[f];
        const float g_ps = g_ptr[f + F];
        const float g_pd = g_ptr[f + 2 * F];

        grad_s_basis[e * F + f] = g_s;

        const int h_base = e * 3 * F + f;

        grad_h_src[h_base]         = r_x * g_ps;
        grad_h_src[h_base + F]     = r_y * g_ps;
        grad_h_src[h_base + 2 * F] = r_z * g_ps;

        grad_h_dst[h_base]         = r_x * g_pd;
        grad_h_dst[h_base + F]     = r_y * g_pd;
        grad_h_dst[h_base + 2 * F] = r_z * g_pd;

        const float hsx = h_src[h_base];
        const float hsy = h_src[h_base + F];
        const float hsz = h_src[h_base + 2 * F];

        const float hdx = h_dst[h_base];
        const float hdy = h_dst[h_base + F];
        const float hdz = h_dst[h_base + 2 * F];

        local0 += hsx * g_ps + hdx * g_pd;
        local1 += hsy * g_ps + hdy * g_pd;
        local2 += hsz * g_ps + hdz * g_pd;
    }

    __shared__ float s0[THREADS];
    __shared__ float s1[THREADS];
    __shared__ float s2[THREADS];

    s0[tid] = local0;
    s1[tid] = local1;
    s2[tid] = local2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            s2[tid] += s2[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_r_hat[e * 3 + 0] = s0[0];
        grad_r_hat[e * 3 + 1] = s1[0];
        grad_r_hat[e * 3 + 2] = s2[0];
    }
}

} // namespace

torch::Tensor gating_proj_forward(
    torch::Tensor r_hat,
    torch::Tensor h_src,
    torch::Tensor h_dst,
    torch::Tensor scalar_basis) {

    const int E = static_cast<int>(r_hat.size(0));
    const int F = static_cast<int>(scalar_basis.size(1));

    auto out = torch::empty({E, 3 * F}, scalar_basis.options());

    dim3 blocks(E);
    dim3 threads(THREADS);

    fused_gating_proj_fwd_kernel<<<blocks, threads>>>(
        r_hat.data_ptr<float>(),
        h_src.data_ptr<float>(),
        h_dst.data_ptr<float>(),
        scalar_basis.data_ptr<float>(),
        out.data_ptr<float>(),
        E,
        F
    );
    cuda_check_last_error("gating_proj_forward launch failed");
    return out;
}

std::vector<torch::Tensor> gating_proj_backward(
    torch::Tensor grad_out,
    torch::Tensor r_hat,
    torch::Tensor h_src,
    torch::Tensor h_dst,
    torch::Tensor scalar_basis) {

    const int E = static_cast<int>(r_hat.size(0));
    const int F = static_cast<int>(scalar_basis.size(1));

    auto grad_r_hat   = torch::empty_like(r_hat);
    auto grad_h_src   = torch::empty_like(h_src);
    auto grad_h_dst   = torch::empty_like(h_dst);
    auto grad_s_basis = torch::empty_like(scalar_basis);

    dim3 blocks(E);
    dim3 threads(THREADS);

    fused_gating_proj_bwd_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        r_hat.data_ptr<float>(),
        h_src.data_ptr<float>(),
        h_dst.data_ptr<float>(),
        grad_r_hat.data_ptr<float>(),
        grad_h_src.data_ptr<float>(),
        grad_h_dst.data_ptr<float>(),
        grad_s_basis.data_ptr<float>(),
        E,
        F
    );
    cuda_check_last_error("gating_proj_backward launch failed");

    return {grad_r_hat, grad_h_src, grad_h_dst, grad_s_basis};
}

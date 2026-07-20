#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

static inline void cuda_check_last_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, msg, ": CUDA error: ", cudaGetErrorString(err));
    }
}

__global__ void bessel_fwd_kernel(
    const float* __restrict__ d,      // (E,)
    const float* __restrict__ freq,   // (F,)
    float* __restrict__ out,          // (E,F)
    float r_max,
    float prefactor,
    int E,
    int F
) {
    int e = blockIdx.x;
    int f = threadIdx.x;
    if (e >= E || f >= F) return;

    float d_val = d[e];
    float inv_r_max = 1.0f / r_max;

    float arg = freq[f] * (d_val * inv_r_max);
    float numerator = prefactor * sinf(arg);
    float denom = d_val + 1e-6f;

    out[e * F + f] = numerator / denom;
}

// grad_out: (E,F) -> grad_d: (E,)
__global__ void bessel_bwd_kernel(
    const float* __restrict__ d,        // (E,)
    const float* __restrict__ freq,     // (F,)
    const float* __restrict__ grad_out, // (E,F)
    float* __restrict__ grad_d,         // (E,)
    float r_max,
    float prefactor,
    int E,
    int F
) {
    int e = blockIdx.x;
    int f = threadIdx.x;
    if (e >= E || f >= F) return;

    float d_val = d[e];
    float w = freq[f];
    float inv_r_max = 1.0f / r_max;
    float eps = 1e-6f;

    float denom = d_val + eps;
    float arg = w * (d_val * inv_r_max);

    // dy/dd = pref * [ cos(arg)*(w/r_max)/(d+eps) - sin(arg)/(d+eps)^2 ]
    float term1 = cosf(arg) * (w * inv_r_max) / denom;
    float term2 = sinf(arg) / (denom * denom);
    float dy_dd = prefactor * (term1 - term2);

    float go = grad_out[e * F + f];
    float contrib = go * dy_dd;

    // reduce across f -> sum to grad_d[e]
    // Minimal: use atomicAdd (OK if F isn't huge; simple and correct)
    atomicAdd(&grad_d[e], contrib);
}

torch::Tensor bessel_cuda_forward(torch::Tensor d, double r_max, torch::Tensor freq) {
    const int E = (int)d.size(0);
    const int F = (int)freq.size(0);

    auto out = torch::empty({E, F}, d.options());
    float rmax_f = (float)r_max;
    float prefactor = sqrtf(2.0f / rmax_f);

    TORCH_CHECK(F <= 1024, "This minimal kernel assumes F <= 1024, got F=", F);

    dim3 blocks(E);
    dim3 threads(F);

    bessel_fwd_kernel<<<blocks, threads>>>(
        d.data_ptr<float>(),
        freq.data_ptr<float>(),
        out.data_ptr<float>(),
        rmax_f,
        prefactor,
        E, F
    );
    cuda_check_last_error("bessel forward launch failed");
    return out;
}

// returns grad_d (E,)
torch::Tensor bessel_cuda_backward(torch::Tensor grad_out, torch::Tensor d, double r_max, torch::Tensor freq) {
    const int E = (int)d.size(0);
    const int F = (int)freq.size(0);

    auto grad_d = torch::zeros({E}, d.options()); // must be zero for atomicAdd
    float rmax_f = (float)r_max;
    float prefactor = sqrtf(2.0f / rmax_f);

    TORCH_CHECK(F <= 1024, "This minimal kernel assumes F <= 1024, got F=", F);

    dim3 blocks(E);
    dim3 threads(F);

    bessel_bwd_kernel<<<blocks, threads>>>(
        d.data_ptr<float>(),
        freq.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_d.data_ptr<float>(),
        rmax_f,
        prefactor,
        E, F
    );
    cuda_check_last_error("bessel backward launch failed");
    return grad_d;
}

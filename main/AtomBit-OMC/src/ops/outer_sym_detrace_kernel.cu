#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void outer_sym_detrace_forward_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ g,
    scalar_t* __restrict__ out,
    int64_t EF,
    int64_t F) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= EF) {
        return;
    }

    const int64_t e = idx / F;
    const int64_t f = idx - e * F;

    const int64_t base_h = e * 3 * F + f;
    const int64_t h0_off = base_h;
    const int64_t h1_off = base_h + F;
    const int64_t h2_off = base_h + 2 * F;

    const scalar_t h0 = h[h0_off];
    const scalar_t h1 = h[h1_off];
    const scalar_t h2 = h[h2_off];

    const scalar_t g0 = g[h0_off];
    const scalar_t g1 = g[h1_off];
    const scalar_t g2 = g[h2_off];

    const scalar_t half = scalar_t(0.5);
    const scalar_t third = scalar_t(1.0 / 3.0);
    const scalar_t t = (h0 * g0 + h1 * g1 + h2 * g2) * third;

    const scalar_t o01 = half * (h0 * g1 + h1 * g0);
    const scalar_t o02 = half * (h0 * g2 + h2 * g0);
    const scalar_t o12 = half * (h1 * g2 + h2 * g1);

    const int64_t base_o = e * 9 * F + f;
    out[base_o + 0 * F] = h0 * g0 - t;
    out[base_o + 1 * F] = o01;
    out[base_o + 2 * F] = o02;
    out[base_o + 3 * F] = o01;
    out[base_o + 4 * F] = h1 * g1 - t;
    out[base_o + 5 * F] = o12;
    out[base_o + 6 * F] = o02;
    out[base_o + 7 * F] = o12;
    out[base_o + 8 * F] = h2 * g2 - t;
}


template <typename scalar_t>
__global__ void outer_sym_detrace_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ g,
    scalar_t* __restrict__ grad_h,
    scalar_t* __restrict__ grad_g,
    int64_t EF,
    int64_t F) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= EF) {
        return;
    }

    const int64_t e = idx / F;
    const int64_t f = idx - e * F;

    const int64_t base_h = e * 3 * F + f;
    const int64_t h0_off = base_h;
    const int64_t h1_off = base_h + F;
    const int64_t h2_off = base_h + 2 * F;

    const scalar_t h0 = h[h0_off];
    const scalar_t h1 = h[h1_off];
    const scalar_t h2 = h[h2_off];

    const scalar_t g0 = g[h0_off];
    const scalar_t g1 = g[h1_off];
    const scalar_t g2 = g[h2_off];

    const int64_t base_go = e * 9 * F + f;
    const scalar_t go00 = grad_out[base_go + 0 * F];
    const scalar_t go01 = grad_out[base_go + 1 * F];
    const scalar_t go02 = grad_out[base_go + 2 * F];
    const scalar_t go10 = grad_out[base_go + 3 * F];
    const scalar_t go11 = grad_out[base_go + 4 * F];
    const scalar_t go12 = grad_out[base_go + 5 * F];
    const scalar_t go20 = grad_out[base_go + 6 * F];
    const scalar_t go21 = grad_out[base_go + 7 * F];
    const scalar_t go22 = grad_out[base_go + 8 * F];

    const scalar_t half = scalar_t(0.5);
    const scalar_t third = scalar_t(1.0 / 3.0);
    const scalar_t diag_avg = (go00 + go11 + go22) * third;
    const scalar_t s01 = half * (go01 + go10);
    const scalar_t s02 = half * (go02 + go20);
    const scalar_t s12 = half * (go12 + go21);

    grad_h[h0_off] = go00 * g0 + s01 * g1 + s02 * g2 - diag_avg * g0;
    grad_h[h1_off] = s01 * g0 + go11 * g1 + s12 * g2 - diag_avg * g1;
    grad_h[h2_off] = s02 * g0 + s12 * g1 + go22 * g2 - diag_avg * g2;

    grad_g[h0_off] = go00 * h0 + s01 * h1 + s02 * h2 - diag_avg * h0;
    grad_g[h1_off] = s01 * h0 + go11 * h1 + s12 * h2 - diag_avg * h1;
    grad_g[h2_off] = s02 * h0 + s12 * h1 + go22 * h2 - diag_avg * h2;
}

}  // namespace

at::Tensor outer_sym_detrace_forward_cuda(const at::Tensor& h_trans, const at::Tensor& geom) {
    const c10::cuda::CUDAGuard device_guard(h_trans.device());

    auto out = at::empty({h_trans.size(0), 3, 3, h_trans.size(2)}, h_trans.options());

    const int64_t E = h_trans.size(0);
    const int64_t F = h_trans.size(2);
    const int64_t EF = E * F;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((EF + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(h_trans.scalar_type(), "outer_sym_detrace_forward_cuda", [&] {
        outer_sym_detrace_forward_kernel<scalar_t><<<blocks, threads>>>(
            h_trans.data_ptr<scalar_t>(),
            geom.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            EF,
            F);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<at::Tensor> outer_sym_detrace_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& h_trans,
    const at::Tensor& geom) {
    const c10::cuda::CUDAGuard device_guard(h_trans.device());

    auto grad_h = at::empty_like(h_trans);
    auto grad_g = at::empty_like(geom);

    const int64_t E = h_trans.size(0);
    const int64_t F = h_trans.size(2);
    const int64_t EF = E * F;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((EF + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(h_trans.scalar_type(), "outer_sym_detrace_backward_cuda", [&] {
        outer_sym_detrace_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            h_trans.data_ptr<scalar_t>(),
            geom.data_ptr<scalar_t>(),
            grad_h.data_ptr<scalar_t>(),
            grad_g.data_ptr<scalar_t>(),
            EF,
            F);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_h, grad_g};
}

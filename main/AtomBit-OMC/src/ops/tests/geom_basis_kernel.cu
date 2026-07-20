
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr double kEps = 1.0e-6;

template <typename scalar_t>
__global__ void compute_rhat_kernel(
    const int64_t E,
    const scalar_t* __restrict__ vec,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ rhat) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }

    const scalar_t inv = scalar_t(1.0) / (d[e] + static_cast<scalar_t>(kEps));
    const int64_t base = e * 3;
    rhat[base + 0] = vec[base + 0] * inv;
    rhat[base + 1] = vec[base + 1] * inv;
    rhat[base + 2] = vec[base + 2] * inv;
}

template <typename scalar_t>
__global__ void compute_basis1_kernel(
    const int64_t EF,
    const int64_t F,
    const scalar_t* __restrict__ rhat,
    const scalar_t* __restrict__ rbf,
    scalar_t* __restrict__ basis1) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= EF) {
        return;
    }

    const int64_t e = idx / F;
    const int64_t f = idx - e * F;

    const scalar_t rf = rbf[idx];
    const int64_t rbase = e * 3;
    const int64_t obase = e * 3 * F + f;

    basis1[obase + 0 * F] = rhat[rbase + 0] * rf;
    basis1[obase + 1 * F] = rhat[rbase + 1] * rf;
    basis1[obase + 2 * F] = rhat[rbase + 2] * rf;
}

template <typename scalar_t>
__global__ void compute_basis2_kernel(
    const int64_t EF,
    const int64_t F,
    const scalar_t* __restrict__ rhat,
    const scalar_t* __restrict__ rbf,
    scalar_t* __restrict__ basis2) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= EF) {
        return;
    }

    const int64_t e = idx / F;
    const int64_t f = idx - e * F;

    const scalar_t rf = rbf[idx];
    const int64_t rbase = e * 3;
    const scalar_t r0 = rhat[rbase + 0];
    const scalar_t r1 = rhat[rbase + 1];
    const scalar_t r2 = rhat[rbase + 2];
    const scalar_t one_third = static_cast<scalar_t>(1.0 / 3.0);

    const int64_t obase = e * 9 * F + f;

    basis2[obase + 0 * F] = (r0 * r0 - one_third) * rf;
    basis2[obase + 1 * F] = (r0 * r1) * rf;
    basis2[obase + 2 * F] = (r0 * r2) * rf;

    basis2[obase + 3 * F] = (r1 * r0) * rf;
    basis2[obase + 4 * F] = (r1 * r1 - one_third) * rf;
    basis2[obase + 5 * F] = (r1 * r2) * rf;

    basis2[obase + 6 * F] = (r2 * r0) * rf;
    basis2[obase + 7 * F] = (r2 * r1) * rf;
    basis2[obase + 8 * F] = (r2 * r2 - one_third) * rf;
}

template <typename scalar_t>
__global__ void geom_basis_backward_kernel(
    const int64_t E,
    const int64_t F,
    const bool need_l1,
    const bool need_l2,
    const scalar_t* __restrict__ grad_rhat,
    const scalar_t* __restrict__ grad_basis1,
    const scalar_t* __restrict__ grad_basis2,
    const scalar_t* __restrict__ vec,
    const scalar_t* __restrict__ d,
    const scalar_t* __restrict__ rbf,
    scalar_t* __restrict__ grad_vec,
    scalar_t* __restrict__ grad_d,
    scalar_t* __restrict__ grad_rbf) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }

    const int64_t vbase = e * 3;
    const scalar_t v0 = vec[vbase + 0];
    const scalar_t v1 = vec[vbase + 1];
    const scalar_t v2 = vec[vbase + 2];

    const scalar_t inv = scalar_t(1.0) / (d[e] + static_cast<scalar_t>(kEps));
    const scalar_t inv2 = inv * inv;
    const scalar_t r0 = v0 * inv;
    const scalar_t r1 = v1 * inv;
    const scalar_t r2 = v2 * inv;
    const scalar_t one_third = static_cast<scalar_t>(1.0 / 3.0);

    scalar_t g_r0 = grad_rhat[vbase + 0];
    scalar_t g_r1 = grad_rhat[vbase + 1];
    scalar_t g_r2 = grad_rhat[vbase + 2];

    const int64_t ef_base = e * F;

    for (int64_t f = 0; f < F; ++f) {
        const int64_t ef = ef_base + f;
        const scalar_t rf = rbf[ef];
        scalar_t g_rf = scalar_t(0);

        if (need_l1) {
            const int64_t b1base = e * 3 * F + f;
            const scalar_t gb10 = grad_basis1[b1base + 0 * F];
            const scalar_t gb11 = grad_basis1[b1base + 1 * F];
            const scalar_t gb12 = grad_basis1[b1base + 2 * F];

            g_r0 += gb10 * rf;
            g_r1 += gb11 * rf;
            g_r2 += gb12 * rf;

            g_rf += gb10 * r0 + gb11 * r1 + gb12 * r2;
        }

        if (need_l2) {
            const int64_t b2base = e * 9 * F + f;

            const scalar_t gb00 = grad_basis2[b2base + 0 * F];
            const scalar_t gb01 = grad_basis2[b2base + 1 * F];
            const scalar_t gb02 = grad_basis2[b2base + 2 * F];
            const scalar_t gb10 = grad_basis2[b2base + 3 * F];
            const scalar_t gb11 = grad_basis2[b2base + 4 * F];
            const scalar_t gb12 = grad_basis2[b2base + 5 * F];
            const scalar_t gb20 = grad_basis2[b2base + 6 * F];
            const scalar_t gb21 = grad_basis2[b2base + 7 * F];
            const scalar_t gb22 = grad_basis2[b2base + 8 * F];

            const scalar_t gT00 = gb00 * rf;
            const scalar_t gT01 = gb01 * rf;
            const scalar_t gT02 = gb02 * rf;
            const scalar_t gT10 = gb10 * rf;
            const scalar_t gT11 = gb11 * rf;
            const scalar_t gT12 = gb12 * rf;
            const scalar_t gT20 = gb20 * rf;
            const scalar_t gT21 = gb21 * rf;
            const scalar_t gT22 = gb22 * rf;

            g_r0 += (gT00 + gT00) * r0 + (gT01 + gT10) * r1 + (gT02 + gT20) * r2;
            g_r1 += (gT10 + gT01) * r0 + (gT11 + gT11) * r1 + (gT12 + gT21) * r2;
            g_r2 += (gT20 + gT02) * r0 + (gT21 + gT12) * r1 + (gT22 + gT22) * r2;

            g_rf += gb00 * (r0 * r0 - one_third);
            g_rf += gb01 * (r0 * r1);
            g_rf += gb02 * (r0 * r2);
            g_rf += gb10 * (r1 * r0);
            g_rf += gb11 * (r1 * r1 - one_third);
            g_rf += gb12 * (r1 * r2);
            g_rf += gb20 * (r2 * r0);
            g_rf += gb21 * (r2 * r1);
            g_rf += gb22 * (r2 * r2 - one_third);
        }

        grad_rbf[ef] = g_rf;
    }

    grad_vec[vbase + 0] = g_r0 * inv;
    grad_vec[vbase + 1] = g_r1 * inv;
    grad_vec[vbase + 2] = g_r2 * inv;
    grad_d[e] = -((g_r0 * v0) + (g_r1 * v1) + (g_r2 * v2)) * inv2;
}

}  // namespace

std::vector<at::Tensor> geom_basis_forward_cuda(
    at::Tensor vec_ij,
    at::Tensor d_ij,
    at::Tensor rbf_feat,
    bool need_l1,
    bool need_l2) {
    const c10::cuda::CUDAGuard device_guard(vec_ij.device());

    auto rhat = at::empty_like(vec_ij);
    auto basis1 = need_l1
        ? at::empty({vec_ij.size(0), 3, rbf_feat.size(1)}, rbf_feat.options())
        : at::empty({0}, rbf_feat.options());
    auto basis2 = need_l2
        ? at::empty({vec_ij.size(0), 3, 3, rbf_feat.size(1)}, rbf_feat.options())
        : at::empty({0}, rbf_feat.options());

    const int threads = 256;
    const int64_t E = vec_ij.size(0);
    const int64_t F = rbf_feat.size(1);
    const int64_t EF = E * F;

    const int blocks_e = static_cast<int>((E + threads - 1) / threads);
    const int blocks_ef = static_cast<int>((EF + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(vec_ij.scalar_type(), "geom_basis_forward_cuda", [&] {
        compute_rhat_kernel<scalar_t><<<blocks_e, threads>>>(
            E,
            vec_ij.data_ptr<scalar_t>(),
            d_ij.data_ptr<scalar_t>(),
            rhat.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (need_l1) {
            compute_basis1_kernel<scalar_t><<<blocks_ef, threads>>>(
                EF,
                F,
                rhat.data_ptr<scalar_t>(),
                rbf_feat.data_ptr<scalar_t>(),
                basis1.data_ptr<scalar_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        if (need_l2) {
            compute_basis2_kernel<scalar_t><<<blocks_ef, threads>>>(
                EF,
                F,
                rhat.data_ptr<scalar_t>(),
                rbf_feat.data_ptr<scalar_t>(),
                basis2.data_ptr<scalar_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });

    return {rhat, basis1, basis2};
}

std::vector<at::Tensor> geom_basis_backward_cuda(
    at::Tensor grad_rhat,
    at::Tensor grad_basis1,
    at::Tensor grad_basis2,
    at::Tensor vec_ij,
    at::Tensor d_ij,
    at::Tensor rbf_feat,
    bool need_l1,
    bool need_l2) {
    const c10::cuda::CUDAGuard device_guard(vec_ij.device());

    auto grad_vec = at::zeros_like(vec_ij);
    auto grad_d = at::zeros_like(d_ij);
    auto grad_rbf = at::zeros_like(rbf_feat);

    const int threads = 256;
    const int64_t E = vec_ij.size(0);
    const int64_t F = rbf_feat.size(1);
    const int blocks_e = static_cast<int>((E + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(vec_ij.scalar_type(), "geom_basis_backward_cuda", [&] {
        geom_basis_backward_kernel<scalar_t><<<blocks_e, threads>>>(
            E,
            F,
            need_l1,
            need_l2,
            grad_rhat.data_ptr<scalar_t>(),
            need_l1 ? grad_basis1.data_ptr<scalar_t>() : nullptr,
            need_l2 ? grad_basis2.data_ptr<scalar_t>() : nullptr,
            vec_ij.data_ptr<scalar_t>(),
            d_ij.data_ptr<scalar_t>(),
            rbf_feat.data_ptr<scalar_t>(),
            grad_vec.data_ptr<scalar_t>(),
            grad_d.data_ptr<scalar_t>(),
            grad_rbf.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return {grad_vec, grad_d, grad_rbf};
}

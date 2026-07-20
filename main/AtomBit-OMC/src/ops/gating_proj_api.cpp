#include <torch/extension.h>
#include <vector>

torch::Tensor gating_proj_forward(
    torch::Tensor r_hat,
    torch::Tensor h_src,
    torch::Tensor h_dst,
    torch::Tensor scalar_basis);

std::vector<torch::Tensor> gating_proj_backward(
    torch::Tensor grad_out,
    torch::Tensor r_hat,
    torch::Tensor h_src,
    torch::Tensor h_dst,
    torch::Tensor scalar_basis);

namespace {

void check_inputs(
    const torch::Tensor& r_hat,
    const torch::Tensor& h_src,
    const torch::Tensor& h_dst,
    const torch::Tensor& scalar_basis) {

    TORCH_CHECK(r_hat.is_cuda(), "r_hat must be a CUDA tensor");
    TORCH_CHECK(h_src.is_cuda(), "h_src must be a CUDA tensor");
    TORCH_CHECK(h_dst.is_cuda(), "h_dst must be a CUDA tensor");
    TORCH_CHECK(scalar_basis.is_cuda(), "scalar_basis must be a CUDA tensor");

    TORCH_CHECK(r_hat.scalar_type() == torch::kFloat32, "r_hat must be float32");
    TORCH_CHECK(h_src.scalar_type() == torch::kFloat32, "h_src must be float32");
    TORCH_CHECK(h_dst.scalar_type() == torch::kFloat32, "h_dst must be float32");
    TORCH_CHECK(scalar_basis.scalar_type() == torch::kFloat32, "scalar_basis must be float32");

    TORCH_CHECK(r_hat.is_contiguous(), "r_hat must be contiguous");
    TORCH_CHECK(h_src.is_contiguous(), "h_src must be contiguous");
    TORCH_CHECK(h_dst.is_contiguous(), "h_dst must be contiguous");
    TORCH_CHECK(scalar_basis.is_contiguous(), "scalar_basis must be contiguous");

    TORCH_CHECK(r_hat.dim() == 2 && r_hat.size(1) == 3, "r_hat must have shape (E, 3)");
    TORCH_CHECK(h_src.dim() == 3 && h_src.size(1) == 3, "h_src must have shape (E, 3, F)");
    TORCH_CHECK(h_dst.dim() == 3 && h_dst.size(1) == 3, "h_dst must have shape (E, 3, F)");
    TORCH_CHECK(scalar_basis.dim() == 2, "scalar_basis must have shape (E, F)");

    const auto E = r_hat.size(0);
    const auto F = scalar_basis.size(1);

    TORCH_CHECK(h_src.size(0) == E, "h_src.size(0) must match r_hat.size(0)");
    TORCH_CHECK(h_dst.size(0) == E, "h_dst.size(0) must match r_hat.size(0)");
    TORCH_CHECK(scalar_basis.size(0) == E, "scalar_basis.size(0) must match r_hat.size(0)");
    TORCH_CHECK(h_src.size(2) == F, "h_src.size(2) must match scalar_basis.size(1)");
    TORCH_CHECK(h_dst.size(2) == F, "h_dst.size(2) must match scalar_basis.size(1)");
}

} // namespace

torch::Tensor gating_proj_forward_checked(
    torch::Tensor r_hat,
    torch::Tensor h_src,
    torch::Tensor h_dst,
    torch::Tensor scalar_basis) {

    check_inputs(r_hat, h_src, h_dst, scalar_basis);
    return gating_proj_forward(r_hat, h_src, h_dst, scalar_basis);
}

std::vector<torch::Tensor> gating_proj_backward_checked(
    torch::Tensor grad_out,
    torch::Tensor r_hat,
    torch::Tensor h_src,
    torch::Tensor h_dst,
    torch::Tensor scalar_basis) {

    check_inputs(r_hat, h_src, h_dst, scalar_basis);

    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be float32");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape (E, 3F)");
    TORCH_CHECK(grad_out.size(0) == r_hat.size(0), "grad_out.size(0) must match E");
    TORCH_CHECK(grad_out.size(1) == 3 * scalar_basis.size(1), "grad_out must have shape (E, 3F)");

    return gating_proj_backward(grad_out, r_hat, h_src, h_dst, scalar_basis);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gating_proj_forward", &gating_proj_forward_checked, "gating_proj forward (CUDA)");
    m.def("gating_proj_backward", &gating_proj_backward_checked, "gating_proj backward (CUDA, first-order only)");
}

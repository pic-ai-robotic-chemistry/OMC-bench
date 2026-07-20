#include <torch/extension.h>

#include <vector>

namespace {

inline void check_inputs(const at::Tensor& h_trans, const at::Tensor& geom) {
    TORCH_CHECK(h_trans.is_cuda(), "h_trans must be a CUDA tensor");
    TORCH_CHECK(geom.is_cuda(), "geom must be a CUDA tensor");
    TORCH_CHECK(h_trans.is_contiguous(), "h_trans must be contiguous");
    TORCH_CHECK(geom.is_contiguous(), "geom must be contiguous");
    TORCH_CHECK(h_trans.dim() == 3, "h_trans must have shape (E, 3, F)");
    TORCH_CHECK(geom.dim() == 3, "geom must have shape (E, 3, F)");
    TORCH_CHECK(h_trans.sizes() == geom.sizes(), "shape mismatch between h_trans and geom");
    TORCH_CHECK(h_trans.size(1) == 3, "expected h_trans.size(1) == 3");
    TORCH_CHECK(h_trans.scalar_type() == geom.scalar_type(), "dtype mismatch between h_trans and geom");
}

inline void check_grad_inputs(
    const at::Tensor& grad_out,
    const at::Tensor& h_trans,
    const at::Tensor& geom) {
    check_inputs(h_trans, geom);
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(grad_out.dim() == 4, "grad_out must have shape (E, 3, 3, F)");
    TORCH_CHECK(grad_out.size(0) == h_trans.size(0), "grad_out.size(0) mismatch");
    TORCH_CHECK(grad_out.size(1) == 3 && grad_out.size(2) == 3, "grad_out inner shape must be (3, 3)");
    TORCH_CHECK(grad_out.size(3) == h_trans.size(2), "grad_out.size(3) mismatch");
    TORCH_CHECK(grad_out.scalar_type() == h_trans.scalar_type(), "grad_out dtype mismatch");
}

}  // namespace

at::Tensor outer_sym_detrace_forward_cuda(const at::Tensor& h_trans, const at::Tensor& geom);
std::vector<at::Tensor> outer_sym_detrace_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& h_trans,
    const at::Tensor& geom);

at::Tensor outer_sym_detrace_forward(const at::Tensor& h_trans, const at::Tensor& geom) {
    check_inputs(h_trans, geom);
    return outer_sym_detrace_forward_cuda(h_trans, geom);
}

std::vector<at::Tensor> outer_sym_detrace_backward(
    const at::Tensor& grad_out,
    const at::Tensor& h_trans,
    const at::Tensor& geom) {
    check_grad_inputs(grad_out, h_trans, geom);
    return outer_sym_detrace_backward_cuda(grad_out, h_trans, geom);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("outer_sym_detrace_forward", &outer_sym_detrace_forward, "outer_sym_detrace forward (CUDA)");
    m.def("outer_sym_detrace_backward", &outer_sym_detrace_backward, "outer_sym_detrace backward (CUDA)");
}

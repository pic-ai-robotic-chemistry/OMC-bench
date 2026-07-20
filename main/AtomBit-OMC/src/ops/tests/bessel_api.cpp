#include <torch/extension.h>

torch::Tensor bessel_cuda_forward(torch::Tensor d, double r_max, torch::Tensor freq);
torch::Tensor bessel_cuda_backward(torch::Tensor grad_out, torch::Tensor d, double r_max, torch::Tensor freq);

static void check_inputs(const torch::Tensor& d, const torch::Tensor& freq) {
    TORCH_CHECK(d.is_cuda(), "d must be CUDA");
    TORCH_CHECK(freq.is_cuda(), "freq must be CUDA");
    TORCH_CHECK(d.dtype() == torch::kFloat32, "d must be float32");
    TORCH_CHECK(freq.dtype() == torch::kFloat32, "freq must be float32");
    TORCH_CHECK(d.is_contiguous(), "d must be contiguous");
    TORCH_CHECK(freq.is_contiguous(), "freq must be contiguous");
    TORCH_CHECK(d.dim() == 1, "d must be (E,)");
    TORCH_CHECK(freq.dim() == 1, "freq must be (F,)");
}

torch::Tensor bessel_forward_checked(torch::Tensor d, double r_max, torch::Tensor freq) {
    check_inputs(d, freq);
    return bessel_cuda_forward(d, r_max, freq);
}

torch::Tensor bessel_backward_checked(torch::Tensor grad_out, torch::Tensor d, double r_max, torch::Tensor freq) {
    check_inputs(d, freq);
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out must be float32");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be (E, F)");
    TORCH_CHECK(grad_out.size(0) == d.size(0), "E mismatch");
    TORCH_CHECK(grad_out.size(1) == freq.size(0), "F mismatch");
    return bessel_cuda_backward(grad_out, d, r_max, freq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bessel_forward", &bessel_forward_checked, "Bessel forward (CUDA)");
    m.def("bessel_backward", &bessel_backward_checked, "Bessel backward (CUDA)");
}

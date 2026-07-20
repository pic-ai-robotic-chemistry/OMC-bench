// mat_mul_sym_api.cpp
#include <torch/extension.h>
#include <vector>

// implemented in .cu
torch::Tensor mat_mul_sym_forward(torch::Tensor h, torch::Tensor geom);
std::vector<torch::Tensor> mat_mul_sym_backward(torch::Tensor grad_out, torch::Tensor h, torch::Tensor geom);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mat_mul_sym_forward", &mat_mul_sym_forward, "mat_mul_sym forward (CUDA)");
    m.def("mat_mul_sym_backward", &mat_mul_sym_backward, "mat_mul_sym backward (CUDA)");
}

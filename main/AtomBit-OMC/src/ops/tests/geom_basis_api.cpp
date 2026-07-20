
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> geom_basis_forward_cuda(
    torch::Tensor vec_ij,
    torch::Tensor d_ij,
    torch::Tensor rbf_feat,
    bool need_l1,
    bool need_l2);

std::vector<torch::Tensor> geom_basis_backward_cuda(
    torch::Tensor grad_rhat,
    torch::Tensor grad_basis1,
    torch::Tensor grad_basis2,
    torch::Tensor vec_ij,
    torch::Tensor d_ij,
    torch::Tensor rbf_feat,
    bool need_l1,
    bool need_l2);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

static void check_forward_inputs(
    const torch::Tensor& vec_ij,
    const torch::Tensor& d_ij,
    const torch::Tensor& rbf_feat) {
    CHECK_INPUT(vec_ij);
    CHECK_INPUT(d_ij);
    CHECK_INPUT(rbf_feat);

    TORCH_CHECK(vec_ij.dim() == 2 && vec_ij.size(1) == 3, "vec_ij must have shape (E, 3)");
    TORCH_CHECK(d_ij.dim() == 1, "d_ij must have shape (E,)");
    TORCH_CHECK(rbf_feat.dim() == 2, "rbf_feat must have shape (E, F)");

    TORCH_CHECK(vec_ij.size(0) == d_ij.size(0), "vec_ij and d_ij must share E");
    TORCH_CHECK(vec_ij.size(0) == rbf_feat.size(0), "vec_ij and rbf_feat must share E");

    TORCH_CHECK(vec_ij.scalar_type() == d_ij.scalar_type(), "vec_ij and d_ij dtypes must match");
    TORCH_CHECK(vec_ij.scalar_type() == rbf_feat.scalar_type(), "vec_ij and rbf_feat dtypes must match");
}

std::vector<torch::Tensor> geom_basis_forward(
    torch::Tensor vec_ij,
    torch::Tensor d_ij,
    torch::Tensor rbf_feat,
    bool need_l1,
    bool need_l2) {
    check_forward_inputs(vec_ij, d_ij, rbf_feat);
    return geom_basis_forward_cuda(vec_ij, d_ij, rbf_feat, need_l1, need_l2);
}

std::vector<torch::Tensor> geom_basis_backward(
    torch::Tensor grad_rhat,
    torch::Tensor grad_basis1,
    torch::Tensor grad_basis2,
    torch::Tensor vec_ij,
    torch::Tensor d_ij,
    torch::Tensor rbf_feat,
    bool need_l1,
    bool need_l2) {
    CHECK_INPUT(grad_rhat);
    CHECK_INPUT(vec_ij);
    CHECK_INPUT(d_ij);
    CHECK_INPUT(rbf_feat);

    if (grad_basis1.numel() > 0) {
        CHECK_INPUT(grad_basis1);
    }
    if (grad_basis2.numel() > 0) {
        CHECK_INPUT(grad_basis2);
    }

    check_forward_inputs(vec_ij, d_ij, rbf_feat);

    TORCH_CHECK(grad_rhat.sizes() == vec_ij.sizes(), "grad_rhat must have shape (E, 3)");

    if (need_l1) {
        TORCH_CHECK(
            grad_basis1.dim() == 3 &&
            grad_basis1.size(0) == vec_ij.size(0) &&
            grad_basis1.size(1) == 3 &&
            grad_basis1.size(2) == rbf_feat.size(1),
            "grad_basis1 must have shape (E, 3, F)"
        );
    }

    if (need_l2) {
        TORCH_CHECK(
            grad_basis2.dim() == 4 &&
            grad_basis2.size(0) == vec_ij.size(0) &&
            grad_basis2.size(1) == 3 &&
            grad_basis2.size(2) == 3 &&
            grad_basis2.size(3) == rbf_feat.size(1),
            "grad_basis2 must have shape (E, 3, 3, F)"
        );
    }

    return geom_basis_backward_cuda(
        grad_rhat,
        grad_basis1,
        grad_basis2,
        vec_ij,
        d_ij,
        rbf_feat,
        need_l1,
        need_l2
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("geom_basis_forward", &geom_basis_forward, "geom_basis forward (CUDA)");
    m.def("geom_basis_backward", &geom_basis_backward, "geom_basis backward (CUDA)");
}

import torch
from torch.utils.cpp_extension import load

cuda_module = load(
    name="gating_proj_cuda",
    sources=["gating_proj_api.cpp", "gating_proj_kernel.cu"],
    verbose=False,
)

def ref_gating_proj(r_hat, h_src, h_dst, scalar_basis):
    p_src = torch.einsum("ed,edf->ef", r_hat, h_src)
    p_dst = torch.einsum("ed,edf->ef", r_hat, h_dst)
    return torch.cat([scalar_basis, p_src, p_dst], dim=1)

class GatingProjFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_hat, h_src, h_dst, scalar_basis):
        out = cuda_module.gating_proj_forward(r_hat, h_src, h_dst, scalar_basis)
        ctx.save_for_backward(r_hat, h_src, h_dst, scalar_basis)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        r_hat, h_src, h_dst, scalar_basis = ctx.saved_tensors
        grad_r, grad_hs, grad_hd, grad_sb = cuda_module.gating_proj_backward(
            grad_out.contiguous(), r_hat, h_src, h_dst, scalar_basis
        )
        return grad_r, grad_hs, grad_hd, grad_sb

def gating_proj(r_hat, h_src, h_dst, scalar_basis):
    return GatingProjFn.apply(r_hat, h_src, h_dst, scalar_basis)

def max_abs(a, b):
    return (a - b).abs().max().item()

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    # IMPORTANT: This minimal CUDA backward assumes F is power-of-two and <=1024
    E, F = 2048, 128

    r_hat = torch.randn(E, 3, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    h_src = torch.randn(E, 3, F, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    h_dst = torch.randn(E, 3, F, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    s_basis = torch.randn(E, F, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()

    # ---- 1) forward compare ----
    with torch.no_grad():
        y_ref = ref_gating_proj(r_hat, h_src, h_dst, s_basis)
        y_ext = cuda_module.gating_proj_forward(r_hat, h_src, h_dst, s_basis)
        print("[forward] max|diff| =", max_abs(y_ref, y_ext))

    # ---- 2) backward compare ----
    # Use same upstream weight to make gradients comparable
    w = torch.randn(E, 3 * F, device="cuda", dtype=torch.float32)

    # ref grads
    y_ref = ref_gating_proj(r_hat, h_src, h_dst, s_basis)
    loss_ref = (y_ref * w).sum()
    grads_ref = torch.autograd.grad(loss_ref, [r_hat, h_src, h_dst, s_basis], retain_graph=False)

    # ext grads
    y_ext = gating_proj(r_hat, h_src, h_dst, s_basis)
    loss_ext = (y_ext * w).sum()
    grads_ext = torch.autograd.grad(loss_ext, [r_hat, h_src, h_dst, s_basis], retain_graph=False)

    print("[backward] max|dr|  =", max_abs(grads_ref[0], grads_ext[0]))
    print("[backward] max|dhs| =", max_abs(grads_ref[1], grads_ext[1]))
    print("[backward] max|dhd| =", max_abs(grads_ref[2], grads_ext[2]))
    print("[backward] max|dsb| =", max_abs(grads_ref[3], grads_ext[3]))

    # ---- 3) check loss.backward works and grads are populated ----
    r_hat2 = r_hat.detach().clone().requires_grad_(True)
    h_src2 = h_src.detach().clone().requires_grad_(True)
    h_dst2 = h_dst.detach().clone().requires_grad_(True)
    s_basis2 = s_basis.detach().clone().requires_grad_(True)

    y = gating_proj(r_hat2, h_src2, h_dst2, s_basis2)
    loss = y.pow(2).mean()
    loss.backward()
    print("[backward()] grads None?",
          r_hat2.grad is None, h_src2.grad is None, h_dst2.grad is None, s_basis2.grad is None)
    print("[backward()] grad norms:",
          r_hat2.grad.norm().item(), h_src2.grad.norm().item(), h_dst2.grad.norm().item(), s_basis2.grad.norm().item())

if __name__ == "__main__":
    main()

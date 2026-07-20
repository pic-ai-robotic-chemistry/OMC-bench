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
    w = torch.randn(E, 3 * F, device="cuda", dtype=torch.float32)

    y_ref = ref_gating_proj(r_hat, h_src, h_dst, s_basis)
    loss_ref = (y_ref * w).sum()
    grads_ref = torch.autograd.grad(loss_ref, [r_hat, h_src, h_dst, s_basis], retain_graph=False)

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

    # ---- 4) force-style 2nd-order / mixed-2nd test (create_graph=True) ----
    # 目标：模拟你的训练路径：
    #   E -> grad(E, input) (create_graph=True) -> loss(force) -> backward() -> param grad
    #
    # 做法：引入一个“参数” alpha，让输入依赖 alpha，然后看 force_loss.backward() 能不能把梯度传到 alpha。
    print("\n[2nd-order test] force-style mixed second derivative...")

    alpha = torch.randn((), device="cuda", dtype=torch.float32, requires_grad=True)

    # 让输入依赖 alpha（模拟网络参数影响能量，再通过力loss回传）
    r_hat3 = (r_hat.detach() * alpha).requires_grad_(True)
    h_src3 = (h_src.detach() * (1.0 + 0.1 * alpha)).requires_grad_(True)
    h_dst3 = (h_dst.detach() * (1.0 - 0.1 * alpha)).requires_grad_(True)
    s_basis3 = s_basis.detach().requires_grad_(True)

    out3 = gating_proj(r_hat3, h_src3, h_dst3, s_basis3)

    # 构造一个标量 energy（模拟 pred_e）
    energy = out3.sum()

    # 一次导：类似“力”（这里只对 r_hat3 求导来触发 create_graph）
    force_like = torch.autograd.grad(
        outputs=energy,
        inputs=r_hat3,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]

    # 用“力”构造 loss（模拟 force loss）
    force_loss = (force_like ** 2).mean()

    # 反传到 alpha（这一步需要 mixed second derivative）
    try:
        force_loss.backward()
        print("[2nd-order test] SUCCESS")
        print("  alpha.grad is None? ", alpha.grad is None)
        if alpha.grad is not None:
            print("  alpha.grad =", float(alpha.grad))
    except RuntimeError as e:
        print("[2nd-order test] FAILED with RuntimeError:")
        print(e)

def partial_second_order_compare():
    print("\n[partial 2nd-order compare] ref vs ext on alpha.grad ...")
    torch.manual_seed(0)

    E, F = 512, 128
    base_r = torch.randn(E, 3, device="cuda", dtype=torch.float32)
    base_hs = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
    base_hd = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
    base_sb = torch.randn(E, F, device="cuda", dtype=torch.float32)

    # 一个“参数”，专门控制 gating 分支
    alpha_ref = torch.tensor(0.7, device="cuda", dtype=torch.float32, requires_grad=True)
    alpha_ext = torch.tensor(0.7, device="cuda", dtype=torch.float32, requires_grad=True)

    def make_inputs(alpha):
        r = (base_r * alpha).clone().requires_grad_(True)
        hs = (base_hs * (1.0 + 0.1 * alpha)).clone().requires_grad_(True)
        hd = (base_hd * (1.0 - 0.1 * alpha)).clone().requires_grad_(True)
        sb = base_sb.clone().requires_grad_(False)
        return r, hs, hd, sb

    # 参考实现：完全 PyTorch，可二阶
    r_ref, hs_ref, hd_ref, sb_ref = make_inputs(alpha_ref)
    e_ref = ref_gating_proj(r_ref, hs_ref, hd_ref, sb_ref).sum()

    # 加一个普通可二阶分支，避免 ext 因“完全不可二阶”直接报错
    e_ref = e_ref + 0.01 * (r_ref ** 2).sum()

    f_ref = torch.autograd.grad(e_ref, r_ref, create_graph=True)[0]
    loss_ref = (f_ref ** 2).mean()
    g_alpha_ref = torch.autograd.grad(loss_ref, alpha_ref, allow_unused=True)[0]

    # 自定义扩展实现
    r_ext, hs_ext, hd_ext, sb_ext = make_inputs(alpha_ext)
    e_ext = gating_proj(r_ext, hs_ext, hd_ext, sb_ext).sum()
    e_ext = e_ext + 0.01 * (r_ext ** 2).sum()

    f_ext = torch.autograd.grad(e_ext, r_ext, create_graph=True)[0]
    loss_ext = (f_ext ** 2).mean()
    g_alpha_ext = torch.autograd.grad(loss_ext, alpha_ext, allow_unused=True)[0]

    print("f_ref.requires_grad:", f_ref.requires_grad)
    print("f_ext.requires_grad:", f_ext.requires_grad)
    print("alpha grad (ref):", None if g_alpha_ref is None else float(g_alpha_ref))
    print("alpha grad (ext):", None if g_alpha_ext is None else float(g_alpha_ext))
    if g_alpha_ref is not None and g_alpha_ext is not None:
        print("abs diff:", abs(float(g_alpha_ref - g_alpha_ext)))

if __name__ == "__main__":
    main()
    partial_second_order_compare()

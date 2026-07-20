import argparse
import math
import traceback
from typing import Callable, Dict, Tuple

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


class GatingProjFnCUDA(torch.autograd.Function):
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


class GatingProjFnTorchBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_hat, h_src, h_dst, scalar_basis):
        out = cuda_module.gating_proj_forward(r_hat, h_src, h_dst, scalar_basis)
        ctx.save_for_backward(r_hat, h_src, h_dst, scalar_basis)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        r_hat, h_src, h_dst, scalar_basis = ctx.saved_tensors
        F = scalar_basis.shape[1]

        g_s = grad_out[:, :F]
        g_ps = grad_out[:, F : 2 * F]
        g_pd = grad_out[:, 2 * F : 3 * F]

        grad_sb = g_s
        grad_hs = r_hat.unsqueeze(-1) * g_ps.unsqueeze(1)
        grad_hd = r_hat.unsqueeze(-1) * g_pd.unsqueeze(1)
        grad_r = (h_src * g_ps.unsqueeze(1)).sum(dim=2) + (h_dst * g_pd.unsqueeze(1)).sum(dim=2)
        return grad_r, grad_hs, grad_hd, grad_sb


def gating_proj_cuda_bwd(r_hat, h_src, h_dst, scalar_basis):
    return GatingProjFnCUDA.apply(r_hat, h_src, h_dst, scalar_basis)


def gating_proj_torch_bwd(r_hat, h_src, h_dst, scalar_basis):
    return GatingProjFnTorchBwd.apply(r_hat, h_src, h_dst, scalar_basis)


def max_abs(a, b):
    return (a - b).abs().max().item()


def clone_inputs(r_hat, h_src, h_dst, s_basis):
    return (
        r_hat.detach().clone().requires_grad_(True),
        h_src.detach().clone().requires_grad_(True),
        h_dst.detach().clone().requires_grad_(True),
        s_basis.detach().clone().requires_grad_(True),
    )


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check_first_order_correctness(E: int, F: int):
    print_header(f"1st-order correctness | E={E}, F={F}")
    torch.manual_seed(0)

    r_hat = torch.randn(E, 3, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    h_src = torch.randn(E, 3, F, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    h_dst = torch.randn(E, 3, F, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()
    s_basis = torch.randn(E, F, device="cuda", dtype=torch.float32, requires_grad=True).contiguous()

    with torch.no_grad():
        y_ref = ref_gating_proj(r_hat, h_src, h_dst, s_basis)
        y_cuda = gating_proj_cuda_bwd(r_hat, h_src, h_dst, s_basis)
        y_torchbwd = gating_proj_torch_bwd(r_hat, h_src, h_dst, s_basis)
        print("[forward] ref vs cuda_bwd    max|diff| =", max_abs(y_ref, y_cuda))
        print("[forward] ref vs torch_bwd   max|diff| =", max_abs(y_ref, y_torchbwd))

    w = torch.randn(E, 3 * F, device="cuda", dtype=torch.float32)

    # ref grads
    r0, hs0, hd0, sb0 = clone_inputs(r_hat, h_src, h_dst, s_basis)
    y0 = ref_gating_proj(r0, hs0, hd0, sb0)
    loss0 = (y0 * w).sum()
    grads_ref = torch.autograd.grad(loss0, [r0, hs0, hd0, sb0], retain_graph=False)

    # cuda-backward grads
    r1, hs1, hd1, sb1 = clone_inputs(r_hat, h_src, h_dst, s_basis)
    y1 = gating_proj_cuda_bwd(r1, hs1, hd1, sb1)
    loss1 = (y1 * w).sum()
    grads_cuda = torch.autograd.grad(loss1, [r1, hs1, hd1, sb1], retain_graph=False)

    # torch-backward grads
    r2, hs2, hd2, sb2 = clone_inputs(r_hat, h_src, h_dst, s_basis)
    y2 = gating_proj_torch_bwd(r2, hs2, hd2, sb2)
    loss2 = (y2 * w).sum()
    grads_torch = torch.autograd.grad(loss2, [r2, hs2, hd2, sb2], retain_graph=False)

    names = ["dr", "dhs", "dhd", "dsb"]
    for i, name in enumerate(names):
        print(f"[backward] ref vs cuda_bwd  max|d{name}| = {max_abs(grads_ref[i], grads_cuda[i]):.6e}")
        print(f"[backward] ref vs torch_bwd max|d{name}| = {max_abs(grads_ref[i], grads_torch[i]):.6e}")



def pure_second_order_case(fn: Callable, base_r, base_hs, base_hd, base_sb):
    alpha = torch.tensor(0.7, device="cuda", dtype=torch.float32, requires_grad=True)
    r = (base_r * alpha).clone().requires_grad_(True)
    hs = (base_hs * (1.0 + 0.1 * alpha)).clone().requires_grad_(True)
    hd = (base_hd * (1.0 - 0.1 * alpha)).clone().requires_grad_(True)
    sb = base_sb.clone().requires_grad_(False)

    out = fn(r, hs, hd, sb)
    energy = out.sum()
    force_like = torch.autograd.grad(energy, r, create_graph=True, retain_graph=True)[0]
    loss = (force_like ** 2).mean()
    grad_alpha = torch.autograd.grad(loss, alpha, allow_unused=True)[0]
    return force_like, grad_alpha



def mixed_second_order_case(fn: Callable, base_r, base_hs, base_hd, base_sb):
    alpha = torch.tensor(0.7, device="cuda", dtype=torch.float32, requires_grad=True)
    r = (base_r * alpha).clone().requires_grad_(True)
    hs = (base_hs * (1.0 + 0.1 * alpha)).clone().requires_grad_(True)
    hd = (base_hd * (1.0 - 0.1 * alpha)).clone().requires_grad_(True)
    sb = base_sb.clone().requires_grad_(False)

    out = fn(r, hs, hd, sb)
    energy = out.sum() + 0.01 * (r ** 2).sum()
    force_like = torch.autograd.grad(energy, r, create_graph=True, retain_graph=True)[0]
    loss = (force_like ** 2).mean()
    grad_alpha = torch.autograd.grad(loss, alpha, allow_unused=True)[0]
    return force_like, grad_alpha



def check_second_order(E: int, F: int):
    print_header(f"2nd-order behavior | E={E}, F={F}")
    torch.manual_seed(0)
    base_r = torch.randn(E, 3, device="cuda", dtype=torch.float32)
    base_hs = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
    base_hd = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
    base_sb = torch.randn(E, F, device="cuda", dtype=torch.float32)

    # Pure gating path
    for name, fn in [
        ("ref", ref_gating_proj),
        ("cuda_bwd", gating_proj_cuda_bwd),
        ("torch_bwd", gating_proj_torch_bwd),
    ]:
        try:
            force_like, grad_alpha = pure_second_order_case(fn, base_r, base_hs, base_hd, base_sb)
            print(f"[pure 2nd] {name:10s} success | force.requires_grad={force_like.requires_grad} | alpha.grad={None if grad_alpha is None else float(grad_alpha):}")
        except Exception as e:
            print(f"[pure 2nd] {name:10s} FAILED | {type(e).__name__}: {e}")

    # Mixed path: adds a small normal differentiable branch so you can detect silent loss of the gating contribution
    try:
        f_ref, g_ref = mixed_second_order_case(ref_gating_proj, base_r, base_hs, base_hd, base_sb)
        print(f"[mixed 2nd] ref        force.requires_grad={f_ref.requires_grad} | alpha.grad={float(g_ref)}")
    except Exception as e:
        print(f"[mixed 2nd] ref        FAILED | {type(e).__name__}: {e}")
        g_ref = None

    for name, fn in [
        ("cuda_bwd", gating_proj_cuda_bwd),
        ("torch_bwd", gating_proj_torch_bwd),
    ]:
        try:
            f_x, g_x = mixed_second_order_case(fn, base_r, base_hs, base_hd, base_sb)
            print(f"[mixed 2nd] {name:10s} force.requires_grad={f_x.requires_grad} | alpha.grad={float(g_x)}")
            if g_ref is not None and g_x is not None:
                print(f"            abs diff vs ref = {abs(float(g_x - g_ref)):.6e}")
        except Exception as e:
            print(f"[mixed 2nd] {name:10s} FAILED | {type(e).__name__}: {e}")



def zero_grads(tensors):
    for t in tensors:
        t.grad = None



def time_fwd_bwd(fn: Callable, r, hs, hd, sb, w, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        zero_grads([r, hs, hd, sb])
        out = fn(r, hs, hd, sb)
        loss = (out * w).sum()
        loss.backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        zero_grads([r, hs, hd, sb])
        out = fn(r, hs, hd, sb)
        loss = (out * w).sum()
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters



def time_force_style_2nd(fn: Callable, base_r, base_hs, base_hd, base_sb, alpha, iters: int, warmup: int) -> float:
    def _run_once():
        if alpha.grad is not None:
            alpha.grad = None
        r = (base_r * alpha).clone().requires_grad_(True)
        hs = (base_hs * (1.0 + 0.1 * alpha)).clone().requires_grad_(True)
        hd = (base_hd * (1.0 - 0.1 * alpha)).clone().requires_grad_(True)
        sb = base_sb.clone().requires_grad_(False)
        out = fn(r, hs, hd, sb)
        energy = out.sum()
        force_like = torch.autograd.grad(energy, r, create_graph=True, retain_graph=True)[0]
        loss = (force_like ** 2).mean()
        loss.backward()

    # warmup
    for _ in range(warmup):
        _run_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _run_once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters



def benchmark_first_order(E_list, F: int, iters: int, warmup: int):
    print_header(f"1st-order speed benchmark | F={F} | iters={iters} | warmup={warmup}")
    print(f"{'E':>10} | {'ref ms':>10} | {'cuda_bwd ms':>12} | {'torch_bwd ms':>13} | {'cuda spd':>9} | {'torch spd':>10}")
    print("-" * 90)

    for E in E_list:
        torch.manual_seed(0)
        base_r = torch.randn(E, 3, device="cuda", dtype=torch.float32)
        base_hs = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
        base_hd = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
        base_sb = torch.randn(E, F, device="cuda", dtype=torch.float32)
        w = torch.randn(E, 3 * F, device="cuda", dtype=torch.float32)

        r0, hs0, hd0, sb0 = clone_inputs(base_r, base_hs, base_hd, base_sb)
        r1, hs1, hd1, sb1 = clone_inputs(base_r, base_hs, base_hd, base_sb)
        r2, hs2, hd2, sb2 = clone_inputs(base_r, base_hs, base_hd, base_sb)

        t_ref = time_fwd_bwd(ref_gating_proj, r0, hs0, hd0, sb0, w, iters, warmup)
        t_cuda = time_fwd_bwd(gating_proj_cuda_bwd, r1, hs1, hd1, sb1, w, iters, warmup)
        t_torch = time_fwd_bwd(gating_proj_torch_bwd, r2, hs2, hd2, sb2, w, iters, warmup)

        print(f"{E:10d} | {t_ref:10.4f} | {t_cuda:12.4f} | {t_torch:13.4f} | {t_ref/t_cuda:9.2f} | {t_ref/t_torch:10.2f}")



def benchmark_second_order(E_list, F: int, iters: int, warmup: int):
    print_header(f"2nd-order speed benchmark (force-style) | F={F} | iters={iters} | warmup={warmup}")
    print("Only ref and torch_bwd are benchmarked here because cuda_bwd fails the pure 2nd-order test.")
    print(f"{'E':>10} | {'ref ms':>10} | {'torch_bwd ms':>13} | {'speedup':>10}")
    print("-" * 65)

    for E in E_list:
        torch.manual_seed(0)
        base_r = torch.randn(E, 3, device="cuda", dtype=torch.float32)
        base_hs = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
        base_hd = torch.randn(E, 3, F, device="cuda", dtype=torch.float32)
        base_sb = torch.randn(E, F, device="cuda", dtype=torch.float32)

        alpha0 = torch.tensor(0.7, device="cuda", dtype=torch.float32, requires_grad=True)
        alpha1 = torch.tensor(0.7, device="cuda", dtype=torch.float32, requires_grad=True)

        t_ref = time_force_style_2nd(ref_gating_proj, base_r, base_hs, base_hd, base_sb, alpha0, iters, warmup)
        t_torch = time_force_style_2nd(gating_proj_torch_bwd, base_r, base_hs, base_hd, base_sb, alpha1, iters, warmup)
        print(f"{E:10d} | {t_ref:10.4f} | {t_torch:13.4f} | {t_ref/t_torch:10.2f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--F", type=int, default=128)
    parser.add_argument("--E", type=int, default=2048)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--bench-warmup", type=int, default=20)
    parser.add_argument(
        "--bench-E-list",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--bench-E-list-2nd",
        type=int,
        nargs="*",
        default=[512, 1024, 2048, 4096],
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required to run this benchmark."
    assert args.F <= 1024, "Current minimal CUDA kernels assume F <= 1024."

    check_first_order_correctness(args.E, args.F)
    check_second_order(min(args.E, 1024), args.F)
    benchmark_first_order(args.bench_E_list, args.F, args.bench_iters, args.bench_warmup)
    benchmark_second_order(args.bench_E_list_2nd, args.F, max(10, args.bench_iters // 5), max(5, args.bench_warmup // 2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

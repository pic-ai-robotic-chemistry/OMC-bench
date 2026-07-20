import torch
from torch.utils.cpp_extension import load

# ---------- load extension ----------
cuda_module = load(
    name="gating_proj_cuda",
    sources=["gating_proj_api.cpp", "gating_proj_kernel.cu"],
    verbose=False,
)

# ---------- reference ----------
def ref_gating_proj(r_hat, h_src, h_dst, scalar_basis):
    p_src = torch.einsum("ed,edf->ef", r_hat, h_src)
    p_dst = torch.einsum("ed,edf->ef", r_hat, h_dst)
    return torch.cat([scalar_basis, p_src, p_dst], dim=1)

# ---------- extension + autograd wrapper ----------
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

def ext_gating_proj(r_hat, h_src, h_dst, scalar_basis):
    return GatingProjFn.apply(r_hat, h_src, h_dst, scalar_basis)

# ---------- timing helpers ----------
def _zero_grads(tensors):
    for t in tensors:
        t.grad = None

@torch.no_grad()
def _sanity_check(E, F):
    r = torch.randn(E, 3, device="cuda", dtype=torch.float32).contiguous()
    hs = torch.randn(E, 3, F, device="cuda", dtype=torch.float32).contiguous()
    hd = torch.randn(E, 3, F, device="cuda", dtype=torch.float32).contiguous()
    sb = torch.randn(E, F, device="cuda", dtype=torch.float32).contiguous()
    y_ref = ref_gating_proj(r, hs, hd, sb)
    y_ext = cuda_module.gating_proj_forward(r, hs, hd, sb)
    diff = (y_ref - y_ext).abs().max().item()
    if diff > 5e-5:
        print(f"WARNING: forward diff too large at E={E}, F={F}: {diff}")
    return diff

def time_fwd_bwd(fn, r_hat, h_src, h_dst, s_basis, w, iters=100, warmup=20):
    # warmup
    for _ in range(warmup):
        _zero_grads([r_hat, h_src, h_dst, s_basis])
        out = fn(r_hat, h_src, h_dst, s_basis)
        loss = (out * w).sum()
        loss.backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _zero_grads([r_hat, h_src, h_dst, s_basis])
        out = fn(r_hat, h_src, h_dst, s_basis)
        loss = (out * w).sum()
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms

def bench(E_list, F=128, iters=100, warmup=20, dtype=torch.float32):
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    print(f"Benchmark (forward+backward): dtype={dtype}, F={F}, iters={iters}, warmup={warmup}")
    print("NOTE: ext minimal backward assumes F is power-of-two and <=1024.")
    print("-" * 100)
    print(f"{'E':>10} | {'ref fwd+bwd (ms)':>18} | {'ext fwd+bwd (ms)':>18} | {'speedup':>10}")
    print("-" * 100)

    for E in E_list:
        # sanity check forward once
        _sanity_check(E=min(E, 4096), F=F)

        # create inputs (requires_grad=True)
        r_hat = torch.randn(E, 3, device="cuda", dtype=dtype, requires_grad=True).contiguous()
        h_src = torch.randn(E, 3, F, device="cuda", dtype=dtype, requires_grad=True).contiguous()
        h_dst = torch.randn(E, 3, F, device="cuda", dtype=dtype, requires_grad=True).contiguous()
        s_basis = torch.randn(E, F, device="cuda", dtype=dtype, requires_grad=True).contiguous()

        # fixed upstream weight to ensure comparable backward path (same loss form)
        w = torch.randn(E, 3 * F, device="cuda", dtype=dtype)

        ref_ms = time_fwd_bwd(ref_gating_proj, r_hat, h_src, h_dst, s_basis, w, iters=iters, warmup=warmup)
        ext_ms = time_fwd_bwd(ext_gating_proj, r_hat, h_src, h_dst, s_basis, w, iters=iters, warmup=warmup)

        speedup = ref_ms / ext_ms if ext_ms > 0 else float("inf")
        print(f"{E:>10} | {ref_ms:>18.4f} | {ext_ms:>18.4f} | {speedup:>9.2f}x")

    print("-" * 100)

if __name__ == "__main__":
    # 你可以按训练真实规模改 E
    E_list = [
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
    ]
    bench(E_list=E_list, F=128, iters=100, warmup=20)

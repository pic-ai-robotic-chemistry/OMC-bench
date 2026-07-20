import torch
from torch.utils.cpp_extension import load

# --- load extension ---
cuda_module = load(
    name="gating_proj_cuda",
    sources=["gating_proj_api.cpp", "gating_proj_kernel.cu"],
    verbose=False,
)

def ref_gating_proj(r_hat, h_src, h_dst, scalar_basis):
    # (E,3) x (E,3,F) -> (E,F)
    p_src = torch.einsum("ed,edf->ef", r_hat, h_src)
    p_dst = torch.einsum("ed,edf->ef", r_hat, h_dst)
    return torch.cat([scalar_basis, p_src, p_dst], dim=1)  # (E,3F)

def ext_gating_proj(r_hat, h_src, h_dst, scalar_basis):
    return cuda_module.gating_proj_forward(r_hat, h_src, h_dst, scalar_basis)

@torch.no_grad()
def time_forward(fn, inputs, iters=200, warmup=30):
    # warmup
    for _ in range(warmup):
        _ = fn(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn(*inputs)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms

def bench(E_list, F=128, iters=200, warmup=30, dtype=torch.float32):
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    print(f"Benchmark: dtype={dtype}, F={F}, iters={iters}, warmup={warmup}")
    print("-" * 90)
    print(f"{'E':>10} | {'ref (ms)':>12} | {'ext (ms)':>12} | {'speedup':>10}")
    print("-" * 90)

    for E in E_list:
        # IMPORTANT: your minimal kernel assumes F<=1024 and backward assumes power-of-two,
        # but this benchmark is forward-only so only F<=1024 matters.
        r_hat = torch.randn(E, 3, device="cuda", dtype=dtype).contiguous()
        h_src = torch.randn(E, 3, F, device="cuda", dtype=dtype).contiguous()
        h_dst = torch.randn(E, 3, F, device="cuda", dtype=dtype).contiguous()
        s_basis = torch.randn(E, F, device="cuda", dtype=dtype).contiguous()

        # quick correctness check once per E (optional)
        y_ref = ref_gating_proj(r_hat, h_src, h_dst, s_basis)
        y_ext = ext_gating_proj(r_hat, h_src, h_dst, s_basis)
        max_diff = (y_ref - y_ext).abs().max().item()
        if max_diff > 5e-5:
            print(f"WARNING: large diff at E={E}: {max_diff}")

        ref_ms = time_forward(ref_gating_proj, (r_hat, h_src, h_dst, s_basis), iters=iters, warmup=warmup)
        ext_ms = time_forward(ext_gating_proj, (r_hat, h_src, h_dst, s_basis), iters=iters, warmup=warmup)

        speedup = ref_ms / ext_ms if ext_ms > 0 else float("inf")
        print(f"{E:>10} | {ref_ms:>12.4f} | {ext_ms:>12.4f} | {speedup:>10.2f}x")

    print("-" * 90)

if __name__ == "__main__":
    # 你可以按需要改这些 E（从小到大）
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
    bench(E_list=E_list, F=128, iters=200, warmup=30)

import os
import torch

from ._extension_utils import load_prebuilt_or_jit

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
cuda_module = load_prebuilt_or_jit(
    module_name="src.ops._gating_proj_cuda",
    fallback_name="gating_proj_cuda_hybrid",
    sources=[
        os.path.join(_THIS_DIR, "gating_proj_api.cpp"),
        os.path.join(_THIS_DIR, "gating_proj_kernel.cu"),
    ],
)


class GatingProjFn(torch.autograd.Function):
    """
    Hybrid autograd behavior:
      - forward: CUDA fused kernel
      - ordinary first-order backward: CUDA fused kernel (fast)
      - higher-order / create_graph=True backward: torch-op backward (differentiable)

    The key switch is torch.is_grad_enabled() inside backward:
      * False -> no higher-order graph is being recorded, so use fast CUDA backward.
      * True  -> higher-order graph is being recorded, so use torch ops.
    """

    @staticmethod
    def forward(ctx, r_hat, h_src, h_dst, scalar_basis):
        r_hat = r_hat.contiguous()
        h_src = h_src.contiguous()
        h_dst = h_dst.contiguous()
        scalar_basis = scalar_basis.contiguous()

        out = cuda_module.gating_proj_forward(r_hat, h_src, h_dst, scalar_basis)
        ctx.save_for_backward(r_hat, h_src, h_dst, scalar_basis)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        r_hat, h_src, h_dst, scalar_basis = ctx.saved_tensors

        # Higher-order path: backward itself must be differentiable.
        if torch.is_grad_enabled():
            F = scalar_basis.shape[1]
            g_s = grad_out[:, :F]
            g_ps = grad_out[:, F:2 * F]
            g_pd = grad_out[:, 2 * F:3 * F]

            grad_sb = g_s
            grad_hs = r_hat.unsqueeze(-1) * g_ps.unsqueeze(1)
            grad_hd = r_hat.unsqueeze(-1) * g_pd.unsqueeze(1)
            grad_r = (h_src * g_ps.unsqueeze(1)).sum(dim=2) + (h_dst * g_pd.unsqueeze(1)).sum(dim=2)
            return grad_r, grad_hs, grad_hd, grad_sb

        # Ordinary first-order path: fastest option.
        grad_r, grad_hs, grad_hd, grad_sb = cuda_module.gating_proj_backward(
            grad_out.contiguous(),
            r_hat,
            h_src,
            h_dst,
            scalar_basis,
        )
        return grad_r, grad_hs, grad_hd, grad_sb


def gating_proj(r_hat, h_src, h_dst, scalar_basis):
    return GatingProjFn.apply(r_hat, h_src, h_dst, scalar_basis)

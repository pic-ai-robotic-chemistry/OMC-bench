import os
import torch

from ._extension_utils import load_prebuilt_or_jit

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
matmul_sym_cuda = load_prebuilt_or_jit(
    module_name="src.ops._mat_mul_sym_cuda",
    fallback_name="mat_mul_sym_cuda",
    sources=[
        os.path.join(_THIS_DIR, "mat_mul_sym_api.cpp"),
        os.path.join(_THIS_DIR, "mat_mul_sym_kernel.cu"),
    ],
)

def _torch_backward(grad_out: torch.Tensor, h: torch.Tensor, g: torch.Tensor):
    # grad_raw = sym(grad_out) - tr(grad_out)/3 I
    go_sym = 0.5 * (grad_out + grad_out.transpose(1, 2))  # (E,3,3,F)

    tr = go_sym[:, 0, 0, :] + go_sym[:, 1, 1, :] + go_sym[:, 2, 2, :]  # (E,F)
    t = tr / 3.0

    dr = go_sym.clone()
    dr[:, 0, 0, :] = dr[:, 0, 0, :] - t
    dr[:, 1, 1, :] = dr[:, 1, 1, :] - t
    dr[:, 2, 2, :] = dr[:, 2, 2, :] - t

    # batched matmul per-feature: treat (E*F) as batch
    E, _, _, F = h.shape
    dr_b = dr.permute(0, 3, 1, 2).reshape(-1, 3, 3)
    h_b  = h.permute(0, 3, 1, 2).reshape(-1, 3, 3)
    g_b  = g.permute(0, 3, 1, 2).reshape(-1, 3, 3)

    grad_h_b = torch.bmm(dr_b, g_b.transpose(1, 2))
    grad_g_b = torch.bmm(h_b.transpose(1, 2), dr_b)

    grad_h = grad_h_b.reshape(E, F, 3, 3).permute(0, 2, 3, 1).contiguous()
    grad_g = grad_g_b.reshape(E, F, 3, 3).permute(0, 2, 3, 1).contiguous()
    return grad_h, grad_g


class MatMulSymFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, geom):
        out = matmul_sym_cuda.mat_mul_sym_forward(h, geom)
        ctx.save_for_backward(h, geom)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        h, geom = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        # 🔥关键：create_graph=True 时，autograd 会在 backward 打开 grad mode
        # 这时必须走 torch backward，否则二阶会错（和你现在看到的一样）
        if torch.is_grad_enabled():
            grad_h, grad_g = _torch_backward(grad_out, h, geom)
        else:
            grad_h, grad_g = matmul_sym_cuda.mat_mul_sym_backward(grad_out, h, geom)

        return grad_h, grad_g


def mat_mul_sym(h: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    # 和 gating 一样：保证 contiguous，避免 stride/对齐问题
    return MatMulSymFn.apply(h.contiguous(), geom.contiguous())

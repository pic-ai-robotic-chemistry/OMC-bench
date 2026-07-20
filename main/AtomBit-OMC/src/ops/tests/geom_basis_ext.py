
import os
import warnings
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EPS = 1.0e-6


def _load_cuda_module():
    if not torch.cuda.is_available():
        return None

    sources = [
        os.path.join(_THIS_DIR, 'geom_basis_api.cpp'),
        os.path.join(_THIS_DIR, 'geom_basis_kernel.cu'),
    ]

    extra_cflags = ['-O3']
    extra_cuda_cflags = ['-O3', '-lineinfo', '--expt-relaxed-constexpr']

    # If the user pins a host compiler, honor it.
    host_cxx = os.environ.get('CUDAHOSTCXX') or os.environ.get('CXX')
    if host_cxx:
        extra_cuda_cflags.append(f'-ccbin={host_cxx}')

    try:
        return load(
            name='geom_basis_cuda_hybrid_v1',
            sources=sources,
            verbose=False,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
    except Exception as exc:
        warnings.warn(
            'Failed to build geom_basis CUDA extension; using torch fallback. '
            f'Build error was: {exc}'
        )
        return None


cuda_module = _load_cuda_module()


def _empty_like(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_empty((0,))


def geom_basis_reference(
    vec_ij: torch.Tensor,
    d_ij: torch.Tensor,
    rbf_feat: torch.Tensor,
    need_l1: bool = True,
    need_l2: bool = True,
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Pure torch reference for the geometric tail.'''
    inv = (d_ij + eps).reciprocal().unsqueeze(-1)
    r_hat = vec_ij * inv

    if need_l1:
        basis1 = r_hat.unsqueeze(-1) * rbf_feat.unsqueeze(1)
    else:
        basis1 = _empty_like(rbf_feat)

    if need_l2:
        trace_less = r_hat.unsqueeze(2) * r_hat.unsqueeze(1)
        eye = torch.eye(3, dtype=vec_ij.dtype, device=vec_ij.device).div_(3.0).view(1, 3, 3)
        trace_less = trace_less - eye
        basis2 = trace_less.unsqueeze(-1) * rbf_feat.unsqueeze(1).unsqueeze(1)
    else:
        basis2 = _empty_like(rbf_feat)

    return r_hat, basis1, basis2


def _geom_basis_backward_torch(
    grad_rhat: torch.Tensor,
    grad_basis1: torch.Tensor,
    grad_basis2: torch.Tensor,
    vec_ij: torch.Tensor,
    d_ij: torch.Tensor,
    rbf_feat: torch.Tensor,
    need_l1: bool,
    need_l2: bool,
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Differentiable backward used for higher-order grads.'''
    inv = (d_ij + eps).reciprocal()
    inv_col = inv.unsqueeze(-1)
    r_hat = vec_ij * inv_col

    grad_r = grad_rhat
    grad_rbf = torch.zeros_like(rbf_feat)

    if need_l1 and grad_basis1.numel() > 0:
        grad_r = grad_r + (grad_basis1 * rbf_feat.unsqueeze(1)).sum(dim=-1)
        grad_rbf = grad_rbf + (grad_basis1 * r_hat.unsqueeze(-1)).sum(dim=1)

    if need_l2 and grad_basis2.numel() > 0:
        gT = (grad_basis2 * rbf_feat.unsqueeze(1).unsqueeze(1)).sum(dim=-1)  # (E, 3, 3)
        grad_r = grad_r + torch.bmm(gT + gT.transpose(1, 2), r_hat.unsqueeze(-1)).squeeze(-1)

        eye = torch.eye(3, dtype=vec_ij.dtype, device=vec_ij.device).div_(3.0).view(1, 3, 3)
        trace_less = r_hat.unsqueeze(2) * r_hat.unsqueeze(1) - eye
        grad_rbf = grad_rbf + (grad_basis2 * trace_less.unsqueeze(-1)).sum(dim=(1, 2))

    grad_vec = grad_r * inv_col
    grad_d = -((grad_r * vec_ij).sum(dim=1)) * inv.square()
    return grad_vec, grad_d, grad_rbf


class GeomBasisTailFn(torch.autograd.Function):
    '''
    Hybrid autograd behavior:
      - forward: CUDA fused kernel if available
      - ordinary first-order backward: CUDA fused kernel if available
      - higher-order / create_graph=True backward: differentiable torch-op backward
    '''

    @staticmethod
    def forward(ctx, vec_ij, d_ij, rbf_feat, need_l1: bool, need_l2: bool):
        vec_ij = vec_ij.contiguous()
        d_ij = d_ij.contiguous()
        rbf_feat = rbf_feat.contiguous()

        use_cuda_kernel = (
            cuda_module is not None
            and vec_ij.is_cuda
            and d_ij.is_cuda
            and rbf_feat.is_cuda
        )

        if use_cuda_kernel:
            r_hat, basis1, basis2 = cuda_module.geom_basis_forward(
                vec_ij,
                d_ij,
                rbf_feat,
                bool(need_l1),
                bool(need_l2),
            )
        else:
            r_hat, basis1, basis2 = geom_basis_reference(
                vec_ij,
                d_ij,
                rbf_feat,
                need_l1=bool(need_l1),
                need_l2=bool(need_l2),
            )

        ctx.save_for_backward(vec_ij, d_ij, rbf_feat)
        ctx.need_l1 = bool(need_l1)
        ctx.need_l2 = bool(need_l2)
        ctx.use_cuda_kernel = bool(use_cuda_kernel)
        return r_hat, basis1, basis2

    @staticmethod
    def backward(ctx, grad_rhat, grad_basis1, grad_basis2):
        vec_ij, d_ij, rbf_feat = ctx.saved_tensors
        E = vec_ij.shape[0]
        F = rbf_feat.shape[1]

        if grad_rhat is None:
            grad_rhat = torch.zeros_like(vec_ij)
        else:
            grad_rhat = grad_rhat.contiguous()

        if ctx.need_l1:
            if grad_basis1 is None:
                grad_basis1 = rbf_feat.new_zeros((E, 3, F))
            else:
                grad_basis1 = grad_basis1.contiguous()
        else:
            grad_basis1 = rbf_feat.new_empty((0,))

        if ctx.need_l2:
            if grad_basis2 is None:
                grad_basis2 = rbf_feat.new_zeros((E, 3, 3, F))
            else:
                grad_basis2 = grad_basis2.contiguous()
        else:
            grad_basis2 = rbf_feat.new_empty((0,))

        # Higher-order path: backward itself must be differentiable.
        if torch.is_grad_enabled():
            grad_vec, grad_d, grad_rbf = _geom_basis_backward_torch(
                grad_rhat,
                grad_basis1,
                grad_basis2,
                vec_ij,
                d_ij,
                rbf_feat,
                ctx.need_l1,
                ctx.need_l2,
            )
            return grad_vec, grad_d, grad_rbf, None, None

        # Ordinary first-order path: fastest option.
        if ctx.use_cuda_kernel:
            grad_vec, grad_d, grad_rbf = cuda_module.geom_basis_backward(
                grad_rhat,
                grad_basis1,
                grad_basis2,
                vec_ij,
                d_ij,
                rbf_feat,
                ctx.need_l1,
                ctx.need_l2,
            )
        else:
            grad_vec, grad_d, grad_rbf = _geom_basis_backward_torch(
                grad_rhat,
                grad_basis1,
                grad_basis2,
                vec_ij,
                d_ij,
                rbf_feat,
                ctx.need_l1,
                ctx.need_l2,
            )

        return grad_vec, grad_d, grad_rbf, None, None


def geom_basis_tail(
    vec_ij: torch.Tensor,
    d_ij: torch.Tensor,
    rbf_feat: torch.Tensor,
    need_l1: bool = True,
    need_l2: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    r_hat, basis1, basis2 = GeomBasisTailFn.apply(vec_ij, d_ij, rbf_feat, need_l1, need_l2)
    return r_hat, (basis1 if need_l1 else None), (basis2 if need_l2 else None)

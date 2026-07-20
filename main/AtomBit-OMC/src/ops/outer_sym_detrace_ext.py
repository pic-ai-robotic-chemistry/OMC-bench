import os
import warnings
from typing import Optional, Tuple

import torch

from ._extension_utils import env_flag_is_true, load_prebuilt_or_jit

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREBUILT_MODULE = "src.ops._outer_sym_detrace_cuda"
_EXT_NAME = "outer_sym_detrace_cuda_ext"
_LOAD_ERROR: Optional[Exception] = None
cuda_module = None


def _maybe_load_extension():
    global cuda_module, _LOAD_ERROR
    if cuda_module is not None:
        return cuda_module
    if _LOAD_ERROR is not None:
        return None
    if not torch.cuda.is_available():
        _LOAD_ERROR = RuntimeError("CUDA is not available in this environment.")
        return None

    try:
        cuda_module = load_prebuilt_or_jit(
            module_name=_PREBUILT_MODULE,
            fallback_name=_EXT_NAME,
            sources=[
                os.path.join(_THIS_DIR, "outer_sym_detrace_api.cpp"),
                os.path.join(_THIS_DIR, "outer_sym_detrace_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
    except Exception as e:  # pragma: no cover
        if env_flag_is_true("ATOMBIT_DISABLE_OPS_JIT"):
            raise
        _LOAD_ERROR = e
        warnings.warn(
            f"Failed to build {_EXT_NAME}, falling back to torch ops. Error: {e}"
        )
        return None
    return cuda_module


def has_cuda_extension() -> bool:
    return _maybe_load_extension() is not None


def get_build_error() -> Optional[Exception]:
    _maybe_load_extension()
    return _LOAD_ERROR


def outer_sym_detrace_ref(h_trans: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    outer = h_trans.unsqueeze(2) * geom.unsqueeze(1)
    sym = 0.5 * (outer + outer.transpose(1, 2))
    trace = (h_trans * geom).sum(dim=1)
    t = trace / 3.0
    res = sym.clone()
    res[:, 0, 0, :].sub_(t)
    res[:, 1, 1, :].sub_(t)
    res[:, 2, 2, :].sub_(t)
    return res


def outer_sym_detrace_torch(h_trans: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    if h_trans.ndim != 3 or geom.ndim != 3:
        raise ValueError("h_trans and geom must have shape (E, 3, F)")
    if h_trans.shape != geom.shape:
        raise ValueError(f"shape mismatch: {h_trans.shape} vs {geom.shape}")
    if h_trans.shape[1] != 3:
        raise ValueError(f"expected shape (E, 3, F), got {h_trans.shape}")

    h0, h1, h2 = h_trans[:, 0, :], h_trans[:, 1, :], h_trans[:, 2, :]
    g0, g1, g2 = geom[:, 0, :], geom[:, 1, :], geom[:, 2, :]

    t = (h0 * g0 + h1 * g1 + h2 * g2) / 3.0
    out = torch.empty(
        (h_trans.shape[0], 3, 3, h_trans.shape[2]),
        device=h_trans.device,
        dtype=h_trans.dtype,
    )

    o01 = 0.5 * (h0 * g1 + h1 * g0)
    o02 = 0.5 * (h0 * g2 + h2 * g0)
    o12 = 0.5 * (h1 * g2 + h2 * g1)

    out[:, 0, 0, :] = h0 * g0 - t
    out[:, 0, 1, :] = o01
    out[:, 0, 2, :] = o02
    out[:, 1, 0, :] = o01
    out[:, 1, 1, :] = h1 * g1 - t
    out[:, 1, 2, :] = o12
    out[:, 2, 0, :] = o02
    out[:, 2, 1, :] = o12
    out[:, 2, 2, :] = h2 * g2 - t
    return out


# grad_out can be non-symmetric, so we must average mirrored entries.
def outer_sym_detrace_backward_torch(
    grad_out: torch.Tensor,
    h_trans: torch.Tensor,
    geom: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g00 = grad_out[:, 0, 0, :]
    g01 = grad_out[:, 0, 1, :]
    g02 = grad_out[:, 0, 2, :]
    g10 = grad_out[:, 1, 0, :]
    g11 = grad_out[:, 1, 1, :]
    g12 = grad_out[:, 1, 2, :]
    g20 = grad_out[:, 2, 0, :]
    g21 = grad_out[:, 2, 1, :]
    g22 = grad_out[:, 2, 2, :]

    h0, h1, h2 = h_trans[:, 0, :], h_trans[:, 1, :], h_trans[:, 2, :]
    x0, x1, x2 = geom[:, 0, :], geom[:, 1, :], geom[:, 2, :]

    diag_avg = (g00 + g11 + g22) / 3.0
    s01 = 0.5 * (g01 + g10)
    s02 = 0.5 * (g02 + g20)
    s12 = 0.5 * (g12 + g21)

    grad_h = torch.empty_like(h_trans)
    grad_x = torch.empty_like(geom)

    grad_h[:, 0, :] = g00 * x0 + s01 * x1 + s02 * x2 - diag_avg * x0
    grad_h[:, 1, :] = s01 * x0 + g11 * x1 + s12 * x2 - diag_avg * x1
    grad_h[:, 2, :] = s02 * x0 + s12 * x1 + g22 * x2 - diag_avg * x2

    grad_x[:, 0, :] = g00 * h0 + s01 * h1 + s02 * h2 - diag_avg * h0
    grad_x[:, 1, :] = s01 * h0 + g11 * h1 + s12 * h2 - diag_avg * h1
    grad_x[:, 2, :] = s02 * h0 + s12 * h1 + g22 * h2 - diag_avg * h2
    return grad_h, grad_x


class OuterSymDetraceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_trans: torch.Tensor, geom: torch.Tensor, min_work: int):
        h_trans = h_trans.contiguous()
        geom = geom.contiguous()
        ctx.save_for_backward(h_trans, geom)
        ctx.min_work = int(min_work)

        mod = _maybe_load_extension()
        work = h_trans.shape[0] * h_trans.shape[2]
        use_cuda = (
            mod is not None
            and h_trans.is_cuda
            and geom.is_cuda
            and work >= ctx.min_work
        )
        ctx.used_cuda_kernel = bool(use_cuda)
        if use_cuda:
            return mod.outer_sym_detrace_forward(h_trans, geom)
        return outer_sym_detrace_torch(h_trans, geom)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h_trans, geom = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        # If higher-order graph recording is on, backward itself must be torch-differentiable.
        if torch.is_grad_enabled():
            grad_h, grad_geom = outer_sym_detrace_backward_torch(grad_out, h_trans, geom)
            return grad_h, grad_geom, None

        mod = _maybe_load_extension()
        if (
            ctx.used_cuda_kernel
            and mod is not None
            and grad_out.is_cuda
            and h_trans.is_cuda
            and geom.is_cuda
        ):
            grad_h, grad_geom = mod.outer_sym_detrace_backward(grad_out, h_trans, geom)
            return grad_h, grad_geom, None

        grad_h, grad_geom = outer_sym_detrace_backward_torch(grad_out, h_trans, geom)
        return grad_h, grad_geom, None


def outer_sym_detrace(
    h_trans: torch.Tensor,
    geom: torch.Tensor,
    min_work: Optional[int] = None,
) -> torch.Tensor:
    if min_work is None:
        min_work = int(os.environ.get("OUTER_SYM_DETRACE_MIN_WORK", "0"))
    return OuterSymDetraceFn.apply(h_trans, geom, int(min_work))

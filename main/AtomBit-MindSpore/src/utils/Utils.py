"""
HTGP utilities: config dataclass, scatter_add, scatter_mean.

Config controls model size, ablation flags (L0/L1/L2, gating, long-range),
and Leibniz coupling paths. Scatter ops are MindSpore-compatible.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import mindspore as ms
from mindspore import mint, ops


@dataclass
class HTGPConfig:
    """
    HTGP model and training configuration.

    Controls hidden dim, cutoffs, RBF, atom types, L0/L1/L2/gating/long-range
    flags, finetune vs pretrain, and active Leibniz coupling paths.
    """

    # Model size
    hidden_dim: int = 128
    num_layers: int = 2
    cutoff: float = 6.0
    num_rbf: int = 12
    atom_types_map: list = field(
        default_factory=lambda: [x for x in range(1, 101)]
    )

    # Module flags
    use_L0: bool = True
    use_L1: bool = True
    use_L2: bool = True
    use_gating: bool = True
    avg_neighborhood: float = 89
    use_long_range: bool = False
    use_charge: bool = False
    use_vdw: bool = False
    use_dipole: bool = False

    # Finetune / training
    FINETUNE_MODE: bool = True
    PRETRAINED_CKPT: str = "Checkpoints_Old/model_epoch_50.pt"
    steps_per_epoch: Optional[int] = None
    long_range_scale: float = 1

    # Leibniz coupling: (node_L_in, edge_L_in, target_L_out, op_type) -> enabled
    active_paths: Dict[Tuple[int, int, int, str], bool] = field(
        default_factory=lambda: {
            (0, 0, 0, "prod"): True,
            (0, 1, 1, "prod"): True,
            (0, 2, 2, "prod"): True,
            (1, 0, 1, "prod"): True,
            (1, 1, 0, "dot"): True,
            (1, 1, 1, "cross"): True,
            (1, 1, 2, "outer"): True,
            (2, 0, 2, "prod"): True,
            (2, 1, 1, "mat_vec"): True,
            (1, 2, 1, "vec_mat"): True,
            (2, 2, 0, "double_dot"): True,
            (2, 2, 2, "mat_mul_sym"): True,
        }
    )


def scatter_add(
    src: ms.Tensor,
    index: ms.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> ms.Tensor:
    """
    Scatter-add: sum src into out at indices along dim.

    Fast path when dim==0 and index is 1D (typical GNN case). Pass dim_size
    when known to avoid sync.

    Args:
        src: Source tensor [E, ...].
        index: Index tensor (same shape as src along dim, or 1D for dim 0).
        dim: Dimension along which to aggregate.
        dim_size: Output size along dim (optional; avoids index.max() sync).

    Returns:
        Tensor of shape (..., dim_size, ...) with scattered sums.
    """
    if dim_size is None:
        if index.numel() == 0:
            d_size = 0
        else:
            d_size = int(index.max()) + 1
    else:
        d_size = dim_size

    out_size = list(src.shape)
    out_size[dim] = d_size
    out = mint.zeros(out_size, dtype=src.dtype)

    if dim == 0 and index.dim() == 1:
        return out.index_add_(0, index, src)

    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index_expand = index.view(view_shape).expand_as(src)
        return out.scatter_add_(dim, index_expand, src)

    return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: ms.Tensor,
    index: ms.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> ms.Tensor:
    """
    Scatter-mean: mean of src grouped by index along dim.

    Implemented as scatter_add(src) / count(index). Zero-count bins yield 0.

    Args:
        src: Source tensor.
        index: Index tensor.
        dim: Aggregation dimension.
        dim_size: Output size along dim (optional).

    Returns:
        Tensor of shape (..., dim_size, ...) with scattered means.
    """
    out = scatter_add(src, index, dim, dim_size)
    d_size = out.shape[dim]

    ones = mint.ones(index.shape, dtype=src.dtype)
    count = scatter_add(ones, index, dim=0, dim_size=d_size)
    count = ops.clamp(count, min=1.0)

    if src.dim() > 1:
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        count = count.view(view_shape)

    return out / count

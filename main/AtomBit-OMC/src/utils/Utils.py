from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field, fields
import torch
from typing import Optional
import torch.nn.functional as F
import numpy as np

# Central runtime defaults for dtype/device.
DEFAULT_FLOAT_DTYPE = torch.float32
DEFAULT_NP_FLOAT_DTYPE = np.float32
DEFAULT_DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DEVICE = torch.device(DEFAULT_DEVICE_STR)


def build_active_path_preset(name: str) -> Dict[Tuple[int, int, int, str], bool]:
    full = {
        (0, 0, 0, "prod"): True,
        (0, 1, 1, "prod"): True,
        (0, 2, 2, "prod"): True,
        (1, 0, 1, "prod"): True,
        (1, 1, 0, "dot"): True,
        (1, 1, 1, "cross"): False,
        (1, 1, 2, "outer"): True,
        (2, 0, 2, "prod"): True,
        (2, 1, 1, "mat_vec"): True,
        (1, 2, 1, "vec_mat"): True,
        (2, 2, 0, "double_dot"): True,
        (2, 2, 2, "mat_mul_sym"): True,
        (1, 2, 2, "vec_cross_tensor"): False,
        (2, 1, 2, "tensor_cross_vector"): False,
        (2, 2, 1, "tensor_commutator"): False,
    }
    presets = {
        "full": full,
        "no_tensor": {
            **full,
            (0, 2, 2, "prod"): False,
            (1, 1, 2, "outer"): False,
            (2, 0, 2, "prod"): False,
            (2, 1, 1, "mat_vec"): False,
            (1, 2, 1, "vec_mat"): False,
            (2, 2, 0, "double_dot"): False,
            (2, 2, 2, "mat_mul_sym"): False,
            (1, 2, 2, "vec_cross_tensor"): False,
            (2, 1, 2, "tensor_cross_vector"): False,
            (2, 2, 1, "tensor_commutator"): False,
        },
        "scalar_vector_only": {
            **full,
            (0, 2, 2, "prod"): False,
            (1, 1, 2, "outer"): False,
            (2, 0, 2, "prod"): False,
            (2, 1, 1, "mat_vec"): False,
            (1, 2, 1, "vec_mat"): False,
            (2, 2, 0, "double_dot"): False,
            (2, 2, 2, "mat_mul_sym"): False,
            (1, 2, 2, "vec_cross_tensor"): False,
            (2, 1, 2, "tensor_cross_vector"): False,
            (2, 2, 1, "tensor_commutator"): False,
            (1, 1, 1, "cross"): True,
        },
        "minimal": {
            **full,
            (0, 1, 1, "prod"): False,
            (0, 2, 2, "prod"): False,
            (1, 0, 1, "prod"): False,
            (1, 1, 0, "dot"): False,
            (1, 1, 1, "cross"): False,
            (1, 1, 2, "outer"): False,
            (2, 0, 2, "prod"): False,
            (2, 1, 1, "mat_vec"): False,
            (1, 2, 1, "vec_mat"): False,
            (2, 2, 0, "double_dot"): False,
            (2, 2, 2, "mat_mul_sym"): False,
            (1, 2, 2, "vec_cross_tensor"): False,
            (2, 1, 2, "tensor_cross_vector"): False,
            (2, 2, 1, "tensor_commutator"): False,
        },
    }
    try:
        return dict(presets[name])
    except KeyError as exc:
        raise ValueError(f"Unknown active_path_preset '{name}'.") from exc

# ==========================================
# 1. 配置与消融控制 (Configuration & Ablation)
# ==========================================
@dataclass
class AtomBitConfig:
    # --- 基础超参 ---
    num_atom_types: int = 60
    hidden_dim: int = 128
    num_layers: int = 2
    cutoff: float = 6.0
    num_rbf: int = 32
    # Explicit atomic-number map. When left empty, models fall back to
    # ``range(1, num_atom_types + 1)``.
    atom_types_map: list = field(default_factory=list)
    # --- 模块开关 (Global Flags) ---
    use_L1: bool = True   # 矢量通道 (偶极/力)
    use_L2: bool = True   # 张量通道 (四极/应力)
    use_gating: bool = True      # 是否开启物理投影门控 (Physics Gating)
    gating_layer_indices: Optional[Tuple[int, ...]] = None
    active_path_preset: str = "full"
    block_impls: Dict[str, str] = field(
        default_factory=lambda: {
            "norm_l1": "equivariant_norm",
            "norm_l2": "equivariant_norm",
            "coupling": "leibniz",
            "gating": "physics_gating",
            "density": "cartesian_density",
            "readout": "mlp_readout",
        }
    )
    avg_neighborhood: float = 89
    # mat_mul_sym implementation: 1 (PyTorch k-loop) or 4 (CUDA ext)
    mat_mul_sym_impl: int = 1
    # outer implementation: 1 (PyTorch) or 4 (CUDA ext)
    outer_impl: int = 1
    # gating implementation: 1 (原始 PyTorch) or 4 (CUDA ext)
    gating_impl: int = 1

    # Legacy training fields kept for old checkpoint compatibility.
    # The current train entrypoint (`train.py` / `src.engine.train_runner`) does
    # not use these fields to decide restart/finetune behavior. Those decisions
    # now come from `configs/train/*.py`.
    FINETUNE_MODE: bool = True
    PRETRAINED_CKPT: str = "Checkpoints_Old/model_epoch_50.pt"
    # This field is still read by the trainer to optionally cap steps per epoch.
    # Keep it on the model config because older checkpoints may already store it.
    steps_per_epoch: Optional[int] = None

    # Legacy-compatible runtime/model behavior flag. The new train config also
    # exposes `use_direct_force`, but we keep this field so old checkpoints load.
    use_direct_force: bool = False

    # --- 莱布尼茨耦合路径字典 (Coupling Matrix) ---
    # 格式: (Node_L_in, Edge_L_in, Target_L_out, Operation_Type)
    # 通过设置 True/False 精确控制每一条物理路径
    active_paths: Dict[Tuple[int, int, int, str], bool] = field(default_factory=dict)

    def __post_init__(self):
        if self.use_L2 and not self.use_L1:
            raise ValueError("AtomBitConfig requires use_L1=True when use_L2=True.")
        if self.gating_layer_indices is not None:
            self.gating_layer_indices = tuple(int(i) for i in self.gating_layer_indices)
            invalid = [i for i in self.gating_layer_indices if i < 0 or i >= self.num_layers]
            if invalid:
                raise ValueError(
                    "gating_layer_indices must be valid zero-based layer indices; "
                    f"got {invalid} for num_layers={self.num_layers}."
                )
        preset_paths = build_active_path_preset(self.active_path_preset)
        if not self.active_paths:
            self.active_paths = preset_paths
        else:
            self.active_paths = {**preset_paths, **self.active_paths}


# # ==========================================
# # 0. 自定义 scatter_add (替代 torch_scatter)
# # ==========================================
# def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
#     """
#     自定义 scatter_add 实现，无需安装 torch_scatter 库。
#     利用 torch.index_add_ 实现高性能聚合。
#     """
#     if dim_size is None:
#         if index.numel() == 0:
#             dim_size = 0
#         else:
#             dim_size = int(index.max().item()) + 1
#     else:
#         dim_size = int(dim_size)

#     # 构建输出张量
#     out_size = list(src.size())
#     out_size[dim] = dim_size
#     out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

#     # 针对 GNN 最常用的 dim=0 且 index 为 1D 的情况进行优化
#     # index_add_ 会自动处理 src 后面的维度 (e.g. [E, 3, F] -> [N, 3, F])
#     if dim == 0 and index.dim() == 1:
#         return out.index_add_(0, index, src)

#     # 通用路径 (处理非 dim=0 或 index 为多维的情况)
#     if index.dim() != src.dim():
#         view_shape = [1] * src.dim()
#         view_shape[dim] = -1
#         index = index.view(view_shape).expand_as(src)

#     return out.scatter_add_(dim, index, src)

# def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
#     # 热路径专用：dim=0, index=1D
#     out = src.new_zeros((dim_size, *src.shape[1:]))
#     return out.index_add_(0, index, src)



# def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
#     """
#     自定义 scatter_add 实现 (Python 版)

#     Args:
#         src: 源数据 [E, F]
#         index: 索引 [E]
#         dim: 聚合维度 (通常为 0)
#         dim_size: 目标节点数 (必须传入以获得最佳性能)
#     """
#     # 1. 确定 dim_size
#     if dim_size is None:
#         if index.numel() == 0:
#             d_size = 0
#         else:
#             # ⚠️ 注意: 这里会有 CPU-GPU 同步，尽量在外部传入 dim_size
#             d_size = int(index.max()) + 1
#     else:
#         d_size = dim_size

#     # 2. 构建输出张量
#     out_size = list(src.size())
#     out_size[dim] = d_size
#     out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

#     # 3. 极速优化 path (GNN 常用)
#     if dim == 0 and index.dim() == 1:
#         return out.index_add_(0, index, src)

#     # 4. 通用路径
#     if index.dim() != src.dim():
#         view_shape = [1] * src.dim()
#         view_shape[dim] = -1
#         index_expand = index.view(view_shape).expand_as(src)
#         return out.scatter_add_(dim, index_expand, src)

#     return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """
    针对 MLIP 优化的原生 scatter_add 实现

    Args:
        src: 源数据 [E, F], [E, 3, F] 或 [E, 3, 3, F]
        index: 索引 [E]
        dim: 聚合维度 (通常为 0)
        dim_size: 目标节点数
    """
    # 1. 确定 dim_size (保持原逻辑不变)
    if dim_size is None:
        if index.numel() == 0:
            d_size = 0
        else:
            # ⚠️ 注意: 这里会有 CPU-GPU 同步
            d_size = int(index.max()) + 1
    else:
        d_size = dim_size

    # 2. 构建输出张量
    out_size = list(src.size())
    out_size[dim] = d_size
    out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

    # 3. 统一使用原生 scatter_add_ 路径
    # 删除了原来的 index_add_ 分支，因为在 L1/L2 且涉及求导时，它比原生慢 5-10 倍
    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index_expand = index.view(view_shape).expand_as(src)
        return out.scatter_add_(dim, index_expand, src)

    return out.scatter_add_(dim, index, src)


def sanitize_model_config_dict(raw: dict) -> dict:
    valid_names = {item.name for item in fields(AtomBitConfig)}
    return {key: value for key, value in raw.items() if key in valid_names}


HTGPConfig = AtomBitConfig


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """
    自定义 scatter_mean 实现 (基于 scatter_add)
    原理: out = sum(src) / count(index)

    Args:
        src: 源数据 [E, F]
        index: 索引 [E]
        dim: 聚合维度 (通常为 0)
        dim_size: 目标节点数
    """
    # 1. 计算分子: Sum
    # 直接调用你现有的 scatter_add
    out = scatter_add(src, index, dim, dim_size)

    # 获取实际的输出大小 (避免在 count 步骤再次进行 CPU-GPU 同步)
    d_size = out.size(dim)

    # 2. 计算分母: Count
    # 创建一个与 index 形状相同的全 1 张量
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    # 聚合全 1 张量得到每个索引的出现次数
    # 注意: index 是 1D 的，所以这里 dim 始终传 0
    count = scatter_add(ones, index, dim=0, dim_size=d_size)

    # 3. 数值稳定性处理
    # 将计数为 0 的位置设为 1，防止除以 0 产生 NaN (这些位置分子也是 0，结果应为 0)
    count.clamp_(min=1.0)

    # 4. 广播处理 (Broadcasting)
    # 如果 src 是 [N, F] 而 count 是 [N]，直接除会报错
    # 我们需要把 count 变成 [N, 1, ..., 1] 以便广播
    if src.dim() > 1:
        # 构建视图形状: [1, 1, ..., -1, ..., 1]
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        count = count.view(view_shape)

    # 5. 执行除法
    return out / count

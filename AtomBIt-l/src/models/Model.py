from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import (
    GeometricBasis, LeibnizCoupling, PhysicsGating, CartesianDensityBlock, LatentLongRange)
from src.utils import scatter_add, HTGPConfig

# ==========================================
# 7. 主模型 (Main Model)
# ==========================================
class HTGPModel(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        
        # ============================================================
        # 🔥 修改 1: 构建原子序数映射表 (Z-Mapper)
        # ============================================================
        # 优先从 config 获取原子列表，如果没有则使用默认的常用有机元素列表
        # 对应: H(1), B(5), C(6), N(7), O(8), F(9), P(15), S(16), Cl(17), Br(35), I(53)
        if hasattr(config, 'atom_types_map'):
            self.used_atomic_numbers = config.atom_types_map
        else:
            self.used_atomic_numbers = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
            
        num_actual_types = len(self.used_atomic_numbers) # 通常为 11
        max_z = max(self.used_atomic_numbers)            # 通常为 53

        # 注册映射表 buffer (会自动转到 GPU，但不更新梯度)
        # 初始化为 -1，方便后续检查非法原子
        self.register_buffer('z_mapper', torch.full((max_z + 1,), -1, dtype=torch.long))
        
        # 填充映射: z -> idx (例如 53 -> 10)
        for idx, z in enumerate(self.used_atomic_numbers):
            self.z_mapper[z] = idx

        # ============================================================
        # 🔥 修改 2: Embedding 尺寸缩小
        # ============================================================
        # Embedding: 只分配 11 行参数，而不是 60 行
        self.embedding = nn.Embedding(num_actual_types, config.hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        # Components (保持不变)
        self.geom_basis = GeometricBasis(config)
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                'coupling': LeibnizCoupling(config),
                'gating': PhysicsGating(config),
                'density': CartesianDensityBlock(config),
                'readout': nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, 1)
                )
            }))
            
        if config.use_long_range:
            self.long_range = LatentLongRange(config)
            
        # Atomic Ref: 同样缩小尺寸
        self.atomic_ref = nn.Embedding(num_actual_types, 1)
        nn.init.zeros_(self.atomic_ref.weight)
            
    def forward(self, data, capture_weights=False, capture_descriptors=False):
        if capture_descriptors:
            self.all_layer_descriptors = []
        
        # ============================================================
        # 🔥 修改 3: Forward 中应用映射
        # ============================================================
        # 获取原始原子序数 (N,)
        z_raw = data.z
        
        # 转换为稠密索引 (N,) -> [0, 2, 10, ...]
        z_idx = self.z_mapper[z_raw]
        
        # (可选) 安全检查: 如果数据里混入了未定义的原子 (如 Fe=26)，这里会是 -1
        # if (z_idx == -1).any():
        #    raise ValueError(f"Input contains undefined atomic numbers! Supported: {self.used_atomic_numbers}")

        # 1. 几何计算
        row, col = data.edge_index
        # 处理 shifts_int (PBC)
        if hasattr(data, 'shifts_int') and data.shifts_int is not None:
            batch_cell = data.cell[data.batch[row]]          # (E, 3, 3)
            current_shifts = torch.bmm(
                data.shifts_int.unsqueeze(1), batch_cell
            ).squeeze(1)                                     # (E, 3)
        else:
            current_shifts = torch.zeros(
                (row.size(0), 3),
                device=data.pos.device,
                dtype=data.pos.dtype
            )

        vec_ij = data.pos[col] - data.pos[row] + current_shifts
        d_ij = torch.norm(vec_ij, dim=-1).clamp(min=1e-8)

        basis_edges, r_hat = self.geom_basis(vec_ij, d_ij)

        # 2. 状态初始化 (使用 z_idx)
        h0 = self.embedding(z_idx) # (N, F) -> 使用映射后的索引
        h1 = None 
        h2 = None
        
        total_energy = 0.0
        
        # 3. 层级传递
        for layer in self.layers:
            # A. 莱布尼茨消息生成
            node_feats = {0: h0, 1: h1, 2: h2}
            raw_msgs = layer['coupling'](node_feats, basis_edges, data.edge_index)
            
            # B. 物理门控
            gated_msgs = layer['gating'](raw_msgs, h0, basis_edges[0], r_hat, h1, data.edge_index, capture_weights=capture_weights)
            
            # C. 密度聚合与更新
            # 注意: 这里 data.z.size(0) 是节点总数，保持不变
            delta_h0, delta_h1, delta_h2 = layer['density'](gated_msgs, row, data.z.size(0))

            # D. 残差更新 (Residual Update)
            h0 = h0 + delta_h0

            if self.cfg.use_L1:
                if h1 is None:
                    h1 = delta_h1 # 第一层直接赋值
                elif delta_h1 is not None:
                    h1 = h1 + delta_h1 # 后续层累加

            if self.cfg.use_L2:
                if h2 is None:
                    h2 = delta_h2
                elif delta_h2 is not None:
                    h2 = h2 + delta_h2

            # h0 h1 h2保存
            if capture_descriptors:
                current_layer_feats = {
                    'h0': h0.detach().cpu(), # ⚠️ 必须 detach 并转到 cpu，否则显存爆炸
                }
                if self.cfg.use_L1 and h1 is not None:
                    current_layer_feats['h1'] = h1.detach().cpu()
                if self.cfg.use_L2 and h2 is not None:
                    current_layer_feats['h2'] = h2.detach().cpu()
                
                self.all_layer_descriptors.append(current_layer_feats)

            # E. 能量读出
            atomic_energy = layer['readout'](h0)
            total_energy = total_energy + scatter_add(atomic_energy, data.batch, dim=0, dim_size=data.num_graphs)
            
        # 4. 长程修正
        if self.cfg.use_long_range and self.cfg.use_L1 and h1 is not None:
            e_long = self.long_range(h1, data.pos, data.batch)
            total_energy = total_energy + e_long
            
        # Atomic Ref (使用 z_idx)
        total_energy = total_energy + scatter_add(self.atomic_ref(z_idx), data.batch, dim=0, dim_size=data.num_graphs)

        return total_energy

    # ============================================================
    # 🔥 新增: 外部 E0 加载辅助函数 (供 train.py 调用)
    # ============================================================
    def load_external_e0(self, e0_dict, device=None, verbose=True, rank = 0):
        """
        从字典加载 E0，自动处理原子序数到内部索引的映射。
        """
        if device is None:
            device = self.atomic_ref.weight.device
            
        count = 0
        with torch.no_grad():
            # 将 mapper 转到 CPU 以便快速查表 (Python int loop)
            mapper_cpu = self.z_mapper.cpu()
            
            for z, e in e0_dict.items():
                z_raw = int(z)
                # 检查 z 是否在 mapper 范围内
                if z_raw < len(mapper_cpu):
                    mapped_idx = mapper_cpu[z_raw].item()
                    # 如果映射有效 (!= -1)
                    if mapped_idx != -1:
                        val = torch.tensor(e, dtype=torch.float32, device=device)
                        self.atomic_ref.weight[mapped_idx] = val
                        count += 1
                        
        # 冻结参数，不参与训练
        self.atomic_ref.weight.requires_grad = False
        if verbose and rank == 0:
            print(f"🔒 [Model Internal] Injected E0 for {count} elements.")
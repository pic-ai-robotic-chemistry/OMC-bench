import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple, List
from src.utils import scatter_add, scatter_mean, HTGPConfig

# ==========================================
# 🔥 核心 JIT 数学引擎 (安全加速区)
# ==========================================

@torch.jit.script
def compute_bessel_math(d: torch.Tensor, r_max: float, freq: torch.Tensor) -> torch.Tensor:
    d_scaled = d / r_max
    prefactor = (2.0 / r_max) ** 0.5
    return prefactor * torch.sin(freq * d_scaled) / (d + 1e-6)
 
@torch.jit.script
def compute_envelope_math(d: torch.Tensor, r_cut: float) -> torch.Tensor:
    x = d / r_cut
    x = torch.clamp(x, min=0.0, max=1.0)
    return 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5

@torch.jit.script
def compute_l2_basis(rbf_feat: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    outer = r_hat.unsqueeze(2) * r_hat.unsqueeze(1) 
    eye = torch.eye(3, dtype=r_hat.dtype, device=r_hat.device).unsqueeze(0)
    trace_less = outer - (1.0/3.0) * eye
    return rbf_feat.unsqueeze(1).unsqueeze(1) * trace_less.unsqueeze(-1)

@torch.jit.script
def compute_invariants(den0: Optional[torch.Tensor], 
                       den1: Optional[torch.Tensor], 
                       den2: Optional[torch.Tensor]) -> torch.Tensor:
    # ✅ 修复：使用标准类型标注
    invariants: List[torch.Tensor] = []
    
    if den0 is not None:
        invariants.append(den0)
        
    if den1 is not None:
        sq_sum = torch.sum(den1.pow(2), dim=1) 
        norm = torch.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
        
    if den2 is not None:
        sq_sum = torch.sum(den2.pow(2), dim=(1, 2))
        norm = torch.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
        
    if len(invariants) > 0:
        return torch.cat(invariants, dim=-1)
    else:
        # 返回空 Tensor (注意处理 device 问题，最好由外部保证 invariants 不为空)
        return torch.zeros(0) 

@torch.jit.script
def compute_gating_projections(h_node1: torch.Tensor, 
                               r_hat: torch.Tensor, 
                               scalar_basis: torch.Tensor,
                               src: torch.Tensor, 
                               dst: torch.Tensor) -> torch.Tensor:
    r_hat_uns = r_hat.unsqueeze(-1)
    p_src = torch.sum(h_node1[src] * r_hat_uns, dim=1)
    p_dst = torch.sum(h_node1[dst] * r_hat_uns, dim=1)
    return torch.cat([scalar_basis, p_src, p_dst], dim=-1)


# ==========================================
# 🧩 模块定义 (普通 nn.Module 区)
# ==========================================

class BesselBasis(nn.Module): 
    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        self.register_buffer("freq", torch.arange(1, num_basis + 1).float() * np.pi)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return compute_bessel_math(d, self.r_max, self.freq)

class PolynomialEnvelope(nn.Module):
    def __init__(self, r_cut: float, p: int = 5):
        super().__init__()
        self.r_cutoff = float(r_cut)
        self.p = int(p)
    
    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        return compute_envelope_math(d_ij, self.r_cutoff)

class GeometricBasis(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, vec_ij, d_ij):
        raw_rbf = self.rbf_mlp(self.rbf(d_ij.unsqueeze(-1)))
        env = self.envelope(d_ij)
        rbf_feat = raw_rbf * env.unsqueeze(-1)

        # ⚠️ r_hat 计算必须在 Python 层保留，确保梯度传导
        r_hat = vec_ij / (d_ij.unsqueeze(-1) + 1e-6)
        
        basis = {}
        basis[0] = rbf_feat
        
        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = rbf_feat.unsqueeze(1) * r_hat.unsqueeze(-1)
            
        if self.cfg.use_L2:
            basis[2] = compute_l2_basis(rbf_feat, r_hat)
            
        return basis, r_hat

class LeibnizCoupling(nn.Module): 
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.path_weights = nn.ModuleDict()
        
        for path_key, active in config.active_paths.items():
            if not active: continue
            l_in, l_edge, l_out, _ = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2: continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1: continue
                
            name = f"{l_in}_{l_edge}_{l_out}_{path_key[3]}"
            self.path_weights[name] = nn.Linear(self.F, self.F, bias=False)

        self.inv_sqrt_f = self.F ** -0.5

    def forward(self, h_nodes: Dict[int, torch.Tensor], basis_edges: Dict[int, torch.Tensor], edge_index):
        src, _ = edge_index
        messages: Dict[int, List[torch.Tensor]] = {0: [], 1: [], 2: []}
        
        for path_key, active in self.cfg.active_paths.items():
            if not active: continue
            l_in, l_edge, l_out, op_type = path_key
            
            if basis_edges.get(l_edge) is None: continue
            
            layer_name = f"{l_in}_{l_edge}_{l_out}_{op_type}"
            if layer_name not in self.path_weights: continue
            
            if h_nodes.get(l_in) is None: continue 
            else: inp = h_nodes[l_in]
            
            h_src = inp[src]
            h_trans = self.path_weights[layer_name](h_src)
            geom = basis_edges[l_edge]
            res = None
            
            # --- Operation Logic ---
            if op_type == 'prod':
                if l_in == 0 and l_edge == 0: res = h_trans * geom
                elif l_in == 0 and l_edge == 1: res = h_trans.unsqueeze(1) * geom
                elif l_in == 0 and l_edge == 2: res = h_trans.unsqueeze(1).unsqueeze(1) * geom
                elif l_in == 1 and l_edge == 0: res = h_trans * geom.unsqueeze(1)
                elif l_in == 2 and l_edge == 0: res = h_trans * geom.unsqueeze(1).unsqueeze(1)
            elif op_type == 'dot':
                res = torch.sum(h_trans * geom, dim=1)
            elif op_type == 'cross':
                g = geom
                if g.dim() == 2: g = g.unsqueeze(-1)
                res = torch.linalg.cross(h_trans, g, dim=1)
            elif op_type == 'outer':
                outer = h_trans.unsqueeze(2) * geom.unsqueeze(1)
                trace = torch.einsum('eiif->ef', outer)
                eye = torch.eye(3, device=outer.device).view(1, 3, 3, 1)
                res = outer - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye
            elif op_type == 'mat_vec':
                res = torch.einsum('eijf, ejf -> eif', h_trans, geom)
            elif op_type == 'vec_mat':
                res = torch.einsum('eif, eijf -> ejf', h_trans, geom)
            elif op_type == 'double_dot':
                res = torch.sum(h_trans * geom, dim=(1, 2))
            elif op_type == 'mat_mul_sym':
                raw = torch.einsum('eikf, ekjf -> eijf', h_trans, geom)
                sym = 0.5 * (raw + raw.transpose(1, 2))
                trace = torch.einsum('eiif->ef', sym)
                eye = torch.eye(3, device=sym.device).view(1, 3, 3, 1)
                res = sym - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye

            if res is not None:
                messages[l_out].append(res * self.inv_sqrt_f)
                
        final_msgs: Dict[int, Optional[torch.Tensor]] = {}
        for l in [0, 1, 2]:
            final_msgs[l] = sum(messages[l]) if len(messages[l]) > 0 else None
        return final_msgs

class PhysicsGating(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        self.W_query = nn.Linear(self.F, self.F, bias=False)
        self.W_key = nn.Linear(self.F, self.F, bias=False)
        
        self.phys_bias_mlp = nn.Sequential(
            nn.Linear(3 * self.F, self.F), 
            nn.SiLU(),            
            nn.Linear(self.F, 3 * self.F) 
        )
        self.channel_mixer = nn.Linear(self.F, 3 * self.F, bias=False)
        self.gate_scale = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, msgs, h_node0, scalar_basis, r_hat, h_node1, edge_index, capture_weights=False):
        if not self.cfg.use_gating: return msgs
        
        src, dst = edge_index
        
        if h_node1 is not None:
            phys_input = compute_gating_projections(h_node1, r_hat, scalar_basis, src, dst)
            split_idx = scalar_basis.shape[-1]
            p_ij = phys_input[:, split_idx:]        
        else:
            p_ij = torch.zeros((scalar_basis.shape[0], 2 * self.F), device=scalar_basis.device)
            phys_input = torch.cat([scalar_basis, p_ij], dim=-1)

        q = self.W_query(h_node0[dst]) 
        k = self.W_key(h_node0[src])   
        chem_score = q * k             
        chem_logits = self.channel_mixer(chem_score)
        phys_logits = self.phys_bias_mlp(phys_input)
        
        raw_gates = chem_logits + phys_logits
        gates = torch.sigmoid(raw_gates) * self.gate_scale
        
        if capture_weights: self.scalar_basis_captured = scalar_basis.detach()
        if capture_weights: self.p_ij_captured = p_ij.detach()
        if capture_weights: self.chem_logits_captured = chem_logits.detach()
        if capture_weights: self.phys_logits_captured = phys_logits.detach()

        g_list = torch.split(gates, self.F, dim=-1)
        g0, g1, g2 = [g.contiguous() for g in g_list]

        if capture_weights: self.g0_captured = g0.detach()
        if capture_weights: self.g1_captured = g1.detach()
        if capture_weights: self.g2_captured = g2.detach()
        
        out_msgs: Dict[int, torch.Tensor] = {}
        if 0 in msgs and msgs[0] is not None: out_msgs[0] = msgs[0] * g0
        if 1 in msgs and msgs[1] is not None: out_msgs[1] = msgs[1] * g1.unsqueeze(1)
        if 2 in msgs and msgs[2] is not None: out_msgs[2] = msgs[2] * g2.unsqueeze(1).unsqueeze(1)
            
        return out_msgs

class CartesianDensityBlock(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        self.cfg = config
        
        in_dim = 0
        if config.use_L0: in_dim += self.F
        if config.use_L1: in_dim += self.F
        if config.use_L2: in_dim += self.F 
        
        self.scalar_update_mlp = nn.Sequential(
            nn.Linear(in_dim, self.F),
            nn.SiLU(),
            nn.Linear(self.F, self.F)
        )

        if config.use_L1: self.L1_linear = nn.Linear(self.F, self.F, bias=False)
        if config.use_L2: self.L2_linear = nn.Linear(self.F, self.F, bias=False)

        scale_out_dim = 0
        if config.use_L1: scale_out_dim += self.F
        if config.use_L2: scale_out_dim += self.F
        
        if scale_out_dim > 0:
            self.scale_mlp = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, scale_out_dim)
            )
        else:
            self.scale_mlp = None
 
        self.inv_sqrt_deg = 1.0 / (config.avg_neighborhood ** 0.5)

    def forward(self, msgs: Dict[int, torch.Tensor], index: torch.Tensor, num_nodes: int):
        # 1. 密度聚合
        # ✅ 修正：标准类型标注，明确 None
        densities: Dict[int, Optional[torch.Tensor]] = {}
        densities[0], densities[1], densities[2] = None, None, None

        for l in [0, 1, 2]:
            if l in msgs and msgs[l] is not None:
                agg = scatter_add(msgs[l], index, dim=0, dim_size=num_nodes)
                densities[l] = agg * self.inv_sqrt_deg 
            else:
                densities[l] = None

        # 2. 提取不变量
        concat = compute_invariants(densities[0], densities[1], densities[2])

        # 3. 标量更新
        # ✅ 修正：使用 index.device 避免歧义报错
        if concat.numel() > 0:
            delta_h0 = self.scalar_update_mlp(concat)
        else:
            delta_h0 = torch.zeros((num_nodes, self.F), device=index.device)

        # 4. 矢量更新
        delta_h1 = None
        delta_h2 = None

        if self.scale_mlp is not None:
            scales = self.scale_mlp(delta_h0)
            curr_dim = 0
            
            if self.cfg.use_L1 and densities[1] is not None:
                alpha1 = scales[:, curr_dim : curr_dim + self.F] 
                h1_mixed = self.L1_linear(densities[1])
                delta_h1 = h1_mixed * alpha1.unsqueeze(1)
                curr_dim += self.F
                
            if self.cfg.use_L2 and densities[2] is not None:
                alpha2 = scales[:, curr_dim : curr_dim + self.F]
                h2_mixed = self.L2_linear(densities[2])
                delta_h2 = h2_mixed * alpha2.unsqueeze(1).unsqueeze(1)

        return delta_h0, delta_h1, delta_h2

# ==========================================
# 6. 长程场 (Latent Long Range) - Ablation Ready
# ==========================================
class LatentLongRange(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        # 物理常数: Coulomb constant in eV * A
        self.KE = 14.3996 
        
        # --- 1. 电荷预测网络 (h0 -> q) ---
        if config.use_charge:
            self.q_proj = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 1, bias=False) # 无偏置，确保空特征输出0电荷
            )
            # 可学习的高斯分布宽度 sigma，初始值设为 1.0 Å
            # 这决定了长程和短程的"交接点"
            self.sigma = nn.Parameter(torch.tensor(1.0))

        # --- 2. 范德华参数预测 (h0 -> C6, Rvdw) ---
        if config.use_vdw:
            self.vdw_proj = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 2) # 输出 [C6系数, 范德华半径]
            )

        # --- 3. 偶极矩预测 (h1 -> mu) ---
        if config.use_dipole:
            self.mu_proj = nn.Linear(self.F, 1, bias=False)

    def forward(self, h0, h1, pos, batch):
        """
        输入:
            h0: (N, F) 标量特征
            h1: (N, 3, F) 矢量特征
            pos: (N, 3) 原子坐标
            batch: (N,) 批次索引
        """
        energy_total = 0.0
        
        # ---------------------------------------------------------
        # 构建全连接几何图 (O(N^2))
        # ---------------------------------------------------------
        # 1. 计算所有原子对的坐标差 (N, N, 3)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0) 
        
        # 2. 计算距离平方 (N, N)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        # 3. 计算距离 (N, N)，加 epsilon 防止除零梯度爆炸
        dist = torch.sqrt(dist_sq + 1e-8)
        
        # 4. 构建 Mask: 
        # batch_mask: 只有同 batch 的原子才计算
        # diag_mask: 排除自己和自己计算 (对角线)
        batch_mask = (batch.unsqueeze(1) == batch.unsqueeze(0))
        diag_mask = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool)
        valid_mask = batch_mask & (~diag_mask)

        # 预计算倒数，减少除法次数
        inv_dist = 1.0 / dist
        
        # ---------------------------------------------------------
        # 模块 1: 静电力 (Electrostatics with erf Screening)
        # 公式: E = k * q_i * q_j / r * erf(r / (sqrt(2)*sigma))
        # ---------------------------------------------------------
        if self.cfg.use_charge:
            # 预测电荷
            q = self.q_proj(h0) # (N, 1)
            
            # [物理约束] 强制电荷中性: 每个分子的总电荷归零
            batch_q_mean = scatter_mean(q, batch, dim=0)
            q = q - batch_q_mean[batch]

            # 电荷乘积 q_i * q_j (N, N)
            qq = q @ q.t()
            
            # 计算屏蔽因子 erf
            # 这里的 math.sqrt(2) 源自高斯积分的标准形式
            scaled_r = dist / (math.sqrt(2) * self.sigma)
            shielding = torch.erf(scaled_r)
            
            # 组合公式
            # valid_mask 确保不计算不同分子间和自相互作用
            E_coul = torch.sum(qq * inv_dist * shielding * valid_mask)
            
            # 乘以 0.5 (避免 i-j 和 j-i 重复计算) 和 库仑常数
            energy_total += 0.5 * self.KE * E_coul

        # ---------------------------------------------------------
        # 模块 2: 范德华力 (VdW with Becke-Johnson Damping)
        # 公式: E = - C6 / (r^6 + f(R_vdw)^6)
        # ---------------------------------------------------------
        if self.cfg.use_vdw:
            # 预测参数，使用 Softplus 确保为正数
            vdw_params = self.vdw_proj(h0)
            c6 = F.softplus(vdw_params[:, 0:1])      # (N, 1)
            r_vdw = F.softplus(vdw_params[:, 1:2])   # (N, 1)
            
            # 组合规则 (Combination Rules)
            # C6_ij = sqrt(C6_i * C6_j)
            c6_ij = torch.sqrt(c6 @ c6.t())
            # R_vdw_ij = sqrt(R_i * R_j)
            r_vdw_ij = torch.sqrt(r_vdw @ r_vdw.t())
            
            # 计算 r^6
            dist6 = dist_sq ** 3
            
            # 构造 BJ 阻尼分母
            # 这里的逻辑是：当 r 很小时，分母趋向于 r_vdw^6 (常数)，避免无穷大
            # 当 r 很大时，分母趋向于 r^6，恢复标准范德华衰减
            damping = dist6 + (r_vdw_ij ** 6)
            
            # 计算能量 (注意符号是负的，吸引力)
            E_vdw = -torch.sum((c6_ij / damping) * valid_mask)
            
            energy_total += 0.5 * E_vdw

        # ---------------------------------------------------------
        # 模块 3: 偶极矩相互作用 (Dipole-Dipole)
        # ---------------------------------------------------------
        if self.cfg.use_dipole and h1 is not None:
            # h1 形状 (N, 3, F) -> 投影 -> (N, 3)
            mu = self.mu_proj(h1).squeeze(-1)
            
            # 计算 mu_i . mu_j
            mu_dot_mu = mu @ mu.t() # (N, N)
            
            # 计算方向向量 n_ij = r_ij / r
            n_ij = diff * inv_dist.unsqueeze(-1) # (N, N, 3)
            
            # 计算 (mu_i . n_ij)
            # (N, 1, 3) * (N, N, 3) -> sum -> (N, N)
            mu_dot_n_i = torch.sum(mu.unsqueeze(1) * n_ij, dim=-1)
            
            # 计算 (mu_j . n_ij)
            # 注意: n_ji = -n_ij, 所以 mu_j . n_ij = - (mu_j . n_ji)
            # 利用矩阵转置性质: A_ij = mu_i . n_ij, 那么 A_ji = mu_j . n_ji
            # 所以 mu_dot_n_j = - mu_dot_n_i.t()
            mu_dot_n_j = -mu_dot_n_i.t()
            
            # 组合项: (mu_i.mu_j) - 3(mu_i.n)(mu_j.n)
            angular_term = mu_dot_mu - 3 * mu_dot_n_i * mu_dot_n_j
            
            # 径向项: 1 / r^3
            # 同样需要 erf 屏蔽防止短程发散 (LES 理论同样适用偶极)
            # 使用 erf(x)^3 是一种常见的偶极屏蔽近似
            r_scaled = dist / self.sigma
            shielding_dip = torch.erf(r_scaled) ** 3
            radial_term = (inv_dist ** 3) * shielding_dip
            
            E_dip = torch.sum(angular_term * radial_term * valid_mask)
            energy_total += 0.5 * self.KE * E_dip

        # 返回总能量，乘以此处的缩放系数可以让训练初期更稳定
        return energy_total * self.cfg.long_range_scale
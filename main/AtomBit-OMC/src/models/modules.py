from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.utils import DEFAULT_FLOAT_DTYPE, AtomBitConfig, scatter_add


def _runtime_device_dtype(*tensors, fallback_dtype=DEFAULT_FLOAT_DTYPE):
    for tensor in tensors:
        if tensor is not None:
            return tensor.device, tensor.dtype
    return torch.device("cpu"), fallback_dtype


class BesselBasis(nn.Module):
    def __init__(self, r_max: float, num_basis: int = 8) -> None:
        super().__init__()
        self.register_buffer("r_max", torch.tensor(r_max, dtype=DEFAULT_FLOAT_DTYPE))
        self.register_buffer("prefactor", torch.tensor((2.0 / r_max) ** 0.5, dtype=DEFAULT_FLOAT_DTYPE))
        self.register_buffer("freq", torch.arange(1, num_basis + 1, dtype=DEFAULT_FLOAT_DTYPE) * torch.pi)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d_scaled = d / self.r_max
        return self.prefactor * torch.sin(self.freq * d_scaled) / (d + 1e-6)


class PolynomialEnvelope(nn.Module):
    def __init__(self, r_max: float, p: int = 5) -> None:
        super().__init__()
        self.register_buffer("inv_r_max", torch.tensor(1.0 / r_max, dtype=DEFAULT_FLOAT_DTYPE))
        self.p = int(p)

    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(d_ij * self.inv_r_max, min=0.0, max=1.0)
        x2 = x * x
        x3 = x2 * x
        return 1.0 - x3 * (10.0 - 15.0 * x + 6.0 * x2)


class GeometricBasis(nn.Module):
    def __init__(self, config: AtomBitConfig) -> None:
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_max=config.cutoff)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        eye_scaled = torch.eye(3, dtype=DEFAULT_FLOAT_DTYPE) / 3.0
        self.register_buffer("_eye_scaled", eye_scaled.view(1, 3, 3))

    def forward(self, vec_ij: torch.Tensor, d_ij: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        bessel = self.rbf(d_ij.unsqueeze(-1))
        raw_rbf = self.rbf_mlp(bessel)
        env = self.envelope(d_ij)
        rbf_feat = raw_rbf * env.unsqueeze(-1)

        r_hat = vec_ij / (d_ij.unsqueeze(-1) + 1e-6)
        safe_eye = self._eye_scaled.to(device=vec_ij.device, dtype=vec_ij.dtype)

        basis: Dict[int, torch.Tensor] = {0: rbf_feat}
        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = torch.einsum("ef,ei->eif", rbf_feat, r_hat)
        if self.cfg.use_L2:
            outer = torch.einsum("ei,ej->eij", r_hat, r_hat)
            basis[2] = torch.einsum("ef,eij->eijf", rbf_feat, outer - safe_eye)
        return basis, r_hat


class LeibnizCoupling(nn.Module):
    def __init__(self, config: AtomBitConfig) -> None:
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.path_weights = nn.ModuleDict()

        for path_key, active in config.active_paths.items():
            if not active:
                continue

            l_in, l_edge, l_out, op_name = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2:
                continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1:
                continue

            name = f"{l_in}_{l_edge}_{l_out}_{op_name}"
            self.path_weights[name] = nn.Linear(self.F, self.F, bias=False)

        self.inv_sqrt_f = self.F ** -0.5

    def forward(
        self,
        h_nodes: Dict[int, Optional[torch.Tensor]],
        basis_edges: Dict[int, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> Dict[int, Optional[torch.Tensor]]:
        _, nbr = edge_index
        messages: Dict[int, Optional[torch.Tensor]] = {0: None, 1: None, 2: None}

        for path_key, active in self.cfg.active_paths.items():
            if not active:
                continue

            l_in, l_edge, l_out, op_type = path_key
            if basis_edges.get(l_edge) is None:
                continue

            layer_name = f"{l_in}_{l_edge}_{l_out}_{op_type}"
            if layer_name not in self.path_weights:
                continue

            inp = h_nodes.get(l_in)
            if inp is None:
                continue

            h_trans = self.path_weights[layer_name](inp)[nbr]
            geom = basis_edges[l_edge]
            res: Optional[torch.Tensor] = None

            if op_type == "prod":
                if l_in == 0 and l_edge == 0:
                    res = h_trans * geom
                elif l_in == 0 and l_edge == 1:
                    res = h_trans.unsqueeze(1) * geom
                elif l_in == 0 and l_edge == 2:
                    res = h_trans.unsqueeze(1).unsqueeze(1) * geom
                elif l_in == 1 and l_edge == 0:
                    res = h_trans * geom.unsqueeze(1)
                elif l_in == 2 and l_edge == 0:
                    res = h_trans * geom.unsqueeze(1).unsqueeze(1)
            elif op_type == "dot":
                res = torch.sum(h_trans * geom, dim=1)
            elif op_type == "cross":
                g = geom.unsqueeze(-1) if geom.dim() == 2 else geom
                res = torch.linalg.cross(h_trans, g, dim=1)
            elif op_type == "outer":
                impl = getattr(self.cfg, "outer_impl", 1)
                if impl == 1:
                    outer = h_trans.unsqueeze(2) * geom.unsqueeze(1)
                    sym = 0.5 * (outer + outer.transpose(1, 2))
                    trace = (h_trans * geom).sum(dim=1) / 3.0
                    res = sym
                    res[:, 0, 0, :] -= trace
                    res[:, 1, 1, :] -= trace
                    res[:, 2, 2, :] -= trace
                elif impl == 4:
                    from src.ops.outer_sym_detrace_ext import outer_sym_detrace

                    res = outer_sym_detrace(h_trans, geom)
                else:
                    raise ValueError(f"Unsupported outer_impl={impl}. Use 1 or 4.")
            elif op_type == "mat_vec":
                res = (h_trans * geom.unsqueeze(1)).sum(dim=2)
            elif op_type == "vec_mat":
                res = (h_trans.unsqueeze(2) * geom).sum(dim=1)
            elif op_type == "double_dot":
                res = torch.sum(h_trans * geom, dim=(1, 2))
            elif op_type == "mat_mul_sym":
                impl = getattr(self.cfg, "mat_mul_sym_impl", 1)
                if impl == 1:
                    e_dim, i_dim, k_dim, f_dim = h_trans.shape
                    j_dim = geom.shape[2]
                    raw = torch.zeros((e_dim, i_dim, j_dim, f_dim), dtype=h_trans.dtype, device=h_trans.device)
                    for k_idx in range(k_dim):
                        raw.add_(h_trans[:, :, k_idx, :].unsqueeze(2) * geom[:, k_idx, :, :].unsqueeze(1))
                    sym = 0.5 * (raw + raw.transpose(1, 2))
                    trace = (sym[:, 0, 0, :] + sym[:, 1, 1, :] + sym[:, 2, 2, :]) / 3.0
                    res = sym
                    res[:, 0, 0, :] -= trace
                    res[:, 1, 1, :] -= trace
                    res[:, 2, 2, :] -= trace
                elif impl == 4:
                    from src.ops.mat_mul_sym_ext import mat_mul_sym

                    res = mat_mul_sym(h_trans, geom)
                else:
                    raise ValueError(f"Unsupported mat_mul_sym_impl={impl}. Use 1 or 4.")
            elif op_type == "vec_cross_tensor":
                v_broad = h_trans.unsqueeze(2)
                res_raw = torch.linalg.cross(v_broad, geom, dim=1)
                res_sym = 0.5 * (res_raw + res_raw.transpose(1, 2))
                trace = (res_sym[:, 0, 0, :] + res_sym[:, 1, 1, :] + res_sym[:, 2, 2, :]) / 3.0
                res = res_sym
                res[:, 0, 0, :] -= trace
                res[:, 1, 1, :] -= trace
                res[:, 2, 2, :] -= trace
            elif op_type == "tensor_cross_vector":
                v_broad = geom.unsqueeze(2)
                term1 = torch.linalg.cross(v_broad, h_trans, dim=1)
                term2 = torch.linalg.cross(v_broad, h_trans.transpose(1, 2), dim=1).transpose(1, 2)
                res_raw = term1 + term2
                res_sym = 0.5 * (res_raw + res_raw.transpose(1, 2))
                trace = (res_sym[:, 0, 0, :] + res_sym[:, 1, 1, :] + res_sym[:, 2, 2, :]) / 3.0
                res = res_sym
                res[:, 0, 0, :] -= trace
                res[:, 1, 1, :] -= trace
                res[:, 2, 2, :] -= trace
            elif op_type == "tensor_commutator":
                a00, a01, a02 = h_trans[:, 0, 0, :], h_trans[:, 0, 1, :], h_trans[:, 0, 2, :]
                a10, a11, a12 = h_trans[:, 1, 0, :], h_trans[:, 1, 1, :], h_trans[:, 1, 2, :]
                a20, a21, a22 = h_trans[:, 2, 0, :], h_trans[:, 2, 1, :], h_trans[:, 2, 2, :]
                b00, b01, b02 = geom[:, 0, 0, :], geom[:, 0, 1, :], geom[:, 0, 2, :]
                b10, b11, b12 = geom[:, 1, 0, :], geom[:, 1, 1, :], geom[:, 1, 2, :]
                b20, b21, b22 = geom[:, 2, 0, :], geom[:, 2, 1, :], geom[:, 2, 2, :]
                res = torch.empty((h_trans.shape[0], 3, h_trans.shape[-1]), dtype=h_trans.dtype, device=h_trans.device)
                res[:, 0, :] = (a20 * b01 + a21 * b11 + a22 * b21) - (b20 * a01 + b21 * a11 + b22 * a21)
                res[:, 1, :] = (a00 * b02 + a01 * b12 + a02 * b22) - (b00 * a02 + b01 * a12 + b02 * a22)
                res[:, 2, :] = (a10 * b00 + a11 * b10 + a12 * b20) - (b10 * a00 + b11 * a10 + b12 * a20)

            if res is not None:
                scaled = res * self.inv_sqrt_f
                messages[l_out] = scaled if messages[l_out] is None else messages[l_out] + scaled

        return messages


class PhysicsGating(nn.Module):
    def __init__(self, config: AtomBitConfig) -> None:
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.W_query = nn.Linear(self.F, self.F, bias=False)
        self.W_key = nn.Linear(self.F, self.F, bias=False)
        self.phys_bias_mlp = nn.Sequential(
            nn.Linear(3 * self.F, self.F, bias=False),
            nn.SiLU(),
            nn.Linear(self.F, 3 * self.F, bias=False),
        )
        self.channel_mixer = nn.Linear(self.F, 3 * self.F, bias=True)
        self.register_buffer("gate_scale", torch.tensor(2.0, dtype=DEFAULT_FLOAT_DTYPE))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.channel_mixer.weight)
        if self.channel_mixer.bias is not None:
            nn.init.zeros_(self.channel_mixer.bias)
        nn.init.zeros_(self.phys_bias_mlp[-1].weight)

    def forward(
        self,
        msgs: Dict[int, Optional[torch.Tensor]],
        h_node0: torch.Tensor,
        scalar_basis: torch.Tensor,
        r_hat: torch.Tensor,
        h_node1: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        capture_weights: bool = False,
    ) -> Dict[int, torch.Tensor]:
        if not self.cfg.use_gating:
            return {key: value for key, value in msgs.items() if value is not None}

        center, nbr = edge_index
        p_ij: Optional[torch.Tensor] = None

        if h_node1 is not None:
            h_center = h_node1[center]
            h_nbr = h_node1[nbr]
            impl = getattr(self.cfg, "gating_impl", 1)
            if impl == 1:
                p_center = torch.einsum("ed,edf->ef", r_hat, h_center)
                p_nbr = torch.einsum("ed,edf->ef", r_hat, h_nbr)
                phys_input = torch.cat([scalar_basis, p_center, p_nbr], dim=-1)
            elif impl == 4:
                from src.ops.gating_proj_ext import gating_proj

                phys_input = gating_proj(r_hat, h_center, h_nbr, scalar_basis)
            else:
                raise ValueError(f"Unsupported gating_impl={impl}. Use 1 or 4.")
        else:
            p_ij = torch.zeros((scalar_basis.shape[0], 2 * self.F), device=scalar_basis.device, dtype=scalar_basis.dtype)
            phys_input = torch.cat([scalar_basis, p_ij], dim=-1)

        chem_logits = self.channel_mixer(self.W_query(h_node0)[center] * self.W_key(h_node0)[nbr])
        phys_logits = self.phys_bias_mlp(phys_input)
        gates = torch.sigmoid(chem_logits + phys_logits) * self.gate_scale.to(h_node0.dtype)

        if capture_weights:
            self.scalar_basis_captured = scalar_basis.detach()
            self.p_ij_captured = None if p_ij is None else p_ij.detach()
            self.chem_logits_captured = chem_logits.detach()
            self.phys_logits_captured = phys_logits.detach()

        g0, g1, g2 = torch.split(gates, self.F, dim=-1)
        if capture_weights:
            self.g0_captured = g0.detach()
            self.g1_captured = g1.detach()
            self.g2_captured = g2.detach()

        out_msgs: Dict[int, torch.Tensor] = {}
        if msgs.get(0) is not None:
            out_msgs[0] = msgs[0] * g0
        if msgs.get(1) is not None:
            out_msgs[1] = msgs[1] * g1.unsqueeze(1)
        if msgs.get(2) is not None:
            out_msgs[2] = msgs[2] * g2.unsqueeze(1).unsqueeze(1)
        return out_msgs


class CartesianDensityBlock(nn.Module):
    def __init__(self, config: AtomBitConfig) -> None:
        super().__init__()
        self.F = config.hidden_dim
        self.cfg = config

        in_dim = self.F
        if config.use_L1:
            in_dim += self.F
        if config.use_L2:
            in_dim += self.F

        hidden_width = self.F * 3
        self.scalar_update_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, self.F),
        )
        if config.use_L1:
            self.L1_linear = nn.Linear(self.F, self.F, bias=False)
        if config.use_L2:
            self.L2_linear = nn.Linear(self.F, self.F, bias=False)

        scale_out_dim = 0
        if config.use_L1:
            scale_out_dim += self.F
        if config.use_L2:
            scale_out_dim += self.F
        self.scale_mlp = nn.Sequential(nn.Linear(self.F, self.F), nn.SiLU(), nn.Linear(self.F, scale_out_dim)) if scale_out_dim > 0 else None

    def forward(
        self,
        msgs: Dict[int, Optional[torch.Tensor]],
        index: torch.Tensor,
        num_nodes: int,
        inv_sqrt_deg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        device, dtype = _runtime_device_dtype(inv_sqrt_deg, msgs.get(0), msgs.get(1), msgs.get(2))
        densities: Dict[int, Optional[torch.Tensor]] = {0: None, 1: None, 2: None}

        if inv_sqrt_deg is None:
            ones = torch.ones(index.shape, dtype=dtype, device=index.device)
            deg = scatter_add(ones, index, dim=0, dim_size=num_nodes)
            deg.clamp_(min=1.0)
            inv_sqrt_deg = torch.rsqrt(deg)

        for degree in (0, 1, 2):
            if msgs.get(degree) is None:
                continue
            agg = scatter_add(msgs[degree], index, dim=0, dim_size=num_nodes)
            view_shape = (num_nodes,) + (1,) * (agg.dim() - 1) if agg.dim() > 1 else (num_nodes,)
            densities[degree] = agg * inv_sqrt_deg.view(view_shape)

        invariants = [densities[0] if densities[0] is not None else torch.zeros((num_nodes, self.F), device=device, dtype=dtype)]
        if self.cfg.use_L1:
            if densities[1] is not None:
                invariants.append(torch.sqrt(torch.sum(densities[1] * densities[1], dim=1) + 1e-8))
            else:
                invariants.append(torch.zeros((num_nodes, self.F), device=device, dtype=dtype))
        if self.cfg.use_L2:
            if densities[2] is not None:
                invariants.append(torch.sqrt(torch.sum(densities[2] * densities[2], dim=(1, 2)) + 1e-8))
            else:
                invariants.append(torch.zeros((num_nodes, self.F), device=device, dtype=dtype))

        concat = torch.cat(invariants, dim=-1) if invariants else torch.zeros((num_nodes, 0), device=device, dtype=dtype)
        delta_h0 = self.scalar_update_mlp(concat) if concat.numel() > 0 else torch.zeros((num_nodes, self.F), device=device, dtype=dtype)

        delta_h1: Optional[torch.Tensor] = None
        delta_h2: Optional[torch.Tensor] = None
        if self.scale_mlp is not None:
            scales = self.scale_mlp(delta_h0)
            curr_dim = 0
            if self.cfg.use_L1:
                alpha1 = scales[:, curr_dim : curr_dim + self.F]
                if densities[1] is not None:
                    delta_h1 = self.L1_linear(densities[1]) * alpha1.unsqueeze(1)
                curr_dim += self.F
            if self.cfg.use_L2:
                alpha2 = scales[:, curr_dim : curr_dim + self.F]
                if densities[2] is not None:
                    delta_h2 = self.L2_linear(densities[2]) * alpha2.unsqueeze(1).unsqueeze(1)

        return delta_h0, delta_h1, delta_h2


class EquivariantLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            norm_sq = torch.mean(torch.sum(x * x, dim=1), dim=-1, keepdim=True)
            return x * torch.rsqrt(norm_sq + 1e-6).unsqueeze(1) * self.weight.unsqueeze(0).unsqueeze(1)

        norm_sq = torch.mean(torch.sum(x * x, dim=(1, 2)), dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_sq + 1e-6).unsqueeze(1).unsqueeze(1) * self.weight.unsqueeze(0).unsqueeze(1).unsqueeze(1)

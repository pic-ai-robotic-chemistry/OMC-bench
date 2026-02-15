"""
HTGP model building blocks: Bessel/envelope basis, geometric basis,
Leibniz coupling, physics gating, Cartesian density, and latent long-range
(electrostatics, vdW, Ewald).
"""

import math
from typing import Dict, List, Optional, Tuple

import mindspore as ms
from mindspore import nn, mint, ops

from src.utils import scatter_add, scatter_mean, HTGPConfig


def compute_bessel_math(
    d: ms.Tensor, r_max: float, freq: ms.Tensor
) -> ms.Tensor:
    d_scaled = d / r_max
    prefactor = (2.0 / r_max) ** 0.5
    return prefactor * mint.sin(freq * d_scaled) / (d + 1e-6)


def compute_envelope_math(d: ms.Tensor, r_cut: float) -> ms.Tensor:
    x = d / r_cut
    x = mint.clamp(x, min=0.0, max=1.0)
    return 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5


def compute_l2_basis(rbf_feat: ms.Tensor, r_hat: ms.Tensor) -> ms.Tensor:
    """L2 (traceless symmetric) basis from RBF features and unit direction."""
    outer = r_hat.unsqueeze(2) * r_hat.unsqueeze(1)
    eye = ms.Tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=r_hat.dtype,
    ).unsqueeze(0)
    trace_less = outer - (1.0 / 3.0) * eye
    return rbf_feat.unsqueeze(1).unsqueeze(1) * trace_less.unsqueeze(-1)


def compute_invariants(
    den0: Optional[ms.Tensor],
    den1: Optional[ms.Tensor],
    den2: Optional[ms.Tensor],
) -> ms.Tensor:
    """Concatenate scalar invariants from L0/L1/L2 densities (L1/L2 as norms)."""
    invariants: List[ms.Tensor] = []

    if den0 is not None:
        invariants.append(den0)
    if den1 is not None:
        sq_sum = mint.sum(den1.pow(ms.Tensor(2, dtype=ms.float32)), dim=1)
        norm = mint.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
    if den2 is not None:
        sq_sum = mint.sum(
            den2.pow(ms.Tensor(2, dtype=ms.float32)), dim=(1, 2)
        )
        norm = mint.sqrt(sq_sum + 1e-8)
        invariants.append(norm)

    if len(invariants) > 0:
        return mint.cat(invariants, dim=-1)
    return mint.zeros(0)


def compute_gating_projections(
    h_node1: ms.Tensor,
    r_hat: ms.Tensor,
    scalar_basis: ms.Tensor,
    src: ms.Tensor,
    dst: ms.Tensor,
) -> ms.Tensor:
    r_hat_uns = r_hat.unsqueeze(-1)
    p_src = mint.sum(h_node1[src] * r_hat_uns, dim=1)
    p_dst = mint.sum(h_node1[dst] * r_hat_uns, dim=1)
    return mint.cat([scalar_basis, p_src, p_dst], dim=-1)


class BesselBasis(nn.Cell):
    """Bessel radial basis functions."""

    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        self.register_buffer(
            "freq",
            mint.arange(1, num_basis + 1).float() * math.pi,
        )

    def construct(self, d: ms.Tensor) -> ms.Tensor:
        return compute_bessel_math(d, self.r_max, self.freq)


class PolynomialEnvelope(nn.Cell):
    """Polynomial envelope (smooth cutoff)."""

    def __init__(self, r_cut: float, p: int = 5):
        super().__init__()
        self.r_cutoff = float(r_cut)
        self.p = int(p)
    
    def construct(self, d_ij: ms.Tensor) -> ms.Tensor:
        return compute_envelope_math(d_ij, self.r_cutoff)


class GeometricBasis(nn.Cell):
    """RBF + envelope and L0/L1/L2 geometric bases; returns basis dict and r_hat."""

    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.SequentialCell(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def construct(self, vec_ij, d_ij):
        raw_rbf = self.rbf_mlp(self.rbf(d_ij.unsqueeze(-1)))
        env = self.envelope(d_ij)
        rbf_feat = raw_rbf * env.unsqueeze(-1)
        r_hat = vec_ij / (d_ij.unsqueeze(-1) + 1e-6)

        basis = {}
        basis[0] = rbf_feat
        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = rbf_feat.unsqueeze(1) * r_hat.unsqueeze(-1)
        if self.cfg.use_L2:
            basis[2] = compute_l2_basis(rbf_feat, r_hat)
        return basis, r_hat


def optimized_cross(a, b):
    """Cross product on last spatial dim: (N, 3, L) with (N, 3, L) -> (N, 3, L)."""
    a_perm = a.transpose(0, 2, 1)
    b_perm = b.transpose(0, 2, 1)
    cross_x = a_perm[..., 1] * b_perm[..., 2] - a_perm[..., 2] * b_perm[..., 1]
    cross_y = a_perm[..., 2] * b_perm[..., 0] - a_perm[..., 0] * b_perm[..., 2]
    cross_z = a_perm[..., 0] * b_perm[..., 1] - a_perm[..., 1] * b_perm[..., 0]
    result_perm = ops.stack((cross_x, cross_y, cross_z), axis=-1)
    return result_perm.transpose(0, 2, 1)


class LeibnizCoupling(nn.Cell):
    """Leibniz-style message generation over L0/L1/L2 paths and op types."""

    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.path_weights = nn.CellDict()

        for path_key, active in config.active_paths.items():
            if not active:
                continue
            l_in, l_edge, l_out, _ = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2:
                continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1:
                continue
            name = f"{l_in}_{l_edge}_{l_out}_{path_key[3]}"
            self.path_weights[name] = nn.Linear(self.F, self.F, bias=False)

        self.inv_sqrt_f = self.F ** -0.5

    def construct(
        self,
        h_nodes: Dict[int, ms.Tensor],
        basis_edges: Dict[int, ms.Tensor],
        edge_index,
        expert_mixing_coefficients: Optional[ms.Tensor] = None,
        batch: Optional[ms.Tensor] = None,
    ):
        src, _ = edge_index
        messages: Dict[int, List[ms.Tensor]] = {0: [], 1: [], 2: []}

        for path_key, active in self.cfg.active_paths.items():
            if not active:
                continue
            l_in, l_edge, l_out, op_type = path_key
            if basis_edges.get(l_edge) is None:
                continue
            layer_name = f"{l_in}_{l_edge}_{l_out}_{op_type}"
            if layer_name not in self.path_weights:
                continue
            if h_nodes.get(l_in) is None:
                continue
            inp = h_nodes[l_in]
            
            h_src = inp[src]
            
            h_trans = self.path_weights[layer_name](h_src)
            geom = basis_edges[l_edge]
            res = None
            
            # --- Operation Logic ---
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
                res = mint.sum(h_trans * geom, dim=1)
            elif op_type == "cross":
                g = geom
                if g.dim() == 2:
                    g = g.unsqueeze(-1)
                res = optimized_cross(h_trans, g)
            elif op_type == "outer":
                outer = h_trans.unsqueeze(2) * geom.unsqueeze(1)
                trace = sum_diag_gather(outer)
                eye = ms.Tensor(
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    dtype=h_trans.dtype,
                ).view(1, 3, 3, 1)
                res = (
                    outer
                    - (1.0 / 3.0)
                    * trace.unsqueeze(1).unsqueeze(1)
                    * eye
                )
            elif op_type == "mat_vec":
                res = (h_trans * geom.unsqueeze(1)).sum(dim=2)
            elif op_type == "vec_mat":
                res = (h_trans.unsqueeze(2) * geom).sum(dim=1)
            elif op_type == "double_dot":
                res = mint.sum(h_trans * geom, dim=(1, 2))
            elif op_type == "mat_mul_sym":
                e, i, k, f = h_trans.shape
                _, _, j, _ = geom.shape
                raw = mint.zeros((e, i, j, f)).astype(h_trans.dtype)
                for k_idx in range(k):
                    raw += (
                        h_trans[:, :, k_idx, :].unsqueeze(2)
                        * geom[:, k_idx, :, :].unsqueeze(1)
                    )
                sym = 0.5 * (raw + raw.transpose(1, 2))
                trace = sum_diag_gather(sym)
                eye = ms.Tensor(
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    dtype=h_trans.dtype,
                ).view(1, 3, 3, 1)
                res = (
                    sym
                    - (1.0 / 3.0)
                    * trace.unsqueeze(1).unsqueeze(1)
                    * eye
                )

            if res is not None:
                messages[l_out].append(res * self.inv_sqrt_f)
                
        final_msgs: Dict[int, Optional[ms.Tensor]] = {}
        for l in [0, 1, 2]:
            final_msgs[l] = sum(messages[l]) if len(messages[l]) > 0 else None
        return final_msgs


def sum_diag_gather(x: ms.Tensor) -> ms.Tensor:
    """
    Sum diagonal of (E, I, I, F) over middle dims: y[e,f] = sum_i x[e,i,i,f].
    """
    E, I, _, F = x.shape
    x2 = x.reshape(E, I * I, F)
    diag = (
        mint.arange(I).astype(ms.int32) * (I + 1)
    ).reshape(1, I, 1)
    diag = mint.broadcast_to(diag, (E, I, 1))
    e = mint.arange(E).astype(ms.int32).reshape(E, 1, 1)
    e = mint.broadcast_to(e, (E, I, 1))
    idx = mint.concat((e, diag), dim=-1)
    diag_vals = ops.gather_nd(x2, idx)
    y = mint.sum(diag_vals, dim=1)
    return y


class PhysicsGating(nn.Cell):
    """Chemistry + physics gates (query/key + phys_bias_mlp) over L0/L1/L2."""

    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        self.W_query = nn.Linear(self.F, self.F, bias=False)
        self.W_key = nn.Linear(self.F, self.F, bias=False)
        
        self.phys_bias_mlp = nn.SequentialCell(
            nn.Linear(3 * self.F, self.F), 
            nn.SiLU(),            
            nn.Linear(self.F, 3 * self.F) 
        )
        self.channel_mixer = nn.Linear(self.F, 3 * self.F, bias=False)
        self.gate_scale = ms.Parameter(mint.ones(1) * 2.0)

    def construct(
        self,
        msgs,
        h_node0,
        scalar_basis,
        r_hat,
        h_node1,
        edge_index,
        capture_weights=False,
    ):
        if not self.cfg.use_gating:
            return msgs

        src, dst = edge_index

        if h_node1 is not None:
            phys_input = compute_gating_projections(h_node1, r_hat, scalar_basis, src, dst)
            split_idx = scalar_basis.shape[-1]
            p_ij = phys_input[:, split_idx:]        
        else:
            p_ij = mint.zeros((scalar_basis.shape[0], 2 * self.F))
            phys_input = mint.cat([scalar_basis, p_ij], dim=-1)

        q = self.W_query(h_node0[dst]) 
        k = self.W_key(h_node0[src])   
        chem_score = q * k             
        chem_logits = self.channel_mixer(chem_score)
        phys_logits = self.phys_bias_mlp(phys_input)
        
        raw_gates = chem_logits + phys_logits
        gates = mint.sigmoid(raw_gates) * self.gate_scale
        
        if capture_weights:
            self.scalar_basis_captured = scalar_basis.detach()
            self.p_ij_captured = p_ij.detach()
            self.chem_logits_captured = chem_logits.detach()
            self.phys_logits_captured = phys_logits.detach()

        g_list = mint.split(gates, self.F, dim=-1)
        g0, g1, g2 = [g.contiguous() for g in g_list]
        if capture_weights:
            self.g0_captured = g0.detach()
            self.g1_captured = g1.detach()
            self.g2_captured = g2.detach()

        out_msgs: Dict[int, ms.Tensor] = {}
        if 0 in msgs and msgs[0] is not None:
            out_msgs[0] = msgs[0] * g0
        if 1 in msgs and msgs[1] is not None:
            out_msgs[1] = msgs[1] * g1.unsqueeze(1)
        if 2 in msgs and msgs[2] is not None:
            out_msgs[2] = msgs[2] * g2.unsqueeze(1).unsqueeze(1)
        return out_msgs


class CartesianDensityBlock(nn.Cell):
    """Aggregate messages to densities, compute invariants, scalar/vector updates."""

    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        self.cfg = config

        in_dim = 0
        if config.use_L0:
            in_dim += self.F
        if config.use_L1:
            in_dim += self.F
        if config.use_L2:
            in_dim += self.F

        self.scalar_update_mlp = nn.SequentialCell(
            nn.Linear(in_dim, self.F),
            nn.SiLU(),
            nn.Linear(self.F, self.F)
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

        if scale_out_dim > 0:
            self.scale_mlp = nn.SequentialCell(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, scale_out_dim)
            )
        else:
            self.scale_mlp = None

        self.inv_sqrt_deg = 1.0 / (config.avg_neighborhood ** 0.5)

    def construct(
        self,
        msgs: Dict[int, ms.Tensor],
        index: ms.Tensor,
        num_nodes: int,
    ):
        densities: Dict[int, Optional[ms.Tensor]] = {
            0: None, 1: None, 2: None,
        }
        for l in [0, 1, 2]:
            if l in msgs and msgs[l] is not None:
                agg = scatter_add(
                    msgs[l], index, dim=0, dim_size=num_nodes
                )
                densities[l] = agg * self.inv_sqrt_deg

        concat = compute_invariants(
            densities[0], densities[1], densities[2]
        )
        if concat.numel() > 0:
            delta_h0 = self.scalar_update_mlp(concat)
        else:
            delta_h0 = mint.zeros((num_nodes, self.F))

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


# Coulomb constant (eV * Angstrom), 1 / (4 * pi * epsilon_0) in eV*A
KE_CONST = 14.3996


def compute_direct_electrostatics_jit(
    q: ms.Tensor,
    dist: ms.Tensor,
    batch_mask: ms.Tensor,
    sigma: ms.Tensor,
    ke_const: float,
) -> ms.Tensor:
    """
    Real-space screened Coulomb sum for finite or short-range correction.
    Uses erf screening; returns 0.5 * ke_const * sum over pairs (no double count).
    """
    qq = q @ q.t()
    inv_dist = 1.0 / (dist + 1e-8)
    sqrt2 = 1.41421356
    scaled_r = dist / (sqrt2 * sigma)
    shielding = mint.erf(scaled_r)
    E_matrix = qq * inv_dist * shielding
    E_sum = mint.sum(E_matrix * batch_mask)
    return 0.5 * ke_const * E_sum


def compute_bj_damping_vdw_jit(
    c6: ms.Tensor,
    r_vdw: ms.Tensor,
    dist_sq: ms.Tensor,
    batch_mask: ms.Tensor,
) -> ms.Tensor:
    """
    Becke-Johnson damped van der Waals (C6/R^6) with geometric combination rules.
    """
    c6_ij = mint.sqrt(c6 @ c6.t())
    rvdw_ij = mint.sqrt(r_vdw @ r_vdw.t())
    dist6 = dist_sq ** 3
    damping = dist6 + (rvdw_ij ** 6)
    E_matrix = -(c6_ij / (damping + 1e-8)) * batch_mask
    return 0.5 * mint.sum(E_matrix)


def generate_k_template(k_cutoff: float) -> ms.Tensor:
    """Generate integer k-grid (n1, n2, n3), excluding (0,0,0)."""
    n_max = 8
    rng = mint.arange(-n_max, n_max + 1, dtype=ms.float32)
    n1, n2, n3 = mint.meshgrid(rng, rng, rng, indexing="ij")
    n = mint.stack(
        [n1.flatten(), n2.flatten(), n3.flatten()], dim=1
    )
    n_sq = mint.sum(n**2, dim=1)
    mask = n_sq > 0
    return n[mask]

def compute_ewald_kspace_jit(
    q: ms.Tensor,
    pos: ms.Tensor,
    batch: ms.Tensor,
    cell: ms.Tensor,
    n_grid: ms.Tensor,
    sigma: ms.Tensor,
    k_cutoff: float,
    num_graphs: int,
    ke_const: float,
) -> ms.Tensor:
    """
    Ewald reciprocal-space sum with soft k-mask so all batches share same M.
    Returns per-graph reciprocal energy minus self-energy.
    """
    recip_cell = 2 * math.pi * mint.inverse(cell).transpose(1, 2)
    k_vecs = mint.matmul(n_grid.unsqueeze(0), recip_cell)
    k_sq = mint.sum(k_vecs**2, dim=-1)
    mask_cutoff = (k_sq < k_cutoff**2).float()

    k_vecs_expanded = k_vecs[batch]
    kr = mint.sum(k_vecs_expanded * pos.unsqueeze(1), dim=-1)
    cos_kr = mint.cos(kr)
    sin_kr = mint.sin(kr)

    M = n_grid.size(0)
    Sk_real = mint.zeros((num_graphs, M)).astype(q.dtype)
    Sk_imag = mint.zeros((num_graphs, M)).astype(q.dtype)
    Sk_real.index_add_(0, batch, q * cos_kr)
    Sk_imag.index_add_(0, batch, q * sin_kr)
    Sk_sq = Sk_real**2 + Sk_imag**2

    prefactor = mint.exp(
        (-0.5 * sigma.astype(ms.float32) ** 2 * k_sq).astype(
            ms.float32
        )
    ) / (k_sq + 1e-12)
    prefactor = prefactor * mask_cutoff

    E_recip_raw = mint.sum(prefactor * Sk_sq, dim=1)
    _, logabsdet = ops.linalg.slogdet(cell)
    vol = mint.exp(logabsdet)
    coeff = (2 * math.pi * ke_const) / vol
    E_recip = coeff * E_recip_raw

    q_sq = q**2
    q_sq_sum = mint.zeros((num_graphs, 1)).astype(dtype=q.dtype)
    q_sq_sum.index_add_(0, batch, q_sq)
    q_sq_sum = q_sq_sum.squeeze(-1)
    self_prefactor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    E_self = ke_const * self_prefactor * q_sq_sum

    return E_recip - E_self


class LatentLongRange(nn.Cell):
    """
    Long-range correction: charge (Coulomb), vdW (BJ damping), optional dipole.
    For PBC uses Ewald k-space; for clusters uses direct screened sum.
    """

    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.KE = KE_CONST

        if config.use_charge:
            self.q_proj = nn.SequentialCell(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 1, bias=False),
            )
            self.sigma = ms.Parameter(ms.Tensor(1.0, ms.float32))

        if config.use_vdw:
            self.vdw_proj = nn.SequentialCell(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 2),
            )

        if config.use_dipole:
            self.mu_proj = nn.Linear(self.F, 1, bias=False)

        self.register_buffer("n_grid_cache", None)

    def construct(
        self,
        h0,
        h1,
        pos,
        batch,
        cell: Optional[ms.Tensor] = None,
        capture_descriptors=False,
    ):
        total_energy = 0.0
        num_graphs = int(batch.max().item()) + 1
        q = None
        c6, r_vdw = None, None

        if self.cfg.use_charge:
            q = self.q_proj(h0)
            q_sum = mint.zeros((num_graphs, 1)).astype(q.dtype)
            q_sum.index_add_(0, batch, q)
            ones = mint.ones_like(q)
            counts = mint.zeros((num_graphs, 1)).astype(q.dtype)
            counts.index_add_(0, batch, ones)
            q_mean = q_sum / counts.clamp(min=1.0)
            q = q - q_mean[batch]
            if capture_descriptors:
                self.charge = q

        if self.cfg.use_vdw:
            vdw_params = self.vdw_proj(h0)
            c6 = ops.softplus(vdw_params[:, 0:1])
            r_vdw = ops.softplus(vdw_params[:, 1:2])

        is_periodic = False
        if cell is not None:
            _, logabsdet = ops.linalg.slogdet(cell)
            det = mint.exp(logabsdet)
            if (det > 1e-6).all():
                is_periodic = True

        if is_periodic:
            if self.n_grid_cache is None:
                self.n_grid_cache = generate_k_template(k_cutoff=6.0)
            e_elec_batch = compute_ewald_kspace_jit(
                q,
                pos,
                batch,
                cell,
                self.n_grid_cache,
                self.sigma,
                k_cutoff=6.0,
                num_graphs=num_graphs,
                ke_const=self.KE,
            )
            total_energy += mint.sum(e_elec_batch)
        else:
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist_sq = mint.sum(diff**2, dim=-1)
            dist = mint.sqrt(dist_sq + 1e-8)
            batch_mask = batch.unsqueeze(1) == batch.unsqueeze(0)
            diag_mask = mint.eye(pos.shape[0], dtype=ms.bool_)
            valid_mask = batch_mask & (~diag_mask)
            mask_float = valid_mask.astype(ms.float32)

            if self.cfg.use_charge and q is not None:
                e_elec = compute_direct_electrostatics_jit(
                    q, dist, mask_float, self.sigma, ke_const=self.KE
                )
                total_energy += e_elec
            if self.cfg.use_vdw and c6 is not None:
                e_vdw = compute_bj_damping_vdw_jit(
                    c6, r_vdw, dist_sq, mask_float
                )
                total_energy += e_vdw

        return total_energy * self.cfg.long_range_scale
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.utils import AtomBitConfig, scatter_add

from .modules import (
    CartesianDensityBlock,
    EquivariantLayerNorm,
    GeometricBasis,
    LeibnizCoupling,
    PhysicsGating,
)


class ScaleShift(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class OptionalEquivariantNorm(nn.Module):
    def __init__(self, hidden_dim: int, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.norm = EquivariantLayerNorm(hidden_dim) if enabled else None

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.enabled or x is None:
            return x
        return self.norm(x)


class IdentityNorm(nn.Module):
    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x


class PassthroughGating(nn.Module):
    def forward(
        self,
        msgs,
        h_node0: torch.Tensor,
        scalar_basis: torch.Tensor,
        r_hat: torch.Tensor,
        h_node1: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        capture_weights: bool = False,
    ):
        return {key: value for key, value in msgs.items() if value is not None}


def build_mlp_readout(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )


COMPONENT_FACTORIES: Dict[str, Callable[..., nn.Module]] = {
    "equivariant_norm": lambda config, *, enabled=True: OptionalEquivariantNorm(config.hidden_dim, enabled=enabled),
    "identity_norm": lambda config, **kwargs: IdentityNorm(),
    "leibniz": lambda config, **kwargs: LeibnizCoupling(config),
    "physics_gating": lambda config, **kwargs: PhysicsGating(config),
    "identity_gating": lambda config, **kwargs: PassthroughGating(),
    "cartesian_density": lambda config, **kwargs: CartesianDensityBlock(config),
    "mlp_readout": lambda config, **kwargs: build_mlp_readout(config.hidden_dim),
}


def build_registered_component(name: str, config: AtomBitConfig, **kwargs) -> nn.Module:
    try:
        factory = COMPONENT_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown AtomBit block component '{name}'.") from exc
    return factory(config, **kwargs)


class AtomBitInteractionBlock(nn.Module):
    def __init__(self, config: AtomBitConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.use_L1 = config.use_L1
        self.use_L2 = config.use_L2
        impls = dict(config.block_impls)
        self.norm = nn.LayerNorm(config.hidden_dim)
        norm_l1_name = impls.get("norm_l1", "equivariant_norm") if config.use_L1 else "identity_norm"
        norm_l2_name = impls.get("norm_l2", "equivariant_norm") if config.use_L2 else "identity_norm"
        gated_layers = getattr(config, "gating_layer_indices", None)
        gate_enabled = config.use_gating and (
            gated_layers is None or self.layer_index in set(gated_layers)
        )
        gating_name = impls.get("gating", "physics_gating") if gate_enabled else "identity_gating"
        self.norm_L1 = build_registered_component(norm_l1_name, config, enabled=config.use_L1)
        self.norm_L2 = build_registered_component(norm_l2_name, config, enabled=config.use_L2)
        self.coupling = build_registered_component(impls.get("coupling", "leibniz"), config)
        self.gating = build_registered_component(gating_name, config)
        self.density = build_registered_component(impls.get("density", "cartesian_density"), config)
        self.readout = build_registered_component(impls.get("readout", "mlp_readout"), config)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(
        self,
        h0: torch.Tensor,
        h1: Optional[torch.Tensor],
        h2: Optional[torch.Tensor],
        basis_edges,
        r_hat: torch.Tensor,
        data,
        inv_sqrt_deg: torch.Tensor,
        capture_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        h0_norm = self.norm(h0)
        h1_in = self.norm_L1(h1)
        h2_in = self.norm_L2(h2)

        node_feats = {0: h0_norm, 1: h1_in, 2: h2_in}
        raw_msgs = self.coupling(node_feats, basis_edges, data.edge_index)
        gated_msgs = self.gating(
            raw_msgs,
            h0_norm,
            basis_edges[0],
            r_hat,
            h1_in,
            data.edge_index,
            capture_weights=capture_weights,
        )
        delta_h0, delta_h1, delta_h2 = self.density(
            gated_msgs,
            data.edge_index[0],
            data.z.size(0),
            inv_sqrt_deg=inv_sqrt_deg,
        )

        next_h0 = h0 + delta_h0
        next_h1 = h1
        next_h2 = h2

        if self.use_L1:
            next_h1 = delta_h1 if h1 is None else (h1 + delta_h1 if delta_h1 is not None else h1)
        if self.use_L2:
            next_h2 = delta_h2 if h2 is None else (h2 + delta_h2 if delta_h2 is not None else h2)

        atomic_energy = self.readout(next_h0)
        return next_h0, next_h1, next_h2, atomic_energy


class AtomBitModel(nn.Module):
    def __init__(self, config: AtomBitConfig) -> None:
        super().__init__()
        self.cfg = config

        configured_map = list(getattr(config, "atom_types_map", []) or [])
        if configured_map:
            self.used_atomic_numbers: List[int] = sorted({int(z) for z in configured_map})
        else:
            self.used_atomic_numbers = list(range(1, int(config.num_atom_types) + 1))

        num_actual_types = len(self.used_atomic_numbers)
        max_z = max(self.used_atomic_numbers)
        self.register_buffer("z_mapper", torch.full((max_z + 1,), -1, dtype=torch.long))
        for idx, atomic_number in enumerate(self.used_atomic_numbers):
            self.z_mapper[atomic_number] = idx

        self.embedding = nn.Embedding(num_actual_types, config.hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.embedding_norm = nn.LayerNorm(config.hidden_dim)
        self.geom_basis = GeometricBasis(config)
        self.blocks = nn.ModuleList(
            [AtomBitInteractionBlock(config, layer_index=i) for i in range(config.num_layers)]
        )

        self.readout_force = None
        if config.use_direct_force and config.use_L1:
            self.readout_force = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim, bias=False),
                nn.Linear(config.hidden_dim, config.hidden_dim, bias=False),
                nn.Linear(config.hidden_dim, 1, bias=False),
            )

        self.atomic_ref = nn.Embedding(num_actual_types, 1)
        nn.init.zeros_(self.atomic_ref.weight)
        self.scale_shift = ScaleShift(mean=0.0, std=1.0)
        self.reset_parameters()

    @property
    def layers(self):
        return self.blocks

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if self.readout_force is not None:
            nn.init.zeros_(self.readout_force[-1].weight)

    def _as_batched_cell(self, cell: torch.Tensor | None, num_graphs: int) -> torch.Tensor | None:
        if cell is None:
            return None
        if cell.dim() == 3:
            return cell
        if cell.dim() == 2:
            if tuple(cell.shape) == (3, 3):
                return cell.unsqueeze(0)
            if cell.shape[1] == 3 and cell.shape[0] == num_graphs * 3:
                return cell.view(num_graphs, 3, 3)
        raise ValueError(f"Unsupported cell shape {tuple(cell.shape)} for num_graphs={num_graphs}")

    def _map_atomic_numbers(self, z_raw: torch.Tensor) -> torch.Tensor:
        z_idx = self.z_mapper[z_raw]
        if (z_idx < 0).any():
            missing = torch.unique(z_raw[z_idx < 0]).detach().cpu().tolist()
            raise ValueError(
                "Input batch contains atomic numbers not covered by the model map: "
                f"{missing}. Config atom_types_map={self.used_atomic_numbers[:20]}"
                + ("..." if len(self.used_atomic_numbers) > 20 else "")
            )
        return z_idx

    def forward(self, data, capture_weights: bool = False, capture_descriptors: bool = False):
        if capture_descriptors:
            self.all_layer_descriptors = []
            self.edge_info = {}

        z_raw = data.z
        z_idx = self._map_atomic_numbers(z_raw)

        center, nbr = data.edge_index
        cell = self._as_batched_cell(getattr(data, "cell", None), data.num_graphs)
        runtime_device = data.pos.device
        runtime_dtype = data.pos.dtype

        if hasattr(data, "shifts_int") and data.shifts_int is not None and cell is not None:
            current_shifts = torch.matmul(data.shifts_int.unsqueeze(1).to(dtype=runtime_dtype), cell[data.batch[center]]).squeeze(1)
        else:
            current_shifts = torch.zeros((center.size(0), 3), device=runtime_device, dtype=runtime_dtype)

        vec_ji = data.pos[center] - data.pos[nbr] - current_shifts
        d_ji = torch.sqrt((vec_ji * vec_ji).sum(dim=-1) + 1e-12)

        if capture_descriptors:
            self.edge_info = {
                "d_ji": d_ji.detach().cpu(),
                "center_z": z_raw[center].detach().cpu(),
                "nbr_z": z_raw[nbr].detach().cpu(),
                "center_idx": center.detach().cpu(),
                "nbr_idx": nbr.detach().cpu(),
                "node_pos": data.pos.detach().cpu(),
                "node_z": z_raw.detach().cpu(),
            }

        basis_edges, r_hat = self.geom_basis(vec_ji, d_ji)
        h0 = self.embedding_norm(self.embedding(z_idx))
        h1 = None
        h2 = None

        total_energy = torch.zeros((data.num_graphs, 1), dtype=runtime_dtype, device=runtime_device)
        total_force = torch.zeros((data.z.shape[0], 3), dtype=runtime_dtype, device=runtime_device) if self.cfg.use_direct_force else None

        ones = torch.ones(center.shape, dtype=runtime_dtype, device=runtime_device)
        deg = scatter_add(ones, center, dim_size=data.z.size(0))
        deg.clamp_(min=1.0)
        inv_sqrt_deg = torch.rsqrt(deg)

        for block in self.blocks:
            h0, h1, h2, atomic_energy = block(
                h0,
                h1,
                h2,
                basis_edges,
                r_hat,
                data,
                inv_sqrt_deg,
                capture_weights=capture_weights,
            )

            if capture_descriptors:
                current_layer_feats = {"h0": h0.detach().cpu()}
                if self.cfg.use_L1 and h1 is not None:
                    current_layer_feats["h1"] = h1.detach().cpu()
                if self.cfg.use_L2 and h2 is not None:
                    current_layer_feats["h2"] = h2.detach().cpu()
                self.all_layer_descriptors.append(current_layer_feats)

            total_energy = total_energy + scatter_add(atomic_energy, data.batch, dim_size=data.num_graphs)
            if self.cfg.use_direct_force and self.readout_force is not None and h1 is not None:
                total_force = total_force + self.readout_force(h1).squeeze(-1)

        total_energy = self.scale_shift(total_energy)
        if self.cfg.use_direct_force:
            return {"energy": total_energy, "force": total_force}
        return total_energy

    def load_external_e0(self, e0_dict, device=None, verbose: bool = True, rank: int = 0) -> None:
        if device is None:
            device = self.atomic_ref.weight.device

        dtype = self.atomic_ref.weight.dtype
        count = 0
        skipped = []
        with torch.no_grad():
            mapper_cpu = self.z_mapper.cpu()
            for atomic_number, energy in e0_dict.items():
                z_raw = int(atomic_number)
                if z_raw < len(mapper_cpu):
                    mapped_idx = mapper_cpu[z_raw].item()
                    if mapped_idx != -1:
                        self.atomic_ref.weight[mapped_idx] = torch.tensor(energy, dtype=dtype, device=device)
                        count += 1
                        continue
                skipped.append(z_raw)

        self.atomic_ref.weight.requires_grad = False
        if verbose and rank == 0:
            print(f"[AtomBitModel] Injected external E0 for {count} elements.")
            if skipped:
                preview = sorted(skipped)[:20]
                suffix = "..." if len(skipped) > 20 else ""
                print(
                    "[AtomBitModel] Warning: skipped E0 entries for atomic numbers "
                    f"{preview}{suffix} because they are not in atom_types_map."
                )

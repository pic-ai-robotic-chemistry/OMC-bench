"""
HTGP main model: message passing with geometric basis, Leibniz coupling,
physics gating, Cartesian density blocks, optional long-range correction,
and atomic reference (E0) embedding.
"""

from typing import Dict

import mindspore as ms
from mindspore import nn, mint, ops
from mindspore.common.initializer import Normal

from src.utils import scatter_add, scatter_mean, HTGPConfig

from .Modules import (
    CartesianDensityBlock,
    GeometricBasis,
    LatentLongRange,
    LeibnizCoupling,
    PhysicsGating,
)


class HTGPModel(nn.Cell):
    """
    HTGP potential model: short-range message passing plus optional
    long-range electrostatic correction. Uses a Z-mapper for atom types
    and atomic reference (E0) embedding.
    """

    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config

        # Z-mapper: map raw atomic numbers to dense embedding indices
        if hasattr(config, "atom_types_map"):
            self.used_atomic_numbers = config.atom_types_map
        else:
            self.used_atomic_numbers = [
                1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53,
            ]

        num_actual_types = len(self.used_atomic_numbers)
        max_z = max(self.used_atomic_numbers)
        self.num_actual_types = num_actual_types

        # Buffer: -1 for unsupported Z, else dense index (no gradient)
        self.register_buffer(
            "z_mapper",
            mint.full((max_z + 1,), -1, dtype=ms.int32),
        )
        for idx, z in enumerate(self.used_atomic_numbers):
            self.z_mapper[z] = idx

        # Embedding: one row per actual atom type (e.g. 11 types)
        self.embedding = nn.Embedding(
            num_actual_types,
            config.hidden_dim,
            embedding_table=Normal(sigma=0.1),
        )

        self.geom_basis = GeometricBasis(config)
        self.layers = nn.CellList()
        for _ in range(config.num_layers):
            self.layers.append(
                nn.CellDict({
                    "coupling": LeibnizCoupling(config),
                    "gating": PhysicsGating(config),
                    "density": CartesianDensityBlock(config),
                })
            )

        self.readout_energy = nn.SequentialCell(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        if config.use_long_range:
            self.long_range = LatentLongRange(config)
        else:
            print("Long-range module not used.")

        # Atomic reference (E0) per atom type; can be loaded from external E0
        self.atomic_ref = nn.Embedding(num_actual_types, 1)
        self.atomic_ref.embedding_table = ms.Parameter(
            self.atomic_ref.embedding_table * 0
        )

    def construct(
        self,
        data,
        capture_weights=False,
        capture_descriptors=False,
    ):
        """
        Forward pass: map Z -> dense index, message passing, readout, E0.

        Returns:
            total_energy: (num_graphs,) per-graph total energy.
        """
        if capture_descriptors:
            self.all_layer_descriptors = []

        if hasattr(data, "num_graphs"):
            num_graphs = data.num_graphs
        else:
            num_graphs = int(data.batch.max().item()) + 1

        z_raw = data.z
        z_idx = self.z_mapper[z_raw]

        # Geometry: edge vectors and distances (with PBC shifts if present)
        row, col = data.edge_index
        cell = getattr(data, "cell", None)
        if (
            hasattr(data, "shifts_int")
            and data.shifts_int is not None
            and cell is not None
        ):
            batch_cell = data.cell[data.batch[row]]
            current_shifts = mint.bmm(
                data.shifts_int.unsqueeze(1), batch_cell
            ).squeeze(1)
        else:
            current_shifts = mint.zeros(
                (row.shape[0], 3), dtype=data.pos.dtype
            )

        vec_ij = data.pos[col] - data.pos[row] + current_shifts
        d_ij = mint.norm(vec_ij, dim=-1).clamp(min=1e-8)
        basis_edges, r_hat = self.geom_basis(vec_ij, d_ij)

        h0 = self.embedding(z_idx)
        h1 = None
        h2 = None
        total_energy = 0.0

        for layer in self.layers:
            node_feats = {0: h0, 1: h1, 2: h2}
            raw_msgs = layer["coupling"](
                node_feats, basis_edges, data.edge_index
            )
            gated_msgs = layer["gating"](
                raw_msgs,
                h0,
                basis_edges[0],
                r_hat,
                h1,
                data.edge_index,
                capture_weights=capture_weights,
            )
            delta_h0, delta_h1, delta_h2 = layer["density"](
                gated_msgs, row, data.z.shape[0]
            )

            h0 = h0 + delta_h0
            if self.cfg.use_L1:
                if h1 is None:
                    h1 = delta_h1
                elif delta_h1 is not None:
                    h1 = h1 + delta_h1
            if self.cfg.use_L2:
                if h2 is None:
                    h2 = delta_h2
                elif delta_h2 is not None:
                    h2 = h2 + delta_h2

            if capture_descriptors:
                current_layer_feats = {
                    "h0": h0.detach().cpu(),
                }
                if self.cfg.use_L1 and h1 is not None:
                    current_layer_feats["h1"] = h1.detach().cpu()
                if self.cfg.use_L2 and h2 is not None:
                    current_layer_feats["h2"] = h2.detach().cpu()
                self.all_layer_descriptors.append(current_layer_feats)

            atomic_energy = self.readout_energy(h0)
            total_energy = total_energy + scatter_add(
                atomic_energy,
                data.batch,
                dim=0,
                dim_size=data.num_graphs,
            )

        if self.cfg.use_long_range and self.cfg.use_L1 and h1 is not None:
            cell = getattr(data, "cell", None)
            e_long = self.long_range(
                h0,
                h1,
                data.pos,
                data.batch,
                cell,
                capture_descriptors=capture_descriptors,
            )
            total_energy = total_energy + e_long

        total_energy = total_energy + scatter_add(
            self.atomic_ref(z_idx),
            data.batch,
            dim=0,
            dim_size=data.num_graphs,
        )

        return total_energy

    def load_external_e0(
        self,
        e0_dict: Dict[int, float],
        verbose: bool = True,
        rank: int = 0,
    ):
        """
        Load E0 values from a dict (raw Z -> e0) into atomic_ref.

        Uses the internal z_mapper so only supported atom types are updated.
        atomic_ref is then frozen (requires_grad=False).

        Args:
            e0_dict: Mapping from atomic number Z to E0 value (float).
            verbose: If True and rank==0, print how many elements were set.
            rank: Process rank (logging only on rank 0).
        """
        count = 0
        with ms._no_grad():
            mapper_cpu = self.z_mapper
            for z, e in e0_dict.items():
                z_raw = int(z)
                if z_raw < len(mapper_cpu):
                    mapped_idx = mapper_cpu[z_raw].item()
                    if mapped_idx != -1:
                        val = ms.Tensor(e, dtype=ms.float32)
                        self.atomic_ref.embedding_table[mapped_idx] = val
                        count += 1

        self.atomic_ref.embedding_table.requires_grad = False
        if verbose and rank == 0:
            print(f"[Model] Loaded E0 for {count} atom types.")

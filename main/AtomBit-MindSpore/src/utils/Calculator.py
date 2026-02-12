"""
ASE Calculator for HTGP: energy, forces, stress from MindSpore model.

Converts ASE atoms to a single-graph format, runs forward with optional
strain for stress, then backward for forces and stress tensor. Supports
capture_weights and capture_descriptors for analysis.
"""

import numpy as np

import mindspore as ms
from mindspore import mint, ops

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress

from sharker.data import Graph


class HTGP_Calculator(Calculator):
    """
    ASE Calculator that uses the same energy/strain logic as PotentialTrainer.

    Computes energy; forces via -dE/dpos; stress (if PBC) via strain derivative.
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "descriptors",
        "weights",
    ]

    def __init__(self, model, cutoff=6.0, **kwargs):
        """
        Args:
            model: HTGPModel instance (MindSpore).
            cutoff: Cutoff for neighbor list; must match training.
            **kwargs: Passed to ASE Calculator; may include capture_weights,
                capture_descriptors.
        """
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.cutoff = cutoff

        self.model.set_train(False)
        self.capture_weights = kwargs.get("capture_weights", False)
        self.capture_descriptors = kwargs.get("capture_descriptors", False)

        for param in self.model.get_parameters():
            param.requires_grad = False

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = ["energy", "forces", "stress"]
        Calculator.calculate(self, atoms, properties, system_changes)

        data = self._atoms_to_pyg_data(atoms)
        original_pos = data.pos
        original_cell = getattr(data, "cell", None)
        displacement = None

        is_periodic = atoms.pbc.any()
        calc_stress = "stress" in properties and is_periodic

        data.pos.requires_grad = True

        if calc_stress:
            displacement = mint.zeros(
                (1, 3, 3), dtype=data.pos.dtype
            )
            displacement.requires_grad = True
            symmetric_strain = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            strain_on_graph = symmetric_strain[0]
            pos_deformed = original_pos + mint.matmul(
                original_pos, strain_on_graph.T
            )
            data.pos = pos_deformed

            if original_cell is not None:
                cell_deformed = original_cell + mint.matmul(
                    original_cell, symmetric_strain
                )
                data.cell = cell_deformed

            inputs_to_grad = [original_pos, displacement]
        else:
            inputs_to_grad = [data.pos]
            data.cell = original_cell

        energy = self.model(
            data,
            capture_weights=self.capture_weights,
            capture_descriptors=self.capture_descriptors,
        )

        self.results["energy"] = energy.asnumpy().item()

        grads = ms.grad(
            outputs=energy,
            inputs=inputs_to_grad,
            retain_graph=False,
            create_graph=False,
        )

        forces = -grads[0]
        self.results["forces"] = forces.asnumpy()

        if calc_stress:
            dE_dStrain = grads[1]
            _, logabsdet = ops.linalg.slogdet(original_cell[0])
            volume = mint.exp(logabsdet)

            if volume > 1e-8:
                stress_tensor = dE_dStrain / volume
                stress_np = stress_tensor.squeeze(0).asnumpy()
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    stress_np
                )
            else:
                self.results["stress"] = np.zeros(6)

        if self.capture_weights:
            self.results["weights"] = self._get_weights()
        if self.capture_descriptors:
            self.results["descriptors"] = self._get_descriptors()
        if getattr(self, "get_charges", False):
            self.results["charges"] = self._get_charges()

    def _get_weights(self):
        """
        Extract captured gating weights from each layer's PhysicsGating.

        Returns:
            List of dicts (one per layer) with g0, g1, g2, chem_logits,
            phys_logits, scalar_basis, p_ij (numpy).
        """
        weights_per_layer = []

        def extract(module, attr_name):
            if hasattr(module, attr_name):
                val = getattr(module, attr_name)
                if val is not None:
                    return val.asnumpy()
            return None

        for layer in self.model.layers:
            gating_module = layer["gating"]
            layer_data = {
                "g0": extract(gating_module, "g0_captured"),
                "g1": extract(gating_module, "g1_captured"),
                "g2": extract(gating_module, "g2_captured"),
                "chem_logits": extract(gating_module, "chem_logits_captured"),
                "phys_logits": extract(gating_module, "phys_logits_captured"),
                "scalar_basis": extract(gating_module, "scalar_basis_captured"),
                "p_ij": extract(gating_module, "p_ij_captured"),
            }
            weights_per_layer.append(layer_data)

        return weights_per_layer

    def _get_descriptors(self):
        """
        Return per-layer atom features (h0, h1, h2) from model.

        Uses model.all_layer_descriptors filled when capture_descriptors=True.
        """
        if not hasattr(self.model, "all_layer_descriptors"):
            return None
        out = []
        for layer_feats in self.model.all_layer_descriptors:
            layer_dict = {}
            for key, val in layer_feats.items():
                if val is not None:
                    layer_dict[key] = (
                        val.asnumpy()
                        if hasattr(val, "asnumpy")
                        else np.array(val)
                    )
                else:
                    layer_dict[key] = None
            out.append(layer_dict)
        return out

    def _get_charges(self):
        """
        Return charges if the model exposes them (e.g. from long-range module).
        """
        if not hasattr(self.model, "charge"):
            return None
        return self.model.charge.asnumpy()

    def _atoms_to_pyg_data(self, atoms):
        """Build a single Graph from ASE atoms (one graph, batch size 1)."""
        z = ms.Tensor(atoms.get_atomic_numbers()).astype(ms.int32)
        pos = ms.Tensor(atoms.get_positions()).astype(ms.float32)

        if atoms.pbc.any():
            cell_np = atoms.get_cell().array
            if np.abs(np.linalg.det(cell_np)) > 1e-6:
                cell = (
                    ms.Tensor(cell_np)
                    .astype(ms.float32)
                    .unsqueeze(0)
                )
            else:
                cell = None
        else:
            cell = None

        i_idx, j_idx, _, S_integers = neighbor_list(
            "ijdS", atoms, self.cutoff
        )
        edge_index = ms.Tensor(
            np.vstack((i_idx, j_idx)), dtype=ms.int32
        )
        shifts_int = ms.Tensor(S_integers).astype(ms.float32)

        num_atoms = len(atoms)
        batch = mint.zeros(num_atoms, dtype=ms.int32)

        data = Graph(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            shifts_int=shifts_int,
            batch=batch,
        )
        data.num_graphs = 1
        return data

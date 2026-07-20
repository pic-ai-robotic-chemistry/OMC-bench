import numpy as np
import torch
from ase.calculators.calculator import (
    Calculator, all_changes, PropertyNotImplementedError
)
from ase.stress import full_3x3_to_voigt_6_stress
from torch_geometric.data import Data

from .Utils import DEFAULT_FLOAT_DTYPE, DEFAULT_NP_FLOAT_DTYPE
from .neighbors import neighbor_list


class AtomBitCalculator(Calculator):
    """ASE calculator for both total-energy and residual-energy models.

    When ``add_e0=True``, model output is interpreted as residual energy and
    total energy is reconstructed as ``E_total = E_model + sum_i E0[z_i]``.
    Forces and stress are unchanged because E0 has no coordinate dependence.
    """

    def __init__(
        self,
        model,
        cutoff=6.0,
        device="cuda",
        enable_stress=True,
        add_e0=False,
        e0_dict=None,
        e0_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.cutoff = float(cutoff)
        self.enable_stress = bool(enable_stress)
        self.add_e0 = bool(add_e0)
        self.e0_dict = self._load_e0_dict(e0_dict=e0_dict, e0_path=e0_path)

        # ASE optimizer 常会尝试 force_consistent energy，所以 free_energy 一并给
        self.implemented_properties = ["energy", "forces", "free_energy"]
        if self.enable_stress:
            self.implemented_properties.append("stress")

        for p in self.model.parameters():
            p.requires_grad_(False)

        # 这两个在 relax 过程中通常是静态的，缓存起来
        self._cached_natoms = None
        self._cached_numbers = None
        self._z = None
        self._batch = None

        self.capture_weights = kwargs.get("capture_weights", False)
        self.capture_descriptors = kwargs.get("capture_descriptors", False)
        self.capture_charges = kwargs.get("capture_charges", False)

    def _load_e0_dict(self, e0_dict=None, e0_path=None):
        if e0_dict is not None and e0_path is not None:
            raise ValueError("Specify either e0_dict or e0_path, not both.")
        if e0_path is not None:
            payload = torch.load(e0_path, map_location="cpu", weights_only=False)
            return payload.get("e0_dict", {})
        return {} if e0_dict is None else dict(e0_dict)

    def _compute_e0_sum(self, atomic_numbers):
        if not self.add_e0:
            return 0.0

        total = 0.0
        for z in atomic_numbers:
            total += float(self.e0_dict.get(int(z), 0.0))
        return total

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        want_stress = "stress" in properties
        # FrechetCellFilter 通常先要 stress，后要 forces；
        # 这里顺手把 forces 也存起来，避免同一步重复 forward
        store_forces = ("forces" in properties) or want_stress

        data = self._atoms_to_pyg_data(atoms)
        original_cell = data.cell

        calc_stress = (
            want_stress
            and self.enable_stress
            and atoms.pbc.any()
            and original_cell is not None
        )
        if want_stress and not calc_stress:
            raise PropertyNotImplementedError("stress is not available for this configuration")

        if store_forces:
            data.pos.requires_grad_(True)
        original_pos = data.pos

        displacement = None
        if calc_stress:
            displacement = torch.zeros(
                (1, 3, 3),
                dtype=data.pos.dtype,
                device=data.pos.device,
                requires_grad=True,
            )
            sym_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
            data.pos = original_pos + original_pos @ sym_strain[0].T
            data.cell = original_cell + original_cell @ sym_strain

        result = self.model(
            data,
            capture_weights=self.capture_weights,
            capture_descriptors=self.capture_descriptors,
        )

        if isinstance(result, dict):
            model_energy = result["energy"]
            direct_force = result.get("force", None)
        else:
            model_energy = result
            direct_force = None

        grad_inputs = []
        if store_forces and direct_force is None:
            grad_inputs.append(original_pos)
        if calc_stress:
            grad_inputs.append(displacement)

        grads = ()
        if grad_inputs:
            grads = torch.autograd.grad(
                outputs=model_energy,
                inputs=grad_inputs,
                grad_outputs=torch.ones_like(model_energy),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

        gidx = 0

        if store_forces:
            if direct_force is not None:
                forces = direct_force
            else:
                forces = -grads[gidx]
                gidx += 1
            self.results["forces"] = forces.detach().cpu().numpy()

        if calc_stress:
            dE_dstrain = grads[gidx]  # (1, 3, 3)
            volume = torch.abs(torch.det(original_cell[0])).clamp_min(1e-12)
            stress_3x3 = (dE_dstrain[0] / volume).detach().cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress_3x3)

        # 如果你后面 profile 发现这里同步很贵，可以只在请求 energy 时再填
        e = float(model_energy.detach().cpu())
        if self.add_e0:
            e += self._compute_e0_sum(atoms.numbers)
        self.results["energy"] = e
        self.results["free_energy"] = e

        if self.capture_weights:
            self.results["weights"] = self._get_weights()

        if self.capture_descriptors:
            self.results["descriptors"] = self._get_descriptors()
            self.results["edge_info"] = self._get_edge_info()

        if self.capture_charges:
            self.results["charges"] = self._get_charges()

    def _atoms_to_pyg_data(self, atoms):
        natoms = len(atoms)

        # 1) 直接拿 ASE 内部数组，避免 get_* 的额外 copy
        numbers_np = np.asarray(atoms.numbers, dtype=np.int64, order="C")
        pos_np = np.asarray(atoms.positions, dtype=DEFAULT_NP_FLOAT_DTYPE, order="C")

        # 2) z 和 batch 基本是静态量，缓存
        if (
            self._z is None
            or self._batch is None
            or self._cached_natoms != natoms
            or self._cached_numbers is None
            or not np.array_equal(self._cached_numbers, numbers_np)
        ):
            self._z = torch.from_numpy(numbers_np).to(
                device=self.device, dtype=torch.long
            )
            self._batch = torch.zeros(natoms, dtype=torch.long, device=self.device)
            self._cached_natoms = natoms
            self._cached_numbers = numbers_np.copy()

        pos = torch.from_numpy(pos_np).to(
            device=self.device, dtype=DEFAULT_FLOAT_DTYPE
        )

        cell = None
        if atoms.pbc.any():
            cell_np = np.asarray(atoms.cell.array, dtype=DEFAULT_NP_FLOAT_DTYPE)
            if abs(np.linalg.det(cell_np)) > 1e-12:
                cell = torch.from_numpy(cell_np).to(
                    device=self.device, dtype=DEFAULT_FLOAT_DTYPE
                ).unsqueeze(0)

        # 3) 只请求真正用到的量：ijS，不要 d
        i_idx, j_idx, S_int = neighbor_list("ijS", atoms, self.cutoff)

        edge_index_np = np.stack([i_idx, j_idx], axis=0).astype(np.int64, copy=False)
        shifts_np = np.asarray(S_int, dtype=DEFAULT_NP_FLOAT_DTYPE, order="C")

        edge_index = torch.from_numpy(edge_index_np).to(
            device=self.device, dtype=torch.long
        )
        shifts_int = torch.from_numpy(shifts_np).to(
            device=self.device, dtype=DEFAULT_FLOAT_DTYPE
        )

        data = Data(
            z=self._z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            shifts_int=shifts_int,
            batch=self._batch,
        )
        data.num_graphs = 1
        return data

    def _get_edge_info(self):
        """
        从模型中提取 edge_info，例如 `d_ji`, `center_z`, `nbr_z`。
        """
        if not hasattr(self.model, 'edge_info') or not self.model.edge_info:
            return None

        info_dict = {}
        # 遍历 model.edge_info 中的所有键值对
        for key, val in self.model.edge_info.items():
            if val is not None:
                # 已经是 Tensor 且 detach().cpu() 过的，直接转 numpy
                if isinstance(val, torch.Tensor):
                    info_dict[key] = val.numpy()
                else:
                    info_dict[key] = val
            else:
                info_dict[key] = None

        return info_dict


    def _get_weights(self):
        """
        从模型各层的 PhysicsGating 模块中提取捕获的权重。
        返回: List[Dict]，列表索引对应层数
        """
        weights_per_layer = []

        # 遍历模型的每一层
        for i, layer in enumerate(self.model.layers):
            # 'gating' 是 ModuleDict 中的 key，对应 PhysicsGating 实例
            gating_module = layer['gating']

            layer_data = {}

            # 辅助函数：如果属性存在且不为None，转numpy
            def extract(attr_name):
                if hasattr(gating_module, attr_name):
                    val = getattr(gating_module, attr_name)
                    if val is not None:
                        return val.detach().cpu().numpy()
                return None

            # 提取你在 PhysicsGating 中定义的捕获变量
            layer_data['g0'] = extract('g0_captured')
            layer_data['g1'] = extract('g1_captured')
            layer_data['g2'] = extract('g2_captured')
            layer_data['chem_logits'] = extract('chem_logits_captured')
            layer_data['phys_logits'] = extract('phys_logits_captured')
            layer_data['scalar_basis'] = extract('scalar_basis_captured') # 如有需要可取消注释
            layer_data['p_ij'] = extract('p_ij_captured') # 如有需要可取消注释

            weights_per_layer.append(layer_data)

        return weights_per_layer


    def _get_descriptors(self):
        """
        从模型中提取每一层的原子特征 (h0, h1, h2)。
        你的模型代码里已经把它们存到了 self.model.all_layer_descriptors 列表里。
        """
        return self._extract_layer_feature_list("all_layer_descriptors")


    def _get_charges(self):
        """
        从模型中提取电荷相关特征。
        优先读取 all_layer_charges，若不存在则回退到 all_layer_descriptors。
        """
        charges = self._extract_layer_feature_list("all_layer_charges")
        if charges is not None:
            return charges
        return self._extract_layer_feature_list("all_layer_descriptors")

    def _extract_layer_feature_list(self, attr_name: str):
        if not hasattr(self.model, attr_name):
            return None

        out = []
        for layer_feats in getattr(self.model, attr_name):
            layer_dict = {}
            for key, val in layer_feats.items():
                if isinstance(val, torch.Tensor):
                    layer_dict[key] = val.detach().cpu().numpy()
                else:
                    layer_dict[key] = val
            out.append(layer_dict)
        return out

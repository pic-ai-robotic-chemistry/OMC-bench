import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ChunkedSmartDataset(Dataset):
    def __init__(self, data_dir, metadata_file, rank=0, world_size=1, cast_float_dtype=None):
        self.data_dir = data_dir
        self.metadata = torch.load(os.path.join(data_dir, metadata_file), weights_only=False)
        self.cast_float_dtype = cast_float_dtype
        if rank == 0:
            print(f"Loading metadata from {os.path.join(data_dir, metadata_file)}...")

    def _maybe_cast(self, tensor):
        if self.cast_float_dtype is None:
            return tensor
        if torch.is_floating_point(tensor):
            return tensor.to(self.cast_float_dtype)
        return tensor

    def _runtime_float_dtype(self, *tensors):
        if self.cast_float_dtype is not None:
            return self.cast_float_dtype
        for tensor in tensors:
            if tensor is not None and torch.is_floating_point(tensor):
                return tensor.dtype
        return torch.float64

    def __getitem__(self, idx):
        info = self.metadata[idx]
        file_name = info["file_path"]
        if file_name.endswith(".pt"):
            file_name = file_name.replace(".pt", ".h5")
        full_path = os.path.join(self.data_dir, file_name)
        inner_idx = info["index_in_file"]

        with h5py.File(full_path, "r") as h5f:
            atom_start = h5f["atom_ptr"][inner_idx]
            atom_end = h5f["atom_ptr"][inner_idx + 1]
            edge_start = h5f["edge_ptr"][inner_idx]
            edge_end = h5f["edge_ptr"][inner_idx + 1]

            z = torch.from_numpy(h5f["z"][atom_start:atom_end].astype(np.int64, copy=False))
            pos = torch.from_numpy(h5f["pos"][atom_start:atom_end])
            force = torch.from_numpy(h5f["force"][atom_start:atom_end])
            edge_index = torch.from_numpy(h5f["edge_index"][:, edge_start:edge_end].astype(np.int64, copy=False))
            y = torch.from_numpy(h5f["y"][inner_idx])
            cell = torch.from_numpy(h5f["cell"][inner_idx])
            float_dtype = self._runtime_float_dtype(pos, force, y, cell)
            shifts_int = torch.from_numpy(h5f["shifts_int"][edge_start:edge_end]).to(float_dtype)

            data = Data(
                z=z,
                pos=self._maybe_cast(pos),
                cell=self._maybe_cast(cell),
                edge_index=edge_index,
                shifts_int=shifts_int,
                y=self._maybe_cast(y),
                force=self._maybe_cast(force),
            )

            if bool(h5f.attrs.get("has_stress", False)):
                stress = torch.from_numpy(h5f["stress"][inner_idx])
                data.stress = self._maybe_cast(stress)
            if bool(h5f.attrs.get("has_charge", False)) and "charge" in h5f:
                data.charge = self._maybe_cast(torch.from_numpy(h5f["charge"][inner_idx]))
            if bool(h5f.attrs.get("has_spin", False)) and "spin" in h5f:
                data.spin = self._maybe_cast(torch.from_numpy(h5f["spin"][inner_idx]))
            if bool(h5f.attrs.get("has_tags", False)) and "tags" in h5f:
                data.tags = torch.from_numpy(h5f["tags"][atom_start:atom_end].astype(np.int64, copy=False))
            if bool(h5f.attrs.get("has_fixed_mask", False)) and "fixed_mask" in h5f:
                data.fixed_mask = torch.from_numpy(h5f["fixed_mask"][atom_start:atom_end].astype(np.bool_, copy=False))
            if "dataset_code" in h5f:
                data.dataset_code = int(h5f["dataset_code"][inner_idx])
            data.energy_mode = h5f.attrs.get("energy_mode", "total")
            data.e0_source = h5f.attrs.get("e0_source", "")

        return data

    def __len__(self):
        return len(self.metadata)

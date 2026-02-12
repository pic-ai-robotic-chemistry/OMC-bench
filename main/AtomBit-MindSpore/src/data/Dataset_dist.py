"""
Chunked dataset backed by HDF5 for distributed training.

Loads graph samples on demand from pre-built H5 chunks using metadata
(file path + index_in_file). No in-memory cache; relies on OS page cache.
"""

import os
import pickle

import h5py
import numpy as np

from mindspore.dataset import Dataset
from sharker.data import Graph


class ChunkedSmartDataset_h5:
    """
    Dataset that reads graph data from HDF5 chunk files via a metadata index.

    Each sample is identified by metadata[idx] giving file_path and
    index_in_file. Files are opened per __getitem__ (no persistent handle)
    to avoid deadlocks with multiprocess DataLoader.
    """

    def __init__(self, data_dir, metadata_file, rank=0, world_size=1):
        """
        Args:
            data_dir: Root directory containing chunk files and metadata.
            metadata_file: Metadata filename (e.g. 'train_metadata.pt').
            rank: Rank of this process (for logging).
            world_size: Total number of processes.
        """
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, metadata_file)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.metadata = self.metadata[:1000]

        if rank == 0:
            print(f"Loading metadata from {meta_path}...")

    def __getitem__(self, idx):
        info = self.metadata[idx]
        file_name = info["file_path"]

        # Support legacy metadata that still references .pt files
        if file_name.endswith(".pt"):
            file_name = file_name.replace(".pt", ".h5")

        inner_idx = info["index_in_file"]
        full_path = os.path.join(self.data_dir, file_name)

        try:
            with h5py.File(full_path, "r") as f:
                a_start = f["atom_ptr"][inner_idx]
                a_end = f["atom_ptr"][inner_idx + 1]
                e_start = f["edge_ptr"][inner_idx]
                e_end = f["edge_ptr"][inner_idx + 1]

                z = f["z"][a_start:a_end].astype(np.int64)
                pos = f["pos"][a_start:a_end]
                force = f["force"][a_start:a_end]
                edge_index = f["edge_index"][:, e_start:e_end].astype(np.int64)
                shifts_int = f["shifts_int"][e_start:e_end].astype(np.float32)

                y = f["y"][inner_idx]
                cell = f["cell"][inner_idx]
                spin = f["spin"][inner_idx]
                charge = f["charge"][inner_idx]
                dataset = f["dataset"][inner_idx].decode("utf-8")
                stress = f["stress"][inner_idx]

            data = Graph(
                z=z,
                pos=pos,
                cell=cell,
                edge_index=edge_index,
                shifts_int=shifts_int,
                y=y,
                force=force,
                spin=spin,
                charge=charge,
                dataset=dataset,
                stress=stress,
            )

            if data.pos.dtype != np.float32:
                data.pos = data.pos.astype(np.float32)
            if data.y.dtype != np.float32:
                data.y = data.y.astype(np.float32)

            return data

        except Exception as e:
            print(f"Error reading {full_path} at index {inner_idx}: {e}")
            return Graph()

    def __len__(self):
        return len(self.metadata)

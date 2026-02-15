"""
Bin-packing sampler for distributed training.

Packs samples by cost (atoms + weighted edges) to balance memory and compute
across batches. Supports deterministic shuffling per epoch for DDP consistency.
"""

import random
from collections import defaultdict

from mindspore.dataset import Sampler


class BinPackingSampler(Sampler):
    """
    Sampler that packs dataset indices into batches under a max cost budget.

    Cost is defined as num_atoms + edge_weight * num_edges. When edge_weight
    is 'auto', it is set to total_atoms / total_edges for memory balance.
    """

    def __init__(
        self,
        metadata,
        max_cost=3000,
        edge_weight="auto",
        shuffle=True,
        world_size=1,
        rank=0,
        seed=42,
    ):
        """
        Args:
            metadata: List of dicts with 'num_atoms' and 'num_edges' per sample.
            max_cost: Maximum cost per batch (atoms + edge_weight * edges).
            edge_weight: Weight for edges in cost; use 'auto' for total_atoms/total_edges.
            shuffle: Whether to shuffle order and add small noise to costs.
            world_size: Number of distributed processes (e.g. GPUs).
            rank: Rank of this process (0 to world_size - 1).
            seed: Base random seed for deterministic behavior across ranks.
        """
        self.metadata = metadata
        self.max_cost = max_cost
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Compute edge weight for cost
        if edge_weight == "auto":
            total_atoms = 0
            total_edges = 0
            for item in metadata:
                total_atoms += item["num_atoms"]
                total_edges += item["num_edges"]

            if total_edges > 0:
                self.edge_weight = total_atoms / total_edges
            else:
                self.edge_weight = 0.0

            if self.rank == 0:
                print(
                    f"[Auto-Balance] Total Atoms: {total_atoms}, Total Edges: {total_edges}"
                )
                print(
                    f"[Auto-Balance] Calculated Edge Weight: {self.edge_weight:.6f}"
                )
                if self.edge_weight > 0:
                    print(
                        f"  (Roughly 1 atom memory per {1/self.edge_weight:.1f} edges)"
                    )
        else:
            self.edge_weight = float(edge_weight)

        # Precompute cost for each sample: cost = atoms + edge_weight * edges
        self.indices_with_cost = []
        for i, item in enumerate(metadata):
            c = item["num_atoms"] + self.edge_weight * item["num_edges"]
            self.indices_with_cost.append((i, c))

        super().__init__()

    def set_epoch(self, epoch):
        """
        Set current epoch. Called at the start of each epoch so that batch
        generation is deterministic and consistent across all ranks.
        """
        self.epoch = epoch

    def _generate_batches(self, epoch_idx):
        """
        Generate batch indices for the given epoch for this rank.

        Steps: sort by cost (with optional shuffle noise), bin-pack into
        batches under max_cost, shuffle batches, then slice for DDP (drop
        last incomplete remainder).

        Returns:
            List of batches; each batch is a list of sample indices.
        """
        rng = random.Random(self.seed + epoch_idx)

        # Sort by cost (with small random perturbation if shuffle)
        indices = self.indices_with_cost.copy()
        if self.shuffle:
            indices.sort(
                key=lambda x: x[1] * rng.uniform(0.99, 1.01), reverse=True
            )
        else:
            indices.sort(key=lambda x: x[1], reverse=True)

        # Bin packing
        batches = []
        current_batch = []
        current_batch_cost = 0

        for idx, cost in indices:
            if (
                current_batch_cost + cost > self.max_cost
                and current_batch
            ):
                batches.append(current_batch)
                current_batch = []
                current_batch_cost = 0
            current_batch.append(idx)
            current_batch_cost += cost
        if current_batch:
            batches.append(current_batch)

        # Shuffle batch order
        if self.shuffle:
            rng.shuffle(batches)

        # DDP: drop last incomplete set of batches, then assign to ranks
        total_batches = len(batches)
        num_samples_per_rank = total_batches // self.world_size
        batches = batches[: num_samples_per_rank * self.world_size]
        my_batches = batches[self.rank :: self.world_size]

        return my_batches

    def __iter__(self):
        batches = self._generate_batches(self.epoch)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self._generate_batches(self.epoch))

    def precompute_total_steps(self, total_epochs):
        """
        Compute the exact total number of steps (batches) over all epochs.

        Useful for schedulers and progress bars. Simulates batch generation
        for each epoch without loading data.

        Args:
            total_epochs: Number of training epochs.

        Returns:
            Total number of batches across all epochs for this rank.
        """
        if self.rank == 0:
            print(f"Pre-computing exact steps for {total_epochs} epochs...")

        total_steps = 0
        for ep in range(total_epochs):
            batches = self._generate_batches(ep)
            total_steps += len(batches)

        if self.rank == 0:
            avg = total_steps / total_epochs if total_epochs else 0
            print(f"Exact total steps: {total_steps} (avg per epoch: {avg:.1f})")

        return total_steps

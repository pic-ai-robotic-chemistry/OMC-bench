import torch
import random
from torch.utils.data import Sampler

class NormalSampler(Sampler):
    def __init__(self, metadata, max_cost=3000, edge_weight='auto', shuffle=True,
                 world_size=1, rank=0, seed=42):
        """
        [普通版修改] 接口完全保持不变，但内部逻辑不再进行 Bin Packing 排序。
        """
        self.metadata = metadata
        self.max_cost = max_cost
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # ---------------------------------------------------
        # 1. 计算权重 (逻辑保持不变，为了兼容 max_cost 参数)
        # ---------------------------------------------------
        if edge_weight == 'auto':
            total_atoms = 0
            total_edges = 0
            for item in metadata:
                total_atoms += item['num_atoms']
                total_edges += item['num_edges']

            if total_edges > 0:
                self.edge_weight = total_atoms / total_edges
            else:
                self.edge_weight = 0.0

            # 仅在主进程打印
            if self.rank == 0:
                print(f"⚖️ [Normal-Mode] Total Atoms: {total_atoms}, Total Edges: {total_edges}")
                print(f"⚖️ [Normal-Mode] Calculated Edge Weight: {self.edge_weight:.6f}")
        else:
            self.edge_weight = float(edge_weight)

        # ---------------------------------------------------
        # 2. 预计算所有 Cost
        # ---------------------------------------------------
        self.indices_with_cost = []
        for i, item in enumerate(metadata):
            c = item['num_atoms'] + self.edge_weight * item['num_edges']
            self.indices_with_cost.append((i, c))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _generate_batches(self, epoch_idx):
        """
        🔥 核心修改：移除排序逻辑，改为完全随机
        """
        rng = random.Random(self.seed + epoch_idx)

        # 1. 复制列表
        indices = self.indices_with_cost.copy()

        # -------------------------------------------------------
        # 🔥 修改点：不再按 cost 排序 (sort)，而是直接随机打乱 (shuffle)
        # 这就变成了普通的 RandomSampler，只是依然受 max_cost 显存限制
        # -------------------------------------------------------
        if self.shuffle:
            rng.shuffle(indices)
        # 如果不 shuffle，那就按原数据集顺序，也不排序

        # 2. 顺序装填 (不再是装箱算法，而是单纯的 FIFO 截断)
        batches = []
        current_batch = []
        current_batch_cost = 0

        for idx, cost in indices:
            # 如果加上当前样本会超显存，就切断当前 batch
            if current_batch_cost + cost > self.max_cost and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_cost = 0

            current_batch.append(idx)
            current_batch_cost += cost

        if current_batch:
            batches.append(current_batch)

        # 3. Batch 间 Shuffle (保持原有逻辑，增强随机性)
        if self.shuffle:
            rng.shuffle(batches)

        # 4. DDP 切片 (Drop Last 逻辑保持不变)
        total_batches = len(batches)
        num_samples_per_rank = total_batches // self.world_size
        batches = batches[:num_samples_per_rank * self.world_size]
        my_batches = batches[self.rank::self.world_size]

        return my_batches

    def __iter__(self):
        batches = self._generate_batches(self.epoch)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self._generate_batches(self.epoch))

    def precompute_total_steps(self, total_epochs):
        if self.rank == 0:
            print(f"🔄 [Normal-Mode] Pre-computing exact steps for {total_epochs} epochs...")

        total_steps = 0
        for ep in range(total_epochs):
            batches = self._generate_batches(ep)
            total_steps += len(batches)

        if self.rank == 0:
            print(f"✅ Exact total steps: {total_steps} (Avg: {total_steps/total_epochs:.1f})")

        return total_steps

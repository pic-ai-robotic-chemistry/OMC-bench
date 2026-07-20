import torch
import random
from torch.utils.data import Sampler

class BatchSampler(Sampler):
    def __init__(self, metadata, max_cost=3000, edge_weight='auto', shuffle=True,
                 world_size=1, rank=0, seed=42):
        """
        【固定 Batch Size 模式】
        虽然名字还叫 BinPackingSampler，但逻辑已经改成了普通的 Fixed Batch Size Sampler。
        接口保持不变，以兼容现有的 train.py。
        """
        self.metadata = metadata
        self.max_cost = max_cost # 注意：在这个模式下，max_cost 参数将被忽略！
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # ---------------------------------------------------
        # 1. 计算权重 (保留这些代码是为了不改变类结构，实际没用到)
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

            if self.rank == 0:
                print(f"⚖️ [Fixed-Batch-Mode] Initialized. (Note: max_cost={max_cost} will be IGNORED)")
        else:
            self.edge_weight = float(edge_weight)

        # ---------------------------------------------------
        # 2. 预计算 Cost (保留结构，但在这个模式下只用索引)
        # ---------------------------------------------------
        self.indices_with_cost = []
        for i, item in enumerate(metadata):
            # 这里的 Cost 计算已经不重要了，因为我们只看数量
            c = item['num_atoms'] + self.edge_weight * item['num_edges']
            self.indices_with_cost.append((i, c))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _generate_batches(self, epoch_idx):
        """
        🔥 核心修改：忽略 Cost，强制使用固定的 Batch Size
        """
        rng = random.Random(self.seed + epoch_idx)

        # -----------------------------------------------------------
        # 🔧在此处修改你的固定 Batch Size
        # 根据你之前的日志，你的显卡大约能跑 15-19 个图，所以我设为 16
        # -----------------------------------------------------------
        FIXED_BATCH_SIZE = 16

        # 1. 复制索引
        indices = [x[0] for x in self.indices_with_cost] # 只取 index，不要 cost

        # 2. 必须乱序 (这是普通 Sampler 的特征)
        if self.shuffle:
            rng.shuffle(indices)

        # 3. 按固定数量切分 (Chunking)
        batches = []
        current_batch = []

        for idx in indices:
            current_batch.append(idx)

            # 🔥 只要数量够了就打包，完全不管显存会不会爆 (这就是普通 Sampler 的风险)
            if len(current_batch) == FIXED_BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []

        # 处理剩下的尾巴
        if current_batch:
            # Drop Last 逻辑：如果剩下的太少（比如少于一半），为了稳定性通常可以丢掉
            # 这里为了简单，我们还是保留它，或者你可以选择 batches.append(current_batch)
            if len(current_batch) >= (FIXED_BATCH_SIZE // 2):
                batches.append(current_batch)

        # 4. Batch 间 Shuffle
        if self.shuffle:
            rng.shuffle(batches)

        # 5. DDP 切片
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
            print(f"🔄 [Fixed-Batch-Mode] Pre-computing steps (Fixed BS)...")

        total_steps = 0
        for ep in range(total_epochs):
            batches = self._generate_batches(ep)
            total_steps += len(batches)

        if self.rank == 0:
            print(f"✅ Exact total steps: {total_steps}")

        return total_steps

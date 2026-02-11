import torch
import random
from torch.utils.data import Sampler

class BinPackingSampler(Sampler):
    def __init__(self, metadata, max_cost=3000, edge_weight='auto', shuffle=True, 
                 world_size=1, rank=0, seed=42): # 🔥 新增 seed 参数
        """
        :param seed: 基础随机种子，保证 DDP 各卡初始状态一致
        """
        self.metadata = metadata
        self.max_cost = max_cost
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.seed = seed      # 🔥 保存种子
        self.epoch = 0        # 🔥 新增 epoch 计数器
        
        # ---------------------------------------------------
        # 1. 计算权重 (逻辑保持不变)
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
                print(f"⚖️ [Auto-Balance] Total Atoms: {total_atoms}, Total Edges: {total_edges}")
                print(f"⚖️ [Auto-Balance] Calculated Edge Weight: {self.edge_weight:.6f}")
                print(f"   (这意味着每 {1/self.edge_weight:.1f} 条边 ≈ 1 个原子的显存消耗)")
        else:
            self.edge_weight = float(edge_weight)

        # ---------------------------------------------------
        # 2. 预计算所有 Cost
        # ---------------------------------------------------
        self.indices_with_cost = []
        for i, item in enumerate(metadata):
            # Cost = Atoms + 权重 * Edges
            c = item['num_atoms'] + self.edge_weight * item['num_edges']
            self.indices_with_cost.append((i, c))

    def set_epoch(self, epoch):
        """
        🔥 关键方法：在每个 Epoch 开始前调用，
        确保每一轮的随机扰动不同，但在所有 GPU 上是一致的。
        """
        self.epoch = epoch

    def _generate_batches(self, epoch_idx):
        """
        🔥 核心修改：将生成逻辑抽离，使其可以被模拟调用
        返回：当前 Rank 在指定 epoch 应该拿到的 batch 列表
        """
        rng = random.Random(self.seed + epoch_idx) # 确定性随机

        # 1. 复制并排序
        indices = self.indices_with_cost.copy()
        if self.shuffle:
            indices.sort(key=lambda x: x[1] * rng.uniform(0.99, 1.01), reverse=True)
        else:
            indices.sort(key=lambda x: x[1], reverse=True)

        # 2. 装箱
        batches = []
        current_batch = []
        current_batch_cost = 0
        
        for idx, cost in indices:
            if current_batch_cost + cost > self.max_cost and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_cost = 0
            current_batch.append(idx)
            current_batch_cost += cost
        if current_batch:
            batches.append(current_batch)

        # 3. Batch 间 Shuffle
        if self.shuffle:
            rng.shuffle(batches)

        # 4. DDP 切片 (Drop Last 逻辑)
        total_batches = len(batches)
        num_samples_per_rank = total_batches // self.world_size
        batches = batches[:num_samples_per_rank * self.world_size]
        my_batches = batches[self.rank::self.world_size]
        
        return my_batches

    def __iter__(self):
        # 直接调用抽离的逻辑
        batches = self._generate_batches(self.epoch)
        for batch in batches:
            yield batch

    def __len__(self):
        # 这个 len 依然只能返回估计值或当前 epoch 的值
        # 但既然我们要精确计算总步数，这个 len 对 Scheduler 已经不重要了，只对 tqdm 有用
        return len(self._generate_batches(self.epoch))

    def precompute_total_steps(self, total_epochs):
        """
        🔥 新增方法：精确计算未来所有 Epoch 的步数总和
        """
        if self.rank == 0:
            print(f"🔄 Pre-computing exact steps for {total_epochs} epochs...")
        
        total_steps = 0
        for ep in range(total_epochs):
            # 模拟生成每一轮的 batch (计算极快，因为只是操作整数列表)
            batches = self._generate_batches(ep)
            total_steps += len(batches)
            
        if self.rank == 0:
            print(f"✅ Exact total steps: {total_steps} (Avg: {total_steps/total_epochs:.1f})")
        
        return total_steps
import torch
import os
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch_geometric.data import Data

class ChunkedSmartDataset(Dataset):
    def __init__(self, data_dir, metadata_file, cache_size=16, rank=0, world_size=1):
        """
        :param data_dir: 数据目录
        :param metadata_file: 元数据文件名 (e.g., 'train_metadata.pt')
        :param cache_size: 内存中最多缓存多少个 .pt 文件块 (根据你的内存大小调整)
        """
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, metadata_file)
        
        
        self.metadata = torch.load(meta_path, weights_only=False)
        if rank == 0:
            print(f"📂 Loading metadata from {meta_path}...")

        # 简单的 LRU 缓存
        self.cache = {} 
        self.cache_keys = [] # 记录顺序
        self.max_cache = cache_size

    def _load_chunk(self, filename):
        # 1. 命中缓存
        if filename in self.cache:
            # 移到队尾表示最近使用
            self.cache_keys.remove(filename)
            self.cache_keys.append(filename)
            return self.cache[filename]
        
        # 2. 未命中，加载新文件
        full_path = os.path.join(self.data_dir, filename)
        try:
            chunk_data = torch.load(full_path, weights_only=False)
        except Exception as e:
            print(f"❌ Error loading chunk {filename}: {e}")
            return [] # 返回空列表防止崩溃

        # 3. 更新缓存
        if len(self.cache_keys) >= self.max_cache:
            # 移除最久未使用的
            oldest = self.cache_keys.pop(0)
            del self.cache[oldest]
            
        self.cache[filename] = chunk_data
        self.cache_keys.append(filename)
        return chunk_data

    def __getitem__(self, idx):
        # 1. 查字典
        info = self.metadata[idx]
        file_name = info['file_path']
        inner_idx = info['index_in_file']
        
        # 2. 拿数据 (带缓存)
        chunk_data = self._load_chunk(file_name)
        data = chunk_data[inner_idx]

        # ========================================================
        # 🔥 强制类型修正 (保持你原有的逻辑)
        # ========================================================
        # --- A. 索引类 Int64 ---
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index = data.edge_index.to(torch.long)
        if hasattr(data, 'z') and data.z is not None:
            data.z = data.z.to(torch.long)
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            data.edge_type = data.edge_type.to(torch.long)

        # --- B. 数值类 Float32 ---
        if hasattr(data, 'pos') and data.pos is not None and data.pos.dtype != torch.float32:
            data.pos = data.pos.to(torch.float32)
        
        if hasattr(data, 'cell') and data.cell is not None and data.cell.dtype != torch.float32:
            data.cell = data.cell.to(torch.float32)

        # 处理 shifts
        if hasattr(data, 'shifts_int') and data.shifts_int is not None and data.shifts_int.dtype != torch.float32:
            data.shifts_int = data.shifts_int.to(torch.float32)
            # del data.shifts_int # 可删可不删，PyG Collate 会忽略不认识的字段
        elif hasattr(data, 'shifts') and data.shifts is not None and data.shifts.dtype != torch.float32:
            data.shifts = data.shifts.to(torch.float32)

        if hasattr(data, 'y') and data.y is not None and data.y.dtype != torch.float32:
            data.y = data.y.to(torch.float32)
        if hasattr(data, 'force') and data.force is not None and data.force.dtype != torch.float32:
            data.force = data.force.to(torch.float32)
        if hasattr(data, 'stress') and data.stress is not None and data.stress.dtype != torch.float32:
            data.stress = data.stress.to(torch.float32)

        return data

    def __len__(self):
        return len(self.metadata)
    

class ChunkedSmartDataset_h5(Dataset):
    def __init__(self, data_dir, metadata_file, rank=0, world_size=1):
        """
        :param data_dir: 数据目录
        :param metadata_file: 元数据文件名 (e.g., 'train_metadata.pt')
        """
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, metadata_file)
        
        # Metadata 依然是必需的，因为它告诉我们要去哪个文件找第 N 个样本
        # 注意：你需要确保 metadata 里的 'file_path' 后缀现在是 .h5 而不是 .pt
        # 如果你没重新生成 metadata，可能需要在这里手动 replace('.pt', '.h5')
        self.metadata = torch.load(meta_path, weights_only=False)
        
        if rank == 0:
            print(f"📂 Loading metadata from {meta_path}...")
            
        # 移除 Cache。HDF5 的 OS Page Cache 已经足够高效，
        # 且避免了 Python 对象的内存开销。

    def __getitem__(self, idx):
        # 1. 查字典
        info = self.metadata[idx]
        file_name = info['file_path']
        
        # 兼容性处理：如果你没重新生成 metadata，这里强制修正后缀
        if file_name.endswith('.pt'):
            file_name = file_name.replace('.pt', '.h5')
            
        inner_idx = info['index_in_file'] # 这是该图在 chunk 中的第几个
        full_path = os.path.join(self.data_dir, file_name)

        # 2. 打开 H5 并切片读取 (Lazy Loading)
        # 这种模式下，不要把 f 存为 self.f，否则多进程 DataLoader 会死锁
        # 每次 getitem 打开并读取是安全的，对于 SSD 来说开销很小
        try:
            with h5py.File(full_path, 'r') as f:
                # 获取指针位置
                a_start = f['atom_ptr'][inner_idx]
                a_end = f['atom_ptr'][inner_idx + 1]
                
                e_start = f['edge_ptr'][inner_idx]
                e_end = f['edge_ptr'][inner_idx + 1]
                
                # 读取数据 (Numpy Slicing) - 只有这一刻才会发生真正的 IO
                # 这里的 [()] 是 h5py 读取全部/标量的语法，切片则直接用 [start:end]
                z = torch.from_numpy(f['z'][a_start:a_end].astype(np.int64)) # PyTorch Embedding 通常需要 Long
                pos = torch.from_numpy(f['pos'][a_start:a_end])
                force = torch.from_numpy(f['force'][a_start:a_end])
                
                edge_index = torch.from_numpy(f['edge_index'][:, e_start:e_end].astype(np.int64))
                shifts_int = torch.from_numpy(f['shifts_int'][e_start:e_end].astype(np.float32)) # 转 float
                
                # Graph 级属性
                y = torch.from_numpy(f['y'][inner_idx])
                cell = torch.from_numpy(f['cell'][inner_idx])
                
                stress = None
                if f.attrs['has_stress']:
                    stress = torch.from_numpy(f['stress'][inner_idx])

            # 3. 组装 PyG Data
            data = Data(
                z=z,
                pos=pos,
                cell=cell,
                edge_index=edge_index,
                shifts_int=shifts_int,
                y=y,
                force=force
            )
            
            if stress is not None:
                data.stress = stress

            # 4. 数据类型微调 (和你原来的逻辑一致)
            # 注意：上面读取时我已经尽量转换了，这里作为双重保险
            if data.pos.dtype != torch.float32: data.pos = data.pos.to(torch.float32)
            if data.y.dtype != torch.float32: data.y = data.y.to(torch.float32)

            return data

        except Exception as e:
            # 容错处理
            print(f"❌ Error reading {full_path} at idx {inner_idx}: {e}")
            # 返回一个空的或者 Dummy 数据，避免训练中断 (根据需要调整)
            return Data()

    def __len__(self):
        return len(self.metadata)
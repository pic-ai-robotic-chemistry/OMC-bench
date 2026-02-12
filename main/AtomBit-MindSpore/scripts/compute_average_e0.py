import mindspore as ms
import numpy as np
def compute_average_e0(dataset):
    """
    使用线性回归 (Ax = b) 计算每种元素的原子平均能量。
    A: (N_samples, N_elements) 矩阵，每行是该样本中各元素的原子计数
    b: (N_samples,) 向量，总能量
    x: (N_elements,) 待求的单原子能量
    """
    print("Computing reference energies via Least Squares...")
    
    # 1. 统计系统中出现过的所有原子序数
    all_z = ms.mint.cat([data.z for data in dataset])
    unique_z = ms.mint.unique(all_z).numpy()
    z_map = {z: i for i, z in enumerate(unique_z)} # Z -> Index
    n_types = len(unique_z)
    
    # 2. 构建 A 矩阵和 b 向量
    num_samples = len(dataset)
    A = np.zeros((num_samples, n_types))
    b = np.zeros(num_samples)
    
    for i, data in enumerate(dataset):
        # 统计该样本中各元素的个数
        z_counts = ms.mint.bincount(data.z)
        for z_val, idx in z_map.items():
            if z_val < len(z_counts):
                A[i, idx] = z_counts[z_val].item()
        b[i] = data.y.item() # Total Energy
        
    # 3. 求解 Ax = b
    # 使用 lstsq 求解最小二乘解
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 4. 构建初始化字典/张量
    e0_dict = {z: e for z, e in zip(unique_z, x)}
    print("Reference Energies:", e0_dict)
    
    return e0_dict

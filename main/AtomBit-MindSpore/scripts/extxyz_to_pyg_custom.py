import mindspore as ms
from mindspore import mint
import numpy as np
from ase.io import iread  # 使用 iread 进行迭代读取，节省内存
from ase.neighborlist import neighbor_list
from sharker.data import Graph
from tqdm import tqdm

def extxyz_to_pyg_custom(xyz_file_path, cutoff=6.0, require_stress=False):
    """
    将 extxyz 文件转换为 PyG Data 列表，并进行内存和存储优化。
    
    Args:
        xyz_file_path (str): 文件路径
        cutoff (float): 邻居搜索截断半径
        require_stress (bool): 如果为 True，则丢弃没有 Stress 信息的帧。默认为 False。
    """
    data_list = []
    
    # 使用 iread 迭代器，避免一次性加载大文件导致内存爆炸
    try:
        frames_iter = iread(xyz_file_path, index=':')
    except Exception as e:
        print(f"❌ 读取错误 {xyz_file_path}: {e}")
        return []

    for atoms in frames_iter:
        try:
            # --- A. 基础张量 (压缩优化) ---
            
            # 1. 原子序数: int32 -> int8 (最大支持 118 号元素 Oganesson)
            z = ms.Tensor(atoms.get_atomic_numbers()).astype(ms.int8)

            # 2. 坐标: float32 (为了精度，坐标通常建议保留 float64)
            pos = ms.Tensor(atoms.get_positions()).astype(ms.float32)
            
            # 3. 晶胞: float32 -> (1, 3, 3)
            # 注意: array 是 numpy 数组，使用 from_numpy 避免不必要的复制
            cell = ms.Tensor(atoms.get_cell().array).astype(ms.float32).unsqueeze(0)

            # --- B. 标签 (Energy, Force, Stress) ---
            
            # 能量: float32 -> (1,)
            energy_val = atoms.info.get('REF_energy', 0.0)
            y = ms.Tensor([energy_val], dtype=ms.float32)
            
            # 力: float32 (与 pos 同维度)
            forces_np = atoms.arrays.get('REF_forces')
            if forces_np is None:
                # 如果没有力，填充 0 或者选择跳过，这里选择填充 0 并保持形状
                forces = mint.zeros_like(pos)
            else:
                forces = ms.Tensor(forces_np).astype(ms.float32)
            
            # 应力 (Stress) 处理
            stress_tensor = None
            stress_info = atoms.info.get('REF_stress', None)
            
            if stress_info is not None:
                # 处理字符串或列表格式
                if isinstance(stress_info, str):
                    s_list = np.fromstring(stress_info, sep=' ')
                else:
                    s_list = np.array(stress_info)
                
                # Voigt Order 转换 (ASE 标准顺序)
                if len(s_list) == 9:
                    s_mat = s_list.reshape(3, 3)
                elif len(s_list) == 6:
                    # Voigt: xx, yy, zz, yz, xz, xy
                    s_mat = np.array([
                        [s_list[0], s_list[5], s_list[4]],
                        [s_list[5], s_list[1], s_list[3]],
                        [s_list[4], s_list[3], s_list[2]]
                    ])
                else:
                    s_mat = np.zeros((3, 3))
                
                # float32 足够应力精度，且为了和 PyTorch 默认行为一致
                stress_tensor = ms.Tensor(s_mat).astype(ms.float32).unsqueeze(0)
            
            # 策略：如果强制要求 Stress 但当前帧没有，则跳过
            if require_stress and stress_tensor is None:
                continue

            # --- C. 纯几何图构建 (邻居列表) ---
            # 使用 ASE 的 neighbor_list
            i_idx, j_idx, _, S_integers = neighbor_list('ijdS', atoms, cutoff)
            
            # [关键压缩 1] edge_index: int32
            # 优化：直接从 numpy 转换，比 stack 快
            edge_index = ms.Tensor(np.vstack((i_idx, j_idx)), dtype=ms.int32)
            
            # [关键压缩 2] shifts_int: int16 (比 int8 更安全)
            # S_integers 虽然通常很小，但极端情况下可能超过 127，int16 更稳妥
            shifts_int = ms.Tensor(S_integers).astype(ms.int32) # 这里用int32兼容性最好，int16也可以
            
            # [注] 不保存 distances，训练时根据 pos 和 shifts 实时计算:
            # vec_ij = pos[j] - pos[i] + shifts @ cell
            # dist = vec_ij.norm(dim=-1)
            
            # --- D. 组装 Data 对象 ---
            data = Graph(
                z=z, 
                pos=pos, 
                cell=cell, 
                edge_index=edge_index, 
                shifts_int=shifts_int, 
                y=y, 
                force=forces
            )
            
            if stress_tensor is not None:
                data.stress = stress_tensor
                
            data_list.append(data)
            
        except Exception as frame_e:
            # 捕获单帧处理错误，不中断整个文件
            # print(f"⚠️ 跳过坏帧 in {xyz_file_path}: {frame_e}")
            continue
            
    return data_list
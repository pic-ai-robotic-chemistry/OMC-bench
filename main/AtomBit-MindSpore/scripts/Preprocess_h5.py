# ==========================================
# 🔥 必须放在文件最最最开头！🔥
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# import torch
import pickle
import random
import multiprocessing
import gc
import h5py
import numpy as np
from tqdm.auto import tqdm
from extxyz_to_pyg_custom import extxyz_to_pyg_custom 

# ==========================================
# H5 写入核心函数
# ==========================================
def save_chunk_to_h5(data_list, save_path):
    """
    写入 H5 并返回统计数据给 Metadata
    """
    n_graphs = len(data_list)
    if n_graphs == 0: return [], 0

    n_atoms_list = [d.num_nodes for d in data_list]
    n_edges_list = [d.num_edges for d in data_list]
    
    total_atoms = sum(n_atoms_list)
    total_edges = sum(n_edges_list)
    
    # 1. 准备指针
    atom_ptr = np.cumsum([0] + n_atoms_list, dtype=np.int64)
    edge_ptr = np.cumsum([0] + n_edges_list, dtype=np.int64)
    
    # 2. 准备大数组
    all_z = np.empty((total_atoms,), dtype=np.int8)
    all_pos = np.empty((total_atoms, 3), dtype=np.float32)
    all_force = np.empty((total_atoms, 3), dtype=np.float32)
    all_edge_index = np.empty((2, total_edges), dtype=np.int32)
    all_shifts = np.empty((total_edges, 3), dtype=np.int32)
    all_y = np.empty((n_graphs, 1), dtype=np.float32)
    all_cell = np.empty((n_graphs, 1, 3, 3), dtype=np.float32)
    
    has_stress = any(hasattr(d, 'stress') and d.stress is not None for d in data_list)
    if has_stress:
        all_stress = np.empty((n_graphs, 1, 3, 3), dtype=np.float32)

    # 3. 填充数据
    for i, data in enumerate(data_list):
        a_s, a_e = atom_ptr[i], atom_ptr[i+1]
        e_s, e_e = edge_ptr[i], edge_ptr[i+1]
        
        all_z[a_s:a_e] = data.z.numpy()
        all_pos[a_s:a_e] = data.pos.numpy()
        all_force[a_s:a_e] = data.force.numpy()
        all_edge_index[:, e_s:e_e] = data.edge_index.numpy()
        all_shifts[e_s:e_e] = data.shifts_int.numpy()
        all_y[i] = data.y.numpy()
        all_cell[i] = data.cell.numpy()
        
        if has_stress:
            if hasattr(data, 'stress') and data.stress is not None:
                all_stress[i] = data.stress.numpy()
            else:
                all_stress[i] = np.zeros((1, 3, 3), dtype=np.float32)

    # 4. 写入 H5
    with h5py.File(save_path, 'w') as f:
        comp = 'lzf' 
        f.create_dataset('atom_ptr', data=atom_ptr)
        f.create_dataset('edge_ptr', data=edge_ptr)
        f.create_dataset('z', data=all_z, compression=comp)
        f.create_dataset('pos', data=all_pos, compression=comp)
        f.create_dataset('force', data=all_force, compression=comp)
        f.create_dataset('edge_index', data=all_edge_index, compression=comp)
        f.create_dataset('shifts_int', data=all_shifts, compression=comp)
        f.create_dataset('y', data=all_y)
        f.create_dataset('cell', data=all_cell)
        if has_stress:
            f.create_dataset('stress', data=all_stress)
            f.attrs['has_stress'] = True
        else:
            f.attrs['has_stress'] = False
            
    # 🔥 返回统计信息给 Metadata
    stats = []
    for i in range(n_graphs):
        stats.append((n_atoms_list[i], n_edges_list[i]))
    return stats, n_graphs

# ==========================================
# Worker
# ==========================================
def worker_task(args):
    # torch.set_num_threads(1)
    (worker_id, file_paths, save_dir, prefix, cutoff, chunk_size) = args
    buffer = []
    save_counter = 0
    generated_info = []

    try:
        for fpath in file_paths:
            if os.path.getsize(fpath) == 0: continue
            try:
                data_list = extxyz_to_pyg_custom(fpath, cutoff=cutoff)
            except: continue
            if not data_list: continue

            for data in data_list:
                buffer.append(data)
                if len(buffer) >= chunk_size:
                    fname = f"{prefix}_w{worker_id}_part_{save_counter}.h5"
                    path = os.path.join(save_dir, fname)
                    stats, count = save_chunk_to_h5(buffer, path)
                    generated_info.append({'file_name': fname, 'count': count, 'stats': stats})
                    buffer = []
                    save_counter += 1
                    gc.collect()

        if len(buffer) > 0:
            fname = f"{prefix}_w{worker_id}_part_{save_counter}.h5"
            path = os.path.join(save_dir, fname)
            stats, count = save_chunk_to_h5(buffer, path)
            generated_info.append({'file_name': fname, 'count': count, 'stats': stats})
            buffer = []
            gc.collect()

        return generated_info
    except Exception as e:
        print(f"❌ Worker-{worker_id} Error: {e}")
        return []

# ==========================================
# Manager
# ==========================================
def process_manager(file_files, save_dir, prefix, num_workers, chunk_size, cutoff):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    real_workers = min(num_workers, len(file_files))
    if real_workers == 0: return

    chunked_files = np.array_split(file_files, real_workers)
    tasks = [(i, chunked_files[i].tolist(), save_dir, prefix, cutoff, chunk_size) for i in range(real_workers)]

    print(f"🚀 [Start] {prefix}: {len(file_files)} files -> {real_workers} Workers")
    all_chunks_info = []
    with multiprocessing.Pool(processes=real_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker_task, tasks), total=real_workers):
            if result: all_chunks_info.extend(result)

    print(f"📊 Generating metadata for {prefix}...")
    all_chunks_info.sort(key=lambda x: x['file_name'])
    
    metadata = []
    for info in all_chunks_info:
        fname = info['file_name']
        stats_list = info['stats']
        for i, (n_atoms, n_edges) in enumerate(stats_list):
            metadata.append({
                'file_path': fname,
                'index_in_file': i,
                'num_atoms': n_atoms, # ✅ 供 Sampler 使用
                'num_edges': n_edges  # ✅ 供 Sampler 使用
            })

    # torch.save(metadata, os.path.join(save_dir, f"{prefix}_metadata.pt"))
    with open(os.path.join(save_dir, f"{prefix}_metadata.pickle"), 'wb') as file:
        pickle.dump(metadata, file)
    print(f"✅ Metadata Saved: {len(metadata)} samples")

# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 1. 尝试修改共享策略 (防止多进程文件描述符耗尽)
    # try:
    #     torch.multiprocessing.set_sharing_strategy('file_system')
    # except:
    #     pass

    # 2. 配置路径 (根据您的实际情况修改)
    # 原代码中的路径列表
    file_dirs = [r"../../../005_all", r"../../../100_all", r"../../../outcar_selected_xyz", r"../../../xyz_grouped", r"/home/wyh/atombit/new/data"]
    SAVE_DIR = "dataset1_h5"  # 建议使用新目录
    
    # 3. 核心参数
    NUM_WORKERS = 168           # 您的原设置
    TRAIN_CHUNK_SIZE = 5000    # H5 模式下，5000 是个健康的数字
    TEST_CHUNK_SIZE = 2000
    CUTOFF = 6.0
    test_ratio = 0.05

    # 4. 收集文件 & 提取唯一标识 (Unique Names)
    # 这一步是为了防止 Data Leakage：必须按结构(stem)划分，而不是按帧划分
    all_files = []
    unique_names = set()

    for d in file_dirs:
        if os.path.exists(d):
            # 获取该目录下所有 xyz 文件
            files_in_dir = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.xyz')]
            all_files.extend(files_in_dir)

            # 提取唯一名 (通常是文件名去掉后缀)
            for f in os.listdir(d):
                if f.endswith('.xyz'):
                    # 假设文件名格式为 "structure_id.xyz"
                    unique_names.add(f.split('.')[0])
    
    print(f"📂 Found {len(all_files)} files with {len(unique_names)} unique names.")

    # 5. 划分数据集 (基于 Unique Name)
    unique_names_list = sorted(list(unique_names)) # 排序保证确定性
    random.seed(42)
    random.shuffle(unique_names_list)

    num_test = max(1, int(len(unique_names_list) * test_ratio))
    
    # 划定集合
    test_names_set = set(unique_names_list[:num_test])
    train_names_set = set(unique_names_list[num_test:])

    # 6. 重新根据 Name 过滤文件路径
    train_files = []
    test_files = []

    for f in all_files:
        # 获取当前文件的文件名 stem
        fname = f.split(os.sep)[-1].split('.')[0]
        
        if fname in train_names_set:
            train_files.append(f)
        elif fname in test_names_set:
            test_files.append(f)

    # 简单打乱文件处理顺序 (不影响数据集归属，只影响写入 chunks 的顺序)
    random.shuffle(train_files)
    random.shuffle(test_files)

    print(f"📝 Tasks: Train={len(train_files)}, Test={len(test_files)}")
    print(f"⚙️ Config: Workers={NUM_WORKERS}, ChunkSize={TRAIN_CHUNK_SIZE}")

    # 7. 执行处理
    print("\n--- Processing Test Set ---")
    process_manager(test_files, SAVE_DIR, "test", NUM_WORKERS, TEST_CHUNK_SIZE, CUTOFF)

    print("\n--- Processing Train Set ---")
    process_manager(train_files, SAVE_DIR, "train", NUM_WORKERS, TRAIN_CHUNK_SIZE, CUTOFF)
 
    print("\n✅ All processing finished.")

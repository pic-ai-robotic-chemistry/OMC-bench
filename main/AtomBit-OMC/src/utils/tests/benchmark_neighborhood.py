import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list as ase_nl
try:
    from matscipy.neighbours import neighbour_list as matscipy_nl
    HAS_MATSCIPY = True
except ImportError:
    HAS_MATSCIPY = False
    print("Error: matscipy not installed.")

def verify_and_benchmark(n_cells=5, cutoff=5.0):
    # 1. 创建一个带有 PBC 的超胞
    atoms = bulk('Cu', 'fcc', a=3.61) * [n_cells, n_cells, n_cells]
    # 稍微扰动原子坐标，模拟真实计算情况（避免完美对称性导致某些极值情况）
    atoms.rattle(stdev=0.1)

    print(f"--- Testing {len(atoms)} atoms, Cutoff: {cutoff} Å ---")

    # 2. 调用 ASE
    i_ase, j_ase, S_ase = ase_nl("ijS", atoms, cutoff)
    print(i_ase.shape, j_ase.shape, S_ase.shape)
    # 3. 调用 matscipy
    if HAS_MATSCIPY:
        i_ms, j_ms, S_ms = matscipy_nl("ijS", atoms, cutoff)
        print(i_ms.shape, j_ms.shape, S_ms.shape)
        # --- 验证逻辑 ---
        # 由于 i, j 顺序可能不同，我们将其拼成 (N, 5) 的矩阵并排序
        def get_sorted_view(i, j, S):
            # 将 i, j 增加一个维度，与 S (N, 3) 拼接成 (N, 5)
            combined = np.column_stack([i, j, S])
            # 使用 lexsort 进行行排序
            order = np.lexsort(combined.T)
            return combined[order]

        ase_res = get_sorted_view(i_ase, j_ase, S_ase)
        ms_res = get_sorted_view(i_ms, j_ms, S_ms)

        # 检查长度
        print(f"ASE pairs:      {len(i_ase)}")
        print(f"matscipy pairs: {len(i_ms)}")

        # 检查数值是否完全一致
        is_same = np.allclose(ase_res, ms_res)
        print(f"Output Exact Match: {is_same}")

        if not is_same:
            # 如果不一致，检查差异（通常是浮点数精度导致的边界判定）
            diff = np.abs(ase_res - ms_res).max()
            print(f"Max Difference: {diff}")

if __name__ == "__main__":
    verify_and_benchmark(n_cells=10, cutoff=4.0)

import numpy as np
import os
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
import multiprocessing as mp

element_covalent_radii = {
    'H': 0.31,
    'B': 0.84,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Br': 1.20,
    'I': 1.39
}

def get_bond_limits(ele1, ele2):
    R1 = element_covalent_radii.get(ele1)
    R2 = element_covalent_radii.get(ele2)
    if R1 is None or R2 is None:
        return None, None
    if ele1 == 'H' or ele2 == 'H':
        bond_max = (R1 + R2) * 1.15 * 1.06
#        bond_max = (R1 + R2) * 1.15 * 1.15
    else:
        bond_max = (R1 + R2) * 1.15 * 1.01
    bond_min = 0.7 * (R1 + R2)
    return bond_min, bond_max

def get_bonded_pairs_ref(struct):
    symbols = struct.get_chemical_symbols()
    N = len(symbols)
    bonded_pairs = set()
    for i in range(N):
        for j in range(i+1, N):
            ele1 = symbols[i].capitalize()
            ele2 = symbols[j].capitalize()
            bond_min, bond_max = get_bond_limits(ele1, ele2)
            if bond_min is None or bond_max is None:
                continue
            dist = struct.get_distance(i, j, mic=True)
            if bond_min < dist < bond_max:
                bonded_pairs.add((i, j))
    return bonded_pairs

def fixed_displacement_sampling(atoms, n_structures, distance, seed=None):
    np.random.seed(seed)
    N = len(atoms)
    structures = []
    for _ in range(n_structures):
        atoms_tmp = atoms.copy()
        directions = np.random.normal(size=(N, 3))
        normalizer = np.linalg.norm(directions, axis=1, keepdims=True)
        displacements = distance * directions / normalizer
        atoms_tmp.positions += displacements
        atoms_tmp.wrap()
        structures.append(atoms_tmp)
    return structures

def filter_structures(structures, bonded_pairs, symbols_ref, print_fail_detail=False):
    filtered = []
    first_fail_info_printed = False
    for struct_idx, struct in enumerate(structures):
        symbols = struct.get_chemical_symbols()
        flag = True
        fail_info = None
        for i, j in bonded_pairs:
            ele1 = symbols[i].capitalize()
            ele2 = symbols[j].capitalize()
            bond_min, bond_max = get_bond_limits(ele1, ele2)
            if bond_min is None or bond_max is None:
                continue
            dist = struct.get_distance(i, j, mic=True)
            if dist < bond_min or dist > bond_max:
                flag = False
                fail_info = {
                    "idx": struct_idx,
                    "atoms": (i, ele1, j, ele2),
                    "dist": dist,
                    "bond_min": bond_min,
                    "bond_max": bond_max
                }
                break
        if flag:
            struct.wrap()
            filtered.append(struct)
        else:
            if (not first_fail_info_printed) and print_fail_detail:
                print(f"  [示例] 未通过结构 #{struct_idx}: 断裂键 ({fail_info['atoms'][1]}{fatoms'][0]}-{fail_info['atoms'][3]}{fail_info['atoms'][2]}), "
                      f"实际距离: {fail_info['dist']:.3f}Å, 允许范围: ({fail_info['bond_min']fail_info['bond_max']:.3f})")
                first_fail_info_printed = True
    return filtered

def get_n_structures(atoms):
    structure = AseAtomsAdaptor.get_structure(atoms)
    n_atoms = len(structure)
    site_factor = n_atoms / 2
    analyzer = SpacegroupAnalyzer(structure)
    N_symm = len(analyzer.get_point_group_operations())
    sym_factor = 1.0 if N_symm == 48 else np.sqrt(48 / N_symm)
    n_structures = int(round(sym_factor * site_factor))
    print(f"检测到{n_atoms}个原子, 点群操作数: {N_symm}, site_factor: {site_factor:.2f}, sym_m_factor:.2f} -> 建议采样数量: {n_structures}")
    return n_structures

def write_compare_txt(ref_atoms, struct, fname):
    ref_symbols = ref_atoms.get_chemical_symbols()
    struct_symbols = struct.get_chemical_symbols()
    ref_positions = ref_atoms.get_positions()
    struct_positions = struct.get_positions()
    with open(fname, 'w') as f:
        f.write(f"{'Idx':>5s} {'Element':>7s} {'Ref_X':>10s} {'Ref_Y':>10s} {'Ref_Z':>10s} "
                f"{'New_X':>10s} {'New_Y':>10s} {'New_Z':>10s} "
                f"{'Abs_dX':>10s} {'Abs_dY':>10s} {'Abs_dZ':>10s} {'Disp':>10s}\n")
        for idx, (el1, el2, pos1, pos2) in enumerate(zip(ref_symbols, struct_symbols, ref_positions, struct_positions)):
            dxyz = np.abs(pos2 - pos1)
            disp = np.linalg.norm(pos2 - pos1)
            f.write(f"{idx:5d} {el1:>7s} {pos1[0]:10.4f} {pos1[1]:10.4f} {pos1[2]:10.4f} "
                    f"{pos2[0]:10.4f} {pos2[1]:10.4f} {pos2[2]:10.4f} "
                    f"{dxyz[0]:10.4f} {dxyz[1]:10.4f} {dxyz[2]:10.4f} {disp:10.4f}\n")

def process_single_cif(args):
    cif_file, input_folder, output_folder, fixed_distances, seed = args
    atoms = read(os.path.join(input_folder, cif_file))
    base = os.path.splitext(cif_file)[0]
    subfolder = os.path.join(output_folder, base)
    os.makedirs(subfolder, exist_ok=True)
    n_structures = get_n_structures(atoms)
    symbols_ref = atoms.get_chemical_symbols()
    bonded_pairs = get_bonded_pairs_ref(atoms)
    print(f"{cif_file}：初始结构成键对数：{len(bonded_pairs)}")
    for fd in fixed_distances:
        if fd == 0.01:
            n_this = max(1, n_structures // 2)
        else:
            n_this = n_structures
        structs = fixed_displacement_sampling(atoms, n_this, fd, seed)
        print(f"  扰动幅度 {fd:.3f}，共采样{len(structs)}个结构。", end='')
        filtered_structs = filter_structures(structs, bonded_pairs, symbols_ref, print_fail_detail=True)
        print(f" 通过筛选结构数：{len(filtered_structs)}")
        for i, struct in enumerate(tqdm(filtered_structs, desc=f"{base} {fd:.3f}", leave=False)):
            outbase = f"{base}_fixed_{str(fd).replace('.', '_')}_{i}"
            outcif = os.path.join(subfolder, outbase + ".cif")
            outtxt = os.path.join(subfolder, outbase + "_compare.txt")
            write(outcif, struct)
            write_compare_txt(atoms, struct, outtxt)

def process_folder(
        input_folder,
        output_folder,
        fixed_distances=[0.01, 0.08, 0.15, 0.2, 0.25, 0.3],
        seed=42,
        num_workers=8):
    os.makedirs(output_folder, exist_ok=True)
    cif_files = [f for f in os.listdir(input_folder) if f.endswith('.cif')]
    args_list = [(cif_file, input_folder, output_folder, fixed_distances, seed) for cif_file in cif_files]

    with mp.get_context('spawn').Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_cif, args_list), total=len(cif_files), desc="全体CIF进度"))

if __name__ == '__main__':
    process_folder(
        input_folder='final_frame_cif_out',
        output_folder='output_folder_fixed',
        fixed_distances=[0.01, 0.08, 0.15, 0.2, 0.25, 0.3],
        seed=42,
        num_workers=100   # 按机器CPU核数调整
    )
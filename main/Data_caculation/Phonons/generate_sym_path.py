from ase.io import read
import os
from seekpath import get_explicit_k_path
import yaml

# ==== 用户参数 ====
POSCAR_FILE = "POSCAR"
current_dir = os.path.basename(os.path.abspath(os.getcwd()))
ATOM_NAME = current_dir
BAND_POINTS = 101

# 读取dim
with open("phonopy_disp.yaml", "r") as f:
    data = yaml.safe_load(f)
dim = data["phonopy"]["configuration"]["dim"]
if isinstance(dim, str):
    DIM = " ".join(dim.split())
elif isinstance(dim, (list, tuple)):
    DIM = " ".join(str(x) for x in dim)
else:
    raise ValueError("无法识别 dim 格式")

# 1. 读入结构
atoms = read(POSCAR_FILE, format="vasp")
cell = atoms.get_cell()
scaled_positions = atoms.get_scaled_positions()  # 就是得到每个原子的分数坐标，常用于还原晶体结构
numbers = atoms.get_atomic_numbers()
structure = (cell.tolist(), scaled_positions.tolist(), numbers.tolist())

# 2. 用SeekPath获取高对称点和所有k点
path_data = get_explicit_k_path(structure)
labels = path_data['explicit_kpoints_labels']
kpoints = path_data['explicit_kpoints_rel']

# 3. 提取所有高对称点（label非空）
band_labels = []
q_points = []
for i, label in enumerate(labels):
    if label:
        band_labels.append(label)
        q_points.append(kpoints[i])

def label_latex(label):
    if label.upper() == 'GAMMA':
        return r'$\Gamma$'
    return label

# 4. 你的原始分段方式
band_label_blocks = []
band_kpoint_blocks = []
current_labels = [label_latex(band_labels[0])]
current_kpoints = [' '.join(f"{x:.8f}" for x in q_points[0])]

for i in range(1, len(band_labels)):
    label = band_labels[i]
    kpt = q_points[i]
    if band_labels[i] == band_labels[i-1]:
        # label相邻且内容一样，分段
        band_label_blocks.append(' '.join(current_labels))
        band_kpoint_blocks.append(' '.join(current_kpoints))
        current_labels = [label_latex(label)]
        current_kpoints = [' '.join(f"{x:.8f}" for x in kpt)]
    else:
        current_labels.append(label_latex(label))
        current_kpoints.append(' '.join(f"{x:.8f}" for x in kpt))
# 加上最后一个分段
band_label_blocks.append(' '.join(current_labels))
band_kpoint_blocks.append(' '.join(current_kpoints))

band_labels_str = ', '.join(band_label_blocks)
band_str = ', '.join(band_kpoint_blocks)

# 5. 输出 band.conf
with open("band.conf", "w") as f:
    f.write(f"ATOM_NAME = {ATOM_NAME}\n")
    f.write(f"DIM = {DIM}\n")
    f.write(f"BAND = {band_str}\n")
    f.write(f"BAND_POINTS = {BAND_POINTS}\n")
    f.write(f"BAND_LABELS = {band_labels_str}\n")
    f.write("BAND_CONNECTION = .TRUE.\n")
    f.write("FC_SYMMETRY = .TRUE.\n")
    f.write("DOS = .TRUE.\n")
    f.write("THERMAL_PROPERTIES = .TRUE.\n")
    f.write("FORCE_CONSTANTS = WRITE\n")
    f.write("MESH = 20 20 20\n")  # 指定网格密度

print("band.conf 已按要求生成！")

from ase.io import read
import os
from seekpath import get_explicit_k_path
import yaml

# ==== User parameters ====
POSCAR_FILE = "POSCAR"
current_dir = os.path.basename(os.path.abspath(os.getcwd()))
ATOM_NAME = current_dir
BAND_POINTS = 101

# Read supercell dimension (DIM) from phonopy_disp.yaml
with open("phonopy_disp.yaml", "r") as f:
    data = yaml.safe_load(f)

dim = data["phonopy"]["configuration"]["dim"]

if isinstance(dim, str):
    DIM = " ".join(dim.split())
elif isinstance(dim, (list, tuple)):
    DIM = " ".join(str(x) for x in dim)
else:
    raise ValueError("Unrecognized dim format")

# 1. Read crystal structure from POSCAR
atoms = read(POSCAR_FILE, format="vasp")
cell = atoms.get_cell()
scaled_positions = atoms.get_scaled_positions()  # Fractional coordinates of atoms (used to reconstruct crystal structure)
numbers = atoms.get_atomic_numbers()

structure = (
    cell.tolist(),
    scaled_positions.tolist(),
    numbers.tolist()
)

# 2. Use SeekPath to obtain high-symmetry path and explicit k-points
path_data = get_explicit_k_path(structure)
labels = path_data['explicit_kpoints_labels']
kpoints = path_data['explicit_kpoints_rel']

# 3. Extract high-symmetry points (non-empty labels only)
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

# 4. Original segmentation logic for band path
band_label_blocks = []
band_kpoint_blocks = []

current_labels = [label_latex(band_labels[0])]
current_kpoints = [' '.join(f"{x:.8f}" for x in q_points[0])]

for i in range(1, len(band_labels)):
    label = band_labels[i]
    kpt = q_points[i]

    # If consecutive labels are identical, start a new segment
    if band_labels[i] == band_labels[i - 1]:
        band_label_blocks.append(' '.join(current_labels))
        band_kpoint_blocks.append(' '.join(current_kpoints))

        current_labels = [label_latex(label)]
        current_kpoints = [' '.join(f"{x:.8f}" for x in kpt)]
    else:
        current_labels.append(label_latex(label))
        current_kpoints.append(' '.join(f"{x:.8f}" for x in kpt)]

# Append the final segment
band_label_blocks.append(' '.join(current_labels))
band_kpoint_blocks.append(' '.join(current_kpoints))

band_labels_str = ', '.join(band_label_blocks)
band_str = ', '.join(band_kpoint_blocks)

# 5. Write band.conf file
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
    f.write("MESH = 20 20 20\n")  # k-point mesh density for phonon DOS

print("band.conf has been generated successfully!")

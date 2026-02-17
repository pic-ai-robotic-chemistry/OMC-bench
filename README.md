# OMC-bench: Machine-Learned Interatomic Potentials for Organic Molecular Crystals

This repository provides a benchmark workflow for machine-learned interatomic potentials (MLIPs) on organic molecular crystals (OMCs). It supports multi-GPU and single-GPU execution for structure optimization, phonon/thermodynamics evaluation, structure matching, and polymorph ranking.

Project website:
```
https://aisci.ustc.edu.cn/mlip-omcs/#/
```

## Table of Contents

1. Project Overview
2. Quick Start
3. Script–Task Mapping
4. Task Instructions
5. Key File Formats
6. AtomBit Notes
7. FAQ & Troubleshooting

---

## 1. Project Overview

Capabilities:
- Batch evaluation of forces, stresses, and (optional) energies
- Batch structure optimization
- Phonon spectra and thermodynamic properties
- Structure matching and RMSD comparison
- Polymorph stability ranking

All scripts are configured via command-line arguments.

---

## 2. Quick Start

1. Install dependencies (Python >= 3.9, `ase`, `phonopy`, `numpy`, `pandas`, `scipy`, `tqdm`, and your MLIP with an ASE calculator interface).
2. Configure `Calculator_defs.json` (model name and model path).
3. Prepare inputs (structures, references, energy tables as needed).
4. Run the script for your task (see mapping below).
5. Check outputs and logs in the specified results directory.

---

## 3. Script–Task Mapping

| Task ID | Description | Script | Main Input | Main Output |
|---|---|---|---|---|
| 1 | Force/Stress/Energy (optional) evaluation | `Task_1.py` | `Structure_files/Task_1.xyz` | `results/task_1/eval_results.csv`, `results/task_1/eval_results_summary.txt` |
| 2-1 | Structure optimization | `Tasks_234_optimize.py` | `Structure_files/Task_2_init_str/` | `results/task_2/optimized_xyz/`, `results/task_2/individual_results/` |
| 2-2 | Phonon/Thermo/DOS | `Task_2_2.py` | `results/task_2/optimized_xyz/` | `results/task_2/phonon_results/` |
| 2-3 | MLIP vs DFT comparison | `Task_2_3.py` | `Structure_files/Task_2_ref.csv`, `results/task_2/phonon_results/ml_phonon_summary.csv` | `results/task_4/metrics_summary.csv` |
| 3-1 | Structure optimization | `Tasks_234_optimize.py` | `Structure_files/Task_3_init_str/` | `results/task_3/optimized_xyz/`, `results/task_3/individual_results/` |
| 3-2 | Structure matching (RMSD) | `Task_3_2.py` | `results/task_3/optimized_xyz/`, `Structure_files/Task_3_ref_cif/` | `results/task_3/structure_matcher_rmsd.csv` |
| 4-1 | Structure optimization | `Tasks_234_optimize.py` | `Structure_files/Task_4_init_str/` | `results/task_4/optimized_xyz/`, `results/task_4/individual_results/` |
| 4-2 | Polymorph ranking | `Task_4_2.py` | `results/task_4/individual_results/`, `Structure_files/Task_4_ref_energy.csv` | `results/task_4/summary_by_polymorph.json` |

Note: All model configurations are defined in `Calculator_defs.json` and resolved by `Calculator_factory.py`.

---

## 4. Task Instructions

### Task 1 — Force/Stress Error Evaluation

Example:
```bash
python Task_1.py \
  --xyz_file ../Structure_files/Task_1.xyz \
  --model_name atombit_omc \
  --config_json Calculator_defs.json \
  --output_csv ../results/task_1/eval_results.csv \
  --n_jobs 5
```

### Task 2 — Phonon & Thermodynamics

1) Structure optimization:
```bash
python Tasks_234_optimize.py \
  --input_dir ../Structure_files/Task_2_init_str \
  --output_dir ../results/task_2 \
  --model_name atombit_omc \
  --config_json Calculator_defs.json \
  --fmax 0.001 \
  --max_steps 3000 \
  --gpus 0 1 2 3 4 5 6 7 \
  --n_jobs 16
```

2) Phonon/Thermo/DOS:
```bash
python Task_2_2.py \
  --input_dir ../results/task_2/optimized_xyz \
  --outdir ../results/task_2/phonon_results \
  --model_name atombit_omc \
  --config_json Calculator_defs.json \
  --n_jobs 12
```

3) MLIP vs DFT comparison (positional arguments):
```bash
python Task_2_3.py \
  ../Structure_files/Task_2_ref.csv \
  ../results/task_2/phonon_results/ml_phonon_summary.csv
```

### Task 3 — Structure Matching

1) Structure optimization:
```bash
python Tasks_234_optimize.py \
  --input_dir ../Structure_files/Task_3_init_str \
  --output_dir ../results/task_3 \
  --model_name atombit_omc \
  --config_json Calculator_defs.json \
  --fmax 0.01 \
  --max_steps 3000 \
  --gpus 0 1 2 3 4 5 6 7 \
  --n_jobs 16
```

2) Structure matching:
```bash
python Task_3_2.py \
  --input_dir ../results/task_3/optimized_xyz \
  --ref_dir ../Structure_files/Task_3_ref_cif \
  --output ../results/task_3/structure_matcher_rmsd.csv
```

### Task 4 — Polymorph Ranking

1) Structure optimization:
```bash
python Tasks_234_optimize.py \
  --input_dir ../Structure_files/Task_4_init_str \
  --output_dir ../results/task_4 \
  --model_name atombit_omc \
  --config_json Calculator_defs.json \
  --compare energy \
  --ref_energy_csv ../Structure_files/Task_4_ref_energy.csv \
  --gpus 4 5 6 7 \
  --n_jobs 16
```

2) Polymorph ranking:
```bash
python Task_4_2.py \
  --input_dir ../results/task_4/individual_results \
  --output ../results/task_4/summary_by_polymorph.json \
  --ref_csv ../Structure_files/Task_4_ref_energy.csv
```

---

## 5. Key File Formats

### 1) `Calculator_defs.json`

```json
{
  "mace_test": {"arch": "mace_mp", "path": "/path/to/mace_model.pt"}
}
```

### 2) `Task_4_ref_energy.csv`

```
name,ref_energy,n_mol,polymorph
ABC,-237.0,2,ABC
ABC01,-239.1,2,ABC
```

### 3) Structure naming

- Optimized structure: `ABC_opt.extxyz`

---

## 6. AtomBit Notes

- Usage examples for AtomBit are available in `main/AtomBIt-l/demo.ipynb`.
- The MindSpore/NPU variant is under `main/AtomBit-MindSpore/`.

---

## 7. FAQ & Troubleshooting

- Benchmark result location:
```
https://huggingface.co/datasets/MUUYUU/OMC-bench
```
- Adding or switching models: update `Calculator_defs.json` and ensure `Calculator_factory.py` supports the architecture.
- File format issues: verify against the templates in the “Key File Formats” section.
- Tasks can be run independently. Missing or failed structures are typically skipped with a warning.


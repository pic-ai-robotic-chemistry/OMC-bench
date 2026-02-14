# <font color="#8B0000">O</font>MC-bench for machine-learned interatomic potential in <font color="#8B0000">O</font>rganic <font color="#8B0000">M</font>olecular <font color="#8B0000">C</font>rystals (<font color="#8B0000">OMC</font>s)

## Table of Contents

1. [Project Introduction](#1-project-introduction)
2. [Quick Start](#2-quick-start)
3. [Script-Task-File Mapping](#3-script-task-file-mapping)
4. [Detailed Task Instructions](#4-detailed-task-instructions)
5. [Key File Formats](#5-key-file-formats)
6. [Use of AtomBit-OMC](#6-AtomBIt-OMC)
7. [FAQ & Troubleshooting](#7-faq--troubleshooting)

---

## 1. Project Introduction

This platform is designed for the automated benchmarking of Machine Learning Potential (MLP) models for organic molecular crystals. It supports both multi-GPU and single-GPU parallelism.

- Batch Evaluation: Error assessment for forces, stresses, and (optionally) energies.
- Batch Geometry Optimization: Automated structure relaxation.
- Properties Calculation: Phonon spectra and thermodynamic properties.
- Structural Analysis: Structure matching and RMSD comparison.
- Ranking: Polymorph stability ranking.

All scripts are fully configurable via command-line arguments.

---

## 2. Quick Start

1. **Install Dependencies** (Python>=3.9, ase, phonopy, numpy, pandas, scipy, tqdm, and the your MLIP with ase calculator interface).
2. **Configure** `Calculator_defs.json` (add the model name and model path).
3. **Prepare Data**, including input structures, reference structures, and energy tables.
4. **Select Script** according to the "Script-Task-File Mapping" table, check inputs, and run the task.
5. **Check Results** in the output files or command-line output.

> **Note:** It is recommended to read this manual thoroughly to avoid errors caused by incorrect input/output file formats.

---

## 3. Script-Task-File Mapping

| Task ID | Task Description | Script File | Main Input File/Directory | Main Output File/Directory |
|---|---|---|---|---|
| 1 | Batch xyz Force/Stress/Energy(optional) Eval | `Task_1.py` | `Task_1.xyz` | `results/task_1/eval_results.csv`, `results/task_2/eval_results_summary.txt` |
| 2-1 | Batch Structure Optimization | `Tasks_234_optimize.py` | `Task_2_init_str/` | `results/task_2/optimized_xyz/`, `results/task_2/individual_results` |
| 2-2 | Phonon/Thermo/DOS Calculation | `Task_2_2.py` | `results/task_2/optimized_xyz/` | `results/task_2/phonon_results` |
| 2-3 | Comparison between MLIP and DFT | `Task_2_3.py` | `phonon_benchmark_summary_ref.csv`, `results/task_2/phonon_results/ml_phonon_summary.csv` | `results/task_2/metrics_summary.csv` |
| 3-1 | Batch Structure Optimization | `Tasks_234_optimize.py` | `Task_3_init_str/` | `results/task_3/optimized_xyz/`, `results/task_3/individual_results` |
| 3-2 | Structure Matching | `Task_3_2.py` | `results/task_3/optimized_xyz/`, `Task_3_ref_cif/` | `results/task_3/summary_by_polymorph.json` |
| 4-1 | Batch Structure Optimization | `Tasks_234_optimize.py` | `Task_4_init_str/` | `results/task_4/optimized_xyz/`, `results/task_4/individual_results` |
| 4-2 | Polymorph Ranking | `Task_3_2.py` | `results/task_3/individual_results/`, `Task_4_ref_energy.csv` | `results/task_4/summary_by_polymorph.json` |


> **Note:** All model configurations must be specified via `Calculator_defs.json and Calculator_factory.py`.

---

## 4. Detailed Task Instructions

### [Task 1] Batch Force/Stress Error Evaluation

**Purpose**: Perform batch error statistics for force/stress on a multi-frame `Task_1.xyz` file.

**Script**: `Task_1.py`

**Example Command**:

```bash
python Task_1.py \
    --xyz_file ../Structure_files/Task_1.xyz \
    --model_name atombit_omc \
    --config_json Calculator_defs.json \
    --output_csv ../results/task_1/eval_results.csv \
    --n_jobs 5
```
---

### [Task 2] Phonon & Thermodynamic Properties Benchmarking

#### 1. Batch Structure Optimization

**Script**: `Tasks_234_optimize.py`

**Example Command**:

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

#### 2. Phonon/Thermo/DOS Calculation

**Script**: `Task_2_2.py`

**Example Command**:

```bash
python Task_2_2.py \
    --input_dir ../results/task_2/optimized_xyz \
    --outdir ../results/task_2/phonon_results \
    --model_name atombit_omc \
    --config_json Calculator_defs.json \
    --n_jobs 12
```

#### 3. Comparison between MLIP and DFT

**Script**: `Task_2_3.py`

**Example Command**:

```bash
python Task_2_3.py \
    --ref_csv ../Structure_files/Task_2_ref.csv \
    --pred_csv ../results/task_2/phonon_results/ml_phonon_summary.csv \
    --output ../results/task_2/metrics_summary.csv
```

---

### [Task 3] Structure Matching (Cocrystal)

#### 1. Batch Structure Optimization

**Script**: `Tasks_234_optimize.py`

**Example Command**:

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

#### 2. Structure Matching

**Script**: `Task_3_2.py`

**Example Command**:

```bash
python Task_3_2.py \
    --input_dir ../results/task_3/optimized_xyz \
    --ref_dir ../Task_3_ref_cif \
    --output ../results/task_3/metrics_summary.csv
```

---

### [Task 4] Polymorph Ranking

#### 1. Batch Structure Optimization

**Script**: `Tasks_234_optimize.py`

**Example Command**:

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

#### 2. Polymorph Ranking

**Script**: `Task_4_2.py`

**Example Command**:

```bash
python Task_4_2.py \
    --input_dir ../results/task_4/individual_results/ \
    --output ../results/task_4/summary_by_polymorph.json \
    --ref_csv ../Structure_files/Task_4_ref_energy.csv
```

## 5. Key File Formats

### 1) `Calculator_defs.json`

```json
{
  "mace_test": {"arch": "mace", "path": "/path/to/mace_model.pt"}
}
```

---

### 2) `Task_4_ref_energy.csv`

```
name,ref_energy,n_mol,polymorph
ABC,-237.0,2,ABC
ABC01,-239.1,2,ABC
```

---

### 3) xyz/cif Structure Naming Convention

- Optimized structure: `ABC_opt.extxyz`

---

## 6. AtomBIt

How to use AtomBit has been demonstrated in the main/AtomBit-OMC-l/demo.ipynb.
In addition, we also provide the AtomBit-OMC in NPU version in the main/AtomBit

## 7. FAQ & Troubleshooting

- **Adding/Switching Models**: Edit `calculator_defs.json` and configure the model name and path.
- **File Format Mismatch/Missing Fields**: Please refer to "Key File Formats" to check your input format and fields.
- **Task Execution**: Tasks can be run independently or sequentially; if some structures fail or are missing, the scripts will automatically skip them and output a warning/hint.

---
```

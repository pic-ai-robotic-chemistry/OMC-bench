# AtomBit-MindSpore (Ascend + MindSpore Usage Guide)

This subproject is an **engineering guide** for running training on **Ascend NPUs with MindSpore**.  
It focuses on setup, data preparation, launch, and troubleshooting, and intentionally does **not** cover model internals.

> Primary training entrypoint: `Train_dist.py`

---

## 1) What this README helps you do

- Prepare a compatible Ascend + MindSpore runtime.
- Build HDF5 training data and metadata.
- Generate E0 reference values and connect them to training.
- Run single-card first, then scale to multi-card with `msrun`.
- Quickly diagnose common startup and runtime issues.

---

## 2) Project structure at a glance

```text
AtomBit-MindSpore/
├── Train_dist.py                 # Main training script (single-card / distributed)
├── train.sh                      # Example launch script for Ascend multi-card with msrun
├── scripts/
│   ├── Preprocess_h5.py          # Data preprocessing: xyz -> h5 chunks + metadata
│   ├── extxyz_to_pyg_custom.py   # Structure file parsing
│   ├── compute_average_e0.py
│   └── compute_save_e0.py
├── src/
│   ├── engine/                   # Training and validation pipeline
│   ├── data/                     # Dataset, sampler, and I/O
│   └── utils/                    # Config, scheduler, helper utilities
└── sharker/                      # Required path for dataloader imports
```

---

## 3) Environment requirements (Ascend + MindSpore)

### 3.1 Version baseline

Use a version-compatible stack:

- CANN: `8.3.RC1`
- MindSpore (Ascend build): `2.7.1`
- Python: `3.9+`

Python packages commonly required by this training pipeline:

- `numpy`
- `h5py`
- `tqdm`

### 3.2 Quick environment checks

1. Verify Ascend devices are visible:

   ```bash
   npu-smi info
   ```

2. Verify `msrun` is available:

   ```bash
   which msrun
   ```

3. Confirm your selected devices are exposed correctly before launch:

   ```bash
   echo "$ASCEND_RT_VISIBLE_DEVICES"
   ```

4. For multi-node/multi-card training, ensure communication config follows Ascend official setup guidance.

> Practical recommendation: always validate on single-card first before any distributed launch.

---

## 4) Data preparation (HDF5 + metadata + E0)

Training expects **HDF5 chunks + metadata**, with optional E0 reference file for corrected energy handling.

### 4.1 Build HDF5 chunks and metadata

```bash
python scripts/Preprocess_h5.py
```

Expected outputs:

- One or more `.h5` chunk files.
- Metadata file(s) containing fields like:
  - `file_path`
  - `index_in_file`
  - `num_atoms`
  - `num_edges`

### 4.2 Generate E0 file (recommended)

```bash
python scripts/compute_save_e0.py
```

Then point `E0_PATH` in `Train_dist.py` to the generated E0 file.

### 4.3 Metadata naming consistency

A common startup failure is metadata suffix mismatch:

- Preprocess script may output: `*_metadata.pickle`
- Training config may expect: `*.pkl`

Ensure `TRAIN_META` and `TEST_META` in `Train_dist.py` exactly match your actual filenames.

---

## 5) Configuration checklist before first run

In `Config` (inside `Train_dist.py`), verify at least:

- `DATA_DIR`: HDF5 root directory
- `TRAIN_META` / `TEST_META`: metadata filenames
- `LOG_DIR`: output path for logs/checkpoints
- `E0_PATH`: E0 file path (if used)

Recommended first-run checks:

- Paths are absolute or unambiguous.
- `DATA_DIR` and metadata files are readable by your runtime user.
- `LOG_DIR` is writable.

---

## 6) Launch training

### 6.1 Single-card (recommended first)

```bash
python Train_dist.py
```

Use this run to validate:

- import path correctness
- data loading correctness
- metric/loss printing and checkpoint saving behavior

### 6.2 Multi-card with `msrun`

```bash
bash train.sh
```

A typical `train.sh` includes:

- `PARALLEL_MODE=DATA_PARALLEL`
- `PYTHONPATH=$(pwd)/sharker:$PYTHONPATH`
- `msrun --worker_num=... Train_dist.py`

When adapting to your machine, confirm:

- worker count equals intended visible devices
- `ASCEND_RT_VISIBLE_DEVICES` matches the selected cards
- each rank can read the same data and metadata paths

---

## 7) Logs and artifacts: what to expect

`LOG_DIR` usually contains:

- training logs
- per-epoch checkpoints (for example: `model_epoch_{k}.ckpt`)

If run quality looks suspicious, check:

- loss trend over early epochs
- validation metrics consistency across epochs
- whether checkpoints are produced at expected intervals

---

## 8) Common issues and fast fixes

### 8.1 "metadata file not found"

- Re-check `DATA_DIR`, `TRAIN_META`, `TEST_META`.
- Confirm suffix consistency (`.pickle` vs `.pkl`).
- Confirm metadata file is physically present in expected location.

### 8.2 Distributed launch starts but dataloader/import fails

- Ensure `PYTHONPATH` includes `sharker`.
- Ensure every rank can access identical data paths.
- Validate `file_path` entries in metadata point to existing `.h5` files.

### 8.3 Training starts but is unstable or slow

- Reproduce on single-card first.
- Then scale gradually (e.g., 1 -> 2 -> N cards).
- Inspect data I/O throughput and per-step latency.

### 8.4 E0 appears not applied

- Confirm `E0_PATH` is set and points to the newly generated E0 file.
- Confirm the file is readable at runtime.

---

## 9) End-to-end minimal runbook (recommended order)

1. Prepare Ascend + MindSpore environment (`npu-smi info`, `msrun` check).
2. Run preprocessing:

   ```bash
   python scripts/Preprocess_h5.py
   ```

3. Generate E0 and configure `E0_PATH`:

   ```bash
   python scripts/compute_save_e0.py
   ```

4. Update `Config` paths in `Train_dist.py` (`DATA_DIR`, `TRAIN_META`, `TEST_META`, `LOG_DIR`, `E0_PATH`).
5. Run single-card sanity test:

   ```bash
   python Train_dist.py
   ```

6. Scale to distributed run:

   ```bash
   bash train.sh
   ```

---

For model architecture, loss definition, and theory, refer to external model documentation. This README is intentionally scoped to Ascend + MindSpore operational usage.

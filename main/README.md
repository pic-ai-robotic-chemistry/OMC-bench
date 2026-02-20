# OMC-bench: Repository Layout

This `main/` directory contains the full benchmark workflow plus related code, data, and utilities.

## Layout

- `main/Scripts/`
  Benchmark scripts for Tasks 1–4, plus `Calculator_defs.json` and `Calculator_factory.py`.
- `main/Structure_files/`
  Benchmark inputs and reference data (initial structures, reference energies, CIFs).
- `main/MLIPs/`
  Example/pretrained MLIP model files.
- `main/AtomBIt-l/`
  AtomBit (PyTorch) code and demo notebook.
- `main/AtomBit-MindSpore/`
  MindSpore/NPU training scripts and source.
- `main/Data_caculation/`
  DFT/phonon input templates and data-generation utilities.
- `main/mini_tools/`
  Small analysis utilities (e.g., SOAP/UMAP/FPS, CSD SMILES, DOE tools).

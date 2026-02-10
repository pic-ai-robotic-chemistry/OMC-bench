#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-consistent ML-based phonon and thermodynamics evaluation using Phonopy.

Key features:
- Automatically builds supercells such that each lattice vector is longer than a
  specified minimum length (default: 12 Å).
- Computes forces using a machine-learned interatomic potential (MLIP) wrapped
  via CalculatorFactory.
- Generates phonon force constants, mesh frequencies, DOS, and thermal
  properties.
- Saves detailed YAML and JSON output for each material.
- Runs in parallel with joblib.

This script is intended for benchmarking MLIPs on phonon stability and
thermodynamic quantities without requiring DFT reference force constants.
"""

import os
import glob
import yaml
import json
import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
import phonopy
from Calculator_factory import CalculatorFactory
from joblib import Parallel, delayed


# ======================================================================
# Utility: Determine supercell matrix based on minimal cell length
# ======================================================================

def get_supercell_matrix(cell, min_len=12.0):
    """
    Compute integer supercell multipliers for each lattice direction such that
    the expanded supercell has length >= min_len (Å) along each axis.

    Parameters
    ----------
    cell : ndarray (3,3)
        ASE cell matrix (in Å).
    min_len : float
        Minimum length required along each direction.

    Returns
    -------
    supercell_matrix : ndarray (3,3)
        Diagonal integer matrix specifying the supercell expansion.
    """
    lengths = np.linalg.norm(cell, axis=1)
    mults = np.ceil(min_len / lengths).astype(int)
    mults = np.maximum(mults, 1)
    return np.diag(mults)


# ======================================================================
# Core phonon evaluation routine
# ======================================================================

def run_phonopy_on_ml_struct(xyz_path, calc, distance=0.01,
                             outdir="./phonon_results", min_len=12.0):
    """
    Run self-consistent phonon calculation using an MLIP-based calculator.

    Workflow:
    1. Load ML-optimized structure (.extxyz).
    2. Automatically generate a supercell large enough for phonon evaluation.
    3. Generate finite-displacement supercells (Phonopy).
    4. Compute forces from MLIP for each displaced supercell.
    5. Build force constants and compute thermodynamic quantities and DOS.
    6. Save YAML and JSON output.

    Parameters
    ----------
    xyz_path : str
        Path to optimized structure (.extxyz).
    calc : ASE Calculator
        MLIP calculator created by CalculatorFactory.
    distance : float
        Displacement amplitude in Å (typically 0.01).
    outdir : str
        Output directory for YAML/JSON files.
    min_len : float
        Minimum expanded cell length for supercell construction.

    Returns
    -------
    result : dict
        Dictionary containing phonon and thermodynamic properties.
    """
    ml_atoms = read(xyz_path)
    name = os.path.splitext(os.path.basename(xyz_path))[0]

    # Build supercell ensuring each direction >= min_len
    cell = ml_atoms.get_cell()
    supercell_matrix = get_supercell_matrix(cell, min_len=min_len)

    print(f"[{name}] Supercell matrix: {supercell_matrix.tolist()} "
          f"(original lengths = {np.linalg.norm(cell, axis=1).round(3)} Å)")

    # Convert ASE → PhonopyAtoms
    ph_atoms = phonopy.structure.atoms.PhonopyAtoms(
        cell=ml_atoms.get_cell(),
        scaled_positions=ml_atoms.get_scaled_positions(),
        symbols=ml_atoms.get_chemical_symbols()
    )

    # Build Phonopy object
    ph = phonopy.Phonopy(
        ph_atoms,
        supercell_matrix=supercell_matrix,
        primitive_matrix=None,
        symprec=1e-3,
    )

    # Generate displaced supercells
    ph.generate_displacements(distance=distance, is_diagonal=False)

    # Compute MLIP forces
    forcesets = []
    for sc in ph.supercells_with_displacements:
        sc_atoms = Atoms(
            cell=sc.cell,
            symbols=sc.symbols,
            scaled_positions=sc.scaled_positions,
            pbc=True
        )
        sc_atoms.calc = calc
        forces = sc_atoms.get_forces()

        # Remove drift force (important for stable force constants)
        drift = forces.sum(axis=0)
        for f in forces:
          f -= drift / forces.shape[0]

        forcesets.append(forces)

    ph.forces = forcesets

    # Build and symmetrize force constants
    ph.produce_force_constants()
    ph.symmetrize_force_constants()

    # Save YAML
    os.makedirs(outdir, exist_ok=True)
    out_yaml = os.path.join(outdir, f"{name}_ml.yaml")
    ph.save(filename=out_yaml, settings={'force_constants': True})

    # Compute mesh frequencies
    mesh = [20, 20, 20]  # customizable
    ph.run_mesh(mesh)
    mesh_dict = ph.get_mesh_dict()
    w_max = np.max(mesh_dict["frequencies"])

    # Thermal properties (selected temperatures)
    T = [0, 75, 150, 300, 600]
    ph.run_thermal_properties(temperatures=T)
    res = ph.get_thermal_properties_dict()

    # DOS
    ph.run_total_dos()
    dos_dict = ph.get_total_dos_dict()

    # Package results
    result = {
        "Material": name,
        "supercell_matrix": supercell_matrix.tolist(),
        "cell_a_b_c(Å)": np.linalg.norm(cell, axis=1).round(6).tolist(),
        "w_max(THz)": float(w_max),
        "Entropy_300K(J/mol/K)": res["entropy"].tolist()[3],
        "Free_energy_300K(meV)": res["free_energy"].tolist()[3],
        "Cv_300K(J/mol/K)": res["heat_capacity"].tolist()[3],
        "DOS_freq": dos_dict["frequency_points"].tolist(),
        "DOS": dos_dict["total_dos"].tolist(),
        "yaml_path": out_yaml
    }

    json_file = os.path.join(outdir, f"{name}_ml.json")
    with open(json_file, "w") as jf:
        json.dump(result, jf, indent=2)

    return result


# ======================================================================
# Wrapper for parallel processing
# ======================================================================

def process_file(xyz_path, model_name, config_json, distance, outdir, min_len):
    """
    Wrapper for processing a single extxyz file.
    Handles caching and JSON reading.

    Returns
    -------
    result dict or None
    """
    name = os.path.splitext(os.path.basename(xyz_path))[0]
    json_file = os.path.join(outdir, f"{name}_ml.json")

    # If result already exists, load instead of recomputing
    if os.path.exists(json_file):
        print(f"[{name}] JSON already exists — skipping.")
        try:
            with open(json_file, "r") as jf:
                return json.load(jf)
        except Exception as e:
            print(f"  [Warning] Failed to read existing JSON, recomputing: {e}")

    # Otherwise compute from scratch
    try:
        calc = CalculatorFactory.from_config(model_name, config_json)
        return run_phonopy_on_ml_struct(
            xyz_path, calc, distance=distance, outdir=outdir, min_len=min_len
        )
    except Exception as e:
        print(f"[{name}] Failed to process: {xyz_path}")
        print(f"  Error: {e}")
        return None


# ======================================================================
# Main entry point
# ======================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Self-consistent ML phonon/thermodynamics evaluation "
                    "(automatic supercell expansion, no DFT reference required)."
    )
    parser.add_argument('--input_dir', required=True, help="Directory containing *.extxyz files")
    parser.add_argument('--outdir', default="phonon_results", help="Output directory")
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--config_json', default="calculator_defs.json")
    parser.add_argument('--distance', type=float, default=0.01)
    parser.add_argument('--n_jobs', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--min_len', type=float, default=12.0, help="Minimum supercell dimension (Å)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    xyz_files = glob.glob(os.path.join(args.input_dir, "*.extxyz"))

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(
            xyz, args.model_name, args.config_json,
            args.distance, args.outdir, args.min_len
        )
        for xyz in xyz_files
    )

    # Remove failed cases
    results = [r for r in results if r is not None]

    # Save summary CSV
    df = pd.DataFrame(results)
    summary_csv = os.path.join(args.outdir, "ml_phonon_summary.csv")
    df.to_csv(summary_csv, index=False)

    print(f"All ML phonon evaluations completed. Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()

"""
This script is for Task 1: Evaluate force and stress errors on large systems of Organic molecular crystals.
The single-point energy is provided when compared in the same level.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.io import read
from Calculator_factory import CalculatorFactory
from joblib import Parallel, delayed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch predict and evaluate energy (No alignment), force, stress from multi-frame xyz")
    parser.add_argument('--xyz_file', required=True, help='Multi-frame xyz file')
    parser.add_argument('--model_name', required=True, help='Model name defined in calculator_defs.json')
    parser.add_argument('--config_json', default='Calculator_defs.json', help='Path to config json')
    parser.add_argument('--output_csv', default='eval_results.csv', help='Path to save result csv')
    parser.add_argument('--n_jobs', type=int, default=4, help='Parallel jobs (default: 4)')
    return parser.parse_args()

def stress9_to_voigt6(stress9):
    """
    Convert a 9-element stress tensor from ASE XYZ (xx, xy, xz, yx, yy, yz, zx, zy, zz) 
    to Voigt order (xx, yy, zz, yz, xz, xy).
    """
    xx = stress9[0]
    yy = stress9[4]
    zz = stress9[8]
    yz = (stress9[5] + stress9[7]) / 2
    xz = (stress9[2] + stress9[6]) / 2
    xy = (stress9[1] + stress9[3]) / 2
    return np.array([xx, yy, zz, yz, xz, xy])

def evaluate_frame(atoms, calc, idx):
    """Evaluate a single frame: calculate energy, force, stress and compare with reference."""
    try:
        atoms.calc = calc
        n_atoms = len(atoms)
        
        # Get predictions
        pred_energy = atoms.get_potential_energy()
        pred_forces = atoms.get_forces()
        pred_stress = atoms.get_stress()
        
        # Get references
        ref_energy = float(atoms.info["REF_energy"])
        ref_stress = np.array(atoms.info["REF_stress"])
        if len(ref_stress) == 9:
            ref_stress = stress9_to_voigt6(ref_stress)
        ref_forces = atoms.arrays["REF_forces"] if "REF_forces" in atoms.arrays else None

        # Calculate errors
        # Energy error per atom (meV/atom)
        energy_err = np.abs((pred_energy - ref_energy) / n_atoms) * 1000 
        # Stress error (GPa). 1 eV/A^3 = 160.21766208 GPa
        stress_err = np.abs((pred_stress - ref_stress)) * 160.21766208 

        if ref_forces is not None:
            force_err = pred_forces - ref_forces
            force_mae = np.mean(np.abs(force_err)) * 1000
            force_rmse = np.sqrt(np.mean(force_err ** 2)) * 1000
        else:
            force_mae = force_rmse = None

        return {
            "frame": idx,
            "energy_err_per_atom": energy_err,
            # FIXED: Renamed key to match the main function's access pattern (stress_mae)
            "stress_mae": np.mean(np.abs(stress_err)), 
            "stress_rmse": np.sqrt(np.mean(stress_err ** 2)),
            "force_mae": force_mae,
            "force_rmse": force_rmse,
            "n_atoms": n_atoms
        }
    except Exception as e:
        print(f"[frame {idx}] error: {e}")
        return {
            "frame": idx,
            "energy_err_per_atom": None,
            "stress_mae": None,
            "stress_rmse": None,
            "force_mae": None,
            "force_rmse": None,
            "n_atoms": len(atoms)
        }

def main():
    args = parse_args()
    print("Loading model...")
    calc = CalculatorFactory.from_config(args.model_name, args.config_json)
    print(f"Reading xyz file: {args.xyz_file}")
    frames = read(args.xyz_file, index=":")
    print(f"Total frames: {len(frames)}")

    # Parallelize per-frame evaluation across multiple workers
    per_frame_results = Parallel(n_jobs=args.n_jobs)(
        delayed(evaluate_frame)(atoms, calc, idx)
        for idx, atoms in enumerate(tqdm(frames))
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(per_frame_results)
    df.to_csv(args.output_csv, index=False)
    
    # --- Statistics and Saving Output ---
    
    # Calculate means
    mean_energy_mae = np.nanmean(np.abs(df["energy_err_per_atom"]))
    mean_stress_mae = np.nanmean(df["stress_mae"])
    mean_force_mae = np.nanmean(df["force_mae"])

    # Prepare the summary text
    summary_content = (
        "--------\n"
        "Mean absolute error:\n"
        f"Energy (No alignment) MAE: {mean_energy_mae:.5f} meV/atom\n"
        f"Stress MAE: {mean_stress_mae:.5f} GPa\n"
        f"Force MAE: {mean_force_mae:.5f} meV/Å\n"
        "--------\n"
        f"Result csv saved: {args.output_csv}\n"
    )

    # Print to console
    print(summary_content)

    summary_file = os.path.splitext(args.output_csv)[0] + "_summary.txt"
    try:
        with open(summary_file, "w", encoding='utf-8') as f:
            f.write(summary_content)
        print(f"Summary text saved to: {summary_file}")
    except Exception as e:
        print(f"Failed to save summary text: {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-processing tool for:
1) Converting optimized .extxyz files into .cif format.
2) Computing RMSD between optimized and reference structures using
   pymatgen's StructureMatcher.

This script automates geometry comparison workflows commonly used in
crystal structure prediction, geometry validation, and MLIP benchmarking.
"""

import os
import json
from pathlib import Path
import pandas as pd
import argparse
from ase.io import read, write
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


# ======================================================================
# Argument Parsing
# ======================================================================

def parse_args():
    """
    Parse command-line arguments for batch conversion and RMSD computation.
    """
    parser = argparse.ArgumentParser(
        description="Batch: extxyz → cif conversion, and RMSD using pymatgen StructureMatcher."
    )

    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing optimized xyz (extxyz) files."
    )

    parser.add_argument(
        "--ref_dir", required=True,
        help="Directory containing reference CIF structures."
    )

    parser.add_argument(
        "--output", default="structure_matcher_rmsd.csv",
        help="Path of output CSV containing RMSD results."
    )

    parser.add_argument(
        "--suffix", default="_opt.extxyz",
        help="Suffix used to identify optimized xyz files. Default: '_opt.extxyz'."
    )

    parser.add_argument(
        "--ref_suffix", default=".cif",
        help="Suffix for reference CIF files. Default: '.cif'."
    )
    parser.add_argument(
        "--json_dir", default=None,
        help="Directory containing optimization JSON files. If omitted, the script tries "
             "<input_dir>/../individual_results and then <input_dir>/.. ."
    )
    parser.add_argument(
        "--base_total", type=int, default=None,
        help="Total number of Task 3 structures. If omitted, this is inferred from ref_dir."
    )

    return parser.parse_args()


# ======================================================================
# Step 1 — Convert optimized extxyz → CIF
# ======================================================================

def xyz2cif_batch(input_dir, output_dir, suffix):
    """
    Convert all optimized .extxyz structures into .cif format.

    Parameters
    ----------
    input_dir : Path
        Directory containing *.extxyz optimized structures.
    output_dir : Path
        Directory where converted CIFs will be written.
    suffix : str
        File suffix to match optimized .extxyz, e.g. '_opt.extxyz'.

    Notes
    -----
    - Each file is read with ASE, and written out as a CIF file.
    - The CIF filename is derived by removing the optimization suffix.
    """
    output_dir.mkdir(exist_ok=True)

    # Collect extxyz files
    xyz_files = list(input_dir.glob(f"*{suffix}"))
    print(f"[INFO] Found {len(xyz_files)} xyz files to convert...")

    for xyz_file in xyz_files:
        try:
            atoms = read(str(xyz_file))

            # Remove suffix (e.g. "_opt") from the stem to form the CIF filename
            prefix = xyz_file.stem.replace(suffix.replace(".extxyz", ""), "")
            cif_file = output_dir / f"{prefix}.cif"

            write(str(cif_file), atoms, format="cif")
            print(f"[OK] {xyz_file.name} → {cif_file.name}")

        except Exception as e:
            print(f"[FAIL] {xyz_file.name}: {e}")


# ======================================================================
# Step 2 — Compute RMSD using pymatgen StructureMatcher
# ======================================================================

def batch_rmsd_match(opt_dir, ref_dir, output_csv, ref_suffix=".cif"):
    """
    Compute RMSD between converted optimized CIFs and reference CIFs.

    Parameters
    ----------
    opt_dir : Path
        Directory containing optimized structures in CIF format.
    ref_dir : Path
        Directory containing reference CIF structures.
    output_csv : str
        Path where the RMSD results CSV will be saved.
    ref_suffix : str
        Suffix appended to reference structure filenames.

    Notes
    -----
    - pymatgen StructureMatcher is used for structural comparison.
    - If structures match, RMSD is computed.
    - If they don't match, RMSD is assigned as 2.0 (arbitrary penalty value).
    - Results are saved as a sorted CSV (ascending RMSD).
    """

    # Collect optimized CIF files
    opt_structs = list(opt_dir.glob("*.cif"))
    results = []

    # The StructureMatcher parameters are responsible for tolerances:
    matcher = StructureMatcher(stol=1.2, ltol=0.3, angle_tol=10)

    for opt_cif in opt_structs:
        prefix = opt_cif.stem

        # Expected reference CIF (same prefix)
        ref_cif = ref_dir / f"{prefix}{ref_suffix}"

        if not ref_cif.exists():
            print(f"Warning: reference structure not found for '{prefix}' ({ref_cif.name})")
            continue

        try:
            # Load structures with pymatgen
            struct_opt = Structure.from_file(str(opt_cif))
            struct_ref = Structure.from_file(str(ref_cif))

            # First check whether the matcher considers them as "same structure"
            if matcher.fit(struct_ref, struct_opt):

                # Compute RMSD
                rmsd_val = matcher.get_rms_dist(struct_ref, struct_opt)

                # Some pymatgen versions return (rmsd, max_dist); extract rmsd
                if isinstance(rmsd_val, tuple):
                    rmsd_val = rmsd_val[0]

                results.append({
                    "name": prefix,
                    "opt_file": opt_cif.name,
                    "ref_file": ref_cif.name,
                    "rmsd": rmsd_val,
                    "matched": True
                })

                print(f"{opt_cif.name} vs {ref_cif.name}: RMSD = {rmsd_val:.4f}")

            else:
                # If not matching under symmetry / tolerance assumptions
                print(f"{opt_cif.name} vs {ref_cif.name}: Not matched by StructureMatcher. Assign RMSD = 2.0")

                results.append({
                    "name": prefix,
                    "opt_file": opt_cif.name,
                    "ref_file": ref_cif.name,
                    "rmsd": 2.0,
                    "matched": False
                })

        except Exception as e:
            print(f"Error comparing {opt_cif.name} vs {ref_cif.name}: {e}")

            results.append({
                "name": prefix,
                "opt_file": opt_cif.name,
                "ref_file": ref_cif.name,
                "rmsd": 2.0,
                "matched": False
            })

    # ------------------------------------------------------------------
    # Save results to CSV
    # ------------------------------------------------------------------
    if results:
        df = pd.DataFrame(results).sort_values("rmsd")
        df.to_csv(output_csv, index=False)

        mean_rmsd = df["rmsd"].mean()
        print(f"Done! Results saved to {output_csv}")
        print(f"Average RMSD: {mean_rmsd:.4f}")

    else:
        print("No valid RMSD results to save.")


# ======================================================================
# Main Workflow
# ======================================================================

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    ref_dir = Path(args.ref_dir)

    # The converted CIF directory is automatically placed next to input_dir
    cif_dir = input_dir.parent / "converted_cif"

    print(f"\n[INFO] Converted CIFs will be stored in: {cif_dir.resolve()}")
    print("\n=== Step 1: Batch conversion extxyz → cif ===")
    xyz2cif_batch(input_dir, cif_dir, args.suffix)

    print("\n=== Step 2: pymatgen RMSD comparison ===")
    batch_rmsd_match(cif_dir, ref_dir, args.output, args.ref_suffix)

    expected_names = {p.stem for p in ref_dir.glob(f"*{args.ref_suffix}")}
    base_total = args.base_total or len(expected_names)

    print(f"\n=== Step 3: Structural-recovery failure counting "
          f"(steps >= 3000, converged=False, rmsd == 2.0, or missing; base {base_total}) ===")

    if args.json_dir:
        individual_dir = Path(args.json_dir)
    else:
        candidate = Path(input_dir).parent / "individual_results"
        individual_dir = candidate if candidate.is_dir() else Path(input_dir).parent
    json_files = list(individual_dir.glob("*.json"))

    failure_names = set()
    json_names = set()

    for jpath in json_files:
        mat = jpath.stem
        json_names.add(mat)

        try:
            with open(jpath, "r", encoding="utf-8") as f:
                content = json.load(f)

            steps = content.get("steps", 3000)
            is_converged = content.get("converged", False)

            if steps is None:
                steps = 3000

            if (steps >= 3000) or (not is_converged):
                failure_names.add(mat)

        except Exception:
            # If JSON cannot be read, treat as failure
            failure_names.add(mat)

    # Read RMSD CSV (from step 2)
    rmsd_names = set()
    rmsd_failure_names = set()
    output_csv = Path(args.output)
    if output_csv.exists():
        df_rmsd = pd.read_csv(output_csv)
        if "name" in df_rmsd.columns:
            name_col = "name"
        else:
            name_col = df_rmsd.columns[0]

        for _, row in df_rmsd.iterrows():
            name = str(row[name_col])
            rmsd_names.add(name)
            try:
                if float(row["rmsd"]) == 2.0:
                    rmsd_failure_names.add(name)
            except Exception:
                # If rmsd is invalid, treat as failure
                rmsd_failure_names.add(name)

    failure_names.update(rmsd_failure_names)

    # Missing items are counted at the structure-name level when references are available.
    total_json = len(json_names)
    total_csv = len(rmsd_names)
    if expected_names:
        missing_names = expected_names - (json_names | rmsd_names)
        failure_names.update(missing_names)
        missing_count = len(missing_names)
        failure_count = len(failure_names)
    else:
        total_items = max(total_json, total_csv)
        missing_count = max(0, base_total - total_items)
        failure_count = len(failure_names) + missing_count

    print(f"Total JSON items: {total_json}")
    print(f"Total CSV items: {total_csv}")
    print(f"Missing items (based on {base_total}): {missing_count}")
    failure_rate = failure_count / base_total if base_total else 0
    print(f"Structural-recovery failure rate (P): {failure_rate:.2%} ({failure_count}/{base_total})")

# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    main()

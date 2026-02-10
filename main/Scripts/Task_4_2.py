#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymorph ordering comparison tool.

Features:
- Read reference polymorph energy rankings from a CSV file.
- Load per-structure results from JSON files produced by an MLIP optimization workflow.
- Convert energies if needed (eV/cell → kJ/mol).
- Produce computed energy rankings for each polymorph.
- Detect large energy jumps.
- Check exact ordering match.
- Compute Kendall distance and inversion counts.
- Save a detailed JSON summary report.

This script is fully self-contained and can be executed directly.
"""

import os
import argparse
import json
import sys
import pandas as pd
from collections import OrderedDict


# ============================================================
# Load reference CSV
# ============================================================

def load_ref_csv(csv_path):
    """
    Load reference polymorph information from a CSV file.

    Expected columns:
        name, polymorph, n_mol, ref_energy

    Returns:
        ref_map: name → dict(polymorph, n_mol, ref_energy)
        poly_to_names: polymorph → list of names sorted by ref_energy ascending
    """

    df = pd.read_csv(csv_path)

    # name → info dict
    ref_map = {
        str(row["name"]): {
            "polymorph": row["polymorph"],
            "n_mol": row["n_mol"],
            "ref_energy": row["ref_energy"]
        }
        for _, row in df.iterrows()
    }

    # polymorph → sorted list of names
    poly_to_names = {
        poly: grp.sort_values("ref_energy")["name"].tolist()
        for poly, grp in df.groupby("polymorph")
    }

    return ref_map, poly_to_names


# ============================================================
# Load per-structure JSON result files
# ============================================================

def load_results(input_dir, ref_map):
    """
    Load predicted energies from JSON files in `input_dir`.

    Each JSON may provide:
        - name
        - polymorph
        - n_mol
        - energy_kjmol
        - energy_eV_cell

    Missing fields are filled using the reference CSV when possible.

    Returns:
        summary: polymorph → list of {name, energy_kjmol}
    """

    summary = {}

    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(input_dir, fname)

        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"[WARNING] Failed to load {fname}: {e}", file=sys.stderr)
            continue

        name = data.get("name", os.path.splitext(fname)[0])
        ref_info = ref_map.get(name, {})

        # Determine polymorph (prefer JSON, fallback to reference CSV)
        polymorph = data.get("polymorph") or ref_info.get("polymorph")

        # Number of molecules per cell
        n_mol = data.get("n_mol") or ref_info.get("n_mol")

        # Direct kJ/mol energy if available
        energy_kjmol = data.get("energy_kjmol")

        # Convert from eV/cell → kJ/mol if needed
        if energy_kjmol is None and "energy_eV_cell" in data and n_mol:
            try:
                energy_kjmol = float(data["energy_eV_cell"]) * 96.4853 / float(n_mol)
            except Exception as e:
                print(f"[WARNING] Failed converting energy for {fname}: {e}")
                continue

        # If still missing, skip this entry
        if polymorph is None or energy_kjmol is None:
            print(f"[INFO] Skipped {fname}: missing polymorph or energy_kjmol", file=sys.stderr)
            continue

        summary.setdefault(polymorph, []).append({
            "name": name,
            "energy_kjmol": energy_kjmol
        })

    return summary


# ============================================================
# Sort inside each polymorph
# ============================================================

def validate_and_sort(summary):
    """
    Sort entries within each polymorph group by energy_kjmol ascending.
    """
    return {
        poly: sorted(entries, key=lambda x: x["energy_kjmol"])
        for poly, entries in summary.items()
    }


# ============================================================
# Detect large energy jumps
# ============================================================

def check_energy_jump(energies, threshold=300):
    """
    Check if any adjacent energies differ by more than `threshold` kJ/mol.
    """
    energies = list(energies)
    for i in range(len(energies) - 1):
        if abs(energies[i + 1] - energies[i]) > threshold:
            return True
    return False


# ============================================================
# Kendall distance + inversion counting
# ============================================================

def _count_inversions(seq):
    """
    Count inversions in O(n log n) using a merge-sort based algorithm.

    seq: list of integers
    Returns:
        D: number of inversions
    """

    def sort_count(a):
        if len(a) <= 1:
            return a, 0

        mid = len(a) // 2
        left, inv_l = sort_count(a[:mid])
        right, inv_r = sort_count(a[mid:])
        merged = []
        i = j = 0
        inv = 0

        # Merge step with inversion counting
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inv += len(left) - i  # key step

        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_l + inv_r + inv

    _, inv = sort_count(list(seq))
    return inv


def kendall_error(computed_names, expected_names):
    """
    Compute normalized Kendall distance:

        E_kendall = D / C(n, 2)

    using only the intersection of computed and expected names.

    Returns:
        (E, D, total_pairs, n_used)
    """

    # Map expected names → rank index
    pos = {name: i for i, name in enumerate(expected_names)}

    # Keep intersection, in computed order
    seq = [pos[name] for name in computed_names if name in pos]

    n = len(seq)
    total_pairs = n * (n - 1) // 2

    if total_pairs == 0:
        return None, 0, 0, n

    D = _count_inversions(seq)
    E = D / total_pairs

    return E, D, total_pairs, n


# ============================================================
# Final comparison + output
# ============================================================

def write_compare_with_energy(path, corrected, poly_to_names, print_to_stdout=True):
    """
    For each polymorph group, compute:
        - computed ordering
        - reference ordering
        - match / mismatch
        - energy jump status
        - Kendall distance & inversion count

    Save as JSON.
    """

    result = {}
    n_total = 0
    n_match = 0

    for poly, entries in corrected.items():

        computed_names = [e["name"] for e in entries]
        computed_energies = [e["energy_kjmol"] for e in entries]

        expected = poly_to_names.get(poly, [])

        # Exact full-order match
        match_order = (computed_names == expected)

        # Check energy jumps
        has_big_jump = check_energy_jump(computed_energies)

        # The strict matching rule
        match = match_order and not has_big_jump

        if match:
            n_match += 1
        n_total += 1

        # Kendall distance
        E_k, D, total_pairs, n_used = kendall_error(computed_names, expected)

        # Store results
        result[poly] = {
            "computed": computed_names,
            "computed_energies": computed_energies,
            "expected": expected,
            "match": match,
            "match_order": match_order,
            "has_big_jump": has_big_jump,
            "kendall_E": E_k,
            "kendall_D": D,
            "kendall_pairs": total_pairs,
            "kendall_n_used": n_used
        }

        # Terminal printing
        if print_to_stdout:
            print(f"== {poly} ==")
            print(f"  computed: {computed_names}")
            print(f"  energies: {[f'{e:.2f}' for e in computed_energies]}")
            print(f"  expected: {expected}")
            print(f"  match: {match}  (order: {match_order}, jump: {has_big_jump})")
            print(f"  kendall_E={E_k}, D={D}, pairs={total_pairs}, used={n_used}")
            print()

    # Global accuracy
    accuracy = n_match / n_total if n_total else None

    # Output JSON
    output_json = OrderedDict()
    output_json["accuracy"] = accuracy
    output_json["matched"] = n_match
    output_json["total"] = n_total
    output_json["details"] = result

    with open(path, "w") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy:.3f} ({n_match}/{n_total})")
    print(f"Detailed results saved to {path}")


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare polymorph energy orderings and compute Kendall distances."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing per-structure JSON files")
    parser.add_argument("--output", required=True,
                        help="Output summary JSON file")
    parser.add_argument("--ref_csv", required=True,
                        help="Reference CSV file containing polymorph data")
    args = parser.parse_args()

    ref_map, poly_to_names = load_ref_csv(args.ref_csv)
    summary = load_results(args.input_dir, ref_map)

    if not summary:
        print(f"[ERROR] No valid results found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    corrected = validate_and_sort(summary)

    write_compare_with_energy(args.output, corrected, poly_to_names)


if __name__ == "__main__":
    main()

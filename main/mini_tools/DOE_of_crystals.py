# Cell A: Compute DOF for each inequivalent site in every CIF file
# and aggregate the results into a CSV file.

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

import spglib
from pymatgen.core.structure import Structure

# Parallel utilities (optional)
import concurrent.futures
import multiprocessing


# --------------------------
# Utility: Space group number → Crystal system
# --------------------------
def get_crystal_system_from_sg(sg_number: int) -> str:
    if sg_number is None:
        return "unknown"
    n = int(sg_number)
    if 1 <= n <= 2:
        return "Triclinic"
    if 3 <= n <= 15:
        return "Monoclinic"
    if 16 <= n <= 74:
        return "Orthorhombic"
    if 75 <= n <= 142:
        return "Tetragonal"
    if 143 <= n <= 167:
        return "Trigonal"
    if 168 <= n <= 194:
        return "Hexagonal"
    if 195 <= n <= 230:
        return "Cubic"
    return "unknown"


# --------------------------
# I/O: Read CIF → convert to spglib cell format
# --------------------------
def load_cif_to_spglib_cell(cifpath):
    """
    Returns:
        cell = (lattice(3x3), frac_positions(Nx3), atomic_numbers(N))
        pmg_struct = pymatgen Structure object (for element symbols, etc.)
    """
    s = Structure.from_file(cifpath)
    lattice = s.lattice.matrix
    frac_coords = [list(frac) for frac in s.frac_coords]
    species = [site.specie.number for site in s]  # Atomic numbers

    cell = (np.array(lattice), np.array(frac_coords), np.array(species))
    return cell, s


# --------------------------
# Symmetry operations & site stabilizer & DOF
# --------------------------
def get_symmetry_ops(cell, symprec=1e-5):
    """
    Return symmetry operations acting on fractional coordinates (R, t):
        rotations: (Nops, 3, 3) integer matrices
        translations: (Nops, 3) fractional translations
    """
    lattice, positions, numbers = cell
    sym = spglib.get_symmetry((lattice, positions, numbers), symprec=symprec)
    return sym["rotations"], sym["translations"]


def site_rotations_for_point(rots, trans, r_frac, tol=1e-5):
    """
    Given symmetry operations (rots, trans) and a fractional coordinate r_frac (3,),
    return rotation matrices that leave the point invariant (mod 1).
    """
    r = np.asarray(r_frac, dtype=float)
    if r.ndim != 1 or r.size != 3:
        raise ValueError(f"r_frac must be shape (3,), got {r.shape}")

    site_rots = []
    for R, t in zip(rots, trans):
        delta = R.dot(r) + t - r
        frac_part = delta - np.round(delta)  # Normalize to [-0.5, 0.5)
        if np.all(np.abs(frac_part) < tol):
            site_rots.append(np.array(R, dtype=float))

    return site_rots


def compute_dof_from_site_rotations(site_rots, tol_svd=1e-8):
    """
    DOF = nullspace dimension of stack(R - I).

    Uses adaptive SVD tolerance:
        tol = max(m,n) * eps * s_max
    """
    if len(site_rots) == 0:
        return 3

    A = np.vstack([R - np.eye(3) for R in site_rots])  # (3*Nops, 3)

    u, s, vh = np.linalg.svd(A)

    if tol_svd is None:
        eps = np.finfo(float).eps
        tol_svd = max(A.shape) * eps * (s[0] if s.size > 0 else 1.0)

    null_dim = int(np.sum(s < tol_svd))
    return max(0, min(3, null_dim))


# --------------------------
# Single-file analysis: one row per inequivalent site
# --------------------------
def analyze_cif_file(cifpath, symprec=1e-4):
    """
    Returns:
        list[dict], each dict corresponds to one representative
        (inequivalent) Wyckoff site.

    All calculations are performed using the original cell
    fractional coordinates and symmetry operations to ensure
    coordinate consistency.
    """
    cell, pmg_struct = load_cif_to_spglib_cell(cifpath)
    lattice, positions, numbers = cell

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    wyckoffs = dataset.get("wyckoffs") or [None]*len(positions)
    site_syms = dataset.get("site_symmetry_symbols") or [None]*len(positions)
    equiv = dataset.get("equivalent_atoms")

    sg_number = dataset.get("number")
    sg_symbol = dataset.get("international")
    crystal_system = get_crystal_system_from_sg(sg_number)

    # Retrieve symmetry operations once (reuse for all representative sites)
    rots, trans = get_symmetry_ops(cell, symprec=symprec)

    # Group atoms by equivalent_atoms → representative index
    groups = defaultdict(list)
    for i, rep in enumerate(equiv):
        groups[rep].append(i)

    rows = []
    for rep, indices in groups.items():
        rep_frac = np.array(positions[rep], dtype=float)

        site_rots = site_rotations_for_point(rots, trans, rep_frac, tol=symprec)
        dof = compute_dof_from_site_rotations(site_rots)

        multiplicity = len(indices)
        wy = wyckoffs[rep] if rep < len(wyckoffs) else None
        ss = site_syms[rep] if rep < len(site_syms) else None
        species = [pmg_struct[i].specie.symbol for i in indices]

        rows.append({
            "cif": os.path.basename(cifpath),
            "spacegroup_number": int(sg_number) if sg_number is not None else None,
            "spacegroup_symbol": sg_symbol,
            "crystal_system": crystal_system,
            "rep_index": int(rep),
            "wyckoff_letter": wy,
            "site_symmetry_symbol": ss,
            "multiplicity": int(multiplicity),
            "dof": int(dof),
            "rep_frac_coords": tuple(np.round(rep_frac, 8)),
            "indices": indices,
            "species_in_orbit": ",".join(species),
        })

    # Compute total structural DOF and total number of atoms
    structure_total_dof = int(sum(r["dof"] for r in rows))
    estimated_natoms = int(sum(r["multiplicity"] for r in rows))

    for r in rows:
        r["structure_total_dof"] = structure_total_dof
        r["structure_estimated_natoms"] = estimated_natoms

    print(structure_total_dof)
    return rows


# --------------------------
# Batch processing (single process)
# --------------------------
def batch_process_cif_dir(cif_dir, symprec=1e-4, out_csv=None):
    all_rows = []

    for fname in tqdm(sorted(os.listdir(cif_dir)), desc="Processing CIFs"):
        if not fname.lower().endswith(".cif"):
            continue

        path = os.path.join(cif_dir, fname)

        try:
            res = analyze_cif_file(path, symprec=symprec)
            all_rows.extend(res)
        except Exception as e:
            print(f"[ERR] {fname}: {e}")

    df = pd.DataFrame(all_rows)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    return df


# --------------------------
# Batch processing (parallel)
# --------------------------
def _worker_analyze(args):
    cifpath, symprec = args
    try:
        rows = analyze_cif_file(cifpath, symprec=symprec)
        return rows, None
    except Exception as e:
        return None, (os.path.basename(cifpath), str(e))


def batch_process_cif_dir_parallel(cif_dir, symprec=1e-4, out_csv=None, nprocs=None):
    cif_paths = [
        os.path.join(cif_dir, f)
        for f in sorted(os.listdir(cif_dir))
        if f.lower().endswith(".cif")
    ]

    if not cif_paths:
        print("No CIF files found in", cif_dir)
        return pd.DataFrame()

    if nprocs is None:
        nprocs = max(1, multiprocessing.cpu_count() - 1)

    tasks = [(p, symprec) for p in cif_paths]
    all_rows, errors = [], []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as exe:
            futures = {exe.submit(_worker_analyze, t): t[0] for t in tasks}

            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing CIFs (parallel)"
            ):
                rows, err = fut.result()
                if err is not None:
                    errors.append(err)
                else:
                    all_rows.extend(rows)

    except Exception as e:
        print("Parallel execution failed:", e)
        print("Falling back to sequential processing...")
        return batch_process_cif_dir(cif_dir, symprec=symprec, out_csv=out_csv)

    if errors:
        print(f"Errors for {len(errors)} files (showing up to 5):")
        for ef in errors[:5]:
            print(ef)

    df = pd.DataFrame(all_rows)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    return df


# --------------------------
# One row per structure (useful for plotting)
# --------------------------
def per_structure_summary(df_reps: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df_reps = per-site dataframe (output of analyze_* functions)

    Output:
        One row per CIF file containing space group,
        crystal system, and total structural DOF.
    """
    if df_reps.empty:
        return df_reps

    per_struct = df_reps.groupby("cif").agg(
        spacegroup_number=("spacegroup_number", "first"),
        spacegroup_symbol=("spacegroup_symbol", "first"),
        crystal_system=("crystal_system", "first"),
        structure_total_dof=("structure_total_dof", "first"),
        n_unique_sites=("rep_index", "nunique"),
    ).reset_index()

    return per_struct


# --------------------------
# Main execution
# --------------------------
# Modify to your CIF directory
cif_dir = r"/Users/lumuyu/Desktop/mp_20/cifs"

use_parallel = True
symprec = 1e-4
out_csv = "/Users/lumuyu/Desktop/wyckoff_all_reps_co_crystal.csv"

if use_parallel:
    df_reps = batch_process_cif_dir_parallel(
        cif_dir,
        symprec=symprec,
        out_csv=out_csv,
        nprocs=None
    )
else:
    df_reps = batch_process_cif_dir(
        cif_dir,
        symprec=symprec,
        out_csv=out_csv
    )

# Generate one-row-per-structure summary (for plotting)
per_struct = per_structure_summary(df_reps)
per_struct.to_csv("wyckoff_per_structure.csv", index=False)

print("Wrote wyckoff_per_structure.csv with", len(per_struct), "rows")
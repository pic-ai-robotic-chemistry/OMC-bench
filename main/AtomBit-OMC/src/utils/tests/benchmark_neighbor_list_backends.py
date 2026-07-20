import time

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list as ase_neighbor_list

from src.utils.neighbors import neighbor_list as matscipy_neighbor_list


def canonicalize(result, quantities):
    arrays = list(result) if isinstance(result, tuple) else [result]
    idx_map = {char: i for i, char in enumerate(quantities)}

    sender = np.asarray(arrays[idx_map["i"]])
    receiver = np.asarray(arrays[idx_map["j"]])
    shifts = np.asarray(arrays[idx_map["S"]]) if "S" in idx_map else None

    if shifts is None:
        order = np.lexsort((receiver, sender))
    else:
        order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], receiver, sender))

    canon = []
    for array in arrays:
        array_np = np.asarray(array)
        canon.append(array_np[order] if array_np.ndim > 0 else array_np)
    return canon


def compare_outputs(atoms, cutoff, quantities):
    ase_result = ase_neighbor_list(quantities, atoms, cutoff)
    matscipy_result = matscipy_neighbor_list(quantities, atoms, cutoff)

    ase_canon = canonicalize(ase_result, quantities)
    matscipy_canon = canonicalize(matscipy_result, quantities)

    for idx, (lhs, rhs) in enumerate(zip(ase_canon, matscipy_canon)):
        if lhs.shape != rhs.shape:
            return False, f"shape mismatch at item {idx}: {lhs.shape} vs {rhs.shape}"
        if np.issubdtype(lhs.dtype, np.floating):
            if not np.allclose(lhs, rhs, atol=1e-12, rtol=0.0):
                return False, f"float mismatch at item {idx}"
        else:
            if not np.array_equal(lhs, rhs):
                return False, f"integer mismatch at item {idx}"
    return True, "ok"


def benchmark(atoms, cutoff, quantities, repeat):
    timings = {}
    for name, func in [("ase", ase_neighbor_list), ("matscipy", matscipy_neighbor_list)]:
        start = time.perf_counter()
        for _ in range(repeat):
            func(quantities, atoms, cutoff)
        timings[name] = (time.perf_counter() - start) / repeat
    return timings


def build_systems():
    systems = []

    water = molecule("H2O")
    water.center(vacuum=6.0)
    systems.append(("water_molecule", water, 6.0))

    ethanol = molecule("CH3CH2OH")
    ethanol.center(vacuum=8.0)
    systems.append(("ethanol_molecule", ethanol, 6.0))

    si = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    systems.append(("si_bulk_64", si, 6.0))

    cu = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
    systems.append(("cu_bulk_108", cu, 6.0))

    mixed = Atoms(
        symbols=["C", "H", "H", "H", "H", "O", "N", "Si", "Cl", "S"],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.8, 0.8],
                [-0.8, -0.8, 0.8],
                [0.8, -0.8, -0.8],
                [-0.8, 0.8, -0.8],
                [4.0, 3.5, 3.0],
                [6.5, 2.5, 5.5],
                [8.0, 8.0, 8.0],
                [9.5, 7.5, 6.5],
                [3.0, 8.5, 2.0],
            ]
        ),
        cell=np.diag([12.0, 12.0, 12.0]),
        pbc=[True, True, True],
    )
    systems.append(("mixed_periodic_10", mixed, 6.0))
    return systems


def main():
    quantities_list = ["ijS", "ijdS"]
    repeat = 20

    print(f"Benchmark repeat count: {repeat}")
    for system_name, atoms, cutoff in build_systems():
        print(f"\n=== {system_name} | natoms={len(atoms)} | cutoff={cutoff} ===")
        for quantities in quantities_list:
            ok, detail = compare_outputs(atoms, cutoff, quantities)
            timings = benchmark(atoms, cutoff, quantities, repeat)
            speedup = timings["ase"] / timings["matscipy"] if timings["matscipy"] > 0 else float("inf")
            print(
                f"{quantities}: match={ok} ({detail}) | "
                f"ase={timings['ase'] * 1e3:.3f} ms | "
                f"matscipy={timings['matscipy'] * 1e3:.3f} ms | "
                f"speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    main()

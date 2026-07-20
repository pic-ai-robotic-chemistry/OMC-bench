from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

from src.sim.io import serialize_atoms


_OPTIMIZERS = {
    "bfgs": BFGS,
    "fire": FIRE,
    "lbfgs": LBFGS,
}


def _max_force(atoms: Atoms) -> float:
    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def run_relax(atoms: Atoms, calculator: Calculator, task_config: dict[str, Any]) -> dict[str, Any]:
    atoms = atoms.copy()
    atoms.calc = calculator

    optimizer_name = str(task_config.get("optimizer", "BFGS")).lower()
    if optimizer_name not in _OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    fmax = float(task_config.get("fmax", 0.05))
    steps = int(task_config.get("steps", 300))
    relax_cell = bool(task_config.get("relax_cell", False))
    record_trajectory = bool(task_config.get("record_trajectory", True))
    trajectory_interval = max(1, int(task_config.get("trajectory_interval", 1)))
    result_structure_format = str(task_config.get("result_structure_format", "cif"))
    return_stress = bool(task_config.get("return_stress", False))

    target = atoms
    if relax_cell:
        if not atoms.pbc.any():
            raise ValueError("relax_cell=True requires a periodic structure.")
        target = FrechetCellFilter(
            atoms,
            hydrostatic_strain=bool(task_config.get("hydrostatic_strain", False)),
            constant_volume=bool(task_config.get("constant_volume", False)),
            scalar_pressure=float(task_config.get("scalar_pressure", 0.0)),
        )

    trajectory = []

    def capture_frame() -> None:
        if not record_trajectory:
            return
        trajectory.append(
            {
                "step": int(getattr(optimizer, "nsteps", 0)),
                "energy": float(atoms.get_potential_energy()),
                "max_force": _max_force(atoms),
            }
        )

    optimizer = _OPTIMIZERS[optimizer_name](target, logfile=None)
    optimizer.attach(capture_frame, interval=trajectory_interval)

    capture_frame()
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    capture_frame()

    summary = {
        "converged": converged,
        "optimizer": optimizer_name.upper(),
        "fmax": fmax,
        "steps_requested": steps,
        "steps_completed": int(optimizer.nsteps),
        "final_energy": float(atoms.get_potential_energy()),
        "final_max_force": _max_force(atoms),
    }

    if return_stress:
        summary["final_stress"] = atoms.get_stress().tolist()

    return {
        "task_type": "relax",
        "summary": summary,
        "artifacts": {
            "final_structure": {
                "format": result_structure_format,
                "text": serialize_atoms(atoms, result_structure_format),
            },
            "trajectory": trajectory,
        },
    }

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase import units
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from src.sim.io import serialize_atoms


def _build_dynamics(atoms: Atoms, task_config: dict[str, Any]):
    ensemble = str(task_config.get("ensemble", "nvt_langevin")).lower()
    timestep_fs = float(task_config.get("timestep_fs", 1.0))

    if ensemble in {"nve", "verlet", "velocityverlet"}:
        return ensemble, VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    if ensemble in {"nvt", "nvt_langevin", "langevin"}:
        temperature_k = float(task_config.get("temperature_K", 300.0))
        friction_inv_fs = float(task_config.get("friction_inv_fs", 0.01))
        return (
            "nvt_langevin",
            Langevin(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=temperature_k,
                friction=friction_inv_fs / units.fs,
            ),
        )

    raise ValueError(f"Unsupported ensemble: {ensemble}")


def _record_observables(atoms: Atoms, step: int, timestep_fs: float) -> dict[str, Any]:
    potential_energy = float(atoms.get_potential_energy())
    kinetic_energy = float(atoms.get_kinetic_energy())
    total_energy = potential_energy + kinetic_energy
    temperature_k = float(atoms.get_temperature())
    return {
        "step": int(step),
        "time_fs": float(step * timestep_fs),
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "total_energy": total_energy,
        "temperature_K": temperature_k,
    }


def run_md(atoms: Atoms, calculator: Calculator, task_config: dict[str, Any]) -> dict[str, Any]:
    atoms = atoms.copy()
    atoms.calc = calculator

    steps = int(task_config.get("steps", 1000))
    timestep_fs = float(task_config.get("timestep_fs", 1.0))
    record_trajectory = bool(task_config.get("record_trajectory", True))
    trajectory_interval = max(1, int(task_config.get("trajectory_interval", 10)))
    result_structure_format = str(task_config.get("result_structure_format", "cif"))

    seed = task_config.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    if bool(task_config.get("initialize_velocities", True)):
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=float(task_config.get("temperature_K", 300.0)),
        )
        if bool(task_config.get("zero_linear_momentum", True)):
            Stationary(atoms)
        if bool(task_config.get("zero_rotation", True)):
            ZeroRotation(atoms)

    ensemble_name, dynamics = _build_dynamics(atoms, task_config)
    observables = []

    def capture_frame() -> None:
        frame = _record_observables(atoms, int(getattr(dynamics, "nsteps", 0)), timestep_fs)
        observables.append(frame)

    dynamics.attach(capture_frame, interval=trajectory_interval)
    capture_frame()
    dynamics.run(steps)
    capture_frame()

    summary = {
        "ensemble": ensemble_name,
        "steps_requested": steps,
        "timestep_fs": timestep_fs,
        "frames_recorded": len(observables),
        "final_energy": float(atoms.get_potential_energy()),
        "final_temperature_K": float(atoms.get_temperature()),
    }

    return {
        "task_type": "md",
        "summary": summary,
        "artifacts": {
            "final_structure": {
                "format": result_structure_format,
                "text": serialize_atoms(atoms, result_structure_format),
            },
            "trajectory": observables if record_trajectory else [],
            "observables": observables,
        },
    }

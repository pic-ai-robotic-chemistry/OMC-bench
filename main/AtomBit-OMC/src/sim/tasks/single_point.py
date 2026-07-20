from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator


def run_single_point(atoms: Atoms, calculator: Calculator, task_config: dict[str, Any]) -> dict[str, Any]:
    atoms = atoms.copy()
    atoms.calc = calculator

    return_forces = bool(task_config.get("return_forces", True))
    return_stress = bool(task_config.get("return_stress", False))
    structure_format = str(task_config.get("result_structure_format", "cif"))

    result = {
        "task_type": "single_point",
        "energy": float(atoms.get_potential_energy()),
        "n_atoms": int(len(atoms)),
        "chemical_symbols": list(atoms.get_chemical_symbols()),
        "structure_format": structure_format,
    }

    if return_forces:
        result["forces"] = atoms.get_forces().tolist()

    if return_stress:
        result["stress"] = atoms.get_stress().tolist()

    return result

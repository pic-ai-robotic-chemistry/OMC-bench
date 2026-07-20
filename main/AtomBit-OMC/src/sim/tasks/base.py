from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from src.sim.tasks.md import run_md
from src.sim.tasks.relax import run_relax
from src.sim.tasks.single_point import run_single_point


_TASK_RUNNERS = {
    "single_point": run_single_point,
    "relax": run_relax,
    "md": run_md,
}


def list_task_types() -> list[str]:
    return sorted(_TASK_RUNNERS.keys())


def run_task(task_type: str, atoms: Atoms, calculator: Calculator, task_config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized_task_type = task_type.lower().strip()
    if normalized_task_type not in _TASK_RUNNERS:
        raise ValueError(f"Unsupported task_type: {task_type}")
    runner = _TASK_RUNNERS[normalized_task_type]
    return runner(atoms=atoms, calculator=calculator, task_config=task_config or {})

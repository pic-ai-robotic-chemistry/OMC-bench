from pathlib import Path
from typing import Any

from src.sim.backends.atombit import AtomBitBackend
from src.sim.backends.base import BaseCalculatorBackend


_BACKEND_TYPES = {
    "atombit": AtomBitBackend,
}


def list_backend_names() -> list[str]:
    return sorted(_BACKEND_TYPES.keys())


def create_backend(backend_name: str, backend_config: dict[str, Any], base_dir: Path) -> BaseCalculatorBackend:
    backend_type = str(backend_config.get("type", "atombit")).lower()
    if backend_type not in _BACKEND_TYPES:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    backend_cls = _BACKEND_TYPES[backend_type]
    return backend_cls(backend_name=backend_name, backend_config=backend_config, base_dir=base_dir)

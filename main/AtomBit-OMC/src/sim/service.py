from pathlib import Path
from typing import Any

import yaml

from src.sim.backends import create_backend, list_backend_names
from src.sim.io import parse_structure_text
from src.sim.task_queue import TaskQueue
from src.sim.tasks import run_task
from src.sim.tasks.base import list_task_types


class AtomisticSimulationService:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self.base_dir = self.config_path.parent
        self.config = self._load_yaml(self.config_path)
        self.backends = self._load_backends()
        self.task_queue = TaskQueue(
            worker_fn=self.run_task_sync,
            max_tasks=int(self.server_settings().get("max_tasks", 1000)),
        )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        return payload

    def _load_backends(self) -> dict[str, Any]:
        backend_configs = self.config.get("backends", {})
        if not isinstance(backend_configs, dict) or not backend_configs:
            raise ValueError("Missing non-empty 'backends' section in simulation config.")

        loaded = {}
        for backend_name, backend_config in backend_configs.items():
            if not isinstance(backend_config, dict):
                raise ValueError(f"Backend config must be a mapping: {backend_name}")
            loaded[backend_name] = create_backend(backend_name, backend_config, self.base_dir)
        return loaded

    def server_settings(self) -> dict[str, Any]:
        return self.config.get("server", {})

    def api_key(self) -> str | None:
        raw_value = str(self.server_settings().get("api_key") or "").strip()
        return raw_value or None

    def list_backends(self) -> list[dict[str, Any]]:
        return [backend.model_info() for backend in self.backends.values()]

    def service_info(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "available_backend_types": list_backend_names(),
            "available_task_types": list_task_types(),
            "loaded_backends": self.list_backends(),
        }

    def run_task_sync(self, request: dict[str, Any]) -> dict[str, Any]:
        task_type = str(request.get("task_type", "")).strip()
        structure = request.get("structure", {})
        backend = request.get("backend", {})
        task_config = request.get("task_config", {}) or {}

        if not task_type:
            raise ValueError("task_type is required.")
        if not isinstance(structure, dict):
            raise ValueError("structure must be a mapping.")
        if not isinstance(backend, dict):
            raise ValueError("backend must be a mapping.")

        structure_text = structure.get("text")
        structure_format = structure.get("format", "cif")
        if not structure_text:
            raise ValueError("structure.text is required.")

        backend_name = str(backend.get("profile", "default"))
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend profile: {backend_name}")

        atoms = parse_structure_text(str(structure_text), str(structure_format))
        backend_instance = self.backends[backend_name]
        calculator = backend_instance.get_calculator()

        result = run_task(
            task_type=task_type,
            atoms=atoms,
            calculator=calculator,
            task_config=task_config,
        )
        result["backend"] = backend_instance.model_info()
        return result

    def resolve_execution_mode(self, request: dict[str, Any]) -> str:
        requested_mode = str(request.get("execution_mode", "") or "").strip().lower()
        task_type = str(request.get("task_type", "") or "").strip().lower()

        if requested_mode:
            if requested_mode not in {"sync", "async"}:
                raise ValueError(f"Unsupported execution_mode: {requested_mode}")
            return requested_mode

        if task_type == "single_point":
            return "sync"
        if task_type in {"relax", "md"}:
            return "async"
        return "sync"

    def submit_task(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.task_queue.submit(request)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.task_queue.get_task(task_id)

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        return self.task_queue.get_result(task_id)

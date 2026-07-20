from pathlib import Path
from typing import Any

import requests
import yaml
from ase import Atoms

from src.sim.io import infer_structure_format_from_path, serialize_atoms


def _resolve_client_config(config_path: str | None) -> tuple[str, str | None, float, str]:
    if config_path is None:
        return "http://127.0.0.1:8000", None, 120.0, "default"

    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("client"), dict):
        raise ValueError(f"Invalid client config: {path}")

    client_cfg = payload["client"]
    server_url = str(client_cfg.get("server_url", "http://127.0.0.1:8000")).rstrip("/")
    api_key = client_cfg.get("api_key")
    timeout = float(client_cfg.get("timeout", 120.0))
    default_backend_profile = str(client_cfg.get("default_backend_profile", "default"))
    return server_url, api_key, timeout, default_backend_profile


class AtomisticSimulationClient:
    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        default_backend_profile: str | None = None,
        config_path: str | None = None,
    ):
        cfg_server_url, cfg_api_key, cfg_timeout, cfg_backend_profile = _resolve_client_config(config_path)
        self.server_url = (server_url or cfg_server_url).rstrip("/")
        self.api_key = api_key if api_key is not None else cfg_api_key
        self.timeout = float(timeout if timeout is not None else cfg_timeout)
        self.default_backend_profile = default_backend_profile or cfg_backend_profile

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _get(self, path: str) -> Any:
        response = requests.get(
            f"{self.server_url}{path}",
            timeout=self.timeout,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = requests.post(
            f"{self.server_url}{path}",
            json=payload,
            timeout=self.timeout,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def service_info(self) -> dict[str, Any]:
        return self._get("/service_info")

    def list_backends(self) -> list[dict[str, Any]]:
        return self._get("/backends")

    def run_task(
        self,
        task_type: str,
        structure_text: str,
        structure_format: str = "cif",
        backend_profile: str | None = None,
        execution_mode: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task_type": task_type,
            "execution_mode": execution_mode,
            "structure": {
                "format": structure_format,
                "text": structure_text,
            },
            "backend": {
                "profile": backend_profile or self.default_backend_profile,
            },
            "task_config": task_config or {},
        }
        return self._post("/run_task", payload)

    def run_task_from_atoms(
        self,
        task_type: str,
        atoms: Atoms,
        structure_format: str = "cif",
        backend_profile: str | None = None,
        execution_mode: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        structure_text = serialize_atoms(atoms, structure_format=structure_format)
        return self.run_task(
            task_type=task_type,
            structure_text=structure_text,
            structure_format=structure_format,
            backend_profile=backend_profile,
            execution_mode=execution_mode,
            task_config=task_config,
        )

    def run_task_from_file(
        self,
        task_type: str,
        input_path: str | Path,
        structure_format: str | None = None,
        backend_profile: str | None = None,
        execution_mode: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = Path(input_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Input structure file not found: {file_path}")

        fmt = structure_format or infer_structure_format_from_path(file_path)
        structure_text = file_path.read_text(encoding="utf-8")
        return self.run_task(
            task_type=task_type,
            structure_text=structure_text,
            structure_format=fmt,
            backend_profile=backend_profile,
            execution_mode=execution_mode,
            task_config=task_config,
        )

    def submit_task(
        self,
        task_type: str,
        structure_text: str,
        structure_format: str = "cif",
        backend_profile: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task_type": task_type,
            "execution_mode": "async",
            "structure": {
                "format": structure_format,
                "text": structure_text,
            },
            "backend": {
                "profile": backend_profile or self.default_backend_profile,
            },
            "task_config": task_config or {},
        }
        return self._post("/submit_task", payload)

    def submit_task_from_atoms(
        self,
        task_type: str,
        atoms: Atoms,
        structure_format: str = "cif",
        backend_profile: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        structure_text = serialize_atoms(atoms, structure_format=structure_format)
        return self.submit_task(
            task_type=task_type,
            structure_text=structure_text,
            structure_format=structure_format,
            backend_profile=backend_profile,
            task_config=task_config,
        )

    def submit_task_from_file(
        self,
        task_type: str,
        input_path: str | Path,
        structure_format: str | None = None,
        backend_profile: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = Path(input_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Input structure file not found: {file_path}")

        fmt = structure_format or infer_structure_format_from_path(file_path)
        structure_text = file_path.read_text(encoding="utf-8")
        return self.submit_task(
            task_type=task_type,
            structure_text=structure_text,
            structure_format=fmt,
            backend_profile=backend_profile,
            task_config=task_config,
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self._get(f"/tasks/{task_id}")

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        return self._get(f"/tasks/{task_id}/result")

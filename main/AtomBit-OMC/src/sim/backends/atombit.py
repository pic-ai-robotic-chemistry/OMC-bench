from pathlib import Path
from typing import Any

import torch

from src.models import AtomBitModel
from src.sim.backends.base import BaseCalculatorBackend
from src.utils import AtomBitCalculator, AtomBitConfig, sanitize_model_config_dict


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


class AtomBitBackend(BaseCalculatorBackend):
    def __init__(self, backend_name: str, backend_config: dict[str, Any], base_dir: Path):
        super().__init__(backend_name=backend_name, backend_config=backend_config)
        self.base_dir = base_dir
        self.device = "cpu"
        self.calculator = self._build_calculator()

    def _build_calculator(self) -> AtomBitCalculator:
        checkpoint_path = self.backend_config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(f"Backend '{self.backend_name}' is missing checkpoint_path.")

        checkpoint_file = _resolve_path(checkpoint_path, self.base_dir)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        requested_device = str(self.backend_config.get("device", "cuda")).lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        checkpoint = torch.load(
            checkpoint_file,
            map_location=self.device,
            weights_only=False,
        )

        model_class = AtomBitModel

        config_override = self.backend_config.get("model_config_override")
        if config_override is not None:
            if not isinstance(config_override, dict):
                raise ValueError("model_config_override must be a mapping when provided.")
            model_config = AtomBitConfig(**sanitize_model_config_dict(config_override))
        else:
            saved_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
            if isinstance(saved_config, dict):
                model_config = AtomBitConfig(**sanitize_model_config_dict(saved_config))
            elif isinstance(saved_config, AtomBitConfig):
                model_config = saved_config
            else:
                model_config = AtomBitConfig()

        cutoff_override = self.backend_config.get("cutoff")
        if cutoff_override is not None:
            model_config.cutoff = float(cutoff_override)

        model = model_class(model_config)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format.")

        strict_load = bool(self.backend_config.get("strict_load", True))
        model.load_state_dict(state_dict, strict=strict_load)

        e0_path = self.backend_config.get("e0_path")
        if e0_path:
            e0_path = str(_resolve_path(e0_path, self.base_dir))

        return AtomBitCalculator(
            model,
            cutoff=model_config.cutoff,
            device=self.device,
            enable_stress=bool(self.backend_config.get("enable_stress", True)),
            add_e0=bool(self.backend_config.get("add_e0", False)),
            e0_dict=self.backend_config.get("e0_dict"),
            e0_path=e0_path,
            capture_weights=False,
            capture_descriptors=False,
            capture_charges=False,
        )

    def get_calculator(self) -> AtomBitCalculator:
        return self.calculator

    def model_info(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_type": "atombit",
            "device": self.device,
            "cutoff": float(self.calculator.cutoff),
            "add_e0": bool(getattr(self.calculator, "add_e0", False)),
        }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from dataclasses import fields
from pathlib import Path


class CalculatorFactory:
    @staticmethod
    def from_config(model_name, config_json="Calculator_defs.json"):
        """
        Return an ASE-compatible calculator object based on the model name
        and a JSON configuration file.
        """
        config_path = Path(config_json).resolve()
        config_dir = config_path.parent

        with open(config_path) as f:
            models = json.load(f)
        assert model_name in models, f"Model '{model_name}' not found in {config_json}!"

        entry = models[model_name]
        arch = entry["arch"]

        def resolve_path(path):
            resolved = Path(path.replace("$HOME", os.environ.get("HOME", ""))).expanduser()
            if not resolved.is_absolute():
                resolved = config_dir / resolved
            return str(resolved)

        model_path = resolve_path(entry["path"])

        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None:
            device = entry.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"

        if arch == "sevenn":
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model=model_path, device=device)

        elif arch == "mace_mp":
            from mace.calculators import mace_mp
            return mace_mp(
                model=model_path,
                dispersion=entry.get("dispersion", False),
                device=device,
            )

        elif arch in {"mace", "mace_model", "mace_omol", "mace_mpa"}:
            from mace.calculators import MACECalculator
            kwargs = {
                "model_paths": model_path,
                "device": device,
            }
            if "default_dtype" in entry:
                kwargs["default_dtype"] = entry["default_dtype"]
            return MACECalculator(**kwargs)

        elif arch == "atombit":
            if torch is None:
                raise ImportError("PyTorch is required for AtomBit calculators.")

            repo_main = Path(__file__).resolve().parents[1]
            atombit_root = repo_main / "AtomBit-OMC"
            if str(atombit_root) not in sys.path:
                sys.path.insert(0, str(atombit_root))

            from src.models import AtomBitModel
            from src.utils import AtomBitCalculator, AtomBitConfig, sanitize_model_config_dict

            print(f"Loading weights from: {model_path}")

            cfg_fields = {f.name for f in fields(AtomBitConfig)}
            model_params = {
                k: v for k, v in entry.get("model_params", {}).items()
                if k in cfg_fields
            }
            model_params.update({k: v for k, v in entry.items() if k in cfg_fields})

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            checkpoint_cfg = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
            if checkpoint_cfg is not None:
                if isinstance(checkpoint_cfg, AtomBitConfig):
                    cfg = checkpoint_cfg
                elif isinstance(checkpoint_cfg, dict):
                    cfg = AtomBitConfig(**sanitize_model_config_dict(checkpoint_cfg))
                else:
                    cfg = checkpoint_cfg
            else:
                cfg = AtomBitConfig(**sanitize_model_config_dict(model_params))

            model = AtomBitModel(cfg)

            state_dict = checkpoint
            for key in ("model_state_dict", "state_dict", "model"):
                if isinstance(checkpoint, dict) and key in checkpoint:
                    state_dict = checkpoint[key]
                    break

            new_state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }

            try:
                model.load_state_dict(new_state_dict, strict=entry.get("strict", True))
                print("Weights loaded successfully.")
            except RuntimeError as e:
                print(f"Weight loading failed: {e}")
                raise e

            e0_path = entry.get("e0_path")
            if e0_path:
                e0_path = resolve_path(e0_path)
                if os.path.isfile(e0_path):
                    e0_data = torch.load(e0_path, map_location="cpu", weights_only=False)
                    if isinstance(e0_data, dict):
                        for key in ("e0", "e0_dict", "atomic_energies"):
                            if key in e0_data and isinstance(e0_data[key], dict):
                                e0_data = e0_data[key]
                                break
                    if entry.get("load_external_e0", True) and hasattr(model, "load_external_e0"):
                        model.load_external_e0(e0_data, device=device, verbose=True)
                else:
                    raise FileNotFoundError(f"AtomBit e0_path not found: {e0_path}")

            calculator_kwargs = {
                "cutoff": entry.get("cutoff", cfg.cutoff),
                "device": device,
                "capture_weights": entry.get("capture_weights", False),
                "capture_descriptors": entry.get("capture_descriptors", False),
            }
            for optional_key in ("enable_stress", "add_e0"):
                if optional_key in entry:
                    calculator_kwargs[optional_key] = entry[optional_key]
            if e0_path and entry.get("pass_e0_to_calculator", False):
                calculator_kwargs["e0_path"] = e0_path

            return AtomBitCalculator(model, **calculator_kwargs)

        # ...You can extend support for more models here
        else:
            raise NotImplementedError(f"Model arch '{arch}' not supported!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


class CalculatorFactory:
    @staticmethod
    def from_config(model_name, config_json="Calculator_defs.json"):
        """
        Return an ASE-compatible calculator object based on the model name 
        and a JSON configuration file.
        """
        with open(config_json) as f:
            models = json.load(f)
        assert model_name in models, f"Model '{model_name}' not found in {config_json}!"

        entry = models[model_name]
        arch = entry["arch"]
        model_path = entry["path"].replace("$HOME", os.environ.get("HOME", ""))

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        if arch == "sevenn":
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model=model, device=cuda)

        elif arch == "mace_mp":
            from mace.calculators import mace_mp
            return mace_mp(model=model_path, dispersion=True, device=device)
        
        elif arch == "atombit":
            
            print(f"Loading weights from: {model_path}")
            
            # A. Load the file
            state_dict = torch.load(model_path, map_location=device)
            
            # B. If a checkpoint dictionary was saved, extract model_state_dict
            # (This ensures compatibility in case the saving format changes later)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # C. Handle the DDP "module." prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove "module." prefix
                else:
                    new_state_dict[k] = v
            
            # D. Load weights into the model
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("Weights loaded successfully.")
            except RuntimeError as e:
                print(f"Weight loading failed: {e}")
                # If strict=True fails, you can try strict=False
                # or check whether the config matches the model architecture
                raise e 
        

            # 3. Return the Calculator
            return HTGP_Calculator(model, cutoff=7.0, device=device)

        # ...You can extend support for more models here
        else:
            raise NotImplementedError(f"Model arch '{arch}' not supported!")

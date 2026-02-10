import json
import os

class CalculatorFactory:
    @staticmethod
    def from_config(model_name, config_json="Calculator_defs.json"):
        """
        从模型名和json配置文件，返回ASE兼容的calculator对象。
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
        
        elif arch == "lmy":
            import sys
            import torch # 确保引入 torch
            sys.path.append("no_topology") # 确保路径能找到 src
            
            from src.models import HTGPModel
            from src.utils import HTGP_Calculator, HTGPConfig
            
            # 1. 配置 (保持你原来的)
            config = HTGPConfig(
                num_atom_types=55, 
                hidden_dim=64, 
                num_layers=3, 
                cutoff=6.0, 
                num_rbf=10,
                use_L0=True, 
                use_L1=True,
                use_L2=True, 
                use_gating=True, 
                use_long_range=False
            )
            
            # 2. 搭建骨架
            model = HTGPModel(config)
            
            # ---------------------------------------------------------
            # 🔥🔥🔥 修正开始：加载权重 🔥🔥🔥
            # ---------------------------------------------------------
            print(f"Loading weights from: {model_path}")
            
            # A. 加载文件
            state_dict = torch.load(model_path, map_location=device)
            
            # B. 如果保存的是 checkpoint 字典，提取 model_state_dict
            # (防止你之后改了保存格式，这里做一个兼容)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # C. 处理 DDP 的 "module." 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v # 去掉 module.
                else:
                    new_state_dict[k] = v
            
            # D. 将权重载入模型
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("✅ Weights loaded successfully!")
            except RuntimeError as e:
                print(f"❌ Weight loading failed: {e}")
                # 如果 strict=True 失败，可以尝试 strict=False，或者检查 config 是否匹配
                raise e 
            
            # ---------------------------------------------------------
            # 🔥🔥🔥 修正结束 🔥🔥🔥
            # ---------------------------------------------------------

            # 3. 返回 Calculator
            return HTGP_Calculator(model, cutoff=6.0, device=device)

        # ...可以继续扩展更多模型
        else:
            raise NotImplementedError(f"Model arch '{arch}' not supported!")

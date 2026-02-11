import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress
from torch_geometric.data import Data

class HTGP_Calculator(Calculator):
    """
    完全适配 PotentialTrainer 训练逻辑的 ASE Calculator
    """
    implemented_properties = ['energy', 'forces', 'stress', 'descriptors', 'weights']
    def __init__(self, model, cutoff=6.0, device='cpu', **kwargs):
        """
        :param model: 你的 HTGPModel 实例
        :param cutoff: 必须与训练时的 cutoff 一致
        :param device: 'cpu' or 'cuda'
        """
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.cutoff = cutoff
        self.device = torch.device(device)
        
        # 1. 模型设置
        self.model.to(self.device)
        self.model.eval() # 必须是 eval 模式
        
        self.capture_weights = kwargs.get("capture_weights", False)
        self.capture_descriptors = kwargs.get("capture_descriptors", False)

        # 冻结模型参数（权重），只对输入求导
        for param in self.model.parameters():
            param.requires_grad = False

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # -----------------------------------------------------------
        # 1. 数据准备 (Data Preparation)
        # -----------------------------------------------------------
        data = self._atoms_to_pyg_data(atoms)
        
        # 开启位置梯度 (用于计算 Force)
        data.pos.requires_grad = True
        is_periodic = atoms.pbc.any()
        # -----------------------------------------------------------
        # 2. 虚拟应变构造 (Virtual Strain Construction) - 复刻训练代码
        # -----------------------------------------------------------
        # 只有在需要计算 Stress 时才构建计算图，但为了保持和训练一致性，
        # 建议始终保持这个路径，或者只在 properties 包含 stress 时开启。
        calc_stress = 'stress' in properties and is_periodic
        
        # ASE 每次只算一个结构，所以 num_graphs = 1
        displacement = torch.zeros((1, 3, 3), dtype=data.pos.dtype, device=self.device)
        
        if calc_stress:
            displacement.requires_grad = True
            
        symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
        
        # --- 应用变形 (Apply Deformation) ---
        # 训练代码: pos_deformed = batch.pos + torch.einsum('ni,nij->nj', batch.pos, strain_per_atom)
        # 对应 ASE 单图逻辑: pos @ strain.T
        # symmetric_strain[0] shape is (3, 3)
        strain_on_graph = symmetric_strain[0] 
        
        # 保存原始用于求导
        original_pos = data.pos
        original_cell = data.cell
        
        # 坐标变形
        pos_deformed = original_pos + torch.matmul(original_pos, strain_on_graph.T)
        
        # 晶胞变形 (训练代码: cell_deformed = original_cell + bmm(original_cell, symmetric_strain))
        cell_deformed = original_cell + torch.bmm(original_cell, symmetric_strain)
        
        # 将变形后的数据喂给模型
        data.pos = pos_deformed
        data.cell = cell_deformed
        
        # -----------------------------------------------------------
        # 3. 前向传播 (Forward)
        # -----------------------------------------------------------
        energy = self.model(data, capture_weights=self.capture_weights, capture_descriptors=self.capture_descriptors)         
        # -----------------------------------------------------------
        # 4. 结果提取与单位转换, 加上atom_ref (Results & Units)
        # -----------------------------------------------------------
        self.results['energy'] = energy.item()

        # -----------------------------------------------------------
        # 5. 反向传播求导 (Backward)
        # -----------------------------------------------------------
        # 我们需要对 original_pos 和 displacement 求导
        inputs_to_grad = [original_pos]
        if calc_stress:
            inputs_to_grad.append(displacement)
            
        grads = torch.autograd.grad(
            outputs=energy,
            inputs=inputs_to_grad,
            retain_graph=False,
            create_graph=False # 推理时不需要二阶导
        )
        
        # --- Force ---
        # F = -dE/dx
        forces = -grads[0]
        self.results['forces'] = forces.detach().cpu().numpy()
        # --- Stress ---
        if calc_stress:
            dE_dStrain = grads[1] # (1, 3, 3)
            
            # 你的训练代码逻辑: pred_stress = dE_dStrain / vol
            # volume calculation
            volume = torch.abs(torch.det(original_cell[0]))
            
            if volume > 1e-8:
                stress_tensor = dE_dStrain / volume
                stress_np = stress_tensor.squeeze(0).detach().cpu().numpy()
                self.results['stress'] = full_3x3_to_voigt_6_stress(stress_np)
            else:
                self.results['stress'] = np.zeros(6)
        if self.capture_weights:
            self.results['weights'] = self._get_weights()
            
        if self.capture_descriptors:
            self.results['descriptors'] = self._get_descriptors()

    def _get_weights(self):
        """
        从模型各层的 PhysicsGating 模块中提取捕获的权重。
        返回: List[Dict]，列表索引对应层数
        """
        weights_per_layer = []
        
        # 遍历模型的每一层
        for i, layer in enumerate(self.model.layers):
            # 'gating' 是 ModuleDict 中的 key，对应 PhysicsGating 实例
            gating_module = layer['gating']
            
            layer_data = {}
            
            # 辅助函数：如果属性存在且不为None，转numpy
            def extract(attr_name):
                if hasattr(gating_module, attr_name):
                    val = getattr(gating_module, attr_name)
                    if val is not None:
                        return val.detach().cpu().numpy()
                return None

            # 提取你在 PhysicsGating 中定义的捕获变量
            layer_data['g0'] = extract('g0_captured')
            layer_data['g1'] = extract('g1_captured')
            layer_data['g2'] = extract('g2_captured')
            layer_data['chem_logits'] = extract('chem_logits_captured')
            layer_data['phys_logits'] = extract('phys_logits_captured')
            layer_data['scalar_basis'] = extract('scalar_basis_captured') # 如有需要可取消注释
            layer_data['p_ij'] = extract('p_ij_captured') # 如有需要可取消注释

            weights_per_layer.append(layer_data)
            
        return weights_per_layer


    def _get_descriptors(self):
        """
        从模型中提取每一层的原子特征 (h0, h1, h2)。
        你的模型代码里已经把它们存到了 self.model.all_layer_descriptors 列表里。
        """
        if not hasattr(self.model, 'all_layer_descriptors'):
            return None
            
        descriptors_numpy = []
        
        # 遍历模型保存的特征列表
        for layer_feats in self.model.all_layer_descriptors:
            layer_dict = {}
            for key, val in layer_feats.items():
                # 模型里已经做了 .detach().cpu()，这里只需要转 numpy
                if val is not None:
                    layer_dict[key] = val.numpy()
                else:
                    layer_dict[key] = None
            descriptors_numpy.append(layer_dict)
            
        return descriptors_numpy


    def _atoms_to_pyg_data(self, atoms):
        """
        转换函数 (保持不变，除了不用压缩数据类型)
        """
        z = torch.from_numpy(atoms.get_atomic_numbers()).to(torch.long).to(self.device)
        pos = torch.from_numpy(atoms.get_positions()).to(torch.float32).to(self.device)
        
        # ASE get_cell returns [a, b, c], shape (3,3)
        # Model expects (1, 3, 3)
        cell_np = atoms.get_cell().array
        cell = torch.from_numpy(cell_np).to(torch.float32).unsqueeze(0).to(self.device)
        
        # Neighbor List
        i_idx, j_idx, _, S_integers = neighbor_list('ijdS', atoms, self.cutoff)
        
        edge_index = torch.tensor(np.vstack((i_idx, j_idx)), dtype=torch.long).to(self.device)
        shifts_int = torch.from_numpy(S_integers).to(torch.float32).to(self.device)
        # print(shifts_int)
        # Batch (ASE always 1 graph)
        num_atoms = len(atoms)
        batch = torch.zeros(num_atoms, dtype=torch.long).to(self.device)
        
        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            shifts_int=shifts_int,
            batch=batch
        )
        # 为 Data 注入 num_graphs 属性，防止模型内部报错
        data.num_graphs = 1
        
        return data
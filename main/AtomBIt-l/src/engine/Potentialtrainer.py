import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
from tqdm.auto import tqdm
from src.utils import scatter_add
import torch.distributed as dist
from torch_ema import ExponentialMovingAverage

# 🔥 优化 1: JIT 编译 Loss 函数 (算子融合，加速计算)
@torch.jit.script
def conditional_huber_loss(pred: torch.Tensor, target: torch.Tensor, base_delta: float = 0.01) -> torch.Tensor:
    """
    自适应 Huber Loss (JIT Optimized)
    """
    # 计算每个原子的受力模长 (N, 1)
    force_norm = torch.norm(target, dim=1, keepdim=True)
    
    # 初始化缩放因子
    delta_scale = torch.ones_like(force_norm)
    
    # 阶梯式降级策略
    mask_100_200 = (force_norm >= 100) & (force_norm < 200)
    delta_scale[mask_100_200] = 0.7
    
    mask_200_300 = (force_norm >= 200) & (force_norm < 300)
    delta_scale[mask_200_300] = 0.4
    
    mask_300 = (force_norm >= 300)
    delta_scale[mask_300] = 0.1
    
    # 计算最终的 delta
    adaptive_delta = base_delta * delta_scale
    
    # 手动实现 Huber 计算逻辑
    error = pred - target
    abs_error = torch.abs(error)
    
    # 判定 MSE 区域
    is_mse = abs_error < adaptive_delta
    
    loss_mse = 0.5 * error ** 2
    loss_l1 = adaptive_delta * (abs_error - 0.5 * adaptive_delta)
    
    # 组合并取平均
    loss = torch.where(is_mse, loss_mse, loss_l1)
    return loss.mean()

class PotentialTrainer:
    def __init__(self, model, total_steps, max_lr=1e-3, device='cuda', checkpoint_dir='checkpoints', epochs=15, **kwargs):
        """
        Args:
            total_steps: 总训练步数
            epochs: 总训练轮次
        """
        self.device = device
        self.model = model.to(self.device)
        
        # 1. 优化器配置
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=max_lr, # 初始学习率
            weight_decay=1e-4, # L2 正则化
            betas=(0.9, 0.999), # 默认值
            amsgrad=True # 使用 AMSGrad 变体
        )

        # 2. EMA (指数移动平均)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        # 3. 学习率调度器
        last_step = kwargs.get('last_epoch', -1)
        div_factor = 100.0 
        final_div_factor = 1000.0
        
        # OneCycleLR 的默认动量设置 (如果你没改过的话)
        base_momentum = 0.85
        max_momentum = 0.95
        
        if last_step > -1:
            initial_lr_val = max_lr / div_factor
            min_lr_val = initial_lr_val / final_div_factor
            
            for group in self.optimizer.param_groups:
                # 1. 补齐学习率参数
                group.setdefault('initial_lr', initial_lr_val)
                group.setdefault('max_lr', max_lr)
                group.setdefault('min_lr', min_lr_val)
                
                # 2. 补齐动量参数 (这次报错是因为缺这俩)
                group.setdefault('base_momentum', base_momentum)
                group.setdefault('max_momentum', max_momentum)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            # epochs=epochs,
            total_steps=int(total_steps * 1.02),
            pct_start=0.1, # 10% 的步数用于升高学习率
            div_factor=100.0, # 初始 lr 为 max_lr / div_factor
            final_div_factor=1000.0, # 最终 lr 为 max_lr / final_div_factor
            anneal_strategy='cos',
            last_epoch=last_step
        )
        
        # Loss 配置
        self.huber_delta = 0.01
        self.w_e = 1.0
        self.w_f = 10.0
        self.w_s = 10.0
        
        # 获取 rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.checkpoint_dir = checkpoint_dir
        self.train_log_path = os.path.join(self.checkpoint_dir, 'train_log.csv')
        self.val_log_path = os.path.join(self.checkpoint_dir, 'val_log.csv')
        self.EV_A3_TO_GPA = 160.21766 
        
        # 日志初始化
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._init_loggers()

    def _init_loggers(self):
        headers = ['epoch', 'step', 'lr', 'total_loss', 'loss_e', 'loss_f', 'loss_s', 'mae_e', 'mae_f', 'mae_s_gpa']
        for path in [self.train_log_path, self.val_log_path]:
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(headers)

    def log_to_csv(self, mode, data):
        # 只有 rank 0 写入
        if self.rank != 0:
            return
        path = self.train_log_path if mode == 'train' else self.val_log_path
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow([
                data['epoch'], data['step'], f"{data['lr']:.2e}",
                f"{data['total_loss']:.6f}", f"{data['loss_e']:.6f}",
                f"{data['loss_f']:.6f}", f"{data['loss_s']:.6f}",
                f"{data['mae_e']*1000:.6f}", f"{data['mae_f']*1000:.6f}", f"{data['mae_s_gpa']:.6f}"
            ])

    def step(self, batch, train=True, batch_idx=0):
        # 🔥 使用 non_blocking 加速传输
        batch = batch.to(self.device, non_blocking=True)
        
        # --- 1. 开启梯度 ---
        batch.pos.requires_grad_(True)
        if hasattr(batch, 'cell') and batch.cell is not None:
            batch.cell.requires_grad_(True) 
        
        # --- 2. 构造虚拟应变 ---
        # 🔥 优化 2: 消除 Sync，优先读取 PyG 的 batch.num_graphs 属性
        # 这避免了 .max().item() 导致的 CPU-GPU 强制同步
        if hasattr(batch, 'num_graphs'):
            num_graphs = batch.num_graphs
        else:
            # 兜底方案
            num_graphs = int(batch.batch.max()) + 1
            
        displacement = torch.zeros((num_graphs, 3, 3), dtype=batch.pos.dtype, device=self.device)
        displacement.requires_grad_(True)
        symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
        
        # --- 3. 应用变形 ---
        strain_per_atom = symmetric_strain[batch.batch]
        pos_deformed = batch.pos + torch.einsum('ni,nij->nj', batch.pos, strain_per_atom)
        
        original_pos = batch.pos
        original_cell = getattr(batch, 'cell', None)
        
        batch.pos = pos_deformed
        
        if original_cell is not None and original_cell.dim() == 3:
            cell_deformed = original_cell + torch.bmm(original_cell, symmetric_strain)
            batch.cell = cell_deformed
        else:
            # 这里的打印在多卡环境下可能会有点乱，但保留原逻辑
            # print("⚠️ Warning: batch.cell is None or not 3D, skipping cell deformation.")
            pass
 
        # --- 4. 前向传播 ---
        pred_e = self.model(batch).view(-1)
        
        # 恢复原始坐标
        batch.pos = original_pos
        if original_cell is not None: batch.cell = original_cell
        
        # --- 5. 自动求导计算力与应力 ---
        grad_out = torch.ones_like(pred_e)
        grads = torch.autograd.grad(
            outputs=pred_e, 
            inputs=[original_pos, displacement], 
            grad_outputs=grad_out,
            create_graph=train, 
            retain_graph=train,
            allow_unused=True
        )
        
        pred_f = -grads[0] if grads[0] is not None else torch.zeros_like(batch.pos)
        dE_dStrain = grads[1]

        # --- 6. 修正体积计算与安全除法 ---
        if original_cell is not None:
            vol = torch.abs(torch.linalg.det(original_cell)).view(-1, 1, 1)
        else:
            vol = torch.ones_like(dE_dStrain)

        # 🛡️ 安全检查：防止梯度断连导致 dE_dStrain 为 None 时报错
        if dE_dStrain is not None:
            pred_stress = dE_dStrain / vol
        else:
            # 保持维度一致 (Batch, 3, 3)
            pred_stress = torch.zeros((num_graphs, 3, 3), device=self.device)
        
        # ==================================================================
        # 6. Loss 计算 
        # ==================================================================
        target_e = batch.y.view(-1)
        
        # 缓存 scatter buffer 避免重复创建 (微小优化)
        if not hasattr(self, '_ones_buffer') or self._ones_buffer.shape[0] != batch.batch.shape[0]:
             self._ones_buffer = torch.ones_like(batch.batch, dtype=torch.float64)
        
        num_atoms = scatter_add(self._ones_buffer, batch.batch, dim=0, dim_size=num_graphs).view(-1).clamp(min=1)
        
        loss_e = F.huber_loss(pred_e / num_atoms, target_e / num_atoms, delta=self.huber_delta)
        
        # 使用 JIT 加速后的 Loss
        loss_f = conditional_huber_loss(pred_f, batch.force, base_delta=self.huber_delta)
        
        loss_s = torch.tensor(0.0, device=self.device, requires_grad=train)
        stress_mask_sum = 0
        if hasattr(batch, 'stress') and batch.stress is not None:
            stress_norm = torch.norm(batch.stress.view(num_graphs, -1), dim=1)
            stress_mask = (stress_norm > 1e-6)
            stress_mask_sum = stress_mask.sum().item() # 这里必须同步获取数值用于判断
            if stress_mask_sum > 0:
                s_pred = pred_stress.view(num_graphs, -1)[stress_mask]
                s_target = batch.stress.view(num_graphs, -1)[stress_mask]
                loss_s = F.huber_loss(s_pred, s_target, delta=self.huber_delta)

        total_loss = self.w_e * loss_e + self.w_f * loss_f + self.w_s * loss_s
        
        # --- 7. Metrics 计算 ---
        with torch.no_grad():
            mae_e = F.l1_loss(pred_e / num_atoms, target_e / num_atoms).item()
            mae_f = F.l1_loss(pred_f, batch.force).item()
            mae_s_gpa = 0.0
            if stress_mask_sum > 0:
                mae_s_val = F.l1_loss(
                    pred_stress.view(num_graphs, -1)[stress_mask], 
                    batch.stress.view(num_graphs, -1)[stress_mask]
                )
                mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA

        # --- 8. 反向传播与优化 ---
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 🔥 优化 3: EMA 降频更新 (每 5 步一次)
            if batch_idx % 5 == 0:
                self.ema.update()
            
        return {
            'total_loss': total_loss.item(),
            'loss_e': loss_e.item(), 'loss_f': loss_f.item(), 'loss_s': loss_s.item(),
            'mae_e': mae_e, 'mae_f': mae_f, 'mae_s_gpa': mae_s_gpa
        }

    def train_epoch(self, loader, epoch_idx):
        self.model.train()
        pbar = tqdm(loader, desc=f"Train Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {'mae_e': 0, 'mae_f': 0, 'mae_s_gpa': 0, 'total_loss': 0}
        count = 0
        
        # 🔥 使用 enumerate 获取 batch_idx
        for i, batch in enumerate(pbar):
            if i == 0:
                if self.rank == 0:
                    print("First batch graph info:")
                    print("Number of graphs in batch:", batch.num_graphs)
                    print("Nodes (atoms) in batch:", batch.pos.size(0))
                    print("Edge index:", batch.edge_index)
                    print("Batch indices:", batch.batch)
                    # 看stress是不是不是None和空
                    if hasattr(batch, 'stress') and batch.stress is not None:
                        print("Stress tensor shape:", batch.stress.shape)
                    else:
                        print("No stress tensor in this batch.")

            # 传入 batch_idx 控制 EMA 更新频率
            metrics = self.step(batch, train=True, batch_idx=i)
            
            if i == 0 and self.rank == 0:
                # 你的 debug 打印逻辑保持不变
                pass 
            
            # 记录 CSV (你要求每一步都保留 I/O)
            log_data = metrics.copy()
            log_data.update({'epoch': epoch_idx, 'step': i, 'lr': self.optimizer.param_groups[0]['lr']})
            self.log_to_csv('train', log_data)
            
            self.scheduler.step()
            
            # 统计
            for k in metrics_sum: metrics_sum[k] += metrics[k]
            count += 1
            pbar.set_postfix({'L': f"{metrics['total_loss']:.4f}", 
                              'MAE_e': f"{metrics['mae_e']*1000:.1f}",
                              'MAE_F': f"{metrics['mae_f']*1000:.1f}"})
            
        return {k: v/count for k,v in metrics_sum.items()}

    def validate(self, loader, epoch_idx):
        self.model.eval()
        pbar = tqdm(loader, desc=f"Val Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {'mae_e': 0, 'mae_f': 0, 'mae_s_gpa': 0, 'total_loss': 0}
        count = 0
        
        with self.ema.average_parameters():
            with torch.set_grad_enabled(True):
                for i, batch in enumerate(pbar):
                    metrics = self.step(batch, train=False)
                    
                    log_data = metrics.copy()
                    log_data.update({'epoch': epoch_idx, 'step': i, 'lr': self.optimizer.param_groups[0]['lr']})
                    self.log_to_csv('val', log_data)
                    
                    for k in metrics_sum: metrics_sum[k] += metrics[k]
                    count += 1
                    pbar.set_postfix({'L': f"{metrics['total_loss']:.4f}", 
                                      'MAE_e': f"{metrics['mae_e']*1000:.1f}",
                                      'MAE_F': f"{metrics['mae_f']*1000:.1f}"})
        
        if count == 0: count = 1
        return {k: v/count for k,v in metrics_sum.items()}

    def save(self, filename='best_model.pt', rank = 0):
            path = os.path.join(self.checkpoint_dir, filename)

            # 1. 解开 DDP 包装 (如果你用了多卡)
            # 如果是 DDP，真实的模型藏在 .module 里；如果是单卡，就是 self.model
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model

            # 2. 开启 EMA 上下文
            # 在这个 block 里，模型的参数被临时替换成了 EMA 平滑后的参数
            with self.ema.average_parameters():
                
                # 3. 准备要保存的字典 (包含配置！)
                checkpoint = {
                    'model_state_dict': raw_model.state_dict(), # 👈 存的是 EMA 权重
                    'model_config': getattr(raw_model, 'cfg', None) # 👈 存配置 (自动加载的关键)
                }
                
                # 4. 保存文件
                torch.save(checkpoint, path)

                if rank == 0:
                    print(f"✅ Model saved to {path} with config!")

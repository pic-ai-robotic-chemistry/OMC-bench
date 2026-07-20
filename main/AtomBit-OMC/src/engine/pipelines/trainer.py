import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import DEFAULT_DEVICE_STR


def _ema_cls():
    from torch_ema import ExponentialMovingAverage

    return ExponentialMovingAverage


def _tqdm():
    from tqdm.auto import tqdm

    return tqdm


def _trainer_logging():
    from .. import trainer_logging

    return trainer_logging


def _trainer_losses():
    from .. import trainer_losses

    return trainer_losses


def _trainer_step_utils():
    from .. import trainer_step_utils

    return trainer_step_utils


class PotentialTrainer:
    """Trainer for AtomBit residual-energy labels."""

    def __init__(
        self,
        model,
        total_steps,
        max_lr=1e-3,
        device=DEFAULT_DEVICE_STR,
        checkpoint_dir="checkpoints",
        epochs=15,
        finetune_mode=False,
        lr_gnn=1e-5,
        **kwargs,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.finetune_mode = finetune_mode
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.checkpoint_dir = checkpoint_dir
        self.use_direct_force = kwargs.get("use_direct_force", False)
        self.optimizer = self._build_optimizer(max_lr=max_lr, lr_gnn=lr_gnn)
        self.ema = _ema_cls()(self.model.parameters(), decay=0.999)
        self.scheduler = self._build_scheduler(total_steps=total_steps, max_lr=max_lr, **kwargs)

        self.huber_delta = kwargs.get("huber_delta", 0.01)
        self.w_e = kwargs.get("energy_loss_weight", 10.0)
        self.w_f = kwargs.get("force_loss_weight", 10.0)
        self.w_s = kwargs.get("stress_loss_weight", 10.0)
        self.EV_A3_TO_GPA = 160.21766

        self.train_log_path = os.path.join(self.checkpoint_dir, "train_log.csv")
        self.val_log_path = os.path.join(self.checkpoint_dir, "val_log.csv")
        self.perf_log_path = os.path.join(self.checkpoint_dir, "performance_log.csv")

        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            _trainer_logging().init_csv_logs(self.train_log_path, self.val_log_path)

    def _build_optimizer(self, max_lr, lr_gnn):
        if not self.finetune_mode:
            return optim.AdamW(self.model.parameters(), lr=max_lr, weight_decay=1e-4, betas=(0.9, 0.999), amsgrad=True)

        if self.rank == 0:
            print(f"[Trainer] Initializing in FINETUNE mode (lr={lr_gnn})")

        trainable_params = []
        trainable_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                trainable_names.append(name)

        self._write_parameter_group_log(trainable_names, lr_gnn)
        return optim.AdamW(
            [{"params": trainable_params, "lr": lr_gnn, "weight_decay": 1e-2}],
            amsgrad=True,
        )

    def _write_parameter_group_log(self, trainable_names, lr):
        if self.rank != 0:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        group_log_path = os.path.join(self.checkpoint_dir, "parameter_groups_check.txt")
        with open(group_log_path, "w") as handle:
            handle.write(f"=== Total Trainable Params: {len(trainable_names)} ===\n\n")
            handle.write(f"--- Trainable Parameters (lr={lr}) [Count: {len(trainable_names)}] ---\n")
            if len(trainable_names) == 0:
                handle.write("WARNING: NO TRAINABLE PARAMETERS FOUND!\n")
            for name in trainable_names:
                handle.write(f"{name}\n")
        print(f"Parameter groups saved to: {group_log_path}")

    def _build_scheduler(self, total_steps, max_lr, **kwargs):
        if self.finetune_mode:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=4, threshold=1e-3, min_lr=1e-7
            )

        last_step = kwargs.get("last_epoch", -1)
        div_factor = 100.0
        final_div_factor = 1000.0
        base_momentum = 0.85
        max_momentum = 0.95
        if last_step > -1:
            initial_lr_val = max_lr / div_factor
            min_lr_val = initial_lr_val / final_div_factor
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", initial_lr_val)
                group.setdefault("max_lr", max_lr)
                group.setdefault("min_lr", min_lr_val)
                group.setdefault("base_momentum", base_momentum)
                group.setdefault("max_momentum", max_momentum)

        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=int(total_steps * 1.02),
            pct_start=0.1,
            div_factor=100.0,
            final_div_factor=1000.0,
            anneal_strategy="cos",
            last_epoch=last_step,
        )

    def log_to_csv(self, mode, data):
        if self.rank != 0:
            return
        _trainer_logging().append_metrics_row(self.train_log_path if mode == "train" else self.val_log_path, data)

    def load_checkpoint(self, checkpoint_dict):
        if "optimizer_state_dict" in checkpoint_dict:
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
            if self.rank == 0:
                print("Optimizer state loaded.")

        if "scheduler_state_dict" in checkpoint_dict:
            self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
            if self.rank == 0:
                print("Scheduler state loaded.")

        if "ema_state_dict" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema_state_dict"])
            self.ema.to(self.device)
            if self.rank == 0:
                print("EMA state loaded.")
            return

        if self.rank == 0:
            print("Warning: No EMA state in checkpoint. Resetting EMA from current model weights.")
        self.ema = _ema_cls()(self.model.parameters(), decay=0.999)
        self.ema.to(self.device)

    def _forward_batch(self, batch, train):
        batch = batch.to(self.device, non_blocking=True)
        batch.pos.requires_grad_(True)
        if hasattr(batch, "cell") and batch.cell is not None:
            batch.cell.requires_grad_(True)

        step_utils = _trainer_step_utils()
        num_graphs, displacement, symmetric_strain = step_utils.build_symmetric_strain(batch, self.device)
        original_pos, original_cell = step_utils.apply_batch_deformation(batch, symmetric_strain)
        result = self.model(batch)
        pred_e = result["energy"] if isinstance(result, dict) else result.view(-1)
        _trainer_losses().check_finite("pred_e", pred_e)
        step_utils.restore_batch_geometry(batch, original_pos, original_cell)

        if self.use_direct_force:
            return self._build_direct_force_outputs(batch, num_graphs, pred_e, result)
        return self._build_autograd_outputs(batch, train, num_graphs, pred_e, original_pos, original_cell, displacement)

    def _build_autograd_outputs(self, batch, train, num_graphs, pred_e, original_pos, original_cell, displacement):
        grads = torch.autograd.grad(
            outputs=pred_e,
            inputs=[original_pos, displacement],
            grad_outputs=torch.ones_like(pred_e),
            create_graph=train,
            retain_graph=train,
            allow_unused=True,
        )
        pred_f = -grads[0] if grads[0] is not None else torch.zeros_like(batch.pos)
        dE_dStrain = grads[1]

        if dE_dStrain is not None:
            if original_cell is not None:
                batched_cell = _trainer_step_utils().as_batched_cell(original_cell, num_graphs)
                vol = torch.abs(torch.linalg.det(batched_cell)).view(-1, 1, 1).clamp(min=1e-12)
            else:
                vol = torch.ones((num_graphs, 1, 1), device=self.device, dtype=pred_e.dtype)
            pred_stress = dE_dStrain / vol
        else:
            pred_stress = torch.zeros((num_graphs, 3, 3), device=self.device)

        losses = _trainer_losses()
        losses.check_finite("pred_f", pred_f)
        losses.check_finite("pred_stress", pred_stress)
        return pred_e, pred_f, pred_stress, num_graphs

    def _build_direct_force_outputs(self, batch, num_graphs, pred_e, result):
        pred_f = result["force"]
        pred_stress = torch.zeros((num_graphs, 3, 3), device=self.device)
        losses = _trainer_losses()
        losses.check_finite("pred_f", pred_f)
        losses.check_finite("pred_stress", pred_stress)
        return pred_e, pred_f, pred_stress, num_graphs

    def _compute_losses(self, batch, train, num_graphs, pred_e, pred_f, pred_stress):
        target_e = batch.y.view(-1)
        step_utils = _trainer_step_utils()
        losses = _trainer_losses()
        num_atoms = step_utils.cached_num_atoms(self, batch, num_graphs)
        loss_e = F.huber_loss(pred_e / num_atoms, target_e / num_atoms, delta=self.huber_delta)
        loss_f = losses.conditional_huber_loss(pred_f, batch.force, base_delta=self.huber_delta)

        loss_s = torch.tensor(0.0, device=self.device, requires_grad=train)
        stress_mask = None
        stress_mask_sum = 0
        if hasattr(batch, "stress") and batch.stress is not None and not self.use_direct_force:
            stress_norm = torch.norm(batch.stress.view(num_graphs, -1), dim=1)
            stress_mask = stress_norm > 1e-6
            stress_mask_sum = stress_mask.sum().item()
            if stress_mask_sum > 0:
                s_pred = pred_stress.view(num_graphs, -1)[stress_mask]
                s_target = batch.stress.view(num_graphs, -1)[stress_mask]
                loss_s = F.huber_loss(s_pred, s_target, delta=self.huber_delta)

        total_loss = self.w_e * loss_e + self.w_f * loss_f + self.w_s * loss_s
        losses.check_finite("loss_e", loss_e)
        losses.check_finite("loss_f", loss_f)
        losses.check_finite("loss_s", loss_s)
        losses.check_finite("total_loss", total_loss)

        return {
            "target_e": target_e,
            "num_atoms": num_atoms,
            "loss_e": loss_e,
            "loss_f": loss_f,
            "loss_s": loss_s,
            "total_loss": total_loss,
            "stress_mask": stress_mask,
            "stress_mask_sum": stress_mask_sum,
        }

    def _compute_metrics(self, batch, num_graphs, pred_e, pred_f, pred_stress, loss_info):
        with torch.no_grad():
            mae_e = F.l1_loss(pred_e / loss_info["num_atoms"], loss_info["target_e"] / loss_info["num_atoms"]).item()
            mae_f = F.l1_loss(pred_f, batch.force).item()
            mae_s_gpa = 0.0
            if loss_info["stress_mask_sum"] > 0:
                mae_s_val = F.l1_loss(
                    pred_stress.view(num_graphs, -1)[loss_info["stress_mask"]],
                    batch.stress.view(num_graphs, -1)[loss_info["stress_mask"]],
                )
                mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA

        return {
            "mae_e": mae_e,
            "mae_f": mae_f,
            "mae_s_gpa": mae_s_gpa,
            "finite_pred_e": True,
            "finite_pred_f": True,
            "finite_pred_stress": True,
            "finite_loss_e": True,
            "finite_loss_f": True,
            "finite_loss_s": True,
            "finite_total_loss": True,
        }

    def _optimize(self, total_loss, batch_idx):
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print("NON-FINITE grad in", name)
                raise FloatingPointError("grad")

        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print("PARAM BECAME NON-FINITE:", name)
                raise FloatingPointError("param")
        if batch_idx % 5 == 0:
            self.ema.update()
        return True, True, grad_norm

    def step(self, batch, train=True, batch_idx=0):
        pred_e, pred_f, pred_stress, num_graphs = self._forward_batch(batch, train=train)
        loss_info = self._compute_losses(batch, train, num_graphs, pred_e, pred_f, pred_stress)
        metrics = self._compute_metrics(batch, num_graphs, pred_e, pred_f, pred_stress, loss_info)
        if train:
            finite_grad, finite_param, grad_norm = self._optimize(loss_info["total_loss"], batch_idx=batch_idx)
        else:
            finite_grad, finite_param = True, True
            grad_norm = torch.tensor(0.0, device=self.device)

        return {
            "total_loss": loss_info["total_loss"].item(),
            "loss_e": loss_info["loss_e"].item(),
            "loss_f": loss_info["loss_f"].item(),
            "loss_s": loss_info["loss_s"].item(),
            **metrics,
            "finite_grad": finite_grad,
            "finite_param": finite_param,
            "grad_norm": float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        }

    def _log_step(self, mode, epoch_idx, step_idx, metrics):
        payload = metrics.copy()
        payload.update({"epoch": epoch_idx, "step": step_idx, "lr": _trainer_step_utils().current_lr(self.optimizer, self.finetune_mode)})
        self.log_to_csv(mode, payload)

    def _maybe_log_first_batch(self, batch, batch_idx):
        if batch_idx != 0 or self.rank != 0:
            return
        print("First batch graph info:")
        print("Number of graphs in batch:", batch.num_graphs)
        print("Nodes (atoms) in batch:", batch.pos.size(0))
        print("Edge index:", batch.edge_index)
        print("Batch indices:", batch.batch)
        if hasattr(batch, "stress") and batch.stress is not None:
            print("Stress tensor shape:", batch.stress.shape)
        else:
            print("No stress tensor in this batch.")

    def _maybe_log_perf(self, batch, batch_idx, step_duration):
        if self.rank != 0:
            return
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        throughput = batch.num_graphs / step_duration if step_duration > 0 else 0.0
        if batch_idx % 10 == 0:
            print(f"Step {batch_idx} | Mem: {max_mem:.0f}MB | Speed: {throughput:.1f} g/s | Atoms: {batch.pos.size(0)}")
        _trainer_logging().append_perf_row(
            self.perf_log_path,
            step=batch_idx,
            memory_mb=max_mem,
            step_duration=step_duration,
            throughput=throughput,
            batch_size_graphs=batch.num_graphs,
            num_atoms=batch.pos.size(0),
        )

    def train_epoch(self, loader, epoch_idx):
        self.model.train()
        pbar = _tqdm()(loader, desc=f"Train Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {"mae_e": 0, "mae_f": 0, "mae_s_gpa": 0, "total_loss": 0}
        count = 0
        max_steps = _trainer_step_utils().resolve_max_steps(self.model, self.finetune_mode)
        if self.rank == 0:
            _trainer_logging().ensure_perf_log(self.perf_log_path)

        for batch_idx, batch in enumerate(pbar):
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            self._maybe_log_first_batch(batch, batch_idx)
            metrics = self.step(batch, train=True, batch_idx=batch_idx)
            self._log_step("train", epoch_idx, batch_idx, metrics)
            if not self.finetune_mode:
                self.scheduler.step()
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
            count += 1
            pbar.set_postfix({"L": f"{metrics['total_loss']:.4f}", "MAE_e": f"{metrics['mae_e'] * 1000:.1f}", "MAE_F": f"{metrics['mae_f'] * 1000:.1f}"})
            self._maybe_log_perf(batch, batch_idx, time.time() - start_time)
            if max_steps is not None and (batch_idx + 1) >= max_steps:
                break
        return _trainer_step_utils().average_metrics(metrics_sum, count)

    def validate(self, loader, epoch_idx):
        self.model.eval()
        pbar = _tqdm()(loader, desc=f"Val Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {"mae_e": 0, "mae_f": 0, "mae_s_gpa": 0, "total_loss": 0}
        count = 0
        max_steps = _trainer_step_utils().resolve_max_steps(self.model, self.finetune_mode)
        with self.ema.average_parameters():
            with torch.set_grad_enabled(True):
                for batch_idx, batch in enumerate(pbar):
                    metrics = self.step(batch, train=False)
                    self._log_step("val", epoch_idx, batch_idx, metrics)
                    for key in metrics_sum:
                        metrics_sum[key] += metrics[key]
                    count += 1
                    pbar.set_postfix({"L": f"{metrics['total_loss']:.4f}", "MAE_e": f"{metrics['mae_e'] * 1000:.1f}", "MAE_F": f"{metrics['mae_f'] * 1000:.1f}"})
                    if max_steps is not None and (batch_idx + 1) >= max_steps:
                        break
        return _trainer_step_utils().average_metrics(metrics_sum, count)

    def step_scheduler_on_val(self, val_loss):
        if self.finetune_mode:
            self.scheduler.step(val_loss)

    def save(self, filename="best_model.pt", rank=0):
        path = os.path.join(self.checkpoint_dir, filename)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        with self.ema.average_parameters():
            torch.save({"model_state_dict": raw_model.state_dict(), "model_config": getattr(raw_model, "cfg", None)}, path)
        if rank == 0:
            print(f"Model saved to {path} with config.")

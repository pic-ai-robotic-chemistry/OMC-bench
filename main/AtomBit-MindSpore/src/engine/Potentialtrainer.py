"""
Potential trainer for MindSpore.

Handles training/validation loops, adaptive Huber loss, EMA, differential
learning rates for finetune (GNN vs long-range), and distributed training.
"""

import csv
import os

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore import save_checkpoint
import mindspore.communication as dist
from mindspore.communication import get_group_size
from mindspore.experimental.optim import AdamW

from src.utils import scatter_add
from src.utils.Scheduler import OneCycleLR
from tqdm.auto import tqdm

mint = ms.mint


def conditional_huber_loss(
    pred: Tensor, target: Tensor, base_delta: float = 0.01
) -> Tensor:
    """
    Adaptive Huber loss with force-magnitude-dependent delta.

    Uses smaller delta for large forces (e.g. >100) to reduce gradient
    dominance of high-force atoms. Implemented for MindSpore.
    """
    force_norm = mint.norm(target, dim=1, keepdim=True)
    delta_scale = mint.ones_like(force_norm)

    mask_100_200 = mint.logical_and(force_norm >= 100, force_norm < 200)
    delta_scale = ops.select(
        mask_100_200, Tensor(0.7, ms.float32), delta_scale
    )
    mask_200_300 = mint.logical_and(force_norm >= 200, force_norm < 300)
    delta_scale = ops.select(
        mask_200_300, Tensor(0.4, ms.float32), delta_scale
    )
    mask_300 = force_norm >= 300
    delta_scale = ops.select(mask_300, Tensor(0.1, ms.float32), delta_scale)

    adaptive_delta = base_delta * delta_scale
    error = pred - target
    abs_error = mint.abs(error)
    is_mse = abs_error < adaptive_delta

    loss_mse = 0.5 * error.astype(ms.float32) ** 2
    loss_l1 = adaptive_delta * (abs_error - 0.5 * adaptive_delta)
    loss = ops.select(is_mse, loss_mse, loss_l1)
    return loss.mean()


class ExponentialMovingAverage:
    """
    Exponential moving average of model parameters (MindSpore implementation).

    MindSpore does not provide torch_ema; this is a manual implementation.
    Parameter swap-in uses set_data; behavior may differ from PyTorch.
    """

    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow_params = []
        self.collected_params = []

        for param in parameters:
            if param.requires_grad:
                self.shadow_params.append(param.data.copy())
            else:
                self.shadow_params.append(None)

    def update(self, parameters=None):
        """Update shadow parameters with current model parameters."""
        if parameters is None:
            return

        for s_param, param in zip(self.shadow_params, parameters):
            if s_param is not None and param.requires_grad:
                s_param.set_data(
                    self.decay * s_param + (1.0 - self.decay) * param.data
                )

    def average_parameters(self):
        """Context manager that temporarily uses EMA parameters for evaluation."""
        class _ContextManager:
            def __init__(self, ema_obj, parameters):
                self.ema_obj = ema_obj
                self.parameters = list(parameters)

            def __enter__(self):
                self.ema_obj.collected_params = []
                for param, s_param in zip(
                    self.parameters, self.ema_obj.shadow_params
                ):
                    if s_param is not None:
                        self.ema_obj.collected_params.append(param.data.copy())
                        param.set_data(s_param)
                    else:
                        self.ema_obj.collected_params.append(None)
                return self

            def __exit__(self, *args):
                for param, c_param in zip(
                    self.parameters, self.ema_obj.collected_params
                ):
                    if c_param is not None:
                        param.set_data(c_param)

        return _ContextManager(self, [])


class PotentialTrainer:
    """
    MindSpore potential trainer: training/validation, EMA, optional
    differential LR for finetune (GNN vs long-range), and distributed support.
    """

    def __init__(
        self,
        model,
        total_steps,
        max_lr=1e-3,
        device="Ascend",
        checkpoint_dir="checkpoints",
        epochs=15,
        finetune_mode=False,
        lr_gnn=1e-5,
        lr_les=1e-3,
        **kwargs,
    ):
        """
        Args:
            model: MindSpore model (e.g. potential network).
            total_steps: Total training steps (for OneCycleLR).
            max_lr: Maximum learning rate.
            device: 'Ascend', 'GPU', or 'CPU'.
            checkpoint_dir: Directory for checkpoints and logs.
            epochs: Total training epochs.
            finetune_mode: If True, use differential LR (GNN low, LES high).
            lr_gnn: Learning rate for GNN parameters in finetune mode.
            lr_les: Learning rate for long-range/sigma parameters in finetune.
        """
        self.device = device
        self.model = model
        self.finetune_mode = finetune_mode
        self.global_step = 0

        try:
            self.rank = dist.get_rank()
        except RuntimeError:
            self.rank = 0

        self.checkpoint_dir = checkpoint_dir

        if kwargs.get("only_les", False):
            for param in self.model.trainable_params():
                param_name = param.name
                if "long_range" in param_name or "sigma" in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if self.finetune_mode:
            if self.rank == 0:
                print(
                    f"[Trainer] Initializing in FINETUNE mode "
                    f"(GNN={lr_gnn}, LES={lr_les})"
                )

            gnn_params, les_params = [], []
            gnn_names, les_names = [], []

            for param in self.model.trainable_params():
                param_name = param.name
                if not param.requires_grad:
                    continue
                if "long_range" in param_name or "sigma" in param_name:
                    les_params.append(param)
                    les_names.append(param_name)
                else:
                    gnn_params.append(param)
                    gnn_names.append(param_name)

            if self.rank == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                group_log_path = os.path.join(
                    self.checkpoint_dir, "parameter_groups_check.txt"
                )
                with open(group_log_path, "w") as f:
                    total_n = len(gnn_names) + len(les_names)
                    f.write(f"=== Total Trainable Params: {total_n} ===\n\n")
                    f.write(
                        f"--- Group LES (High LR: {lr_les}) "
                        f"[Count: {len(les_names)}] ---\n"
                    )
                    if len(les_names) == 0:
                        f.write(
                            "WARNING: NO PARAMETERS FOUND IN LES GROUP!\n"
                        )
                    for n in les_names:
                        f.write(f"{n}\n")
                    if len(gnn_names) > 0:
                        f.write(
                            f"\n--- Group GNN (Low LR: {lr_gnn}) "
                            f"[Count: {len(gnn_names)}] ---\n"
                        )
                        for n in gnn_names:
                            f.write(f"{n}\n")
                print(f"Parameter groups saved to: {group_log_path}")

            if len(gnn_names) > 0:
                group_params = [
                    {
                        "params": gnn_params,
                        "lr": lr_gnn,
                        "weight_decay": 1e-2,
                    },
                    {"params": les_params, "lr": lr_les, "weight_decay": 0},
                ]
                self.optimizer = nn.AdamWeightDecay(group_params)
            else:
                self.optimizer = nn.AdamWeightDecay(
                    les_params,
                    learning_rate=lr_les,
                    weight_decay=1e-3,
                )
        else:
            self.optimizer = AdamW(
                self.model.trainable_params(),
                lr=max_lr,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )
            self.scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                total_steps=int(total_steps * 1.02),
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=100.0,
                final_div_factor=1000.0,
                three_phase=False,
            )

        self.ema = ExponentialMovingAverage(
            self.model.trainable_params(), decay=0.999
        )
        self.finetune_mode_flag = finetune_mode
        self.max_lr = max_lr
        self.total_steps = total_steps

        self.huber_delta = 0.01
        self.w_e = 10.0
        self.w_f = 10.0
        self.w_s = 10.0
        self.train_log_path = os.path.join(
            self.checkpoint_dir, "train_log.csv"
        )
        self.val_log_path = os.path.join(
            self.checkpoint_dir, "val_log.csv"
        )
        self.EV_A3_TO_GPA = 160.21766

        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._init_loggers()

        self.grad_fn = ms.value_and_grad(
            self.step, None, self.optimizer.parameters, has_aux=True
        )
        self.parallel_mode = os.environ.get(
            "PARALLEL_MODE", "NONE"
        ).upper()
        if self.parallel_mode == "DATA_PARALLEL":
            self.grad_reducer = nn.DistributedGradReducer(
                self.optimizer.parameters,
                mean=True,
                degree=get_group_size(),
            )

    def _init_loggers(self):
        headers = [
            "epoch", "step", "lr", "total_loss", "loss_e", "loss_f",
            "loss_s", "mae_e", "mae_f", "mae_s_gpa",
        ]
        for path in [self.train_log_path, self.val_log_path]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def log_to_csv(self, mode, data):
        if self.rank != 0:
            return
        path = (
            self.train_log_path
            if mode == "train"
            else self.val_log_path
        )
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                data["epoch"], data["step"], f"{data['lr']}",
                f"{data['total_loss']}", f"{data['loss_e']}",
                f"{data['loss_f']}", f"{data['loss_s']}",
                f"{data['mae_e']*1000}", f"{data['mae_f']*1000}",
                f"{data['mae_s_gpa']}",
            ])

    def step(self, batch, train=True, batch_idx=0):
        """
        Single forward/backward step: energy and stress via strain derivative.

        Uses value_and_grad on a closure that applies virtual strain to pos/cell,
        runs the model, and returns energy. Forces and stress come from gradients.
        """
        if hasattr(batch, "num_graphs"):
            num_graphs = batch.num_graphs
        else:
            batch_max = (
                batch.batch
                if batch.batch is not None
                else ms.Tensor(1)
            )
            num_graphs = int(ops.reduce_max(batch_max)) + 1

        original_pos = batch.pos
        original_cell = getattr(batch, "cell", None)
        displacement = mint.zeros((num_graphs, 3, 3)).astype(ms.float32)

        def get_energy(pos, disp):
            symmetric_strain = 0.5 * (
                disp + ops.transpose(disp, (0, 2, 1))
            )
            strain_per_atom = symmetric_strain[batch.batch]
            pos_deformed = pos + mint.einsum(
                "ni,nij->nj", pos, strain_per_atom
            )
            batch.pos = pos_deformed
            if original_cell is not None and len(original_cell.shape) == 3:
                batch.cell = original_cell + ops.matmul(
                    original_cell, symmetric_strain
                )
            pred_e_inner = self.model(batch).view(-1)
            return pred_e_inner, original_cell

        grads_fn = ms.value_and_grad(
            get_energy, grad_position=(0, 1), has_aux=True
        )
        (pred_e, _), grads = grads_fn(original_pos, displacement)

        batch.pos = original_pos
        if original_cell is not None:
            batch.cell = original_cell

        pred_f = (
            -grads[0]
            if grads[0] is not None
            else mint.zeros_like(batch.pos)
        )
        dE_dStrain = grads[1]

        if dE_dStrain is not None:
            if original_cell is not None:
                vol = mint.abs(
                    mint.exp(
                        ops.logdet(original_cell).astype(ms.float32)
                    )
                ).view(-1, 1, 1)
            else:
                vol = mint.ones((num_graphs, 1, 1), ms.float32)
            pred_stress = dE_dStrain / vol
        else:
            pred_stress = mint.zeros(
                (num_graphs, 3, 3)
            ).astype(ms.float32)

        target_e = batch.y.view(-1)
        if (
            not hasattr(self, "_ones_buffer")
            or self._ones_buffer.shape[0] != batch.batch.shape[0]
        ):
            self._ones_buffer = mint.ones(
                batch.batch.shape
            ).astype(ms.float32)
        num_atoms = scatter_add(
            self._ones_buffer,
            batch.batch,
            dim=0,
            dim_size=num_graphs,
        ).view(-1).clamp(min=1)

        loss_e = ops.huber_loss(
            pred_e / num_atoms,
            target_e / num_atoms,
            delta=self.huber_delta,
        )
        loss_f = conditional_huber_loss(
            pred_f, batch.force, base_delta=self.huber_delta
        )
        loss_s = Tensor(0.0, ms.float32)
        stress_mask_sum = 0

        if hasattr(batch, "stress") and batch.stress is not None:
            stress_norm = mint.norm(
                batch.stress.view(num_graphs, -1), dim=1
            )
            stress_mask = stress_norm > 1e-6
            stress_mask_sum = stress_mask.sum().item()
            if stress_mask_sum > 0:
                s_pred = pred_stress.view(num_graphs, -1)[stress_mask]
                s_target = batch.stress.view(num_graphs, -1)[stress_mask]
                loss_s = ops.huber_loss(
                    s_pred, s_target, delta=self.huber_delta
                )

        total_loss = (
            self.w_e * loss_e + self.w_f * loss_f + self.w_s * loss_s
        )

        with ms._no_grad():
            mae_e = ops.reduce_mean(
                ops.abs(pred_e / num_atoms - target_e / num_atoms)
            ).item()
            mae_f = ops.reduce_mean(
                ops.abs(pred_f - batch.force)
            ).item()
            mae_s_gpa = 0.0

        if stress_mask_sum > 0:
            mae_s_val = ops.reduce_mean(
                ops.abs(
                    pred_stress.view(num_graphs, -1)[stress_mask]
                    - batch.stress.view(num_graphs, -1)[stress_mask]
                )
            )
            mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA

        return total_loss, {
            "total_loss": total_loss.asnumpy().item(),
            "loss_e": loss_e.asnumpy().item(),
            "loss_f": loss_f.asnumpy().item(),
            "loss_s": loss_s.asnumpy().item(),
            "mae_e": mae_e,
            "mae_f": mae_f,
            "mae_s_gpa": mae_s_gpa,
        }

    def train_epoch(self, loader, epoch_idx):
        self.model.set_train(True)
        pbar = tqdm(
            loader,
            desc=f"Train Ep {epoch_idx}",
            leave=False,
            disable=(self.rank != 0),
        )
        metrics_sum = {
            "mae_e": 0,
            "mae_f": 0,
            "mae_s_gpa": 0,
            "total_loss": 0,
        }
        count = 0

        model_for_cfg = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        max_steps = getattr(
            model_for_cfg.cfg, "steps_per_epoch", None
        )
        if self.finetune_mode and max_steps is None:
            max_steps = 500

        for i, batch in enumerate(pbar):
            if i == 0 and self.rank == 0:
                print("First batch graph info:")
                print("  Number of graphs in batch:", batch.num_graphs)
                print("  Nodes (atoms) in batch:", batch.pos.shape[0])
                if hasattr(batch, "stress") and batch.stress is not None:
                    print("  Stress tensor shape:", batch.stress.shape)
                else:
                    print("  No stress tensor in this batch.")

            (loss, metrics), grads = self.grad_fn(
                batch, train=True, batch_idx=i
            )
            if self.parallel_mode == "DATA_PARALLEL":
                grads = self.grad_reducer(grads)
            grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
            self.optimizer(grads)

            if i % 5 == 0:
                self.ema.update()

            self.global_step += 1
            log_data = metrics.copy()
            if self.finetune_mode:
                groups = self.optimizer.param_groups
                lr = (
                    groups[1]["lr"].item()
                    if len(groups) > 1
                    else groups[0]["lr"].item()
                )
                log_data.update(
                    {"epoch": epoch_idx, "step": i, "lr": lr}
                )
            else:
                log_data.update(
                    {
                        "epoch": epoch_idx,
                        "step": i,
                        "lr": self.optimizer.lrs[0].item(),
                    }
                )

            self.log_to_csv("train", log_data)
            if not self.finetune_mode:
                self.scheduler.step()

            for k in metrics_sum:
                metrics_sum[k] += metrics[k]
            count += 1
            pbar.set_postfix({
                "L": f"{metrics['total_loss']:.4f}",
                "MAE_e": f"{metrics['mae_e']*1000:.1f}",
                "MAE_F": f"{metrics['mae_f']*1000:.1f}",
            })

            if max_steps is not None and (i + 1) >= max_steps:
                if self.rank == 0:
                    print(
                        f"  Virtual Epoch Reached ({max_steps} steps). "
                        "Stopping for Validation..."
                    )
                break

        return {k: v / count for k, v in metrics_sum.items()}

    def validate(self, loader, epoch_idx):
        """Run validation loop using EMA parameters."""
        self.model.set_train(False)
        pbar = tqdm(
            loader,
            desc=f"Val Ep {epoch_idx}",
            leave=False,
            disable=(self.rank != 0),
        )
        metrics_sum = {
            "mae_e": 0,
            "mae_f": 0,
            "mae_s_gpa": 0,
            "total_loss": 0,
        }
        count = 0

        model_for_cfg = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        max_steps = getattr(
            model_for_cfg.cfg, "steps_per_epoch", None
        )
        if self.finetune_mode and max_steps is None:
            max_steps = 500

        with self.ema.average_parameters():
            for i, batch in enumerate(pbar):
                metrics = self.step(batch, train=False)[1]
                log_data = metrics.copy()
                if self.finetune_mode:
                    groups = self.optimizer.param_groups
                    lr = (
                        groups[1]["lr"]
                        if len(groups) > 1
                        else groups[0]["lr"]
                    )
                    log_data.update(
                        {"epoch": epoch_idx, "step": i, "lr": lr}
                    )
                else:
                    log_data.update({
                        "epoch": epoch_idx,
                        "step": i,
                        "lr": self.optimizer.lrs[0].item(),
                    })

                self.log_to_csv("val", log_data)
                for k in metrics_sum:
                    metrics_sum[k] += metrics[k]
                count += 1
                pbar.set_postfix({
                    "L": f"{metrics['total_loss']:.4f}",
                    "MAE_e": f"{metrics['mae_e']*1000:.1f}",
                    "MAE_F": f"{metrics['mae_f']*1000:.1f}",
                })
                if max_steps is not None and (i + 1) >= max_steps:
                    if self.rank == 0:
                        print(
                            f"  Virtual Epoch Reached ({max_steps} steps). "
                            "Stopping validation."
                        )
                    break

        if count == 0:
            count = 1
        return {k: v / count for k, v in metrics_sum.items()}

    def step_scheduler_on_val(self, val_loss):
        """Step LR scheduler on validation loss (e.g. ReduceLROnPlateau)."""
        if self.finetune_mode:
            self.scheduler.step(val_loss)

    def save(self, filename="best_model.pt", rank=0):
        """Save model checkpoint using EMA parameters."""
        path = os.path.join(self.checkpoint_dir, filename)
        raw_model = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        with self.ema.average_parameters():
            save_checkpoint(raw_model, path)
        if self.rank == 0:
            print(f"Model saved to {path}")

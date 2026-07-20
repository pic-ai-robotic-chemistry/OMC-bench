import argparse
import importlib
import importlib.util
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.engine.train_config import RunConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified training entrypoint.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a Python config file that defines CONFIG.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Floating-point precision for Python/Torch model code.",
    )
    return parser.parse_args(argv)


def resolve_default_config_path() -> Path:
    return Path("configs/train/latest.py")


def load_run_config(config_path: str | None) -> tuple[RunConfig, Path]:
    resolved_path = resolve_default_config_path() if config_path is None else Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("train_config_module", resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config from {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, RunConfig):
        raise TypeError(f"{resolved_path} must define CONFIG = RunConfig(...)")
    return config, resolved_path


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return torch.float64 if dtype_name == "float64" else torch.float32


def apply_precision_globals(dtype: torch.dtype):
    utils_pkg = importlib.import_module("src.utils")
    utils_mod = importlib.import_module("src.utils.Utils")

    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    utils_pkg.DEFAULT_FLOAT_DTYPE = dtype
    utils_pkg.DEFAULT_NP_FLOAT_DTYPE = np_dtype
    utils_mod.DEFAULT_FLOAT_DTYPE = dtype
    utils_mod.DEFAULT_NP_FLOAT_DTYPE = np_dtype


def init_global_runtime(dtype: torch.dtype):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_default_dtype(dtype)
    torch.backends.cuda.matmul.allow_tf32 = dtype == torch.float32
    torch.backends.cudnn.allow_tf32 = dtype == torch.float32
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def init_distributed_mode(config: RunConfig):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"[startup] rank={rank} local_rank={local_rank} world_size={world_size} initializing process group", flush=True)
        torch.cuda.set_device(local_rank)

        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "world_size": world_size,
            "rank": rank,
        }
        if config.distributed.init_timeout_minutes is not None:
            init_kwargs["timeout"] = timedelta(minutes=config.distributed.init_timeout_minutes)

        dist.init_process_group(**init_kwargs)
        print(f"[startup] rank={rank} process group initialized", flush=True)
        return local_rank, rank, world_size

    print("Warning: Running in single-GPU mode")
    return 0, 0, 1


def log_info(message, rank):
    if rank == 0:
        print(message)


def import_runtime_components(dtype: torch.dtype):
    data_module = importlib.import_module("src.data")
    utils_module = importlib.import_module("src.utils")
    model_module = importlib.import_module("src.models")
    trainer_module = importlib.import_module("src.engine.pipelines.trainer")
    dataset_class = data_module.ChunkedSmartDataset
    model_class = model_module.AtomBitModel
    trainer_class = trainer_module.PotentialTrainer

    sampler_registry = {
        "bin_packing": data_module.BinPackingSampler,
        "batch": data_module.BatchSampler,
        "normal": data_module.NormalSampler,
    }
    return {
        "Dataset": dataset_class,
        "ModelConfig": utils_module.AtomBitConfig,
        "Model": model_class,
        "Trainer": trainer_class,
        "sampler_registry": sampler_registry,
    }


def build_dataloader(
    config: RunConfig,
    components: dict,
    meta_file: str,
    rank: int,
    world_size: int,
    is_train: bool,
    dtype: torch.dtype,
):
    from torch_geometric.loader import DataLoader

    full_path = os.path.join(config.paths.data_dir, meta_file)
    if not os.path.exists(full_path):
        if is_train:
            raise FileNotFoundError(f"Missing metadata file: {meta_file}")
        log_info(f"Warning: {meta_file} not found, skipping.", rank)
        return None, None

    dataset_kwargs = {
        "data_dir": config.paths.data_dir,
        "metadata_file": meta_file,
        "rank": rank,
        "world_size": world_size,
        "cast_float_dtype": dtype,
    }

    dataset = components["Dataset"](**dataset_kwargs)

    sampler_cls = components["sampler_registry"][config.loader.sampler]
    sampler_kwargs = {
        "max_cost": config.training.max_cost_per_batch,
        "edge_weight": "auto",
        "shuffle": is_train,
        "world_size": world_size,
        "rank": rank,
    }
    if config.distributed.seed is not None:
        sampler_kwargs["seed"] = config.distributed.seed
    sampler = sampler_cls(dataset.metadata, **sampler_kwargs)

    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": config.loader.num_workers,
        "pin_memory": config.loader.pin_memory,
    }
    if config.loader.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.loader.prefetch_factor

    return DataLoader(dataset, **loader_kwargs), sampler


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def enforce_precision_compatible_ops(model_config, dtype: torch.dtype, rank: int):
    if dtype != torch.float64:
        return

    changed = []
    for field_name in ("gating_impl", "mat_mul_sym_impl", "outer_impl"):
        if getattr(model_config, field_name, 1) != 1:
            setattr(model_config, field_name, 1)
            changed.append(field_name)

    if changed and rank == 0:
        joined = ", ".join(changed)
        print(f"float64 mode forces pure PyTorch ops. Reset fields to 1: {joined}")


def build_model(
    config: RunConfig,
    components: dict,
    device,
    rank: int,
    model_config,
    dtype: torch.dtype,
    state_dict=None,
    strict_load=True,
    load_e0=True,
):
    enforce_precision_compatible_ops(model_config, dtype=dtype, rank=rank)
    model = components["Model"](model_config).to(device=device, dtype=dtype)

    if state_dict is not None:
        if rank == 0:
            log_info("Loading state_dict from checkpoint...", rank)
        model.load_state_dict(clean_state_dict(state_dict), strict=strict_load)
        model.atomic_ref.weight.requires_grad = False
    elif load_e0:
        if os.path.exists(config.paths.e0_path):
            meta_data = torch.load(config.paths.e0_path, map_location="cpu", weights_only=False)
            model.load_external_e0(meta_data.get("e0_dict", None))
            model.atomic_ref.weight.requires_grad = False
            if rank == 0:
                log_info(f"Injected E0 from {config.paths.e0_path}", rank)
        else:
            model.atomic_ref.weight.data = model.atomic_ref.weight.data.to(dtype=dtype)
            log_info("E0 file not found, skipping injection.", rank)

    if rank == 0:
        param_count = sum(parameter.numel() for parameter in model.parameters())
        log_info(f"Model parameters: {param_count:,}", rank)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
    return model


def resolve_training_state(config: RunConfig, components: dict, device, rank: int, train_sampler, train_total_steps: int, dtype: torch.dtype):
    model_config = components["ModelConfig"](**config.model_params)
    start_epoch = 0
    checkpoint = None
    checkpoint_state = None
    resume_step = -1
    strict_load = True
    is_finetuning = config.training.finetune_mode

    if config.restart.enabled:
        checkpoint_path = config.restart.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("Restart is enabled but no checkpoint_path was provided.")

        log_info(f"Resuming from {checkpoint_path}...", rank)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_epoch = checkpoint.get("epoch", 2)

        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
            log_info(f"Loaded config from checkpoint (avg_neigh={model_config.avg_neighborhood:.2f})", rank)
        else:
            log_info("No config in checkpoint, using default derived from data.", rank)
            model_config.avg_neighborhood = 1.0 / train_sampler.edge_weight

        checkpoint_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        steps_per_epoch_est = train_total_steps // config.training.epochs
        resume_step = start_epoch * steps_per_epoch_est - 1
        log_info(f"Resuming OneCycleLR from step: {resume_step}", rank)
    elif config.finetune.enabled:
        checkpoint_path = config.finetune.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("Finetune is enabled but no checkpoint_path was provided.")

        log_info(f"Starting fine-tuning from {checkpoint_path}...", rank)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        strict_load = config.finetune.strict_load

        if "model_config" in checkpoint and getattr(checkpoint["model_config"], "avg_neighborhood", None) is not None:
            model_config = checkpoint["model_config"]
        model_config.avg_neighborhood = getattr(model_config, "avg_neighborhood", 1.0 / train_sampler.edge_weight)
        resume_step = -1
        is_finetuning = True
    else:
        log_info("Starting new training...", rank)
        model_config.avg_neighborhood = 1.0 / train_sampler.edge_weight

    enforce_precision_compatible_ops(model_config, dtype=dtype, rank=rank)
    return {
        "checkpoint": checkpoint,
        "checkpoint_state": checkpoint_state,
        "is_finetuning": is_finetuning,
        "model_config": model_config,
        "resume_step": resume_step,
        "start_epoch": start_epoch,
        "strict_load": strict_load,
    }


def save_epoch_checkpoint(config: RunConfig, epoch: int, model, model_config, trainer, config_path: Path, dtype: torch.dtype):
    save_dict = {
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "ema_state_dict": trainer.ema.state_dict(),
        "run_config_path": str(config_path),
        "run_config": config.to_dict(),
        "precision_dtype": str(dtype),
        "label_mode": "residual",
    }
    checkpoint_name = config.training.checkpoint_name_template.format(epoch=epoch)
    torch.save(save_dict, os.path.join(config.paths.log_dir, checkpoint_name))


def run_training(config: RunConfig, config_path: Path, dtype: torch.dtype):
    init_global_runtime(dtype)
    apply_precision_globals(dtype)
    print(f"[startup] loading runtime components for residual training, dtype={dtype}", flush=True)
    components = import_runtime_components(dtype=dtype)
    print("[startup] runtime components loaded", flush=True)

    local_rank, rank, world_size = init_distributed_mode(config)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(config.paths.log_dir, exist_ok=True)
        log_info(f"Start: {config.name} | label_mode=residual | dtype={dtype} | world_size={world_size} | device={device}", rank)
        log_info(f"Config: {config_path}", rank)
        if config.distributed.seed is not None:
            log_info(f"Seed: {config.distributed.seed}", rank)

    train_loader, train_sampler = build_dataloader(
        config,
        components,
        config.paths.train_meta,
        rank,
        world_size,
        is_train=True,
        dtype=dtype,
    )
    log_info("Train dataloader ready", rank)
    test_loader, _ = build_dataloader(
        config,
        components,
        config.paths.test_meta,
        rank,
        world_size,
        is_train=False,
        dtype=dtype,
    )
    log_info("Validation dataloader ready", rank)

    if os.environ.get("ATOMBIT_FAST_STEP_ESTIMATE", "0") == "1":
        steps_per_epoch_est = len(train_sampler)
        train_total_steps = steps_per_epoch_est * config.training.epochs
        log_info(
            f"Total scheduled steps: {train_total_steps} "
            f"(estimated: {steps_per_epoch_est} steps/epoch x {config.training.epochs})",
            rank,
        )
    else:
        train_total_steps = train_sampler.precompute_total_steps(config.training.epochs)
        log_info(f"Total scheduled steps: {train_total_steps}", rank)
    state = resolve_training_state(config, components, device, rank, train_sampler, train_total_steps, dtype=dtype)
    log_info("Training state resolved", rank)

    model = build_model(
        config=config,
        components=components,
        device=device,
        rank=rank,
        model_config=state["model_config"],
        dtype=dtype,
        state_dict=state["checkpoint_state"],
        strict_load=state["strict_load"],
        load_e0=state["checkpoint_state"] is None,
    )
    log_info("Model built", rank)

    trainer = components["Trainer"](
        model,
        total_steps=train_total_steps,
        max_lr=config.training.lr,
        device=device,
        checkpoint_dir=config.paths.log_dir,
        finetune_mode=state["is_finetuning"],
        lr_gnn=config.training.lr_gnn,
        last_epoch=state["resume_step"],
        use_direct_force=config.training.use_direct_force,
        huber_delta=config.training.huber_delta,
        energy_loss_weight=config.training.energy_loss_weight,
        force_loss_weight=config.training.force_loss_weight,
        stress_loss_weight=config.training.stress_loss_weight,
    )
    log_info("Trainer built", rank)

    if config.restart.enabled and state["checkpoint"] is not None:
        log_info("Restoring optimizer, scheduler and EMA states...", rank)
        trainer.load_checkpoint(state["checkpoint"])

    for epoch in range(state["start_epoch"] + 1, config.training.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch)

        if test_loader:
            val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
            if config.training.step_scheduler_on_val:
                trainer.step_scheduler_on_val(val_metrics["total_loss"])
        else:
            val_metrics = {"total_loss": 0.0, "mae_f": 0.0}

        if rank == 0:
            print(
                f"Ep {epoch:03d} | "
                f"T_Loss: {train_metrics['total_loss']:.4f} | "
                f"V_Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE_F: {train_metrics['mae_f'] * 1000:.1f}/{val_metrics['mae_f'] * 1000:.1f} meV/A"
            )
            save_epoch_checkpoint(config, epoch, model, state["model_config"], trainer, config_path, dtype)

        if dist.is_initialized():
            dist.barrier()

    log_info("Training finished.", rank)
    if dist.is_initialized():
        dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    dtype = resolve_dtype(args.dtype)
    config, config_path = load_run_config(args.config)
    run_training(config, config_path, dtype=dtype)

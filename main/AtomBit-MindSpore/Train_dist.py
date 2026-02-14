"""
Distributed training script for HTGP: data loaders, model build, E0 load,
and train/validate loop with MindSpore (single or data-parallel).
"""

import os
import pickle

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from sharker.loader.dataloader import Dataloader

from src.data import BinPackingSampler, ChunkedSmartDataset_h5
from src.engine import PotentialTrainer
from src.models import HTGPModel
from src.utils import HTGPConfig


class Config:
    """Training and data paths; model and loader settings."""

    # Paths
    DATA_DIR = "/home/hxy/chendanyang/UMA/OMC_dataset_h5"
    TRAIN_META = "OMC_train_metadata.pkl"
    TEST_META = "OMC_test_metadata.pkl"
    E0_PATH = "/home/hxy/chendanyang/UMA/OMC_dataset_h5/meta_data.pt"
    LOG_DIR = "Checkpoints"

    # Training
    MAX_COST_PER_BATCH = 5000
    LR = 1e-3
    EPOCHS = 10

    # Loader
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 2

    # Model (HTGP)
    MODEL_PARAMS = dict(
        hidden_dim=128,
        num_layers=2,
        cutoff=6.0,
        num_rbf=10,
        use_L0=True,
        use_L1=True,
        use_L2=True,
        use_gating=True,
        use_long_range=False,
    )


def init_distributed_mode():
    """
    Initialize data-parallel context if PARALLEL_MODE=DATA_PARALLEL.

    Returns:
        (rank, world_size); (0, 1) if not distributed.
    """
    parallel_mode = os.environ.get("PARALLEL_MODE", "NONE").upper()
    if parallel_mode == "DATA_PARALLEL":
        init()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )
        return get_rank(), get_group_size()
    print("Warning: Running in single-device mode.")
    return 0, 1


def log_info(msg, rank):
    """Print message only on rank 0."""
    if rank == 0:
        print(msg)


def get_dataloader(data_dir, meta_file, rank, world_size, is_train=True):
    """
    Build ChunkedSmartDataset_h5, BinPackingSampler, and Dataloader.

    Returns:
        (loader, sampler). Sampler is used for set_epoch and precompute_total_steps.
    """
    full_path = os.path.join(data_dir, meta_file)
    if not os.path.exists(full_path):
        if is_train:
            raise FileNotFoundError(
                f"Metadata not found: {meta_file}. Run preprocess first."
            )
        log_info(f"Warning: {meta_file} not found, skipping.", rank)

    dataset = ChunkedSmartDataset_h5(
        data_dir,
        metadata_file=meta_file,
        rank=rank,
        world_size=world_size,
    )
    sampler = BinPackingSampler(
        dataset.metadata,
        max_cost=Config.MAX_COST_PER_BATCH,
        edge_weight="auto",
        shuffle=is_train,
        world_size=world_size,
        rank=rank,
    )
    loader = Dataloader(
        dataset,
        sampler=sampler,
        num_parallel_workers=Config.NUM_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )
    return loader, sampler


def build_model(rank, avg_neighborhood, **kwargs):
    """
    Build HTGPModel, optionally load checkpoint and E0.

    kwargs may include: restart (bool), state_dict (for restart), etc.
    """
    model_config = HTGPConfig(**Config.MODEL_PARAMS)
    model_config.avg_neighborhood = avg_neighborhood
    model = HTGPModel(model_config)

    if kwargs.get("restart"):
        state_dict = kwargs["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        param_not_load, _ = ms.load_param_into_net(model, new_state_dict)
        if param_not_load:
            print(f"Warning: {param_not_load} parameters not loaded.")
        else:
            print("All parameters loaded successfully.")

    if rank == 0:
        param_count = sum(p.numel() for p in model.get_parameters())
        log_info(f"Model parameters: {param_count:,}", rank)

    if not kwargs.get("restart"):
        e0_path = Config.E0_PATH
        if os.path.exists(e0_path):
            with open(e0_path, "rb") as f:
                meta_data = pickle.load(f)
                e0_dict = meta_data.get("e0_dict", None)
            if e0_dict:
                model.load_external_e0(
                    e0_dict, verbose=(rank == 0), rank=rank
                )
        model.atomic_ref.embedding_table.requires_grad = False

    return model


def main():
    rank, world_size = init_distributed_mode()

    if rank == 0:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        log_info(f"\n[Start] World size: {world_size}", rank)
        log_info("=" * 60, rank)

    log_info("\n[1/4] Initializing DataLoaders...", rank)
    train_loader, train_sampler = get_dataloader(
        Config.DATA_DIR,
        Config.TRAIN_META,
        rank,
        world_size,
        is_train=True,
    )
    test_loader, test_sampler = get_dataloader(
        Config.DATA_DIR,
        Config.TEST_META,
        rank,
        world_size,
        is_train=False,
    )

    log_info("\n[2/4] Building Model...", rank)
    avg_neighborhood = 1 / train_sampler.edge_weight
    restart = False

    if not restart:
        model = build_model(rank, avg_neighborhood)
    else:
        checkpoint_path = "Checkpoints/model_epoch_2.ckpt"
        checkpoint_weights = ms.load_checkpoint(checkpoint_path)
        model = build_model(
            rank,
            avg_neighborhood,
            restart=True,
            state_dict=checkpoint_weights,
        )

    log_info("\n[3/4] Initializing Trainer...", rank)
    train_total_steps = train_sampler.precompute_total_steps(Config.EPOCHS)
    log_info(
        f"Estimated total training steps: {train_total_steps}",
        rank,
    )
    if test_sampler is not None:
        test_total_steps = test_sampler.precompute_total_steps(Config.EPOCHS)
        log_info(
            f"Estimated total test steps: {test_total_steps}",
            rank,
        )

    trainer = PotentialTrainer(
        model,
        total_steps=train_total_steps,
        max_lr=Config.LR,
        checkpoint_dir=Config.LOG_DIR,
    )

    log_info("\n[4/4] Starting training loop...", rank)
    log_info("=" * 60, rank)

    for epoch in range(1, Config.EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_epoch(
            train_loader, epoch_idx=epoch
        )
        if test_loader:
            val_metrics = trainer.validate(
                test_loader, epoch_idx=epoch
            )
        else:
            val_metrics = {"total_loss": 0.0, "mae_f": 0.0}

        if rank == 0:
            log_msg = (
                f"Ep {epoch:03d} | "
                f"T_Loss: {train_metrics['total_loss']:.4f} | "
                f"V_Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE_F: {train_metrics['mae_f']*1000:.1f}/"
                f"{val_metrics['mae_f']*1000:.1f} meV/A"
            )
            print(log_msg)
            trainer.save(f"model_epoch_{epoch}.ckpt")

    log_info("\nTraining finished.", rank)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    main()

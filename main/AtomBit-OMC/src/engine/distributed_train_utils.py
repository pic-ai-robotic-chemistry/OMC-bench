import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

from src.data import BinPackingSampler, ChunkedSmartDataset_h5
from src.models import AtomBitModel


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
        return local_rank, rank, world_size

    print("⚠️ Warning: Running in Single GPU Mode")
    return 0, 0, 1


def log_info(message, rank):
    if rank == 0:
        print(message)


def get_dataloader(config, meta_file, rank, world_size, is_train=True):
    full_path = os.path.join(config.DATA_DIR, meta_file)
    if not os.path.exists(full_path):
        if is_train:
            raise FileNotFoundError(f"❌ Error: {meta_file} not found!")
        log_info(f"⚠️ Warning: {meta_file} not found, skipping...", rank)
        return None, None

    dataset = ChunkedSmartDataset_h5(
        config.DATA_DIR,
        metadata_file=meta_file,
        rank=rank,
        world_size=world_size,
    )
    sampler = BinPackingSampler(
        dataset.metadata,
        max_cost=config.MAX_COST_PER_BATCH,
        edge_weight="auto",
        shuffle=is_train,
        world_size=world_size,
        rank=rank,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
    )
    return loader, sampler


def build_model(device, rank, model_config, e0_path, state_dict=None):
    model = AtomBitModel(model_config).to(device)

    if state_dict is not None:
        if rank == 0:
            log_info("📥 Loading state_dict from checkpoint...", rank)

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_state_dict[key[7:] if key.startswith("module.") else key] = value

        model.load_state_dict(cleaned_state_dict, strict=False)
    else:
        if os.path.exists(e0_path):
            meta_data = torch.load(e0_path, map_location="cpu", weights_only=False)
            e0_dict = meta_data.get("e0_dict", None)
            model.load_external_e0(e0_dict)
            model.atomic_ref.weight.requires_grad = False
            if rank == 0:
                log_info(f"✨ Injected E0 from {e0_path}", rank)
        else:
            model.atomic_ref.weight = model.atomic_ref.weight.float()
            log_info("⚠️ E0 file not found, skipping injection.", rank)

    if rank == 0:
        param_count = sum(parameter.numel() for parameter in model.parameters())
        log_info(f"🧠 Model Parameters: {param_count:,}", rank)

    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )

    return model

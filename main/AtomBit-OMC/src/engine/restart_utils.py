import torch


def resolve_restart_state(restart_enabled, checkpoint_path, device, default_model_config, train_sampler, log_info, rank, epochs):
    start_epoch = 0
    checkpoint = None
    checkpoint_state = None
    model_config = default_model_config
    resume_step = -1

    if not restart_enabled:
        log_info("\n🆕 Starting New Training...", rank)
        model_config.avg_neighborhood = 1.0 / train_sampler.edge_weight
        return {
            "start_epoch": start_epoch,
            "checkpoint": checkpoint,
            "checkpoint_state": checkpoint_state,
            "model_config": model_config,
            "resume_step": resume_step,
        }

    log_info(f"\n🔄 Resuming from {checkpoint_path}...", rank)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    start_epoch = checkpoint.get("epoch", 2)

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        log_info(
            f"   Loaded config from checkpoint (avg_neigh={model_config.avg_neighborhood:.2f})",
            rank,
        )
    else:
        log_info("⚠️ No config in checkpoint, using default derived from data.", rank)
        model_config.avg_neighborhood = 1.0 / train_sampler.edge_weight

    checkpoint_state = checkpoint.get("model_state_dict", checkpoint)
    steps_per_epoch_est = train_sampler.precompute_total_steps(epochs) // epochs
    resume_step = start_epoch * steps_per_epoch_est - 1
    log_info(f"   Resuming OneCycleLR from step: {resume_step}", rank)

    return {
        "start_epoch": start_epoch,
        "checkpoint": checkpoint,
        "checkpoint_state": checkpoint_state,
        "model_config": model_config,
        "resume_step": resume_step,
    }

import torch


def conditional_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base_delta: float = 0.01,
) -> torch.Tensor:
    """Adaptive Huber loss used by the original trainer."""
    force_norm = torch.norm(target, dim=1, keepdim=True)

    delta_scale = torch.ones_like(force_norm)
    delta_scale[(force_norm >= 100) & (force_norm < 200)] = 0.7
    delta_scale[(force_norm >= 200) & (force_norm < 300)] = 0.4
    delta_scale[force_norm >= 300] = 0.1

    adaptive_delta = base_delta * delta_scale
    error = pred - target
    abs_error = torch.abs(error)
    is_mse = abs_error < adaptive_delta

    loss_mse = 0.5 * error**2
    loss_l1 = adaptive_delta * (abs_error - 0.5 * adaptive_delta)
    return torch.where(is_mse, loss_mse, loss_l1).mean()


def check_finite(name: str, value: torch.Tensor):
    if value is None:
        return
    if not torch.isfinite(value).all():
        print(f"!!! NON-FINITE in {name}")
        print(
            "min:", torch.nan_to_num(value).min().item(),
            "max:", torch.nan_to_num(value).max().item(),
        )
        raise FloatingPointError(name)

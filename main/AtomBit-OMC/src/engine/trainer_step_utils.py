import torch

from src.utils import DEFAULT_FLOAT_DTYPE, scatter_add


def get_num_graphs(batch) -> int:
    if hasattr(batch, "num_graphs"):
        return batch.num_graphs
    return int(batch.batch.max()) + 1


def as_batched_cell(cell, num_graphs: int):
    if cell is None:
        return None
    if cell.dim() == 3:
        return cell
    if cell.dim() == 2:
        if cell.shape == (3, 3):
            return cell.unsqueeze(0)
        if cell.shape[1] == 3 and cell.shape[0] == num_graphs * 3:
            return cell.view(num_graphs, 3, 3)
    raise ValueError(f"Unsupported cell shape {tuple(cell.shape)} for num_graphs={num_graphs}")


def build_symmetric_strain(batch, device):
    num_graphs = get_num_graphs(batch)
    displacement = torch.zeros((num_graphs, 3, 3), dtype=batch.pos.dtype, device=device)
    displacement.requires_grad_(True)
    symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
    return num_graphs, displacement, symmetric_strain


def apply_batch_deformation(batch, symmetric_strain):
    strain_per_atom = symmetric_strain[batch.batch]
    pos_deformed = batch.pos + torch.einsum("ni,nij->nj", batch.pos, strain_per_atom)

    original_pos = batch.pos
    original_cell = getattr(batch, "cell", None)
    num_graphs = get_num_graphs(batch)

    batch.pos = pos_deformed
    if original_cell is not None:
        batched_cell = as_batched_cell(original_cell, num_graphs)
        batch.cell = batched_cell + torch.bmm(batched_cell, symmetric_strain)

    return original_pos, original_cell


def restore_batch_geometry(batch, original_pos, original_cell):
    batch.pos = original_pos
    if original_cell is not None:
        batch.cell = original_cell


def cached_num_atoms(trainer, batch, num_graphs):
    if not hasattr(trainer, "_ones_buffer") or trainer._ones_buffer.shape[0] != batch.batch.shape[0]:
        trainer._ones_buffer = torch.ones_like(batch.batch, dtype=DEFAULT_FLOAT_DTYPE)

    return scatter_add(
        trainer._ones_buffer,
        batch.batch,
        dim=0,
        dim_size=num_graphs,
    ).view(-1).clamp(min=1)


def current_lr(optimizer, finetune_mode: bool) -> float:
    if finetune_mode and len(optimizer.param_groups) > 1:
        return optimizer.param_groups[1]["lr"]
    return optimizer.param_groups[0]["lr"]


def resolve_max_steps(model, finetune_mode: bool):
    cfg = model.module.cfg if hasattr(model, "module") else model.cfg
    max_steps = getattr(cfg, "steps_per_epoch", None)
    if finetune_mode and max_steps is None:
        return 500
    return max_steps


def average_metrics(metrics_sum: dict, count: int) -> dict:
    if count == 0:
        count = 1
    return {key: value / count for key, value in metrics_sum.items()}

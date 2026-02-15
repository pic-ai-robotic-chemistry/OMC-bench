"""
Learning rate schedulers for MindSpore.

Provides OneCycleLR (warm-up + cosine/linear decay), stepped per batch/iteration.
"""

import math
from typing import List, Optional, Sequence, Union

import mindspore as ms
from mindspore.experimental import optim


def _to_list(x, n: int, name: str) -> List[float]:
    """Broadcast scalar to length n or validate list/tuple length."""
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(
                f"{name} length must match optimizer.param_groups ({n}), "
                f"but got {len(x)}."
            )
        return [float(v) for v in x]
    return [float(x) for _ in range(n)]


def _anneal_linear(start: float, end: float, pct: float) -> float:
    """Linear interpolation from start to end as pct goes 0 -> 1."""
    return start + (end - start) * pct


def _anneal_cos(start: float, end: float, pct: float) -> float:
    """Cosine annealing from start to end as pct goes 0 -> 1."""
    cos_out = (1.0 + math.cos(math.pi * pct)) / 2.0
    return end + (start - end) * cos_out


class OneCycleLR(optim.lr_scheduler.LRScheduler):
    """
    One-cycle learning rate policy for MindSpore (experimental).

    Step this scheduler every iteration/batch. LR schedule:
    - initial_lr = max_lr / div_factor
    - Warm-up (pct_start of steps): initial_lr -> max_lr
    - Anneal: max_lr -> min_lr, where min_lr = max_lr / (div_factor * final_div_factor)

    With three_phase=True, the anneal phase is split: max_lr -> initial_lr -> min_lr.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: Union[float, Sequence[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: MindSpore experimental optimizer with param_groups.
            max_lr: Maximum LR (float or list per param_group).
            total_steps: Total number of step() calls in the cycle.
            pct_start: Fraction of total_steps used for warm-up (0 < pct_start < 1).
            anneal_strategy: "cos" or "linear" for decay.
            div_factor: initial_lr = max_lr / div_factor.
            final_div_factor: min_lr = initial_lr / final_div_factor.
            three_phase: If True, anneal in two segments (max->initial->min).
            last_epoch: Initial step index (-1 = start from 0).
        """
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError(
                f"total_steps must be a positive int, but got {total_steps}."
            )
        if not (0.0 < float(pct_start) < 1.0):
            raise ValueError(
                f"pct_start must be in (0, 1), but got {pct_start}."
            )
        if div_factor <= 0 or final_div_factor <= 0:
            raise ValueError(
                "div_factor and final_div_factor must be > 0."
            )
        anneal_strategy = str(anneal_strategy).lower()
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(
                "anneal_strategy must be 'cos' or 'linear'."
            )

        self.total_steps = total_steps
        self.pct_start = float(pct_start)
        self.anneal_strategy = anneal_strategy
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.three_phase = bool(three_phase)

        n_groups = len(optimizer.param_groups)
        self.max_lrs = _to_list(max_lr, n_groups, "max_lr")
        self.initial_lrs = [lr / self.div_factor for lr in self.max_lrs]
        self.min_lrs = [
            lr / self.final_div_factor for lr in self.initial_lrs
        ]

        for g, init_lr in zip(optimizer.param_groups, self.initial_lrs):
            g["lr"] = ms.Parameter(init_lr)

        up_steps = int(self.total_steps * self.pct_start)
        up_steps = max(1, up_steps)

        if self.three_phase:
            rem = self.total_steps - up_steps
            down_steps = max(1, rem // 2)
            annihilate_steps = max(
                1, self.total_steps - up_steps - down_steps
            )
            self._phase_ends = (
                up_steps,
                up_steps + down_steps,
                up_steps + down_steps + annihilate_steps,
            )
        else:
            down_steps = max(1, self.total_steps - up_steps)
            self._phase_ends = (up_steps, up_steps + down_steps)

        super().__init__(optimizer, last_epoch)

    def _anneal(self, start: float, end: float, pct: float) -> float:
        pct = max(0.0, min(1.0, pct))
        if self.anneal_strategy == "cos":
            return _anneal_cos(start, end, pct)
        return _anneal_linear(start, end, pct)

    def get_lr(self):
        """
        Return current LRs for each param_group.
        last_epoch is the number of step() calls already executed.
        """
        step_num = self.last_epoch
        if step_num < 0:
            return list(self.initial_lrs)
        if step_num >= self.total_steps:
            return list(self.min_lrs)

        if not self.three_phase:
            up_end, cycle_end = self._phase_ends
            if step_num <= up_end:
                pct = step_num / float(up_end)
                return [
                    self._anneal(s, m, pct)
                    for s, m in zip(self.initial_lrs, self.max_lrs)
                ]
            pct = (step_num - up_end) / float(max(1, cycle_end - up_end))
            return [
                self._anneal(m, mn, pct)
                for m, mn in zip(self.max_lrs, self.min_lrs)
            ]

        up_end, down_end, cycle_end = self._phase_ends
        if step_num <= up_end:
            pct = step_num / float(up_end)
            return [
                self._anneal(s, m, pct)
                for s, m in zip(self.initial_lrs, self.max_lrs)
            ]
        if step_num <= down_end:
            pct = (step_num - up_end) / float(max(1, down_end - up_end))
            return [
                self._anneal(m, s, pct)
                for s, m in zip(self.initial_lrs, self.max_lrs)
            ]
        pct = (step_num - down_end) / float(max(1, cycle_end - down_end))
        return [
            self._anneal(s, mn, pct)
            for s, mn in zip(self.initial_lrs, self.min_lrs)
        ]

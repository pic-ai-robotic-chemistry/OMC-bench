from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    data_dir: str
    train_meta: str
    test_meta: str
    e0_path: str
    log_dir: str


@dataclass
class LoaderConfig:
    sampler: str = "bin_packing"
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    max_cost_per_batch: int
    lr: float
    epochs: int
    huber_delta: float = 0.01
    energy_loss_weight: float = 10.0
    force_loss_weight: float = 10.0
    stress_loss_weight: float = 10.0
    finetune_mode: bool = False
    lr_gnn: float = 1e-5
    use_direct_force: bool = False
    checkpoint_name_template: str = "model_epoch_{epoch}.pt"
    step_scheduler_on_val: bool = False


@dataclass
class RestartConfig:
    enabled: bool = False
    checkpoint_path: str | None = None


@dataclass
class FinetuneConfig:
    enabled: bool = False
    checkpoint_path: str | None = None
    strict_load: bool = False


@dataclass
class DistributedConfig:
    init_timeout_minutes: int | None = None
    seed: int | None = None


@dataclass
class RunConfig:
    name: str
    paths: PathsConfig
    loader: LoaderConfig
    training: TrainingConfig
    model_params: dict[str, Any]
    restart: RestartConfig = field(default_factory=RestartConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_config_path(config_path: str | None) -> Path:
    if config_path is None:
        return Path("configs/train/latest.py")
    return Path(config_path)

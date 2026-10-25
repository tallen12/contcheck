"""Training settings, read from ``configs/train.yaml``."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainConfig(BaseModel):
    """Everything a training run needs besides the device."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoder: str = "google/electra-small-discriminator"
    seed: int = 13
    window: int = Field(default=256, ge=8)
    stride: int = Field(default=128, ge=1)
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=3, ge=1)
    patience: int = Field(default=1, ge=0)
    """Epochs without a better val F0.5 before stopping."""
    encoder_lr: float = Field(default=3e-5, gt=0)
    head_lr: float = Field(default=1e-3, gt=0)
    warmup_ratio: float = Field(default=0.06, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    outside_weight: float = Field(default=0.2, gt=0)
    """Loss weight of the ``O`` label; every other label weighs 1."""
    fuse_dim: int = Field(default=128, ge=1)
    use_doc: bool = False
    train: Path = Path("data/synthetic/train.jsonl")
    val: Path = Path("data/synthetic/val.jsonl")
    out: Path = Path("runs/baseline")
    log_every: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def _check_stride(self) -> TrainConfig:
        if self.stride > self.window - 2:
            raise ValueError("stride must leave windows overlapping or touching")
        return self


def load_train_config(path: Path) -> TrainConfig:
    """Read and validate a training config file."""
    with path.open(encoding="utf-8") as file:
        data: object = yaml.safe_load(file)
    return TrainConfig.model_validate(data)

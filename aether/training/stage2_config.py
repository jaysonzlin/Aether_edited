"""Strict configuration for fixed-view SimGen Stage-2 refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


class Stage2TrainingConfigError(ValueError):
    """Raised when Stage-2 configuration violates its immutable contract."""


@dataclass(frozen=True)
class Stage2TrainingConfig:
    data_root: str
    sample_ids: tuple[int, ...]
    num_frames: int
    height: int
    width: int
    left_padding: int
    history_slots: int
    max_train_steps: int
    output_interval: int
    mixed_precision: str
    learning_rate: float
    onecycle_pct_start: float
    train_batch_size: int
    gradient_accumulation_steps: int
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float
    dataloader_num_workers: int
    pin_memory: bool
    report_to: str | None
    wandb_project: str
    seed: int
    output_dir: str
    stage1_checkpoint: str
    aether_model_id: str
    cogvideox_model_id: str
    rgb_loss_weight: float
    depth_loss_weight: float
    pointmap_loss_weight: float


_FIELDS = frozenset(Stage2TrainingConfig.__dataclass_fields__)


def _parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise Stage2TrainingConfigError(f"override must use key=value syntax: {override}")
    key, value = override.split("=", 1)
    if key not in _FIELDS:
        raise Stage2TrainingConfigError(f"unknown configuration key: {key}")
    return key, yaml.safe_load(value)


def _validate(raw: dict[str, Any]) -> Stage2TrainingConfig:
    unknown, missing = set(raw) - _FIELDS, _FIELDS - set(raw)
    if unknown:
        raise Stage2TrainingConfigError(f"unknown configuration keys: {sorted(unknown)}")
    if missing:
        raise Stage2TrainingConfigError(f"missing configuration keys: {sorted(missing)}")
    try:
        config = Stage2TrainingConfig(
            **{**raw, "sample_ids": tuple(int(sample_id) for sample_id in raw["sample_ids"])}
        )
    except (TypeError, ValueError) as error:
        raise Stage2TrainingConfigError("configuration fields have invalid types") from error
    if (config.num_frames, config.height, config.width, config.left_padding) != (41, 480, 720, 120):
        raise Stage2TrainingConfigError("Stage-2 requires 41 frames and 480x720 padding geometry")
    if config.history_slots != 4 or config.sample_ids != tuple(range(128)):
        raise Stage2TrainingConfigError("Stage-2 requires four history slots and all 128 samples")
    if config.max_train_steps != 2500:
        raise Stage2TrainingConfigError("max_train_steps must be 2500 for Stage-2")
    if config.output_interval != 500:
        raise Stage2TrainingConfigError("output_interval must be 500 for Stage-2")
    if config.mixed_precision != "bf16":
        raise Stage2TrainingConfigError("mixed_precision must be bf16")
    if config.learning_rate != 1e-5:
        raise Stage2TrainingConfigError("learning_rate must be 1e-5 for Stage-2")
    if config.onecycle_pct_start != 0.1:
        raise Stage2TrainingConfigError("onecycle_pct_start must be 0.1 for Stage-2")
    if not config.stage1_checkpoint.rstrip("/").endswith("checkpoint-010000"):
        raise Stage2TrainingConfigError("stage1_checkpoint must name checkpoint-010000")
    numeric = (
        config.learning_rate,
        config.adam_beta1,
        config.adam_beta2,
        config.adam_epsilon,
        config.weight_decay,
        config.max_grad_norm,
        config.rgb_loss_weight,
        config.depth_loss_weight,
        config.pointmap_loss_weight,
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
        raise Stage2TrainingConfigError("numeric configuration values must be finite")
    if min(config.rgb_loss_weight, config.depth_loss_weight, config.pointmap_loss_weight) <= 0:
        raise Stage2TrainingConfigError("loss weights must be positive")
    if config.train_batch_size != 1 or config.gradient_accumulation_steps != 1:
        raise Stage2TrainingConfigError("Stage-2 requires batch size and accumulation of one")
    return config


def load_stage2_training_config(
    path: str | Path, overrides: Sequence[str] = ()
) -> Stage2TrainingConfig:
    with Path(path).open() as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise Stage2TrainingConfigError("configuration root must be a mapping")
    raw = dict(raw)
    for override in overrides:
        key, value = _parse_override(override)
        raw[key] = value
    return _validate(raw)

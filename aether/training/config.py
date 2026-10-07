"""Strict, portable configuration for the first fixed-view SimGen run."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


class TrainingConfigError(ValueError):
    """Raised when a training configuration violates the fixed-view contract."""


@dataclass(frozen=True)
class FixedViewTrainingConfig:
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
    train_batch_size: int
    gradient_accumulation_steps: int
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    dataloader_num_workers: int
    pin_memory: bool
    report_to: str | None
    wandb_project: str
    seed: int
    output_dir: str
    aether_model_id: str
    cogvideox_model_id: str


_FIELDS = frozenset(FixedViewTrainingConfig.__dataclass_fields__)


def _parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise TrainingConfigError(f"override must use key=value syntax: {override}")
    key, value = override.split("=", 1)
    if key not in _FIELDS:
        raise TrainingConfigError(f"unknown override key: {key}")
    return key, yaml.safe_load(value)


def _validate(raw: dict[str, Any]) -> FixedViewTrainingConfig:
    unknown = set(raw) - _FIELDS
    missing = _FIELDS - set(raw)
    if unknown:
        raise TrainingConfigError(f"unknown configuration keys: {sorted(unknown)}")
    if missing:
        raise TrainingConfigError(f"missing configuration keys: {sorted(missing)}")

    try:
        config = FixedViewTrainingConfig(
            **{
                **raw,
                "sample_ids": tuple(int(sample_id) for sample_id in raw["sample_ids"]),
            }
        )
    except (TypeError, ValueError) as error:
        raise TrainingConfigError("configuration fields have invalid types") from error

    if config.num_frames != 41:
        raise TrainingConfigError("num_frames must be 41 for the Aether fixed-view path")
    if (config.height, config.width, config.left_padding) != (480, 720, 120):
        raise TrainingConfigError("height, width, and left_padding must be 480, 720, and 120")
    if config.history_slots != 4:
        raise TrainingConfigError("history_slots must be 4")
    if config.max_train_steps != 10_000:
        raise TrainingConfigError("max_train_steps must be 10000 for the first run")
    if config.output_interval != 1_000:
        raise TrainingConfigError("output_interval must be 1000 for the first run")
    if config.mixed_precision != "bf16":
        raise TrainingConfigError("mixed_precision must be bf16")
    if config.sample_ids != tuple(range(128)):
        raise TrainingConfigError("sample_ids must contain every ID from 0 through 127")
    if config.learning_rate <= 0 or config.train_batch_size <= 0:
        raise TrainingConfigError("learning_rate and train_batch_size must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise TrainingConfigError("gradient_accumulation_steps must be positive")
    numeric_fields = ("adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm")
    if any(
        isinstance(getattr(config, name), bool)
        or not isinstance(getattr(config, name), (int, float))
        for name in numeric_fields
    ):
        raise TrainingConfigError(f"{', '.join(numeric_fields)} must be numeric")
    if not 0.0 <= config.adam_beta1 < 1.0:
        raise TrainingConfigError("adam_beta1 must be in [0, 1)")
    if not 0.0 <= config.adam_beta2 < 1.0:
        raise TrainingConfigError("adam_beta2 must be in [0, 1)")
    if not math.isfinite(config.adam_epsilon) or config.adam_epsilon <= 0:
        raise TrainingConfigError("adam_epsilon must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise TrainingConfigError("weight_decay must be finite and non-negative")
    if type(config.warmup_steps) is not int or not 0 <= config.warmup_steps < config.max_train_steps:
        raise TrainingConfigError("warmup_steps must be an integer in [0, max_train_steps)")
    if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
        raise TrainingConfigError("max_grad_norm must be finite and positive")
    if type(config.dataloader_num_workers) is not int or config.dataloader_num_workers < 0:
        raise TrainingConfigError("dataloader_num_workers must be a non-negative integer")
    if not isinstance(config.pin_memory, bool):
        raise TrainingConfigError("pin_memory must be boolean")
    if config.report_to not in (None, "wandb"):
        raise TrainingConfigError("report_to must be 'wandb' or null")
    if not isinstance(config.wandb_project, str) or not config.wandb_project.strip():
        raise TrainingConfigError("wandb_project must be a non-empty string")
    return config


def load_training_config(
    path: str | Path, overrides: Sequence[str] = ()
) -> FixedViewTrainingConfig:
    """Load one YAML file and validate the deliberate first-run constants."""
    config_path = Path(path)
    with config_path.open() as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise TrainingConfigError("configuration root must be a mapping")
    raw = dict(raw)
    for override in overrides:
        key, value = _parse_override(override)
        raw[key] = value
    return _validate(raw)

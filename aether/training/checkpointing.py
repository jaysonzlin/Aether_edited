"""Numbered, resumable Accelerate checkpoints for fixed-view training."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Protocol


class AcceleratorState(Protocol):
    def save_state(self, output_dir: str | Path) -> None: ...

    def load_state(self, input_dir: str | Path) -> None: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(directory: Path, show_progress: bool = False) -> list[dict[str, object]]:
    from tqdm.auto import tqdm

    entries = []
    paths = sorted(item for item in directory.rglob("*") if item.is_file())
    for path in tqdm(paths, desc=f"Hashing {directory.name}", unit="file", disable=not show_progress):
        entry: dict[str, object] = {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        entry["sha256"] = _file_sha256(path)
        entries.append(entry)
    return entries


def build_run_manifest(
    config, prediction_type: str, show_progress: bool = False
) -> dict[str, object]:
    """Build path-independent run identity for safe checkpoint restoration."""
    aether_root = Path(config.aether_model_id) / "transformer"
    cogvideox_root = Path(config.cogvideox_model_id)
    data_root = Path(config.data_root)
    model_identity = {
        "aether_transformer": _inventory(aether_root, show_progress=show_progress),
        "cogvideox": {
            component: _inventory(cogvideox_root / component, show_progress=show_progress)
            for component in ("tokenizer", "text_encoder", "vae", "scheduler")
        },
    }
    sample_identity = []
    from tqdm.auto import tqdm
    for sample_id in tqdm(config.sample_ids, desc="Hashing SimGen", unit="sample", disable=not show_progress):
        view_dir = data_root / f"sample_{sample_id}" / "view_0"
        files = [view_dir / f"{frame_index:08d}.png" for frame_index in range(config.num_frames)]
        files.extend((view_dir / "depth.h5", view_dir / "cameras.json"))
        sample_identity.append(
            {
                "sample_id": int(sample_id),
                "files": [
                    {
                        "path": path.relative_to(data_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _file_sha256(path),
                    }
                    for path in files
                ],
            }
        )
    return {
        "schema_version": 2,
        "objective": {
            "name": "stage1_diffusion_prediction_mse",
            "prediction_type": prediction_type,
        },
        "training": {
            "sample_ids": list(config.sample_ids),
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "left_padding": config.left_padding,
            "history_slots": config.history_slots,
            "max_train_steps": config.max_train_steps,
            "output_interval": config.output_interval,
            "mixed_precision": config.mixed_precision,
            "learning_rate": config.learning_rate,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_epsilon": config.adam_epsilon,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "max_grad_norm": config.max_grad_norm,
            "dataloader_num_workers": config.dataloader_num_workers,
            "pin_memory": config.pin_memory,
            "report_to": config.report_to,
            "wandb_project": config.wandb_project,
            "seed": config.seed,
        },
        "models": model_identity,
        "dataset": sample_identity,
    }


def _checkpoint_path(output_dir: str | Path, global_step: int) -> Path:
    if global_step <= 0:
        raise ValueError("global_step must be positive")
    return Path(output_dir) / f"checkpoint-{global_step:06d}"


def _trim_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for checkpoint in checkpoints[:-keep_last]:
        if checkpoint.parent.resolve() != output_dir.resolve():
            raise RuntimeError(f"refusing to remove checkpoint outside output directory: {checkpoint}")
        shutil.rmtree(checkpoint)


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the newest checkpoint with valid global-step metadata."""
    output_path = Path(output_dir)
    checkpoints = sorted(
        (path for path in output_path.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for checkpoint in checkpoints:
        try:
            metadata = json.loads((checkpoint / "metadata.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if isinstance(metadata.get("global_step"), int) and metadata["global_step"] > 0:
            return checkpoint
    return None


def save_checkpoint(
    accelerator: AcceleratorState,
    output_dir: str | Path,
    global_step: int,
    run_manifest: Mapping[str, object],
    keep_last: int = 3,
) -> Path:
    """Save full Accelerate state and global-step metadata without overwriting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_path(output_path, global_step)
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists and will not be overwritten: {checkpoint}")
    accelerator.save_state(checkpoint)
    wait_for_everyone = getattr(accelerator, "wait_for_everyone", None)
    if wait_for_everyone is not None:
        wait_for_everyone()
    if getattr(accelerator, "is_main_process", True):
        (checkpoint / "metadata.json").write_text(
            json.dumps(
                {"global_step": global_step, "run_manifest": dict(run_manifest)},
                sort_keys=True,
            )
            + "\n"
        )
        _trim_checkpoints(output_path, keep_last)
    if wait_for_everyone is not None:
        wait_for_everyone()
    return checkpoint


def restore_checkpoint(
    accelerator: AcceleratorState,
    checkpoint: str | Path,
    expected_manifest: Mapping[str, object],
) -> int:
    """Validate run identity, restore Accelerator state, and return the saved step."""
    checkpoint_path = Path(checkpoint)
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    global_step = metadata.get("global_step")
    if not isinstance(global_step, int) or global_step <= 0:
        raise ValueError(f"invalid global_step in checkpoint metadata: {metadata_path}")
    saved_manifest = metadata.get("run_manifest")
    if not isinstance(saved_manifest, dict):
        raise ValueError(
            f"run manifest missing from checkpoint {checkpoint_path}; "
            "legacy checkpoints cannot be resumed"
        )
    if saved_manifest != dict(expected_manifest):
        raise ValueError(
            f"incompatible run manifest in checkpoint {checkpoint_path}; "
            "model, data, objective, or training configuration differs"
        )
    accelerator.load_state(checkpoint_path)
    return global_step

"""Numbered, resumable Accelerate checkpoints for fixed-view training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Protocol


class AcceleratorState(Protocol):
    def save_state(self, output_dir: str | Path) -> None: ...

    def load_state(self, input_dir: str | Path) -> None: ...


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
            json.dumps({"global_step": global_step}) + "\n"
        )
        _trim_checkpoints(output_path, keep_last)
    if wait_for_everyone is not None:
        wait_for_everyone()
    return checkpoint


def restore_checkpoint(accelerator: AcceleratorState, checkpoint: str | Path) -> int:
    """Restore Accelerator state and return the continuing global step."""
    checkpoint_path = Path(checkpoint)
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    global_step = metadata.get("global_step")
    if not isinstance(global_step, int) or global_step <= 0:
        raise ValueError(f"invalid global_step in checkpoint metadata: {metadata_path}")
    accelerator.load_state(checkpoint_path)
    return global_step

"""Numbered, resumable Accelerate checkpoints for fixed-view training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
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


def _read_fingerprint_cache(cache_path: Path | None) -> dict[str, dict[str, object]]:
    if cache_path is None:
        return {}
    try:
        payload = json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {}
    entries = payload.get("files")
    if not isinstance(entries, dict):
        return {}
    return {
        key: entry
        for key, entry in entries.items()
        if isinstance(key, str) and isinstance(entry, dict)
    }


def _write_fingerprint_cache(
    cache_path: Path | None, entries: dict[str, dict[str, object]]
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "files": entries}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fingerprinted_file(
    path: Path,
    relative_path: str,
    cached_entries: dict[str, dict[str, object]],
    updated_entries: dict[str, dict[str, object]],
    force_rehash: bool,
) -> dict[str, object]:
    stat = path.stat()
    cache_key = str(path.resolve())
    cached = cached_entries.get(cache_key)
    sha256 = None
    if not force_rehash and cached is not None:
        if (
            cached.get("size_bytes") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            sha256 = cached["sha256"]
    if sha256 is None:
        sha256 = _file_sha256(path)
    updated_entries[cache_key] = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256,
    }
    return {
        "path": relative_path,
        "size_bytes": stat.st_size,
        "sha256": sha256,
    }


def _inventory(
    directory: Path,
    show_progress: bool = False,
    cached_entries: dict[str, dict[str, object]] | None = None,
    updated_entries: dict[str, dict[str, object]] | None = None,
    force_rehash: bool = False,
) -> list[dict[str, object]]:
    from tqdm.auto import tqdm

    cached_entries = cached_entries or {}
    updated_entries = updated_entries if updated_entries is not None else {}
    entries = []
    paths = sorted(item for item in directory.rglob("*") if item.is_file())
    for path in tqdm(
        paths,
        desc=f"Fingerprinting {directory.name}",
        unit="file",
        disable=not show_progress,
    ):
        entries.append(
            _fingerprinted_file(
                path,
                path.relative_to(directory).as_posix(),
                cached_entries,
                updated_entries,
                force_rehash,
            )
        )
    return entries


def build_run_manifest(
    config,
    prediction_type: str,
    show_progress: bool = False,
    fingerprint_cache_path: str | Path | None = None,
    force_rehash: bool = False,
) -> dict[str, object]:
    """Build path-independent run identity for safe checkpoint restoration."""
    cache_path = Path(fingerprint_cache_path) if fingerprint_cache_path is not None else None
    cached_entries = {} if force_rehash else _read_fingerprint_cache(cache_path)
    updated_entries: dict[str, dict[str, object]] = {}
    aether_root = Path(config.aether_model_id) / "transformer"
    cogvideox_root = Path(config.cogvideox_model_id)
    data_root = Path(config.data_root)
    model_identity = {
        "aether_transformer": _inventory(
            aether_root,
            show_progress=show_progress,
            cached_entries=cached_entries,
            updated_entries=updated_entries,
            force_rehash=force_rehash,
        ),
        "cogvideox": {
            component: _inventory(
                cogvideox_root / component,
                show_progress=show_progress,
                cached_entries=cached_entries,
                updated_entries=updated_entries,
                force_rehash=force_rehash,
            )
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
                    _fingerprinted_file(
                        path,
                        path.relative_to(data_root).as_posix(),
                        cached_entries,
                        updated_entries,
                        force_rehash,
                    )
                    for path in files
                ],
            }
        )
    _write_fingerprint_cache(cache_path, updated_entries)
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

"""Fixed-name MP4 artifacts for qualitative fixed-view training checks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import imageio.v3 as imageio
import numpy as np


VideoWriter = Callable[[Path, np.ndarray, int], None]


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    imageio.imwrite(path, frames, fps=fps)


def _as_uint8_video(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("video frames must have shape [frames, height, width, 3]")
    return (np.clip(frames, 0.0, 1.0) * 255).round().astype(np.uint8)


def save_fixed_rollout(
    output_dir: str | Path,
    global_step: int,
    target_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    fps: int,
    writer: VideoWriter = _write_mp4,
) -> tuple[Path, Path]:
    """Save target and fixed-seed generated videos with deterministic filenames."""
    if global_step <= 0:
        raise ValueError("global_step must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_path = output_path / f"step_{global_step:06d}_target.mp4"
    generated_path = output_path / f"step_{global_step:06d}_generated.mp4"
    writer(target_path, _as_uint8_video(target_rgb), fps)
    writer(generated_path, _as_uint8_video(generated_rgb), fps)
    return target_path, generated_path

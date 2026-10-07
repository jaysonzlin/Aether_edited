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


def normalized_disparity_to_viridis_frames(disparity: np.ndarray) -> np.ndarray:
    """Map normalized relative-disparity frames [F,C,H,W] or [F,H,W] to viridis RGB."""
    from matplotlib import colormaps

    disparity = np.asarray(disparity, dtype=np.float32)
    if disparity.ndim == 4 and disparity.shape[1] == 3:
        disparity = disparity.mean(axis=1)
    if disparity.ndim != 3:
        raise ValueError("disparity must have shape [frames, height, width] or [frames, 3, height, width]")
    relative_disparity = np.square(np.clip(disparity * 0.5 + 0.5, 0.0, 1.0))
    return colormaps["viridis"](relative_disparity)[..., :3].astype(np.float32)


def decoded_disparity_to_viridis_frames(decoded_disparity) -> np.ndarray:
    """Apply Aether's decoded-disparity channel mean, range map, square, and viridis."""
    decoded = np.asarray(decoded_disparity)
    if decoded.ndim != 5 or decoded.shape[0] != 1:
        raise ValueError("decoded disparity must have shape [1, 3, frames, height, width]")
    channel_mean = decoded.mean(axis=1)[0]
    return normalized_disparity_to_viridis_frames(channel_mean)


def save_fixed_rollout(
    output_dir: str | Path,
    global_step: int,
    target_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    target_disparity: np.ndarray,
    generated_disparity: np.ndarray,
    fps: int,
    writer: VideoWriter = _write_mp4,
) -> tuple[Path, Path, Path, Path]:
    """Save RGB and normalized-disparity target/generated MP4s."""
    if global_step <= 0:
        raise ValueError("global_step must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_path = output_path / f"step_{global_step:06d}_target.mp4"
    generated_path = output_path / f"step_{global_step:06d}_generated.mp4"
    target_disparity_path = output_path / f"step_{global_step:06d}_target_disparity.mp4"
    generated_disparity_path = output_path / f"step_{global_step:06d}_generated_disparity.mp4"
    writer(target_path, _as_uint8_video(target_rgb), fps)
    writer(generated_path, _as_uint8_video(generated_rgb), fps)
    writer(target_disparity_path, _as_uint8_video(target_disparity), fps)
    writer(generated_disparity_path, _as_uint8_video(generated_disparity), fps)
    return target_path, generated_path, target_disparity_path, generated_disparity_path

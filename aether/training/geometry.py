"""Geometry preprocessing shared by the fixed-view SimGen dataset and trainer."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SOURCE_HEIGHT = 480
SOURCE_WIDTH = 480
TARGET_HEIGHT = 480
TARGET_WIDTH = 720
LEFT_PADDING = 120
RAYMAP_DOWNSAMPLE = 8


def pad_intrinsics(
    camera: Mapping[str, object],
    left_padding: int = LEFT_PADDING,
    right_padding: int | None = None,
) -> np.ndarray:
    """Return the padded 3x3 camera matrix without changing focal length."""
    width = int(camera["width"])
    height = int(camera["height"])
    if (height, width) != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError(
            f"expected {SOURCE_WIDTH}x{SOURCE_HEIGHT} camera, got {width}x{height}"
        )
    if right_padding is not None and width + left_padding + right_padding != TARGET_WIDTH:
        raise ValueError("left and right padding must produce the 720-pixel target width")

    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera.get("cx", width / 2)) + left_padding
    cy = float(camera.get("cy", height / 2))
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics must be finite with positive focal lengths")

    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def normalized_disparity(depth: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert depth to Aether's per-clip, square-root normalized disparity.

    Invalid pixels are filled with the clip's farthest valid depth.  The result is
    in ``[-1, 1]`` and is returned with the pre-square-root disparity maximum used
    to scale Aether ray directions.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("depth contains no positive finite values")

    filled_depth = depth.copy()
    filled_depth[~valid] = depth[valid].max()
    raw_disparity = 1.0 / filled_depth
    dmax = float(raw_disparity[valid].max())
    disparity = np.sqrt(np.clip(raw_disparity / dmax, 0.0, 1.0))
    return (disparity * 2.0 - 1.0).astype(np.float32), dmax


def padded_disparity(depth: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize depth and pad it with far-depth values on both horizontal sides."""
    disparity, dmax = normalized_disparity(depth)
    padded = np.pad(
        disparity,
        ((0, 0), (0, 0), (LEFT_PADDING, TARGET_WIDTH - SOURCE_WIDTH - LEFT_PADDING)),
        mode="constant",
        constant_values=-1.0,
    )
    return padded.astype(np.float32), dmax


def camera_poses(cameras: list[Mapping[str, object]]) -> np.ndarray:
    """Build camera-to-world matrices from SimGen camera records."""
    poses = []
    for frame_index, camera in enumerate(cameras):
        try:
            rotation = np.asarray(camera["rotation"], dtype=np.float32)
            position = np.asarray(camera["position"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed camera at frame {frame_index}") from error
        if rotation.shape != (3, 3) or position.shape != (3,):
            raise ValueError(f"malformed camera pose at frame {frame_index}")
        if not np.isfinite(rotation).all() or not np.isfinite(position).all():
            raise ValueError(f"non-finite camera pose at frame {frame_index}")
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        poses.append(pose)
    return np.stack(poses)


def build_raymaps(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    dmax: float,
    ray_o_scale_factor: float = 10.0,
) -> np.ndarray:
    """Construct Aether-compatible six-channel low-resolution raymaps.

    This is algebraically equivalent to Aether's full-resolution ray creation
    followed by bilinear downsampling with ``align_corners=False``.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [frames, 4, 4]")
    if intrinsics.shape != (poses.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [frames, 3, 3]")
    if not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("dmax must be finite and positive")

    height = TARGET_HEIGHT // RAYMAP_DOWNSAMPLE
    width = TARGET_WIDTH // RAYMAP_DOWNSAMPLE
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    sample_offset = (RAYMAP_DOWNSAMPLE - 1) / 2

    directions = []
    origins = []
    for pose, intrinsic in zip(poses, intrinsics):
        x = (u * RAYMAP_DOWNSAMPLE + sample_offset - intrinsic[0, 2]) / intrinsic[0, 0]
        y = (v * RAYMAP_DOWNSAMPLE + sample_offset - intrinsic[1, 2]) / intrinsic[1, 1]
        camera_directions = np.stack((x, y, np.ones_like(x)), axis=-1)
        directions.append(camera_directions @ pose[:3, :3].T)

        scaled_translation = pose[:3, 3] * dmax * ray_o_scale_factor
        origins.append(np.sign(scaled_translation) * np.log1p(np.abs(scaled_translation)))

    ray_directions = np.stack(directions).transpose(0, 3, 1, 2)
    ray_origins = np.stack(origins)[:, :, None, None]
    ray_origins = np.broadcast_to(ray_origins, ray_directions.shape)
    return np.concatenate((ray_directions, ray_origins), axis=1).astype(np.float32)


def pack_raymaps(raymaps: np.ndarray) -> np.ndarray:
    """Pack six-channel source raymaps into Aether's four-frame 24-channel form."""
    raymaps = np.asarray(raymaps, dtype=np.float32)
    if raymaps.ndim != 4 or raymaps.shape[1] != 6:
        raise ValueError("raymaps must have shape [frames, 6, height, width]")

    frame_count = raymaps.shape[0]
    remainder = frame_count % 4
    if remainder:
        raymaps = np.concatenate((raymaps[: 4 - remainder], raymaps), axis=0)
    return raymaps.reshape(-1, 4, *raymaps.shape[1:]).reshape(
        -1, 24, raymaps.shape[2], raymaps.shape[3]
    )

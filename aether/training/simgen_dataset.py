"""Validated fixed-view SimGen clips for Aether's 41-frame training contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from PIL import Image

from aether.training.geometry import (
    LEFT_PADDING,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_WIDTH,
    build_raymaps,
    camera_poses,
    normalized_disparity,
    pack_raymaps,
    pad_intrinsics,
    padded_disparity,
)


FRAME_COUNT = 41


class SimGenFixedViewDataset:
    """Load and validate 41-frame, stationary-camera SimGen samples.

    RGB is float32 in ``[0, 1]`` with replicated edge padding.  Disparity is
    three-channel float32 in ``[-1, 1]`` and uses far-depth padding.
    """

    def __init__(
        self,
        root: str | Path,
        sample_ids: Sequence[int] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SimGen root does not exist: {self.root}")
        self.sample_ids = list(sample_ids) if sample_ids is not None else self._discover_ids()
        if not self.sample_ids:
            raise ValueError(f"no sample_<id> directories found under {self.root}")
        self._validate_manifest()

    def _discover_ids(self) -> list[int]:
        sample_ids = []
        for sample_dir in self.root.glob("sample_*"):
            try:
                sample_ids.append(int(sample_dir.name.removeprefix("sample_")))
            except ValueError:
                continue
        return sorted(sample_ids)

    def _view_dir(self, sample_id: int) -> Path:
        return self.root / f"sample_{sample_id}" / "view_0"

    def _validate_manifest(self) -> None:
        for sample_id in self.sample_ids:
            view_dir = self._view_dir(sample_id)
            if not view_dir.is_dir():
                raise FileNotFoundError(f"missing SimGen view directory: {view_dir}")
            for frame_index in range(FRAME_COUNT):
                frame_path = view_dir / f"{frame_index:08d}.png"
                if not frame_path.is_file():
                    raise FileNotFoundError(f"missing required RGB frame: {frame_path}")
            if not (view_dir / "depth.h5").is_file():
                raise FileNotFoundError(f"missing depth file: {view_dir / 'depth.h5'}")
            if not (view_dir / "cameras.json").is_file():
                raise FileNotFoundError(f"missing camera file: {view_dir / 'cameras.json'}")
            for frame_index in range(FRAME_COUNT):
                frame_path = view_dir / f"{frame_index:08d}.png"
                with Image.open(frame_path) as image:
                    if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                        raise ValueError(
                            f"expected 480x480 RGB frame, got {image.size}: {frame_path}"
                        )
            normalized_disparity(self._load_depth(view_dir / "depth.h5"))
            cameras = self._load_cameras(view_dir / "cameras.json")
            camera_poses(cameras)
            for camera in cameras:
                pad_intrinsics(camera)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        sample_id = self.sample_ids[index]
        view_dir = self._view_dir(sample_id)
        rgb = self._load_rgb(view_dir)
        depth = self._load_depth(view_dir / "depth.h5")
        cameras = self._load_cameras(view_dir / "cameras.json")

        disparity, dmax = padded_disparity(depth)
        intrinsics = np.stack([pad_intrinsics(camera) for camera in cameras])
        raymap = pack_raymaps(build_raymaps(camera_poses(cameras), intrinsics, dmax))
        content_mask = np.zeros((1, SOURCE_HEIGHT, TARGET_WIDTH), dtype=bool)
        content_mask[:, :, LEFT_PADDING : LEFT_PADDING + SOURCE_WIDTH] = True

        return {
            "sample_id": f"sample_{sample_id}",
            "rgb": rgb,
            "disparity": np.repeat(disparity[:, None], 3, axis=1),
            "raymap": raymap,
            "content_mask": content_mask,
        }

    @staticmethod
    def _load_rgb(view_dir: Path) -> np.ndarray:
        frames = []
        for frame_index in range(FRAME_COUNT):
            frame_path = view_dir / f"{frame_index:08d}.png"
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    raise ValueError(
                        f"expected 480x480 RGB frame, got {image.size}: {frame_path}"
                    )
                frame = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(frame)
        rgb = np.stack(frames).transpose(0, 3, 1, 2)
        return np.pad(rgb, ((0, 0), (0, 0), (0, 0), (LEFT_PADDING, LEFT_PADDING)), mode="edge")

    @staticmethod
    def _load_depth(depth_path: Path) -> np.ndarray:
        with h5py.File(depth_path, "r") as depth_file:
            if "depth" not in depth_file:
                raise ValueError(f"depth dataset missing from {depth_path}")
            depth = np.asarray(depth_file["depth"][:FRAME_COUNT], dtype=np.float32)
        if depth.shape == (FRAME_COUNT, 1, SOURCE_HEIGHT, SOURCE_WIDTH):
            depth = depth[:, 0]
        if depth.shape != (FRAME_COUNT, SOURCE_HEIGHT, SOURCE_WIDTH):
            raise ValueError(
                f"expected depth shape {(FRAME_COUNT, SOURCE_HEIGHT, SOURCE_WIDTH)}, "
                f"got {depth.shape}: {depth_path}"
            )
        return depth

    @staticmethod
    def _load_cameras(camera_path: Path) -> list[dict[str, object]]:
        with camera_path.open() as camera_file:
            cameras = json.load(camera_file)
        if not isinstance(cameras, list) or len(cameras) < FRAME_COUNT:
            raise ValueError(f"expected at least {FRAME_COUNT} cameras: {camera_path}")
        cameras = cameras[:FRAME_COUNT]
        if not all(isinstance(camera, dict) for camera in cameras):
            raise ValueError(f"malformed camera records: {camera_path}")
        return cameras


# Keep the concise name used by early experiment scripts while exposing the
# plan's explicit public interface.
FixedViewSimGenDataset = SimGenFixedViewDataset

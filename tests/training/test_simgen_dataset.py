import json

import h5py
import numpy as np
import pytest
from PIL import Image

from aether.training.simgen_dataset import FixedViewSimGenDataset


def _write_simgen_sample(
    root, *, missing_frame=None, depth_value=2.0, singleton_depth_channel=False
):
    view_dir = root / "sample_0" / "view_0"
    view_dir.mkdir(parents=True)

    for frame_index in range(41):
        if frame_index == missing_frame:
            continue
        Image.new("RGB", (480, 480), (frame_index, 0, 0)).save(
            view_dir / f"{frame_index:08d}.png"
        )

    with h5py.File(view_dir / "depth.h5", "w") as depth_file:
        depth = np.full((41, 480, 480), depth_value, dtype=np.float16)
        if singleton_depth_channel:
            depth = depth[:, None]
        depth_file.create_dataset(
            "depth",
            data=depth,
        )

    camera = {
        "width": 480,
        "height": 480,
        "fx": 720.0,
        "fy": 720.0,
        "rotation": np.eye(3).tolist(),
        "position": [0.0, 0.0, 0.0],
    }
    (view_dir / "cameras.json").write_text(json.dumps([camera] * 41))


def test_dataset_loads_a_41_frame_clip_with_padded_geometry(tmp_path):
    _write_simgen_sample(tmp_path)

    item = FixedViewSimGenDataset(tmp_path)[0]

    assert item["sample_id"] == "sample_0"
    assert item["rgb"].shape == (41, 3, 480, 720)
    assert item["disparity"].shape == (41, 3, 480, 720)
    assert item["raymap"].shape == (11, 24, 60, 90)
    assert item["content_mask"].shape == (1, 480, 720)
    assert item["content_mask"][:, :, :120].sum() == 0
    assert item["content_mask"][:, :, 120:600].all()
    assert item["content_mask"][:, :, 600:].sum() == 0


def test_dataset_rejects_a_clip_with_a_missing_required_frame(tmp_path):
    _write_simgen_sample(tmp_path, missing_frame=40)

    with pytest.raises(FileNotFoundError, match="00000040.png"):
        FixedViewSimGenDataset(tmp_path)


def test_dataset_accepts_simgen_singleton_depth_channel(tmp_path):
    _write_simgen_sample(tmp_path, singleton_depth_channel=True)

    item = FixedViewSimGenDataset(tmp_path)[0]

    assert item["disparity"].shape == (41, 3, 480, 720)


def test_dataset_rejects_depth_with_no_positive_finite_values(tmp_path):
    _write_simgen_sample(tmp_path, depth_value=0.0)

    with pytest.raises(ValueError, match="positive finite"):
        FixedViewSimGenDataset(tmp_path)

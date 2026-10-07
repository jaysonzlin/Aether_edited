import numpy as np
from types import SimpleNamespace
import sys

from aether.training.visualization import (
    decoded_disparity_to_viridis_frames,
    normalized_disparity_to_viridis_frames,
    save_fixed_rollout,
)


def test_fixed_rollout_uses_numbered_target_and_generated_filenames(tmp_path):
    writes = []

    def writer(path, frames, fps):
        writes.append((path.name, frames.shape, fps))

    saved = save_fixed_rollout(
        output_dir=tmp_path,
        global_step=1_000,
        target_rgb=np.zeros((2, 4, 4, 3), dtype=np.float32),
        generated_rgb=np.ones((2, 4, 4, 3), dtype=np.float32),
        target_disparity=np.zeros((2, 4, 4, 3), dtype=np.float32),
        generated_disparity=np.ones((2, 4, 4, 3), dtype=np.float32),
        fps=12,
        writer=writer,
    )

    assert [path.name for path in saved] == [
        "step_001000_target.mp4",
        "step_001000_generated.mp4",
        "step_001000_target_disparity.mp4",
        "step_001000_generated_disparity.mp4",
    ]
    assert writes == [
        ("step_001000_target.mp4", (2, 4, 4, 3), 12),
        ("step_001000_generated.mp4", (2, 4, 4, 3), 12),
        ("step_001000_target_disparity.mp4", (2, 4, 4, 3), 12),
        ("step_001000_generated_disparity.mp4", (2, 4, 4, 3), 12),
    ]


def test_normalized_disparity_uses_aether_transform_then_viridis(monkeypatch):
    calls = []

    def viridis(values):
        calls.append(values.copy())
        return np.stack((values, values * 0 + 0.5, values * 0 + 0.25, values * 0 + 1), axis=-1)

    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(colormaps={"viridis": viridis}))
    disparity = np.array([[[ -1.0, 1.0 ]]], dtype=np.float32)

    frames = normalized_disparity_to_viridis_frames(disparity)

    assert frames.shape == (1, 1, 2, 3)
    assert np.allclose(calls[0], np.array([[[0.0, 1.0]]]))
    assert np.allclose(frames[0, 0, 0], [0.0, 0.5, 0.25])
    assert np.allclose(frames[0, 0, 1], [1.0, 0.5, 0.25])


def test_decoded_disparity_means_rgb_channels_before_range_mapping(monkeypatch):
    calls = []

    def viridis(values):
        calls.append(values.copy())
        return np.stack((values, values, values, values * 0 + 1), axis=-1)

    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(colormaps={"viridis": viridis}))
    decoded = np.array([[[[[ -1.0, 1.0 ]]], [[[ -1.0, 1.0 ]]], [[[ -1.0, 1.0 ]]]]])

    frames = decoded_disparity_to_viridis_frames(decoded)

    assert frames.shape == (1, 1, 2, 3)
    assert np.allclose(calls[0], np.array([[[0.0, 1.0]]]))

import numpy as np

from aether.training.visualization import save_fixed_rollout


def test_fixed_rollout_uses_numbered_target_and_generated_filenames(tmp_path):
    writes = []

    def writer(path, frames, fps):
        writes.append((path.name, frames.shape, fps))

    saved = save_fixed_rollout(
        output_dir=tmp_path,
        global_step=1_000,
        target_rgb=np.zeros((2, 4, 4, 3), dtype=np.float32),
        generated_rgb=np.ones((2, 4, 4, 3), dtype=np.float32),
        fps=12,
        writer=writer,
    )

    assert [path.name for path in saved] == [
        "step_001000_target.mp4",
        "step_001000_generated.mp4",
    ]
    assert writes == [
        ("step_001000_target.mp4", (2, 4, 4, 3), 12),
        ("step_001000_generated.mp4", (2, 4, 4, 3), 12),
    ]

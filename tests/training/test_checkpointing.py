from pathlib import Path

from aether.training.checkpointing import restore_checkpoint, save_checkpoint


class FakeAccelerator:
    def __init__(self):
        self.restored_path = None

    def save_state(self, path):
        path = Path(path)
        path.mkdir()
        (path / "state.txt").write_text("state")

    def load_state(self, path):
        self.restored_path = Path(path)


def test_restore_returns_the_saved_global_step(tmp_path):
    accelerator = FakeAccelerator()
    checkpoint = save_checkpoint(accelerator, tmp_path, global_step=1_000)

    restored_step = restore_checkpoint(accelerator, checkpoint)

    assert restored_step == 1_000
    assert accelerator.restored_path == checkpoint


def test_checkpoint_retention_keeps_the_most_recent_numbered_directories(tmp_path):
    accelerator = FakeAccelerator()
    for step in (1_000, 2_000, 3_000):
        save_checkpoint(accelerator, tmp_path, global_step=step, keep_last=2)

    assert not (tmp_path / "checkpoint-001000").exists()
    assert (tmp_path / "checkpoint-002000").exists()
    assert (tmp_path / "checkpoint-003000").exists()

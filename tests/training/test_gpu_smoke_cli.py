import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT_PATH = Path("scripts/train_fixed_view_simgen.py")


def _training_script_module():
    spec = importlib.util.spec_from_file_location("train_fixed_view_simgen", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_smoke_mode_runs_exactly_one_optimizer_step(monkeypatch):
    training_script = _training_script_module()
    monkeypatch.setattr(sys, "argv", ["train_fixed_view_simgen.py", "--gpu-smoke-test"])

    args = training_script.parse_args()

    assert args.gpu_smoke_test is True
    assert training_script.training_steps(10_000, gpu_smoke_test=True) == 1


def test_training_defaults_to_resuming_the_latest_checkpoint(monkeypatch):
    training_script = _training_script_module()
    monkeypatch.setattr(sys, "argv", ["train_fixed_view_simgen.py"])

    args = training_script.parse_args()

    assert args.resume == "latest"


def test_completed_run_does_not_take_an_extra_optimizer_step():
    training_script = _training_script_module()

    assert training_script.has_remaining_steps(completed_steps=9_999, max_train_steps=10_000)
    assert not training_script.has_remaining_steps(completed_steps=10_000, max_train_steps=10_000)


def test_fixed_rollout_batch_uses_sample_zero_without_dataloader_shuffle():
    training_script = _training_script_module()

    class Dataset:
        sample_ids = (3, 0, 1)

        def __getitem__(self, index):
            assert index == 1
            return {
                "rgb": np.zeros((41, 3, 480, 720), dtype=np.float32),
                "disparity": np.zeros((41, 3, 480, 720), dtype=np.float32),
                "raymap": np.zeros((11, 24, 60, 90), dtype=np.float32),
                "sample_id": 0,
            }

    batch = training_script.fixed_rollout_batch(Dataset(), torch.device("cpu"))

    assert set(batch) == {"rgb", "disparity", "raymap"}
    assert batch["rgb"].shape == (1, 41, 3, 480, 720)
    assert batch["raymap"].shape == (1, 11, 24, 60, 90)

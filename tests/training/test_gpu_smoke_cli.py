import importlib.util
from pathlib import Path
import sys


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

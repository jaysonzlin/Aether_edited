from pathlib import Path


def test_stage2_launcher_uses_python_module_for_accelerate_cli():
    launcher = Path("submit_fixed_view_simgen_stage2_4gpu_mamba.sh").read_text()

    assert '"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch' in launcher

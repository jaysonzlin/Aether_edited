from pathlib import Path


def test_stage2_launcher_uses_python_module_for_accelerate_cli():
    launcher = Path("submit_fixed_view_simgen_stage2_4gpu_mamba.sh").read_text()

    assert '"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch' in launcher


def test_stage2_launcher_requests_email_on_lifecycle_events():
    launcher = Path("submit_fixed_view_simgen_stage2_4gpu_mamba.sh").read_text()

    assert "#SBATCH --mail-user=jlin3@college.harvard.edu" in launcher
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in launcher


def test_stage2_launcher_has_stage1_operational_diagnostics():
    launcher = Path("submit_fixed_view_simgen_stage2_4gpu_mamba.sh").read_text()

    assert "#SBATCH --ntasks-per-node=1" in launcher
    assert "#SBATCH --open-mode=append" in launcher
    assert 'NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-aether-fixed-view-stage2-${SLURM_JOB_ID}/nccl.%h.%p.log"' in launcher
    assert 'mkdir -p logs outputs/fixed_view_simgen_stage2' in launcher
    assert '"${PYTHON_BIN}" -c' in launcher
    assert "import accelerate" in launcher
    assert "wandb" in launcher

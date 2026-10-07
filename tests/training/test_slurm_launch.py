from pathlib import Path


def test_four_h200_mamba_launcher_uses_local_models_and_automatic_resume():
    launcher = Path("submit_fixed_view_simgen_4gpu_mamba.sh")
    contents = launcher.read_text()

    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --gres=gpu:nvidia_h200:4" in contents
    assert "--time=24:00:00" in contents
    assert "MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env" in contents
    assert "--config_file configs/accelerate/h200_4gpu.yaml" in contents
    assert "--resume latest" in contents
    assert "aether_model_id=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/AetherV1" in contents
    assert "cogvideox_model_id=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/CogVideoX-5b-I2V" in contents


def test_accelerate_config_is_one_machine_four_gpu_bf16_ddp():
    contents = Path("configs/accelerate/h200_4gpu.yaml").read_text()

    assert "distributed_type: MULTI_GPU" in contents
    assert "mixed_precision: bf16" in contents
    assert "num_machines: 1" in contents
    assert "num_processes: 4" in contents

from pathlib import Path

import pytest

from aether.training.stage2_config import Stage2TrainingConfigError, load_stage2_training_config


def _write_config(path: Path, **overrides) -> None:
    values = {
        "data_root": "../simgen/runs/panda_ball_can",
        "sample_ids": list(range(128)),
        "num_frames": 41,
        "height": 480,
        "width": 720,
        "left_padding": 120,
        "history_slots": 4,
        "max_train_steps": 2500,
        "output_interval": 500,
        "mixed_precision": "bf16",
        "learning_rate": 1e-5,
        "onecycle_pct_start": 0.1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.1,
        "max_grad_norm": 2.0,
        "dataloader_num_workers": 4,
        "pin_memory": True,
        "report_to": "wandb",
        "wandb_project": "aether-fixed-view-simgen-stage2",
        "seed": 42,
        "output_dir": "outputs/fixed_view_simgen_stage2",
        "stage1_checkpoint": "outputs/fixed_view_simgen/checkpoint-010000",
        "aether_model_id": "models/AetherV1",
        "cogvideox_model_id": "models/CogVideoX-5b-I2V",
        "rgb_loss_weight": 1.0,
        "depth_loss_weight": 1.0,
        "pointmap_loss_weight": 1.0,
    }
    values.update(overrides)
    import yaml

    path.write_text(yaml.safe_dump(values))


def test_stage2_config_has_the_approved_fixed_values(tmp_path):
    path = tmp_path / "stage2.yaml"
    _write_config(path)

    config = load_stage2_training_config(path)

    assert config.max_train_steps == 2500
    assert config.output_interval == 500
    assert config.learning_rate == 1e-5
    assert config.onecycle_pct_start == 0.1
    assert config.stage1_checkpoint.endswith("checkpoint-010000")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_train_steps", 2499, "max_train_steps"),
        ("output_interval", 250, "output_interval"),
        ("onecycle_pct_start", 0.2, "onecycle_pct_start"),
        ("stage1_checkpoint", "outputs/fixed_view_simgen/checkpoint-009500", "checkpoint-010000"),
        ("rgb_loss_weight", 0.0, "loss weights"),
    ],
)
def test_stage2_config_rejects_values_outside_contract(tmp_path, field, value, message):
    path = tmp_path / "stage2.yaml"
    _write_config(path, **{field: value})

    with pytest.raises(Stage2TrainingConfigError, match=message):
        load_stage2_training_config(path)

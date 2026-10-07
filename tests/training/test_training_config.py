from pathlib import Path

import pytest

from aether.training.config import TrainingConfigError, load_training_config


CONFIG_PATH = Path("configs/train/fixed_view_simgen_4h200.yaml")


def test_first_run_config_has_the_agreed_fixed_view_constants():
    config = load_training_config(CONFIG_PATH)

    assert config.data_root == "../simgen/runs/panda_ball_can"
    assert config.num_frames == 41
    assert (config.height, config.width, config.left_padding) == (480, 720, 120)
    assert config.history_slots == 4
    assert config.max_train_steps == 10_000
    assert config.output_interval == 1_000
    assert config.mixed_precision == "bf16"
    assert config.sample_ids == tuple(range(128))
    assert config.cogvideox_model_id == "models/CogVideoX-5b-I2V"
    assert config.aether_model_id == "models/AetherV1"
    assert (config.adam_beta1, config.adam_beta2) == (0.9, 0.95)
    assert config.adam_epsilon == 1e-8
    assert config.weight_decay == 0.1
    assert config.warmup_steps == 200
    assert config.max_grad_norm == 2.0
    assert config.dataloader_num_workers == 4
    assert config.pin_memory is True
    assert config.report_to == "wandb"
    assert config.wandb_project == "aether-fixed-view-simgen"
    assert config.train_batch_size == 1


def test_config_allows_disabling_wandb_tracking():
    config = load_training_config(CONFIG_PATH, ["report_to=null"])

    assert config.report_to is None


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ("warmup_steps=-1", "warmup_steps"),
        ("max_grad_norm=0", "max_grad_norm"),
        ("dataloader_num_workers=-1", "dataloader_num_workers"),
        ("adam_beta1=1", "adam_beta1"),
        ("adam_beta2=-0.1", "adam_beta2"),
        ("adam_epsilon=0", "adam_epsilon"),
        ("adam_epsilon=not-a-number", "adam_epsilon"),
        ("weight_decay=-0.1", "weight_decay"),
        ("report_to=tensorboard", "report_to"),
    ],
)
def test_config_rejects_invalid_wan_training_feature_values(override, field):
    with pytest.raises(TrainingConfigError, match=field):
        load_training_config(CONFIG_PATH, [override])


def test_config_rejects_incompatible_temporal_or_padding_values(tmp_path):
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        CONFIG_PATH.read_text().replace("num_frames: 41", "num_frames: 49")
    )

    with pytest.raises(TrainingConfigError, match="num_frames"):
        load_training_config(invalid_config)

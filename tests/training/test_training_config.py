from pathlib import Path

import pytest

from aether.training.config import TrainingConfigError, load_training_config


CONFIG_PATH = Path("configs/train/fixed_view_simgen_4h200.yaml")


def test_first_run_config_has_the_agreed_fixed_view_constants():
    config = load_training_config(CONFIG_PATH)

    assert config.num_frames == 41
    assert (config.height, config.width, config.left_padding) == (480, 720, 120)
    assert config.history_slots == 4
    assert config.max_train_steps == 10_000
    assert config.output_interval == 1_000
    assert config.mixed_precision == "bf16"
    assert config.sample_ids == tuple(range(128))


def test_config_rejects_incompatible_temporal_or_padding_values(tmp_path):
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        CONFIG_PATH.read_text().replace("num_frames: 41", "num_frames: 49")
    )

    with pytest.raises(TrainingConfigError, match="num_frames"):
        load_training_config(invalid_config)

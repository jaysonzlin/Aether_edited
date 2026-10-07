from pathlib import Path
from types import SimpleNamespace

from aether.training.checkpointing import (
    latest_checkpoint,
    build_run_manifest,
    restore_checkpoint,
    save_checkpoint,
)

RUN_MANIFEST = {
    "schema_version": 2,
    "objective": {"name": "stage1_diffusion_prediction_mse", "prediction_type": "v_prediction"},
}


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
    checkpoint = save_checkpoint(
        accelerator, tmp_path, global_step=1_000, run_manifest=RUN_MANIFEST
    )

    restored_step = restore_checkpoint(
        accelerator, checkpoint, expected_manifest=RUN_MANIFEST
    )

    assert restored_step == 1_000
    assert accelerator.restored_path == checkpoint


def test_checkpoint_retention_keeps_the_most_recent_numbered_directories(tmp_path):
    accelerator = FakeAccelerator()
    for step in (1_000, 2_000, 3_000):
        save_checkpoint(
            accelerator, tmp_path, global_step=step, keep_last=2, run_manifest=RUN_MANIFEST
        )

    assert not (tmp_path / "checkpoint-001000").exists()
    assert (tmp_path / "checkpoint-002000").exists()
    assert (tmp_path / "checkpoint-003000").exists()


def test_latest_checkpoint_selects_the_highest_valid_step(tmp_path):
    accelerator = FakeAccelerator()
    save_checkpoint(accelerator, tmp_path, global_step=1_000, run_manifest=RUN_MANIFEST)
    save_checkpoint(accelerator, tmp_path, global_step=3_000, run_manifest=RUN_MANIFEST)
    (tmp_path / "checkpoint-999999").mkdir()

    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-003000"


def test_restore_rejects_manifest_mismatch_before_loading_accelerator_state(tmp_path):
    accelerator = FakeAccelerator()
    checkpoint = save_checkpoint(
        accelerator, tmp_path, global_step=1_000, run_manifest=RUN_MANIFEST
    )

    try:
        restore_checkpoint(
            accelerator,
            checkpoint,
            expected_manifest={**RUN_MANIFEST, "training": {"learning_rate": 0.1}},
        )
    except ValueError as error:
        assert "incompatible run manifest" in str(error)
    else:
        raise AssertionError("expected incompatible checkpoint manifest to be rejected")

    assert accelerator.restored_path is None


def test_restore_rejects_legacy_checkpoint_without_run_manifest(tmp_path):
    accelerator = FakeAccelerator()
    checkpoint = tmp_path / "checkpoint-001000"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text('{"global_step": 1000}\n')

    try:
        restore_checkpoint(accelerator, checkpoint, expected_manifest=RUN_MANIFEST)
    except ValueError as error:
        assert "run manifest missing" in str(error)
    else:
        raise AssertionError("expected legacy checkpoint to be rejected")

    assert accelerator.restored_path is None


def _write_manifest_fixture(root):
    aether = root / "AetherV1"
    cogvideox = root / "CogVideoX-5b-I2V"
    data = root / "simgen"
    (aether / "transformer").mkdir(parents=True)
    (aether / "transformer" / "config.json").write_text('{"hidden_size": 8}')
    (aether / "transformer" / "model.safetensors").write_bytes(b"weights")
    for component in ("tokenizer", "text_encoder", "vae", "scheduler"):
        component_dir = cogvideox / component
        component_dir.mkdir(parents=True)
        (component_dir / "config.json").write_text('{"component": true}')
        (component_dir / "model.safetensors").write_bytes(b"weights")
    view = data / "sample_0" / "view_0"
    view.mkdir(parents=True)
    (view / "00000000.png").write_bytes(b"rgb")
    (view / "depth.h5").write_bytes(b"depth")
    (view / "cameras.json").write_text("[]")
    return aether, cogvideox, data


def _manifest_config(paths):
    aether, cogvideox, data = paths
    return SimpleNamespace(
        aether_model_id=str(aether),
        cogvideox_model_id=str(cogvideox),
        data_root=str(data),
        sample_ids=(0,),
        num_frames=1,
        height=480,
        width=720,
        left_padding=120,
        history_slots=4,
        max_train_steps=10_000,
        output_interval=1_000,
        mixed_precision="bf16",
        learning_rate=1e-5,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        warmup_steps=200,
        max_grad_norm=2.0,
        dataloader_num_workers=4,
        pin_memory=True,
        report_to="wandb",
        wandb_project="aether-fixed-view-simgen",
        seed=42,
    )


def test_run_manifest_is_path_independent_but_captures_prediction_type(tmp_path):
    config_one = _manifest_config(_write_manifest_fixture(tmp_path / "one"))
    config_two = _manifest_config(_write_manifest_fixture(tmp_path / "two"))

    manifest_one = build_run_manifest(config_one, "v_prediction")
    manifest_two = build_run_manifest(config_two, "v_prediction")
    manifest_with_progress = build_run_manifest(config_two, "v_prediction", show_progress=True)
    epsilon_manifest = build_run_manifest(config_two, "epsilon")

    assert manifest_one == manifest_two
    assert manifest_with_progress == manifest_two
    assert manifest_one["objective"]["prediction_type"] == "v_prediction"
    assert manifest_one["schema_version"] == 2
    assert manifest_one["training"]["warmup_steps"] == 200
    assert epsilon_manifest != manifest_two


def test_run_manifest_detects_same_size_model_weight_replacement(tmp_path):
    paths = _write_manifest_fixture(tmp_path / "run")
    config = _manifest_config(paths)
    before = build_run_manifest(config, "v_prediction")

    (paths[0] / "transformer" / "model.safetensors").write_bytes(b"changed")
    after = build_run_manifest(config, "v_prediction")

    assert before != after

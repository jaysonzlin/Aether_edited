import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from aether.training.checkpointing import (
    latest_checkpoint,
    build_run_manifest,
    restore_checkpoint,
    restore_latest_checkpoint_with_fallback,
    save_checkpoint,
)


def test_stage1_transformer_loader_requires_completed_checkpoint_and_loads_weights(tmp_path):
    import aether.training.checkpointing as checkpointing

    checkpoint = tmp_path / "checkpoint-010000"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text('{"global_step": 10000, "run_manifest": {}}')
    (checkpoint / "model.safetensors").touch()

    class Target:
        def __init__(self):
            self.received = None

        def load_state_dict(self, state_dict, strict):
            self.received = (state_dict, strict)
            return [], []

    target = Target()
    checkpointing._load_safetensor_state = lambda _: {"weight": "source"}

    checkpointing.load_stage1_transformer_weights(target, checkpoint)

    assert target.received == ({"weight": "source"}, True)


def test_stage1_transformer_loader_rejects_incomplete_or_wrong_step_checkpoint(tmp_path):
    from aether.training.checkpointing import load_stage1_transformer_weights

    checkpoint = tmp_path / "checkpoint-010000"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text('{"global_step": 9999, "run_manifest": {}}')
    (checkpoint / "model.safetensors").touch()

    with pytest.raises(ValueError, match="10000"):
        load_stage1_transformer_weights(object(), checkpoint)

RUN_MANIFEST = {
    "schema_version": 2,
    "objective": {"name": "stage1_diffusion_prediction_mse", "prediction_type": "v_prediction"},
}


class FakeAccelerator:
    def __init__(self):
        self.restored_path = None
        self.load_attempts = []
        self.failing_checkpoints = set()
        self.process_index = 0
        self.num_processes = 1

    def save_state(self, path):
        path = Path(path)
        path.mkdir()
        (path / "state.txt").write_text("state")

    def load_state(self, path):
        path = Path(path)
        self.load_attempts.append(path)
        if path.name in self.failing_checkpoints:
            raise RuntimeError(f"cannot load {path.name}")
        self.restored_path = path


def _write_complete_accelerate_state(checkpoint, num_processes=1):
    for filename in ("model.safetensors", "optimizer.bin", "scheduler.bin"):
        (checkpoint / filename).touch()
    for process_index in range(num_processes):
        (checkpoint / f"random_states_{process_index}.pkl").touch()


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


def test_save_checkpoint_replaces_an_incomplete_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint-000500"
    checkpoint.mkdir()
    (checkpoint / "partial-state.txt").write_text("interrupted")

    saved_checkpoint = save_checkpoint(
        FakeAccelerator(), tmp_path, global_step=500, run_manifest=RUN_MANIFEST
    )

    assert saved_checkpoint == checkpoint
    assert (checkpoint / "state.txt").read_text() == "state"
    assert not (checkpoint / "partial-state.txt").exists()


def test_resume_latest_skips_checkpoint_missing_required_accelerate_state(tmp_path):
    accelerator = FakeAccelerator()
    checkpoint = save_checkpoint(
        accelerator, tmp_path, global_step=500, run_manifest=RUN_MANIFEST
    )
    _write_complete_accelerate_state(checkpoint)
    (tmp_path / "checkpoint-001000").mkdir()

    resumed_checkpoint, step = restore_latest_checkpoint_with_fallback(
        accelerator, tmp_path, RUN_MANIFEST
    )

    assert resumed_checkpoint == checkpoint
    assert step == 500
    assert accelerator.load_attempts == [checkpoint]


def test_resume_latest_falls_back_when_the_newest_complete_state_cannot_load(tmp_path):
    accelerator = FakeAccelerator()
    checkpoint_500 = save_checkpoint(
        accelerator, tmp_path, global_step=500, run_manifest=RUN_MANIFEST
    )
    checkpoint_1000 = save_checkpoint(
        accelerator, tmp_path, global_step=1_000, run_manifest=RUN_MANIFEST
    )
    _write_complete_accelerate_state(checkpoint_500)
    _write_complete_accelerate_state(checkpoint_1000)
    accelerator.failing_checkpoints.add(checkpoint_1000.name)

    resumed_checkpoint, step = restore_latest_checkpoint_with_fallback(
        accelerator, tmp_path, RUN_MANIFEST
    )

    assert resumed_checkpoint == checkpoint_500
    assert step == 500
    assert accelerator.load_attempts == [checkpoint_1000, checkpoint_500]


def test_resume_latest_starts_fresh_when_no_checkpoint_can_be_restored(tmp_path):
    accelerator = FakeAccelerator()
    checkpoint = save_checkpoint(
        accelerator, tmp_path, global_step=500, run_manifest=RUN_MANIFEST
    )
    _write_complete_accelerate_state(checkpoint)
    accelerator.failing_checkpoints.add(checkpoint.name)

    resumed_checkpoint, step = restore_latest_checkpoint_with_fallback(
        accelerator, tmp_path, RUN_MANIFEST
    )

    assert resumed_checkpoint is None
    assert step == 0
    assert accelerator.load_attempts == [checkpoint]


def test_resume_latest_requires_random_state_for_every_rank(tmp_path):
    accelerator = FakeAccelerator()
    accelerator.num_processes = 4
    checkpoint_500 = save_checkpoint(
        accelerator, tmp_path, global_step=500, run_manifest=RUN_MANIFEST
    )
    checkpoint_1000 = save_checkpoint(
        accelerator, tmp_path, global_step=1_000, run_manifest=RUN_MANIFEST
    )
    _write_complete_accelerate_state(checkpoint_500, num_processes=4)
    _write_complete_accelerate_state(checkpoint_1000, num_processes=3)

    resumed_checkpoint, step = restore_latest_checkpoint_with_fallback(
        accelerator, tmp_path, RUN_MANIFEST
    )

    assert resumed_checkpoint == checkpoint_500
    assert step == 500
    assert accelerator.load_attempts == [checkpoint_500]


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


def test_run_manifest_reuses_cached_hashes_when_file_metadata_matches(tmp_path, monkeypatch):
    import aether.training.checkpointing as checkpointing

    config = _manifest_config(_write_manifest_fixture(tmp_path / "run"))
    cache_path = tmp_path / "output" / "input_fingerprint_cache.json"
    first = build_run_manifest(
        config, "v_prediction", fingerprint_cache_path=cache_path
    )

    def unexpected_hash(_path):
        raise AssertionError("unchanged inputs should reuse their cached SHA-256")

    monkeypatch.setattr(checkpointing, "_file_sha256", unexpected_hash)
    second = build_run_manifest(
        config, "v_prediction", fingerprint_cache_path=cache_path
    )

    assert second == first


def test_run_manifest_rehashes_only_files_with_changed_metadata(tmp_path, monkeypatch):
    import aether.training.checkpointing as checkpointing

    paths = _write_manifest_fixture(tmp_path / "run")
    config = _manifest_config(paths)
    cache_path = tmp_path / "output" / "input_fingerprint_cache.json"
    before = build_run_manifest(
        config, "v_prediction", fingerprint_cache_path=cache_path
    )
    changed_file = paths[0] / "transformer" / "model.safetensors"
    old_stat = changed_file.stat()
    changed_file.write_bytes(b"changed")
    os.utime(
        changed_file,
        ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 2_000_000_000),
    )

    real_hash = checkpointing._file_sha256
    hashed_paths = []

    def record_hash(path):
        hashed_paths.append(Path(path).resolve())
        return real_hash(path)

    monkeypatch.setattr(checkpointing, "_file_sha256", record_hash)
    after = build_run_manifest(
        config, "v_prediction", fingerprint_cache_path=cache_path
    )

    assert after != before
    assert hashed_paths == [changed_file.resolve()]


def test_force_rehash_bypasses_cached_hashes(tmp_path, monkeypatch):
    import aether.training.checkpointing as checkpointing

    config = _manifest_config(_write_manifest_fixture(tmp_path / "run"))
    cache_path = tmp_path / "output" / "input_fingerprint_cache.json"
    build_run_manifest(config, "v_prediction", fingerprint_cache_path=cache_path)

    real_hash = checkpointing._file_sha256
    hashed_paths = []

    def record_hash(path):
        hashed_paths.append(Path(path).resolve())
        return real_hash(path)

    monkeypatch.setattr(checkpointing, "_file_sha256", record_hash)
    build_run_manifest(
        config,
        "v_prediction",
        fingerprint_cache_path=cache_path,
        force_rehash=True,
    )

    assert len(hashed_paths) == 13

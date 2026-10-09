import pytest
from types import SimpleNamespace

import scripts.train_fixed_view_simgen_stage2 as stage2

from scripts.train_fixed_view_simgen_stage2 import (
    enable_transformer_gradient_checkpointing,
    enable_vae_gradient_checkpointing,
    parse_args,
    smoke_result_message,
    smoke_trace_message,
)


class RecordingAccelerator:
    def __init__(self):
        self.project = None
        self.metadata = None
        self.finished = False

    def init_trackers(self, project, config):
        self.project = project
        self.metadata = config

    def end_training(self):
        self.finished = True


def _config(report_to="wandb"):
    return SimpleNamespace(
        report_to=report_to,
        wandb_project="aether-fixed-view-simgen-stage2",
        learning_rate=1e-5,
        max_train_steps=2500,
        onecycle_pct_start=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        max_grad_norm=2.0,
        output_interval=500,
        stage1_checkpoint="outputs/fixed_view_simgen/checkpoint-010000",
        rgb_loss_weight=1.0,
        depth_loss_weight=1.0,
        pointmap_loss_weight=1.0,
    )


def test_stage2_tracking_helpers_exist():
    assert hasattr(stage2, "initialize_stage2_tracking")
    assert hasattr(stage2, "finish_stage2_tracking")
    assert hasattr(stage2, "stage2_tracker_config")
    assert hasattr(stage2, "stage2_metric_values")
    assert hasattr(stage2, "stage2_calibration_metric_values")


def test_initialize_stage2_tracking_uses_project_and_stage2_metadata():
    accelerator = RecordingAccelerator()

    initialized = stage2.initialize_stage2_tracking(accelerator, _config())

    assert initialized is True
    assert accelerator.project == "aether-fixed-view-simgen-stage2"
    assert accelerator.metadata["onecycle_pct_start"] == 0.1
    assert accelerator.metadata["rgb_loss_weight"] == 1.0


def test_initialize_stage2_tracking_skips_disabled_reporting():
    accelerator = RecordingAccelerator()

    initialized = stage2.initialize_stage2_tracking(accelerator, _config(report_to=None))

    assert initialized is False
    assert accelerator.project is None


def test_stage2_metric_values_report_raw_losses_and_optimizer_health():
    losses = SimpleNamespace(rgb=2.0, depth=3.0, pointmap=4.0)

    metrics = stage2.stage2_metric_values(
        total=10.0,
        mse=1.0,
        losses=losses,
        learning_rate=5e-6,
        grad_norm=0.25,
    )

    assert metrics == {
        "train/loss": 10.0,
        "train/mse": 1.0,
        "train/rgb_ms_ssim": 2.0,
        "train/depth_ssi": 3.0,
        "train/pointmap": 4.0,
        "train/learning_rate": 5e-6,
        "train/grad_norm": 0.25,
    }


def test_stage2_calibration_metric_values_report_effective_weights():
    metrics = stage2.stage2_calibration_metric_values(
        {"rgb": 2.0, "depth": 3.0, "pointmap": 4.0}
    )

    assert metrics == {
        "train/calibrated_rgb_weight": 2.0,
        "train/calibrated_depth_weight": 3.0,
        "train/calibrated_pointmap_weight": 4.0,
    }


def test_finish_stage2_tracking_only_closes_initialized_tracker_and_preserves_errors():
    uninitialized = RecordingAccelerator()
    stage2.finish_stage2_tracking(uninitialized, initialized=False)
    assert uninitialized.finished is False

    initialized = RecordingAccelerator()
    with pytest.raises(RuntimeError, match="training failed"):
        try:
            raise RuntimeError("training failed")
        finally:
            stage2.finish_stage2_tracking(initialized, initialized=True)
    assert initialized.finished is True


def test_stage2_rollout_uses_keyword_only_helper_contract():
    received = {}

    def save_rollout_artifacts(*, accelerator, config, dataset, pipeline, transformer, vae, scheduler, prompt_embeds, global_step):
        received.update(
            accelerator=accelerator,
            config=config,
            dataset=dataset,
            pipeline=pipeline,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            global_step=global_step,
        )
        return ["rollout.mp4"]

    artifacts = stage2.save_stage2_rollout(
        save_rollout_artifacts,
        accelerator="accelerator",
        config="config",
        dataset="dataset",
        pipeline="pipeline",
        transformer="transformer",
        vae="vae",
        scheduler="scheduler",
        prompts="prompts",
        step=500,
    )

    assert artifacts == ["rollout.mp4"]
    assert received == {
        "accelerator": "accelerator",
        "config": "config",
        "dataset": "dataset",
        "pipeline": "pipeline",
        "transformer": "transformer",
        "vae": "vae",
        "scheduler": "scheduler",
        "prompt_embeds": "prompts",
        "global_step": 500,
    }


def test_smoke_result_message_reports_completed_update_and_loss():
    assert smoke_result_message(1, 2.5) == "Stage-2 GPU smoke test passed: completed 1 optimizer update (loss=2.500000)"


def test_smoke_trace_message_labels_execution_boundary():
    assert smoke_trace_message("components loaded") == "Stage-2 GPU smoke trace: components loaded"


def test_gpu_smoke_test_accepts_a_two_update_limit():
    args = parse_args(["--gpu-smoke-test", "--gpu-smoke-test-steps", "2"])

    assert args.gpu_smoke_test_steps == 2


def test_gpu_smoke_step_limit_requires_gpu_smoke_mode():
    with pytest.raises(SystemExit):
        parse_args(["--gpu-smoke-test-steps", "2"])


def test_stage2_enables_transformer_gradient_checkpointing_when_supported():
    class Transformer:
        def __init__(self):
            self.enabled = False

        def enable_gradient_checkpointing(self):
            self.enabled = True

    transformer = Transformer()

    enable_transformer_gradient_checkpointing(transformer)

    assert transformer.enabled


def test_stage2_enables_frozen_vae_gradient_checkpointing_when_supported():
    class Vae:
        def __init__(self):
            self.enabled = False

        def enable_gradient_checkpointing(self):
            self.enabled = True

    vae = Vae()

    enable_vae_gradient_checkpointing(vae)

    assert vae.enabled

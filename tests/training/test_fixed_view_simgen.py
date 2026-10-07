from types import SimpleNamespace

import pytest

from scripts.train_fixed_view_simgen import (
    build_optimizer,
    diffusion_training_target,
    optimizer_update,
)


class FakeScheduler:
    def __init__(self, prediction_type):
        self.config = SimpleNamespace(prediction_type=prediction_type)
        self.velocity_args = None

    def get_velocity(self, latents, noise, timesteps):
        self.velocity_args = (latents, noise, timesteps)
        return "velocity-target"


def test_v_prediction_uses_scheduler_velocity_target():
    scheduler = FakeScheduler("v_prediction")
    target_latents, noise, timesteps = object(), object(), object()

    result = diffusion_training_target(scheduler, target_latents, noise, timesteps)

    assert result == "velocity-target"
    assert scheduler.velocity_args == (target_latents, noise, timesteps)


def test_epsilon_prediction_uses_sampled_noise_target():
    scheduler = FakeScheduler("epsilon")
    noise = object()

    result = diffusion_training_target(scheduler, object(), noise, object())

    assert result is noise
    assert scheduler.velocity_args is None


def test_unsupported_scheduler_prediction_type_fails_explicitly():
    scheduler = FakeScheduler("sample")

    with pytest.raises(ValueError, match="unsupported scheduler prediction_type.*sample"):
        diffusion_training_target(scheduler, object(), object(), object())


def test_run_manifest_is_computed_once_on_rank_zero_and_broadcast(monkeypatch):
    import aether.training.checkpointing as checkpointing
    from scripts.train_fixed_view_simgen import shared_run_manifest

    expected_manifest = {"schema_version": 2}
    calls = []
    monkeypatch.setattr(
        checkpointing,
        "build_run_manifest",
        lambda config, prediction_type, show_progress=False: calls.append((config, prediction_type, show_progress))
        or expected_manifest,
    )

    class Distributed:
        @staticmethod
        def broadcast_object_list(values, src):
            assert src == 0
            values[0] = expected_manifest

    accelerator = SimpleNamespace(is_main_process=False, num_processes=4)
    result = shared_run_manifest(
        object(), "v_prediction", accelerator, SimpleNamespace(distributed=Distributed())
    )

    assert result == expected_manifest
    assert calls == []


def test_adamw_uses_wan_video_hyperparameters():
    import torch

    parameter = torch.nn.Parameter(torch.ones(()))
    transformer = SimpleNamespace(parameters=lambda: iter((parameter,)))
    config = SimpleNamespace(
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
    )

    optimizer = build_optimizer(transformer, config, torch)
    group = optimizer.param_groups[0]

    assert group["lr"] == 1e-5
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.1


def test_constant_schedule_warms_up_then_stays_at_configured_learning_rate(monkeypatch):
    import torch
    import sys
    from scripts.train_fixed_view_simgen import build_lr_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1e-5)
    calls = []

    def constant_warmup(optimizer, num_warmup_steps):
        calls.append(num_warmup_steps)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / num_warmup_steps),
        )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(get_constant_schedule_with_warmup=constant_warmup),
    )
    scheduler = build_lr_scheduler(optimizer, SimpleNamespace(warmup_steps=2))

    assert scheduler.get_last_lr() == [0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [5e-6]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [1e-5]
    assert calls == [2]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [1e-5]


def test_accelerator_does_not_auto_advance_prepared_scheduler_per_ddp_rank():
    from scripts.train_fixed_view_simgen import accelerator_options

    options = accelerator_options(SimpleNamespace(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        report_to="wandb",
    ))

    assert options["step_scheduler_with_optimizer"] is False


@pytest.mark.parametrize("sync_gradients", [False, True])
def test_optimizer_update_clips_only_at_synchronized_update(sync_gradients):
    events = []

    class FakeAccelerator:
        def __init__(self):
            self.sync_gradients = sync_gradients

        def clip_grad_norm_(self, parameters, maximum):
            events.append(("clip", maximum))
            return 2.5

    class FakeOptimizer:
        def step(self):
            events.append(("optimizer",))

        def zero_grad(self, set_to_none):
            events.append(("zero", set_to_none))

    class FakeScheduler:
        def step(self):
            events.append(("scheduler",))

    grad_norm = optimizer_update(
        FakeAccelerator(), FakeOptimizer(), FakeScheduler(),
        SimpleNamespace(parameters=lambda: ()), 2.0
    )

    expected = ([('clip', 2.0), ('optimizer',), ('scheduler',), ('zero', True)]
                if sync_gradients else [('optimizer',), ('zero', True)])
    assert events == expected
    assert grad_norm == (2.5 if sync_gradients else None)

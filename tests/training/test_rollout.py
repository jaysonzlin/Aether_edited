import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from aether.training.rollout import (
    composite_history,
    dynamic_guidance_scale,
    require_41_frames,
    sample_aether_latents,
    unconditional_conditions,
)


class FakeScheduler:
    """Tiny DPM-compatible scheduler that records sampler inputs."""

    init_noise_sigma = 1.0

    def __init__(self):
        self.timesteps = torch.tensor([99])
        self.calls = []

    def set_timesteps(self, steps, device):
        self.timesteps = torch.tensor([2, 0], device=device)
        self.requested_steps = steps

    def scale_model_input(self, latents, timestep):
        return latents

    def step(self, prediction, old_prediction, timestep, previous_timestep, latents, return_dict):
        self.calls.append((prediction.clone(), timestep, previous_timestep, return_dict))
        return latents - prediction, prediction


class FakeTransformer:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch, frames, _, height, width = kwargs["hidden_states"].shape
        prediction = torch.zeros((batch, frames, 56, height, width))
        prediction[batch // 2 :] = 1
        return (prediction,)


def test_composite_history_replaces_the_observed_raw_frame_prefix():
    generated = torch.zeros((1, 41, 3, 2, 2))
    observed = torch.ones((1, 41, 3, 2, 2))

    composite = composite_history(generated, observed, history_frames=13)

    assert torch.equal(composite[:, :13], observed[:, :13])
    assert torch.count_nonzero(composite[:, 13:]) == 0


def test_unconditional_conditions_zero_all_four_rgb_history_slots_only():
    conditions = torch.ones((1, 11, 40, 60, 90))

    unconditional = unconditional_conditions(conditions, history_slots=4)

    assert torch.count_nonzero(unconditional[:, :4, :16]) == 0
    assert torch.equal(unconditional[:, 4:, :16], conditions[:, 4:, :16])
    assert torch.equal(unconditional[:, :, 16:], conditions[:, :, 16:])


def test_dynamic_guidance_matches_aether_prediction_schedule():
    assert dynamic_guidance_scale(3.0, timestep=50, num_inference_steps=50) == 1.0
    assert dynamic_guidance_scale(3.0, timestep=0, num_inference_steps=50) == 4.0


def test_rollout_requires_exactly_41_decoded_frames():
    require_41_frames(torch.zeros((1, 41, 3, 2, 2)))
    try:
        require_41_frames(torch.zeros((1, 40, 3, 2, 2)))
    except ValueError as error:
        assert "41" in str(error)
    else:
        raise AssertionError("expected a non-41 decoded clip to fail")


def test_sampler_clones_the_scheduler_and_uses_seeded_aether_cfg_inputs():
    scheduler = FakeScheduler()
    transformer = FakeTransformer()
    conditions = torch.ones((1, 11, 40, 60, 90))
    prompt_embeds = torch.zeros((1, 2, 3))

    first = sample_aether_latents(
        transformer,
        scheduler,
        conditions,
        prompt_embeds,
        rotary_emb=None,
        ofs=None,
        seed=42,
        num_inference_steps=2,
    )
    second = sample_aether_latents(
        FakeTransformer(),
        scheduler,
        conditions,
        prompt_embeds,
        rotary_emb=None,
        ofs=None,
        seed=42,
        num_inference_steps=2,
    )

    assert torch.equal(first, second)
    assert torch.equal(scheduler.timesteps, torch.tensor([99]))
    call = transformer.calls[0]
    assert call["hidden_states"].shape == (2, 11, 96, 60, 90)
    assert call["timestep"].shape == (2,)
    unconditional = call["hidden_states"][0, :, 56:]
    conditional = call["hidden_states"][1, :, 56:]
    assert torch.count_nonzero(unconditional[:4, :16]) == 0
    assert torch.equal(unconditional[:, 16:], conditional[:, 16:])
    assert torch.equal(conditional, conditions[0])

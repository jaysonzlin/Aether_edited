"""Four-history Aether rollout helpers used for fixed checkpoint previews."""

from __future__ import annotations

import copy
import math

import torch


def unconditional_conditions(
    condition_latents: torch.Tensor, history_slots: int = 4
) -> torch.Tensor:
    """Remove RGB history for CFG while preserving RGB padding and raymaps."""
    unconditional = condition_latents.clone()
    unconditional[:, :history_slots, :16] = 0
    return unconditional


def dynamic_guidance_scale(
    guidance_scale: float, timestep: int, num_inference_steps: int
) -> float:
    """Aether prediction's time-varying classifier-free guidance scale."""
    progress = (num_inference_steps - timestep) / num_inference_steps
    return 1 + guidance_scale * (1 - math.cos(math.pi * progress**5)) / 2


def require_41_frames(rgb_clip: torch.Tensor) -> None:
    if rgb_clip.ndim != 5 or rgb_clip.shape[1] != 41:
        raise ValueError("decoded rollout must contain exactly 41 frames")


def composite_history(
    generated_rgb: torch.Tensor,
    observed_rgb: torch.Tensor,
    history_frames: int = 13,
) -> torch.Tensor:
    """Replace generated prefix frames with the observed four-latent history."""
    if generated_rgb.shape != observed_rgb.shape:
        raise ValueError("generated and observed RGB clips must have matching shapes")
    if generated_rgb.ndim != 5:
        raise ValueError("RGB clips must have shape [batch, frames, channels, height, width]")
    if not 0 < history_frames <= generated_rgb.shape[1]:
        raise ValueError("history_frames must be within the RGB clip length")
    composite = generated_rgb.clone()
    composite[:, :history_frames] = observed_rgb[:, :history_frames]
    return composite


def _validate_rollout_latents(condition_latents: torch.Tensor) -> None:
    if condition_latents.ndim != 5:
        raise ValueError("condition latents must have shape [batch, frames, channels, height, width]")
    if condition_latents.shape[1:] != (11, 40, 60, 90):
        raise ValueError("condition latents must have shape [batch, 11, 40, 60, 90]")


@torch.no_grad()
def sample_aether_latents(
    transformer,
    scheduler_template,
    condition_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    rotary_emb,
    ofs: torch.Tensor | None,
    seed: int = 42,
    num_inference_steps: int = 50,
    show_progress: bool = False,
) -> torch.Tensor:
    """Denoise a fixed-seed Aether target state with four-history conditions.

    The scheduler is copied before calling ``set_timesteps`` so a checkpoint
    preview cannot change the scheduler used by the next training batch.
    """
    _validate_rollout_latents(condition_latents)
    scheduler = copy.deepcopy(scheduler_template)
    generator = torch.Generator(device=condition_latents.device).manual_seed(seed)
    latents = torch.randn(
        condition_latents.shape[0], condition_latents.shape[1], 56,
        condition_latents.shape[3], condition_latents.shape[4],
        device=condition_latents.device, dtype=condition_latents.dtype, generator=generator,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    old_prediction = None
    from tqdm.auto import tqdm
    timesteps = tqdm(
        scheduler.timesteps,
        desc="Aether rollout",
        unit="step",
        disable=not show_progress,
    )
    for index, timestep in enumerate(timesteps):
        model_input = scheduler.scale_model_input(torch.cat((latents, latents)), timestep)
        latent_conditions = torch.cat(
            (unconditional_conditions(condition_latents), condition_latents)
        )
        model_input = torch.cat((model_input, latent_conditions), dim=2)
        prediction = transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds.repeat(model_input.shape[0], 1, 1),
            timestep=timestep.expand(model_input.shape[0]),
            ofs=ofs,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0].float()
        unconditioned, conditioned = prediction.chunk(2)
        prediction = unconditioned + dynamic_guidance_scale(
            3.0, int(timestep.item()), num_inference_steps
        ) * (conditioned - unconditioned)
        latents, old_prediction = scheduler.step(
            prediction, old_prediction, timestep,
            scheduler.timesteps[index - 1] if index else None,
            latents, return_dict=False,
        )
        latents = latents.to(prompt_embeds.dtype)
    return latents

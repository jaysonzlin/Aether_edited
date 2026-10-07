"""Frozen-VAE latent assembly for fixed-view Aether diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


RGB_LATENT_CHANNELS = 16
RAYMAP_CHANNELS = 24
HISTORY_SLOTS = 4


class VAEEncoder(Protocol):
    """The minimal CogVideoX VAE interface used by latent assembly."""

    config: Any
    dtype: torch.dtype

    def encode(self, video: torch.Tensor) -> Any: ...


@dataclass(frozen=True)
class AetherTrainingBatch:
    """Aether target and condition states for one batch of 41-frame clips."""

    target_latents: torch.Tensor
    condition_latents: torch.Tensor
    history_positions: tuple[int, ...]


def _retrieve_latents(encoder_output: Any) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("VAE encoder output must provide latent_dist or latents")


@torch.no_grad()
def _encode_video(vae: VAEEncoder, video: torch.Tensor) -> torch.Tensor:
    """Encode ``[batch, frames, channels, height, width]`` into Aether layout."""
    video = video.to(dtype=vae.dtype)
    encoded = _retrieve_latents(vae.encode(video.permute(0, 2, 1, 3, 4)))
    if encoded.ndim != 5:
        raise ValueError("VAE latents must have shape [batch, channels, frames, height, width]")
    return encoded.permute(0, 2, 1, 3, 4)


def _apply_vae_scaling(vae: VAEEncoder, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    if getattr(vae.config, "invert_scale_latents", False):
        return latents / scaling_factor
    return latents * scaling_factor


def assemble_aether_training_batch(
    batch: dict[str, torch.Tensor],
    vae: VAEEncoder,
    history_slots: int = HISTORY_SLOTS,
) -> AetherTrainingBatch:
    """Encode RGB-D and combine it with packed raymaps in Aether channel order.

    ``rgb`` is supplied by the dataset in ``[0, 1]`` and is converted to the
    VAE's ``[-1, 1]`` range.  The dataset's disparity is already in ``[-1, 1]``.
    """
    rgb = batch["rgb"]
    disparity = batch["disparity"]
    raymap = batch["raymap"]
    if not all(isinstance(value, torch.Tensor) for value in (rgb, disparity, raymap)):
        raise TypeError("rgb, disparity, and raymap must be torch tensors")
    if rgb.shape != disparity.shape or rgb.ndim != 5 or rgb.shape[2:] != (3, 480, 720):
        raise ValueError("rgb and disparity must have shape [batch, 41, 3, 480, 720]")
    if rgb.shape[1] != 41:
        raise ValueError("fixed-view training requires exactly 41 RGB-D frames")

    rgb_latents = _apply_vae_scaling(vae, _encode_video(vae, rgb * 2.0 - 1.0))
    disparity_latents = _apply_vae_scaling(vae, _encode_video(vae, disparity))
    if rgb_latents.shape != disparity_latents.shape:
        raise ValueError("RGB and disparity VAE latent shapes must match")
    if rgb_latents.shape[2] != RGB_LATENT_CHANNELS:
        raise ValueError("Aether requires 16-channel CogVideoX VAE latents")
    if raymap.shape != (
        rgb.shape[0],
        rgb_latents.shape[1],
        RAYMAP_CHANNELS,
        rgb_latents.shape[3],
        rgb_latents.shape[4],
    ):
        raise ValueError("packed raymap shape must match the VAE latent time and spatial dimensions")
    if not 0 < history_slots <= rgb_latents.shape[1]:
        raise ValueError("history_slots must be between one and the latent sequence length")

    raymap = raymap.to(device=rgb_latents.device, dtype=rgb_latents.dtype)
    target_latents = torch.cat((rgb_latents, disparity_latents, raymap), dim=2)
    rgb_conditions = torch.zeros_like(rgb_latents)
    rgb_conditions[:, :history_slots] = rgb_latents[:, :history_slots]
    condition_latents = torch.cat((rgb_conditions, raymap), dim=2)
    return AetherTrainingBatch(
        target_latents=target_latents,
        condition_latents=condition_latents,
        history_positions=tuple(range(history_slots)),
    )

"""Four-history Aether rollout helpers used for fixed checkpoint previews."""

from __future__ import annotations

import torch


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

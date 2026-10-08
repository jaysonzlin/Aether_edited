"""Image-space losses for fixed-view Aether Stage-2 refinement."""

from __future__ import annotations

from dataclasses import dataclass


# The 41-frame clip supports three 11x11x11 SSIM pyramid levels. The default
# five levels reduce time to five frames, where reflected temporal padding fails.
MS_SSIM_41_FRAME_BETAS = (0.0448, 0.2856, 0.6696)


def reconstruct_clean_latents(scheduler, noisy_latents, prediction, timesteps):
    """Recover x0 from an epsilon or velocity prediction without scheduler.step."""
    import torch

    alphas = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha = alphas[timesteps].view(-1, 1, 1, 1, 1).sqrt()
    sigma = (1 - alphas[timesteps]).view(-1, 1, 1, 1, 1).sqrt()
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (noisy_latents - sigma * prediction) / alpha.clamp_min(torch.finfo(alpha.dtype).eps)
    if prediction_type == "v_prediction":
        return alpha * noisy_latents - sigma * prediction
    raise ValueError(f"unsupported scheduler prediction_type {prediction_type!r}")


def decoded_disparity(decoded):
    """Return Aether's normalized relative disparity from decoded VAE output."""
    return ((decoded.mean(dim=1) * 0.5 + 0.5).clamp(0, 1)).square()


def unpack_raymaps(latents, frame_count=41):
    """Turn [B, 11, 24, H, W] packed Aether raymaps into 41 six-channel frames."""
    batch, _, channels, height, width = latents.shape
    if channels != 24:
        raise ValueError("raymap latents must have 24 channels")
    raymaps = latents.reshape(batch, -1, 4, 6, height, width).reshape(batch, -1, 6, height, width)
    if raymaps.shape[1] < frame_count:
        raise ValueError("packed raymaps do not contain enough frames")
    return raymaps[:, -frame_count:]


def _content(video, left=120, right=600):
    return video[..., left:right]


def depth_ssi_loss(predicted, target):
    """Per-video scale/shift aligned L1 data plus equal gradient residuals."""
    import torch

    predicted, target = predicted.float(), target.float()
    p, t = predicted.flatten(1), target.flatten(1)
    one = torch.ones_like(p)
    a00, a01, a11 = (p * p).sum(1), p.sum(1), one.sum(1)
    b0, b1 = (p * t).sum(1), t.sum(1)
    determinant = (a00 * a11 - a01.square()).clamp_min(1e-6)
    scale = ((a11 * b0 - a01 * b1) / determinant).view(-1, 1, 1, 1)
    shift = ((a00 * b1 - a01 * b0) / determinant).view(-1, 1, 1, 1)
    aligned = scale * predicted + shift
    data = (aligned - target).abs().mean()
    gradient = (aligned[..., 1:, :] - aligned[..., :-1, :] - (target[..., 1:, :] - target[..., :-1, :])).abs().mean()
    gradient = gradient + (aligned[..., :, 1:] - aligned[..., :, :-1] - (target[..., :, 1:] - target[..., :, :-1])).abs().mean()
    return data + gradient


@dataclass(frozen=True)
class Stage2Losses:
    rgb: object
    depth: object
    pointmap: object


def calibrate_auxiliary_weights(mse, losses, configured):
    """Scale nonzero auxiliary losses to the initial detached MSE magnitude."""
    target = float(mse.detach().float().item())
    result = {}
    for name, loss in losses.items():
        raw = float(loss.detach().float().item())
        if raw <= 0:
            raise ValueError(f"cannot calibrate zero {name} loss")
        result[name] = target * float(configured[name]) / raw
    return result


def _decode(vae, latents):
    """Decode [B,T,C,H,W] latent video while retaining gradients to latents."""
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    decoder_input = latents / scaling if not getattr(vae.config, "invert_scale_latents", False) else latents * scaling
    return vae.decode(decoder_input.to(dtype=vae.dtype).permute(0, 2, 1, 3, 4)).sample


def _pointmap_loss(predicted_disparity, predicted_raymaps, target_disparity, target_raymaps):
    import torch
    import torch.nn.functional as functional

    batch, frames, _, low_h, low_w = predicted_raymaps.shape
    size = predicted_disparity.shape[-2:]
    def upsample(value):
        return functional.interpolate(value.flatten(0, 1), size=size, mode="bilinear", align_corners=False).unflatten(0, (batch, frames))
    pred_rays, target_rays = upsample(predicted_raymaps), upsample(target_raymaps)
    # Ray origins are stored as signed log1p values by fixed-view geometry.
    for raymaps in (pred_rays, target_rays):
        origins = raymaps[:, :, 3:]
        raymaps[:, :, 3:] = origins.sign() * (origins.abs().exp() - 1)
    pred_depth = predicted_disparity.detach().clamp_min(1e-3).reciprocal()
    target_depth = target_disparity.clamp_min(1e-3).reciprocal()
    pred = pred_depth.unsqueeze(2) * pred_rays[:, :, :3] + pred_rays[:, :, 3:]
    target = target_depth.unsqueeze(2) * target_rays[:, :, :3] + target_rays[:, :, 3:]
    pred, target = _content(pred), _content(target)
    pred_center, target_center = pred.mean(dim=(1, 3, 4), keepdim=True), target.mean(dim=(1, 3, 4), keepdim=True)
    centered_pred, centered_target = pred - pred_center, target - target_center
    scale = (centered_pred * centered_target).sum(dim=(1, 2, 3, 4), keepdim=True) / centered_pred.square().sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    weights = _content(target_depth).reciprocal().clamp_min(1e-6).unsqueeze(2)
    return (weights * (scale * centered_pred - centered_target).abs()).sum() / weights.sum().clamp_min(1e-6)


def compute_stage2_losses(vae, clean_latents, batch):
    """Compute the paper's three decoded losses over valid fixed-view content."""
    import torch
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

    predicted_rgb = _decode(vae, clean_latents[:, :, :16])
    predicted_disparity = _decode(vae, clean_latents[:, :, 16:32])
    pred_rgb = _content((predicted_rgb * 0.5 + 0.5).clamp(0, 1))
    target_rgb = _content(batch["rgb"].permute(0, 2, 1, 3, 4))
    rgb_loss = 1 - multiscale_structural_similarity_index_measure(
        pred_rgb.float(),
        target_rgb.float(),
        data_range=1.0,
        betas=MS_SSIM_41_FRAME_BETAS,
    )
    pred_depth = decoded_disparity(predicted_disparity)
    target_depth = ((batch["disparity"][:, :, 0] * 0.5 + 0.5).clamp(0, 1)).square()
    depth_loss = depth_ssi_loss(_content(pred_depth), _content(target_depth))
    pointmap_loss = _pointmap_loss(pred_depth, unpack_raymaps(clean_latents[:, :, 32:]), target_depth, unpack_raymaps(batch["raymap"]))
    if not all(torch.isfinite(value) for value in (rgb_loss, depth_loss, pointmap_loss)):
        raise RuntimeError("Stage-2 decoded loss became non-finite")
    return Stage2Losses(rgb_loss, depth_loss, pointmap_loss)

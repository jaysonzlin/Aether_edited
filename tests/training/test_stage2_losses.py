from types import SimpleNamespace

import torch

from aether.training.stage2_losses import MS_SSIM_41_FRAME_BETAS, _decode, _pointmap_loss


def test_41_frame_ms_ssim_uses_a_three_scale_temporal_pyramid():
    assert MS_SSIM_41_FRAME_BETAS == (0.0448, 0.2856, 0.6696)


def test_decode_casts_clean_latents_to_the_frozen_vae_dtype_and_keeps_gradients():
    class Vae:
        dtype = torch.bfloat16
        config = SimpleNamespace(scaling_factor=1.0, invert_scale_latents=False)

        def decode(self, latents):
            self.decoder_input = latents
            return SimpleNamespace(sample=latents)

    vae = Vae()
    latents = torch.ones((1, 2, 16, 2, 2), requires_grad=True)

    decoded = _decode(vae, latents)
    decoded.float().sum().backward()

    assert vae.decoder_input.dtype is torch.bfloat16
    assert latents.grad is not None


def test_pointmap_loss_backpropagates_through_signed_log_ray_origins_without_inplace_mutation():
    predicted_disparity = torch.rand((1, 41, 4, 600), requires_grad=True)
    raw_predicted_raymaps = torch.randn((1, 41, 6, 2, 300), requires_grad=True)
    predicted_raymaps = raw_predicted_raymaps * 0.1
    target_disparity = torch.rand((1, 41, 4, 600))
    target_raymaps = torch.randn((1, 41, 6, 2, 300)) * 0.1

    loss = _pointmap_loss(
        predicted_disparity,
        predicted_raymaps,
        target_disparity,
        target_raymaps,
    )
    loss.backward()

    assert raw_predicted_raymaps.grad is not None

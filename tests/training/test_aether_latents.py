import os
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from aether.training.aether_latents import assemble_aether_training_batch


class FakeVAE:
    config = SimpleNamespace(scaling_factor=0.5, invert_scale_latents=False)

    def __init__(self):
        self.inputs = []

    def encode(self, video):
        self.inputs.append(video.clone())
        value = 2.0 * len(self.inputs)
        latents = torch.full(
            (video.shape[0], 16, 11, 60, 90), value, dtype=video.dtype
        )
        return SimpleNamespace(latents=latents)


def test_assembly_preserves_aether_modality_order_and_four_latent_history():
    vae = FakeVAE()
    rgb = torch.full((1, 41, 3, 480, 720), 0.25)
    disparity = torch.full((1, 41, 3, 480, 720), -0.5)
    raymap = torch.full((1, 11, 24, 60, 90), 7.0)

    assembled = assemble_aether_training_batch(
        {"rgb": rgb, "disparity": disparity, "raymap": raymap}, vae
    )

    assert assembled.target_latents.shape == (1, 11, 56, 60, 90)
    assert assembled.condition_latents.shape == (1, 11, 40, 60, 90)
    assert assembled.history_positions == (0, 1, 2, 3)
    torch.testing.assert_close(assembled.target_latents[:, :, :16], torch.ones(1, 11, 16, 60, 90))
    torch.testing.assert_close(assembled.target_latents[:, :, 16:32], torch.full((1, 11, 16, 60, 90), 2.0))
    torch.testing.assert_close(assembled.target_latents[:, :, 32:], raymap)
    torch.testing.assert_close(assembled.condition_latents[:, :4, :16], torch.ones(1, 4, 16, 60, 90))
    assert torch.count_nonzero(assembled.condition_latents[:, 4:, :16]) == 0
    torch.testing.assert_close(assembled.condition_latents[:, :, 16:], raymap)


def test_assembly_converts_rgb_to_vae_range_but_keeps_normalized_disparity():
    vae = FakeVAE()
    rgb = torch.full((1, 41, 3, 480, 720), 0.25)
    disparity = torch.full((1, 41, 3, 480, 720), -0.5)
    raymap = torch.zeros((1, 11, 24, 60, 90))

    assemble_aether_training_batch({"rgb": rgb, "disparity": disparity, "raymap": raymap}, vae)

    torch.testing.assert_close(vae.inputs[0], torch.full((1, 3, 41, 480, 720), -0.5))
    torch.testing.assert_close(vae.inputs[1], disparity.permute(0, 2, 1, 3, 4))

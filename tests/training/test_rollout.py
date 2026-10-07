import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from aether.training.rollout import composite_history


def test_composite_history_replaces_the_observed_raw_frame_prefix():
    generated = torch.zeros((1, 41, 3, 2, 2))
    observed = torch.ones((1, 41, 3, 2, 2))

    composite = composite_history(generated, observed, history_frames=13)

    assert torch.equal(composite[:, :13], observed[:, :13])
    assert torch.count_nonzero(composite[:, 13:]) == 0

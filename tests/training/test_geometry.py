import numpy as np

from aether.training.geometry import pack_raymaps, pad_intrinsics


def test_pad_intrinsics_widens_principal_point_without_rescaling():
    intrinsic = {"fx": 720.0, "fy": 720.0, "width": 480, "height": 480}

    matrix = pad_intrinsics(intrinsic, left_padding=120, right_padding=120)

    np.testing.assert_allclose(
        matrix,
        np.array(
            [[720.0, 0.0, 360.0], [0.0, 720.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )


def test_pack_raymaps_uses_aether_temporal_padding_and_four_frame_packing():
    raymaps = np.zeros((41, 6, 1, 1), dtype=np.float32)
    for frame_index in range(41):
        raymaps[frame_index] = frame_index

    packed = pack_raymaps(raymaps)

    assert packed.shape == (11, 24, 1, 1)
    np.testing.assert_array_equal(packed[0, :, 0, 0], np.repeat([0, 1, 2, 0], 6))
    np.testing.assert_array_equal(packed[-1, :, 0, 0], np.repeat([37, 38, 39, 40], 6))

"""Image conversion contracts for ComfyUI's actual torch tensor layout."""

import numpy as np
import pytest
import torch
from PIL import Image

from hordeai.utils.images import images_to_tensor, pil_to_tensor, tensor_to_pil


@pytest.mark.parametrize(
    ("mode", "pixels", "expected"),
    [
        ("L", [[0, 128], [255, 64]], [[[0], [128]], [[255], [64]]]),
        (
            "RGBA",
            [[[255, 0, 64, 0], [0, 128, 255, 255]], [[64, 32, 16, 128], [1, 2, 3, 0]]],
            [[[255, 0, 64], [0, 128, 255]], [[64, 32, 16], [1, 2, 3]]],
        ),
    ],
)
def test_pil_conversion_preserves_grayscale_and_removes_only_rgba_alpha(mode, pixels, expected):
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode=mode)
    result = pil_to_tensor(image)
    expected_tensor = torch.tensor(expected, dtype=torch.float32)[None] / 255

    assert result.dtype == torch.float32
    assert result.shape == expected_tensor.shape
    torch.testing.assert_close(result, expected_tensor, rtol=0, atol=0)
    assert image.mode == mode
    np.testing.assert_array_equal(np.asarray(image), pixels)


@pytest.mark.parametrize("shape", [(1, 1, 2, 3), (1, 2, 1, 3)])
def test_tensor_conversion_keeps_single_pixel_spatial_dimensions(shape):
    pixels = torch.tensor([0, 128, 255, 255, 64, 0], dtype=torch.float32).reshape(shape)
    tensor = pixels / 255

    image = tensor_to_pil(tensor)

    assert image.mode == "RGB"
    assert image.size == (shape[2], shape[1])
    np.testing.assert_array_equal(np.asarray(image), pixels[0].numpy().astype(np.uint8))
    torch.testing.assert_close(pil_to_tensor(image), tensor, rtol=0, atol=0)


def test_tensor_conversion_clamps_out_of_range_values_without_mutating_input():
    tensor = torch.tensor([[[[-0.5, 0.5, 1.5], [1.0, 0.0, 128 / 255]]]], dtype=torch.float32)
    original = tensor.clone()

    image = tensor_to_pil(tensor)

    assert image.size == (2, 1)
    np.testing.assert_array_equal(np.asarray(image), [[[0, 127, 255], [255, 0, 128]]])
    torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_batch_conversion_normalizes_grayscale_and_rgba_to_rgb_in_input_order():
    grayscale = Image.new("L", (2, 2), 128)
    transparent_red = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    green = Image.new("RGB", (2, 2), (0, 255, 0))

    batch = images_to_tensor([grayscale, transparent_red, green])

    assert batch.dtype == torch.float32
    assert batch.shape == (3, 2, 2, 3)
    expected_pixels = torch.tensor([[128 / 255] * 3, [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    torch.testing.assert_close(batch, expected_pixels[:, None, None, :].expand(3, 2, 2, 3), rtol=0, atol=0)
    assert grayscale.mode == "L"
    assert transparent_red.mode == "RGBA"

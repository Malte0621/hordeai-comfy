"""Preserved generation semantics translated into upstream SDK request types."""

import base64
from io import BytesIO

import numpy as np
import pytest
import torch
from horde_sdk import ANON_API_KEY
from horde_sdk.ai_horde_api.apimodels import ImageGenerateAsyncRequest, LorasPayloadEntry, TIPayloadEntry
from PIL import Image

from hordeai.generation.image import ImageSamplingSettings
from hordeai.generation.request import GenerationSettings


def settings(**overrides):
    values = {"prompt": "  A landscape  ", "width": 512, "height": 512, "num_images": 2, "model": "Deliberate"}
    values.update(overrides)
    return GenerationSettings(**values)


def decode_webp(value):
    image = Image.open(BytesIO(base64.b64decode(value, validate=True)))
    image.load()
    assert image.format == "WEBP"
    return image


@pytest.mark.parametrize(
    ("negative_prompt", "api_key", "seed", "expected_prompt", "expected_key", "expected_seed"),
    [
        ("blurry, noisy", "0123456789abcdefghijkl", 42, "  A landscape   ### blurry, noisy", "0123456789abcdefghijkl", "42"),
        (" \t", " \t", -1, "  A landscape  ", ANON_API_KEY, None),
        ("", "", 0, "  A landscape  ", ANON_API_KEY, "0"),
    ],
)
def test_positive_negative_prompts_random_seed_and_anonymous_key(
    negative_prompt, api_key, seed, expected_prompt, expected_key, expected_seed
):
    request = settings(negative_prompt=negative_prompt, api_key=api_key, sampling=ImageSamplingSettings(seed=seed)).to_request()

    assert isinstance(request, ImageGenerateAsyncRequest)
    assert request.prompt == expected_prompt
    assert request.apikey == expected_key
    assert request.params.seed == expected_seed
    assert request.models == ["Deliberate"]
    assert (request.params.width, request.params.height, request.params.n) == (512, 512, 2)
    if expected_seed is None:
        assert "seed" not in request.params.model_dump(exclude_none=True)


def test_lora_and_textual_inversion_fields_and_input_lists_are_preserved():
    loras = [LorasPayloadEntry(name="76693", model=0.6, clip=1.4, is_version=True, inject_trigger="painting")]
    tis = [TIPayloadEntry(name="72437", inject_ti="negprompt", strength=0.75)]
    expected_loras = [entry.model_dump() for entry in loras]
    expected_tis = [entry.model_dump() for entry in tis]

    request = settings(loras=loras, textual_inversions=tis).to_request()

    assert all(isinstance(entry, LorasPayloadEntry) for entry in request.params.loras)
    assert all(isinstance(entry, TIPayloadEntry) for entry in request.params.tis)
    assert [entry.model_dump() for entry in request.params.loras] == expected_loras
    assert [entry.model_dump() for entry in request.params.tis] == expected_tis
    assert [entry.model_dump() for entry in loras] == expected_loras
    assert [entry.model_dump() for entry in tis] == expected_tis


@pytest.mark.parametrize("processing", ["img2img", "inpainting", "outpainting", "remix"])
def test_source_image_is_rgb_webp_and_mask_keeps_white_modified_regions(processing):
    source = torch.empty((1, 64, 64, 4), dtype=torch.float32)
    source[:] = torch.tensor([1.0, 0.0, 0.0, 0.25])
    mask = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
    mask[:, :, 32:, :] = 1.0

    request = settings(
        source_image=source,
        source_processing=processing,
        source_mask=mask,
        sampling=ImageSamplingSettings(denoising_strength=0.35),
    ).to_request()
    decoded_source = decode_webp(request.source_image)
    decoded_mask = decode_webp(request.source_mask)

    assert request.source_processing == processing
    assert request.params.denoising_strength == 0.35
    assert decoded_source.mode == "RGB"
    assert decoded_source.size == decoded_mask.size == (64, 64)
    np.testing.assert_allclose(np.asarray(decoded_source)[32, 16], [255, 0, 0], rtol=0, atol=3)
    mask_pixels = np.asarray(decoded_mask.convert("L"))
    assert mask_pixels[32, 8] == 0
    assert mask_pixels[32, 56] == 255
    assert torch.all(source[..., 3] == 0.25)
    assert torch.count_nonzero(mask) == 64 * 32


@pytest.mark.parametrize("processing", ["img2img", "inpainting", "outpainting", "remix"])
def test_missing_source_falls_back_to_txt2img_and_ignores_mask(processing):
    request = settings(
        source_processing=processing,
        source_mask=torch.ones((1, 64, 64, 1)),
    ).to_request()

    assert request.source_processing == "txt2img"
    assert request.source_image is None
    assert request.source_mask is None
    body = request.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
    assert "source_processing" not in body
    assert "source_image" not in body
    assert "source_mask" not in body


def test_txt2img_does_not_send_unused_source_or_mask():
    request = settings(
        source_processing="txt2img",
        source_image=torch.ones((1, 64, 64, 3)),
        source_mask=torch.ones((1, 64, 64, 1)),
    ).to_request()

    assert request.source_processing == "txt2img"
    assert request.source_image is None
    assert request.source_mask is None
    body = request.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
    assert "source_processing" not in body
    assert "source_image" not in body
    assert "source_mask" not in body


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prompt": " \t\n"}, "Prompt cannot be empty"),
        ({"width": 65}, "multiples of 64"),
        ({"height": 513}, "multiples of 64"),
        ({"width": 1280, "height": 1024}, "dimensions too large"),
        ({"sampling": ImageSamplingSettings(steps=0)}, "Steps must be between 1 and 100"),
        ({"sampling": ImageSamplingSettings(steps=101)}, "Steps must be between 1 and 100"),
        ({"sampling": ImageSamplingSettings(cfg_scale=0.5)}, "CFG scale must be between 1.0 and 20.0"),
        ({"sampling": ImageSamplingSettings(cfg_scale=20.5)}, "CFG scale must be between 1.0 and 20.0"),
    ],
)
def test_existing_user_validation_remains_in_force(overrides, message):
    with pytest.raises(ValueError, match=message):
        settings(**overrides).to_request()

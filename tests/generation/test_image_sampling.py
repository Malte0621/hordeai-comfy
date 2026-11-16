"""Reusable image sampling settings preserve request semantics across workflow branches."""

from dataclasses import FrozenInstanceError

import pytest
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from hordeai.generation.image import ImageSamplingSettings
from hordeai.generation.request import GenerationSettings
from hordeai.nodes.image.sampling import AIHordeImageSampling


def test_shared_sampling_preserves_values_with_independent_request_dimensions_and_modes():
    (sampling,) = AIHordeImageSampling().make_sampling(37, 9.0, 0, 0.35, "k_euler_a", False)
    loras = [LorasPayloadEntry(name="76693", model=0.6, clip=1.4)]
    first = GenerationSettings(
        prompt="A landscape", width=64, height=128, num_images=2, model="Deliberate", sampling=sampling, loras=loras
    ).to_request()
    second = GenerationSettings(
        prompt="A portrait", width=512, height=512, num_images=1, model="Deliberate", sampling=sampling,
        source_processing="img2img",
    ).to_request()
    assert (first.params.width, first.params.height, first.params.n) == (64, 128, 2)
    assert (second.params.width, second.params.height, second.params.n) == (512, 512, 1)
    for payload in (first.params, second.params):
        assert payload.steps == 37
        assert payload.cfg_scale == 9.0
        assert payload.seed == "0"
        assert payload.sampler_name == "k_euler_a"
        assert payload.karras is False
    assert first.params.denoising_strength == 1.0
    assert second.params.denoising_strength == 0.35
    assert second.params.loras is None
    assert first.params is not second.params
    assert sampling.denoising_strength == 0.35
    with pytest.raises(FrozenInstanceError):
        sampling.seed = 42
    first.params.loras.append(LorasPayloadEntry(name="second"))
    assert [entry.name for entry in loras] == ["76693"]


def test_default_sampling_and_unconnected_sampling_produce_identical_payloads():
    values = {"prompt": "A landscape", "width": 64, "height": 64, "num_images": 1, "model": "Deliberate"}
    omitted = GenerationSettings(**values).to_request()
    explicit = GenerationSettings(**values, sampling=ImageSamplingSettings()).to_request()
    assert omitted.params.model_dump() == explicit.params.model_dump()
    assert omitted.params.steps == 20
    assert omitted.params.cfg_scale == 7.5
    assert omitted.params.sampler_name == "k_lms"
    assert omitted.params.karras is True
    assert omitted.params.seed is None
    assert "seed" not in omitted.params.model_dump(exclude_none=True, exclude_unset=True)

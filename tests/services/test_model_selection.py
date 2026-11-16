"""Blank model dispatch and txt2img payload acceptance through real SDK transport."""

import asyncio
from uuid import UUID

import pytest
from horde_model_reference import horde_model_reference_settings
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from PIL import Image

from hordeai.generation.request import GenerationSettings
from hordeai.services.horde import HordeClient
from tests.horde_server import encoded_image, enqueue_job
from tests.horde_server import horde_server as horde_server

REFERENCE_PATH = "/model_references/v2/image_generation"


def online(names, model_type="image"):
    return [{"name": name, "type": model_type, "count": 1} for name in names]


def model_record(name, **kwargs):
    return ImageGenerationModelRecord(name=name, baseline="stable_diffusion_1", nsfw=False, **kwargs).model_dump(mode="json", exclude_none=True)


def test_omitted_image_model_resolves_online_models_and_omits_invalid_txt2img_field(horde_server, monkeypatch):
    monkeypatch.setattr(horde_model_reference_settings, "primary_api_url", horde_server.origin)
    names = ["second", "first", "paint", "special-sampler", "custom", "SDXL_beta::stability.ai#6901"]
    horde_server.expect("GET", "/v2/status/models", online(names))
    horde_server.expect("GET", REFERENCE_PATH, {
        "first": model_record("first"),
        "second": model_record("second"),
        "paint": model_record("paint", inpainting=True),
        "special-sampler": model_record("special-sampler", requirements={"samplers": ["k_euler"]}),
    })
    job_id = str(UUID(int=601))
    enqueue_job(horde_server, job_id, [encoded_image(Image.new("RGB", (2, 2), "red"))])
    request = GenerationSettings(prompt="A landscape", width=64, height=64, num_images=1).to_request()
    images = asyncio.run(HordeClient().generate(request, lambda: None))
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    submitted = next(record for record in horde_server.requests if record.method == "POST")
    assert submitted.body["models"] == ["custom", "first", "second"]
    assert "source_processing" not in submitted.body
    assert "source_image" not in submitted.body and "source_mask" not in submitted.body
    assert request.models == [], "Model discovery must not mutate caller settings"
    assert horde_server.requests[0].query["type"] == ["image"]


def test_blank_model_with_no_online_workers_never_submits_a_generation(horde_server):
    horde_server.expect("GET", "/v2/status/models", [])
    request = GenerationSettings(prompt="A landscape", width=64, height=64, num_images=1, model=" \t").to_request()
    with pytest.raises(RuntimeError, match="No online image models"):
        asyncio.run(HordeClient().generate(request, lambda: None))
    assert horde_server.paths("POST") == []
    assert horde_server.paths("DELETE") == []


def test_explicit_image_model_needs_no_discovery_and_preserves_exact_name(horde_server):
    job_id = str(UUID(int=602))
    enqueue_job(horde_server, job_id, [encoded_image(Image.new("RGB", (2, 2), "blue"))])
    request = GenerationSettings(prompt="A landscape", width=64, height=64, num_images=1, model=" custom/model ").to_request()
    asyncio.run(HordeClient().generate(request, lambda: None))
    assert horde_server.requests[0].body["models"] == ["custom/model"]
    assert "/v2/status/models" not in horde_server.paths("GET")


def test_server_validation_failure_has_no_remote_job_and_does_not_claim_cleanup_failed(horde_server, caplog):
    horde_server.expect("POST", "/v2/generate/async", {
        "message": "Input payload validation failed",
        "errors": {"source_processing": "Invalid source mode"},
    }, status_code=400)
    request = GenerationSettings(prompt="A landscape", width=64, height=64, num_images=1, model="explicit").to_request()
    with pytest.raises(Exception, match="Input payload validation failed"):
        asyncio.run(HordeClient().generate(request, lambda: None))
    assert horde_server.paths("DELETE") == []
    assert "Could not cancel" not in caplog.text

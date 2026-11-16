"""Resolve blank image model inputs against online model capabilities."""

from __future__ import annotations

import aiohttp
from horde_model_reference import horde_model_reference_settings
from horde_model_reference.image_baseline import ImageBaselineCatalog
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateAsyncRequest


async def compatible_image_models(
    session: aiohttp.ClientSession,
    names: list[str],
    request: ImageGenerateAsyncRequest,
) -> list[str]:
    """Keep online models that support the requested source mode and sampling settings."""
    root = horde_model_reference_settings.primary_api_url
    if not root:
        raise RuntimeError("Image model discovery requires the AI Horde model-reference API")
    root = root.rstrip("/") + "/model_references/v2/image_generation"
    async with session.get(root) as response:
        response.raise_for_status()
        records = await response.json()

    remix_baselines: set[str] | None = None
    if request.source_processing == "remix":
        async with session.get(root + "/baselines/export") as response:
            response.raise_for_status()
            catalog = ImageBaselineCatalog.model_validate(await response.json())
        remix_baselines = {name for name, record in catalog.baselines.items() if record.capabilities.remix}

    selected: list[str] = []
    params = request.params
    for name in names:
        # This upstream beta is private and overrides every other model in a request.
        if name == "SDXL_beta::stability.ai#6901":
            continue
        raw = records.get(name)
        if raw is None:
            if remix_baselines is None:
                selected.append(name)
            continue
        record = ImageGenerationModelRecord.model_validate(raw)
        if record.inpainting and request.source_processing not in ("inpainting", "outpainting"):
            continue
        if remix_baselines is not None and str(record.baseline) not in remix_baselines:
            continue
        requirements = record.requirements or {}
        if params is not None:
            samplers = requirements.get("samplers")
            scheduler = params.scheduler or ("karras" if params.karras else "normal")
            schedulers = requirements.get("schedulers")
            if isinstance(samplers, list) and params.sampler_name not in samplers:
                continue
            if isinstance(schedulers, list) and scheduler not in schedulers:
                continue
        selected.append(name)
    return selected

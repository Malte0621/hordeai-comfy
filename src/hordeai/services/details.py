"""Public worker snapshots and generation metadata for ComfyUI string outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import aiohttp
from PIL import Image
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncManualClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    SingleWorkerDetailsRequest,
    SingleWorkerDetailsResponse,
)
from horde_sdk.exceptions import HordeException

from .lifecycle import GenerationResponse, GenerationSession


_PUBLIC_WORKER_FIELDS = {
    "type", "id", "name", "online", "performance", "threads", "uptime", "requests_fulfilled",
    "kudos_rewards", "kudos_details", "uncompleted_jobs", "models", "team", "bridge_agent",
    "trusted", "maintenance_mode", "paused", "max_pixels", "megapixelsteps_generated",
    "max_length", "max_context_length", "tokens_generated", "img2img", "painting", "lora",
    "post-processing", "controlnet", "sdxl_controlnet", "extended_controlnet",
}


async def _worker_stats(client: AIHordeAPIAsyncManualClient, worker_id: str) -> dict[str, Any]:
    try:
        response = await client.submit_request(SingleWorkerDetailsRequest(worker_id=worker_id), SingleWorkerDetailsResponse)
        if isinstance(response, RequestErrorResponse):
            return {"available": False, "error": response.message}
        data = response.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)
        return {"available": True, **{key: value for key, value in data.items() if key in _PUBLIC_WORKER_FIELDS}}
    except (aiohttp.ClientError, TimeoutError, HordeException, ValueError) as error:
        return {"available": False, "error": str(error)}


async def generation_details(
    client: AIHordeAPIAsyncManualClient,
    generation: GenerationSession,
    status: GenerationResponse,
    *,
    images: Sequence[Image.Image] | None = None,
) -> list[str]:
    """Fetch each worker once; preserve cancellation and report missing stats without losing completed outputs."""
    request = generation.request
    request_data: dict[str, Any] = {
        "id": str(generation.job_id),
        "kind": "image" if isinstance(request, ImageGenerateAsyncRequest) else "text",
        "prompt": request.prompt,
        "models": request.models,
        "parameters": request.params.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True) if request.params else {},
        "status": status.model_dump(mode="json", by_alias=True, exclude={"generations"}, exclude_none=True, exclude_unset=True),
    }
    if isinstance(request, ImageGenerateAsyncRequest):
        request_data["source_processing"] = str(request.source_processing)
    else:
        request_data["context_fit"] = str(request.context_fit) if request.context_fit is not None else None

    workers: dict[str, dict[str, Any]] = {}
    details: list[str] = []
    for index, result in enumerate(status.generations):
        worker_id = str(result.worker_id)
        if worker_id not in workers:
            workers[worker_id] = await generation.wait_for(_worker_stats(client, worker_id))
        result_data = result.model_dump(mode="json", by_alias=True, exclude={"img", "text", "worker_id", "worker_name"}, exclude_none=True)
        if "id_" in result_data:
            result_data["id"] = result_data.pop("id_")
        result_data["index"] = index
        if images is not None:
            result_data["width"], result_data["height"] = images[index].size
        details.append(json.dumps({
            "request": request_data,
            "generation": result_data,
            "worker": {"id": worker_id, "name": result.worker_name, "stats": workers[worker_id]},
        }, ensure_ascii=False, indent=2))
    return details

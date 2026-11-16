from __future__ import annotations

from collections.abc import Callable
from typing import cast

import aiohttp
from PIL import Image
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncManualClient, AIHordeAPIAsyncSimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerateStatusResponse,
    ImageStatsModelsRequest,
    ImageStatsModelsResponse,
    GenerationKobold,
    HordeStatusModelsAllRequest,
    HordeStatusModelsAllResponse,
    TextGenerateAsyncRequest,
    TextGenerateStatusResponse,
)
from horde_sdk.ai_horde_api.consts import MODEL_STATE, MODEL_TYPE
from horde_sdk.worker.model_meta import ImageModelLoadResolver


from ..generation.results import GenerationResult
from .details import generation_details
from .lifecycle import GenerationSession
from .model_selection import compatible_image_models


class HordeClient:
    """Own the HTTP session and remote lifecycle of one generation per call."""

    _request_timeout = 30.0

    def __init__(self, *, poll_interval: float = 4.0) -> None:
        self._poll_interval = poll_interval

    async def generate(
        self,
        request: ImageGenerateAsyncRequest,
        check_interrupt: Callable[[], None],
    ) -> GenerationResult[Image.Image]:
        """Generate images, deleting the remote request when local execution stops."""
        check_interrupt()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._request_timeout)) as session:
            client = AIHordeAPIAsyncManualClient(session)
            if not request.models:
                probe = GenerationSession(client, request, check_interrupt, poll_interval=self._poll_interval)
                models = await probe.wait_for(self._active_models(client, MODEL_TYPE.image))
                if models:
                    models = await probe.wait_for(compatible_image_models(session, models, request))
                check_interrupt()
                if not models:
                    raise RuntimeError("No online image models support the requested generation settings")
                request = request.model_copy(update={"models": models})
            async with GenerationSession(client, request, check_interrupt, poll_interval=self._poll_interval) as generation:
                status = cast(ImageGenerateStatusResponse, await generation.wait())
                downloader = AIHordeAPIAsyncSimpleClient(aiohttp_session=session)
                print(f"Downloading {len(status.generations)} generated image(s)...")
                images: list[Image.Image] = []
                for result in status.generations:
                    image, _ = await generation.wait_for(downloader.download_image_from_generation(result))
                    image.load()
                    images.append(image)
                details = await generation_details(client, generation, status, images=images)
                check_interrupt()
                return GenerationResult(images, details)

    async def get_models(self, top_n: int = 20) -> list[str]:
        """Return model names ranked by monthly usage without loading model references."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._request_timeout)) as session:
            client = AIHordeAPIAsyncManualClient(session)
            response = await client.submit_request(
                ImageStatsModelsRequest(model_state=MODEL_STATE.known), ImageStatsModelsResponse
            )
            if isinstance(response, RequestErrorResponse):
                raise RuntimeError(f"Error getting stats for models: {response.message}")
            return ImageModelLoadResolver.resolve_top_n_model_names(
                top_n, response, ImageModelLoadResolver.default_timeframe
            )

    async def generate_text(
        self,
        request: TextGenerateAsyncRequest,
        check_interrupt: Callable[[], None],
    ) -> GenerationResult[GenerationKobold]:
        """Return completed LLM responses in server order; never return partial jobs."""
        check_interrupt()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._request_timeout)) as session:
            client = AIHordeAPIAsyncManualClient(session)
            async with GenerationSession(client, request, check_interrupt, poll_interval=self._poll_interval) as generation:
                status = cast(TextGenerateStatusResponse, await generation.wait())
                expected = request.params.n if request.params is not None and request.params.n is not None else 1
                if len(status.generations) != expected:
                    raise RuntimeError(f"AI Horde completed {len(status.generations)} of {expected} requested text generations")
                if any(result.text is None for result in status.generations):
                    raise RuntimeError("AI Horde returned a text generation without text")
                details = await generation_details(client, generation, status)
                check_interrupt()
                return GenerationResult(status.generations, details)

    async def get_active_models(self, model_type: str, top_n: int = 100) -> list[str]:
        """Discover online models without restricting custom or newly added model names."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._request_timeout)) as session:
            client = AIHordeAPIAsyncManualClient(session)
            return (await self._active_models(client, MODEL_TYPE(model_type)))[:top_n]

    @staticmethod
    async def _active_models(client: AIHordeAPIAsyncManualClient, model_type: MODEL_TYPE) -> list[str]:
        response = await client.submit_request(
            HordeStatusModelsAllRequest(type=model_type, min_count=1), HordeStatusModelsAllResponse
        )
        if isinstance(response, RequestErrorResponse):
            raise RuntimeError(f"Error getting active models: {response.message}")
        return sorted({model.name for model in response.root if model.name and model.type_ == model_type and (model.count or 0) > 0})

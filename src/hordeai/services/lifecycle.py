"""Cancellation-safe lifecycle shared by image and text generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import suppress
from types import TracebackType
from typing import TypeVar, cast

from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncManualClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerateAsyncResponse,
    ImageGenerateCheckResponse,
    ImageGenerateStatusResponse,
    TextGenerateAsyncRequest,
    TextGenerateAsyncResponse,
    TextGenerateStatusResponse,
)
from horde_sdk.ai_horde_api.consts import GENERATION_MAX_LIFE
from horde_sdk.ai_horde_api.fields import GenerationID


logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")
type GenerationRequest = ImageGenerateAsyncRequest | TextGenerateAsyncRequest
type SubmissionResponse = ImageGenerateAsyncResponse | TextGenerateAsyncResponse
type ProgressResponse = ImageGenerateCheckResponse | ImageGenerateStatusResponse | TextGenerateStatusResponse
type GenerationResponse = ImageGenerateStatusResponse | TextGenerateStatusResponse


class GenerationSession:
    """Own one submitted job, including cancellation during submission and result use."""

    _interrupt_interval = 0.1

    def __init__(
        self,
        client: AIHordeAPIAsyncManualClient,
        request: GenerationRequest,
        check_interrupt: Callable[[], None],
        *,
        poll_interval: float,
    ) -> None:
        self.client = client
        self.request = request
        self.check_interrupt = check_interrupt
        self.poll_interval = poll_interval
        self.job_id: GenerationID | None = None
        self._submission: asyncio.Task[SubmissionResponse | RequestErrorResponse] | None = None
        self._response: SubmissionResponse | None = None

    async def __aenter__(self) -> GenerationSession:
        self.check_interrupt()
        self._submission = asyncio.create_task(
            self.client.submit_request(self.request, self.request.get_default_success_response_type())
        )
        try:
            # Keep POST alive until its response identifies the job that needs cleanup.
            while not self._submission.done():
                self.check_interrupt()
                await asyncio.wait((self._submission,), timeout=self._interrupt_interval)
            response = self._submission.result()
            if isinstance(response, RequestErrorResponse):
                raise RuntimeError(f"AI Horde API Error: {response.message}")
            self._response = response
            self.job_id = response.id_
            self.check_interrupt()
            logger.info("AI Horde request submitted, job ID: %s", self.job_id)
            return self
        except BaseException:
            await self._cancel_and_wait()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self._cancel_and_wait()

    async def wait(self) -> GenerationResponse:
        """Follow the SDK's image check or text status requests until the whole job finishes."""
        if self._response is None:
            raise RuntimeError("Generation has not been submitted")
        follow_up_data = self._response.get_follow_up_returned_params()[0]
        progress_type = self._response.get_follow_up_default_request_type()
        progress_request = progress_type.model_validate(follow_up_data)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GENERATION_MAX_LIFE

        while True:
            response = await self.wait_for(
                self.client.submit_request(progress_request, progress_request.get_default_success_response_type())
            )
            if isinstance(response, RequestErrorResponse):
                raise RuntimeError(f"AI Horde API Error: {response.message}")
            progress = cast(ProgressResponse, response)
            if progress.faulted:
                raise RuntimeError(f"AI Horde generation failed, job ID: {self.job_id}")
            if progress.done:
                break
            if loop.time() >= deadline:
                raise TimeoutError(f"AI Horde generation timed out, job ID: {self.job_id}")
            await self.wait_for(asyncio.sleep(self.poll_interval))

        finalize_type = progress.get_finalize_success_request_type()
        if finalize_type is not None:
            finalize = finalize_type.model_validate(follow_up_data)
            response = await self.wait_for(self.client.submit_request(finalize, finalize.get_default_success_response_type()))
            if isinstance(response, RequestErrorResponse):
                raise RuntimeError(f"AI Horde API Error: {response.message}")
            status = cast(GenerationResponse, response)
        else:
            status = cast(GenerationResponse, progress)
        if status.faulted:
            raise RuntimeError(f"AI Horde generation failed, job ID: {self.job_id}")
        if not status.generations:
            raise RuntimeError("No generations returned in the response")
        return status

    async def wait_for(self, operation: Coroutine[object, object, _Result]) -> _Result:
        task = asyncio.create_task(operation)
        try:
            while not task.done():
                self.check_interrupt()
                await asyncio.wait((task,), timeout=self._interrupt_interval)
            self.check_interrupt()
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            # Always retrieve failures, including when an interrupt wins the response race.
            with suppress(asyncio.CancelledError, Exception):
                await task

    async def _cancel_and_wait(self) -> None:
        cleanup = asyncio.create_task(self._cleanup())
        # A second cancel must not abort DELETE or replace the first interruption.
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue

    async def _cleanup(self) -> None:
        response = self._response
        if response is None:
            if self._submission is None:
                return
            try:
                submitted = await self._submission
            except Exception:
                # Rejected submissions have no known remote job to cancel.
                return
            if isinstance(submitted, RequestErrorResponse):
                return
            response = submitted
            self.job_id = response.id_

        try:
            cleanup_type = response.get_follow_up_failure_cleanup_request_type()
            cleanup_request = cleanup_type.model_validate(response.get_follow_up_returned_params()[0])
            result = await self.client.submit_request(cleanup_request, cleanup_request.get_default_success_response_type())
            if isinstance(result, RequestErrorResponse):
                raise RuntimeError(result.message)
            logger.info("Cancelled AI Horde generation %s", self.job_id)
        except Exception:
            logger.exception("Could not cancel AI Horde generation %s; remote work may continue", self.job_id)

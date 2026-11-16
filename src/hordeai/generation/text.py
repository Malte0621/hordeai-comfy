"""Typed LLM request settings independent of the ComfyUI interface."""

from __future__ import annotations

from dataclasses import dataclass

from horde_sdk import ANON_API_KEY
from horde_sdk.ai_horde_api.apimodels import ContextFit, ModelGenerationInputKobold, TextGenerateAsyncRequest


@dataclass(frozen=True, slots=True)
class TextSamplingSettings:
    """Reusable text sampling parameters with no shared mutable payload."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    stop_sequences: str = ""

    def to_payload(self, *, max_length: int, max_context_length: int, num_texts: int) -> ModelGenerationInputKobold:
        # Each nonempty line is one exact stop sequence; whitespace within a sequence is significant.
        stops = [line for line in self.stop_sequences.splitlines() if line]
        return ModelGenerationInputKobold(
            n=num_texts,
            max_length=max_length,
            max_context_length=max_context_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            rep_pen=self.repetition_penalty,
            stop_sequence=stops or None,
        )


@dataclass(frozen=True, slots=True)
class TextGenerationSettings:
    """A completion prompt and its generation limits; blank model accepts any LLM."""

    prompt: str
    max_length: int = 256
    max_context_length: int = 2048
    num_texts: int = 1
    model: str = ""
    api_key: str = ""
    context_fit: str = "reject"
    sampling: TextSamplingSettings | None = None

    def to_request(self) -> TextGenerateAsyncRequest:
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if self.max_length >= self.max_context_length:
            raise ValueError("max_context_length must exceed max_length to leave room for the prompt")
        sampling = self.sampling if self.sampling is not None else TextSamplingSettings()
        return TextGenerateAsyncRequest(
            prompt=self.prompt,
            models=[self.model.strip()] if self.model.strip() else [],
            apikey=self.api_key if self.api_key.strip() else ANON_API_KEY,
            params=sampling.to_payload(
                max_length=self.max_length, max_context_length=self.max_context_length, num_texts=self.num_texts
            ),
            context_fit=ContextFit(self.context_fit),
        )

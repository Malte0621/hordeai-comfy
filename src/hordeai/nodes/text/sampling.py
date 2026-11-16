from __future__ import annotations

from typing import Any

from ...generation.text import TextSamplingSettings


class AIHordeTextSampling:
    """Build reusable LLM sampling settings for one or more AI Horde Text Generate nodes."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Sampling randomness; 0 favors deterministic choices."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.001, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability; 1 disables filtering."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Candidate token count; 0 disables filtering."}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Relative minimum probability; 0 disables filtering."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "Discourage repeated tokens; 1 disables the penalty."}),
            },
            "optional": {
                "stop_sequences": ("STRING", {"default": "", "multiline": True, "tooltip": "One exact stop sequence per line. Empty lines are ignored; spaces are preserved."}),
            },
        }

    RETURN_TYPES = ("HORDE_TEXT_SAMPLING",)
    RETURN_NAMES = ("sampling",)
    FUNCTION = "make_sampling"
    CATEGORY = "AI Horde/Text"
    DESCRIPTION = __doc__

    def make_sampling(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        stop_sequences: str = "",
    ) -> tuple[TextSamplingSettings]:
        settings = TextSamplingSettings(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )
        return (settings,)

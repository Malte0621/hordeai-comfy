from __future__ import annotations

from typing import Any, cast

from ...generation.text import TextGenerationSettings, TextSamplingSettings
from ...services.horde import HordeClient
from ..inputs import api_key_input, model_input


class AIHordeTextGenerate:
    """Generate LLM completions using AI Horde. Feed text into prompts, text tools, or Preview as Text."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe a beautiful landscape in one sentence.",
                        "tooltip": "Completion prompt sent unchanged. Include the selected model's instruction format if needed.",
                    },
                ),
            },
            "optional": {
                "model": model_input(),
                "api_key": api_key_input(),
                "max_length": ("INT", {"default": 256, "min": 16, "max": 16384, "tooltip": "Maximum output tokens per completion."}),
                "max_context_length": (
                    "INT",
                    {"default": 2048, "min": 80, "max": 1048576, "tooltip": "Total prompt and output token budget. Must exceed max_length."},
                ),
                "num_texts": ("INT", {"default": 1, "min": 1, "max": 20, "tooltip": "Number of independent completions."}),
                "context_fit": (
                    ["reject", "ignore", "grow"],
                    {"default": "reject", "tooltip": "Reject oversized prompts, let workers truncate, or grow context at additional kudos cost."},
                ),
                "sampling": (
                    "HORDE_TEXT_SAMPLING",
                    {"tooltip": "Optional reusable sampling settings from AI Horde Text Sampling. Unconnected uses standard defaults."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "model", "seed")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "generate_text"
    CATEGORY = "AI Horde/Text"
    DESCRIPTION = __doc__

    async def generate_text(
        self,
        prompt: str,
        model: str = "",
        api_key: str = "",
        max_length: int = 256,
        max_context_length: int = 2048,
        num_texts: int = 1,
        context_fit: str = "reject",
        sampling: TextSamplingSettings | None = None,
    ) -> tuple[list[str], list[str], list[int]]:
        from comfy import model_management

        check_interrupt = model_management.throw_exception_if_processing_interrupted
        check_interrupt()
        settings = TextGenerationSettings(
            prompt=prompt,
            model=model,
            api_key=api_key,
            max_length=max_length,
            max_context_length=max_context_length,
            num_texts=num_texts,
            context_fit=context_fit,
            sampling=sampling,
        )
        request = settings.to_request()
        results = await HordeClient().generate_text(request, check_interrupt)
        check_interrupt()
        return (
            [cast(str, result.text) for result in results],
            [result.model for result in results],
            [result.seed if result.seed is not None else -1 for result in results],
        )

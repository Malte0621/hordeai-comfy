from __future__ import annotations

from typing import Any

from ...services.horde import HordeClient


class AIHordeTextModelList:
    """List exact online LLM model names, including custom models. Connect to AI Horde Model Select."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "top_n": ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Maximum online LLM names, sorted alphabetically."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "get_models"
    CATEGORY = "AI Horde/Text"
    DESCRIPTION = __doc__

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        # Availability changes independently of workflow input values.
        return float("nan")

    async def get_models(self, top_n: int = 100) -> tuple[list[str]]:
        return (await HordeClient().get_active_models("text", top_n),)

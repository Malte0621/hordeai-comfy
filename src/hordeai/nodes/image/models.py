from __future__ import annotations

from typing import Any

from ...services.horde import HordeClient


class AIHordeModelList:
    """
    Get available models from AI Horde

    This node fetches the current list of available models from the AI Horde network.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "top_n": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of top models to retrieve based on their usage",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "get_models"
    CATEGORY = "AI Horde/Image"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("nan")

    async def get_models(self, top_n: int = 20) -> tuple[list[str]]:
        return (await HordeClient().get_models(top_n),)

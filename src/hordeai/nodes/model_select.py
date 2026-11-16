from __future__ import annotations

from typing import Any


class AIHordeModelSelect:
    """Select one discovered model without automatically generating with every listed model."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Zero-based position in the connected model list."}),
            },
            "optional": {
                "model_list": ("STRING", {"forceInput": True, "tooltip": "Connect an image or text model-list node. The whole list is consumed once."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "select_model"
    CATEGORY = "AI Horde/Workflow"
    DESCRIPTION = __doc__

    def select_model(self, index: list[int], model_list: list[str] | None = None) -> tuple[str]:
        if model_list is None:
            return ("",)
        if not index or len(index) != 1:
            raise ValueError("Provide one model selection index")
        selected = index[0]
        if selected < 0 or selected >= len(model_list):
            raise ValueError(f"Model index {selected} is outside the available list of {len(model_list)} models")
        model = model_list[selected].strip()
        if not model:
            raise ValueError("Selected model name is empty")
        return (model,)

from __future__ import annotations

from inspect import cleandoc
from typing import Any

from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry


class AIHordeLora:
    """
    Create LoRA configuration entries for AI Horde

    This node creates LoRA (Low-Rank Adaptation) configuration entries that can be used
    with the AI Horde Image Generate node to apply fine-tuned model adaptations.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "lora_name": (
                    "STRING",
                    {
                        "default": "76693",
                        "tooltip": "LoRA model name or ID (e.g., 76693)",
                    },
                ),
                "model_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Model strength multiplier (0.0-2.0)",
                    },
                ),
                "clip_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "CLIP strength multiplier (0.0-2.0)",
                    },
                ),
                "is_version": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether the LoRA name is a version ID (True) or a name (False)",
                    },
                ),
            },
            "optional": {
                "loras_list": (
                    "LORAS_LIST",
                    {
                        "tooltip": "Optional existing LoRA list to append to",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LORAS_LIST",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "create_lora_entry"
    CATEGORY = "AI Horde/Adapters"
    DESCRIPTION = cleandoc(__doc__)

    def create_lora_entry(
        self,
        lora_name: str,
        model_strength: float,
        clip_strength: float,
        is_version: bool,
        loras_list: list[LorasPayloadEntry] | None = None,
    ) -> tuple[list[LorasPayloadEntry]]:
        # Create new LoRA entry
        lora_entry = LorasPayloadEntry(
            name=lora_name,
            model=model_strength,
            clip=clip_strength,
            is_version=is_version,
        )

        # Start with existing list or create new one
        if loras_list is None:
            result_list = []
        else:
            result_list = list(loras_list)

        # Add the new LoRA entry
        result_list.append(lora_entry)

        print(
            f"Added LoRA: {lora_name} (model: {model_strength}, clip: {clip_strength})"
        )
        print(f"Total LoRAs in list: {len(result_list)}")

        return (result_list,)

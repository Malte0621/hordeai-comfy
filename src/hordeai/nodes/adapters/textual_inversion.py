from __future__ import annotations

from inspect import cleandoc
from typing import Any

from horde_sdk.ai_horde_api.apimodels import TIPayloadEntry


class AIHordeTextualInversion:
    """
    Create Textual Inversion configuration entries for AI Horde

    This node creates Textual Inversion configuration entries that can be used
    with the AI Horde Image Generate node to apply textual embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "ti_name": (
                    "STRING",
                    {
                        "default": "72437",
                        "tooltip": "Textual Inversion name or ID (e.g., 72437)",
                    },
                ),
                "inject_ti": (
                    ["prompt", "negprompt"],
                    {
                        "default": "negprompt",
                        "tooltip": "Where to inject the textual inversion (prompt or negprompt)",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Textual inversion strength (0.0-5.0)",
                    },
                ),
            },
            "optional": {
                "tis_list": (
                    "TIS_LIST",
                    {
                        "tooltip": "Optional existing Textual Inversion list to append to",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TIS_LIST",)
    RETURN_NAMES = ("textual_inversions",)
    FUNCTION = "create_ti_entry"
    CATEGORY = "AI Horde/Adapters"
    DESCRIPTION = cleandoc(__doc__)

    def create_ti_entry(
        self,
        ti_name: str,
        inject_ti: str,
        strength: float,
        tis_list: list[TIPayloadEntry] | None = None,
    ) -> tuple[list[TIPayloadEntry]]:
        # Create new Textual Inversion entry
        ti_entry = TIPayloadEntry(
            name=ti_name,
            inject_ti=inject_ti,
            strength=strength,
        )

        # Start with existing list or create new one
        if tis_list is None:
            result_list = []
        else:
            result_list = list(tis_list)

        # Add the new TI entry
        result_list.append(ti_entry)

        print(
            f"Added Textual Inversion: {ti_name} (inject: {inject_ti}, strength: {strength})"
        )
        print(f"Total Textual Inversions in list: {len(result_list)}")

        return (result_list,)

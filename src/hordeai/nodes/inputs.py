"""Shared, independently allocated ComfyUI widget specifications."""

from typing import Any


def model_input() -> tuple[str, dict[str, Any]]:
    return (
        "STRING",
        {
            "default": "",
            "tooltip": "Optional exact AI Horde model name. Leave blank for any online model, or connect AI Horde Model Select.",
        },
    )


def api_key_input() -> tuple[str, dict[str, Any]]:
    return (
        "STRING",
        {
            "default": "",
            "tooltip": "AI Horde API key. Leave blank for anonymous generation. Convert to input to share a key across nodes.",
        },
    )

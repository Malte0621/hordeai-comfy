from __future__ import annotations

from typing import Any

from ...generation.image import ImageSamplingSettings


class AIHordeImageSampling:
    """Build reusable sampling settings for one or more AI Horde Image Generate nodes."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of sampling steps",
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 7.5,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Classifier-free guidance scale",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "tooltip": "Random seed (-1 for random)",
                    },
                ),
                "denoising_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoising strength for img2img (0.0-1.0)",
                    },
                ),
                "sampler_name": (
                    [
                        "k_lms",
                        "k_heun",
                        "k_euler",
                        "k_euler_a",
                        "k_dpm_2",
                        "k_dpm_2_a",
                        "k_dpm_fast",
                        "k_dpm_adaptive",
                        "k_dpmpp_2s_a",
                        "k_dpmpp_2m",
                        "dpmsolver",
                        "k_dpmpp_sde",
                        "lcms",
                        "DDIM",
                    ],
                    {
                        "default": "k_lms",
                        "tooltip": "Sampler algorithm to use",
                    },
                ),
                "karras": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use Karras noise schedule (if supported by model)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("HORDE_IMAGE_SAMPLING",)
    RETURN_NAMES = ("sampling",)
    FUNCTION = "make_sampling"
    CATEGORY = "AI Horde/Image"
    DESCRIPTION = __doc__

    def make_sampling(
        self,
        steps: int,
        cfg_scale: float,
        seed: int,
        denoising_strength: float,
        sampler_name: str,
        karras: bool,
    ) -> tuple[ImageSamplingSettings]:
        return (
            ImageSamplingSettings(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                denoising_strength=denoising_strength,
                sampler_name=sampler_name,
                karras=karras,
            ),
        )

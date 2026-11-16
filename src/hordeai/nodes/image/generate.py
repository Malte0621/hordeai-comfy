from __future__ import annotations

from inspect import cleandoc
from typing import Any

import torch
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry, TIPayloadEntry

from ...generation.request import GenerationSettings
from ...services.horde import HordeClient
from ...utils.images import images_to_tensor
from ..inputs import api_key_input, model_input


class AIHordeImageGenerate:
    """
    Generate images using AI Horde API

    This node connects to the AI Horde distributed network to generate images
    using various Stable Diffusion models available on the network.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful landscape",
                        "tooltip": "Text prompt describing the image to generate",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Width of the generated image",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Height of the generated image",
                    },
                ),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "Number of images to generate",
                    },
                ),
            },
            "optional": {
                "model": model_input(),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt to avoid certain elements",
                    },
                ),
                "api_key": api_key_input(),
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
                "loras": (
                    "LORAS_LIST",
                    {
                        "tooltip": "Optional LoRA list from AIHordeLora node",
                    },
                ),
                "textual_inversions": (
                    "TIS_LIST",
                    {
                        "tooltip": "Optional Textual Inversion list from AIHordeTextualInversion node",
                    },
                ),
                "source_image": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "Optional source image for img2img, inpainting, outpainting, or remix",
                    },
                ),
                "source_processing": (
                    ["txt2img", "img2img", "inpainting", "outpainting", "remix"],
                    {
                        "default": "txt2img",
                        "tooltip": "Processing type for source image",
                    },
                ),
                "source_mask": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "Optional mask image for inpainting or outpainting (white areas are modified)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "AI Horde/Image"
    DESCRIPTION = cleandoc(__doc__)

    async def generate_images(
        self,
        prompt: str,
        width: int,
        height: int,
        num_images: int,
        model: str = "",
        negative_prompt: str = "",
        api_key: str = "",
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: int = -1,
        denoising_strength: float = 1.0,
        sampler_name: str = "k_lms",
        karras: bool = True,
        loras: list[LorasPayloadEntry] | None = None,
        textual_inversions: list[TIPayloadEntry] | None = None,
        source_image: torch.Tensor | None = None,
        source_processing: str = "txt2img",
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        from comfy import model_management

        check_interrupt = model_management.throw_exception_if_processing_interrupted
        check_interrupt()
        settings = GenerationSettings(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_images,
            model=model,
            negative_prompt=negative_prompt,
            api_key=api_key,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            denoising_strength=denoising_strength,
            sampler_name=sampler_name,
            karras=karras,
            loras=loras,
            textual_inversions=textual_inversions,
            source_image=source_image,
            source_processing=source_processing,
            source_mask=source_mask,
        )

        print(f"Generating {num_images} image(s) with AI Horde...")
        print(
            f"Model: {model}, Size: {width}x{height}, Steps: {steps}, CFG: {cfg_scale}"
        )

        try:
            request = settings.to_request()
            print(f"Sending request to AI Horde: {request.prompt[:100]}...")
            pil_images = await HordeClient().generate(request, check_interrupt)
            result_tensor = images_to_tensor(pil_images)
            check_interrupt()
            print(f"Successfully generated {len(pil_images)} image(s)")

            return (result_tensor,)

        except model_management.InterruptProcessingException:
            raise
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating images: {error_msg}")

            # Provide more specific error information for debugging
            if "Input payload validation failed" in error_msg:
                print("Debug info:")
                print(f"  - Model: {model}")
                print(f"  - Dimensions: {width}x{height}")
                print(f"  - Steps: {steps}")
                print(f"  - CFG Scale: {cfg_scale}")
                print(f"  - Has source image: {source_image is not None}")
                print(f"  - Source processing: {source_processing}")
                print(f"  - Has source mask: {source_mask is not None}")
                print(f"  - Number of LoRAs: {len(loras) if loras else 0}")
                print(
                    f"  - Number of TIs: {len(textual_inversions) if textual_inversions else 0}"
                )

            raise RuntimeError(f"Error generating images: {error_msg}") from e

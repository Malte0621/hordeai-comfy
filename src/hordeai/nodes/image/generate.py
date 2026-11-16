from __future__ import annotations

from inspect import cleandoc
from typing import Any

import torch
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry, TIPayloadEntry

from ...generation.image import ImageSamplingSettings
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
                "sampling": (
                    "HORDE_IMAGE_SAMPLING",
                    {"tooltip": "Optional reusable settings from AI Horde Image Sampling. Unconnected uses standard defaults."},
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

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "details")
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
        loras: list[LorasPayloadEntry] | None = None,
        textual_inversions: list[TIPayloadEntry] | None = None,
        source_image: torch.Tensor | None = None,
        source_processing: str = "txt2img",
        source_mask: torch.Tensor | None = None,
        sampling: ImageSamplingSettings | None = None,
    ) -> tuple[torch.Tensor, str]:
        from comfy import model_management

        check_interrupt = model_management.throw_exception_if_processing_interrupted
        check_interrupt()
        image_sampling = sampling if sampling is not None else ImageSamplingSettings()
        settings = GenerationSettings(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_images,
            model=model,
            negative_prompt=negative_prompt,
            api_key=api_key,
            sampling=image_sampling,
            loras=loras,
            textual_inversions=textual_inversions,
            source_image=source_image,
            source_processing=source_processing,
            source_mask=source_mask,
        )

        print(f"Generating {num_images} image(s) with AI Horde...")
        print(
            f"Model: {model}, Size: {width}x{height}, Steps: {image_sampling.steps}, CFG: {image_sampling.cfg_scale}"
        )

        try:
            request = settings.to_request()
            print(f"Sending request to AI Horde: {request.prompt[:100]}...")
            result = await HordeClient().generate(request, check_interrupt)
            result_tensor = images_to_tensor(result.items)
            check_interrupt()
            print(f"Successfully generated {len(result.items)} image(s)")

            return (result_tensor, "[" + ",".join(result.details) + "]")

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
                print(f"  - Steps: {image_sampling.steps}")
                print(f"  - CFG Scale: {image_sampling.cfg_scale}")
                print(f"  - Has source image: {source_image is not None}")
                print(f"  - Source processing: {source_processing}")
                print(f"  - Has source mask: {source_mask is not None}")
                print(f"  - Number of LoRAs: {len(loras) if loras else 0}")
                print(
                    f"  - Number of TIs: {len(textual_inversions) if textual_inversions else 0}"
                )

            raise RuntimeError(f"Error generating images: {error_msg}") from e

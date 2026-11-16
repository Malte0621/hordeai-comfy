from __future__ import annotations

from dataclasses import dataclass

import torch
from horde_sdk import ANON_API_KEY
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerationInputPayload,
    LorasPayloadEntry,
    TIPayloadEntry,
)
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SOURCE_PROCESSING

from ..utils.images import tensor_to_base64


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Validated inputs for a single AI Horde image generation."""

    prompt: str
    width: int
    height: int
    num_images: int
    model: str = ""
    negative_prompt: str = ""
    api_key: str = ""
    steps: int = 20
    cfg_scale: float = 7.5
    seed: int = -1
    denoising_strength: float = 1.0
    sampler_name: str = "k_lms"
    karras: bool = True
    loras: list[LorasPayloadEntry] | None = None
    textual_inversions: list[TIPayloadEntry] | None = None
    source_image: torch.Tensor | None = None
    source_processing: str = "txt2img"
    source_mask: torch.Tensor | None = None

    def to_request(self) -> ImageGenerateAsyncRequest:
        # Validate inputs
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate dimensions (AI Horde has specific requirements)
        if self.width % 64 != 0 or self.height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        if self.width * self.height > 1024 * 1024:
            raise ValueError("Image dimensions too large (max 1024x1024 for most models)")

        # Validate steps
        if not 1 <= self.steps <= 100:
            raise ValueError("Steps must be between 1 and 100")

        # Validate CFG scale
        if not 1.0 <= self.cfg_scale <= 20.0:
            raise ValueError("CFG scale must be between 1.0 and 20.0")

        # Use anonymous key if no API key provided
        api_key = self.api_key
        if not api_key.strip():
            api_key = ANON_API_KEY
            print(
                "Using anonymous API key. Consider getting your own API key at https://aihorde.net/ for better performance."
            )

        # Add denoising_strength if applicable
        denoising_strength = (
            self.denoising_strength
            if self.source_processing in ("img2img", "inpainting", "outpainting", "remix")
            else 1.0
        )

        # Add seed if specified
        seed = str(self.seed) if self.seed >= 0 else None

        # Add loras if provided
        loras = self.loras or None
        if loras:
            print(f"Using {len(loras)} LoRA(s)")

        # Add textual inversions if provided
        textual_inversions = self.textual_inversions or None
        if textual_inversions:
            print(f"Using {len(textual_inversions)} Textual Inversion(s)")

        # Prepare generation parameters
        params = ImageGenerationInputPayload(
            height=self.height,
            width=self.width,
            n=self.num_images,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            karras=self.karras,
            denoising_strength=denoising_strength,
            loras=loras,
            tis=textual_inversions,
            **({"seed": seed} if seed is not None else {}),
        )

        # Prepare prompt with negative prompt if provided
        final_prompt = self.prompt
        if self.negative_prompt.strip():
            # AI Horde uses ### to separate positive and negative prompts
            final_prompt = f"{self.prompt} ### {self.negative_prompt}"

        # Handle source image processing
        processed_source_image = None
        processed_source_processing = "txt2img"  # Default to txt2img
        if self.source_processing != "txt2img":
            if self.source_image is not None:
                processed_source_image = tensor_to_base64(self.source_image)
                processed_source_processing = self.source_processing
            else:
                # If no source image but processing type is not txt2img, default to txt2img
                print(
                    "Warning: No source image provided but source_processing is not txt2img. Defaulting to txt2img."
                )

        # Handle source mask
        processed_source_mask = None
        if self.source_mask is not None and self.source_processing != "txt2img":
            if processed_source_image is None:
                print("Warning: Source mask provided but no source image. Ignoring mask.")
            else:
                processed_source_mask = tensor_to_base64(self.source_mask, mask=True)

        source_fields = {}
        if processed_source_image is not None:
            source_fields = {
                "source_image": processed_source_image,
                "source_processing": KNOWN_IMAGE_SOURCE_PROCESSING(processed_source_processing),
                "source_mask": processed_source_mask,
            }

        # Create request
        return ImageGenerateAsyncRequest(
            apikey=api_key,
            prompt=final_prompt,
            models=[self.model.strip()] if self.model.strip() else [],
            params=params,
            **source_fields,
        )

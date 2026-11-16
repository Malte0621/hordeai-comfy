from __future__ import annotations

from dataclasses import dataclass

import torch
from horde_sdk import ANON_API_KEY
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    LorasPayloadEntry,
    TIPayloadEntry,
)
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SOURCE_PROCESSING

from .image import ImageSamplingSettings
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
    loras: list[LorasPayloadEntry] | None = None
    textual_inversions: list[TIPayloadEntry] | None = None
    source_image: torch.Tensor | None = None
    source_processing: str = "txt2img"
    source_mask: torch.Tensor | None = None
    sampling: ImageSamplingSettings | None = None

    def to_request(self) -> ImageGenerateAsyncRequest:
        # Validate inputs
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate dimensions (AI Horde has specific requirements)
        if self.width % 64 != 0 or self.height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        if self.width * self.height > 1024 * 1024:
            raise ValueError("Image dimensions too large (max 1024x1024 for most models)")

        # Use anonymous key if no API key provided
        api_key = self.api_key
        if not api_key.strip():
            api_key = ANON_API_KEY
            print(
                "Using anonymous API key. Consider getting your own API key at https://aihorde.net/ for better performance."
            )

        # Add loras if provided
        loras = self.loras or None
        if loras:
            print(f"Using {len(loras)} LoRA(s)")

        # Add textual inversions if provided
        textual_inversions = self.textual_inversions or None
        if textual_inversions:
            print(f"Using {len(textual_inversions)} Textual Inversion(s)")

        # Prepare generation parameters
        sampling = self.sampling if self.sampling is not None else ImageSamplingSettings()
        params = sampling.to_payload(
            height=self.height,
            width=self.width,
            num_images=self.num_images,
            source_processing=self.source_processing,
            loras=loras,
            textual_inversions=textual_inversions,
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

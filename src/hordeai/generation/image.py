"""Reusable image sampling settings independent of ComfyUI workflows."""

from __future__ import annotations

from dataclasses import dataclass

from horde_sdk.ai_horde_api.apimodels import ImageGenerationInputPayload, LorasPayloadEntry, TIPayloadEntry


@dataclass(frozen=True, slots=True)
class ImageSamplingSettings:
    """Sampling parameters shared without mutation across image-generation requests."""

    steps: int = 20
    cfg_scale: float = 7.5
    seed: int = -1
    denoising_strength: float = 1.0
    sampler_name: str = "k_lms"
    karras: bool = True

    def to_payload(
        self,
        *,
        width: int,
        height: int,
        num_images: int,
        source_processing: str,
        loras: list[LorasPayloadEntry] | None = None,
        textual_inversions: list[TIPayloadEntry] | None = None,
    ) -> ImageGenerationInputPayload:
        # Validate steps
        if not 1 <= self.steps <= 100:
            raise ValueError("Steps must be between 1 and 100")

        # Validate CFG scale
        if not 1.0 <= self.cfg_scale <= 20.0:
            raise ValueError("CFG scale must be between 1.0 and 20.0")

        # Add denoising_strength if applicable
        denoising_strength = (
            self.denoising_strength
            if source_processing in ("img2img", "inpainting", "outpainting", "remix")
            else 1.0
        )

        # Add seed if specified
        seed = str(self.seed) if self.seed >= 0 else None

        # Prepare generation parameters
        return ImageGenerationInputPayload(
            height=height,
            width=width,
            n=num_images,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            karras=self.karras,
            denoising_strength=denoising_strength,
            loras=loras,
            tis=textual_inversions,
            **({"seed": seed} if seed is not None else {}),
        )

from __future__ import annotations

import base64
from io import BytesIO
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image"""
    # tensor shape: [batch, height, width, channels]
    pixels = tensor.detach().cpu().numpy()
    if pixels.ndim == 4:
        if pixels.shape[0] != 1:
            raise ValueError("Source image must contain exactly one image")
        pixels = pixels[0]
    pixels = np.clip(255.0 * pixels, 0, 255).astype(np.uint8)
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]
    return Image.fromarray(pixels)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor"""
    img_array = np.array(pil_img, dtype=np.float32)
    img_array /= 255.0
    if img_array.ndim == 2:  # Grayscale
        img_array = np.expand_dims(img_array, axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel

    # Add batch dimension: [1, height, width, channels]
    return torch.from_numpy(img_array).unsqueeze(0)


def tensor_to_base64(tensor: torch.Tensor, *, mask: bool = False) -> str:
    """Encode one source image or mask as a base64 WebP image."""
    # Convert tensor to PIL Image first
    image = tensor_to_pil(tensor)
    if mask:
        # Ensure mask is in the correct format (grayscale or RGB)
        if image.mode not in ("L", "RGB"):
            image = image.convert("L")
    else:
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

    # Convert to WebP format in memory
    with BytesIO() as buffer:
        image.save(buffer, format="WEBP")
        # Encode as base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def images_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    """Build a ComfyUI RGB image batch in generation order."""
    if not images:
        raise RuntimeError("No images were downloaded")

    # Convert PIL images to ComfyUI tensors
    tensors = [pil_to_tensor(image if image.mode == "RGB" else image.convert("RGB")) for image in images]
    # Concatenate all images into a single tensor
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

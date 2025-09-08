import sys
from typing import Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
from inspect import cleandoc
import base64
from io import BytesIO

try:
    from horde_sdk import ANON_API_KEY, RequestErrorResponse
    from horde_sdk.ai_horde_api.ai_horde_clients import (
        AIHordeAPISimpleClient,
        AIHordeAPIManualClient,
    )
    from horde_sdk.ai_horde_api.apimodels import (
        ImageGenerateAsyncRequest,
        ImageGenerateStatusResponse,
        ImageGenerationInputPayload,
        LorasPayloadEntry,
        TIPayloadEntry,
        ImageStatsModelsRequest,
        ImageStatsModelsResponse,
        StatsModelsTimeframe,
    )
    from horde_sdk.ai_horde_api.consts import MODEL_STATE
    from horde_sdk.ai_horde_worker.model_meta import ImageModelLoadResolver

    HORDE_SDK_AVAILABLE = True
except ImportError:
    print("Warning: horde-sdk not installed. Install with: pip install horde-sdk")
    HORDE_SDK_AVAILABLE = False


def tensor_to_pil(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    # tensor shape: [batch, height, width, channels]
    i = 255.0 * tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(pil_img):
    """Convert PIL Image to ComfyUI tensor"""
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.expand_dims(img_array, axis=2)
    if img_array.shape[2] == 3:  # RGB
        pass
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel

    # Add batch dimension: [1, height, width, channels]
    tensor = torch.from_numpy(img_array)[None,]
    return tensor


class AIHordeImageGenerate:
    """
    Generate images using AI Horde API

    This node connects to the AI Horde distributed network to generate images
    using various Stable Diffusion models available on the network.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
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
                "model": (
                    "STRING",
                    {
                        "default": "Deliberate",
                        "tooltip": "AI model name to use for generation (e.g., Deliberate, SDXL_beta::stability.ai#6901, etc.)",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt to avoid certain elements",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Your AI Horde API key (leave empty for anonymous)",
                    },
                ),
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
    CATEGORY = "AI Horde"
    DESCRIPTION = cleandoc(__doc__)

    def generate_images(
        self,
        prompt,
        width,
        height,
        num_images,
        model,
        negative_prompt="",
        api_key="",
        steps=20,
        cfg_scale=7.5,
        seed=-1,
        denoising_strength=1.0,
        sampler_name="k_lms",
        karras=True,
        loras=None,
        textual_inversions=None,
        source_image=None,
        source_processing="txt2img",
        source_mask=None,
    ):
        if not HORDE_SDK_AVAILABLE:
            raise RuntimeError(
                "horde-sdk is not installed. Please install it with: pip install horde-sdk"
            )

        # Validate inputs
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Use anonymous key if no API key provided
        if not api_key.strip():
            api_key = ANON_API_KEY
            print(
                "Using anonymous API key. Consider getting your own API key at https://aihorde.net/ for better performance."
            )

        print(f"Generating {num_images} image(s) with AI Horde...")
        print(
            f"Model: {model}, Size: {width}x{height}, Steps: {steps}, CFG: {cfg_scale}"
        )

        # Create synchronous client
        simple_client = AIHordeAPISimpleClient()

        try:
            # Validate dimensions (AI Horde has specific requirements)
            if width % 64 != 0 or height % 64 != 0:
                raise ValueError("Width and height must be multiples of 64")

            if width * height > 1024 * 1024:
                raise ValueError(
                    "Image dimensions too large (max 1024x1024 for most models)"
                )

            # Validate steps
            if steps < 1 or steps > 100:
                raise ValueError("Steps must be between 1 and 100")

            # Validate CFG scale
            if cfg_scale < 1.0 or cfg_scale > 20.0:
                raise ValueError("CFG scale must be between 1.0 and 20.0")

            # Prepare generation parameters
            params_dict = {
                "height": height,
                "width": width,
                "n": num_images,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler_name,
                "karras": karras,
            }

            # Add denoising_strength if applicable
            if source_processing in ["img2img", "inpainting", "outpainting", "remix"]:
                params_dict["denoising_strength"] = denoising_strength

            # Add seed if specified
            if seed >= 0:
                params_dict["seed"] = str(seed)

            # Add loras if provided
            if loras is not None and len(loras) > 0:
                params_dict["loras"] = loras
                print(f"Using {len(loras)} LoRA(s)")

            # Add textual inversions if provided
            if textual_inversions is not None and len(textual_inversions) > 0:
                params_dict["tis"] = textual_inversions
                print(f"Using {len(textual_inversions)} Textual Inversion(s)")

            params = ImageGenerationInputPayload(**params_dict)

            # Prepare prompt with negative prompt if provided
            final_prompt = prompt
            if negative_prompt.strip():
                # AI Horde uses ### to separate positive and negative prompts
                final_prompt = f"{prompt} ### {negative_prompt}"

            # Handle source image processing
            processed_source_image = None
            processed_source_processing = "txt2img"  # Default to txt2img

            if source_image is not None:
                # Convert tensor to PIL Image first
                source_pil = tensor_to_pil(source_image)
                # Ensure RGB mode
                if source_pil.mode != "RGB":
                    source_pil = source_pil.convert("RGB")

                # Convert to WebP format in memory
                source_buffer = BytesIO()
                source_pil.save(source_buffer, format="WEBP")
                source_buffer.seek(0)

                # Encode as base64
                processed_source_image = base64.b64encode(
                    source_buffer.getvalue()
                ).decode("utf-8")
                processed_source_processing = source_processing
            elif source_processing != "txt2img":
                # If no source image but processing type is not txt2img, default to txt2img
                print(
                    "Warning: No source image provided but source_processing is not txt2img. Defaulting to txt2img."
                )
                processed_source_processing = "txt2img"

            # Handle source mask
            processed_source_mask = None

            # Handle source mask
            processed_source_mask = None
            if source_mask is not None:
                if processed_source_image is None:
                    print(
                        "Warning: Source mask provided but no source image. Ignoring mask."
                    )
                else:
                    # Convert mask tensor to PIL Image
                    mask_pil = tensor_to_pil(source_mask)
                    # Ensure mask is in the correct format (grayscale or RGB)
                    if mask_pil.mode not in ["L", "RGB"]:
                        mask_pil = mask_pil.convert("L")

                    # Convert to WebP format in memory
                    mask_buffer = BytesIO()
                    mask_pil.save(mask_buffer, format="WEBP")
                    mask_buffer.seek(0)

                    # Encode as base64
                    processed_source_mask = base64.b64encode(
                        mask_buffer.getvalue()
                    ).decode("utf-8")

            # Create request
            if source_processing == "txt2img":
                request = ImageGenerateAsyncRequest(
                    apikey=api_key,
                    prompt=final_prompt,
                    models=[model],
                    params=params,
                )
            else:
                request = ImageGenerateAsyncRequest(
                    apikey=api_key,
                    prompt=final_prompt,
                    models=[model],
                    source_image=processed_source_image,
                    source_processing=processed_source_processing,
                    source_mask=processed_source_mask,
                    params=params,
                )

            print(f"Sending request to AI Horde: {final_prompt[:100]}...")

            # Generate images synchronously
            generation_response, job_id = simple_client.image_generate_request(request)

            if isinstance(generation_response, RequestErrorResponse):
                raise RuntimeError(f"AI Horde API Error: {generation_response.message}")

            print(f"Request completed, job ID: {job_id}")
            print(
                f"Downloading {len(generation_response.generations)} generated image(s)..."
            )

            if len(generation_response.generations) == 0:
                raise RuntimeError("No generations returned in the response")

            # Download images synchronously
            pil_images = []
            for i, generation in enumerate(generation_response.generations):
                print(
                    f"Downloading image {i+1}/{len(generation_response.generations)}..."
                )
                image_pil = simple_client.download_image_from_generation(generation)
                pil_images.append(image_pil)

            print(f"Successfully downloaded {len(pil_images)} image(s)")

            if not pil_images:
                raise RuntimeError("No images were downloaded")

            print(f"Successfully generated {len(pil_images)} image(s)")

            # Convert PIL images to ComfyUI tensors
            tensor_images = []
            for i, pil_img in enumerate(pil_images):
                # Ensure RGB mode
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                tensor_img = pil_to_tensor(pil_img)
                tensor_images.append(tensor_img)
                print(f"Converted image {i+1} to tensor: {tensor_img.shape}")

            # Concatenate all images into a single tensor
            result_tensor = torch.cat(tensor_images, dim=0)

            return (result_tensor,)

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

            raise RuntimeError(f"Error generating images: {error_msg}")


class AIHordeModelList:
    """
    Get available models from AI Horde

    This node fetches the current list of available models from the AI Horde network.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "top_n": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of top models to retrieve based on their usage",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "get_models"
    CATEGORY = "AI Horde"
    OUTPUT_NODE = True

    def get_models(self, top_n: Optional[int] = 20) -> Tuple[List[str]]:
        if not HORDE_SDK_AVAILABLE:
            return (
                "horde-sdk not installed - please install with: pip install horde-sdk",
            )

        try:
            client = AIHordeAPIManualClient()

            stats_response = client.submit_request(
                ImageStatsModelsRequest(model_state=MODEL_STATE.known),
                ImageStatsModelsResponse,
            )
            if isinstance(stats_response, RequestErrorResponse):
                raise Exception(
                    f"Error getting stats for models: {stats_response.message}"
                )

            models = ImageModelLoadResolver.resolve_top_n_model_names(
                top_n, stats_response, ImageModelLoadResolver.default_timeframe
            )

            return (models,)

        except Exception as e:
            return (f"Error getting model list: {str(e)}",)


class AIHordeLora:
    """
    Create LoRA configuration entries for AI Horde

    This node creates LoRA (Low-Rank Adaptation) configuration entries that can be used
    with the AI Horde Image Generate node to apply fine-tuned model adaptations.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = "AI Horde"
    DESCRIPTION = cleandoc(__doc__)

    def create_lora_entry(
        self,
        lora_name: str,
        model_strength: float,
        clip_strength: float,
        is_version: bool,
        loras_list: Optional[List] = None,
    ):
        if not HORDE_SDK_AVAILABLE:
            raise RuntimeError(
                "horde-sdk is not installed. Please install it with: pip install horde-sdk"
            )

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


class AIHordeTextualInversion:
    """
    Create Textual Inversion configuration entries for AI Horde

    This node creates Textual Inversion configuration entries that can be used
    with the AI Horde Image Generate node to apply textual embeddings.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = "AI Horde"
    DESCRIPTION = cleandoc(__doc__)

    def create_ti_entry(
        self,
        ti_name: str,
        inject_ti: str,
        strength: float,
        tis_list: Optional[List] = None,
    ):
        if not HORDE_SDK_AVAILABLE:
            raise RuntimeError(
                "horde-sdk is not installed. Please install it with: pip install horde-sdk"
            )

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AIHordeImageGenerate": AIHordeImageGenerate,
    "AIHordeModelList": AIHordeModelList,
    "AIHordeLora": AIHordeLora,
    "AIHordeTextualInversion": AIHordeTextualInversion,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIHordeImageGenerate": "AI Horde Image Generate",
    "AIHordeModelList": "AI Horde Model List",
    "AIHordeLora": "AI Horde LoRA",
    "AIHordeTextualInversion": "AI Horde Textual Inversion",
}

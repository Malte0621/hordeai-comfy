"""Stable ComfyUI node identifiers and display names."""

from .nodes.adapters.lora import AIHordeLora
from .nodes.adapters.textual_inversion import AIHordeTextualInversion
from .nodes.image.generate import AIHordeImageGenerate
from .nodes.image.models import AIHordeModelList
from .nodes.image.sampling import AIHordeImageSampling
from .nodes.model_select import AIHordeModelSelect
from .nodes.text.generate import AIHordeTextGenerate
from .nodes.text.models import AIHordeTextModelList
from .nodes.text.sampling import AIHordeTextSampling

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AIHordeImageGenerate": AIHordeImageGenerate,
    "AIHordeModelList": AIHordeModelList,
    "AIHordeImageSampling": AIHordeImageSampling,
    "AIHordeLora": AIHordeLora,
    "AIHordeTextualInversion": AIHordeTextualInversion,
    "AIHordeTextGenerate": AIHordeTextGenerate,
    "AIHordeTextModelList": AIHordeTextModelList,
    "AIHordeTextSampling": AIHordeTextSampling,
    "AIHordeModelSelect": AIHordeModelSelect,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIHordeImageGenerate": "AI Horde Image Generate",
    "AIHordeModelList": "AI Horde Model List",
    "AIHordeImageSampling": "AI Horde Image Sampling",
    "AIHordeLora": "AI Horde LoRA",
    "AIHordeTextualInversion": "AI Horde Textual Inversion",
    "AIHordeTextGenerate": "AI Horde Text Generate",
    "AIHordeTextModelList": "AI Horde Text Model List",
    "AIHordeTextSampling": "AI Horde Text Sampling",
    "AIHordeModelSelect": "AI Horde Model Select",
}

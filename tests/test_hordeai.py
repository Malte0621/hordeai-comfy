"""Saved workflow compatibility and end-to-end public node contracts."""

import asyncio
import inspect
import sys
from types import ModuleType
from uuid import UUID

import pytest
import torch
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry, TIPayloadEntry
from PIL import Image

import hordeai
from hordeai import registry
from hordeai.nodes.adapters.lora import AIHordeLora
from hordeai.nodes.adapters.textual_inversion import AIHordeTextualInversion
from hordeai.nodes.image.generate import AIHordeImageGenerate
from hordeai.nodes.image.models import AIHordeModelList
from hordeai.nodes.image.sampling import AIHordeImageSampling
from tests.horde_server import (
    GENERATION_TIMEOUT,
    SUBMIT_PATH,
    check_path,
    encoded_image,
    enqueue_job,
    progress,
    status,
    status_path,
    submission,
    wait_for_event,
)
from tests.horde_server import horde_server as horde_server


SAMPLERS = [
    "k_lms", "k_heun", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", "k_dpm_fast", "k_dpm_adaptive",
    "k_dpmpp_2s_a", "k_dpmpp_2m", "dpmsolver", "k_dpmpp_sde", "lcms", "DDIM",
]
SOURCE_PROCESSING = ["txt2img", "img2img", "inpainting", "outpainting", "remix"]
WORKFLOW_CONTRACTS = {
    "AIHordeImageGenerate": {
        "class": AIHordeImageGenerate,
        "display": "AI Horde Image Generate",
        "required": {"prompt": "STRING", "width": "INT", "height": "INT", "num_images": "INT"},
        "optional": {
            "model": "STRING",
            "negative_prompt": "STRING", "api_key": "STRING", "sampling": "HORDE_IMAGE_SAMPLING", "loras": "LORAS_LIST",
            "textual_inversions": "TIS_LIST", "source_image": "IMAGE", "source_processing": SOURCE_PROCESSING,
            "source_mask": "IMAGE",
        },
        "outputs": ("IMAGE",),
        "output_names": ("images",),
        "method": "generate_images",
    },
    "AIHordeImageSampling": {
        "class": AIHordeImageSampling,
        "display": "AI Horde Image Sampling",
        "required": {
            "steps": "INT", "cfg_scale": "FLOAT", "seed": "INT", "denoising_strength": "FLOAT",
            "sampler_name": SAMPLERS, "karras": "BOOLEAN",
        },
        "optional": {},
        "outputs": ("HORDE_IMAGE_SAMPLING",),
        "output_names": ("sampling",),
        "method": "make_sampling",
    },
    "AIHordeModelList": {
        "class": AIHordeModelList,
        "display": "AI Horde Model List",
        "required": {},
        "optional": {"top_n": "INT"},
        "outputs": ("STRING",),
        "output_names": ("model_list",),
        "method": "get_models",
    },
    "AIHordeLora": {
        "class": AIHordeLora,
        "display": "AI Horde LoRA",
        "required": {"lora_name": "STRING", "model_strength": "FLOAT", "clip_strength": "FLOAT", "is_version": "BOOLEAN"},
        "optional": {"loras_list": "LORAS_LIST"},
        "outputs": ("LORAS_LIST",),
        "output_names": ("loras",),
        "method": "create_lora_entry",
    },
    "AIHordeTextualInversion": {
        "class": AIHordeTextualInversion,
        "display": "AI Horde Textual Inversion",
        "required": {"ti_name": "STRING", "inject_ti": ["prompt", "negprompt"], "strength": "FLOAT"},
        "optional": {"tis_list": "TIS_LIST"},
        "outputs": ("TIS_LIST",),
        "output_names": ("textual_inversions",),
        "method": "create_ti_entry",
    },
}

class ComfyUIInterrupted(BaseException):
    """The interruption type supplied at the lazy ComfyUI integration boundary."""



@pytest.fixture
def comfy_management(monkeypatch):
    """Only the ComfyUI integration boundary is supplied; torch and the SDK remain real."""
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    management = ModuleType("comfy.model_management")
    management.InterruptProcessingException = ComfyUIInterrupted
    management.throw_exception_if_processing_interrupted = lambda: None
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return management


def test_registry_keeps_all_saved_workflow_identifiers_and_display_names():
    assert hordeai.NODE_CLASS_MAPPINGS is registry.NODE_CLASS_MAPPINGS
    assert hordeai.NODE_DISPLAY_NAME_MAPPINGS is registry.NODE_DISPLAY_NAME_MAPPINGS
    for identifier, contract in WORKFLOW_CONTRACTS.items():
        assert hordeai.NODE_CLASS_MAPPINGS[identifier] is contract["class"]
        assert hordeai.NODE_DISPLAY_NAME_MAPPINGS[identifier] == contract["display"]


@pytest.mark.parametrize("identifier", WORKFLOW_CONTRACTS)
def test_saved_workflow_sockets_outputs_and_dispatch_methods_survive_module_move(identifier):
    contract = WORKFLOW_CONTRACTS[identifier]
    node_class = hordeai.NODE_CLASS_MAPPINGS[identifier]
    inputs = node_class.INPUT_TYPES()

    assert set(inputs) == ({"required"} if identifier == "AIHordeImageSampling" else {"required", "optional"})
    for group in ("required", "optional"):
        assert {name: specification[0] for name, specification in inputs.get(group, {}).items()} == contract[group]
    assert node_class.RETURN_TYPES == contract["outputs"]
    assert node_class.RETURN_NAMES == contract["output_names"]
    assert node_class.FUNCTION == contract["method"]
    expected_category = "AI Horde/Image" if identifier in {"AIHordeImageGenerate", "AIHordeModelList", "AIHordeImageSampling"} else "AI Horde/Adapters"
    assert node_class.CATEGORY == expected_category
    method = getattr(node_class, node_class.FUNCTION)
    assert set(inspect.signature(method).parameters) - {"self"} == set(contract["required"]) | set(contract["optional"])
    assert inspect.iscoroutinefunction(method) == (identifier in {"AIHordeImageGenerate", "AIHordeModelList"})
    if identifier == "AIHordeModelList":
        assert node_class.OUTPUT_IS_LIST == (True,)
        assert node_class.OUTPUT_NODE is True


def test_lora_chains_branch_without_mutating_the_shared_input_list():
    node = AIHordeLora()
    (base,) = node.create_lora_entry("76693", 0.6, 1.4, True)
    before = [entry.model_dump() for entry in base]
    (left,) = node.create_lora_entry("second", 1.2, 0.5, False, base)
    (right,) = node.create_lora_entry("third", 0.8, 1.0, True, base)

    assert [entry.model_dump() for entry in base] == before
    assert base is not left and base is not right and left is not right
    assert [entry.name for entry in left] == ["76693", "second"]
    assert [entry.name for entry in right] == ["76693", "third"]
    assert all(isinstance(entry, LorasPayloadEntry) for entry in left + right)
    assert (base[0].model, base[0].clip, base[0].is_version) == (0.6, 1.4, True)
    assert (left[1].model, left[1].clip, left[1].is_version) == (1.2, 0.5, False)


def test_textual_inversion_chains_preserve_injection_target_and_shared_input_list():
    node = AIHordeTextualInversion()
    (base,) = node.create_ti_entry("72437", "negprompt", 0.75)
    before = [entry.model_dump() for entry in base]
    (left,) = node.create_ti_entry("second", "prompt", 1.5, base)
    (right,) = node.create_ti_entry("third", "negprompt", 0.5, base)

    assert [entry.model_dump() for entry in base] == before
    assert base is not left and base is not right and left is not right
    assert [entry.name for entry in left] == ["72437", "second"]
    assert [entry.name for entry in right] == ["72437", "third"]
    assert all(isinstance(entry, TIPayloadEntry) for entry in left + right)
    assert (base[0].inject_ti, base[0].strength) == ("negprompt", 0.75)
    assert (left[1].inject_ti, left[1].strength) == ("prompt", 1.5)


def test_generate_node_returns_rgb_tensor_batch_from_real_sdk_results(horde_server, comfy_management):
    job_id = str(UUID(int=101))
    enqueue_job(
        horde_server,
        job_id,
        [encoded_image(Image.new("RGB", (2, 2), "green")), encoded_image(Image.new("L", (2, 2), 64))],
    )
    checked = []

    def check_interrupt():
        checked.append(True)

    comfy_management.throw_exception_if_processing_interrupted = check_interrupt
    (loras,) = AIHordeLora().create_lora_entry("76693", 0.6, 1.4, True)
    (tis,) = AIHordeTextualInversion().create_ti_entry("72437", "negprompt", 0.75)
    (sampling,) = AIHordeImageSampling().make_sampling(37, 9.0, 42, 0.35, "k_euler_a", False)
    result = asyncio.run(
        asyncio.wait_for(
            AIHordeImageGenerate().generate_images(
                "A landscape", 64, 64, 2, "Deliberate", negative_prompt="blurry", sampling=sampling,
                loras=loras, textual_inversions=tis,
            ),
            GENERATION_TIMEOUT,
        )
    )

    assert isinstance(result, tuple) and len(result) == 1
    batch = result[0]
    assert batch.shape == (2, 2, 2, 3)
    assert batch.dtype == torch.float32
    expected = torch.tensor([[0, 128 / 255, 0], [64 / 255] * 3], dtype=torch.float32)
    torch.testing.assert_close(batch, expected[:, None, None, :].expand(2, 2, 2, 3), rtol=0, atol=0)
    assert checked, "Node must pass the actual ComfyUI interruption callback to the generation service"
    submitted = next(request for request in horde_server.requests if request.method == "POST")
    assert submitted.body["prompt"] == "A landscape ### blurry"
    assert submitted.body["params"]["seed"] == "42"
    assert submitted.body["params"]["steps"] == 37
    assert submitted.body["params"]["cfg_scale"] == 9.0
    assert submitted.body["params"]["sampler_name"] == "k_euler_a"
    assert submitted.body["params"]["karras"] is False
    assert submitted.body["params"]["denoising_strength"] == 1.0
    assert submitted.body["params"]["loras"][0]["name"] == "76693"
    assert submitted.body["params"]["tis"][0]["inject_ti"] == "negprompt"
    assert horde_server.paths("DELETE") == []


@pytest.mark.parametrize("error_base", [BaseException, Exception])
def test_generate_node_propagates_comfyui_interrupt_and_cancels_remote_job(horde_server, comfy_management, error_base):
    class NodeInterrupted(error_base):
        pass

    job_id = str(UUID(int=102))
    horde_server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202)
    stalled = horde_server.expect("GET", check_path(job_id), progress(), hold="response")
    horde_server.expect("DELETE", status_path(job_id), status())
    interruption = NodeInterrupted("ComfyUI cancelled processing")
    comfy_management.InterruptProcessingException = NodeInterrupted

    def check_interrupt():
        if stalled.arrived.is_set():
            raise interruption

    comfy_management.throw_exception_if_processing_interrupted = check_interrupt

    async def scenario():
        try:
            with pytest.raises(NodeInterrupted) as raised:
                await asyncio.wait_for(
                    AIHordeImageGenerate().generate_images("A landscape", 64, 64, 1, "Deliberate"), GENERATION_TIMEOUT
                )
            assert raised.value is interruption
            await wait_for_event(stalled.disconnected)
        finally:
            stalled.release.set()

    asyncio.run(scenario())

    assert horde_server.paths("GET") == [check_path(job_id)]
    assert horde_server.paths("DELETE") == [status_path(job_id)]


def test_model_node_returns_most_used_monthly_models_as_comfyui_list(horde_server):
    horde_server.expect(
        "GET", "/v2/stats/img/models",
        {"day": {"least": 500}, "month": {"least": 1, "winner": 99, "middle": 20}, "total": {"old": 9999}},
    )

    result = asyncio.run(asyncio.wait_for(AIHordeModelList().get_models(top_n=2), GENERATION_TIMEOUT))

    assert result == (["winner", "middle"],)
    assert horde_server.paths("GET") == ["/v2/stats/img/models"]
    assert horde_server.requests[0].query == {"model_state": ["known"]}
    assert horde_server.paths("DELETE") == []


def test_model_node_reports_api_failure_instead_of_returning_an_error_as_a_model(horde_server):
    horde_server.expect("GET", "/v2/stats/img/models", {"message": "Statistics unavailable", "rc": "ServiceUnavailable"}, status_code=503)

    with pytest.raises(RuntimeError, match="Statistics unavailable"):
        asyncio.run(asyncio.wait_for(AIHordeModelList().get_models(top_n=2), GENERATION_TIMEOUT))
    assert horde_server.paths("DELETE") == []

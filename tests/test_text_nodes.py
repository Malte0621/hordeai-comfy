"""ComfyUI list outputs compose without multiplying model selections."""

import asyncio
from uuid import UUID

import pytest

import hordeai
from hordeai.nodes.model_select import AIHordeModelSelect
from hordeai.nodes.text.generate import AIHordeTextGenerate
from hordeai.nodes.text.models import AIHordeTextModelList
from hordeai.nodes.text.sampling import AIHordeTextSampling
from tests.horde_server import GENERATION_TIMEOUT
from tests.horde_server import horde_server as horde_server
from tests.test_hordeai import comfy_management as comfy_management
from tests.text_horde import TEXT_SUBMIT_PATH, text_status, text_status_path, text_submission


def test_text_node_returns_aligned_string_model_and_seed_lists(horde_server, comfy_management):
    job_id = str(UUID(int=701))
    horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
    horde_server.expect("GET", text_status_path(job_id), text_status(["A red sunrise.", "A snowy mountain."]))
    (sampling,) = AIHordeTextSampling().make_sampling(0.3, 0.8, 10, 0.1, 1.2, "END")
    result = asyncio.run(asyncio.wait_for(AIHordeTextGenerate().generate_text("Describe a landscape.", num_texts=2, sampling=sampling), GENERATION_TIMEOUT))
    assert result == (["A red sunrise.", "A snowy mountain."], ["koboldcpp/model-0", "koboldcpp/model-1"], [41, 42])
    submitted = horde_server.requests[0]
    assert submitted.body["params"]["temperature"] == 0.3
    assert submitted.body["params"]["stop_sequence"] == ["END"]
    assert submitted.body["models"] == []
    assert AIHordeTextGenerate.OUTPUT_IS_LIST == (True, True, True)
    assert hordeai.NODE_CLASS_MAPPINGS["AIHordeTextGenerate"] is AIHordeTextGenerate


def test_text_model_list_selects_one_online_custom_model(horde_server):
    horde_server.expect("GET", "/v2/status/models", [
        {"name": "z/custom", "type": "text", "count": 1},
        {"name": "a/new-model", "type": "text", "count": 2},
        {"name": "offline", "type": "text", "count": 0},
        {"name": "image-model", "type": "image", "count": 3},
    ])
    (models,) = asyncio.run(AIHordeTextModelList().get_models())
    assert models == ["a/new-model", "z/custom"]
    assert AIHordeModelSelect().select_model([1], models) == ("z/custom",)
    assert horde_server.requests[0].query["type"] == ["text"]
    assert horde_server.requests[0].query["min_count"] == ["1"]
    assert AIHordeModelSelect.INPUT_IS_LIST is True


@pytest.mark.parametrize("models,index", [([], [0]), ([""], [0]), (["first"], [1]), (["first"], [-1]), (["first"], [0, 1])])
def test_selector_rejects_invalid_choices_instead_of_accidentally_using_any_model(models, index):
    with pytest.raises(ValueError):
        AIHordeModelSelect().select_model(index, models)


def test_unconnected_selector_is_any_model_not_a_literal_name():
    assert AIHordeModelSelect().select_model([0]) == ("",)

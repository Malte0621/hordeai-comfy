"""LLM request semantics and reusable sampling parameters."""

import pytest
from horde_sdk import ANON_API_KEY

from hordeai.generation.text import TextGenerationSettings, TextSamplingSettings
from hordeai.nodes.text.sampling import AIHordeTextSampling


@pytest.mark.parametrize("model", ["", " \t", "koboldcpp/custom model"])
def test_completion_prompt_and_optional_model_are_preserved(model):
    prompt = "  User: Describe a landscape.\nAssistant: "
    request = TextGenerationSettings(prompt=prompt, model=model, max_length=80).to_request()
    assert request.prompt == prompt
    assert request.models == ([model] if model.strip() else [])
    assert request.apikey == ANON_API_KEY
    assert request.params.max_length == 80
    assert request.context_fit == "reject"


def test_shared_sampling_keeps_stop_sequence_whitespace_and_independent_limits():
    (sampling,) = AIHordeTextSampling().make_sampling(0.2, 0.8, 20, 0.05, 1.2, " stop \n\nEND\n")
    first = TextGenerationSettings(prompt="first", num_texts=2, sampling=sampling, api_key="0123456789abcdefghijkl").to_request()
    second = TextGenerationSettings(prompt="second", max_length=80, sampling=sampling).to_request()
    assert first.params.stop_sequence == second.params.stop_sequence == [" stop ", "END"]
    assert first.params.n == 2 and second.params.n == 1
    assert first.params.max_length == 256 and second.params.max_length == 80
    assert first.apikey == "0123456789abcdefghijkl"
    assert first.params.temperature == 0.2
    assert first.params.rep_pen == 1.2
    assert first.params is not second.params
    first.params.stop_sequence.append("changed")
    assert second.params.stop_sequence == [" stop ", "END"]
    assert sampling.stop_sequences == " stop \n\nEND\n"


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": " \t"},
        {"max_length": 2048, "max_context_length": 2048},
        {"max_length": 15},
        {"num_texts": 21},
        {"context_fit": "truncate_silently"},
        {"sampling": TextSamplingSettings(temperature=-1)},
        {"sampling": TextSamplingSettings(top_p=0)},
    ],
)
def test_invalid_generation_settings_fail_before_remote_submission(overrides):
    values = {"prompt": "A story"} | overrides
    with pytest.raises(ValueError):
        TextGenerationSettings(**values).to_request()

"""Worker snapshots remain aligned, public, and cancellable."""

import asyncio
import json
from uuid import UUID

import pytest
from PIL import Image

from hordeai.generation.request import GenerationSettings
from hordeai.generation.text import TextGenerationSettings
from hordeai.services.horde import HordeClient
from tests.horde_server import GENERATION_TIMEOUT, SUBMIT_PATH, encoded_image, progress, status, status_path, submission, wait_for_event
from tests.horde_server import horde_server as horde_server
from tests.text_horde import TEXT_SUBMIT_PATH, text_status, text_status_path, text_submission
from tests.worker_details import worker_details, worker_path


class Interrupted(BaseException):
    pass


@pytest.mark.parametrize("kind", ["image", "text"])
def test_details_preserve_generation_worker_alignment_and_redact_private_payloads(horde_server, kind):
    job_id = str(UUID(int=801))
    first_worker = str(UUID(int=810))
    second_worker = str(UUID(int=811))
    if kind == "image":
        request = GenerationSettings(prompt="A landscape", width=64, height=64, num_images=2, model="chosen", api_key="0123456789abcdefghijkl").to_request()
        response = status([encoded_image(Image.new("RGB", (3, 4), "red")), encoded_image(Image.new("RGB", (3, 4), "blue"))])
        horde_server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202)
        horde_server.expect("GET", f"/v2/generate/check/{job_id}", progress(done=True, count=2))
        result_path = status_path(job_id)
        run = HordeClient().generate
    else:
        request = TextGenerationSettings(prompt="A story", num_texts=2, model="chosen", api_key="0123456789abcdefghijkl").to_request()
        response = text_status(["first", "second"])
        horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
        result_path = text_status_path(job_id)
        run = HordeClient().generate_text
    for index, worker in enumerate([first_worker, second_worker]):
        response["generations"][index]["worker_id"] = worker
        response["generations"][index]["worker_name"] = f"worker-{index}"
    response["kudos"] = 12.5
    horde_server.expect("GET", result_path, response)
    for index, worker in enumerate([first_worker, second_worker]):
        horde_server.expect("GET", worker_path(worker), worker_details(
            worker, kind=kind, name=f"worker-{index}", requests_fulfilled=100 + index,
            ipaddr="192.0.2.1", owner="private owner", contact="private@example.com",
        ))
    result = asyncio.run(run(request, lambda: None))
    entries = [json.loads(value) for value in result.details]
    assert [entry["generation"]["index"] for entry in entries] == [0, 1]
    assert [entry["worker"]["id"] for entry in entries] == [first_worker, second_worker]
    assert [entry["worker"]["stats"]["requests_fulfilled"] for entry in entries] == [100, 101]
    assert all(entry["request"]["id"] == job_id and entry["request"]["status"]["kudos"] == 12.5 for entry in entries)
    for entry in entries:
        assert "apikey" not in entry["request"]
        assert "img" not in entry["generation"] and "text" not in entry["generation"]
        assert not {"ipaddr", "owner", "contact"} & entry["worker"]["stats"].keys()
        if kind == "image":
            assert (entry["generation"]["width"], entry["generation"]["height"]) == (3, 4)
    assert "0123456789abcdefghijkl" not in "".join(result.details)


@pytest.mark.parametrize("unavailable", ["not_found", "malformed"])
def test_missing_worker_stats_do_not_discard_completed_text(horde_server, unavailable):
    job_id = str(UUID(int=802))
    worker_id = str(UUID(int=3000))
    horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
    horde_server.expect("GET", text_status_path(job_id), text_status(["usable text", "another usable text"]))
    body = {"message": "Worker went offline", "rc": "WorkerNotFound"} if unavailable == "not_found" else {"name": "incomplete worker"}
    horde_server.expect("GET", worker_path(worker_id), body, status_code=404 if unavailable == "not_found" else 200)
    result = asyncio.run(HordeClient().generate_text(TextGenerationSettings(prompt="A story", num_texts=2).to_request(), lambda: None))
    assert [item.text for item in result.items] == ["usable text", "another usable text"]
    entries = [json.loads(value) for value in result.details]
    assert all(entry["worker"]["stats"]["available"] is False and entry["worker"]["stats"]["error"] for entry in entries)
    assert horde_server.paths("GET").count(worker_path(worker_id)) == 1
    assert horde_server.paths("DELETE") == []


def test_interruption_during_worker_stats_preserves_comfyui_cancellation(horde_server):
    job_id = str(UUID(int=803))
    worker_id = str(UUID(int=3000))
    horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
    horde_server.expect("GET", text_status_path(job_id), text_status(["completed text"]))
    stalled = horde_server.expect("GET", worker_path(worker_id), worker_details(worker_id, kind="text"), hold="response")
    horde_server.expect("DELETE", text_status_path(job_id), text_status())
    interruption = Interrupted("Cancel metadata lookup")

    def check_interrupt():
        if stalled.arrived.is_set():
            raise interruption

    async def scenario():
        try:
            with pytest.raises(Interrupted) as raised:
                await asyncio.wait_for(HordeClient().generate_text(TextGenerationSettings(prompt="A story").to_request(), check_interrupt), GENERATION_TIMEOUT)
            assert raised.value is interruption
            await wait_for_event(stalled.disconnected)
        finally:
            stalled.release.set()

    asyncio.run(scenario())
    assert horde_server.paths("DELETE") == [text_status_path(job_id)]

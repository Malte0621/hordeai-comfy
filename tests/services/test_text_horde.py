"""Text lifecycle proof against real SDK clients and isolated HTTP responses."""

import asyncio
from threading import Event
from uuid import UUID

import pytest

from hordeai.generation.text import TextGenerationSettings
from hordeai.services.horde import HordeClient
from tests.horde_server import GENERATION_TIMEOUT, wait_for_event
from tests.horde_server import horde_server as horde_server
from tests.text_horde import TEXT_SUBMIT_PATH, text_status, text_status_path, text_submission
from tests.worker_details import expect_text_worker, worker_path


class Interrupted(BaseException):
    pass


def request(count=1):
    return TextGenerationSettings(prompt="Continue this story.", num_texts=count, max_length=80).to_request()


def test_text_status_waits_for_all_requested_results_without_image_endpoints(horde_server):
    job_id = str(UUID(int=501))
    horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
    horde_server.expect("GET", text_status_path(job_id), text_status(["first partial"], done=False))
    horde_server.expect("GET", text_status_path(job_id), text_status(["first complete", "second complete"]))
    expect_text_worker(horde_server)
    results = asyncio.run(HordeClient(poll_interval=0.01).generate_text(request(2), lambda: None))
    assert [result.text for result in results.items] == ["first complete", "second complete"]
    assert [result.model for result in results.items] == ["koboldcpp/model-0", "koboldcpp/model-1"]
    assert [result.seed for result in results.items] == [41, 42]
    assert horde_server.paths("DELETE") == []
    assert horde_server.paths("GET") == [text_status_path(job_id)] * 2 + [worker_path(str(UUID(int=3000)))]
    submitted = horde_server.requests[0]
    assert submitted.body["models"] == []
    assert submitted.body["params"]["n"] == 2


@pytest.mark.parametrize("failure", ["api", "fault", "missing_text", "incomplete"])
def test_text_failures_cancel_the_text_job_not_an_image_job(horde_server, failure):
    job_id = str(UUID(int=502))
    horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
    if failure == "api":
        body = {"message": "Text worker failed", "rc": "WorkerUnavailable"}
        status_code = 503
    elif failure == "fault":
        body, status_code = text_status(faulted=True), 200
    elif failure == "missing_text":
        body, status_code = text_status([None]), 200
    else:
        body, status_code = text_status(["only one"]), 200
    horde_server.expect("GET", text_status_path(job_id), body, status_code=status_code)
    horde_server.expect("DELETE", text_status_path(job_id), text_status())
    with pytest.raises(RuntimeError):
        asyncio.run(HordeClient(poll_interval=0.01).generate_text(request(2 if failure == "incomplete" else 1), lambda: None))
    assert horde_server.paths("DELETE") == [text_status_path(job_id)]


@pytest.mark.parametrize("error_body", [
    {"message": "Invalid API key", "rc": "InvalidAPIKey"},
    {"message": "Input payload validation failed", "errors": {"params": "Invalid token count"}},
])
def test_rejected_text_submission_never_deletes_a_remote_job(horde_server, error_body):
    horde_server.expect("POST", TEXT_SUBMIT_PATH, error_body, status_code=400)
    with pytest.raises(Exception, match=error_body["message"]):
        asyncio.run(HordeClient().generate_text(request(), lambda: None))
    assert horde_server.paths("GET") == []
    assert horde_server.paths("DELETE") == []


@pytest.mark.parametrize("stage", ["post", "status", "poll_delay"])
def test_text_interrupt_preserves_exception_and_cleans_up_once(horde_server, stage):
    job_id = str(UUID(int=503))
    post = horde_server.expect(
        "POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202,
        hold="response" if stage == "post" else None,
    )
    stalled = None
    if stage != "post":
        stalled = horde_server.expect("GET", text_status_path(job_id), text_status(done=False), hold="response" if stage == "status" else None)
    horde_server.expect("DELETE", text_status_path(job_id), text_status())
    interruption = Interrupted("Stop text generation")
    interrupted = Event()
    stop = Event()

    def callback():
        trigger = post.arrived.is_set() if stage == "post" else (stalled.arrived.is_set() if stage == "status" else stop.is_set())
        if trigger:
            interrupted.set()
            raise interruption
        if stage == "poll_delay" and stalled.sent.is_set():
            asyncio.get_running_loop().call_soon(stop.set)

    async def scenario():
        task = asyncio.create_task(HordeClient(poll_interval=60 if stage == "poll_delay" else 0.01).generate_text(request(), callback))
        try:
            if stage == "post":
                await wait_for_event(interrupted)
                assert not task.done()
                post.release.set()
            with pytest.raises(Interrupted) as raised:
                await asyncio.wait_for(task, GENERATION_TIMEOUT)
            assert raised.value is interruption
            if stage == "status":
                await wait_for_event(stalled.disconnected)
        finally:
            post.release.set()
            if stalled:
                stalled.release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert horde_server.paths("DELETE") == [text_status_path(job_id)]


@pytest.mark.parametrize("delete_fails", [False, True])
def test_cancelled_async_text_task_preserves_cancellation_reason(horde_server, delete_fails):
    job_id = str(UUID(int=504))
    horde_server.expect("POST", TEXT_SUBMIT_PATH, text_submission(job_id), status_code=202)
    stalled = horde_server.expect("GET", text_status_path(job_id), text_status(done=False), hold="response")
    horde_server.expect("DELETE", text_status_path(job_id), {"message": "Delete failed", "rc": "CleanupFailed"} if delete_fails else text_status(), status_code=500 if delete_fails else 200)

    async def scenario():
        task = asyncio.create_task(HordeClient().generate_text(request(), lambda: None))
        try:
            await wait_for_event(stalled.arrived)
            task.cancel("User cancelled LLM")
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, GENERATION_TIMEOUT)
            assert raised.value.args == ("User cancelled LLM",)
            await wait_for_event(stalled.disconnected)
        finally:
            stalled.release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert horde_server.paths("DELETE") == [text_status_path(job_id)]

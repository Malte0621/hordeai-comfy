"""Consumer behavior through actual SDK clients and a local HTTP transport."""

import asyncio
from threading import Event
from uuid import UUID

import pytest
import torch
from horde_sdk import ANON_API_KEY
from horde_sdk.ai_horde_api.apimodels import ImageGenerateAsyncRequest, ImageGenerationInputPayload
from PIL import Image

from hordeai.services.horde import HordeClient
from hordeai.utils.images import images_to_tensor
from tests.horde_server import (
    GENERATION_TIMEOUT,
    SUBMIT_PATH,
    check_path,
    encoded_image,
    enqueue_job,
    image_bytes,
    progress,
    status,
    status_path,
    submission,
    wait_for_event,
)
from tests.horde_server import horde_server as horde_server


class ProcessingInterrupted(BaseException):
    """Like a ComfyUI interruption, this must escape ordinary Exception handlers."""


def make_request(*, num_images=1):
    return ImageGenerateAsyncRequest(
        apikey=ANON_API_KEY,
        prompt="A small landscape",
        models=["Deliberate"],
        params=ImageGenerationInputPayload(width=64, height=64, n=num_images),
    )


async def bounded_generate(client, request, callback=lambda: None):
    return await asyncio.wait_for(client.generate(request, callback), GENERATION_TIMEOUT)


async def cancel_remaining(task):
    if not task.done():
        task.cancel()
        done, _ = await asyncio.wait((task,), timeout=GENERATION_TIMEOUT)
        assert task in done, "Generation did not exit after cancellation"
    try:
        task.result()
    except BaseException:
        pass



def expect_stalled_operation(server, job_id, stage):
    if stage == "post":
        return server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202, hold="response")
    server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202)
    if stage == "check":
        return server.expect("GET", check_path(job_id), progress(), hold="response")
    server.expect("GET", check_path(job_id), progress(done=True))
    server.expect("GET", status_path(job_id), status([f"{server.origin}/images/stalled.webp"]))
    return server.expect("GET", "/images/stalled.webp", image_bytes(Image.new("RGB", (2, 2), "red")), hold="body")


def test_pending_generation_downloads_url_and_base64_in_rgb_batch_order(horde_server):
    job_id = str(UUID(int=1))
    remote_image = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    embedded_image = Image.new("L", (2, 2), 128)
    enqueue_job(
        horde_server,
        job_id,
        [f"{horde_server.origin}/images/red.webp", encoded_image(embedded_image)],
        pending=True,
    )
    horde_server.expect("GET", "/images/red.webp", image_bytes(remote_image))

    images = asyncio.run(bounded_generate(HordeClient(poll_interval=0.01), make_request(num_images=2)))
    result = images_to_tensor(images)

    assert result.shape == (2, 2, 2, 3)
    assert result.dtype == torch.float32
    expected = torch.tensor([[1.0, 0.0, 0.0], [128 / 255, 128 / 255, 128 / 255]], dtype=torch.float32)
    torch.testing.assert_close(result, expected[:, None, None, :].expand(2, 2, 2, 3), rtol=0, atol=0)
    assert horde_server.paths("POST") == [SUBMIT_PATH]
    assert horde_server.paths("GET") == [check_path(job_id), check_path(job_id), status_path(job_id), "/images/red.webp"]
    assert horde_server.paths("DELETE") == []


def test_rejected_submission_reports_api_error_without_delete(horde_server):
    horde_server.expect("POST", SUBMIT_PATH, {"message": "API key was rejected", "rc": "InvalidAPIKey"}, status_code=401)

    with pytest.raises(Exception, match="API key was rejected"):
        asyncio.run(bounded_generate(HordeClient(poll_interval=0.01), make_request()))

    assert horde_server.paths("POST") == [SUBMIT_PATH]
    assert horde_server.paths("GET") == []
    assert horde_server.paths("DELETE") == []


@pytest.mark.parametrize("failure", ["check_api", "status_api", "check_fault", "status_fault", "empty_status", "download"])
def test_accepted_job_errors_delete_only_the_failed_job(horde_server, failure):
    job_id = str(UUID(int=2))
    horde_server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202)
    error_message = "Remote worker unavailable"
    if failure == "check_api":
        horde_server.expect("GET", check_path(job_id), {"message": error_message, "rc": "WorkerUnavailable"}, status_code=503)
        match = error_message
    elif failure == "check_fault":
        horde_server.expect("GET", check_path(job_id), progress(done=True, faulted=True))
        match = "(?i)(failed|fault)"
    else:
        horde_server.expect("GET", check_path(job_id), progress(done=True))
        if failure == "status_api":
            horde_server.expect("GET", status_path(job_id), {"message": error_message, "rc": "WorkerUnavailable"}, status_code=503)
            match = error_message
        elif failure == "status_fault":
            horde_server.expect("GET", status_path(job_id), status([encoded_image(Image.new("RGB", (2, 2), "blue"))], faulted=True))
            match = "(?i)(failed|fault)"
        elif failure == "empty_status":
            horde_server.expect("GET", status_path(job_id), status())
            match = "(?i)(generation|image)"
        else:
            horde_server.expect("GET", status_path(job_id), status([f"{horde_server.origin}/images/missing.webp"]))
            horde_server.expect("GET", "/images/missing.webp", {"message": "Image expired"}, status_code=404)
            match = "404"
    horde_server.expect("DELETE", status_path(job_id), status())

    with pytest.raises(Exception, match=match):
        asyncio.run(bounded_generate(HordeClient(poll_interval=0.01), make_request()))

    assert horde_server.paths("DELETE") == [status_path(job_id)]
    expected_gets = [check_path(job_id)]
    if failure not in {"check_api", "check_fault"}:
        expected_gets.append(status_path(job_id))
    if failure == "download":
        expected_gets.append("/images/missing.webp")
    assert horde_server.paths("GET") == expected_gets


@pytest.mark.parametrize(
    ("error_type", "delete_fails"),
    [(ProcessingInterrupted, False), (ProcessingInterrupted, True), (ValueError, False)],
)
def test_processing_interrupt_during_long_poll_delay_preserves_original_error(horde_server, error_type, delete_fails):
    job_id = str(UUID(int=3))
    horde_server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202)
    pending = horde_server.expect("GET", check_path(job_id), progress())
    horde_server.expect(
        "DELETE",
        status_path(job_id),
        {"message": "Delete failed", "rc": "CleanupFailed"} if delete_fails else status(),
        status_code=500 if delete_fails else 200,
    )
    interruption = error_type("ComfyUI stopped this generation")
    stop = Event()
    armed = False

    def check_interrupt():
        nonlocal armed
        if stop.is_set():
            raise interruption
        if pending.sent.is_set() and not armed:
            armed = True
            # Interrupt on the next event-loop turn, not inside the completed check callback.
            asyncio.get_running_loop().call_soon(stop.set)

    with pytest.raises(error_type) as raised:
        asyncio.run(bounded_generate(HordeClient(poll_interval=60.0), make_request(), check_interrupt))

    assert raised.value is interruption
    assert horde_server.paths("GET") == [check_path(job_id)]
    assert horde_server.paths("DELETE") == [status_path(job_id)]


def test_pre_submit_interrupt_never_creates_or_deletes_a_job(horde_server):
    interruption = ProcessingInterrupted("Already interrupted")

    def check_interrupt():
        raise interruption

    with pytest.raises(ProcessingInterrupted) as raised:
        asyncio.run(bounded_generate(HordeClient(poll_interval=0.01), make_request(), check_interrupt))

    assert raised.value is interruption
    assert horde_server.requests == []


@pytest.mark.parametrize("accepted", [True, False])
def test_interrupt_during_post_waits_for_response_and_recovers_job_id(horde_server, accepted):
    job_id = str(UUID(int=4))
    post = horde_server.expect(
        "POST",
        SUBMIT_PATH,
        submission(job_id) if accepted else {"message": "Rejected request", "rc": "InvalidAPIKey"},
        status_code=202 if accepted else 401,
        hold="response",
    )
    if accepted:
        horde_server.expect("DELETE", status_path(job_id), status())
    interrupted = Event()
    interruption = ProcessingInterrupted("Interrupted while submitting")

    def check_interrupt():
        if post.arrived.is_set():
            interrupted.set()
            raise interruption

    async def scenario():
        task = asyncio.create_task(HordeClient(poll_interval=0.01).generate(make_request(), check_interrupt))
        try:
            await wait_for_event(post.arrived)
            await wait_for_event(interrupted)
            assert not post.sent.is_set()
            assert not task.done(), "Interruption must wait for the recoverable POST response"
            assert horde_server.paths("DELETE") == []
            post.release.set()
            with pytest.raises(ProcessingInterrupted) as raised:
                await asyncio.wait_for(task, GENERATION_TIMEOUT)
            assert raised.value is interruption
        finally:
            post.release.set()
            await cancel_remaining(task)

    asyncio.run(scenario())

    assert post.sent.is_set(), "Submission response must remain recoverable after the interrupt"
    assert horde_server.paths("GET") == []
    assert horde_server.paths("DELETE") == ([status_path(job_id)] if accepted else [])


@pytest.mark.parametrize("stage", ["check", "download"])
def test_callback_interrupt_stops_stalled_http_before_delete(horde_server, stage):
    job_id = str(UUID(int=5))
    stalled = expect_stalled_operation(horde_server, job_id, stage)
    horde_server.expect("DELETE", status_path(job_id), status())
    interruption = ProcessingInterrupted(f"Interrupted during {stage}")
    ready = stalled.headers_sent if stage == "download" else stalled.arrived

    def check_interrupt():
        if ready.is_set():
            raise interruption

    async def scenario():
        try:
            with pytest.raises(ProcessingInterrupted) as raised:
                await bounded_generate(HordeClient(poll_interval=0.01), make_request(), check_interrupt)
            assert raised.value is interruption
            await wait_for_event(stalled.disconnected)
            assert not stalled.sent.is_set(), "The blocked HTTP response must not be awaited to completion"
        finally:
            stalled.release.set()

    asyncio.run(scenario())

    assert horde_server.paths("DELETE") == [status_path(job_id)]
    if stage == "check":
        assert horde_server.paths("GET") == [check_path(job_id)]
    else:
        assert horde_server.paths("GET") == [check_path(job_id), status_path(job_id), "/images/stalled.webp"]


@pytest.mark.parametrize(("stage", "delete_fails"), [("post", False), ("check", False), ("download", False), ("check", True)])
def test_asyncio_task_cancellation_preserves_reason_and_cleans_up_once(horde_server, stage, delete_fails):
    job_id = str(UUID(int=6))
    stalled = expect_stalled_operation(horde_server, job_id, stage)
    horde_server.expect(
        "DELETE",
        status_path(job_id),
        {"message": "Delete failed", "rc": "CleanupFailed"} if delete_fails else status(),
        status_code=500 if delete_fails else 200,
    )
    ready = stalled.headers_sent if stage == "download" else stalled.arrived

    async def scenario():
        task = asyncio.create_task(HordeClient(poll_interval=0.01).generate(make_request(), lambda: None))
        try:
            await wait_for_event(ready)
            task.cancel("Caller cancelled generation")
            if stage == "post":
                # The POST may already have created a remote job; its response supplies the ID for cleanup.
                stalled.release.set()
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, GENERATION_TIMEOUT)
            assert raised.value.args == ("Caller cancelled generation",)
            assert task.cancelled()
            if stage != "post":
                await wait_for_event(stalled.disconnected)
                assert not stalled.sent.is_set()
        finally:
            stalled.release.set()
            await cancel_remaining(task)

    asyncio.run(scenario())

    assert horde_server.paths("DELETE") == [status_path(job_id)]
    if stage == "post":
        assert stalled.sent.is_set()
        assert horde_server.paths("GET") == []


def test_second_task_cancellation_during_delete_keeps_first_reason_and_single_cleanup(horde_server):
    job_id = str(UUID(int=9))
    stalled = expect_stalled_operation(horde_server, job_id, "check")
    deletion = horde_server.expect("DELETE", status_path(job_id), status(), hold="response")

    async def scenario():
        task = asyncio.create_task(HordeClient(poll_interval=0.01).generate(make_request(), lambda: None))
        try:
            await wait_for_event(stalled.arrived)
            task.cancel("First cancellation")
            await wait_for_event(deletion.arrived)
            task.cancel("Second cancellation")
            deletion.release.set()
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, GENERATION_TIMEOUT)
            assert raised.value.args == ("First cancellation",)
            assert task.cancelled()
            await wait_for_event(stalled.disconnected)
        finally:
            stalled.release.set()
            deletion.release.set()
            await cancel_remaining(task)

    asyncio.run(scenario())

    assert deletion.sent.is_set(), "The second cancellation must not abandon the cleanup HTTP response"
    assert horde_server.paths("GET") == [check_path(job_id)]
    assert horde_server.paths("DELETE") == [status_path(job_id)]


def test_successive_runs_do_not_reuse_a_failed_jobs_id(horde_server):
    failed_id = str(UUID(int=7))
    success_id = str(UUID(int=8))
    horde_server.expect("POST", SUBMIT_PATH, submission(failed_id), status_code=202)
    horde_server.expect("GET", check_path(failed_id), {"message": "First job failed", "rc": "WorkerUnavailable"}, status_code=503)
    horde_server.expect("DELETE", status_path(failed_id), status())
    enqueue_job(horde_server, success_id, [encoded_image(Image.new("RGB", (2, 2), "blue"))])
    client = HordeClient(poll_interval=0.01)
    interruption = ProcessingInterrupted("No third job")

    def pre_submit_interrupt():
        raise interruption

    async def scenario():
        with pytest.raises(Exception, match="First job failed"):
            await bounded_generate(client, make_request())
        images = await bounded_generate(client, make_request())
        assert images[0].convert("RGB").getpixel((0, 0)) == (0, 0, 255)
        with pytest.raises(ProcessingInterrupted) as raised:
            await bounded_generate(client, make_request(), pre_submit_interrupt)
        assert raised.value is interruption

    asyncio.run(scenario())

    assert horde_server.paths("POST") == [SUBMIT_PATH, SUBMIT_PATH]
    assert horde_server.paths("DELETE") == [status_path(failed_id)]
    assert horde_server.paths("GET") == [check_path(failed_id), check_path(success_id), status_path(success_id)]

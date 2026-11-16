"""Local HTTP transport for tests against the real Horde SDK clients."""

import asyncio
import base64
import json
import select
import socket
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Event, Lock, Thread
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from horde_sdk.ai_horde_api import endpoints
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncResponse,
    ImageGenerateCheckResponse,
    ImageGenerateStatusResponse,
    ImageGeneration,
)
from horde_sdk.ai_horde_api.apimodels import base as request_base
from PIL import Image

SUBMIT_PATH = "/v2/generate/async"
SYNC_TIMEOUT = 5.0
GENERATION_TIMEOUT = 10.0


def check_path(job_id: str) -> str:
    return f"/v2/generate/check/{job_id}"


def status_path(job_id: str) -> str:
    return f"/v2/generate/status/{job_id}"


def submission(job_id: str) -> dict[str, Any]:
    return ImageGenerateAsyncResponse(id=job_id, kudos=1.0).model_dump(mode="json", by_alias=True, exclude_none=True)


def progress(*, done: bool = False, faulted: bool = False, count: int = 1) -> dict[str, Any]:
    return ImageGenerateCheckResponse(
        finished=count if done else 0,
        processing=0 if done else count,
        restarted=0,
        waiting=0,
        done=done,
        faulted=faulted,
        wait_time=0,
        queue_position=0,
        kudos=1.0,
        is_possible=True,
    ).model_dump(mode="json", by_alias=True)


def generation(image: str, index: int = 1) -> dict[str, Any]:
    return ImageGeneration(
        id=str(UUID(int=1000 + index)),
        img=image,
        seed=str(index),
        censored=False,
        model="Deliberate",
        state="ok",
        worker_id=str(UUID(int=2000)),
        worker_name="local worker",
    ).model_dump(mode="json", by_alias=True, exclude_none=True)


def status(images: list[str] | None = None, *, faulted: bool = False) -> dict[str, Any]:
    images = [] if images is None else images
    payload = progress(done=True, faulted=faulted, count=len(images))
    payload["generations"] = [generation(image, index) for index, image in enumerate(images, start=1)]
    return ImageGenerateStatusResponse.model_validate(payload).model_dump(mode="json", by_alias=True, exclude_none=True)


def image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="WEBP", lossless=True)
    return buffer.getvalue()


def encoded_image(image: Image.Image) -> str:
    return base64.b64encode(image_bytes(image)).decode("ascii")


@dataclass(slots=True)
class ResponsePlan:
    body: bytes
    status: int
    content_type: str
    hold: Literal["response", "body"] | None = None
    arrived: Event = field(default_factory=Event)
    headers_sent: Event = field(default_factory=Event)
    sent: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    disconnected: Event = field(default_factory=Event)
    finished: Event = field(default_factory=Event)


@dataclass(frozen=True, slots=True)
class RequestRecord:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any


class LocalHordeServer:
    def __init__(self) -> None:
        self._lock = Lock()
        self._routes: dict[tuple[str, str], deque[ResponsePlan]] = defaultdict(deque)
        self._plans: list[ResponsePlan] = []
        self._requests: list[RequestRecord] = []
        self.unexpected: list[RequestRecord] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                self.respond()

            def do_GET(self) -> None:
                self.respond()

            def do_DELETE(self) -> None:
                self.respond()

            def wait_for_release(self, plan: ResponsePlan) -> bool:
                deadline = time.monotonic() + 15.0
                while not plan.release.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    readable, _, _ = select.select([self.connection], [], [], min(0.05, remaining))
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        plan.disconnected.set()
                        return False
                return True

            def respond(self) -> None:
                url = urlsplit(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else None
                record = RequestRecord(self.command, url.path, parse_qs(url.query), dict(self.headers), body)
                plan = owner.take_response(record)
                plan.arrived.set()
                self.close_connection = True
                try:
                    if plan.hold == "response" and not self.wait_for_release(plan):
                        return
                    self.send_response(plan.status)
                    self.send_header("Content-Type", plan.content_type)
                    self.send_header("Content-Length", str(len(plan.body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.flush()
                    plan.headers_sent.set()
                    if plan.hold == "body" and not self.wait_for_release(plan):
                        return
                    self.wfile.write(plan.body)
                    self.wfile.flush()
                    plan.sent.set()
                except OSError:
                    plan.disconnected.set()
                finally:
                    plan.finished.set()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self.origin = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    @property
    def requests(self) -> list[RequestRecord]:
        with self._lock:
            return list(self._requests)

    def paths(self, method: str) -> list[str]:
        return [request.path for request in self.requests if request.method == method]

    def expect(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | bytes,
        *,
        status_code: int = 200,
        hold: Literal["response", "body"] | None = None,
    ) -> ResponsePlan:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        plan = ResponsePlan(raw, status_code, "image/webp" if isinstance(body, bytes) else "application/json", hold)
        if hold is None:
            plan.release.set()
        with self._lock:
            self._routes[method, path].append(plan)
            self._plans.append(plan)
        return plan

    def take_response(self, record: RequestRecord) -> ResponsePlan:
        with self._lock:
            self._requests.append(record)
            route = self._routes[record.method, record.path]
            if route:
                return route.popleft()
            self.unexpected.append(record)
        return ResponsePlan(
            json.dumps({"message": f"Unexpected {record.method} {record.path}", "rc": "UnexpectedRequest"}).encode(),
            500,
            "application/json",
        )

    def close(self) -> None:
        for plan in self._plans:
            plan.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=SYNC_TIMEOUT)
        assert not self._thread.is_alive(), "Local HTTP server did not shut down"
        deadline = time.monotonic() + SYNC_TIMEOUT
        for plan in self._plans:
            if plan.arrived.is_set():
                assert plan.finished.wait(max(0.0, deadline - time.monotonic())), "Local HTTP handler did not shut down"


def enqueue_job(server: LocalHordeServer, job_id: str, images: list[str], *, pending: bool = False) -> None:
    server.expect("POST", SUBMIT_PATH, submission(job_id), status_code=202)
    if pending:
        server.expect("GET", check_path(job_id), progress(count=len(images)))
    server.expect("GET", check_path(job_id), progress(done=True, count=len(images)))
    server.expect("GET", status_path(job_id), status(images))


async def wait_for_event(event: Event) -> None:
    assert await asyncio.to_thread(event.wait, SYNC_TIMEOUT), "Expected synchronization event did not arrive"


@pytest.fixture
def horde_server(monkeypatch):
    server = LocalHordeServer()
    # SDK requests resolve their endpoint through apimodels.base, not the client's base_url setter.
    monkeypatch.setattr(request_base, "AI_HORDE_BASE_URL", f"{server.origin}/")
    monkeypatch.setattr(endpoints, "AI_HORDE_BASE_URL", f"{server.origin}/")
    try:
        yield server
    finally:
        server.close()
    assert not server.unexpected, f"Unplanned HTTP requests: {server.unexpected}"

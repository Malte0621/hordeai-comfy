"""Typed text responses for the isolated Horde HTTP fixture."""

from uuid import UUID

from horde_sdk.ai_horde_api.apimodels import GenerationKobold, TextGenerateAsyncResponse, TextGenerateStatusResponse

from tests.horde_server import progress

TEXT_SUBMIT_PATH = "/v2/generate/text/async"


def text_status_path(job_id):
    return f"/v2/generate/text/status/{job_id}"


def text_submission(job_id):
    return TextGenerateAsyncResponse(id=job_id, kudos=1).model_dump(mode="json", by_alias=True, exclude_none=True)


def text_status(texts=(), *, done=True, faulted=False):
    payload = progress(done=done, faulted=faulted, count=len(texts) or 1)
    payload["generations"] = [
        GenerationKobold(
            id_=str(UUID(int=4000 + index)),
            text=text,
            model=f"koboldcpp/model-{index}",
            seed=index + 41,
            state="ok",
            worker_id=str(UUID(int=3000)),
            worker_name="local text worker",
        ).model_dump(mode="json", exclude_none=True)
        for index, text in enumerate(texts)
    ]
    return TextGenerateStatusResponse.model_validate(payload).model_dump(mode="json", by_alias=True, exclude_none=True)

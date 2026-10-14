"""ReLoCache routes composed with native vLLM serving handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.chat_completion.api_router import create_chat_completion
from vllm.entrypoints.openai.completion.api_router import create_completion
from vllm.entrypoints.openai.responses.api_router import create_responses

from relocache.content import ContentError
from relocache.webchat import mount_webchat

from .content import ContentService
from .protocol import (
    CompileInput,
    ContentInput,
    ReusableChatRequest,
    ReusableCompletionRequest,
    ReusableResponsesRequest,
)

router = APIRouter()


async def content_error_handler(request: Request, error: ContentError):
    return JSONResponse(
        status_code=404 if error.code == "unknown_content" else 400,
        content={
            "error": {
                "message": str(error),
                "type": "invalid_request_error",
                "code": error.code,
            }
        },
    )


@router.post("/v1/reusable_contents/register")
async def register_content(payload: ContentInput, request: Request):
    service = request.app.state.relocache
    content = await service.register(payload.model_dump(exclude_none=True))
    return service.public_content(content)


@router.post("/v1/reusable_contents/compile")
async def compile_content(payload: CompileInput, request: Request):
    service = request.app.state.relocache
    return await service.compile(payload.model_dump(exclude_none=True))


@router.get("/v1/reusable_contents/{content_id}")
async def get_content(content_id: str, request: Request):
    service = request.app.state.relocache
    return service.public_content(await service.resolve({"id": content_id}))


@router.get("/v1/reusable_contents/{content_id}/status")
async def content_status(content_id: str, request: Request):
    return await request.app.state.relocache.status(content_id)


@router.get("/v1/relocache")
async def deployment(request: Request):
    return request.app.state.relocache.deployment()


@router.post("/v1/reusable_contents/flush")
async def flush_contents(request: Request):
    await request.app.state.relocache.flush()
    return {"flushed": True}


@router.post("/v1/completions")
async def completion(payload: ReusableCompletionRequest, request: Request):
    return await create_completion(payload, request)


@router.post("/v1/chat/completions")
async def chat(payload: ReusableChatRequest, request: Request):
    return await create_chat_completion(payload, request)


@router.post("/v1/responses")
async def responses(payload: ReusableResponsesRequest, request: Request):
    return await create_responses(payload, request)


class ReLoCacheEndpoints:
    name = "relocache"
    required_tasks = ("generate",)

    def attach_router(self, app):
        # Replace request schemas while retaining native generation handlers.
        replaced = {"/v1/completions", "/v1/chat/completions", "/v1/responses"}
        app.router.routes[:] = [
            route
            for route in app.router.routes
            if not (
                getattr(route, "path", None) in replaced
                and "POST" in getattr(route, "methods", set())
            )
        ]
        app.include_router(router)
        mount_webchat(app)
        app.add_exception_handler(ContentError, content_error_handler)

    async def init_state(self, engine_client, state, args):
        config = engine_client.vllm_config.additional_config["relocache"]
        state.relocache = ContentService(engine_client, config)
        state.online_renderer.renderer.service = state.relocache

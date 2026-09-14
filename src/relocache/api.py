"""ReLoCache routes composed with native vLLM serving handlers."""

from functools import partial

from fastapi import APIRouter, HTTPException, Request
from vllm.entrypoints.openai.chat_completion.api_router import create_chat_completion
from vllm.entrypoints.openai.responses.api_router import create_responses

from .content import ContentStore
from .protocol import (
    ContentInput,
    ReusableChatRequest,
    ReusableResponsesRequest,
)
from .rendering import render_reusable_chat

router = APIRouter()


def public_content(content):
    return {
        "content_id": content["content_id"],
        "input_ids": content["input_ids"],
        "logical_length": len(content["input_ids"]),
    }


@router.post("/v1/reusable_contents/register")
async def register_content(payload: ContentInput, request: Request):
    return public_content(
        await request.app.state.relocache.register(
            payload.model_dump(exclude_none=True),
        )
    )


@router.post("/v1/reusable_contents/compile")
async def compile_content(payload: ContentInput, request: Request):
    try:
        return public_content(
            await request.app.state.relocache.compile(
                payload.model_dump(exclude_none=True),
            )
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.get("/v1/reusable_contents/{content_id}")
async def get_content(content_id: str, request: Request):
    try:
        return public_content(await request.app.state.relocache.resolve({"id": content_id}))
    except KeyError as error:
        raise HTTPException(404, f"Unknown content: {content_id}") from error


@router.post("/v1/reusable_contents/flush")
async def flush_contents(request: Request):
    await request.app.state.relocache.flush()
    return {"flushed": True}


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
        # Replace these two schemas while retaining vLLM's serving handlers.
        replaced = {"/v1/chat/completions", "/v1/responses"}
        app.router.routes[:] = [
            route
            for route in app.router.routes
            if not (
                getattr(route, "path", None) in replaced
                and "POST" in getattr(route, "methods", set())
            )
        ]
        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        config = engine_client.vllm_config.additional_config["relocache"]
        state.relocache = ContentStore(engine_client, config)
        state.online_renderer.chat_input_renderer = partial(
            render_reusable_chat,
            resolver=state.relocache.resolve,
        )

"""Serve the demo UI alongside an engine or through a streaming HTTP proxy."""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ASSETS = Path(__file__).with_name("webchat_assets")


def mount_webchat(app):
    @app.get("/demo", include_in_schema=False)
    async def webchat():
        return FileResponse(ASSETS / "index.html")

    app.mount("/demo/assets", StaticFiles(directory=ASSETS), name="webchat-assets")


def create_app(api_base, *, api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(
            base_url=api_base.rstrip("/") + "/", headers=headers, timeout=None, trust_env=False
        ) as client:
            app.state.upstream = client
            yield

    app = FastAPI(title="ReLoCache Web Chat", lifespan=lifespan)
    mount_webchat(app)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/demo")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"], include_in_schema=False)
    async def proxy(path: str, request: Request):
        client = request.app.state.upstream
        upstream_request = client.build_request(
            request.method,
            f"v1/{path}",
            params=request.query_params,
            content=await request.body(),
            headers={"Content-Type": request.headers.get("content-type", "application/json")},
        )
        try:
            response = await client.send(upstream_request, stream=True)
        except httpx.RequestError as error:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Cannot reach inference service: {error}"}},
            )

        async def stream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            stream(),
            status_code=response.status_code,
            headers={
                "Content-Type": response.headers.get("content-type", "application/json"),
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app

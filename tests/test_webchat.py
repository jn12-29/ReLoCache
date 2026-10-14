"""Exercise the browser entry point and the engine-independent streaming proxy."""

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from relocache.webchat import create_app


def test_demo_assets_are_available_without_an_engine():
    with TestClient(create_app("http://engine")) as client:
        page = client.get("/demo")
        assert page.status_code == 200
        assert "TTFT" in page.text
        assert client.get("/demo/assets/presets.mjs").status_code == 200
        assert "星河科技" in client.get("/demo/assets/presets.mjs").text
        assert client.get("/demo/assets/handbook.mjs").status_code == 200
        assert client.get("/demo/assets/app.mjs").status_code == 200
        assert client.get("/demo/assets/stream.mjs").status_code == 200
        assert client.get("/demo/assets/style.css").status_code == 200
        assert client.get("/").url.path == "/demo"


def test_proxy_preserves_body_queries_auth_errors_and_closes_streams():
    async def check():
        closed = []
        requests = []
        reset_results = iter([False, True])

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                closed.append(True)

        def handle(request):
            requests.append(request)
            if request.url.path == "/reset_prefix_cache":
                return httpx.Response(200, json={"success": next(reset_results)})
            if request.method == "GET":
                return httpx.Response(401, json={"error": {"message": "denied"}})
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Stream()
            )

        app = create_app("http://engine", api_key="demo-key")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                base_url="http://engine/",
                headers={"Authorization": "Bearer demo-key"},
                transport=httpx.MockTransport(handle),
            ) as upstream:
                app.state.upstream = upstream
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://demo"
                ) as client:
                    payload = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
                    response = await client.post("/v1/chat/completions?example=1", json=payload)
                    assert response.status_code == 200
                    assert response.headers["content-type"] == "text/event-stream"
                    assert response.headers["x-accel-buffering"] == "no"
                    assert response.text.endswith("data: [DONE]\n\n")
                    assert closed
                    assert requests[0].url == "http://engine/v1/chat/completions?example=1"
                    assert json.loads(requests[0].content) == payload
                    assert requests[0].headers["authorization"] == "Bearer demo-key"
                    error = await client.get("/v1/models")
                    assert error.status_code == 401
                    assert error.json()["error"]["message"] == "denied"
                    failed_reset = await client.post("/reset_prefix_cache")
                    assert failed_reset.status_code == 200
                    assert failed_reset.json() == {"success": False}
                    reset = await client.post("/reset_prefix_cache")
                    assert reset.json() == {"success": True}
                    assert requests[-1].url == "http://engine/reset_prefix_cache"
                    assert requests[-1].headers["authorization"] == "Bearer demo-key"
                    assert requests[-1].url.query == b""

    asyncio.run(check())


def test_disconnected_engine_returns_a_browser_visible_error():
    async def check():
        def handle(request):
            raise httpx.ConnectError("connection refused", request=request)

        app = create_app("http://engine")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handle), base_url="http://engine/"
            ) as upstream:
                app.state.upstream = upstream
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://demo"
                ) as client:
                    response = await client.get("/v1/models")
                    assert response.status_code == 502
                    assert "connection refused" in response.json()["error"]["message"]

    asyncio.run(check())

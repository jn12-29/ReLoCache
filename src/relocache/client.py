"""Lightweight asynchronous HTTP access; no engine imports."""

import asyncio
from time import perf_counter

import httpx


class Client:
    def __init__(self, api_base, *, timeout=None):
        self.http = httpx.AsyncClient(base_url=api_base, timeout=timeout, trust_env=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def post(self, endpoint, payload):
        started = perf_counter()
        response = await self.http.post(endpoint, json=payload)
        response.raise_for_status()
        result = response.json()
        result["http_latency_seconds"] = perf_counter() - started
        return result

    async def register(self, content_id, *, input_ids=None, text=None):
        source = {"text": text} if input_ids is None else {"input_ids": input_ids}
        return await self.post(
            "/v1/reusable_contents/register", {"content_id": content_id, **source}
        )

    async def compile(self, content_id, *, input_ids=None, text=None, persist=False):
        source = {"text": text} if input_ids is None else {"input_ids": input_ids}
        return await self.post(
            "/v1/reusable_contents/compile",
            {
                "content_id": content_id,
                **source,
                "persist": persist,
            },
        )

    async def flush(self):
        return await self.post("/v1/reusable_contents/flush", {})


async def completed_requests(request, items, concurrency):
    """Refill a bounded request window as individual requests finish."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    items = iter(items)
    pending = set()
    try:
        for _ in range(concurrency):
            item = next(items, None)
            if item is None:
                break
            pending.add(asyncio.create_task(request(item)))
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                pending.remove(task)
                item = next(items, None)
                if item is not None:
                    pending.add(asyncio.create_task(request(item)))
                yield result
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

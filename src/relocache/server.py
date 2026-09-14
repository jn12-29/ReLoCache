"""Manage a ReLoCache subprocess or connect to an existing service."""

import asyncio
import json
import socket
import sys
from pathlib import Path
from time import perf_counter

import httpx

from .client import Client


class Server:
    def __init__(self, config, output, *, api_base=None):
        self.config = config
        self.output = Path(output)
        self.api_base = api_base
        self.process = self.log = self.client = None
        self.startup_seconds = 0.0

    async def __aenter__(self):
        started = perf_counter()
        if self.api_base is None:
            self.output.mkdir(parents=True, exist_ok=True)
            path = self.output / "server.json"
            path.write_text(json.dumps(self.config, indent=2) + "\n")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [
                sys.executable,
                "-m",
                "relocache.cli",
                "serve",
                "--config",
                str(path.resolve()),
                "--port",
                str(port),
            ]
            self.log = (self.output / "server.log").open("w")
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=self.log,
                stderr=asyncio.subprocess.STDOUT,
            )
            (self.output / "server-process.json").write_text(
                json.dumps(
                    {
                        "pid": self.process.pid,
                        "command": command,
                    },
                    indent=2,
                )
                + "\n"
            )
            self.api_base = f"http://127.0.0.1:{port}"
        self.client = Client(self.api_base)
        try:
            while True:
                if self.process is not None and self.process.returncode is not None:
                    raise RuntimeError(
                        f"Service exited with {self.process.returncode}; see server.log"
                    )
                try:
                    response = await self.client.http.get("/health", timeout=2)
                    response.raise_for_status()
                    break
                except (httpx.ConnectError, httpx.TimeoutException):
                    await asyncio.sleep(1)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        self.startup_seconds = perf_counter() - started
        return self.client

    async def __aexit__(self, *args):
        if self.client is not None:
            await self.client.http.aclose()
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=60)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.log.close()

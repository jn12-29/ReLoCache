"""Compilation coordination independent of execution queues and KV ownership."""

import time


class CompilationTracker:
    def __init__(self):
        self.active = {}
        self.started = {}

    def start(self, compiled_id, request_id):
        self.active.setdefault(compiled_id, set()).add(request_id)
        self.started[request_id] = time.monotonic()

    def finish(self, compiled_id, request_id):
        pending = self.active[compiled_id]
        pending.remove(request_id)
        if not pending:
            del self.active[compiled_id]
        return time.monotonic() - self.started.pop(request_id)


def compilation_action(*, missing, pending, auto_compile):
    if missing:
        return "compile" if auto_compile else "fail"
    return "wait" if pending else "ready"

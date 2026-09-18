"""Source snapshots and their optional persistent catalog."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


class ContentError(Exception):
    def __init__(self, message, *, code="invalid_content"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Content:
    content_id: str
    input_ids: tuple[int, ...]


def wrapped_token_ids(input_ids, pre_length=0, post_length=0, *, placeholder=0):
    return [placeholder] * pre_length + list(input_ids) + [placeholder] * post_length


class ContentStore:
    def __init__(self, writer_id, *, metadata_dir=None):
        self.contents: dict[str, Content] = {}
        self.metadata_dir = (
            Path(metadata_dir) / f"writer-{quote(writer_id, safe='')}"
            if metadata_dir is not None
            else None
        )
        if self.metadata_dir is not None:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def metadata_path(self, content_id):
        return self.metadata_dir / f"content-{quote(content_id, safe='')}.json"

    def register(self, content_id, input_ids):
        if not input_ids:
            raise ContentError("Reusable content must contain at least one token")
        content = Content(content_id, tuple(input_ids))
        if self.contents.get(content_id) == content:
            return self.contents[content_id]
        self.contents[content_id] = content
        if self.metadata_dir is not None:
            self.metadata_path(content_id).write_text(
                json.dumps({"content_id": content_id, "input_ids": content.input_ids}) + "\n"
            )
        return content

    def resolve(self, content_id):
        if content_id in self.contents:
            return self.contents[content_id]
        if self.metadata_dir is not None:
            try:
                metadata = json.loads(self.metadata_path(content_id).read_text())
            except FileNotFoundError:
                pass
            else:
                content = Content(content_id, tuple(metadata["input_ids"]))
                self.contents[content_id] = content
                return content
        raise ContentError(f"Unknown content: {content_id}", code="unknown_content")

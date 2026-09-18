"""Compiled content occurrences in native request coordinates."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ReusableSpan:
    start: int
    length: int
    content: dict[str, Any]

"""Logical PIC occurrences over shared storage slots."""

from dataclasses import dataclass


@dataclass
class PICContentLayout:
    length: int
    # Missing layers use contiguous identity positions.
    entry_positions: dict[str, list[float]]


@dataclass
class PICOccurrence:
    start: int
    content_index: int
    # (storage slot, entry offset, allocated length), including tail padding.
    segments: list[tuple[int, int, int]]


@dataclass
class PICLayout:
    canonical_length: int
    storage_length: int
    # (storage slot, canonical position, length), for cached and live Dynamic Content.
    dynamic_content_ranges: list[tuple[int, int, int]]
    occurrences: list[PICOccurrence]
    contents: list[PICContentLayout]
    layer_names: list[str]

    @property
    def precomputed_length(self) -> int:
        return max(
            start + length
            for occurrence in self.occurrences
            for start, _, length in occurrence.segments
        )


def dynamic_intervals(canonical_length, spans):
    """Canonical intervals remaining after reusable occurrences are removed."""
    ranges = []
    start = 0
    for position, length in spans:
        if start < position:
            ranges.append((start, position))
        start = position + length
    if start < canonical_length:
        ranges.append((start, canonical_length))
    return ranges


def merge_segments(segments):
    merged = []
    for start, offset, length in sorted(segments, key=lambda segment: segment[1]):
        if (
            merged
            and merged[-1][0] + merged[-1][2] == start
            and merged[-1][1] + merged[-1][2] == offset
        ):
            previous_start, previous_offset, previous_length = merged[-1]
            merged[-1] = (previous_start, previous_offset, previous_length + length)
        else:
            merged.append((start, offset, length))
    return merged


def map_dynamic_intervals(intervals, cached_tokens, prefix_slots):
    mapped = []
    offset = 0
    for start, end in intervals:
        while start < end:
            length = end - start
            if offset < cached_tokens:
                length = min(length, cached_tokens - offset)
                storage_start = offset
            else:
                storage_start = offset + prefix_slots
            mapped.append((storage_start, start, length))
            offset += length
            start += length
    return mapped


def reused_token_count(layout):
    """Count distinct valid storage rows, excluding padding and repeat references."""
    ranges = sorted(
        (start, start + min(capacity, layout.contents[occurrence.content_index].length - offset))
        for occurrence in layout.occurrences
        for start, offset, capacity in occurrence.segments
    )
    end = total = 0
    for start, stop in ranges:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total

"""Map native storage coordinates into Reader positions before model execution."""

from vllm.triton_utils import tl, triton


@triton.jit
def _map_positions(storage, logical, request_indices, query_offsets, maps, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    mapping = tl.full((), 0, tl.int64)
    if maps is not None:
        index = tl.load(request_indices + request)
        mapping = tl.load(maps + index)
    start = tl.load(query_offsets + request)
    end = tl.load(query_offsets + request + 1)
    for offset in tl.range(start, end, BLOCK):
        token = offset + tl.arange(0, BLOCK)
        mask = token < end
        position = tl.load(storage + token, mask=mask, other=0)
        if mapping != 0:
            position = tl.load(mapping.to(tl.pointer_type(tl.int64)) + position, mask=mask, other=0)
        tl.store(logical + token, position, mask=mask)


def map_positions(storage, logical, request_indices, query_offsets, maps):
    _map_positions[(request_indices.shape[0],)](
        storage, logical, request_indices, query_offsets, maps, BLOCK=256
    )

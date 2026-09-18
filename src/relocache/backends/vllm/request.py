"""Native request specialization for independently compiled content."""

from dataclasses import asdict

from vllm.v1.request import Request

from .cache import dynamic_content_ranges, hash_reusable_request, pack_prompt_embeds
from .transport import decode_input
from .types import ReusableSpan


class PICRequest(Request):
    def __init__(self, *args, extension=None, reusable_spans=None, **kwargs):
        data = decode_input(extension) if extension is not None else None
        self.pic_compile = data["contents"][0] if data and data["compile"] else None
        self.reusable_spans = reusable_spans or (
            [
                ReusableSpan(start, length, data["contents"][index])
                for start, length, index in data["spans"]
            ]
            if data and not data["compile"]
            else []
        )
        self.canonical_prompt_token_ids = kwargs.get("prompt_token_ids", args[1] if args else None)
        self.pic_layout = None
        self._pic_dynamic_content_hashes = []
        self._pic_hash_block_size = 0
        mask = kwargs.get("prompt_is_token_ids")
        self._pic_dynamic_content_prompt_token_ids = (
            [
                token if mask is None or mask[index] else 0
                for start, end in dynamic_content_ranges(self)
                for index, token in enumerate(self.canonical_prompt_token_ids[start:end], start)
            ]
            if self.reusable_spans
            else []
        )
        self.pic_metrics = {
            "referenced_tokens": sum(span.length for span in self.reusable_spans),
            "occurrences": len(self.reusable_spans),
            "auto_compiles": 0,
            "wait_seconds": 0.0,
            "prefill_tokens_executed": 0,
        }
        self.wait_started = None
        super().__init__(*args, extension=extension, **kwargs)
        self.internal = bool(data and data["internal"])
        self.operation_id = data["operation_id"] if data else None
        self.required_cached_tokens = sum(
            c["groups"][0]["num_slots"]
            for c in {s.content["compiled_id"]: s.content for s in self.reusable_spans}.values()
        )

    def get_prompt_embeds(self):
        return pack_prompt_embeds(self)

    def install_layout(self):
        self.required_cached_tokens = self.pic_layout.precomputed_length
        self.worker_metadata = asdict(self.pic_layout)


def pic_hasher(native_hasher, block_size, hash_fn):
    def hash_request(request):
        if request.pic_compile is not None:
            return request.pic_compile["groups"][0]["block_hashes"][len(request.block_hashes) :]
        if request.reusable_spans:
            return hash_reusable_request(request, block_size, hash_fn)
        return native_hasher(request)

    return hash_request

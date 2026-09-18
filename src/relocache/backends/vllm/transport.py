"""Value-only PIC metadata carried by native SamplingParams.extra_args."""

from .compiler import source_signature


def input_metadata(contents, spans=(), *, compile=False, internal=False, operation_id=None):
    return {
        "contents": [
            {
                **{key: value for key, value in content.items() if key != "writer_soft_tokens"},
                "signature": content.get("signature") or source_signature(content),
            }
            for content in contents
        ],
        "spans": list(spans),
        "compile": compile,
        "internal": internal,
        "operation_id": operation_id,
    }

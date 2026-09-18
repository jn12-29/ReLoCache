"""Keep PIC payloads opaque to native IPC while retaining zero-copy tensors."""

from vllm.v1.engine import RequestExtension


def encode_input(contents, spans=(), *, compile=False, internal=False, operation_id=None):
    tensors = {}
    descriptors = []
    for index, content in enumerate(contents):
        descriptor = {key: value for key, value in content.items() if key != "writer_soft_tokens"}
        descriptor["writer_soft_tokens"] = {}
        for name, tensor in content.get("writer_soft_tokens", {}).items():
            key = f"{index}:{name}"
            tensors[key] = tensor
            descriptor["writer_soft_tokens"][name] = key
        descriptors.append(descriptor)
    return RequestExtension(
        {
            "contents": descriptors,
            "spans": list(spans),
            "compile": compile,
            "internal": internal,
            "operation_id": operation_id,
        },
        tensors,
    )


def decode_input(extension):
    data = dict(extension.data)
    data["contents"] = [
        {
            **content,
            "writer_soft_tokens": {
                name: extension.tensors[key]
                for name, key in content.get("writer_soft_tokens", {}).items()
            },
        }
        for content in data["contents"]
    ]
    return data

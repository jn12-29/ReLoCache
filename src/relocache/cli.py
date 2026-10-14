"""Launch native vLLM serving or a standalone Web Chat."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    webchat = subparsers.add_parser("webchat", help="Serve Web Chat for an existing API server")
    webchat.add_argument("--api-base", default="http://127.0.0.1:8000")
    webchat.add_argument("--host", default="127.0.0.1")
    webchat.add_argument("--port", type=int, default=8080)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--cuda-visible-devices")
    serve.add_argument("--data-parallel-size", type=int)
    serve.add_argument("--tensor-parallel-size", type=int)
    args = parser.parse_args()
    if args.command == "webchat":
        import uvicorn

        from .webchat import create_app

        app = create_app(args.api_base, api_key=os.environ.get("RELOCACHE_API_KEY"))
        uvicorn.run(app, host=args.host, port=args.port)
        return
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    plugins = list(filter(None, os.environ.get("VLLM_PLUGINS", "").split(",")))
    os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*plugins, "relocache"]))
    cache = Path(".cache").resolve()
    for variable, name in (
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("HF_HOME", "huggingface"),
        ("XDG_CACHE_HOME", "xdg"),
        ("NUMBA_CACHE_DIR", "numba"),
    ):
        os.environ.setdefault(variable, str(cache / name))

    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from .backends.vllm.config import engine_options

    config = json.loads(Path(args.config).read_text())
    options = engine_options(config)
    options.update(host=args.host, port=args.port)
    for key in ("data_parallel_size", "tensor_parallel_size"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    native_parser = make_arg_parser(FlexibleArgumentParser())
    native_parser.set_defaults(**options)
    native_args = native_parser.parse_args([])
    validate_parsed_serve_args(native_args)
    ServeSubcommand.cmd(native_args)


if __name__ == "__main__":
    main()

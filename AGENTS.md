# Agent Instructions

- Treat the existing `sempic_iclr`, `sempic_open`, `ref_repo`, and their environments as read-only. Do not modify or manage their running jobs.
- Make extraction, implementation, and validation changes only in new repositories and independent environments. Keep caches, logs, exported artifacts, and build outputs in new locations.
- Derive the service and engine implementation from the current `sempic_iclr` source. Use `sempic_open` to establish consumer contracts.
- Keep `third_party/vllm` an independent Git repository registered as a submodule. Commit engine changes there, then update the parent gitlink.
- ReLoCache owns serving and content management. SemPIC owns methods, Writer compilation semantics, datasets, prompts, and evaluation semantics.
- ReLoCache calls vLLM directly. Do not add multi-backend abstractions or reverse imports into ReLoCache or SemPIC from the fork.
- Preserve native KV ownership, scheduler admission, cache lookup, and offloading semantics when moving service logic out of the fork.
- KV Packet uses soft tokens during Writer compilation; Joint additionally enables the LoRA Writer. Reuse the soft-token representation and keep the execution stage explicit.
- Follow the fork's own instructions for changes inside `third_party/vllm`.

# Integration Boundary

This document records the agreed target; service extraction is not implemented yet.

## SemPIC to ReLoCache

SemPIC exports a deployment description containing model/tokenizer identity, Writer identity, a standard PEFT adapter when used, and compilation settings. ReLoCache must not depend on SemPIC checkpoint layout or import `sempic`.

Support both rendered Chat/Responses inputs and pretokenized requests containing canonical token IDs, reusable spans, and sampling parameters. Pretokenized requests preserve the consumer's prompt boundaries without a decode/re-encode round trip. Responses provide generated token IDs, text, termination information, usage, and observed timings.

Datasets, answers, Teacher/Student roles, soft-token placement policy, scoring, aggregation, and latency experiment protocols remain in SemPIC. The client package must be usable without installing vLLM into the research environment.

The intended `sempic_open` integration operates at the generation boundary, rather than returning its local `PrefillResult` tensors over the service interface.

## Methods and soft tokens

- Full recompute uses ordinary generation.
- No recompute uses independently compiled base-Writer PICs.
- SemPIC uses independently compiled LoRA-Writer PICs.
- KV Packet maps to Pre/Post soft tokens supplied during Writer compilation.
- Joint combines those compilation soft tokens with the LoRA Writer.

KV Packet and Joint are included in the integration scope. Reuse the soft-token representation; specify whether embeddings enter Writer compilation or Reader input. Reader-only soft tokens do not change Writer cache identity. Soft tokens used during compilation do affect that identity.

## ReLoCache to vLLM

Move HTTP routes, content resolution, message rendering, metadata persistence, explicit compilation orchestration, and service configuration into ReLoCache. Reuse vLLM serving components and call the fork directly.

Keep native page identity and geometry, execution metadata transport, scheduler miss handling, concurrent compilation tracking, physical/logical placement, pre-RoPE KV, fused-RoPE attention, and offload lifecycle in the fork. Pending native loads are not compilation misses. Automatic Writer requests remain on the Reader's DP engine.

Scientific Writer tensor exports remain analysis artifacts; the current serving path compiles and persists native vLLM pages. Compression extensions beyond the existing identity-reduction path require separate implementation.

## Validation ownership

The fork verifies cache and execution behavior; ReLoCache verifies service behavior; SemPIC verifies method outputs and evaluation semantics. Run all checks in new repositories and environments. Existing experiment results remain reference evidence, not validation of the extracted service.

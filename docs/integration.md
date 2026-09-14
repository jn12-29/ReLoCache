# Integration Boundary

This document records the integration boundary after reviewing the initial extraction. The Qwen3-8B P16 demo completed six Full Prefill/PIC sample pairs on GPUs 4–7 with DP=4 and TP=1; results are in `runs/sempic-demo-20260914`. This covers on-demand LoRA Writer compilation and Reader soft tokens. The default abort shutdown logged an `EngineDeadError` after all responses and flush completed; all test GPU allocations were released.

## SemPIC to ReLoCache

SemPIC exports a deployment description containing model/tokenizer identity, Writer identity, a standard PEFT adapter when used, and compilation settings. ReLoCache must not depend on SemPIC checkpoint layout or import `sempic`.

Full Prefill uses `/v1/completions` for text or token IDs, or `/v1/chat/completions` for messages. PIC uses structured reusable-content references and, when requested by the method, prompt embeddings through the Chat/Responses APIs. ReLoCache resolves references and renders them for native execution. Both paths retain native generation-parameter handling, streaming, and response semantics.

SemPIC chooses the task prompt, chat template, content boundaries, and soft-token placement. ReLoCache executes the supplied rendering instructions; it does not choose them from dataset or method names. Content registration accepts text or token IDs. Token IDs remain useful for research-controlled inputs without becoming the required public generation format.

Datasets, answers, Teacher/Student roles, soft-token placement policy, scoring, aggregation, and latency experiment protocols remain in SemPIC. The client package must be usable without installing vLLM into the research environment.

The intended `sempic_open` integration operates at the generation boundary, rather than returning its local `PrefillResult` tensors over the service interface.

## Methods and soft tokens

- Full recompute uses ordinary generation.
- No recompute uses independently compiled base-Writer PICs.
- SemPIC uses independently compiled LoRA-Writer PICs.
- KV Packet supplies Pre/Post soft tokens during Writer compilation.
- Joint combines those compilation soft tokens with the LoRA Writer.

KV Packet and Joint are included in the integration scope. Reuse the soft-token representation; specify whether embeddings enter Writer compilation or Reader input. Reader-only soft tokens do not change Writer cache identity. Soft tokens used during compilation do affect that identity.

The current ICLR implementation inserts Pre/Post embeddings into Reader input around each reusable occurrence. They occupy Reader positions and produce ordinary causal Reader states; they are not part of the persistent content representation.

Writer compilation accepts deployment `writer_soft_tokens` with optional `pre` and `post` entries, using the same `{ "data": "<base64 torch.save tensor>" }` representation as Reader prompt embeddings. ReLoCache loads these tensors once in the model's dtype and enables native prompt embeddings. Tensor row counts determine the lengths; a separate `count` is unnecessary. Omit `writer_adapter` for a base Writer (KV Packet), or supply a standard PEFT adapter for Joint. The service does not select methods by name.

Explicit and automatic compilation both execute `[Pre, document, Post, page padding, observation]` through native mixed token/embedding input. All Pre/document/Post KV rows are retained, with contiguous local positions starting at zero; padding and the observation row are excluded from Reader attention. Each reusable occurrence occupies `pre_length + document_length + post_length` Reader positions, so subsequent Inline and decode positions advance by that full extent. Reader-side soft tokens remain separate ordinary inputs and are not implicitly inserted from the Writer configuration. This implements the uncompressed packet path; KV selection and reduction are outside this extension.

Native page identity includes Writer soft-token values, shapes, dtypes, and Pre/Post placement, in addition to method identity and document tokens. Reusable descriptors carry typed tensors through native vLLM IPC; only compilation requests build full mixed embeddings. Workers consuming cached contents still receive compact continuous layouts. Registration and metadata persistence retain source document tokens; restoring a source binds it to the current deployment's Writer and soft tokens.

For example, add the following to the deployment JSON; either soft-token side may be omitted:

```json
{
  "writer_soft_tokens": {
    "pre": {"data": "<base64 torch.save tensor shaped [P, hidden_size]>"},
    "post": {"data": "<base64 torch.save tensor shaped [S, hidden_size]>"}
  }
}
```

Content register/compile/get responses return the original document `input_ids` and the full wrapped `logical_length`. Content references and explicit compile calls use the existing API without additional request fields.

## ReLoCache to vLLM

Move HTTP routes, content resolution, message rendering, metadata persistence, explicit compilation orchestration, and service configuration into ReLoCache. Reuse vLLM serving components and call the fork directly.

Keep native page identity and geometry, execution metadata transport, scheduler miss handling, concurrent compilation tracking, physical/logical placement, pre-RoPE KV, fused-RoPE attention, and offload lifecycle in the fork. Pending native loads are not compilation misses. Automatic Writer requests remain on the Reader's DP engine.

Registration stores source tokens and Writer identity without an engine RPC. Reader input carries a content table and occurrence indices; the engine derives native page metadata before request hashing. Explicit compilation receives page padding and its final observation row in native input processing, before prompt-length validation. Workers receive only unique position mappings, occurrence placements, and physical segments. Identity mappings use contiguous ranges without entry-position tensors; explicit per-layer mappings share tensors across repeated occurrences.

Scientific Writer tensor exports remain analysis artifacts; the current serving path compiles and persists native vLLM pages. Compression extensions beyond the existing identity-reduction path require separate implementation.

ReLoCache owns content registration and optional metadata persistence. vLLM owns KV residency and connector writes. Registering metadata is not evidence that KV has been compiled or persisted. A persistence request must require a configured persistent KV tier and wait for native writes to finish. Ordinary in-memory serving must not require a metadata database.

## Evaluation input semantics

ICLR constructs canonical inputs by independently tokenizing Inline and Content blocks and concatenating their IDs. Its Teacher performs full causal recomputation on those IDs; its Student uses the same hard IDs with optional Reader soft tokens. This is an explicit training and evaluation convention, not a requirement of ordinary Full Prefill serving. See the read-only source's `src/sempic/data.py`, `src/sempic/evaluation/server.py`, and `docs/overview/Training_Design.md`.

The service demo expands the source documents into ordinary messages for Full Prefill and references those same documents for PIC, using the same task text, template, and generation settings. Evaluation compares quality and performance without requiring identical token IDs. Whole-prompt and segmented tokenization are the respective paths' normal behavior; their difference does not require alignment or correction. Canonical token alignment is a training requirement.

Datasets, artifact conversion, reference answers, scoring, and aggregation stay in `examples/sempic`, outside the service package. ReLoCache receives a standard adapter and explicit execution inputs, without reading SemPIC artifacts itself.

## Service configuration and inputs

Content metadata stays in memory by default. Optional `metadata_dir` stores JSON source records by Writer identity and content alias, allowing references to be resolved after restart. KV pages remain owned by vLLM; metadata records do not imply KV residency or persistence.

`compile` with `persist=true` requires an `OffloadingConnector` using `TieringOffloadingSpec` with a filesystem (`fs`) or object-store (`obj`) secondary tier. It waits for all native DP engines to drain requests and connector writes. `flush` drains native work without asserting that a persistent tier is configured.

Deployment `engine.kv_transfer_config` retains its connector settings, with `key_encoding=pre_rope` applied to the cache namespace. Optional `cpu_bytes_to_use` overrides the staging budget; `kv_dir` appends a filesystem tier and requires a staging budget supplied either there or in the native connector configuration.

Chat and Responses retain native request types for ordinary inputs. Messages containing `reusable_content` use typed text, content-reference, and prompt-embedding parts. Responses also accepts prompt-embedding messages alongside content references. Responses preprocessing preserves assistant content lists only when they contain reusable references or prompt embeddings; other input items follow native conversion.

Dependency version differences do not need to be reverted. HTTP connection policy and GPU/CPU cache budgets are deployment or measurement settings; record the actual settings when interpreting timings instead of treating historical values as method requirements.

## Validation ownership

The fork verifies cache and execution behavior; ReLoCache verifies service behavior; SemPIC verifies method outputs and evaluation semantics. Run all checks in new repositories and environments. Existing experiment results remain reference evidence, not validation of the extracted service.

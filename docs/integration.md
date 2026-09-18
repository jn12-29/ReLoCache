# Integration Boundary

This document records the integration boundary after reviewing the initial extraction. The Qwen3-8B P16 demo completed six Full Prefill/PIC sample pairs on GPUs 4–7 with DP=4 and TP=1; results are in `runs/sempic-demo-20260914`. This covers on-demand LoRA Writer compilation and Reader soft tokens. The default abort shutdown logged an `EngineDeadError` after all responses and flush completed; all test GPU allocations were released.

## Content terminology

- **Reusable Content** is independently compiled by the Writer into PIC KV and referenced by the Reader at different logical positions. The API uses `reusable_content` references.
- **Dynamic Content** is executed in the current Reader context, including ordinary tokens and Reader-side embeddings. Its KV can be reused through native prefix caching under the corresponding context dependencies. Code identifiers use `dynamic_content`.

PIC names the reusable cache mechanism; pre-RoPE KV describes the storage representation. These are distinct from the two content categories.

## SemPIC to ReLoCache

SemPIC exports a deployment description containing model/tokenizer identity, Writer identity, a standard PEFT adapter when used, and compilation settings. ReLoCache must not depend on SemPIC checkpoint layout or import `sempic`.

Full Prefill uses `/v1/completions` for text or token IDs, or `/v1/chat/completions` for messages. PIC accepts ordered token content blocks through `/v1/completions`, or structured reusable-content references and prompt embeddings through Chat/Responses. ReLoCache resolves both input paths into the same logical Reader representation before native execution. Generation retains native sampling, streaming, and response handling.

For training and evaluation, SemPIC chooses the task prompt, chat template, content boundaries, and soft-token placement. ReLoCache executes the supplied rendering instructions; it does not choose them from dataset or method names. Content registration accepts text or token IDs. Token IDs remain useful for research-controlled inputs without becoming the required public generation format.

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

Explicit and automatic compilation both execute `[Pre, document, Post, page padding, observation]` through native mixed token/embedding input. All Pre/document/Post KV rows are retained, with contiguous local positions starting at zero; padding and the observation row are excluded from Reader attention. Each reusable occurrence occupies `pre_length + document_length + post_length` Reader positions, so subsequent Dynamic Content and decode positions advance by that full extent. Reader-side soft tokens remain separate ordinary inputs and are not implicitly inserted from the Writer configuration.

Native page identity includes Writer soft-token values, shapes, dtypes, and Pre/Post placement, in addition to method identity and document tokens. Reusable descriptors carry value-only plugin metadata through native SamplingParams. Each EngineCore loads deployment Writer soft tokens once; compilation requests use native mixed embeddings. Workers consuming cached contents still receive compact continuous layouts. Registration and metadata persistence retain source document tokens; restoring a source binds it to the current deployment's Writer and soft tokens.

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

ReLoCache targets efficient PIC-based inference serving and implements PIC semantics and core components, including cache compilation, content reuse, and PIC-specific inference components such as compilation coordination, position mapping, and attention. SemPIC primarily owns training and evaluation semantics and exports the trained artifacts and configuration used for serving.

The architecture targets multiple backends; only vLLM is implemented today. Keep backend details such as block hashes, page geometry, scheduler messages, and attention metadata in backend integration modules. Extend from concrete needs rather than building a speculative backend framework.

Inference backends call ReLoCache components through plugins or extension points, minimizing changes to backend source code. The vLLM fork may depend on and call ReLoCache; avoid package installation cycles and keep ReLoCache optional for upstream vLLM. Reuse native KV allocation, reference counting, scheduling, cache lookup, and offload mechanisms. Necessary vLLM changes should be narrow interfaces or execution capabilities that can be understood and proposed upstream independently. Preserve native structure, functionality, and performance; reduce private implementation coupling rather than copying native logic to reduce the patch's file count.

PIC implementation now lives in ReLoCache:

| Module | Responsibility |
| --- | --- |
| `content`, `inputs`, `identity` | Immutable sources, ordered Reader input, Writer/content identity. |
| `compilation`, `layout` | Shared compilation tracking, miss policy, logical intervals and shared storage coordinates. |
| `kernels.attention` | PIC range traversal and pre-RoPE key rotation, fused into the native attention kernel. |
| `backends.vllm.compiler`, `cache`, `request`, `scheduler` | Native page hashes, physical page ordering, Request/Scheduler specializations and compile/load coordination. |
| `backends.vllm.transport`, `worker`, `positions`, `attention`, `rotary` | Native plugin parameters, ModelState buffers, an external attention backend, and query-only RoPE through the CustomOp registry. |
| `backends.vllm.storage` | Residency inspection through native per-tier lookup, batch submission and cleanup. |
| `backends.vllm.api`, `rendering`, `content`, `config` | Serving integration and deployment configuration. |

The fork contains no ReLoCache imports or content-protocol names. It retains request construction and cache-admission hooks, configurable OnlineRenderer and ModelState, fused attention transforms, request-scoped offload work tracking and transfer statistics, and native error/stream metadata propagation. Request and scheduler-output schemas remain native. Native KV ownership, scheduling, offload and attention tiling/softmax remain in vLLM. Ordinary vLLM installation does not require ReLoCache. The cumulative patch against `upstream-base` is described in [vLLM patch boundary](vllm-patch.md).

ReLoCache registers query-only RoPE through `vllm.general_plugins` and the existing CustomOp registry; the native rotary source is unchanged. Standard and Llama 3 frequency scaling retain their native computation. Worker input packing uses the existing scheduler output path. Logical positions are prepared by the external ModelState into a stable GPU buffer after native storage positions and slot mappings; native worker preparation is unchanged. Residency inspection calls each native tier directly and flushes deferred existence lookups through its existing schedule-end method; it does not call the tiering manager promotion path. Scheduler utility calls expose concrete adapter methods, with an explicit `all_results` client option; DP aggregation and global-drain decisions belong to the service. PICOnlineRenderer composes the existing BaseRenderer, overrides structured completion preprocessing, and binds request metadata to native sampling parameters. The native OnlineRenderer retains tool/reasoning request adjustment; Responses conversion has a generic item override.

Registration stores source tokens without an engine RPC. Reader input carries a content table and occurrence indices through SamplingParams.extra_args. ReLoCache request types attach the metadata after rendering; the native InputProcessor is unchanged. The scheduler adapter derives native page metadata before hashing. The service constructs explicit compilation padding and the observation row before native prompt validation; automatic Writer requests use the same constructor on the Reader's DP engine. Workers receive unique layouts and shared physical segments, with no source tensors for cached contents. Pending native loads are not compilation misses.

Recompute methods such as EPIC and CacheBlend, and compression methods, remain deferred; the [roadmap](roadmap.md) retains their future plans.

Scientific Writer tensor exports remain analysis artifacts; the current serving path compiles and persists native vLLM pages.

ReLoCache owns content registration and optional metadata persistence. vLLM owns KV residency and connector writes. Registering metadata is not evidence that KV has been compiled or persisted. A persistence request must require a configured persistent KV tier and wait for native writes to finish. Ordinary in-memory serving must not require a metadata database.

## Compilation, residency and metrics

Registration/get responses expose `source_registered`, source `input_ids` and wrapped `logical_length`. Compilation additionally returns `compilation` (`compiled`, DP `rank`, `pending`, `persisted`, optional cache snapshot) and execution `metrics`. `persisted=null` means persistence was not requested; `true` records the successful check at completion, not a permanent residency guarantee. Failed persistence returns `persistence_failed`.

`GET /v1/reusable_contents/{content_id}/status` / `Client.status` inspects the current source snapshot on every DP replica. Each entry reports `compiled_id`, `compiling`, total pages, GPU-resident pages and per-tier `HIT`/`MISS` arrays. Inspection waits for asynchronous existence lookups and releases their bookkeeping; it does not promote KV or compile content. A registered source can have no resident KV. Compilation completion and current residency are separate observations.

With `auto_compile=false`, a confirmed missing page terminates only the Reader request through native `finish_reason=error`; HTTP errors retain `reusable_cache_miss` in their message. Asynchronous native promotions wait normally. `reusable_compilation_failed` identifies a dependent Writer failure. Neither failure permits placeholder tokens to execute as ordinary inputs.

`GET /v1/relocache` / `Client.deployment` reports model, Writer, Pre/Post lengths, automatic compilation, block size, TP/DP sizes and storage configuration.

PIC request metrics use the existing `kv_transfer_params.relocache` response field in completions, Chat and Responses. Completion/Chat streaming includes this metadata on the final usage chunk (`stream_options.include_usage=true`). Explicit compile responses include the same metrics directly.

| Metric | Scope and unit |
| --- | --- |
| `referenced_tokens`, `occurrences` | Logical wrapped PIC token references and occurrences, including repeated references. |
| `reused_tokens`, `reused_pages` | Distinct valid PIC storage rows and native pages consumed by this Reader; padding and duplicate references do not add valid rows. Pages are counted at the shared full-attention group geometry. |
| `auto_compiles` | Writer requests initiated by this Reader; waiting for another Reader's Writer does not increment it. |
| `wait_seconds` | Reader elapsed time from deferred PIC admission to execution, including compilation, lookup/load and intervening admission delay. |
| `prefill_tokens_executed` | Actually scheduled prefill rows, including repeated prefill after preemption. Writer rows include padding and observation; Reader rows exclude reused pages. |
| `compile_seconds` | Explicit Writer request elapsed time within the native scheduler, excluding later persistence waiting. |
| `loads.cpu_load_jobs`, `cpu_loaded_pages`, `cpu_load_seconds` | Completed CPU→GPU transfer jobs, pages and summed job durations for the request. |
| `loads.<tier>:<index>` | Secondary→CPU `load_jobs`, successful `loaded_pages` and summed `load_seconds`; includes `fs:0` for the first filesystem tier. |

Load counts cover the request's native transfers, which can include Dynamic Content pages. A page loaded from filesystem through CPU appears in both transfer stages. Work may overlap; do not sum compilation, admission waiting and transfer durations into end-to-end latency. Native timing metrics, TTFT, queueing and tracing remain available. SemPIC records initial cache state and reset scope when comparing runs.

## Evaluation input semantics

ICLR constructs canonical inputs by independently tokenizing Dynamic Content and Reusable Content blocks and concatenating their IDs. Its Teacher performs full causal recomputation on those IDs; its Student uses the same hard IDs with optional Reader soft tokens. This is an explicit training and evaluation convention, not a requirement of ordinary Full Prefill serving. See the read-only source's `src/sempic/data.py`, `src/sempic/evaluation/server.py`, and `docs/overview/Training_Design.md`.

The service demo expands the source documents into ordinary messages for Full Prefill and references those same documents for PIC, using the same task text, template, and generation settings. Evaluation compares quality and performance without requiring identical token IDs. Whole-prompt and segmented tokenization are the respective paths' normal behavior; their difference does not require alignment or correction. Canonical token alignment is a training requirement.

Datasets, artifact conversion, reference answers, scoring, and aggregation stay in `examples/sempic`, outside the service package. ReLoCache receives a standard adapter and explicit execution inputs, without reading SemPIC artifacts itself.

## Service configuration and inputs

Content metadata stays in memory by default. Optional `metadata_dir` stores JSON source records by Writer identity and content alias, allowing references to be resolved after restart. KV pages remain owned by vLLM; metadata records do not imply KV residency or persistence.

Registration requires `content_id` and exactly one of `input_ids` or `text`. Compilation accepts the same source fields, or only `content_id` to compile an existing registration. Each resolved occurrence binds an immutable source snapshot. Updating an ID affects subsequent resolutions, including later blocks within the same request; already resolved occurrences retain their source. Reference blocks are resolved in input order.

Content errors use an `error` object containing `message`, `type`, and `code`: unknown IDs return HTTP 404 with `unknown_content`; empty sources return HTTP 400 with `invalid_content`; unavailable persistent KV storage returns HTTP 400 with `persistence_unavailable`. Request schema errors retain native validation handling.

`compile` with `persist=true` requires an `OffloadingConnector` using `TieringOffloadingSpec` with a filesystem (`fs`) or object-store (`obj`) secondary tier. It waits only for that compilation and its native GPU→CPU→persistent-tier writes, then checks presence in each configured persistent tier. Concurrent requests can continue. `flush` remains a global drain across DP replicas and does not assert that a persistent tier is configured.

Deployment `engine.kv_transfer_config` retains its connector settings, with a `pre_rope` subdirectory appended to filesystem roots and a `pre_rope` component appended to object-store prefixes. Native FileMapper code is unchanged. Optional `cpu_bytes_to_use` overrides the staging budget; `kv_dir` appends a filesystem tier and requires a staging budget supplied either there or in the native connector configuration.

Chat and Responses retain native request types for ordinary inputs. Messages containing `reusable_content` use typed text, content-reference, and prompt-embedding parts. Responses also accepts prompt-embedding messages alongside content references. Responses preprocessing preserves assistant content lists only when they contain reusable references or prompt embeddings; other input items follow native conversion.

### Exact token content blocks

`POST /v1/completions` accepts a single structured `prompt`:

```json
{
  "model": "reader",
  "prompt": {
    "blocks": [
      {"type": "tokens", "input_ids": [101, 102]},
      {"type": "reusable_content", "id": "document"},
      {"type": "prompt_embeds", "data": "<base64 torch.save tensor>"},
      {"type": "tokens", "input_ids": [103]}
    ]
  },
  "max_tokens": 32,
  "temperature": 0
}
```

Token IDs execute in the supplied order without decoding, re-tokenization, or inserted special tokens. Reader embedding rows occupy ordinary Dynamic Content positions. Each content reference advances by the Writer's full Pre/document/Post length; clients supply no spans or offsets. Repeated references may share native KV pages while retaining distinct logical positions. References may include `text` or `input_ids` to register their source inline.

Structured prompts reject `truncate_prompt_tokens`, top-level `prompt_embeds`, `echo`, and `prompt_logprobs`. The Reader input must end with Dynamic Content because terminal PIC observation is not implemented. Chat truncation may remove whole occurrences but cannot split reusable content. Ordinary completion prompts retain native behavior.

`relocache.protocol` exports `TokenPart`, `PromptEmbedsPart`, `ReusableContentPart`, `ContentBlock`, `BlockPrompt`, and `ContentInfo` without vLLM, Torch, or Pydantic imports. The asynchronous client uses the same wire format:

```python
from relocache.client import Client

async with Client("http://127.0.0.1:8000") as client:
    await client.register("document", input_ids=[201, 202])
    compiled = await client.compile("document")
    response = await client.complete(
        [
            {"type": "tokens", "input_ids": [101]},
            {"type": "reusable_content", "id": "document"},
            {"type": "tokens", "input_ids": [103]},
        ],
        model="reader",
        max_tokens=32,
    )
```

`Client.complete` consumes a non-streaming JSON response. `Client.complete_stream(blocks, **generation)` yields native SSE JSON events, includes the final usage chunk by default, and closes the HTTP stream when its iterator is closed. Use `contextlib.aclosing` when breaking out early. Native request cancellation remains responsible for aborting execution.

Dependency version differences do not need to be reverted. HTTP connection policy and GPU/CPU cache budgets are deployment or measurement settings; record the actual settings when interpreting timings instead of treating historical values as method requirements.

## Validation ownership

ReLoCache verifies PIC core components, cache semantics, backend integration, and service behavior; the fork verifies changes to native execution; SemPIC verifies training and evaluation semantics and assesses method quality. Run all checks in new repositories and environments. Existing experiment results remain reference evidence, not validation of the extracted service.

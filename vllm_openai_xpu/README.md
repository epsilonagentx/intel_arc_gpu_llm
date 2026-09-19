# Upstream vLLM XPU engine

This folder runs **upstream's own** XPU image, `vllm/vllm-openai-xpu`, rather than
Intel's fork used by `scaler/`. It serves the same OpenAI-compatible API on the
same port, so it's a drop-in replacement for the other two engines here.

It's configured for **two validated models** — `gemma-4-26B-A4B-it` and
`gpt-oss-20b` — and switching between them is a `.env` edit plus a recreate.
It is also the only engine in this repo that can load gemma-4, which is why it's
the one currently serving.

Everything below was measured on an Arc Pro B60 (24 GB). Where a number is a
prediction or an extrapolation, it says so.

> **One GPU, one engine.** Exactly one may run at a time — `docker compose down`
> in `vllm_xpu/` or `scaler/` before starting this one. All three deliberately
> share the Compose project name `llm` so they reuse the `hf-cache` volume, which
> is also why you must **never pass `--remove-orphans`**: in a shared project it
> deletes the other engines' containers.

**Current state**

| | |
|---|---|
| Image | `vllm/vllm-openai-xpu:v0.29.0` |
| Model runner | V2 (upstream default from 0.29.0) |
| Model served | `gemma-4-26B-A4B-it`, offline int4 group-32 |
| Context | 131,072 |
| KV pool | 152,592 tokens = 1.16× concurrency |
| Reasoning | opt-in per request |
| `smoke.sh` | ALL PASS |

Kept as the cross-engine comparison point, measured for **gpt-oss-20b** on this
same engine: 83.5 tok/s @200, 83.1 @400, TTFT ~76 ms, KV pool 338,928 tokens
= 2.59× @128k.

---

## Contents

- [Quick start](#quick-start)
- [Configuration — `.env`](#configuration--env)
- [The two models](#the-two-models)
- [Switching models](#switching-models)
- [Verifying which model is live](#verifying-which-model-is-live)
- [Reasoning / thinking](#reasoning--thinking)
- [Model naming](#model-naming)
- [`bench.sh` and `smoke.sh`](#benchsh-and-smokesh)
- [Downstream consumers](#downstream-consumers)
- [gemma-4 in depth](#gemma-4-in-depth) — checkpoint, context, prefill, prefix cache
- [Performance](#performance)
- [Why it is configured this way](#why-it-is-configured-this-way)
- [Shared project and volumes](#shared-project-and-volumes)
- [Troubleshooting](#troubleshooting)
- [Open questions](#open-questions)

---

## Quick start

```bash
cd vllm_openai_xpu
docker compose up -d --force-recreate
cd ..                                   # the scripts live at the repo root
./smoke.sh        # correctness: served, content, reasoning, tool-calling
./bench.sh 400    # speed: TTFT and tok/s
```

Cold boot is ~2.5 min. `/v1/models` starts answering **before** generation is
ready, so trust `smoke.sh` over the first `200`.

### Why `--force-recreate` is not optional

vLLM bakes its CLI arguments into the container at creation. A plain `up -d` sees
an already-running container and leaves it alone, so **a `.env` edit appears to
do nothing**. Always recreate after changing configuration.

---

## Configuration — `.env`

`compose.yaml` reads every model-specific knob from a `${VAR:-default}`
placeholder, and Compose auto-reads `.env` from this folder — so the compose file
is never edited to change models. Copy the template and edit:

```bash
[ -f .env ] || cp .env.example .env    # never clobber an existing .env
```

`.env` is gitignored; `.env.example` is tracked and documents every variable with
its compose default and the measured consequence of changing it.

These are all **model-specific** — every one differs between the two models, so
they move as a set. `!` marks the three whose wrong value breaks a boot or
silently degrades it.

| Variable | What it sets |
|----------|--------------|
| `VLLM_MODEL` | Hugging Face repo ID |
| `VLLM_SERVED_MODEL_NAME` | The id clients call it by — see *Model naming* |
| `VLLM_REASONING_PARSER` | Model-family specific; wrong value = empty reasoning, **not** a crash |
| `VLLM_TOOL_CALL_PARSER` | Model-family specific, same quiet failure mode |
| `VLLM_MAX_MODEL_LEN` **!** | Context window; must fit VRAM after weights |
| `VLLM_KV_CACHE_MEMORY` **!** | KV pool in **absolute bytes**; overrides util, skips profiling, and OOMs rather than shrinking |
| `VLLM_EAGER_FLAG` **!** | `--enforce-eager` (required by gpt-oss's 8.6 GiB pin) or `--no-enforce-eager` (gemma-4, measured fine at 131,072). Passed whole — `--enforce-eager=False` does not parse |
| `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` | Server-side chat-template defaults. Currently unset, so reasoning is opt-in |

Shell variables beat `.env`, but **only for the names you pass** — so the
temporary-override block under *Switching models* has to set all of them, not
just `VLLM_MODEL`.

> **The variables are not independent.** `VLLM_KV_CACHE_MEMORY` is sized for a
> specific model's weights and OOMs rather than shrinking. gemma-4's weights
> (16.93 GiB) plus gpt-oss's KV pin (8.0 GiB) exceed the 22.33 GiB free on the
> card and fail at boot. Switch a whole block at once.

---

## The two models

| | gpt-oss-20b | gemma-4-26B-A4B-it (int4) |
|---|---|---|
| weights on device | 12.87 GiB | 16.93 GiB |
| decode | 83.2 tok/s (**chunks** — understated, see note) | **56.2 tok/s** (true tokens) |
| TTFT, ~55-token prompt | ~76 ms | ~89 ms reasoning off / ~128 ms on |
| KV pool | 338,928 tok | 152,592 tok |
| concurrency | 2.59× @128k | 1.16× @131k |
| max context | 131,072 | 131,072 (ceiling 162,496) |
| reasoning | always on, no off switch | opt-in per request |
| large cold prompts | linear, fine | **quadratic — see warning** |

**gpt-oss-20b is still the better choice for coding-CLI traffic** through a
gateway, because its prefill stays linear. gemma-4 is the newer and stronger
model, and is vision-capable.

> ⚠ **`bench.sh` under-reports gemma-4 by ~15%, and the reason matters.** It
> counts SSE *chunks*, and with reasoning on the reasoning channel packs **1.15
> tokens per chunk**, so it reads ~48 chunk/s where the true rate is ~56 tok/s.
> Measure with `stream_options.include_usage` instead: TTFT from the first chunk,
> true counts from the final usage chunk. With thinking off, chunks and tokens go
> exactly 1:1 — which confirms the mechanism rather than merely correlating with
> it.
>
> **gpt-oss's 83.2 is also a chunk count** and is therefore understated too, by
> an unmeasured amount — it always reasons, so it can't be re-measured with
> reasoning off. Treat the cross-model gap as indicative, not exact.

---

## Switching models

### Temporary — shell overrides

Shell variables beat `.env`, so this reverts by itself and edits nothing. To run
gpt-oss-20b while `.env` is configured for gemma-4, **paste the whole block** — a
shell variable only wins for the variables you actually name, so every value
`.env` sets for gemma has to be overridden here:

```bash
cd vllm_openai_xpu
docker compose down

VLLM_MODEL=openai/gpt-oss-20b \
VLLM_SERVED_MODEL_NAME=gpt-oss-20b \
VLLM_REASONING_PARSER=openai_gptoss \
VLLM_TOOL_CALL_PARSER=openai \
VLLM_MAX_MODEL_LEN=131072 \
VLLM_KV_CACHE_MEMORY=8603448832 \
VLLM_EAGER_FLAG=--enforce-eager \
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=null \
docker compose up -d --force-recreate
```

Confirm it took over before trusting it:

```bash
cd .. && ./smoke.sh          # should report gpt-oss-20b -> openai/gpt-oss-20b
```

Two of those are **not optional**, and they fail differently:

| variable | if you omit it while `.env` holds gemma's value |
|---|---|
| `VLLM_KV_CACHE_MEMORY` | boots, but the 4.25 GiB pin strands ~4 GiB and roughly halves the pool |
| `VLLM_EAGER_FLAG` | **boot fails** — compiled mode with gpt-oss's 8.6 GiB pin starves the KV pool, which is the case `--enforce-eager` exists for |

Both models now run at `131,072`, so `VLLM_MAX_MODEL_LEN` no longer differs
between the blocks — but keep it in the block anyway, since the two are free to
diverge again. `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=null` is tidiness only; an
unknown template kwarg is verified harmless on gpt-oss, which always reasons
regardless.

The served name changes with the model, so update your consumer's model mapping
too — see *Model naming*.

### Back to whatever `.env` says

```bash
docker compose down
docker compose up -d --force-recreate
```

No variables. If you used `export` rather than a one-line prefix, the old values
are still in your shell and will silently win:

```bash
env | grep ^VLLM_     # expect no output
```

Clear them with `unset`, or just open a new terminal.

### Permanent — edit `.env`

Comment out one model block in `.env` and uncomment the other, then recreate.
`.env.example` ships both blocks ready to swap.

---

## Verifying which model is live

The model id is a configured label, not proof of what loaded. `root` always
reports the real checkpoint:

```bash
curl -s localhost:8000/v1/models \
  | python3 -c 'import sys,json;[print(m["id"],"->",m["root"]) for m in json.load(sys.stdin)["data"]]'
```

`/v1/models` also reports `max_model_len`, which is the authoritative answer to
"what context is actually live" — the `.env` file tells you only what the *next*
container will get.

Expected in the container log:

| model | log lines |
|---|---|
| gpt-oss-20b | `Model loading took 12.87 GiB`, `GPU KV cache size: 338,928 tokens … 2.59x` |
| gemma-4 | `Model loading took 16.93 GiB`, `GPU KV cache size: 152,592 tokens … 1.16x` |

For gemma-4 the log should also show the full quantisation chain, which confirms
the int4 kernels actually engaged rather than silently falling back:

```
Using XPUwNa16LinearKernel for CompressedTensorsWNA16
Using CompressedTensorsWNA16MoEMethod
Using 'XPU' WNA16 MoE backend.
Using XPUExpertsWNA16
```

---

## Reasoning / thinking

vLLM puts the trace in **`message.reasoning`** (and `delta.reasoning` when
streaming), **never `reasoning_content`**. Anything downstream expecting
DeepSeek's `reasoning_content` spelling will silently see nothing and conclude
the parser is broken. It isn't — check the field name first.

- **gpt-oss-20b** — always reasons, no off switch.
- **gemma-4** — **opt-in, and that is the deployed default.**
  `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` is unset, so the compose default `null`
  applies and the server adds no template defaults. A request asks for reasoning
  explicitly:

```json
"chat_template_kwargs": {"enable_thinking": true}
```

To turn it back on for *every* request, set this in `.env` and recreate:

```dotenv
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}
```

Either way a request can override the server default in both directions, so it is
a default and not a lock.

### What reasoning actually costs

**Not decode speed.** Measured on one solvable word problem, streaming with
`include_usage`:

| | thinking off | thinking on |
|---|---|---|
| decode | **56.2 tok/s** | **53.7 tok/s** |
| completion tokens | 433 | 1,033 |
| wall-clock | **7.8 s** | **19.4 s** |
| answer | correct | correct, same length |

The per-token rate barely moves. What moves is the **token count** — 2.39× more
tokens, so 2.48× the wall-clock. On that particular problem the extra 600 tokens
of trace changed the answer not at all, though that's one arithmetic question and
not a quality evaluation; arithmetic is reasoning's weakest case.

This is why the deployed default is off: reasoning is a cost you opt into for the
requests that earn it. If you want graded control rather than a switch, it
belongs at the gateway — route `enable_thinking` per consumer.

> **There is no middle setting.** `enable_thinking` is binary. `reasoning_effort`
> exists in vLLM 0.29.0 but only for DeepSeek-V4, and there is no
> `thinking_budget` / `max_thinking_tokens` anywhere in the codebase. "Off" is a
> designed mode rather than a degradation — the chat template pre-emits an
> opened-and-immediately-closed thought channel, so the model *cannot* think
> rather than being asked not to.

### ⚠ Never use `max_tokens` as a thinking cap

Reasoning consumes `max_tokens` **first**. If the budget runs out mid-trace you
get `finish_reason: length` and **zero answer characters** — not a truncated
answer, nothing at all. Measured: a prompt with no clean solution burned **6,000
tokens over 133 seconds and returned nothing.** With reasoning on, `max_tokens`
must cover trace *plus* answer; allow 1500+ for ordinary work and more if the
task might make the model search.

> `null` is the compose default for this variable, not `{}` — a brace inside a
> `${VAR:-default}` breaks Compose interpolation. `json.loads("null")` is `None`,
> which vLLM treats as no defaults.

---

## Model naming

**One truthful name per model** — `gpt-oss-20b` and `gemma-4-26b-a4b`. The id is
set by `VLLM_SERVED_MODEL_NAME` and is not validated against the checkpoint, so
it *could* be used as a fixed label to hide a swap. Don't. The two models now
share a context cap but differ enormously in prefill cost, so a consumer that
keeps sending gpt-oss-sized cold prompts to gemma-4 gets multi-minute stalls that
look like a gateway fault rather than a deliberate model change.

A swap therefore means updating the consumer's model mapping. That is the point —
it makes the change visible where the behaviour actually differs.

`smoke.sh` prints the id→checkpoint mapping on every run, and warns if
`/v1/models` ever returns a duplicate id.

---

## `bench.sh` and `smoke.sh`

Both live at the repo root and work against any engine. Two conveniences matter
here:

- **`MODEL` is auto-detected** from `/v1/models` when unset, so neither script
  needs editing across a model swap. The output labels which path was used
  (`auto-detected` vs `explicit`).
- **`THINKING` is tri-state**, which is what makes the deployed reasoning default
  observable:

| `THINKING` | sends | asserts / reports |
|---|---|---|
| `auto` (default) | nothing | whatever the **server** default does |
| `1` | `enable_thinking: true` | a reasoning trace comes back |
| `0` | `enable_thinking: false` | reasoning is **suppressed** (`smoke.sh` inverts the check) |

```bash
./smoke.sh                 # ALL PASS on both models as shipped
THINKING=1 ./smoke.sh      # proves a request can opt in to the trace
./bench.sh 1500            # reasoning needs the larger budget
VLLM_ENDPOINT=http://<host>:8000 ./smoke.sh        # remote target

# MODEL is only needed to pin an id explicitly; it must match what is loaded,
# or you get a 404 that names the served id back to you.
MODEL=gemma-4-26b-a4b ./bench.sh 400
```

---

## Downstream consumers

The endpoint is OpenAI-compatible and **unauthenticated** — see *Firewall* in the
[top-level README](../README.md). Any AI gateway works; point it at
`http://<host>:8000/v1` using a served model name.

**A model swap changes the served name, so the consumer 404s until its model
mapping is updated.** That is deliberate: the two models have very different
prefill costs, and a fixed label would hide that until it surfaced as a stall.

Two integration details that cause silent breakage:

- The reasoning trace is in **`reasoning`**, not `reasoning_content`.
- gemma-4 reasoning is **opt-in**, so a gateway that doesn't pass
  `chat_template_kwargs` gets no trace. That's configuration, not a fault.

---

## gemma-4 in depth

### Checkpoint constraint

The checkpoint must be **int4-symmetric with `group_size` 32 or channelwise**.
The XPU expert kernel (`XPUExpertsWNA16`) accepts only those two schemes, so the
smaller and far more popular **group-64** builds are rejected at load despite
being ~2 GiB lighter. Check
`quantization_config.config_groups.*.weights.group_size` before trying any other
MoE checkpoint.

### Context is nearly free — the KV formula

gemma-4 is **25 sliding-attention layers (window 1024) + 5 full-attention
layers**, a 5:1 pattern, and its checkpoint declares
`max_position_embeddings: 262144`. In vLLM 0.29.0
`SlidingWindowSpec.max_memory_usage_bytes` bounds those 25 layers at
`min(sliding_window - 1 + max_in_flight_tokens, max_model_len)` — **independent
of `--max-model-len`**. Only the 5 full-attention layers scale with context, and
they're the cheap ones: 2 KV heads × 512, against the sliding layers' 8 × 256.

So KV cost is a large fixed block plus a small linear term:

```
fixed  (25 sliding layers, window-bounded) = 1.235 GB      # context-independent
linear (5 full layers)                     = 20 KiB/token
```

Measured against that formula at the 4.25 GiB pin — **raising `--max-model-len`
costs no VRAM at all**, and prediction tracks the engine's own log to 0.003%:

| `--max-model-len` | per request | concurrency | KV pool (predicted / **logged**) |
|---|---|---|---|
| 32,768 | 1.906 GB | 2.39× | 78,435 / **78,433** |
| 65,536 | 2.578 GB | 1.77× | 116,028 / **116,025** |
| **131,072** (shipped) | 3.920 GB | **1.16×** | 152,596 / **152,592** |
| 162,496 | 4.563 GB | 1.00× | hard ceiling |

Weights, pool, decode and TTFT were byte-for-byte identical across all three
boots; only concurrency moves. A 64,708-token prompt answered a question about
its *last* record correctly, so the window is real and not merely allocated.

The reported pool isn't a flat token count:
`kv_cache_utils.py` computes `num_tokens = int(max_concurrency * max_model_len)`,
which is why it lands on an odd number.

> **⚠ Raising `--max-num-batched-tokens` LOWERS the context ceiling.** It feeds
> `max_in_flight_tokens`, which inflates the **sliding** reservation, not the
> full-attention one. At 8192 the fixed block balloons 1.235 → 3.568 GB and the
> ceiling collapses to 48,576. An earlier revision of this file reported that
> collapse as a property of the model — it's an artifact of the flag. Leave it at
> the default (2496).

> **Per-token KV is meaningless for a hybrid model.** 4.25 GiB / 78,433 tok =
> 56.8 KiB/token was an artifact of the old 32k cap; the *marginal* cost is
> 20 KiB/token. When sizing any hybrid model read `layer_types`,
> `sliding_window`, `num_key_value_heads` **and** `num_global_key_value_heads` —
> the head *counts* differ per layer type, not just the dims.

### The cap is a guardrail, not a speed setting

**Changing the cap does not change decode speed.** Measured same-session, three
warm runs at each setting: 46.2 / 46.3 / 46.3 chunk/s and 106 / 106 / 105 ms TTFT
at 32k / 64k / 128k — identical. (Chunk counts, same instrument and same bias
across all three, so the comparison holds even though the absolute figure is
understated.)

The pool is pinned in absolute bytes, so it is 4.25 GiB at every cap, and a given
prompt costs the same wherever the cap sits. Concurrency is only
pool ÷ one-max-length-request — a ratio, not a capacity.

What the cap **does** control is the longest prefill a client can trigger:

| | 32k | 64k | 96k | 131k (shipped) |
|---|---|---|---|---|
| largest prompt accepted | 32,768 | 65,536 | 98,304 | 131,072 |
| worst-case cold prefill | ~4.6 min | ~22 min | ~51 min | **~92 min** |
| concurrency | 2.39× | 1.77× | 1.41× | **1.16×** |

A lower cap makes an oversized prompt fail fast with a `400` instead of occupying
the card for tens of minutes and then very likely timing out upstream anyway, and
it leaves more pool for prefix cache. The shipped `131,072` trades that
protection for reach. If you run it, make sure client timeouts along the whole
path match — and note the top of that range is **unexercised**: the largest
prompt ever pushed through end-to-end is 64,708 tokens.

### ⚠ Prefill is quadratic, and it is not tunable from here

This, not VRAM, is the real limit on usable context.

| cold prompt | wall |
|---|---|
| 2,421 tok | 2.2 s |
| 4,821 tok | 5.6 s |
| 9,621 tok | 19.4 s |
| 19,221 tok | 80.8 s |
| **64,708 tok** | **1297.6 s (21.6 min)** |

Least-squares over three same-harness points gives exponent **2.048** —
essentially plain quadratic — reproducing all three to within **±0.1%**:

```
T ≈ 26.3 s × (n / 9643)^2.048
   9,643 → 26.3 s   |   31,957 → 305.3 s   |   64,708 → 1297.6 s
```

Beyond 64,708 this is extrapolation, not measurement: ~51 min at 96k and ~92 min
at 131k have never been observed.

> An earlier revision claimed exponent **2.287** and ~109 min at 128k. That came
> from mixing one session's 64.7k measurement with a *different* session's
> 19,221 → 80.8 s datapoint taken under another config. Never fit a curve across
> sessions.

**The tell that the kernel is at fault rather than the algorithm:** prefill runs
at **49.9 tok/s** while decode does ~56. Prefill being *slower than decode*
should be impossible on a healthy path, because prefill batches 2,496 tokens per
chunk and is compute-bound while decode is memory-bound at one token per step.
The kernel is discarding essentially all of prefill's parallelism.

#### Why the slow kernel gets chosen

The engine says so at boot:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}.
FA4 not available, forcing TRITON_ATTN backend.
```

The chain, all verifiable inside the image:

| step | source | result |
|---|---|---|
| head-512 gate | `flash_attn.py supports_head_size()` — `>256` needs FA4 | needs FA4 |
| is FA4 available? | `is_fa_version_supported(4)` → imports `vllm.vllm_flash_attn` | **ImportError** |
| why | that module *"requires the CUDA flash attention extensions (`_vllm_fa2_C` or `_vllm_fa3_C`)"* | CUDA-only |
| and even if it imported | `_is_fa4_supported()` requires CUDA compute capability 9.x/10.x/11.x | unreachable on Arc |

⇒ `supports_head_size(512)` is `False`, FLASH_ATTN is ineligible, and
`Gemma4Config` selects `TRITON_ATTN` — the only backend that JIT-compiles for
both 256 and 512. vLLM puts *all* layers on it deliberately: mixing causes
*"mixed backend selection and numerical divergence"*.

> **A head-512 XPU kernel does exist.** `vllm_xpu_kernels`'
> `libattn_kernels_xe_2.so` in this image contains `chunk_policy_head512` **and**
> `chunk_policy_head512_b16` — the block-size-16 paged variant, which is this
> deployment's geometry — plus a head-512 paged-decode kernel. It is unreachable
> because the gate above asks *"are you FlashAttention 4?"*, a question about
> CUDA lineage, rather than *"can you do head_size 512?"*. An earlier revision of
> this file said the fix was a kernel that doesn't exist; the kernel exists and
> the plumbing doesn't. Nothing in this repo can reach it — `is_xpu() → return 2`
> in `fa_utils.py` sits *before* the config override path, so no env var helps.

`VLLM_ATTENTION_BACKEND` is **honoured, then refused** on the head-size gate —
not ignored. Don't spend time forcing it.

And a hypothetical per-layer split wouldn't rescue long prompts: at 131k the five
512-dim layers do **92.7%** of the attention work. It *would* help ordinary
traffic, though — the two layer groups cross over at
`25 × n × 1024 = 5 × n²/2`, i.e. **n ≈ 10,240 tokens**, so below that the
twenty-five 256-dim sliding layers carry most of the attention cost and they are
FA2-eligible on paper.

### Prefix caching is what makes this usable

Same prompt sent twice on a virgin engine:

| n | cold | warm | speedup |
|---|---|---|---|
| 9,643 | 26.3 s | 0.31 s | **84×** |
| 31,957 | 305.3 s | 0.63 s | **481×** |

Warm time stays roughly flat while cold grows quadratically, so the ratio climbs
with prompt size. This also **refutes** a plausible worry: sliding-window layers
free their out-of-window blocks mid-request (`remove_skipped_blocks`), which
looked like it should defeat reuse on long prompts. It doesn't.

So conversation and growing context are fine; a large **cold** context (fresh RAG
document, big paste) is where it hurts. **You pay it once per document per engine
lifetime** — the cache is VRAM-resident, so a `--force-recreate` throws it away.

Two practical consequences:

- **Put stable content first.** A document at the top of the prompt stays a
  reusable prefix across many different questions; the same document placed
  *after* a varying question caches nothing, because matching runs from token 0.
- **Growing a conversation costs the same total as one big prefill** — it's the
  same triangle either way. What changes is that it's spread out, so per-turn
  latency creeps up: a ~2k-token turn costs ~40 s at 32k of context and ~2.2 min
  at 100k.

---

## Performance

### Eager vs compiled — take compiled on gemma-4

Upstream runs `torch.compile` even with cudagraphs off, and those buffers are
**not** capped by `--gpu-memory-utilization`. With **gpt-oss's 8.6 GiB pin** they
starved the KV pool and killed the boot:

```
Available KV cache memory: 2.89 GiB
ValueError: max seq len (131072) needs 3.1 GiB KV cache > available 2.89 GiB
```

Missing 128k by 0.21 GiB — which is why `--enforce-eager` is the compose default,
and note it's for a *different* reason than on the scaler, where eager prevents
silently-empty content.

gemma-4's smaller 4.25 GiB pin leaves headroom, and compiled mode is measured
free — at 32,768, 65,536 **and** 131,072, all boot-tested with the pool intact.
The compile buffers scale with `max_num_batched_tokens` (2496), *not* with
context, so context length doesn't change this trade-off:

| config | decode (true tokens) | KV pool | concurrency |
|---|---|---|---|
| eager | 52.0 tok/s | 78,433 | 2.39× |
| **compiled** | **~56 tok/s** (+7%) | 78,433 | 2.39× |
| compiled + XPU graph | +0.5% more | 46,138 | 1.41× |

Those pool figures were taken at the then-shipped 32,768 cap; at today's 131,072
the pool is 152,592 tokens.

```dotenv
VLLM_EAGER_FLAG=--no-enforce-eager
```

Compiled costs ~34 s of `torch.compile` per boot (the SYCL cache must stay off,
so it never persists — though it does load from the AOT compile cache in well
under a second once warm).

**XPU graph is not worth it.** +0.5% for 1.54 GiB of capture memory. At the old
32k cap that dropped the pool to 46,138 tokens; at **131,072 it doesn't fit at
all** — 1.54 GiB out of a 4.25 GiB pool leaves less than one max-length request,
so the engine won't boot. It does genuinely capture graphs in compiled mode; the
earlier "no-op" result was an artifact of only ever testing it under eager.

**Compiled mode does not help prefill** (2.28 / 5.56 / 19.77 s at 2.4k / 4.8k /
9.6k tokens, vs eager's 2.15 / 5.61 / 19.38). That confirms the quadratic prefill
is the forced Triton attention backend, not kernel-launch overhead, and so isn't
fixable from configuration.

### Why gemma-4 is slower than gpt-oss

Not bandwidth — gemma reads *less* per token (~3.35 GB vs ~3.71 GB, counting
int4/MXFP4 weights plus the unquantized embedding). It's op count and
granularity:

| | gemma-4 | gpt-oss |
|---|---|---|
| layers | 30 | 24 |
| experts activated / layer | 8 of 128 | 4 of 32 |
| expert intermediate size | 768 | 2880 |
| expert GEMVs per token | **240** | 96 |
| attention backend | TRITON_ATTN (forced) | FLASH_ATTN |
| expert kernel | generic `XPUExpertsWNA16` | `XPUExpertsMxFp4` (tuned) |

2.5× as many expert matrix multiplies per token, each ~3.75× narrower. Small
matrices use the GPU poorly and each costs a launch. Compiled mode recovers part
of that; the rest is architectural, and it sets the ceiling — there is no
configuration that makes this model reach gpt-oss's rate.

### Standings against the scaler

Both engines have the `--kv-cache-memory-bytes` lever, measured on the same box
with the same scripts (gpt-oss-20b, 128k):

| | upstream 0.29.0 + V2 | scaler 0.26.0-b2 | winner |
|---|---|---|---|
| tok/s @400 | 83.1 | **85.6** | scaler, +3.0% |
| tok/s @200 | 83.5 | **86.1** | scaler, +3.1% |
| KV pool | 338,928 tok | **340,663 tok** | scaler, +0.5% |
| Concurrency @128k | 2.59× | **2.60×** | scaler |

**The scaler leads on both axes**, so there's no capacity argument for migrating
gpt-oss here. Upstream's advantages are non-performance: several vLLM minors
newer, half the image size, mainline rather than an undocumented beta, a
trustworthy `latest`, and `xpu-smi` in the image — plus it is **the only engine
here that runs gemma-4**, which is the actual reason it's serving.

---

## Why it is configured this way

### Why this engine exists

The old assumption was that a B60 needs Intel's fork, because upstream vLLM
didn't cover Arc Pro B-series or MXFP4 gpt-oss. Both halves are false: upstream's
own XPU docs list **Intel® Arc™ Pro B-Series** as validated hardware, and
`platforms/xpu.py` `supported_quantization` includes `mxfp4` and `gpt_oss_mxfp4`.
MXFP4 was confirmed native here — weights load at 12.87 GiB where bf16 would be
~42 GB.

> ⚠️ **Being listed in upstream's model table is an architecture claim, not a fit
> claim.** `gpt-oss-120b` is listed too, and its 60.7 GiB of weights fit neither
> one B60 nor 2×B70.

**Image pin.** Upstream's XPU builds live in a **separate Docker Hub repo** —
`vllm/vllm-openai-xpu`, not `vllm/vllm-openai`. Checking the CUDA repo shows zero
XPU tags and wrongly suggests no image exists. Unlike Intel's images, `latest`
here **does** track newest stable; the pin is still explicit so an upgrade stays a
deliberate commit.

### `SYCL_CACHE_PERSISTENT=0` — mandatory with the V2 runner

The first V2 boot died after a clean memory profile, with no Python traceback:

```
!!!!!!! Segfault encountered !!!!!!!
  at::native::xpu::topk_kernel → ... → sycl::handler::finalize()
  → PersistentDeviceCodeCache::getItemFromDisc → getSortedImages   ← segfault
```

Traced to source: `gpu_worker.py` runs `warmup_kernels(...)` **for V2 only**, and
that warmup builds its batch with `SamplingParams.for_sampler_warmup()`, which
hardcodes `logprobs=5, prompt_logprobs=1` to exercise all sampler logic. Logprobs
reach `torch.topk(...)`, whose XPU kernel segfaults while building the SYCL
program from the **persistent** device-code cache.

Notes for whoever meets this next:

- The crashing feature is **logprobs, which this deployment never requests.** V2
  died proving a path normal serving never reaches.
- Not memory: `OOMKilled=false`, the profile printed, the pool allocated.
- **No config knob avoids it.** `for_sampler_warmup()` hardcodes its params,
  `--max-logprobs` never reaches it, and there's no env var to skip
  `warmup_kernels` (`KernelConfig.enable_jit_warmup` governs a *different*
  warmup, which succeeds).
- Cost of the fix is a device-kernel rebuild each boot; total boot still ~70 s.
- **Unreported upstream** — worth filing.

`SYCL_CACHE_PERSISTENT` is set by this repo, not a vLLM default. Note the stock
`intel/vllm` engine in `vllm_xpu/` runs fine with it at `1`.

### The V2 model runner

At 0.28.0 the V2 gate required
`arch ∈ DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES or not is_moe`;
`GptOssForCausalLM` was in neither set, so **0.28.0 silently ran V1**. 0.29.0
removed that gate, so V2 is now the default here.

V2 is faster and leaner:

| | V1 | V2 |
|---|---|---|
| Non-torch | ~1.50 GiB | **~1.03 GiB** |
| Peak activation | 0.69 GiB | **0.27 GiB** |
| Advised "fully utilize" pool | 6.63 GiB | **8.01 GiB** |

Confirming which runner is live — the absence of a log line is the V1 signal:

| | V1 | V2 |
|---|---|---|
| Log line | *(none)* | `xpu_worker.py Using V2 Model Runner` |
| Module in log paths | `gpu_model_runner.py` | `worker/gpu/model_runner.py` |

`Initializing a V1 LLM engine` refers to the **engine**, a different axis — don't
read it as the runner. `VLLM_USE_V2_MODEL_RUNNER` overrides, and is **integer
only**: a blank value crashes on `bool(int(""))`.

### `--kv-cache-memory-bytes` — the big capacity lever

`--gpu-memory-utilization 0.80` caps the engine at 18.17 GiB of a 22.71 GiB card,
stranding ~4.5 GiB. The engine prints the exact byte value to reclaim it:

```
Replace gpu_memory_utilization config with
  `--kv-cache-memory=8603448832` (8.01 GiB) to fully utilize gpu memory
```

Taking it moved gpt-oss's pool 3.12 → 8.01 GiB, **169,123 → 338,928 tokens
(1.29× → 2.59× @128k)** at unchanged decode and TTFT.

Four things to know before touching it:

1. **The value is absolute.** It overrides util and skips memory profiling, and it
   **OOMs rather than shrinking** if the model doesn't fit.
2. **It is runner-specific.** The byte value comes from that runner's profile;
   carrying V1's 6.63 GiB onto V2 leaves ~1.4 GiB unclaimed. Re-derive after any
   runner or image change by commenting the flag out for one boot and reading the
   advised value back.
3. **It is model-specific.** gemma-4's weights plus gpt-oss's 8.0 GiB pin exceed
   the 22.33 GiB free and fail at boot. Switch a whole `.env` block.
4. **Canonical spelling is `--kv-cache-memory-bytes`.** The log advises
   `--kv-cache-memory`, which only works as an argparse prefix abbreviation.

gemma-4's 4.25 GiB pin leaves ~0.15 GiB of card headroom. That's defensible only
because the compile buffers are bounded by `max_num_batched_tokens` rather than by
context.

### Device passthrough

**Both the whole `/dev/dri` device and a `/dev/dri/by-path:ro` bind are
required**, at any world size, because oneCCL enumerates devices via `by-path`.
This is upstream's documented recipe, not a local workaround — it matches the
`ze_fd_manager.cpp:144 … opendir failed` crash reverse-engineered on the scaler
([`scaler/README.md`](../scaler/README.md) §5). 0.29.0 skips the oneCCL warm-up at
world_size=1, so the bind may now be redundant; keep it until a boot without it
proves otherwise.

Upstream's example uses `--privileged` and `--network=host`. This repo uses
`group_add` (render / video GIDs) plus an explicit port map instead — **measured
sufficient**, no `ze_fd_manager` failure. Less privilege for no cost. Those GIDs
are host-specific; check yours with `getent group render video`.

Three things the platform sets automatically: `UCX_MEMTYPE_CACHE=n`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `shutdown_timeout=5`. That last one
matters — XPU needs a graceful shutdown to release oneCCL/Level Zero resources,
or *"subsequent server startups on the same devices may hang during CCL
initialization."*

**Graph mode is single-GPU-only upstream** ("XPU Graph support is experimental and
currently only supports single-GPU execution"), so the single-B60 layout is the
only one that could use it — going dual would forfeit it. See *Performance* for
why it isn't taken.

### What upstream does not provide

Intel's fork keeps some B60-specific surface upstream lacks: the online int4 path
and `VLLM_QUANTIZE_Q40_LIB`, extra arch registrations, and Battlematrix multi-GPU
tuning.

Requirements: **Python 3.12 exactly** (the `vllm-xpu-kernels` wheels are
3.12-only and upstream flags this as a MUST), `torch==2.13.0` XPU,
`triton==3.7.2+xpu` — all satisfied inside the image. The host needs only the
in-kernel `xe` driver. The image also bundles `xpu-smi` 2.1.0, which the scaler
image does not.

---

## Shared project and volumes

All three engine folders pin `name: llm`, and Compose prefixes declared volumes
with the project name, so a plain `hf-cache:` in each file resolves to the *same*
real volume. Whichever engine starts first creates `llm_hf-cache`; the others
attach. No `external:` needed — the three are peers in one project. Compile
caches stay per-engine, because kernels are image-specific.

| | |
|---|---|
| Project | `llm` (shared with `vllm_xpu/`, `scaler/`) |
| Weights (shared) | `llm_hf-cache` |
| Compile cache (own) | `llm_vllm-openai-cache` |

- **Orphan-container warnings are expected** when another engine's container
  lingers. Fix by running `down` in that folder. **Never `--remove-orphans`** —
  in one shared project it deletes the other engines, possibly killing prod.
- **`down -v` from ANY engine folder deletes `llm_hf-cache`**, taking the cached
  weights with it. Plain project volumes are removable by any project member. Use
  bare `down`.

**Swap procedure** — one GPU, so exactly one engine runs at a time:

```bash
docker compose -f vllm_xpu/compose.yaml down     # if the stock engine is up
docker compose -f scaler/compose.yaml down       # if the scaler is up
docker compose -f vllm_openai_xpu/compose.yaml up -d
```

Stop any local `llama.cpp` stack first if one is running — it shares the GPU and
port 8000, and if it's under a process supervisor it will come back by itself.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `.env` edit had no effect | Missing `--force-recreate`; the container kept its baked-in args |
| `HTTP 404 ... model does not exist` | Stale `MODEL` in your shell, or the gateway points at the other block's name. `bench.sh` prints what *is* served |
| Reasoning field always empty | Two candidates: gemma-4 without `enable_thinking` (it's opt-in by default), or the client reading `reasoning_content` instead of `reasoning` |
| Trace returned but no answer | Reasoning consumed the whole `max_tokens`. Raise it — and note a hard prompt can burn 6,000 tokens without closing the trace |
| OOM at boot | KV pin from the other model; `VLLM_KV_CACHE_MEMORY` is absolute and OOMs rather than shrinking |
| `400` on long prompts | Above the cap. Both blocks now run 131,072; the ceiling is 162,496 and costs no VRAM |
| A big prompt seems to hang | Not hung — quadratic prefill. 9.6k ≈ 26 s, 32k ≈ 5 min, 64.7k ≈ 22 min. Raise your client timeout |
| Context ceiling dropped after a tuning change | You raised `--max-num-batched-tokens`; it inflates the sliding-window reservation |
| Boot segfaults in `getSortedImages` | `SYCL_CACHE_PERSISTENT=1` with the V2 runner. Must stay `0` |
| Same id listed twice | A `--served-model-name` value is repeated in `compose.yaml`; vLLM does not dedupe |
| Orphan-container warning | The other engine's container lingers; `down` that folder. Never `--remove-orphans` |

---

## Open questions

1. A genuine **concurrent-load** test — every concurrency figure here is
   *allocated* capacity; `bench.sh` is single-stream.
2. A true **~131k prompt** has never completed end-to-end. The cap is
   boot-verified and exercised to ~64.7k; everything above that is the fitted
   curve, not measurement.
3. **fp8 KV cache, untested.** `TritonAttentionBackend.supported_kv_cache_dtypes`
   includes `fp8`, and the SM89 guard that would reject it sits inside
   `if current_platform.is_cuda():` — **not gated on XPU**. It would halve both KV
   terms (131k at ~2.3×, or the full 262,144 in budget), does nothing for prefill,
   and its accuracy cost is unmeasured. Needs a `--kv-cache-dtype` passthrough.
4. **Speculative decoding, untested and the only real decode lever left.**
   `gemma4_mtp` is a registered method, the proposer ships at
   `vllm/v1/spec_decode/gemma4.py`, `platforms/xpu.py` has no guard against it,
   and Google publishes a matching 0.84 GB assistant checkpoint. It needs the
   context cap down to ~96k or below to make VRAM room, and its CUDA-graph path
   won't apply here.
5. **File the V2 segfault upstream** — clean repro, no existing issue.
6. **The head-512 capability gate** is arguably also worth reporting: the kernel
   is compiled and unreachable. See *Why the slow kernel gets chosen*.
7. Whether `/dev/dri/by-path` is still needed at world_size=1.
8. The stock `intel/vllm:0.21.0` baseline in `vllm_xpu/`, still unmeasured.

---

See also: the [top-level README](../README.md) for the stack overview, firewall
and volumes; [`scaler/README.md`](../scaler/README.md) for the alternative engine
and its measurement log; [`vllm_xpu/README.md`](../vllm_xpu/README.md) for the
stock Intel baseline.

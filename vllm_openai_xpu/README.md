# Upstream vLLM XPU engine

This folder runs **upstream's own** XPU image, `vllm/vllm-openai-xpu`, rather than
Intel's fork used by `scaler/`. It serves the same OpenAI-compatible API on the
same port, so it's a drop-in replacement for the other two engines here.

It's configured for **three validated models**: `gemma-4-26B-A4B-it`,
`Qwen3.8-27B` and `gpt-oss-20b`. Switching between them is a `.env` edit plus a
recreate. It is also the only engine in this repo that can load gemma-4, which
is why it's the one currently serving.

Most measurements below were taken on an Arc Pro B60 (24 GB), and the ones from
the B70 say so. The Arc Pro B70 replaced the B60 on 2026-09-26 and runs gemma-4
about 1.3× faster at decode, so don't compare B70 numbers with B60 ones
directly. The settings that depend on the card, and the headline numbers for
each, are in [INTEL_ARC_B60.md](../INTEL_ARC_B60.md) and
[INTEL_ARC_B70.md](../INTEL_ARC_B70.md). Where a number is a prediction or an
extrapolation, it says so.

> **One GPU, one engine.** Exactly one may run at a time — `docker compose down`
> in `vllm_xpu/` or `scaler/` before starting this one. All three deliberately
> share the Compose project name `llm` so they reuse the `hf-cache` volume, which
> is also why you must **never pass `--remove-orphans`**: in a shared project it
> deletes the other engines' containers.

**Current state**

| | |
|---|---|
| GPU | Arc Pro B70 |
| Image | `vllm/vllm-openai-xpu:v0.30.0` |
| Model runner | V2 (upstream default from 0.29.0) |
| Model served | `gemma-4-26B-A4B-it`, offline int4 group-32 |
| Context | 131,072 |
| KV pool | 394,408 tokens = 3.01× concurrency, from a 10.5 GiB `VLLM_KV_CACHE_MEMORY` pin |
| Reasoning | on by default (`VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}`); a request can still turn it off |
| Attention | Intel's flash-attention kernel, so **text-only** — see [*Intel attention for gemma-4*](#intel-attention-for-gemma-4) and [PLUGIN.md](PLUGIN.md) |
| Per-request stats | on — `metrics` in chat and completions responses; a stream needs `include_usage` |
| `smoke.sh` | ALL PASS, with reasoning on and off |

Kept as the cross-engine comparison point, measured for **gpt-oss-20b** on this
same engine on the B60: 83.5 tok/s @200, 83.1 @400, TTFT ~76 ms, KV pool 338,928 tokens
= 2.59× @128k.

---

## Contents

- [Quick start](#quick-start)
- [Configuration — `.env`](#configuration--env)
- [The models](#the-models)
- [Switching models](#switching-models)
- [Verifying which model is live](#verifying-which-model-is-live)
- [Reasoning / thinking](#reasoning--thinking)
- [Model naming](#model-naming)
- [`bench.sh` and `smoke.sh`](#benchsh-and-smokesh)
- [Downstream consumers](#downstream-consumers)
- [Per-request stats — TTFT and tok/s](#per-request-stats--ttft-and-toks)
- [gemma-4 in depth](#gemma-4-in-depth) — checkpoint, context, prefill, Intel attention, prefix cache
- [Qwen3.8-27B](#qwen38-27b) — checkpoint, Intel attention with images, measured against Triton
- [Performance](#performance)
- [Why it is configured this way](#why-it-is-configured-this-way)
- [Shared project and volumes](#shared-project-and-volumes)
- [Troubleshooting](#troubleshooting)
- [Open questions](#open-questions)

The plugin that lets gemma-4 use Intel's attention kernel has its own page,
[PLUGIN.md](PLUGIN.md): what it is, how vLLM loads it, where the idea came from,
and why it costs image input.

---

## Quick start

```bash
cd vllm_openai_xpu
docker compose up -d
cd ..                                   # the scripts live at the repo root
./smoke.sh        # correctness: served, content, reasoning, tool-calling
./bench.sh 400    # speed: TTFT and tok/s
```

Cold boot is ~2.5 min. `/v1/models` starts answering **before** generation is
ready, so trust `smoke.sh` over the first `200`.

### Applying a `.env` change

A plain `docker compose up -d` applies it. Compose keeps a hash of each
container's settings, with the `.env` values already filled in, and recreates the
container whenever that hash changes. Switching between model blocks, a shell
override, or changing `VLLM_KV_CACHE_MEMORY` by a single byte all
produce a new hash (checked on Compose v5.5.1 with `docker compose config --hash`
and `up -d --dry-run`). Running from the repo root with
`-f vllm_openai_xpu/compose.yaml` reads the same `.env`.

Two things do **not** pick up an edit: `docker compose restart`, which restarts
the existing container with its old settings, and an `export`ed shell variable,
which beats `.env` until you `unset` it. Add `--force-recreate` only when you want
a fresh container with unchanged settings, for example to start from an empty
prefix cache.

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

These are all **model-specific**, so they move as a set, and each block in
`.env.example` sets every one of them. `!` marks the three whose wrong value breaks a boot or
silently degrades it.

| Variable | What it sets |
|----------|--------------|
| `VLLM_MODEL` | Hugging Face repo ID |
| `VLLM_SERVED_MODEL_NAME` | The id clients call it by — see *Model naming* |
| `VLLM_REASONING_PARSER` | Model-family specific; wrong value = empty reasoning, **not** a crash |
| `VLLM_TOOL_CALL_PARSER` | Model-family specific, same quiet failure mode |
| `VLLM_MAX_MODEL_LEN` **!** | Context window; must fit VRAM after weights |
| `VLLM_KV_CACHE_MEMORY` **!** | KV pool in **absolute bytes**; overrides util, skips profiling, and OOMs rather than shrinking |
| `VLLM_EAGER_FLAG` **!** | `--enforce-eager` (required by gpt-oss's 8.0 GiB B60 pin) or `--no-enforce-eager` (gemma-4 and Qwen3.8, measured fine at 131,072). Passed whole — `--enforce-eager=False` does not parse |
| `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` | Server-side chat-template defaults. The gemma-4 block sets `{"enable_thinking":true}`, which turns reasoning on for every request; `null` leaves it to each request. Qwen3.8 thinks by default without it |
| `VLLM_ATTN_BACKEND` | `auto` (default) lets vLLM choose; `FLASH_ATTN` asks for Intel's kernel. gemma-4 needs the next variable with it — see *Intel attention for gemma-4*. Qwen3.8 doesn't |
| `VLLM_TEXT_ONLY_FLAG` | `--no-language-model-only` (default) or `--language-model-only`, which turns image input off. Passed whole, like `VLLM_EAGER_FLAG` |

Shell variables beat `.env`, but **only for the names you pass** — so the
temporary-override block under *Switching models* has to set all of them, not
just `VLLM_MODEL`.

> **The variables are not independent.** `VLLM_KV_CACHE_MEMORY` is sized for a
> specific model's weights on a specific card, and it OOMs rather than
> shrinking. On a B60, gemma-4's weights (15.76 GiB) plus gpt-oss's KV pin
> (8.0 GiB) exceed the 22.33 GiB free and fail at boot. Switch a whole block at
> once, and use the line for your card: each card's values are in its card file.

---

## The models

| | gpt-oss-20b | gemma-4-26B-A4B-it (int4) | Qwen3.8-27B (int4) |
|---|---|---|---|
| type | MoE | MoE | dense, with linear-attention layers |
| weights on device | 12.87 GiB | 15.76 GiB (`adeepv`) | 17.56 GiB |
| max context | 131,072 | 131,072 (ceiling 162,496 on the B60's 4.25 GiB pin) | 131,072 (B70) |
| reasoning | always on, no off switch | opt-in per request | on by default, a request can turn it off |
| large cold prompts | linear, fine | **quadratic on Triton — see warning**; fast with Intel attention | fast with Intel attention, slow on Triton |
| image input | no | yes, unless Intel attention is on | yes, also with Intel attention |
| decode on the B70 | not measured | 86 tok/s with Intel attention | 32.9 tok/s with Intel attention |

Speed, first-token time and KV pool depend on the card, so they're in the card
files: [B60](../INTEL_ARC_B60.md#measured-results) and
[B70](../INTEL_ARC_B70.md#measured-gemma-4-on-the-upstream-engine). On the B60,
for example, gpt-oss decoded at 83.2 tok/s (a chunk count, see the note below)
and gemma-4 at 56.2 tok/s (true tokens).

gpt-oss-20b's prefill stays linear, which made it the better choice for
coding-CLI traffic through a gateway while gemma-4 ran on Triton. gemma-4 is the
newer and stronger model and is vision-capable; with Intel attention switched on
it reads a 24k-token prompt in 4.2 s on the B70, at the cost of image input.
Qwen3.8-27B is a dense model, so it decodes at well under half gemma-4's rate,
but it keeps image input on Intel attention — see [*Qwen3.8-27B*](#qwen38-27b).

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
VLLM_ATTN_BACKEND=auto \
VLLM_TEXT_ONLY_FLAG=--no-language-model-only \
docker compose up -d
```

The same for Qwen3.8-27B, with the B70's KV value:

```bash
cd vllm_openai_xpu
docker compose down

VLLM_MODEL=RedHatAI/Qwen3.8-27B-INT4 \
VLLM_SERVED_MODEL_NAME=qwen3.8-27b \
VLLM_REASONING_PARSER=qwen3 \
VLLM_TOOL_CALL_PARSER=qwen3_coder \
VLLM_MAX_MODEL_LEN=131072 \
VLLM_KV_CACHE_MEMORY=8603448832 \
VLLM_EAGER_FLAG=--no-enforce-eager \
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=null \
VLLM_ATTN_BACKEND=FLASH_ATTN \
VLLM_TEXT_ONLY_FLAG=--no-language-model-only \
docker compose up -d
```

Confirm it took over before trusting it:

```bash
cd .. && ./smoke.sh          # should report the new id -> its checkpoint
```

The KV value in that block is the B60's. gpt-oss hasn't been measured on the B70
with this engine, so it has no B70 value yet; the B60 one should boot there but
leaves memory unused (inferred, see [INTEL_ARC_B70.md](../INTEL_ARC_B70.md)).

On the B60, two of those are **not optional**, and they fail differently:

| variable | if you omit it while `.env` holds gemma's value |
|---|---|
| `VLLM_KV_CACHE_MEMORY` | boots, but the 4.25 GiB pin strands ~4 GiB and roughly halves the pool |
| `VLLM_EAGER_FLAG` | **boot fails** — compiled mode with gpt-oss's 8.0 GiB pin starves the KV pool, which is the case `--enforce-eager` exists for |

Every model runs at `131,072`, so `VLLM_MAX_MODEL_LEN` doesn't differ between
the blocks today — but keep it in each block anyway, so they are free to
diverge. `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=null` is tidiness only; an
unknown template kwarg is verified harmless on gpt-oss, which always reasons
regardless.

The served name changes with the model, so update your consumer's model mapping
too — see *Model naming*.

### Back to whatever `.env` says

```bash
docker compose down
docker compose up -d
```

No variables. If you used `export` rather than a one-line prefix, the old values
are still in your shell and will silently win:

```bash
env | grep ^VLLM_     # expect no output
```

Clear them with `unset`, or just open a new terminal.

### Permanent — edit `.env`

Comment out the active model block in `.env` and uncomment another, then
`docker compose up -d`. `.env.example` ships all three blocks ready to swap,
and each one sets every model-specific variable.

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

| model | log line |
|---|---|
| gpt-oss-20b | `Model loading took 12.87 GiB` |
| gemma-4 | `Model loading took 15.76 GiB`, or 14.69 GiB with Intel attention |
| Qwen3.8-27B | `Model loading took 17.56 GiB` |

The pool size comes next, as `XPU KV cache size: … tokens, Maximum concurrency
for … tokens per request: …x` (earlier docs here quote it as `GPU KV cache
size`). It depends
on the card and the KV setting, so the figure to expect is in the card files:
on the B70, gemma-4 shows 394,408 tokens and 3.01× with Intel attention.

For gemma-4 the log should also show the full quantization chain, which confirms
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
- **gemma-4** — **opt-in out of the box.** With
  `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` unset, the compose default `null` applies
  and the server adds no template defaults. A request asks for reasoning
  explicitly:

```json
"chat_template_kwargs": {"enable_thinking": true}
```

To turn it on for *every* request, set this in `.env` and recreate. The
gemma-4 block in `.env.example` ships with it on:

```dotenv
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}
```

Either way a request can override the server default in both directions, so it is
a default and not a lock.

- **Qwen3.8-27B** — thinks by default with no server setting. A request turns it
  off with `"chat_template_kwargs": {"enable_thinking": false}`.

### What reasoning actually costs

**Not decode speed.** Measured on the B60 on one solvable word problem,
streaming with `include_usage`:

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

This is why the template leaves it off: reasoning is a cost you opt into for the
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

**One truthful name per model**: `gpt-oss-20b`, `gemma-4-26b-a4b` and
`qwen3.8-27b`. The id is
set by `VLLM_SERVED_MODEL_NAME` and is not validated against the checkpoint, so
it *could* be used as a fixed label to hide a swap. Don't. The models share a
context cap but differ enormously in prefill cost, so a consumer that
keeps sending gpt-oss-sized cold prompts to gemma-4 gets multi-minute stalls that
look like a gateway fault rather than a deliberate model change.

A swap therefore means updating the consumer's model mapping. That is the point —
it makes the change visible where the behavior actually differs.

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
./smoke.sh                 # ALL PASS on every model as shipped
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
mapping is updated.** That is deliberate: the models have very different
prefill costs, and a fixed label would hide that until it surfaced as a stall.

Two integration details that cause silent breakage:

- The reasoning trace is in **`reasoning`**, not `reasoning_content`.
- gemma-4 reasoning is **opt-in** unless `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS`
  turns it on, so with it unset a gateway that doesn't pass
  `chat_template_kwargs` gets no trace. That's configuration, not a fault.

---

## Per-request stats — TTFT and tok/s

The OpenAI API returns token counts (`usage`) but no timing. vLLM v0.30.0 can
add timing as well, and this engine turns it on with
`--enable-per-request-metrics`. A `/v1/chat/completions` or `/v1/completions`
response then carries a `metrics` object (streams need one more step, below):

| Field | What it measures |
|---|---|
| `time_to_first_token_ms` | From the moment the scheduler picks the request up to the first token, so it covers the whole prefill |
| `queue_time_ms` | Time spent waiting before that, when the engine was busy |
| `generation_time_ms` | First token to last token — decode only |
| `mean_itl_ms` | Average gap between tokens during decode |
| `tokens_per_second` | All generated tokens ÷ (prefill + decode) |
| `speculative_decoding` | Draft-token stats. Always `null` here, since this engine runs no speculative decoding. Streams leave it out |

A real one, from gemma-4 on the Arc Pro B70 (2026-09-26, 29-token prompt, 193
tokens back, thinking off, rounded):

```json
"metrics": {
  "time_to_first_token_ms": 79.3,
  "generation_time_ms": 2691.1,
  "queue_time_ms": 0.01,
  "mean_itl_ms": 14.02,
  "tokens_per_second": 69.7,
  "speculative_decoding": null
}
```

**Streaming needs one more thing from the client.** A stream carries `metrics`
only in its final usage chunk (the one with `"choices": []`), and vLLM sends that
chunk only when the request asks for it:

```json
"stream": true,
"stream_options": {"include_usage": true}
```

Without it a stream has no `metrics`.

**When `metrics` is `null` or missing.** vLLM leaves it out for a request with
`n` > 1, for a `/v1/completions` call with a list of prompts, and on the
`/v1/responses` and `/v1/messages` endpoints. A single `null` field is normal
too: `mean_itl_ms` is `null` when only one token came back.

Things to know when reading the numbers:

- **`tokens_per_second` is not the decode rate.** It includes prefill, so on a
  long prompt it reads far below the real decode speed. The decode rate is
  `1000 / mean_itl_ms`. Measured on the same B70 with an 8,097-token cold
  prompt: TTFT 13.3 s and decode 47.7 tok/s (decode itself slows as the
  context grows), but `tokens_per_second` only 7.9.
- **TTFT here leaves out queueing and the HTTP side.** It starts when the
  scheduler takes the request, so waiting in the queue is reported separately
  as `queue_time_ms`, and chat-template rendering, tokenization, image
  preprocessing and network time aren't counted at all. On an idle engine a
  client-side stopwatch reads a little higher (32 ms against 27.7 ms,
  measured). Under load, add `queue_time_ms` on top.
- **Reasoning counts.** With thinking on, `time_to_first_token_ms` is the time
  to the first reasoning token, not to the first word of the answer, and
  `tokens_per_second` and `mean_itl_ms` include the trace, the same way
  `usage.completion_tokens` does.

It needs the engine's stats logging, which is on by default. vLLM refuses to
start if `--disable-log-stats` is added alongside it.

**Don't add `--enable-force-include-usage` to save clients from sending
`stream_options`.** It puts a running `usage` total on every chunk, and Open
WebUI adds those up, so its token counts come out inflated. Reproduced offline
with Open WebUI's own merge code: 600 prompt tokens reported for 100.

### What the clients do with it

This comes from reading Open WebUI's source and a live test on this host.
Other versions may behave differently.

| Client | What happens to `metrics` |
|---|---|
| Open WebUI v0.11.4 | Ignored: it reads only `usage` and llama.cpp's `timings` |

---

## gemma-4 in depth

### Checkpoint constraint

The checkpoint must be **int4-symmetric with `group_size` 32 or channelwise**.
The XPU expert kernel (`XPUExpertsWNA16`) accepts only those two schemes, so the
smaller and far more popular **group-64** builds are rejected at load despite
being ~0.7 GiB lighter. Check
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

Measured on the B60 against that formula at the 4.25 GiB pin — **raising
`--max-model-len` costs no VRAM at all**, and prediction tracks the engine's own
log to 0.003%:

| `--max-model-len` | per request | concurrency | KV pool (predicted / **logged**) |
|---|---|---|---|
| 32,768 | 1.906 GB | 2.39× | 78,435 / **78,433** |
| 65,536 | 2.578 GB | 1.77× | 116,028 / **116,025** |
| **131,072** (shipped) | 3.920 GB | **1.16×** | 152,596 / **152,592** |
| 162,496 | 4.563 GB | 1.00× | hard ceiling |

Weights, pool, decode and TTFT were byte-for-byte identical across all three
boots; only concurrency moves. A 64,708-token prompt answered a question about
its *last* record correctly, so the window is real and not merely allocated.

The B70 follows the same formula: at its 10.5 GiB setting and 131,072 context the
formula predicts 2.877× and the log says 2.88×. On that card the 1.00× point lies
beyond the checkpoint's own 262,144 limit, so the model is the ceiling there,
not memory (computed, not booted).

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

**Changing the cap does not change decode speed.** Measured on the B60 in one
session, three warm runs at each setting: 46.2 / 46.3 / 46.3 chunk/s and 106 / 106 / 105 ms TTFT
at 32k / 64k / 128k — identical. (Chunk counts, same instrument and same bias
across all three, so the comparison holds even though the absolute figure is
understated.)

The pool is pinned in absolute bytes, so it is 4.25 GiB at every cap, and a given
prompt costs the same wherever the cap sits. Concurrency is only
pool ÷ one-max-length-request — a ratio, not a capacity.

What the cap **does** control is the longest prefill a client can trigger. On the
B60, on Triton:

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

### ⚠ Prefill on Triton is quadratic

On the default Triton backend this, not VRAM, is the real limit on usable
context. [Intel attention](#intel-attention-for-gemma-4) removes it if you can
do without image input.

Measured on the B60. The B70 follows the same curve about 1.4–1.5× faster
(11,782 tokens in 27.7 s, 23,307 in 107.2 s).

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
> `libattn_kernels_xe_2.so` in this image contains the prefill kernels
> `chunk_policy_head512` and `chunk_policy_head512_b16`, plus a head-512
> paged-decode kernel built for 64-token blocks (`q8_h512_p64`: 8 query heads per
> KV head, exactly gemma-4's full-attention layout). vLLM doesn't use it because
> the gate above asks *"are you FlashAttention 4?"*, a question about CUDA
> lineage, rather than *"can you do head_size 512?"*. No setting reaches past that
> gate; the small plugin in `head512_plugin/` does — see the next section and
> [PLUGIN.md](PLUGIN.md).

Requesting `--attention-backend=FLASH_ATTN` on its own is **honoured, then
refused** on the head-size gate. (vLLM 0.30 has no `VLLM_ATTENTION_BACKEND`
environment variable any more; the flag is the only way to ask.)

And a hypothetical per-layer split wouldn't rescue long prompts: at 131k the five
512-dim layers do **92.7%** of the attention work. It *would* help ordinary
traffic, though — the two layer groups cross over at
`25 × n × 1024 = 5 × n²/2`, i.e. **n ≈ 10,240 tokens**, so below that the
twenty-five 256-dim sliding layers carry most of the attention cost and they are
FA2-eligible on paper.

### Intel attention for gemma-4

Two lines in `.env` put gemma-4 on Intel's head-512 kernel instead of Triton:

```dotenv
VLLM_ATTN_BACKEND=FLASH_ATTN
VLLM_TEXT_ONLY_FLAG=--language-model-only
```

Then `docker compose up -d`. To go back, comment both out and run `up -d` again.

**What it costs: image input.** gemma-4 attends to image tokens in both
directions (`use_bidirectional_attention: vision` in its config), and vLLM only
supports that on FlashAttention 4 or Triton. `--language-model-only` turns image
input off, and that is what makes FLASH_ATTN eligible. Setting only
`VLLM_ATTN_BACKEND` fails the boot with *"mm_prefix (PrefixLM bidirectional
attention) requires FlashAttention v4"* (read from the source, not boot-tested).

**How it works.** The only thing in the way is the head-size gate: vLLM accepts
head sizes above 256 for FLASH_ATTN only with FA4. `head512_plugin/` is a vLLM
general plugin that makes it accept exactly 512 on XPU. Compose mounts it at
`/opt/vllm-plugins` and puts that on `PYTHONPATH`, so every vLLM process loads
it and logs `xpu_head512: FLASH_ATTN on XPU now accepts head size 512`. It
changes nothing else: with the switch off, gemma-4 still gets Triton, and
gpt-oss's head size is 64. FLASH_ATTN on XPU uses 64-token KV blocks, which is
the block size the head-512 decode kernel is built for. [PLUGIN.md](PLUGIN.md)
walks through the plugin file by file.

**Measured** on the B70 on 2026-09-30. Both columns use the same checkpoint, the
131,072 context cap, the 10.5 GiB KV pin and the same scripts, with true token
counts:

| | Triton (switch off) | Intel attention |
|---|---|---|
| decode, 512 tokens, thinking off / on | 73.7–74.3 / 74.4–74.5 tok/s | **85.7–86.3 / 86.5–86.7 tok/s** |
| cold prefill, ~11.8k tokens | 27.7 s | **1.7 s** |
| cold prefill, ~24.2k tokens | 116.6 s | **4.2 s** |
| 16,049-token prompt: first token | 51.1 s | **2.5 s** |
| 16,049-token prompt: decode after it | 38.1 tok/s | **74.8 tok/s** |
| KV pool | 376,999 tokens (2.88×) | 394,408 tokens (3.01×) |
| image input | yes | no |

Longer prompts, measured with Intel attention only: 67,956 tokens in 17.8 s and
129,331 tokens in 51.7 s. For scale, Triton took about 13 minutes for a
60,924-token prompt on the same card (one run, with a second request
overlapping).

The answers held up. The fixed questions got the same answers at temperature 0,
and `smoke.sh` passed with thinking on and off. A code hidden in documents of
11,991, 67,956 and 129,331 tokens (5–50% of the way in) was found every time.
That code sits far outside the 1024-token sliding window, so only the head-512
layers can retrieve it.

**After an image upgrade, check the boot log before trusting it:**

- `xpu_head512: FLASH_ATTN on XPU now accepts head size 512` means the plugin
  loaded;
- `Using Flash Attention backend.` and `Setting kv cache block size to 64` mean
  it took effect;
- there must be no `[vllm_xpu_kernels] XPU kernel not compiled … falling back to
  PyTorch reference attention`. Intel's wrapper silently switches to a slow
  reference path when a kernel variant is missing, and that line is the only
  sign.

If vLLM starts asking the XPU kernels what they support, the plugin does nothing
and can be deleted.

### Prefix caching is what makes this usable

Same prompt sent twice on a virgin engine:

| n | cold | warm | speedup |
|---|---|---|---|
| 9,643 | 26.3 s | 0.31 s | **84×** |
| 31,957 | 305.3 s | 0.63 s | **481×** |

These are Triton numbers from the B60. Warm time stays roughly flat while cold grows
quadratically, so the ratio climbs with prompt size. This also **refutes** a plausible worry: sliding-window layers
free their out-of-window blocks mid-request (`remove_skipped_blocks`), which
looked like it should defeat reuse on long prompts. It doesn't.

So conversation and growing context are fine; a large **cold** context (fresh RAG
document, big paste) is where it hurts. **You pay it once per document per engine
lifetime** — the cache is VRAM-resident, so recreating the container throws it away.

Two practical consequences:

- **Put stable content first.** A document at the top of the prompt stays a
  reusable prefix across many different questions; the same document placed
  *after* a varying question caches nothing, because matching runs from token 0.
- **Growing a conversation costs the same total as one big prefill** — it's the
  same triangle either way. What changes is that it's spread out, so per-turn
  latency creeps up: on Triton a ~2k-token turn costs ~40 s at 32k of context and
  ~2.2 min at 100k.

With Intel attention a cold prompt is 16–28× cheaper, so the cache matters much
less, though it still works the same way.

---

## Qwen3.8-27B

Block C in `.env.example`. It's a dense 27B model: 48 of its 64 layers are
linear attention (Gated DeltaNet) and 16 are full attention, so the KV cache
grows with only those 16 layers. Every token reads all of its weights, which
is why it decodes at well under half gemma-4's rate.

**Checkpoint.** `RedHatAI/Qwen3.8-27B-INT4`: int4 symmetric, group size 128,
in compressed-tensors format, with the vision encoder kept at full precision.
It loads as 17.56 GiB and runs on `XPUwNa16LinearKernel`, the same int4
kernel gemma-4's non-expert layers use. This engine can't quantize a
full-precision checkpoint to int4 as it loads (the scaler can), so it needs a
pre-quantized one. For a dense model the XPU kernel takes symmetric or
asymmetric int4 with any group size that's a multiple of 32, which most int4
builds on Hugging Face meet. Check `quantization_config` before trying another.

**Intel attention, with images.** The full-attention heads are size 256, which
Intel's kernel takes without the head-512 plugin. Qwen also doesn't use
gemma-4's bidirectional attention over image tokens, so `auto` already picks
FLASH_ATTN and image input keeps working. The block sets `FLASH_ATTN`
explicitly anyway, so the choice is visible. In the log you should see
`Using Flash Attention backend.`, and `Setting attention block size to 1600
tokens`: vLLM makes the attention blocks large enough to hold the
linear-attention state, and this is normal.

The log also mentions Triton and even CUDA, and none of it means attention fell
back. `Using Triton/FLA GDN prefill kernel` and `GDN decode kernel: cuda` come
from shared setup code that runs on every platform. On XPU the linear-attention
layers then call Intel's own kernel (`torch.ops._xpu_C.gdn_attention`) instead.
`Warmed M-RoPE Triton kernels` is the position encoding, a small step outside
attention. All of this is read from the v0.30.0 source.

**Measured** on the B70 on 2026-10-02, same checkpoint, 131,072 context, the
8.0 GiB KV value and compiled mode, true token counts, thinking off:

| | Triton | Intel attention |
|---|---|---|
| decode, 512 tokens | 29.7 tok/s | **32.9 tok/s** |
| 9,411-token prompt: first token | 73.6 s | **5.9 s** |
| 18,786-token prompt: first token | 283.0 s | **12.8 s** |
| 18,786-token prompt: decode after it | 5.3 tok/s | **30.2 tok/s** |
| code hidden 10% into those prompts | found | found |
| image input | works | works |
| KV pool | 244,270 tokens (1.86×) | 244,270 tokens (1.86×) |
| `smoke.sh` | ALL PASS | ALL PASS |

On Triton the first-token time roughly quadruples when the prompt doubles, so
long prompts get slow fast. With Intel attention it grows in line with the
prompt. For comparison, the scaler runs the official BF16 checkpoint
quantized to int4 as it loads, at 28.7 tok/s, and fits only 98,304 tokens of
context on the same card ([INTEL_ARC_B70.md](../INTEL_ARC_B70.md)).

**Leave room for thinking.** Qwen3.8 thinks by default and at length. On
`bench.sh`'s default question it reasoned for 2,477 tokens, about 80 s, before
the first word of the answer, so `./bench.sh 400` and `./bench.sh 2000` both
end with no answer at all. Use `THINKING=0 ./bench.sh 400` to measure speed
(32.4 tok/s, measured), and give clients a `max_tokens` of several thousand,
or have them turn thinking off for quick replies.

Compose doesn't pin a revision, so a new upload to that repo loads on the next
boot. The measured one is `91bd022d5b49442a868bc35008f6c21e1860edfa`.

---

## Performance

### Eager vs compiled — take compiled on gemma-4

Upstream runs `torch.compile` even with cudagraphs off, and those buffers are
**not** capped by `--gpu-memory-utilization`. On the B60, with **gpt-oss's
8.0 GiB pin**, they starved the KV pool and killed the boot:

```
Available KV cache memory: 2.89 GiB
ValueError: max seq len (131072) needs 3.1 GiB KV cache > available 2.89 GiB
```

Missing 128k by 0.21 GiB — which is why `--enforce-eager` is the compose default,
and note it's for a *different* reason than on the scaler, where eager prevents
silently-empty content.

On the same card, gemma-4's smaller 4.25 GiB pin leaves headroom, and compiled
mode is measured free — at 32,768, 65,536 **and** 131,072, all boot-tested with the pool intact.
The compile buffers scale with `max_num_batched_tokens` (2496), *not* with
context, so context length doesn't change this trade-off:

| config | decode (true tokens) | KV pool | concurrency |
|---|---|---|---|
| eager | 52.0 tok/s | 78,433 | 2.39× |
| **compiled** | **~56 tok/s** (+7%) | 78,433 | 2.39× |
| compiled + XPU graph | +0.5% more | 46,138 | 1.41× |

Those are B60 figures, taken at the then-shipped 32,768 cap; at 131,072 the
B60's pool is 152,592 tokens. The B70 runs compiled too, at its 10.5 GiB setting.

```dotenv
VLLM_EAGER_FLAG=--no-enforce-eager
```

Compiled costs ~34 s of `torch.compile` per boot (the SYCL cache must stay off,
so it never persists — though it does load from the AOT compile cache in well
under a second once warm).

**XPU graph is not worth it.** On the B60 it gave +0.5% for 1.54 GiB of capture
memory. At the old 32k cap that dropped the pool to 46,138 tokens; at **131,072
it doesn't fit at all** — 1.54 GiB out of a 4.25 GiB pool leaves less than one
max-length request, so the engine won't boot. It does genuinely capture graphs
in compiled mode; the earlier "no-op" result was an artifact of only ever
testing it under eager.

Re-checked on the B70 with Intel attention, where each token takes less GPU
time: during a long single-request decode the GPU was still 99–100% busy
(`sudo nvtop`), so there are no kernel-launch gaps for graphs to remove. The
boot-log line *"XPU Graph is disabled by environment variable"* is expected and
harmless.

**Compiled mode does not help prefill** (2.28 / 5.56 / 19.77 s at 2.4k / 4.8k /
9.6k tokens, vs eager's 2.15 / 5.61 / 19.38). That confirms the quadratic prefill
is the forced Triton attention backend, not kernel-launch overhead. No compile or
batching setting fixes it; changing the attention backend does (*Intel attention
for gemma-4*).

### Why gemma-4 is slower than gpt-oss

Not bandwidth — gemma reads *less* per token (~3.23 GB vs ~3.71 GB, counting
int4/MXFP4 weights plus the unquantized embedding). It's op count and
granularity:

| | gemma-4 | gpt-oss |
|---|---|---|
| layers | 30 | 24 |
| experts activated / layer | 8 of 128 | 4 of 32 |
| expert intermediate size | 704 | 2880 |
| expert GEMVs per token | **240** | 96 |
| attention backend | TRITON_ATTN (forced), or FLASH_ATTN with the Intel-attention switch | FLASH_ATTN |
| expert kernel | generic `XPUExpertsWNA16` | `XPUExpertsMxFp4` (tuned) |

2.5× as many expert matrix multiplies per token, each ~4.1× narrower. Small
matrices use the GPU poorly and each costs a launch. Compiled mode recovers part
of that; the rest is architectural, and it sets the ceiling — there is no
configuration that makes this model reach gpt-oss's rate.

### Standings against the scaler

Both engines have the `--kv-cache-memory-bytes` lever, measured on the same B60
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

### 0.29.0 → 0.30.0 — a like-for-like swap

Measured 2026-09-22 on the B60 with gemma-4 (int4 group-32, 45,056 context,
5.25 GiB pinned pool, compiled), same script against both images, true token counts from the
streamed usage chunk:

| | 0.29.0 | 0.30.0 |
|---|---|---|
| decode, 512 tok, thinking off | 56.38 tok/s | 56.46 tok/s |
| TTFT, short prompt | 40 ms | 40 ms |
| cold prefill, 8,161 tok | 19.14 s | 19.14 s |
| cold prefill, 16,301 tok | 74.11 s | 74.33 s |
| 4 concurrent, aggregate | 181.8 tok/s | 182.1 tok/s |
| KV pool | 117,681 tok (2.61×) | 117,681 tok (2.61×) |
| weights on device | 15.87 GiB | 15.76 GiB |
| engine init | 65 s | 109 s cold cache |
| `smoke.sh`, thinking on and off | ALL PASS | ALL PASS |

Nothing moved. The release's XPU work (fused GemmaRMSNorm, SYCL activation ops,
triton-xpu 3.8.0) doesn't show up in decode, and prefill is still quadratic
because the backend is still forced to TRITON_ATTN — the boot log prints the same
"FA4 not available" line. The same `--kv-cache-memory-bytes` value produced the
same pool, so no `.env` value needs to change.

Two things to know before you read your own numbers after an upgrade:

- **The first concurrent batch after a cache wipe is slow.** It read 165 tok/s
  once, then 181.6–182.4 on four repeats — Triton compiling the new batch shapes
  (`jit_monitor` logs it). Warm the engine before measuring.
- **The 109 s init includes a 33.6 s compile** into the freshly emptied
  `vllm-openai-cache` volume; later boots reuse it (the next compiled boot:
  72 s, 2.6 s of it compiling).

**The release's new XPU kernels only run in eager mode, and compiled still
wins.** Under torch.compile the XPU platform sets op priority to `native`, so
the fused GemmaRMSNorm and SYCL activation ops are bypassed — the boot log shows
`custom_ops: ['none']`. `--enforce-eager` switches them on (`custom_ops: ['all']`,
`vllm_c` first), measured on 0.30.0 with the same pool:

| | compiled | eager + fused kernels |
|---|---|---|
| decode | **56.46 tok/s** | 53.25 tok/s (−5.7%) |
| 4 concurrent, aggregate | **182.1 tok/s** | 176.0 tok/s (−3.3%) |
| cold prefill, 8,161 tok | 19.14 s | 19.22 s |

Keep `VLLM_EAGER_FLAG=--no-enforce-eager`.

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

On the B60, `--gpu-memory-utilization 0.80` caps the engine at 18.17 GiB of a
22.71 GiB card, stranding ~4.5 GiB. The engine prints the exact byte value to
reclaim it:

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
3. **It is model-specific and card-specific.** On a B60, gemma-4's weights plus
   gpt-oss's 8.0 GiB pin exceed the 22.33 GiB free and fail at boot. Switch a
   whole `.env` block, and use the value for your card.
4. **Canonical spelling is `--kv-cache-memory-bytes`.** The log advises
   `--kv-cache-memory`, which only works as an argparse prefix abbreviation.

On the B60, gemma-4's 4.25 GiB pin leaves ~0.15 GiB of card headroom. That's
defensible only because the compile buffers are bounded by
`max_num_batched_tokens` rather than by context.

On the B70 the same one-boot check printed 11,916,317,184 bytes (11.1 GiB) for
gemma-4, and the setting used is 11,274,289,152 (10.5 GiB), which keeps about
1.1 GiB free for those buffers. Each card's values are in
[INTEL_ARC_B60.md](../INTEL_ARC_B60.md#settings-for-this-card) and
[INTEL_ARC_B70.md](../INTEL_ARC_B70.md#settings-for-this-card).

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
currently only supports single-GPU execution"), so a single-card layout is the
only one that could use it — going dual would forfeit it. See *Performance* for
why it isn't taken.

### What upstream does not provide

Intel's fork keeps some Arc B-series-specific surface upstream lacks: the online int4 path
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
| `.env` edit had no effect | `docker compose restart` was used (it keeps the old settings; use `up -d`), the file wasn't saved, or an `export`ed shell variable is overriding it (`env \| grep VLLM_`) |
| `HTTP 404 ... model does not exist` | Stale `MODEL` in your shell, or the gateway still asks for the previous block's name. `bench.sh` prints what *is* served |
| Reasoning field always empty | Two candidates: gemma-4 without `enable_thinking` (opt-in unless `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` sets it), or the client reading `reasoning_content` instead of `reasoning` |
| Trace returned but no answer | Reasoning consumed the whole `max_tokens`. Raise it — and note a hard prompt can burn 6,000 tokens without closing the trace |
| OOM at boot | KV pin from the other model; `VLLM_KV_CACHE_MEMORY` is absolute and OOMs rather than shrinking |
| `400` on long prompts | Above the cap. Every block runs 131,072; on gemma-4 the ceiling is 162,496 and costs no VRAM |
| A big prompt seems to hang | Not hung — quadratic prefill on Triton. 9.6k ≈ 26 s, 32k ≈ 5 min, 64.7k ≈ 22 min (B60). Raise your client timeout, or switch on Intel attention if you can do without images |
| Boot fails: `mm_prefix … requires FlashAttention v4` | `VLLM_ATTN_BACKEND=FLASH_ATTN` without `VLLM_TEXT_ONLY_FLAG=--language-model-only`. Set both or neither |
| Boot fails with FLASH_ATTN refusing head size 512 | The plugin didn't load. Check the `head512_plugin` mount and `PYTHONPATH` in `compose.yaml`, and look for `xpu_head512` in the log. [PLUGIN.md](PLUGIN.md) covers how it loads |
| `400 At most 0 image(s) may be provided in one prompt` | `--language-model-only` is set. On gemma-4 that comes with Intel attention. Comment out both switch lines and `up -d` to get images back. On Qwen3.8 it's a leftover from the gemma-4 block: set `VLLM_TEXT_ONLY_FLAG=--no-language-model-only` |
| Much slower than *Intel attention for gemma-4* says, after an image upgrade | Look for `XPU kernel not compiled … falling back` in the log: a kernel variant is missing from the new image |
| Context ceiling dropped after a tuning change | You raised `--max-num-batched-tokens`; it inflates the sliding-window reservation |
| Boot segfaults in `getSortedImages` | `SYCL_CACHE_PERSISTENT=1` with the V2 runner. Must stay `0` |
| `metrics` is `null` or missing | Engine not recreated since the flag was added; the stream didn't send `stream_options.include_usage`; the request went through a gateway that drops it (see *What the clients do with it*); `n` > 1 or a `/v1/completions` call with a list of prompts; or the endpoint is `/v1/responses` or `/v1/messages`. Only `mean_itl_ms` `null` just means one token came back |
| Same id listed twice | A `--served-model-name` value is repeated in `compose.yaml`; vLLM does not dedupe |
| Orphan-container warning | The other engine's container lingers; `down` that folder. Never `--remove-orphans` |

---

## Open questions

1. A genuine **concurrent-load** test — every concurrency figure here is
   *allocated* capacity; `bench.sh` is single-stream.
2. On Triton, a true **~131k prompt** has never completed end-to-end. The cap
   is boot-verified and exercised to ~64.7k; everything above that is the
   fitted curve, not measurement. With Intel attention, a 129,331-token prompt
   completed in 51.7 s.
3. **fp8 KV cache, untested.** `TritonAttentionBackend.supported_kv_cache_dtypes`
   includes `fp8`, and the SM89 guard that would reject it sits inside
   `if current_platform.is_cuda():` — **not gated on XPU**. It would halve both KV
   terms (131k at ~2.3×, or the full 262,144 in budget), does nothing for prefill,
   and its accuracy cost is unmeasured. Needs a `--kv-cache-dtype` passthrough.
4. **Speculative decoding, untested and the only real decode lever left.**
   `gemma4_mtp` is a registered method, the proposer ships at
   `vllm/v1/spec_decode/gemma4.py`, `platforms/xpu.py` has no guard against it,
   and Google publishes a matching 0.84 GB assistant checkpoint. On the B60 it
   needs the context cap down to ~96k or below to make VRAM room, and its
   CUDA-graph path won't apply here.
5. **File the V2 segfault upstream** — clean repro, no existing issue.
6. **The head-512 capability gate** is worth reporting: the kernel is compiled,
   and only `head512_plugin/` makes it reachable. A fix upstream would let the
   plugin go. See *Why the slow kernel gets chosen*.
7. **Image input with Intel attention.** It needs bidirectional attention over
   image tokens. vLLM's FLASH_ATTN path allows that only with FA4, and nobody
   has checked whether Intel's kernel could do it. See
   [PLUGIN.md](PLUGIN.md#the-drawback-no-image-input).
8. **Concurrent load with Intel attention**, untested. Every figure in its table
   is a single stream.
9. Whether `/dev/dri/by-path` is still needed at world_size=1.
10. The stock `intel/vllm:0.21.0` baseline in `vllm_xpu/`, still unmeasured.

---

See also: the [top-level README](../README.md) for the stack overview, firewall
and volumes; [`scaler/README.md`](../scaler/README.md) for the alternative engine
and its measurement log; [`vllm_xpu/README.md`](../vllm_xpu/README.md) for the
stock Intel baseline.

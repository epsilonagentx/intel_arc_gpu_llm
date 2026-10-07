# Upstream vLLM XPU engine

This folder runs **upstream's own** XPU image, `vllm/vllm-openai-xpu`, rather than
Intel's fork used by `scaler/`. It serves the same OpenAI-compatible API on the
same port, so it's a drop-in replacement for the other two engines here.

It's configured for **three validated models**: `gemma-4-26B-A4B-it`,
`Qwen3.8-27B` and `gpt-oss-20b`. Switching between them is a `.env` edit plus a
recreate. It is also the only engine in this repo that can load gemma-4, which
is why it's the one currently serving.

This page covers the engine and what every model shares. What's specific to
one model has its own page: [GEMMA_4_26B_A4B.md](GEMMA_4_26B_A4B.md) and [QWEN3_8_27B.md](QWEN3_8_27B.md).

Every measurement says which card it was taken on. The B70 runs gemma-4 about
1.3× faster than the B60 at decode, so don't compare numbers across cards. The
settings that depend on the card are in [INTEL_ARC_B60.md](../INTEL_ARC_B60.md)
and [INTEL_ARC_B70.md](../INTEL_ARC_B70.md). Where a number is a prediction
rather than a measurement, it says so.

> **One GPU, one engine.** Exactly one may run at a time — `docker compose down`
> in `vllm_xpu/` or `scaler/` before starting this one. All three deliberately
> share the Compose project name `llm` so they reuse the `hf-cache` volume, which
> is also why you must **never pass `--remove-orphans`**: in a shared project it
> deletes the other engines' containers.

**Current state**

| | |
|---|---|
| GPU | Arc Pro B70 |
| Image | `vllm/vllm-openai-xpu:v0.31.0` |
| Model runner | V2 (upstream default from 0.29.0) |
| XPU graphs | on, 0.31.0's default (0.74 GiB for gemma-4) |
| Model served | `gemma-4-26B-A4B-it`, offline int4 group-32 |
| Context | 131,072 |
| KV pool | 393,915 tokens = 3.01× concurrency, from an 11.0 GiB `VLLM_KV_CACHE_MEMORY` pin; with 256 requests at once only 0.24 GiB of the card stays free — see [*Images with Intel attention*](GEMMA_4_26B_A4B.md#images-with-intel-attention) |
| Speculative decoding | on, Google's gemma-4 draft model with 3 tokens: about 1.7× faster decode — see [*Faster decode with a draft model*](GEMMA_4_26B_A4B.md#faster-decode-with-a-draft-model) |
| Reasoning | on by default (`VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}`); a request can still turn it off |
| Attention | Intel's flash-attention kernel, with image input — see [*Intel attention for gemma-4*](GEMMA_4_26B_A4B.md#intel-attention-for-gemma-4) and [PLUGIN.md](PLUGIN.md) |
| Per-request stats | on — `metrics` in chat and completions responses; a stream needs `include_usage` |
| `smoke.sh` | ALL PASS, with reasoning on and off |
| ⚠ Known issue | gemma-4 can get stuck thinking in long coding-agent sessions and repeat itself until the token limit — see [*Known issue: thinking loops in agent sessions*](GEMMA_4_26B_A4B.md#known-issue-thinking-loops-in-agent-sessions) |

---

## Contents

- [Quick start](#quick-start)
- [Configuration — `.env`](#configuration--env)
- [The models](#the-models)
- [Switching models](#switching-models)
- [Verifying which model is live](#verifying-which-model-is-live)
- [Reasoning / thinking](#reasoning--thinking)
- [Model names and clients](#model-names-and-clients)
- [`bench.sh` and `smoke.sh`](#benchsh-and-smokesh)
- [Per-request stats — TTFT and tok/s](#per-request-stats--ttft-and-toks)
- [Performance](#performance)
- [Why it is configured this way](#why-it-is-configured-this-way)
- [Shared project and volumes](#shared-project-and-volumes)
- [Troubleshooting](#troubleshooting)
- [Open questions](#open-questions)

Other pages in this folder:

- [GEMMA_4_26B_A4B.md](GEMMA_4_26B_A4B.md): gemma-4's checkpoint, context, prefill, Intel attention, draft
  model, prefix cache and the known thinking-loop issue
- [QWEN3_8_27B.md](QWEN3_8_27B.md): Qwen3.8's checkpoint, Intel attention with images, and MTP with
  an FP8 output layer
- [PLUGIN.md](PLUGIN.md): the plugin that lets gemma-4 use Intel's attention
  kernel, and how images are read with it
- [tools/README.md](tools/README.md): the output-layer converter for Qwen3.8

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

A plain `docker compose up -d` applies it: Compose recreates the container
whenever any filled-in setting changes, even by one byte. Two things do **not**
pick up an edit: `docker compose restart`, which keeps the old settings, and an
`export`ed shell variable, which beats `.env` until you `unset` it. Use
`--force-recreate` only for a fresh container with the same settings, for
example to empty the prefix cache.

---

## Configuration — `.env`

`compose.yaml` takes every model-specific setting from `.env` in this folder,
so you never edit the compose file to change models. Copy the template and
edit:

```bash
[ -f .env ] || cp .env.example .env    # never clobber an existing .env
```

`.env` is gitignored; `.env.example` documents every variable. Each model block
in it sets all of the variables below, because they belong together. `!` marks
the three whose wrong value breaks the boot or quietly makes things worse.

| Variable | What it sets |
|----------|--------------|
| `VLLM_MODEL` | Hugging Face repo ID, or a local folder such as Qwen3.8's FP8 copy ([QWEN3_8_27B.md](QWEN3_8_27B.md#faster-decode-mtp-and-an-fp8-output-layer)) |
| `VLLM_SERVED_MODEL_NAME` | The id clients call it by — see *Model names and clients* |
| `VLLM_REASONING_PARSER` | Model-family specific; wrong value = empty reasoning, **not** a crash |
| `VLLM_TOOL_CALL_PARSER` | Model-family specific, same quiet failure mode |
| `VLLM_MAX_MODEL_LEN` **!** | Context window; must fit VRAM after weights |
| `VLLM_KV_CACHE_MEMORY` **!** | KV pool in **absolute bytes**; overrides util, skips profiling, and OOMs rather than shrinking |
| `VLLM_EAGER_FLAG` **!** | `--enforce-eager` (required by gpt-oss's 8.0 GiB B60 pin) or `--no-enforce-eager` (gemma-4 and Qwen3.8, measured fine at 131,072). Passed whole — `--enforce-eager=False` does not parse |
| `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` | Server-side chat-template defaults. The gemma-4 block sets `{"enable_thinking":true}`, which turns reasoning on for every request; `null` leaves it to each request. Qwen3.8 thinks by default without it |
| `VLLM_ATTN_BACKEND` | `auto` (default) lets vLLM choose; `FLASH_ATTN` asks for Intel's kernel, which gemma-4 can use only through `head512_plugin/` — see [*Intel attention for gemma-4*](GEMMA_4_26B_A4B.md#intel-attention-for-gemma-4) |
| `VLLM_TEXT_ONLY_FLAG` | `--no-language-model-only` (default) or `--language-model-only`, which turns image input off and gives gemma-4 about 5% more KV pool. Passed whole, like `VLLM_EAGER_FLAG` |
| `VLLM_SPECULATIVE_CONFIG` | Speculative decoding as JSON; `null` (default) is off. The gemma-4 block names Google's draft model with 3 draft tokens ([GEMMA_4_26B_A4B.md](GEMMA_4_26B_A4B.md#faster-decode-with-a-draft-model)). Qwen3.8's block uses the MTP head its checkpoint ships, `{"method":"mtp","num_speculative_tokens":3}` ([QWEN3_8_27B.md](QWEN3_8_27B.md#faster-decode-mtp-and-an-fp8-output-layer)). Each draft fits only its own model, so gpt-oss's block sets `null` |
| `VLLM_MAX_NUM_SEQS` | Most requests at once; 256 (default) is vLLM's own. Qwen3.8's block sets 128, because graph capture needs one linear-attention state block per request and its KV setting holds only 154 |

> **Switch a whole block at once.** `VLLM_KV_CACHE_MEMORY` is sized for one
> model's weights on one card, and the boot fails rather than shrinking it: on
> a B60, gemma-4's weights plus gpt-oss's KV value don't fit. Use the value for
> your card; each card file lists them.

---

## The models

| | gpt-oss-20b | gemma-4-26B-A4B-it (int4) | Qwen3.8-27B (int4) |
|---|---|---|---|
| type | MoE | MoE | dense, with linear-attention layers |
| weights on device | 12.87 GiB | 15.76 GiB (`adeepv`) | 17.56 GiB, 17.18 with an FP8 output layer |
| max context | 131,072 | 131,072 (ceiling 162,496 on the B60's 4.25 GiB pin) | 131,072 (B70) |
| reasoning | always on, no off switch | opt-in per request | on by default, a request can turn it off |
| large cold prompts | linear, fine | **quadratic on Triton — see warning**; fast with Intel attention | fast with Intel attention, slow on Triton |
| image input | no | yes, also with Intel attention (from 0.31.0) | yes, also with Intel attention |
| decode on the B70 | not measured | 137–156 tok/s with Intel attention and the draft model, 86 without the draft | 32.9 tok/s with Intel attention, 65.8 with MTP and an FP8 output layer |

Each card's measurements are in its card file:
[B60](../INTEL_ARC_B60.md#measured-results) and
[B70](../INTEL_ARC_B70.md#measured-gemma-4-on-the-upstream-engine). gemma-4 is
the newest and strongest of the three; with Intel attention it reads a
24k-token prompt in 4.2 s on the B70 and still takes images. Qwen3.8-27B
decodes at well under half gemma-4's rate. The details
are in [GEMMA_4_26B_A4B.md](GEMMA_4_26B_A4B.md) and [QWEN3_8_27B.md](QWEN3_8_27B.md). On the B60, gpt-oss decoded at 83.2 tok/s.

> ⚠ **`bench.sh` counts stream chunks, not tokens.** With reasoning on, gemma-4
> packs 1.15 tokens per chunk, so `bench.sh` reads about 15% low. For true
> rates, use the usage chunk (`stream_options.include_usage`) or the `metrics`
> object — see *Per-request stats*. gpt-oss always reasons, so its 83.2 is a
> chunk count too and understated by an unknown amount.

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
VLLM_SPECULATIVE_CONFIG=null \
VLLM_MAX_NUM_SEQS=256 \
docker compose up -d
```

The same for Qwen3.8-27B on the B70, with MTP and the FP8 output layer. Make
the FP8 copy first, as
[QWEN3_8_27B.md](QWEN3_8_27B.md#faster-decode-mtp-and-an-fp8-output-layer)
explains; for the plain checkpoint use `RedHatAI/Qwen3.8-27B-INT4` and `null`
instead:

```bash
cd vllm_openai_xpu
docker compose down

VLLM_MODEL=/cache/huggingface/local/Qwen3.8-27B-INT4-fp8head \
VLLM_SERVED_MODEL_NAME=qwen3.8-27b \
VLLM_REASONING_PARSER=qwen3 \
VLLM_TOOL_CALL_PARSER=qwen3_coder \
VLLM_MAX_MODEL_LEN=131072 \
VLLM_KV_CACHE_MEMORY=8603448832 \
VLLM_EAGER_FLAG=--no-enforce-eager \
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=null \
VLLM_ATTN_BACKEND=FLASH_ATTN \
VLLM_TEXT_ONLY_FLAG=--no-language-model-only \
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}' \
VLLM_MAX_NUM_SEQS=128 \
docker compose up -d
```

Confirm it took over before trusting it:

```bash
cd .. && ./smoke.sh          # should report the new id -> its checkpoint
```

The gpt-oss KV value is the B60's. gpt-oss has no B70 value yet; the B60 one
should boot there but leaves memory unused (inferred). On the B60, leaving out
`VLLM_EAGER_FLAG=--enforce-eager` makes the gpt-oss boot fail, and leaving out
its KV value roughly halves the pool.

The served name changes with the model, so update your client's model mapping
too — see *Model names and clients*.

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

It also reports `max_model_len`, the context that's actually live; `.env`
only says what the next container will get. Expected in the container log:

| model | log line |
|---|---|
| gpt-oss-20b | `Model loading took 12.87 GiB` |
| gemma-4 | `Model loading took 15.76 GiB`, or 14.69 GiB with Intel attention, or 15.47 GiB with Intel attention and the draft model, or 16.55 GiB with images on as well |
| Qwen3.8-27B | `Model loading took 17.56 GiB` |

The pool size comes next, as `XPU KV cache size: … tokens, Maximum concurrency
for … tokens per request: …x`. It depends on the card and the KV setting: on
the B70, gemma-4 with Intel attention, the draft model and images on shows
393,915 tokens and 3.01× at the 11.0 GiB setting.

For gemma-4 the log should also show the int4 kernels in use:

```
Using XPUwNa16LinearKernel for CompressedTensorsWNA16
Using CompressedTensorsWNA16MoEMethod
Using 'XPU' WNA16 MoE backend.
Using XPUExpertsWNA16
```

---

## Reasoning / thinking

vLLM puts the trace in **`message.reasoning`** (`delta.reasoning` when
streaming), **never `reasoning_content`**. A client that reads
`reasoning_content` sees nothing and looks like a broken parser.

- **gpt-oss-20b** always reasons; there's no off switch.
- **gemma-4** reasons only when asked. A request turns it on with
  `"chat_template_kwargs": {"enable_thinking": true}`. The gemma-4 block in
  `.env.example` turns it on for every request with
  `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}`, and a request
  can still turn it off. The compose default, `null`, adds nothing (not `{}`:
  a brace inside `${VAR:-default}` breaks Compose).
- **Qwen3.8-27B** thinks by default. A request turns it off with
  `"chat_template_kwargs": {"enable_thinking": false}`.

### What reasoning actually costs

**Not decode speed, but tokens.** On the B60, one word problem took 433 tokens
in 7.8 s with thinking off and 1,033 tokens in 19.4 s with it on: 56.2 against
53.7 tok/s, and the same correct answer. Reasoning is worth it only for the
requests that need it. For finer control than on/off, route `enable_thinking`
per client at the gateway.

**Capping the thinking.** `enable_thinking` is on or off; gemma-4's template
has nothing in between. Since 0.30.0, vLLM has a per-request cap,
`thinking_token_budget`, which works on gemma-4 with just the `gemma4`
reasoning parser. Measured: a budget of 64 gives exactly 64 reasoning tokens
and then the answer. The server has no default for it, so each request has to
send it.

### ⚠ Never use `max_tokens` as a thinking cap

Reasoning uses up `max_tokens` **first**. If it runs out mid-trace you get
`finish_reason: length` and **no answer at all**. Measured: a prompt with no
clean solution burned 6,000 tokens over 133 seconds and returned nothing. With
reasoning on, allow 1,500+ tokens for ordinary work, or cap the thinking with
`thinking_token_budget` instead.

---

## Model names and clients

Each model is served under its own name: `gpt-oss-20b`, `gemma-4-26b-a4b` and
`qwen3.8-27b`, set by `VLLM_SERVED_MODEL_NAME`. Don't reuse one name to hide a
swap. The models behave very differently (gemma-4 on Triton takes minutes over
a large cold prompt that gpt-oss reads in seconds), and a fixed name makes that
look like a gateway fault. A swap means updating the client's model mapping,
and until then the client gets a `404`. `smoke.sh` prints which name maps to
which checkpoint.

The endpoint is OpenAI-compatible and **unauthenticated** — see *Firewall* in
the [top-level README](../README.md). Any AI gateway works; point it at
`http://<host>:8000/v1`. Two things break quietly: a client reading
`reasoning_content` instead of `reasoning`, and a gateway that doesn't pass
`chat_template_kwargs` when you rely on it to turn gemma-4's thinking on or off.

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

## Per-request stats — TTFT and tok/s

The OpenAI API returns token counts (`usage`) but no timing. This engine adds
timing with `--enable-per-request-metrics`: a `/v1/chat/completions` or
`/v1/completions` response then carries a `metrics` object.

| Field | What it measures |
|---|---|
| `time_to_first_token_ms` | From when the scheduler picks the request up to the first token, so the whole prefill |
| `queue_time_ms` | Time spent waiting before that, when the engine was busy |
| `generation_time_ms` | First token to last token — decode only |
| `mean_itl_ms` | Average gap between tokens during decode; `null` if only one token came back |
| `tokens_per_second` | All generated tokens ÷ (prefill + decode) |
| `speculative_decoding` | Stays `null`, even with the draft model on: vLLM fills it only with `per_request_spec_decode_metrics`, which isn't set here. The engine log's `SpecDecoding metrics` lines carry the same numbers every 10 s |

A real one, from gemma-4 on the B70 (29-token prompt, 193 tokens back,
thinking off, rounded):

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

**A stream carries `metrics` only if it asks for usage**, in the final chunk:

```json
"stream": true,
"stream_options": {"include_usage": true}
```

Reading the numbers:

- **`tokens_per_second` is not the decode rate.** It includes prefill. The
  decode rate is `1000 / mean_itl_ms`. Measured on the B70 with an 8,097-token
  cold prompt: decode 47.7 tok/s, but `tokens_per_second` only 7.9.
- **TTFT leaves out queueing and the HTTP side** (template, tokenizing,
  network). Under load, add `queue_time_ms`.
- **Reasoning counts.** With thinking on, TTFT is the time to the first
  reasoning token, and the rates include the trace.

Don't add `--enable-force-include-usage` to spare clients the
`stream_options`: it puts a running `usage` total on every chunk, and Open
WebUI adds them up, so its token counts come out inflated. Open WebUI v0.11.4
ignores `metrics` anyway; it reads only `usage` and llama.cpp's `timings`.

---

## Performance

### Eager vs compiled — take compiled on gemma-4

gemma-4 and Qwen3.8 run compiled (`VLLM_EAGER_FLAG=--no-enforce-eager`). On the
B60, compiled decoded gemma-4 at about 56 tok/s against eager's 52.0 (+7%),
with the same KV pool. It costs about 34 s of `torch.compile` on the first boot;
later boots load it from the compile cache.

The compose default is eager because of gpt-oss on the B60: the compile
buffers aren't covered by `--gpu-memory-utilization`, and with gpt-oss's
8.0 GiB KV value they left too little memory for 131,072 tokens and the boot
failed. The buffers grow with `max_num_batched_tokens`, not with context, so a
longer context doesn't change this. That budget is vLLM's 2048 for cards
under 70 GiB, raised to 2496 for gemma-4 with image input on.

Compiled doesn't help prefill; only the attention backend does. On 0.30.0,
the release's new fused XPU kernels ran only in eager mode, and compiled still
beat them (56.46 against 53.25 tok/s, measured on the B60).

**XPU graphs are on from 0.31.0.** Compiled mode now also captures XPU graphs,
and the old `VLLM_XPU_ENABLE_XPU_GRAPH` switch is gone (the log calls it an
unknown variable if it's still set). On the B70 they fit next to both models'
KV settings: 0.74 GiB for gemma-4, 0.96 GiB for Qwen3.8. They make gemma-4
decode about 8% faster after a long prompt and change little else (see
*0.30.0 → 0.31.0* below). Qwen3.8 needs `VLLM_MAX_NUM_SEQS=128` for the
capture to fit. On 0.30.0 graphs gave only +0.5% on the B60 for 1.54 GiB, and
at 131,072 context they didn't fit there; 0.31.0 hasn't been booted on a B60.

### Against the scaler and across vLLM versions

For gpt-oss-20b on the B60 the scaler is about 3% faster (85.6 against 83.1
tok/s) with a slightly larger pool, so there's no speed reason to move gpt-oss
here. This engine's case is that it is mainline vLLM, several versions newer,
and **the only engine here that runs gemma-4**. Going from 0.29.0 to 0.30.0
changed nothing measurable on gemma-4 (decode, prefill, concurrency and pool
all within 0.3%), and the same `.env` values carried over. Warm the engine up
before measuring after an upgrade: the first concurrent batch after a cache
wipe reads about 9% slow while Triton compiles new shapes.

### 0.30.0 → 0.31.0

Measured on the B70 on 2026-10-05, same `.env` values, same benchmark (three
512-token answers with thinking off, a long prompt with a hidden code, and 6
requests at once), warm engines:

| | 0.30.0 | 0.31.0, graphs off | 0.31.0, graphs on |
|---|---|---|---|
| gemma-4 decode, temperature 0 | 148 tok/s | 147 | 149 |
| gemma-4 decode after a 12,940-token prompt | 107.7 tok/s | 119.0 | **128.0** |
| gemma-4, 6 requests at once | 392 tok/s | 394–396 | 389–397 |
| Qwen3.8 decode, temperature 0 | 69–71 tok/s | 65–66 | 65–66 |
| Qwen3.8 decode after a 14,386-token prompt | 67.4 tok/s | 68.2 | 70.3 |
| Qwen3.8, 6 requests at once | 217–219 tok/s | 220–221 | 218–220 |

First-token times and KV pools didn't change, and `smoke.sh` passed on every
boot. gemma-4 with Intel attention and the draft model gains the most after a
long prompt. Qwen3.8's lower temperature-0 figure comes entirely from one of
the three test prompts, the other two are unchanged; why is unknown. 0.31.0
also uses about 2 GiB less card memory for Qwen3.8. The head-size check that
`head512_plugin/` patches is unchanged, so the plugin works as before.

The run above was text-only. 0.31.0 also lets gemma-4 keep image input on
Intel attention, which 0.30.0 refused at boot. That costs 4.7% of the KV pool
and no decode speed; see
[*Images with Intel attention*](GEMMA_4_26B_A4B.md#images-with-intel-attention).

---

## Why it is configured this way

### Why this engine exists

Upstream vLLM lists **Intel Arc Pro B-Series** as validated hardware and
supports MXFP4 on XPU, so a B-series card doesn't need Intel's fork. gpt-oss's
MXFP4 weights load natively, at 12.87 GiB. Being in upstream's model list says
the architecture is supported, not that it fits: `gpt-oss-120b` is listed too,
and its 60.7 GiB fit neither card.

The XPU images are in their own Docker Hub repo, `vllm/vllm-openai-xpu`, not
`vllm/vllm-openai`, which has no XPU tags. `latest` there tracks the newest
release, but compose pins a version so an upgrade is a deliberate change.

### `SYCL_CACHE_PERSISTENT=0` — mandatory with the V2 runner

With `1`, the V2 runner segfaults at boot, in `getSortedImages`, with no Python
traceback. Its warm-up exercises `torch.topk` (for logprobs, which this setup
never asks for), and the XPU kernel crashes while loading from the persistent
SYCL code cache. No setting skips that warm-up, so the cache stays off; the
cost is a kernel rebuild on every boot. Not yet reported upstream. The stock
engine in `vllm_xpu/` runs fine with `1`.

### The V2 model runner

V2 is the default since 0.29.0 and uses less memory than V1 (about 1.03 GiB of
non-torch memory against 1.50, and a larger KV pool). The log line
`xpu_worker.py Using V2 Model Runner` confirms it; V1 prints nothing.
`Initializing a V1 LLM engine` is about the engine, not the runner.
`VLLM_USE_V2_MODEL_RUNNER` overrides it and takes only `0` or `1`; a blank
value crashes.

### `--kv-cache-memory-bytes` — the big capacity lever

`--gpu-memory-utilization` leaves memory unused: on the B60, 0.80 left about
4.5 GiB idle. On boot the engine prints the exact byte value that would use it
all:

```
Replace gpu_memory_utilization config with
  `--kv-cache-memory=8603448832` (8.01 GiB) to fully utilize gpu memory
```

Taking it doubled gpt-oss's pool on the B60, from 169,123 to 338,928 tokens, at
the same speed. That value is what `VLLM_KV_CACHE_MEMORY` holds. Before you
change it:

1. **It's absolute.** It replaces the utilization setting and skips memory
   profiling, and it **fails the boot instead of shrinking** if it doesn't fit.
2. **It belongs to one model, one card and one runner.** Re-derive it after an
   image or runner change: comment it out for one boot and read the value back.
3. **Leave room for the compile buffers.** On the B70 the boot advised 11.1 GiB
   for gemma-4 text-only, and 10.5 GiB was used, keeping about 1.1 GiB free.
   With the draft model and images on it's 11.0 GiB, found by a load test
   rather than the boot's advice, and the margin under full load is thin —
   see [*Images with Intel attention*](GEMMA_4_26B_A4B.md#images-with-intel-attention).

The log spells it `--kv-cache-memory`, which works only as an abbreviation; the
full flag is `--kv-cache-memory-bytes`. Each card's values are in
[INTEL_ARC_B60.md](../INTEL_ARC_B60.md#settings-for-this-card) and
[INTEL_ARC_B70.md](../INTEL_ARC_B70.md#settings-for-this-card).

### Device passthrough

The container gets the whole `/dev/dri` device **and** a read-only
`/dev/dri/by-path` bind, because oneCCL finds devices through `by-path`. That's
upstream's documented recipe. Since 0.29.0 the bind may be unnecessary on one
GPU, but it stays until a boot without it proves that.

Upstream's example uses `--privileged` and `--network=host`. This repo uses
`group_add` (the `render` and `video` groups) and a normal port mapping instead,
which is enough. The group IDs differ per host: check yours with
`getent group render video`.

---

## Shared project and volumes

All three engine folders use the Compose project name `llm`, so their
`hf-cache:` volume is the same real volume, `llm_hf-cache`, and they share the
weights. Compile caches stay per engine, because the kernels belong to the
image.

| | |
|---|---|
| Project | `llm` (shared with `vllm_xpu/`, `scaler/`) |
| Weights (shared) | `llm_hf-cache` |
| Compile cache (own) | `llm_vllm-openai-cache` |

- **Orphan-container warnings are expected** when another engine's container
  lingers. Fix by running `down` in that folder. **Never `--remove-orphans`** —
  in one shared project it deletes the other engines' containers.
- **`down -v` from any engine folder deletes `llm_hf-cache`** and the weights
  in it. Use plain `down`.
- **`HF_CACHE` points the weights at a host folder** instead of `llm_hf-cache`,
  and `VLLM_USER` runs the engine as you so the files stay yours. Engines still
  share the weights as long as every folder's `.env` uses the same `HF_CACHE`,
  and `down -v` can't delete a host folder. [../README.md](../README.md) *Volumes*

**Swap procedure** — one GPU, so exactly one engine runs at a time:

```bash
docker compose -f vllm_xpu/compose.yaml down     # if the stock engine is up
docker compose -f scaler/compose.yaml down       # if the scaler is up
docker compose -f vllm_openai_xpu/compose.yaml up -d
```

Stop any other server on the GPU and port 8000 first, such as a `llama.cpp`
stack; if a process supervisor runs it, it will come back by itself.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `.env` edit had no effect | `docker compose restart` was used (it keeps the old settings; use `up -d`), the file wasn't saved, or an `export`ed shell variable is overriding it (`env \| grep VLLM_`) |
| `HTTP 404 ... model does not exist` | Stale `MODEL` in your shell, or the gateway still asks for the previous block's name. `bench.sh` prints what *is* served |
| Reasoning field always empty | Two candidates: gemma-4 without `enable_thinking` (opt-in unless `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` sets it), or the client reading `reasoning_content` instead of `reasoning` |
| Trace returned but no answer | Reasoning consumed the whole `max_tokens`. Raise it — and note a hard prompt can burn 6,000 tokens without closing the trace |
| Thinking never ends and the same sentence repeats, in an agent session | The known gemma-4 thinking loop. Stop the request; the next one normally works. See [*Known issue: thinking loops in agent sessions*](GEMMA_4_26B_A4B.md#known-issue-thinking-loops-in-agent-sessions) |
| OOM at boot | KV pin from the other model; `VLLM_KV_CACHE_MEMORY` is absolute and OOMs rather than shrinking |
| `400` on long prompts | Above the cap. Every block runs 131,072; on gemma-4 the ceiling is 162,496 and costs no VRAM |
| A big prompt seems to hang | Not hung — quadratic prefill on Triton. 9.6k ≈ 26 s, 32k ≈ 5 min, 64.7k ≈ 22 min (B60). Raise your client timeout, or switch on Intel attention |
| Boot fails: `mm_prefix … requires FlashAttention v4` | vLLM 0.30.0 or older with `VLLM_ATTN_BACKEND=FLASH_ATTN` and images on. Those versions need `VLLM_TEXT_ONLY_FLAG=--language-model-only` with it; 0.31.0 boots without |
| Log warns `prefix-LM bidirectional mask cannot be applied, so image/video inputs will produce incorrect results` | Expected for gemma-4 on Intel attention with images on. Nine test images gave the same answers as on Triton — see [*Images with Intel attention*](GEMMA_4_26B_A4B.md#images-with-intel-attention) |
| Log warns `max_num_scheduled_tokens is set to 2048` (or 2496) `based on the speculative decoding settings` | Harmless here. vLLM prints it whenever the draft model is on and the per-step budget is under 8192, which is the default only on cards of 70 GiB or more. With 3 draft tokens 2048 still fits 512 requests per step. Don't raise `--max-num-batched-tokens`: it lowers gemma-4's context ceiling |
| Boot fails with FLASH_ATTN refusing head size 512 | The plugin didn't load. Check the `head512_plugin` mount and `PYTHONPATH` in `compose.yaml`, and look for `xpu_head512` in the log. [PLUGIN.md](PLUGIN.md) covers how it loads |
| `400 At most 0 image(s) may be provided in one prompt` | `--language-model-only` is set. Set `VLLM_TEXT_ONLY_FLAG=--no-language-model-only` and `up -d` to get images back; on 0.31.0 gemma-4 keeps Intel attention with it |
| Much slower than [*Intel attention for gemma-4*](GEMMA_4_26B_A4B.md#intel-attention-for-gemma-4) says, after an image upgrade | Look for `XPU kernel not compiled … falling back` in the log: a kernel variant is missing from the new image |
| Context ceiling dropped after a tuning change | You raised `--max-num-batched-tokens`; it inflates the sliding-window reservation |
| Boot fails: `max_num_seqs (256) exceeds available Mamba cache blocks` | Qwen3.8 with graphs needs `VLLM_MAX_NUM_SEQS=128` (or anything up to the number in the message) |
| Boot segfaults in `getSortedImages` | `SYCL_CACHE_PERSISTENT=1` with the V2 runner. Must stay `0` |
| `metrics` is `null` or missing | Engine not recreated since the flag was added; the stream didn't send `stream_options.include_usage`; the request went through a gateway that drops it; `n` > 1 or a `/v1/completions` call with a list of prompts; or the endpoint is `/v1/responses` or `/v1/messages`. Only `mean_itl_ms` `null` just means one token came back |
| Same id listed twice | A `--served-model-name` value is repeated in `compose.yaml`; vLLM does not dedupe |
| Orphan-container warning | The other engine's container lingers; `down` that folder. Never `--remove-orphans` |

---

## Open questions

gemma-4's own are in [GEMMA_4_26B_A4B.md](GEMMA_4_26B_A4B.md#open-questions).

1. **Heavier concurrent load.** The only real test is 6 requests at a time,
   512 tokens each (323 tok/s in total, 386 with the draft model). Every other
   concurrency figure is allocated capacity, and `bench.sh` sends one request.
2. **Report the V2 segfault upstream.** It reproduces cleanly, and there's no
   existing issue.
3. **The stock `intel/vllm:0.21.0` baseline** in `vllm_xpu/`, still unmeasured.

---

See also: the [top-level README](../README.md) for the stack overview, firewall
and volumes; [`scaler/README.md`](../scaler/README.md) for the alternative engine
and its measurement log; [`vllm_xpu/README.md`](../vllm_xpu/README.md) for the
stock Intel baseline.

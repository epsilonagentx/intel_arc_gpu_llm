# Upstream vLLM XPU engine

This folder runs **upstream's own** XPU image, `vllm/vllm-openai-xpu`, rather than
Intel's fork used by `scaler/`. It serves the same OpenAI-compatible API on the
same port, so it's a drop-in replacement for the other two engines here.

It's configured for **three validated models**: `gemma-4-26B-A4B-it`,
`Qwen3.8-27B` and `gpt-oss-20b`. Switching between them is a `.env` edit plus a
recreate. It is also the only engine in this repo that can load gemma-4, which
is why it's the one currently serving.

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
| Image | `vllm/vllm-openai-xpu:v0.30.0` |
| Model runner | V2 (upstream default from 0.29.0) |
| Model served | `gemma-4-26B-A4B-it`, offline int4 group-32 |
| Context | 131,072 |
| KV pool | 356,893 tokens = 2.72× concurrency, from a 9.5 GiB `VLLM_KV_CACHE_MEMORY` pin |
| Speculative decoding | on, Google's gemma-4 draft model with 3 tokens: about 1.7× faster decode — see [*Faster decode with a draft model*](#faster-decode-with-a-draft-model) |
| Reasoning | on by default (`VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}`); a request can still turn it off |
| Attention | Intel's flash-attention kernel, so **text-only** — see [*Intel attention for gemma-4*](#intel-attention-for-gemma-4) and [PLUGIN.md](PLUGIN.md) |
| Per-request stats | on — `metrics` in chat and completions responses; a stream needs `include_usage` |
| `smoke.sh` | ALL PASS, with reasoning on and off |
| ⚠ Known issue | gemma-4 can get stuck thinking in long coding-agent sessions and repeat itself until the token limit — see [*Known issue: thinking loops in agent sessions*](#known-issue-thinking-loops-in-agent-sessions) |

---

## Contents

- [Quick start](#quick-start)
- [Configuration — `.env`](#configuration--env)
- [The models](#the-models)
- [Switching models](#switching-models)
- [Verifying which model is live](#verifying-which-model-is-live)
- [Reasoning / thinking](#reasoning--thinking)
- [Known issue: thinking loops in agent sessions](#known-issue-thinking-loops-in-agent-sessions)
- [Model names and clients](#model-names-and-clients)
- [`bench.sh` and `smoke.sh`](#benchsh-and-smokesh)
- [Per-request stats — TTFT and tok/s](#per-request-stats--ttft-and-toks)
- [gemma-4 in depth](#gemma-4-in-depth) — checkpoint, context, prefill, Intel attention, draft model, prefix cache
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
| `VLLM_MODEL` | Hugging Face repo ID |
| `VLLM_SERVED_MODEL_NAME` | The id clients call it by — see *Model names and clients* |
| `VLLM_REASONING_PARSER` | Model-family specific; wrong value = empty reasoning, **not** a crash |
| `VLLM_TOOL_CALL_PARSER` | Model-family specific, same quiet failure mode |
| `VLLM_MAX_MODEL_LEN` **!** | Context window; must fit VRAM after weights |
| `VLLM_KV_CACHE_MEMORY` **!** | KV pool in **absolute bytes**; overrides util, skips profiling, and OOMs rather than shrinking |
| `VLLM_EAGER_FLAG` **!** | `--enforce-eager` (required by gpt-oss's 8.0 GiB B60 pin) or `--no-enforce-eager` (gemma-4 and Qwen3.8, measured fine at 131,072). Passed whole — `--enforce-eager=False` does not parse |
| `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` | Server-side chat-template defaults. The gemma-4 block sets `{"enable_thinking":true}`, which turns reasoning on for every request; `null` leaves it to each request. Qwen3.8 thinks by default without it |
| `VLLM_ATTN_BACKEND` | `auto` (default) lets vLLM choose; `FLASH_ATTN` asks for Intel's kernel. gemma-4 needs the next variable with it — see *Intel attention for gemma-4*. Qwen3.8 doesn't |
| `VLLM_TEXT_ONLY_FLAG` | `--no-language-model-only` (default) or `--language-model-only`, which turns image input off. Passed whole, like `VLLM_EAGER_FLAG` |
| `VLLM_SPECULATIVE_CONFIG` | `null` (default) is off. The gemma-4 block names Google's draft model and 3 draft tokens — see *Faster decode with a draft model*. The draft belongs to one model, so every other block sets `null` |

> **Switch a whole block at once.** `VLLM_KV_CACHE_MEMORY` is sized for one
> model's weights on one card, and the boot fails rather than shrinking it: on
> a B60, gemma-4's weights plus gpt-oss's KV value don't fit. Use the value for
> your card; each card file lists them.

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
| decode on the B70 | not measured | 137–156 tok/s with Intel attention and the draft model, 86 without the draft | 32.9 tok/s with Intel attention |

Each card's measurements are in its card file:
[B60](../INTEL_ARC_B60.md#measured-results) and
[B70](../INTEL_ARC_B70.md#measured-gemma-4-on-the-upstream-engine). gemma-4 is
the newest and strongest of the three; with Intel attention it reads a
24k-token prompt in 4.2 s on the B70, but loses image input. Qwen3.8-27B
decodes at well under half gemma-4's rate but keeps image input — see
[*Qwen3.8-27B*](#qwen38-27b). On the B60, gpt-oss decoded at 83.2 tok/s.

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
VLLM_SPECULATIVE_CONFIG=null \
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
| gemma-4 | `Model loading took 15.76 GiB`, or 14.69 GiB with Intel attention, or 15.47 GiB with Intel attention and the draft model |
| Qwen3.8-27B | `Model loading took 17.56 GiB` |

The pool size comes next, as `XPU KV cache size: … tokens, Maximum concurrency
for … tokens per request: …x`. It depends on the card and the KV setting: on
the B70, gemma-4 with Intel attention and the draft model shows 356,893 tokens
and 2.72×.

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
has nothing in between. vLLM 0.30.0 adds a per-request cap,
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

## Known issue: thinking loops in agent sessions

**Not fixed yet.** In long sessions with a coding agent, gemma-4 sometimes
starts thinking and never stops, so the answer never comes. It goes on until
the client gives up or the request's token limit runs out. The next request in
the same session usually works fine. It has been seen on the B70 with Intel
attention, with and without the draft model.

**What the captured loops look like.** Five were caught on 3 and 4 October
2026, with a coding agent (opencode) talking to the engine through a gateway:

- All five were stuck in the thinking. None reached the answer or a tool call.
- They started 25,000–55,000 tokens into the conversation, so it isn't only a
  problem near the context limit.
- Most came straight after a failed or broken file edit, at a point where the
  model had to choose what to do next.
- The model usually had the right idea in its first few hundred characters.
  Then it kept reopening the decision ("Actually, I'll use `write`." / "Wait,
  I'll check…") until it settled into a block of 54 to 357 characters that it
  repeated word for word, 150–235 times.
- While it loops, the draft model's guesses are accepted 98–100% of the time,
  against 50–65% for normal text. The engine log's `SpecDecoding metrics` lines
  show it. Copying a file word for word also gives about 100%, so this hints at
  a loop but doesn't prove one.

**Capping the thinking isn't enough on its own.** A temporary plugin gave every
request a default `thinking_token_budget` of 4,096 (see *What reasoning actually
costs* above). It did cut the loop, and the model then made a valid tool call.
But the agent sends the model's reasoning back with each tool step, the chat
template puts it into the next prompt, and the next step looped again within
its first 1,000 characters. The cut-off loop seeded the next one. The plugin
was taken out again, and nothing from it is in this repository.

**The likely cause: the chat template feeds old reasoning back in.** gemma-4's
template keeps the reasoning of every assistant step after the last user
message and renders it as a thought block in the next prompt. In this
checkpoint's `chat_template.jinja` that's lines 239–242:

```jinja
{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}
{%- if thinking_text and thinking_gate -%}
    {{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}
```

The first half of the gate is the one that fires here. The second half,
`preserve_thinking`, is a template option that defaults to off. During a chain
of tool calls there's no new user message, so each step sees all of its own
earlier thinking again, and a loop can feed on itself. The design looks
deliberate: it lets the model carry its reasoning through a tool chain.

A Hugging Face discussion on Google's model page describes exactly this, and
proposes a fix:

- [*Chat template may re-inject prior-turn reasoning during multi-turn tool use → repetition loops*](https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/48),
  google/gemma-4-26B-A4B-it discussion #48, by ManniX-ITA. With the stock
  template, 4 of 12 seeds of a multi-turn agent test looped. With the
  re-injection switched off (`{%- if false and thinking_text … %}`), none did,
  and code benchmarks held up (HumanEval+ 92.07%). That was measured on a
  pruned derivative of the 26B-A4B, not on this checkpoint.
- [The fixed template](https://huggingface.co/ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF/blob/main/chat_template.fixed.jinja)
  and [its unit test](https://huggingface.co/ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF/blob/main/template_loop_unittest.py),
  linked from that discussion.

**Not tested here yet.** In this checkpoint's template, the same fix means
changing line 241 to `{%- if false and thinking_text and thinking_gate -%}`.
The edited copy would be passed to the server with `--chat-template`, so no
client has to change. What it would cost is unknown: the model would no longer
see its own reasoning from earlier steps in the same tool chain.

**Until then:** stop the looping request; the next one normally works. Don't
reach for repetition or frequency penalties as a workaround: they also push
the model away from copying code exactly, which a coding agent depends on.

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

## gemma-4 in depth

### Checkpoint constraint

The checkpoint must be **int4-symmetric with `group_size` 32 or channelwise**.
The XPU expert kernel (`XPUExpertsWNA16`) accepts only those two schemes, so the
smaller and far more popular **group-64** builds are rejected at load despite
being ~0.7 GiB lighter. Check
`quantization_config.config_groups.*.weights.group_size` before trying any other
MoE checkpoint.

### Context is nearly free — the KV formula

gemma-4 has **25 sliding-window layers (window 1024) and 5 full-attention
layers**. vLLM sizes the sliding layers by the window, not by
`--max-model-len`, so only the 5 full-attention layers grow with context, and
they're the cheap ones (2 KV heads × 512, against 8 × 256):

```
fixed  (25 sliding layers, window-bounded) = 1.235 GB      # context-independent
linear (5 full layers)                     = 20 KiB/token
```

So **raising `--max-model-len` costs no VRAM**; it only lowers the concurrency
figure. On the B60's 4.25 GiB setting the formula matched the logged pool to
0.003% at 32k, 64k and 131k, with 162,496 as the ceiling. On the B70's 10.5 GiB
setting it predicted 2.877× at 131,072 and the log said 2.88×; there the
checkpoint's own 262,144 limit comes before memory runs out (computed, not
booted). A 64,708-token prompt answered a question about its last record
correctly, so the long context really works.

> **⚠ Raising `--max-num-batched-tokens` lowers the context ceiling.** It
> enlarges the sliding layers' reservation: at 8192 the fixed block grows from
> 1.235 to 3.568 GB and the B60's ceiling falls to 48,576. Leave it at the
> default (2496).

When sizing any hybrid model like this, per-token KV figures mislead. Read
`layer_types`, `sliding_window`, `num_key_value_heads` **and**
`num_global_key_value_heads` from its config.

### The cap is a guardrail, not a speed setting

**The cap doesn't change decode speed** (identical at 32k, 64k and 128k on the
B60). The KV pool is set in bytes, so a given prompt costs the same wherever
the cap sits. What the cap does control is the longest prompt a client can
send. A lower cap turns an oversized prompt into a quick `400` instead of tens
of minutes of prefill on Triton. If you keep 131,072, make sure client
timeouts along the whole path allow for it.

### ⚠ Prefill on Triton is quadratic

On the default Triton backend, prefill time, not VRAM, limits usable context.
[Intel attention](#intel-attention-for-gemma-4) removes the problem if you can
do without image input. Measured on the B60 (the B70 is 1.4–1.5× faster):

| cold prompt | wall |
|---|---|
| 2,421 tok | 2.2 s |
| 9,621 tok | 19.4 s |
| 19,221 tok | 80.8 s |
| **64,708 tok** | **21.6 min** |

That's almost exactly quadratic, `T ≈ 26.3 s × (n / 9643)^2.048`. Beyond 64,708
tokens it's extrapolation: about 51 min at 96k and 92 min at 131k, never run.
Prefill here is even slower per token than decode, which a healthy kernel would
never be.

#### Why the slow kernel gets chosen

The boot log says it:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}.
FA4 not available, forcing TRITON_ATTN backend.
```

vLLM lets FLASH_ATTN take head size 512 only on FlashAttention 4, which is
CUDA-only, so it puts every gemma-4 layer on Triton. Intel's head-512 kernel is
compiled into this image but sits behind that check. No setting gets past it;
the small plugin in `head512_plugin/` does. [PLUGIN.md](PLUGIN.md) walks
through the check and the source.

### Intel attention for gemma-4

Two lines in `.env` put gemma-4 on Intel's head-512 kernel instead of Triton:

```dotenv
VLLM_ATTN_BACKEND=FLASH_ATTN
VLLM_TEXT_ONLY_FLAG=--language-model-only
```

Then `docker compose up -d`. To go back, comment both out and run `up -d` again.

**What it costs: image input.** gemma-4 attends to image tokens in both
directions, which vLLM supports only on FlashAttention 4 or Triton.
`--language-model-only` turns image input off, and that makes FLASH_ATTN
eligible. Setting only `VLLM_ATTN_BACKEND` fails the boot with *"mm_prefix
(PrefixLM bidirectional attention) requires FlashAttention v4"* (read from the
source, not boot-tested).

**How it works.** `head512_plugin/` makes FLASH_ATTN accept head size 512 on
XPU, and changes nothing else: with the switch off gemma-4 still gets Triton.
Compose mounts it and puts it on `PYTHONPATH`, so every vLLM process loads it.
[PLUGIN.md](PLUGIN.md) walks through it file by file.

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

Longer prompts, with Intel attention: 67,956 tokens in 17.8 s and 129,331 in
51.7 s. Triton took about 13 minutes for 60,924 tokens on the same card.

The answers held up: the same answers at temperature 0, `smoke.sh` passed with
thinking on and off, and a code hidden in documents of 11,991, 67,956 and
129,331 tokens was found every time. It sat far outside the 1024-token sliding
window, so only the head-512 layers could have found it.

**After an image upgrade, check the boot log before trusting it:**

- `xpu_head512: FLASH_ATTN on XPU now accepts head size 512` means the plugin
  loaded;
- `Using Flash Attention backend.` and `Setting kv cache block size to 64` mean
  it took effect;
- there must be no `[vllm_xpu_kernels] XPU kernel not compiled … falling back to
  PyTorch reference attention`. Intel's wrapper silently switches to a slow
  reference path when a kernel variant is missing, and that line is the only
  sign.

### Faster decode with a draft model

Google publishes a small "assistant" model for gemma-4,
[`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant):
4 layers, 0.42B parameters, 0.78 GiB. It guesses the next few tokens, and
gemma-4 checks all of them in one pass instead of producing one token per pass.
Wrong guesses are thrown away, so the answers come from gemma-4 exactly as
before. Only the speed changes. Take the QAT variant: the `adeepv` checkpoint
is built from Google's QAT weights, so the matching draft should guess better.

Two lines in the gemma-4 block of `.env` switch it on:

```dotenv
VLLM_SPECULATIVE_CONFIG={"model":"google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant","num_speculative_tokens":3}
VLLM_KV_CACHE_MEMORY=10200547328
```

Then `docker compose up -d`. The first boot downloads the draft. vLLM sees
`model_type: gemma4_assistant` in its config and picks the MTP method itself,
so there's no `"method"` key. The KV setting drops 1 GiB, from 10.5 to 9.5 GiB,
to make room for the draft's weights. The draft adds no KV cache of its own: its
layers have only query projections and read gemma-4's cache. The boot log
shows it as four `Gemma4 MTP: draft layer N … -> language_model.model.layers.28`
(and `.29`) lines.

**Measured** on the B70 on 2026-10-03. Same checkpoint, Intel attention, 131,072
context, compiled. Decode is the median of three 512-token answers (a story,
code, an explanation) with thinking off, each set run twice:

| | without | 3 draft tokens | 4 | 5 |
|---|---|---|---|---|
| decode, default sampling | 82.1 tok/s | **137–146** | 136 | 125 |
| decode, temperature 0 | 86.6 tok/s | **156** | 155 | 152 |
| 6 requests at once, total | 323 tok/s | **386** | — | — |
| 16,853-token prompt: first token | 2.65 s | 3.05 s | — | — |
| 16,853-token prompt: decode after it | 75.0 tok/s | **109.5** | — | — |
| KV pool | 394,408 (3.01×) | 356,893 (2.72×) | same | same |
| weights on the card | 14.69 GiB | 15.47 GiB | same | same |

The gain depends on the text. Code is easiest to guess (167–175 tok/s) and the
story hardest (109–117). The draft guesses well: about 2.75–3.05 tokens are
accepted per pass, and its first, second and third guesses are right about 78%,
55% and 43% of the time. With 4 or 5 tokens the extra guesses are wrong too
often to pay for themselves, so 3 it is.

With several requests at a time the gain shrinks to about 20%, because the GPU
is already busy with the other requests. A long prompt's first token arrives
about 0.4 s later. `smoke.sh` passed, and a code hidden in the 16,853-token
prompt was found with and without the draft.

Google [warns](https://ai.google.dev/gemma/docs/mtp/overview) that on the
26B-A4B mixture-of-experts model the draft may not speed up a single request,
since checking several tokens pulls in more experts. That didn't happen on
this card.

To turn it off, set `VLLM_SPECULATIVE_CONFIG=null`, put the KV setting back to
`11274289152` and run `up -d`.

**Tried and not worth it: a smaller output layer.** gemma-4 shares its 16-bit
output layer with its embeddings, `[262144, 2816]`, 1.47 GB read for every
token. A copy stored as FP8 made no measurable difference on top of the draft
model (143–145 and 158 tok/s). As int4 with group size 32 it added about 5% (147–159 and 164)
but slightly changes the model's output, and that wasn't measured.

### Prefix caching

vLLM reuses the KV of a prompt prefix it has already seen. The same prompt sent
twice on the B60, on Triton:

| n | cold | warm | speedup |
|---|---|---|---|
| 9,643 | 26.3 s | 0.31 s | **84×** |
| 31,957 | 305.3 s | 0.63 s | **481×** |

It works on gemma-4's sliding-window layers too. A growing conversation only
pays for the new turn, though on Triton each turn gets slower as the
conversation grows (a 2k-token turn takes about 40 s at 32k of context and
2.2 min at 100k, on the B60). What hurts most is a large prompt the engine hasn't seen
(a fresh document, a big paste), once per engine lifetime, since the cache
lives in VRAM and a recreate empties it. **Put stable content first:** matching
starts from token 0, so a document placed after a varying question is never
reused. With Intel attention a cold prompt is 16–28× cheaper, so this matters
much less.

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

gemma-4 and Qwen3.8 run compiled (`VLLM_EAGER_FLAG=--no-enforce-eager`). On the
B60, compiled decoded gemma-4 at about 56 tok/s against eager's 52.0 (+7%),
with the same KV pool. It costs about 34 s of `torch.compile` on the first boot;
later boots load it from the compile cache.

The compose default is eager because of gpt-oss on the B60: the compile
buffers aren't covered by `--gpu-memory-utilization`, and with gpt-oss's
8.0 GiB KV value they left too little memory for 131,072 tokens and the boot
failed. The buffers grow with `max_num_batched_tokens` (2496), not with
context, so a longer context doesn't change this.

Compiled doesn't help prefill; only the attention backend does. vLLM 0.30.0's
new fused XPU kernels run only in eager mode, and compiled still beats them
(56.46 against 53.25 tok/s, measured on the B60).

**XPU graph isn't worth it.** On the B60 it gave +0.5% for 1.54 GiB of memory,
which at 131,072 no longer fits at all. On the B70 the GPU was already 99–100%
busy during decode, so there's nothing for graphs to remove. The boot-log line
*"XPU Graph is disabled by environment variable"* is expected.

### Why gemma-4 is slower than gpt-oss per pass

Not memory bandwidth: gemma-4 reads less per token (about 3.23 GB against
3.71 GB). It runs 240 small expert matrix multiplies per token (30 layers × 8
of 128 experts, each 704 wide) where gpt-oss runs 96 that are four times
wider, and small ones use the GPU poorly. The draft model gets around that by
checking several tokens per pass — see *Faster decode with a draft model*.

### Against the scaler and across vLLM versions

For gpt-oss-20b on the B60 the scaler is about 3% faster (85.6 against 83.1
tok/s) with a slightly larger pool, so there's no speed reason to move gpt-oss
here. This engine's case is that it is mainline vLLM, several versions newer,
and **the only engine here that runs gemma-4**. Going from 0.29.0 to 0.30.0
changed nothing measurable on gemma-4 (decode, prefill, concurrency and pool
all within 0.3%), and the same `.env` values carried over. Warm the engine up
before measuring after an upgrade: the first concurrent batch after a cache
wipe reads about 9% slow while Triton compiles new shapes.

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
   for gemma-4, and 10.5 GiB is used, keeping about 1.1 GiB free; with the
   draft model it's 9.5 GiB.

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
| Thinking never ends and the same sentence repeats, in an agent session | The known gemma-4 thinking loop. Stop the request; the next one normally works. See *Known issue: thinking loops in agent sessions* |
| OOM at boot | KV pin from the other model; `VLLM_KV_CACHE_MEMORY` is absolute and OOMs rather than shrinking |
| `400` on long prompts | Above the cap. Every block runs 131,072; on gemma-4 the ceiling is 162,496 and costs no VRAM |
| A big prompt seems to hang | Not hung — quadratic prefill on Triton. 9.6k ≈ 26 s, 32k ≈ 5 min, 64.7k ≈ 22 min (B60). Raise your client timeout, or switch on Intel attention if you can do without images |
| Boot fails: `mm_prefix … requires FlashAttention v4` | `VLLM_ATTN_BACKEND=FLASH_ATTN` without `VLLM_TEXT_ONLY_FLAG=--language-model-only`. Set both or neither |
| Boot fails with FLASH_ATTN refusing head size 512 | The plugin didn't load. Check the `head512_plugin` mount and `PYTHONPATH` in `compose.yaml`, and look for `xpu_head512` in the log. [PLUGIN.md](PLUGIN.md) covers how it loads |
| `400 At most 0 image(s) may be provided in one prompt` | `--language-model-only` is set. On gemma-4 that comes with Intel attention. Comment out both switch lines and `up -d` to get images back. On Qwen3.8 it's a leftover from the gemma-4 block: set `VLLM_TEXT_ONLY_FLAG=--no-language-model-only` |
| Much slower than *Intel attention for gemma-4* says, after an image upgrade | Look for `XPU kernel not compiled … falling back` in the log: a kernel variant is missing from the new image |
| Context ceiling dropped after a tuning change | You raised `--max-num-batched-tokens`; it inflates the sliding-window reservation |
| Boot segfaults in `getSortedImages` | `SYCL_CACHE_PERSISTENT=1` with the V2 runner. Must stay `0` |
| `metrics` is `null` or missing | Engine not recreated since the flag was added; the stream didn't send `stream_options.include_usage`; the request went through a gateway that drops it; `n` > 1 or a `/v1/completions` call with a list of prompts; or the endpoint is `/v1/responses` or `/v1/messages`. Only `mean_itl_ms` `null` just means one token came back |
| Same id listed twice | A `--served-model-name` value is repeated in `compose.yaml`; vLLM does not dedupe |
| Orphan-container warning | The other engine's container lingers; `down` that folder. Never `--remove-orphans` |

---

## Open questions

1. **Heavier concurrent load.** The only real test is 6 requests at a time,
   512 tokens each (323 tok/s in total, 386 with the draft model). Every other
   concurrency figure is allocated capacity, and `bench.sh` sends one request.
2. **A ~131k prompt on Triton** has never been run to the end; 64.7k is the
   largest. With Intel attention, 129,331 tokens took 51.7 s.
3. **fp8 KV cache, untested.** Triton accepts it, and the check that would
   reject it applies only to CUDA. It would halve the KV memory; the accuracy
   cost is unknown. It needs a `--kv-cache-dtype` setting in compose.
4. **The draft model on the B60.** Its 0.78 GiB would most likely come out of
   the 4.25 GiB KV setting, which means a lower context cap (inferred, not
   booted).
5. **Report the V2 segfault upstream.** It reproduces cleanly, and there's no
   existing issue.
6. **Report the head-512 check upstream.** The kernel is compiled in, and only
   `head512_plugin/` makes it reachable. A fix upstream would let the plugin go.
7. **Image input with Intel attention.** It needs bidirectional attention over
   image tokens, which vLLM's FLASH_ATTN path allows only on FA4. Nobody has
   checked whether Intel's kernel could do it. See
   [PLUGIN.md](PLUGIN.md#the-drawback-no-image-input).
8. **The stock `intel/vllm:0.21.0` baseline** in `vllm_xpu/`, still unmeasured.
9. **gemma-4's thinking loops.** Does the template fix from Hugging Face
   discussion #48 stop them on this checkpoint, and what does it cost? Do the
   loops happen on Triton attention too? See *Known issue: thinking loops in
   agent sessions*.

---

See also: the [top-level README](../README.md) for the stack overview, firewall
and volumes; [`scaler/README.md`](../scaler/README.md) for the alternative engine
and its measurement log; [`vllm_xpu/README.md`](../vllm_xpu/README.md) for the
stock Intel baseline.

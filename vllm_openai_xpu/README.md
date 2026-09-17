# Upstream vLLM XPU engine — operator guide (devops)

This folder runs **upstream's own** XPU image, `vllm/vllm-openai-xpu`, as opposed
to Intel's fork used by `scaler/`. It serves the same OpenAI-compatible API on
the same port, so it is a drop-in replacement for the other engines.

It is configured for **two validated models** and switching between them is a
`.env` edit plus a recreate. Everything below was measured on an Arc Pro B60
(24 GB), not estimated.

> **Single GPU.** Exactly one engine may run at a time. `docker compose down` in
> `vllm_xpu/` or `scaler/` before starting this one. All three share the Compose
> project name `llm` on purpose so they reuse the `hf-cache` volume — never pass
> `--remove-orphans`, it deletes the other engine's container.

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

vLLM bakes its CLI arguments into the container at creation. A plain `up -d`
sees an already-running container and leaves it alone, so **a `.env` edit
appears to do nothing**. Always recreate after changing configuration.

---

## Configuration — `.env`

`compose.yaml` reads every model-specific knob from a `${VAR:-default}`
placeholder, and Compose auto-reads `.env` from this folder. So the compose file
itself is never edited to change models. Copy the template and edit:

```bash
[ -f .env ] || cp .env.example .env    # never clobber an existing .env
```

`.env` is gitignored; `.env.example` is tracked and documents every variable
with its compose default and the measured consequence of changing it.

These are all **model-specific** — every one of them differs between the two
models, so they move as a set. `!` marks the three whose wrong value breaks a
boot or silently degrades it.

| Variable | What it sets |
|----------|--------------|
| `VLLM_MODEL` | Hugging Face repo ID |
| `VLLM_SERVED_MODEL_NAME` | The id clients call it by — see *Model naming* |
| `VLLM_REASONING_PARSER` | Model-family specific; wrong value = empty reasoning, **not** a crash |
| `VLLM_TOOL_CALL_PARSER` | Model-family specific, same quiet failure mode |
| `VLLM_MAX_MODEL_LEN` **!** | Context window; must fit VRAM after weights |
| `VLLM_KV_CACHE_MEMORY` **!** | KV pool in **absolute bytes**; overrides util, skips profiling, and OOMs rather than shrinking |
| `VLLM_EAGER_FLAG` **!** | `--enforce-eager` (required by gpt-oss's 8.6 GiB pin) or `--no-enforce-eager` (gemma-4, measured fine at 128k). Passed whole — `--enforce-eager=False` does not parse. See *Performance* |
| `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` | Server-side chat-template defaults, e.g. always-on reasoning |

Shell variables beat `.env`, but **only for the names you pass** — so the
temporary-override block under *Switching models* has to set all of them, not
just `VLLM_MODEL`.

> **The variables are not independent.** `VLLM_KV_CACHE_MEMORY` is sized for a
> specific model's weights, and it OOMs rather than shrinking. gemma-4's weights
> (16.93 GiB) plus gpt-oss's KV pin (8.0 GiB) exceed the 22.33 GiB free on the
> card and fail at boot. Switch a whole block at once.

---

## The two models

| | gpt-oss-20b | gemma-4-26B-A4B-it (int4) |
|---|---|---|
| weights on device | 12.87 GiB | 16.93 GiB |
| decode | 83.2 tok/s (**chunks** — understated, see note) | **56.1 tok/s** (true tokens) |
| TTFT, short prompt | ~76 ms | 93 ms reasoning on / **42 ms** off |
| KV pool | 338,928 tok | 78,433 tok |
| concurrency | 2.59× @128k | **2.39× @32k** |
| max context | 131,072 | **32,768 shipped** — 131,072 verified, ceiling 162,496 |
| reasoning | always on | opt-in (see below) |
| large cold prompts | linear, fine | **quadratic — see warning** |

**gemma-4's context was never VRAM-limited.** It boots, serves and passes
`smoke.sh` at 131,072 on the same 4.25 GiB pool, with decode and TTFT unchanged
(verified 2026-09-17). It ships at **65,536 as a deliberate guardrail**, not
because of a memory limit — see *gemma-4 context scaling* below for the formula
that predicts its pool to within 0.003%, and *why 65,536* for the reasoning.

### Why 32,768, and why the cap is not a speed setting

**Lowering the cap does not make the model faster.** Measured same-session, three
warm `bench.sh 400` runs at each setting:

| cap | decode | TTFT | KV pool reserved |
|---|---|---|---|
| 32k | 46.2 tok/s | 106 ms | 4.25 GiB |
| 64k | 46.3 tok/s | 106 ms | 4.25 GiB |
| 128k | 46.3 tok/s | 105 ms | 4.25 GiB |

The pool is pinned in absolute bytes, so it is identical at every cap, and a
given prompt costs the same wherever the cap sits. The concurrency figure is only
pool ÷ one-max-length-request — a ratio, not a capacity.

What the cap **does** control is the longest prefill a client can trigger:

| | 32k | 64k | 96k | 128k |
|---|---|---|---|---|
| largest prompt accepted | 32,768 | 65,536 | 98,304 | 131,072 |
| **worst-case prefill** | **~4.6 min** | ~22 min | ~51 min | ~92 min |
| concurrency | **2.39×** | 1.77× | 1.41× | 1.16× |

At 32,768 an oversized prompt fails fast with a `400` instead of occupying the
B60 for tens of minutes and then very likely timing out upstream anyway — and it
leaves the most pool for prefix cache. Raise the cap when you have a use for
prompts that size **and** timeouts along the whole path to match.

**gpt-oss-20b is still the better choice for coding-CLI traffic** through a
gateway, because its prefill stays linear. gemma-4 is the newer and stronger
model, and is vision-capable.

> ⚠ **`bench.sh` under-reports gemma-4 by ~15%, and the reason matters.** It
> counts SSE *chunks*, but with reasoning on the reasoning channel packs **1.15
> tokens per chunk**, so it reads ~48 chunk/s where the true rate is **56.1
> tok/s**. RESOLVED 2026-09-17 by streaming with `stream_options.include_usage`,
> which gives TTFT from the first chunk and true counts from the final usage
> chunk: 56.1 tok/s with thinking on, 55.8 with it off (so **reasoning costs no
> throughput**), and with thinking off chunks and tokens go exactly 1:1 —
> confirming the mechanism rather than merely correlating with it.
>
> **gpt-oss's 83.2 is also a chunk count** and is therefore understated too, by
> an unmeasured amount — it always reasons, so it can't be measured with
> reasoning off. Treat the cross-model gap as indicative, not exact.

### gemma-4 checkpoint constraint

The checkpoint must be **int4-symmetric with `group_size` 32 or channelwise**.
The XPU expert kernel (`XPUExpertsWNA16`) accepts only those two schemes, so the
smaller and far more popular **group-64** builds are rejected at load despite
being ~2 GiB lighter. Check `quantization_config.config_groups.*.weights.group_size`
before trying any other MoE checkpoint.

### gemma-4 context scaling — why 131,072 fits in a 4.25 GiB pool

gemma-4 is **25 sliding-attention layers (window 1024) + 5 full-attention
layers**, a 5:1 pattern, and its checkpoint declares
`max_position_embeddings: 262144`. In vLLM 0.29.0
`SlidingWindowSpec.max_memory_usage_bytes` bounds those 25 layers at
`min(sliding_window - 1 + max_in_flight_tokens, max_model_len)` — **independent
of `--max-model-len`**. Only the 5 full-attention layers scale with context, and
they are the cheap ones: 2 KV heads × 512, against the sliding layers' 8 × 256.

So the KV cost is a large fixed block plus a small linear term:

```
fixed  (25 sliding layers, window-bounded) = 1.235 GB      # context-independent
linear (5 full layers)                     = 20 KiB/token
```

Measured against that formula at the shipped 4.25 GiB pin — **raising
`--max-model-len` costs no VRAM at all**:

| `--max-model-len` | per request | concurrency | KV pool (predicted / **logged**) |
|---|---|---|---|
| 32,768 | 1.906 GB | 2.39× | 78,435 / **78,433** |
| 65,536 | 2.578 GB | 1.77× | 116,028 / **116,025** |
| **131,072** | 3.920 GB | **1.16×** | 152,596 / **152,592** |
| 162,496 | 4.563 GB | 1.00× | hard ceiling |

Weights (16.93 GiB), pool (4.25 GiB), decode and TTFT were byte-for-byte
identical across all three boots; only concurrency moves. A 64,708-token prompt
answered a question about its *last* record correctly, so the window is real and
not merely allocated.

> **⚠ Raising `--max-num-batched-tokens` LOWERS the context ceiling.** It feeds
> `max_in_flight_tokens`, which inflates the **sliding** reservation, not the
> full-attention one. At 8192 the fixed block balloons 1.235 → 3.568 GB and the
> ceiling collapses to 48,576. An earlier revision of this file reported that
> collapse as a property of the model ("hard ceiling ~48.5k") — it is an
> artifact of the flag. Leave it at the default.

### ⚠ gemma-4 prefill is quadratic, and it is not tunable

This, not VRAM, is the real limit on usable context — and the cause is a kernel
capability gate, not a missing XPU backend.

**XPU has its own flash-attention kernel and it is the default.**
`xpu_ops.flash_attn_varlen_func` is what gpt-oss uses, which is why *its* prefill
is fast. gemma-4 is locked out of it:

| step | source | result |
|---|---|---|
| XPU flash is FA2-class | `fa_utils.py`: `if is_xpu(): return 2` | version 2 |
| flash caps head_size at 256 without FA4 | `flash_attn.py supports_head_size()` | 512 needs FA4 |
| gemma-4's full-attention layers | `global_head_dim: 512` | 512 |
| FA4 on XPU | `import vllm.vllm_flash_attn` → *"requires the CUDA flash attention extensions"* | **unavailable** |

⇒ `supports_head_size(512)` is `False`, FLASH_ATTN is ineligible, and
`Gemma4Config` selects `TRITON_ATTN` — the only backend that JIT-compiles for
both 256 and 512. vLLM puts *all* layers on it deliberately: mixing backends
causes *"mixed backend selection and numerical divergence"*.

> **`VLLM_ATTENTION_BACKEND` is honoured, then refused** — not ignored, as an
> earlier revision of this file said. The override is
> `elif attention_config.backend is None`, so an explicit request survives
> config, and is then rejected on the head-size gate above. Don't spend time
> forcing it.

The tell that the *kernel* is at fault rather than the algorithm: prefill runs at
**49.9 tok/s** while decode does **56.1**. Prefill is *slower than decode* —
which should be impossible on a healthy path, because prefill batches 2,496
tokens per chunk and is compute-bound, while decode is memory-bound at one token
per step. The kernel is discarding essentially all of prefill's parallelism. At 127k the 5
full-attention layers account for **92.5%** of attention work, so a hypothetical
per-layer split (flash for the twenty-five 256-dim layers, Triton for the five
512-dim ones) would recover only ~7.5%. **The only real fix is a fast
head_dim-512 XPU kernel** — upstream work, nothing configurable here.

| cold prompt | wall |
|---|---|
| 2,421 tok | 2.2 s |
| 4,821 tok | 5.6 s |
| 9,621 tok | 19.4 s |
| 19,221 tok | 80.8 s |
| **64,708 tok** | **1297.6 s (21.6 min)** |

Least-squares fit over the three same-harness points below gives exponent
**2.048** — essentially plain quadratic — and reproduces all three to within
**±0.1%**:

```
T ≈ 26.3 s × (n / 9643)^2.048
```

| n | measured | fit |
|---|---|---|
| 9,643 | 26.3 s | 26.3 s (−0.1%) |
| 31,957 | 305.3 s | 305.7 s (+0.1%) |
| 64,708 | 1297.6 s | 1296.5 s (−0.1%) |

⇒ **~22 min at the 65,536 cap**, ~51 min at 96k, ~92 min at 128k. Prefill
throughput at 64.7k is only 49.9 tok/s — barely faster than *decode* (~46), which
is the tell that the forced Triton kernel is wasting the parallelism prefill
should enjoy.

> An earlier revision of this file claimed exponent **2.287** and ~109 min at
> 128k. That came from mixing one session's 64.7k measurement with a *different*
> session's 19,221 → 80.8 s datapoint taken under another config. Never fit a
> curve across sessions.

**Prefix caching is what makes this usable, and it improves with size.** Same
prompt sent twice on a virgin engine:

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
- **Growing a conversation costs the same total as one big prefill** — the work
  is the same triangle either way. What changes is that it's spread out, so
  per-turn latency creeps up: a ~2k-token turn costs ~40 s at 32k of context and
  ~2.2 min at 100k.

---

## Switching models

### Temporary — shell overrides

Shell variables beat `.env`, so this reverts by itself and edits nothing. To run
gpt-oss-20b while `.env` is configured for gemma-4, **paste the whole block** —
a shell variable only wins for the variables you actually name, so every value
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

Three of those are **not optional**, and each fails differently:

| variable | if you omit it while `.env` holds gemma's value |
|---|---|
| `VLLM_KV_CACHE_MEMORY` | boots, but the 4.25 GiB pin strands ~4 GiB and roughly halves the pool |
| `VLLM_EAGER_FLAG` | **boot fails** — compiled mode with gpt-oss's 8.6 GiB pin starves the KV pool, which is the case `--enforce-eager` exists for |
| `VLLM_MAX_MODEL_LEN` | **silently caps gemma-4 at 65,536 instead of 131,072** — the blocks differ again, so this is load-bearing |

`VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=null` is tidiness only — an unknown template
kwarg is verified harmless on gpt-oss, which always reasons regardless.

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

Expected in the container log:

| model | log lines |
|---|---|
| gpt-oss-20b | `Model loading took 12.87 GiB`, `GPU KV cache size: 338,928 tokens … 2.59x` |
| gemma-4 | `Model loading took 16.93 GiB`, `GPU KV cache size: 78,433 tokens … 2.39x` |

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
streaming), never `reasoning_content`.

- **gpt-oss-20b** — always reasons, no off switch.
- **gemma-4** — **opt-in.** Its chat template defaults `enable_thinking` to
  false, so without action the reasoning field comes back empty. That is the
  flag being unset, not a broken parser.

Turn it on for every request by setting this **in `.env`** (not a shell command),
then recreating:

```dotenv
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}
```

A request can still override that **both ways**, so it is a default and not a
lock:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

Reasoning does **not** cost decode speed — measured by `usage.completion_tokens`
over wall time, thinking on and off are both ~52 tok/s. (`bench.sh` reports a
lower rate with reasoning on because it counts SSE *chunks*, and the reasoning
channel does not stream 1 chunk per token. Trust `completion_tokens` for a true
rate.)

The real cost is budget: reasoning consumes `max_tokens` **first**, so at
`max_tokens: 400` the whole budget goes to the trace and you get no answer.
Allow 1500+.

> `null` is the compose default for this variable, not `{}` — a brace inside a
> `${VAR:-default}` breaks Compose interpolation. `json.loads("null")` is `None`,
> which vLLM treats as no defaults.

---

## Model naming

**One truthful name per model** — `gpt-oss-20b` and `gemma-4-26b-a4b`. The id is
set by `VLLM_SERVED_MODEL_NAME` and is not validated against the checkpoint, so
it *could* be used as a fixed label to hide a swap. Don't: the two models now
differ in context cap (131,072 vs 65,536) *and* cost far more than that ratio
suggests, so a consumer that keeps sending gpt-oss-sized prompts to gemma-4 gets
hard `400`s that look like a gateway fault rather than a deliberate model
change. Repoint the consumer's context cap as well as its model id.

A swap therefore means updating the consumer's model mapping. That is the point
— it makes the change visible where the limits actually differ.

Confirm the name matches reality with `root`, which reports the loaded repo:

```bash
curl -s localhost:8000/v1/models \
  | python3 -c 'import sys,json;[print(m["id"],"->",m["root"]) for m in json.load(sys.stdin)["data"]]'
```

`smoke.sh` prints the same mapping on every run, and warns if `/v1/models` ever
returns a duplicate id.

---

## `bench.sh` and `smoke.sh`

Both live at the repo root and work against any engine. Two conveniences matter
here:

- **`MODEL` is auto-detected** from `/v1/models` when unset, so neither script
  needs editing across a model swap. The output labels which path was used
  (`auto-detected` vs `explicit`).
- **`THINKING` is tri-state**, which is what makes the deployed reasoning
  default observable:

| `THINKING` | sends | asserts / reports |
|---|---|---|
| `auto` (default) | nothing | whatever the **server** default does |
| `1` | `enable_thinking: true` | a reasoning trace comes back |
| `0` | `enable_thinking: false` | reasoning is **suppressed** (`smoke.sh` inverts the check) |

```bash
./smoke.sh                 # ALL PASS on both models as shipped
THINKING=0 ./smoke.sh      # proves a request can override the server default
./bench.sh 1500            # reasoning needs the larger budget
VLLM_ENDPOINT=http://192.168.x.x:8000 ./smoke.sh   # remote target

# MODEL is only needed to pin an id explicitly; it must match what is loaded,
# or you get a 404 that names the served id back to you.
MODEL=gemma-4-26b-a4b ./bench.sh 400
```

---

## Downstream consumers

The endpoint is OpenAI-compatible and **unauthenticated** — see *Firewall* in the
[top-level README](../README.md). Any AI gateway works (LiteLLM and Bifrost are
just examples); point it at `http://<host>:8000/v1` using a served model name.

**A model swap changes the served name, so the consumer 404s until its model
mapping is updated.** That is deliberate — the two models charge very different
different context caps (131,072 vs 65,536) and very different prefill costs, and
a fixed label would hide both until one surfaced as a `400` or a stall. Update
both together.

---

## Engine knobs

Rarely changed; all have safe compose defaults.

| Variable | Default | Note |
|---|---|---|
| `SYCL_CACHE_PERSISTENT` | `0` | **Must stay 0** while the V2 runner is on — `1` segfaults at boot inside the SYCL persistent code-cache read. Costs a kernel rebuild each boot (~70 s). |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Upstream default from 0.29.0. Integer only — a blank value crashes at boot. |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | `0` | Only does anything in **compiled** mode — it is a genuine no-op under `--enforce-eager`. See *Performance*. |
| `HF_TOKEN` | empty | Anonymous. Neither model above is gated. |

---

## Performance

### Eager vs compiled — take compiled on gemma-4

Upstream runs `torch.compile` even with cudagraphs off, and those buffers are
**not** capped by `--gpu-memory-utilization`. With **gpt-oss's 8.6 GiB pin** they
starved the KV pool and killed the boot, which is why `--enforce-eager` is the
compose default. gemma-4's 4.25 GiB pin leaves headroom, and compiled mode is
measured free — at 32,768, 65,536 **and** 131,072, all boot-tested on
2026-09-17 with the full pool intact. The compile buffers scale with
`max_num_batched_tokens` (2496), *not* with context, so context length does not
change this trade-off:

| config | decode (true tokens) | KV pool | concurrency |
|---|---|---|---|
| eager | 52.0 tok/s | 78,433 | 2.39× |
| **compiled** | **55.7 tok/s** (+7.1%) | 78,433 | 2.39× |
| compiled + XPU graph | 56.0 tok/s | 46,138 | 1.41× |

> ✅ **This table's 55.7 is confirmed.** Re-measured 2026-09-17 with
> `include_usage`: **56.1 tok/s** true. The ~46 figure seen that day was
> `bench.sh`'s chunk count, not a regression. Context length does not affect
> decode (46.2 / 46.3 / 46.3 chunk/s at 32k / 64k / 128k — identical).

```dotenv
VLLM_EAGER_FLAG=--no-enforce-eager
```

Compiled costs nothing but ~34 s of `torch.compile` on every boot (the SYCL
cache must stay off, so it never persists). **XPU graph is not worth it**:
+0.5% more for 1.54 GiB of capture memory, which drops the pool to 46,138 tokens
and concurrency to 1.41×. It does, however, genuinely capture graphs in compiled
mode — the "no-op" result was an artifact of only ever testing it under eager.

**Compiled mode does not help prefill** (2.28 / 5.56 / 19.77 s at 2.4k / 4.8k /
9.6k tokens, vs eager's 2.15 / 5.61 / 19.38). That confirms the quadratic
prefill is the forced Triton attention backend, not kernel-launch overhead, and
so is not fixable from configuration.

### Why gemma-4 is slower than gpt-oss

Not bandwidth — gemma reads *less* per token (~3.35 GB vs ~3.71 GB, counting
int4/MXFP4 weights plus the unquantized embedding). It is op count and
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
of that; the rest is architectural.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `.env` edit had no effect | Missing `--force-recreate`; the container kept its baked-in args |
| `HTTP 404 ... model does not exist` | Stale `MODEL` in your shell, or the gateway points at the other block's name. `bench.sh` prints what *is* served |
| Reasoning field always empty | gemma-4 without `enable_thinking` — set it per request or via `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` |
| Trace returned but no answer | Reasoning consumed the whole `max_tokens`; raise it to 1500+ |
| OOM at boot | KV pin from the other model; `VLLM_KV_CACHE_MEMORY` is absolute and OOMs rather than shrinking |
| `400` on long prompts | Above the block's cap — 131,072 for gpt-oss, **32,768 for gemma-4 as shipped**. Raising gemma's cap is safe up to `162,496` and costs no VRAM; see *Why 32,768* first |
| A big prompt seems to hang | Not hung — quadratic prefill on gemma-4. 9.6k ≈ 26 s, 32k ≈ 5 min, **64.7k ≈ 22 min** (the shipped cap's worst case). Raise your client timeout |
| Context ceiling dropped after a tuning change | You raised `--max-num-batched-tokens`; it inflates the sliding-window reservation. See *gemma-4 context scaling* |
| Same id listed twice | A `--served-model-name` value is repeated in `compose.yaml`; vLLM does not dedupe |
| Orphan-container warning | The other engine's container lingers; `down` that folder. Never `--remove-orphans` |

See also: [top-level README](../README.md) for the stack overview, firewall and
volumes; [UPSTREAM_VLLM_NOTES.md](../UPSTREAM_VLLM_NOTES.md) for why this engine
is configured the way it is.

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
| `VLLM_EAGER_FLAG` **!** | `--enforce-eager` (required at 128k) or `--no-enforce-eager` (+7.1% at 32k). Passed whole — `--enforce-eager=False` does not parse. See *Performance* |
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
| decode | **83.2 tok/s** (chunks) | **55.7 tok/s** compiled / 52.0 eager (true tokens) |
| TTFT, short prompt | ~76 ms | ~56 ms |
| KV pool | 338,928 tok | 78,433 tok |
| concurrency | 2.59× @128k | 2.39× @32k |
| max context | 131,072 | 32,768 (hard ceiling ~48.5k) |
| reasoning | always on | opt-in (see below) |
| large cold prompts | linear, fine | **quadratic — see warning** |

**gpt-oss-20b is the better choice for coding-CLI traffic** through a gateway,
because its prefill stays linear at large context and it has 4× the context.
gemma-4 is the newer and stronger model, and is vision-capable.

### gemma-4 checkpoint constraint

The checkpoint must be **int4-symmetric with `group_size` 32 or channelwise**.
The XPU expert kernel (`XPUExpertsWNA16`) accepts only those two schemes, so the
smaller and far more popular **group-64** builds are rejected at load despite
being ~2 GiB lighter. Check `quantization_config.config_groups.*.weights.group_size`
before trying any other MoE checkpoint.

### ⚠ gemma-4 prefill is quadratic, and it is not tunable

gemma-4 has heterogeneous head dims (`sliding_attention` 256 /
`full_attention` 512). XPU has no FA4 kernel, so vLLM force-selects
`TRITON_ATTN` at config level — `VLLM_ATTENTION_BACKEND` is **ignored**, and
raising `--max-num-batched-tokens` buys only ~15% while inflating the KV
requirement enough to break a 64k boot.

| cold prompt | wall |
|---|---|
| 2,421 tok | 2.2 s |
| 4,821 tok | 5.6 s |
| 9,621 tok | 19.4 s |
| 19,221 tok | **80.8 s** |

2× tokens ⇒ 4× time. **Prefix caching hides this** for repeated or growing
prompts: the same 9.6k prompt went 26.9 s cold → **0.38 s** warm (71×). So
conversation and growing context are fine; a large **cold** context (fresh RAG
document, big paste) is where it hurts.

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
| `VLLM_EAGER_FLAG` | **boot fails** — compiled mode at 128k starves the KV pool, which is the case `--enforce-eager` exists for |
| `VLLM_MAX_MODEL_LEN` | silently caps gpt-oss at 32,768 instead of 131,072 |

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
it *could* be used as a fixed label to hide a swap. Don't: the two models differ
in context limit (131,072 vs 32,768) and prefill cost, so a consumer that keeps
sending gpt-oss-sized prompts to gemma-4 gets hard `400`s that look like a
gateway fault rather than a deliberate model change.

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
mapping is updated.** That is deliberate — the two models have different context
limits, and a fixed label would hide that mismatch until it surfaced as a `400`
on a long prompt. Update both together.

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

### Eager vs compiled — take compiled at 32k

Upstream runs `torch.compile` even with cudagraphs off, and those buffers are
**not** capped by `--gpu-memory-utilization`. At **128k** (gpt-oss) they starved
the KV pool and killed the boot, which is why `--enforce-eager` is the default.
At **32k** (gemma-4) there is headroom, and compiled mode is measured free:

| config | decode (true tokens) | KV pool | concurrency |
|---|---|---|---|
| eager | 52.0 tok/s | 78,433 | 2.39× |
| **compiled** | **55.7 tok/s** (+7.1%) | 78,433 | 2.39× |
| compiled + XPU graph | 56.0 tok/s | 46,138 | 1.41× |

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
| `400` on long prompts | gemma-4 caps at 32,768, not gpt-oss's 131,072 |
| A big prompt seems to hang | Not hung — quadratic prefill. 19.2k tok ≈ 81 s |
| Same id listed twice | A `--served-model-name` value is repeated in `compose.yaml`; vLLM does not dedupe |
| Orphan-container warning | The other engine's container lingers; `down` that folder. Never `--remove-orphans` |

See also: [top-level README](../README.md) for the stack overview, firewall and
volumes; [UPSTREAM_VLLM_NOTES.md](../UPSTREAM_VLLM_NOTES.md) for why this engine
is configured the way it is.

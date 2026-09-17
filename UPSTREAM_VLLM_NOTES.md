# Upstream vLLM XPU engine — why it is configured this way

Companion to [`vllm_openai_xpu/compose.yaml`](vllm_openai_xpu/compose.yaml), the
way [`SCALER_NOTES.md`](SCALER_NOTES.md) is to `scaler/compose.yaml`.

**Division of labour:** [`vllm_openai_xpu/README.md`](vllm_openai_xpu/README.md)
is the operator guide — how to run it, switch models, read the logs. **This file
explains the non-obvious settings and what was measured to justify them**, so the
compose file can stay short. If you only want to run the thing, you don't need
this file.

Everything here was measured on an Arc Pro B60 (24 GB), not estimated.

---

## Current state

| | |
|---|---|
| Image | `vllm/vllm-openai-xpu:v0.29.0` |
| Model runner | V2 (upstream default from 0.29.0) |
| Model served | gemma-4-26B-A4B-it, offline int4 group-32 |
| Context | 32,768 (131,072 verified; see §7) |
| KV pool | 78,433 tok = 2.39× @32k |
| `smoke.sh` | ALL PASS |

Measured for **gpt-oss-20b** on this same engine, kept because it is the
cross-engine comparison point: 83.5 tok/s @200, 83.1 @400, TTFT ~76 ms, KV pool
338,928 tok = 2.59× @128k.

---

## 1. Why this engine exists

The old assumption was that a B60 needs Intel's fork, because upstream vLLM
didn't cover Arc Pro B-series or MXFP4 gpt-oss. Both halves are false: upstream's
own XPU docs list **Intel® Arc™ Pro B-Series** as validated hardware, and
`platforms/xpu.py` `supported_quantization` includes `mxfp4` and
`gpt_oss_mxfp4`. MXFP4 was confirmed native here — weights load at 12.87 GiB,
where bf16 would be ~42 GB.

⚠️ **Being listed in upstream's model table is an architecture claim, not a fit
claim.** `gpt-oss-120b` is listed too, and its 60.7 GiB of weights fit neither
one B60 nor 2×B70.

**Image pin.** Upstream's XPU builds live in a **separate Docker Hub repo** —
`vllm/vllm-openai-xpu`, not `vllm/vllm-openai`. Checking the CUDA repo shows zero
XPU tags and wrongly suggests no image exists. Unlike Intel's images, `latest`
here **does** track newest stable. The pin is still explicit, so an upgrade is a
deliberate commit. Upstream is ahead of both other engines in this repo
(`intel/vllm` 0.21.0, `intel/llm-scaler-vllm` 0.26.0), and its image is roughly
half the size of the scaler's.

---

## 2. `--enforce-eager` — required by the *pin*, not by the context size

Upstream defaults `cudagraph_mode: NONE` but **still runs torch.compile /
Inductor** (`CompilationMode.VLLM_COMPILE`). Those two settings are independent,
and assuming "graphs off ⇒ already eager" cost a failed boot.

Inductor's kernel and workspace buffers are **not capped by
`--gpu-memory-utilization`**. With gpt-oss's 8.6 GiB KV pin they consumed the
space the pool needed:

```
Available KV cache memory: 2.89 GiB
ValueError: max seq len (131072) needs 3.1 GiB KV cache > available 2.89 GiB
```

Missing 128k by 0.21 GiB. So **eager is mandatory for gpt-oss here** — for a
different reason than on the scaler, where it prevents silently-empty content.

**gemma-4 runs compiled.** Its smaller 4.25 GiB pin leaves headroom, verified at
32k, 64k *and* 131,072 with the pool intact. The compile buffers scale with
`--max-num-batched-tokens` (2496), **not** with context, so context length does
not change this trade-off.

Rejected alternatives: raising `--gpu-memory-utilization` (the documented
anti-pattern — util doesn't cap compile buffers, and 0.86 OOMs), and
`--max-model-len 122048` (hides the cause and breaks 128k parity).

The flag is passed **whole** — argparse builds it as a `BooleanOptionalAction`,
so `--enforce-eager=False` does not parse; it must be `--no-enforce-eager`.

---

## 3. `SYCL_CACHE_PERSISTENT=0` — mandatory with the V2 runner

The first V2 boot died after a clean memory profile, with no Python traceback:

```
!!!!!!! Segfault encountered !!!!!!!
  at::native::xpu::topk_kernel → ... → sycl::handler::finalize()
  → PersistentDeviceCodeCache::getItemFromDisc → getSortedImages   ← segfault
```

Traced to source: `gpu_worker.py` runs `warmup_kernels(...)` **for V2 only**.
That warmup builds its batch with `SamplingParams.for_sampler_warmup()`, which
hardcodes `logprobs=5, prompt_logprobs=1` to exercise all sampler logic —
logprobs reach `torch.topk(...)`, whose XPU kernel segfaults while building the
SYCL program from the **persistent** device-code cache.

Notes for whoever meets this next:

- The crashing feature is **logprobs, which this deployment never requests**. V2
  died proving a path normal serving never reaches.
- Not memory: `OOMKilled=false`, the profile printed, the pool allocated.
- **No config knob avoids it.** `for_sampler_warmup()` hardcodes its params,
  `--max-logprobs` never reaches it, and there is no env var to skip
  `warmup_kernels` (`KernelConfig.enable_jit_warmup` governs a *different*
  warmup, which succeeds).
- Cost of the fix is a device-kernel rebuild each boot; total boot still ~70 s.
- **Unreported upstream** as of 2026-09-09 — worth filing.

`SYCL_CACHE_PERSISTENT` is set by this repo, not a vLLM default.

---

## 4. The V2 model runner

At 0.28.0 the V2 gate required `arch ∈ DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES or
not is_moe`; `GptOssForCausalLM` was in neither set, so **0.28.0 silently ran
V1**. 0.29.0 removed that gate, so V2 is now the default here.

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

`Initializing a V1 LLM engine` refers to the **engine**, a different axis — do
not read it as the runner. `VLLM_USE_V2_MODEL_RUNNER` overrides, and is
**integer only**: a blank value crashes on `bool(int(""))`.

---

## 5. `--kv-cache-memory-bytes` — the big capacity lever

`--gpu-memory-utilization 0.80` caps the engine at 18.17 GiB of a 22.71 GiB card,
stranding ~4.5 GiB. The engine prints the exact byte value to reclaim it:

```
Replace gpu_memory_utilization config with
  `--kv-cache-memory=8603448832` (8.01 GiB) to fully utilize gpu memory
```

Taking it moved gpt-oss's pool 3.12 → 8.01 GiB, **169,123 → 338,928 tokens
(1.29× → 2.59× @128k)** at unchanged decode and TTFT.

Four things to know before touching it:

1. **The value is absolute.** It overrides util and skips memory profiling, and
   it **OOMs rather than shrinking** if the model doesn't fit.
2. **It is runner-specific.** The byte value comes from that runner's profile;
   carrying V1's 6.63 GiB onto V2 leaves ~1.4 GiB unclaimed. Re-derive after any
   runner or image change by commenting the flag out for one boot and reading the
   advised value back.
3. **It is model-specific.** gemma-4's weights (16.93 GiB) plus gpt-oss's 8.0 GiB
   pin exceed the 22.33 GiB free and fail at boot. Switch a whole `.env` block.
4. **Canonical spelling is `--kv-cache-memory-bytes`.** The log advises
   `--kv-cache-memory`, which only works as an argparse prefix abbreviation.

The pinned pool leaves ~0.15 GiB of card headroom — defensible only because
`--enforce-eager` means no growing compile buffers.

---

## 6. Device passthrough and graph mode

**Both the whole `/dev/dri` device and a `/dev/dri/by-path:ro` bind are
required**, at any world size, because oneCCL enumerates devices via `by-path`.
This is upstream's documented recipe, not a local workaround — it matches the
`ze_fd_manager.cpp:144 … opendir failed` crash reverse-engineered on the scaler
([`SCALER_NOTES.md`](SCALER_NOTES.md) §5). 0.29.0 skips the oneCCL warm-up at
world_size=1, so the bind may now be redundant; keep it until a boot without it
proves otherwise.

Upstream's example uses `--privileged` and `--network=host`. This repo uses
`group_add` (render 992 / video 44) plus an explicit port map instead —
**measured sufficient**, no `ze_fd_manager` failure. Less privilege for no cost.

Three things the platform sets automatically: `UCX_MEMTYPE_CACHE=n`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `shutdown_timeout=5`. That last one
matters — XPU needs a graceful shutdown to release oneCCL/Level Zero resources,
or *"subsequent server startups on the same devices may hang during CCL
initialization."*

**Graph mode is single-GPU-only upstream** ("XPU Graph support is experimental
and currently only supports single-GPU execution"), so the single-B60 layout is
the only one that could use it — going dual would forfeit it. It defaults off,
and it is a genuine **no-op under eager**. In compiled mode it does capture, but
costs 1.54 GiB for +0.5%, dropping the pool 78,433 → 46,138 tokens. **Not worth
it.**

---

## 7. gemma-4: context is nearly free, prefill is not

Full operator detail is in
[`vllm_openai_xpu/README.md`](vllm_openai_xpu/README.md); the mechanism is here.

gemma-4 is **25 `sliding_attention` layers (window 1024) + 5 `full_attention`
layers** (5:1), declaring `max_position_embeddings: 262144`. In vLLM 0.29.0,
`SlidingWindowSpec.max_memory_usage_bytes` bounds the sliding layers at
`min(sliding_window - 1 + max_in_flight_tokens, max_model_len)` — **independent
of `max_model_len`** — while `FullAttentionSpec` scales with it. So:

```
fixed  (25 sliding layers, window-bounded) = 1.235 GB    # context-independent
linear (5 full layers)                     = 20 KiB/token
```

Predictions vs the engine's own log, at the 4.25 GiB pin — **agreement 0.003%**:

| `--max-model-len` | concurrency | predicted / **logged** tokens |
|---|---|---|
| 32,768 | 2.39× | 78,435 / **78,433** |
| 65,536 | 1.77× | 116,028 / **116,025** |
| 131,072 | 1.16× | 152,596 / **152,592** |
| 162,496 | 1.00× | hard ceiling |

The reported pool is not a flat token count: `kv_cache_utils.py` computes
`num_tokens = int(max_concurrency * max_model_len)`, which is why it's an odd
number.

**Two traps.** Raising `--max-num-batched-tokens` **lowers** the context ceiling
— it feeds `max_in_flight_tokens`, inflating the *sliding* reservation; at 8192
the ceiling collapses to 48,576. And per-token KV is meaningless for a hybrid
model: 4.25 GiB / 78,433 tok = 56.8 KiB/token is an artifact of 32k, while the
*marginal* cost is 20 KiB/token. When sizing any hybrid model read
`layer_types`, `sliding_window`, `num_key_value_heads` **and**
`num_global_key_value_heads` — the head *counts* differ per layer type, not just
the dims.

**Why it ships at 32,768 anyway.** Prefill is quadratic and the pool is 4.25 GiB
at *any* cap, so changing the cap reserves nothing and — **measured** — changes
neither decode nor TTFT (46.2 / 46.3 / 46.3 **chunk/s** and 105–106 ms at
32k / 64k / 128k — identical). The cap is a guardrail, not a speed setting.
*(Those are `bench.sh` chunk counts. The true decode rate is **56.1 tok/s**; see
§11 item 2. The comparison across caps is unaffected — same instrument, same
bias.)* What it controls is the
longest prefill a client can trigger. Least-squares over three same-harness
points, fitting to **±0.1%**:

```
T ≈ 26.3 s × (n / 9643)^2.048
   9,643 → 26.3 s   |   31,957 → 305.3 s   |   64,708 → 1297.6 s
```

⇒ **~4.6 min at the shipped 32,768**, ~22 min at 64k, ~51 min at 96k, ~92 min at
128k. At 32,768 an oversized prompt fails fast with a `400` instead of occupying
the GPU for tens of minutes and then timing out upstream anyway. 32k also leaves
the most pool for prefix cache (2.39× vs 1.16× at 128k).

**Prefill is slower than decode, which is the diagnostic.** Prefill measured
**49.9 tok/s** at 64.7k against decode's **56.1**. That should be impossible on a
healthy path: prefill batches 2,496 tokens per chunk and is compute-bound, while
decode is memory-bound at one token per step. The kernel is discarding
essentially all of prefill's parallelism.

**The cause is a kernel capability gate, not a missing XPU backend.** XPU *has* a
flash kernel (`xpu_ops.flash_attn_varlen_func`, the default, used by gpt-oss),
but it reports as FA2 and flash only handles `head_size > 256` when FA4 is
present. gemma-4's full-attention layers are **512**, and FA4 ships only as a
CUDA extension — so `supports_head_size(512)` is `False` and `Gemma4Config`
selects `TRITON_ATTN`, the only backend that JIT-compiles for both 256 and 512.
vLLM puts *all* layers on it deliberately: mixing causes *"mixed backend
selection and numerical divergence"*.

`VLLM_ATTENTION_BACKEND` is **honoured then refused** on that head-size gate —
not ignored. Don't spend time forcing it. And a hypothetical per-layer split
wouldn't help: at 127k the five 512-dim layers do **92.5%** of the attention work.

**Prefix caching is what makes long context usable, and it improves with size:**
9,643 tok 26.3 s → 0.31 s (**84×**); 31,957 tok 305.3 s → 0.63 s (**481×**).
Warm time stays ~flat while cold grows quadratically. Paid once per document per
engine lifetime — the cache is VRAM-resident and does not survive a recreate.

---

## 8. Shared project and volumes — two hazards

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
  weights with it. Plain project volumes are removable by any project member.
  Use bare `down`.

**Swap procedure** — one GPU, so exactly one engine runs at a time:

```bash
docker compose -f vllm_xpu/compose.yaml down     # if the stock engine is up
docker compose -f scaler/compose.yaml down       # if the scaler is up
docker compose -f vllm_openai_xpu/compose.yaml up -d
```

Stop the `~/llama_cpp` stack first if it is running — it shares the B60 and
:8000 and is `restart: unless-stopped`.

---

## 9. What upstream does not provide

Intel's fork keeps some B60-specific surface upstream lacks: the online int4 path
and `VLLM_QUANTIZE_Q40_LIB`, extra arch registrations, and Battlematrix
multi-GPU tuning.

Requirements: **Python 3.12 exactly** (the `vllm-xpu-kernels` wheels are
3.12-only and upstream flags this as a MUST), `torch==2.13.0` XPU,
`triton==3.7.2+xpu` — all satisfied inside the image. The host needs only the
in-kernel `xe` driver. The image also bundles `xpu-smi` 2.1.0, which the scaler
image does not.

---

## 10. Standings against the scaler

Both engines now have the `--kv-cache-memory-bytes` lever, measured on the same
box with the same scripts (gpt-oss-20b, 128k):

| | upstream 0.29.0 + V2 | scaler 0.26.0-b2 | winner |
|---|---|---|---|
| tok/s @400 | 83.1 | **85.6** | scaler, +3.0% |
| tok/s @200 | 83.5 | **86.1** | scaler, +3.1% |
| KV pool | 338,928 tok | **340,663 tok** | scaler, +0.5% |
| Concurrency @128k | 2.59× | **2.60×** | scaler |

**The scaler leads on both axes**, so there is no capacity argument for
migrating gpt-oss. Upstream's advantages are non-performance: several vLLM minors
newer, half the image size, mainline rather than an undocumented beta, a
trustworthy `latest`, and `xpu-smi` in the image — plus it is **the only engine
here that runs gemma-4**, which is why it is the one currently serving.

---

## 11. Open

1. A genuine **concurrent-load** test — every concurrency figure in this file is
   *allocated* capacity; `bench.sh` is single-stream.
2. ~~gemma-4's decode figure is unresolved~~ — **RESOLVED 2026-09-17.** True
   rate is **56.1 tok/s**, confirming the earlier 55.7. `bench.sh` counts SSE
   chunks and the reasoning channel packs 1.15 tokens/chunk, so it reads ~48.
   Measured with `stream_options.include_usage`. Reasoning costs no throughput
   (56.1 on / 55.8 off). ⚠ gpt-oss's 83.2 is also a chunk count and likewise
   understated — it always reasons, so it can't be re-measured with it off.
3. A true **~127k prompt** has never completed end-to-end; 131,072 is
   boot-verified and exercised to ~64.7k.
4. **fp8 KV cache, untested.** `TritonAttentionBackend.supported_kv_cache_dtypes`
   includes `fp8`, and the SM89 guard that would reject it sits inside
   `if current_platform.is_cuda():` — **not gated on XPU**. It would halve both
   KV terms (128k at ~2.3×, or the full 262,144 in budget), does nothing for
   prefill, and its accuracy cost is unmeasured. Needs a `--kv-cache-dtype`
   passthrough in the compose file.
5. **File the V2 segfault upstream** (§3) — clean repro, no existing issue.
6. Whether `/dev/dri/by-path` is still needed at world_size=1 (§6).
7. The stock `intel/vllm:0.21.0` baseline, still unmeasured.

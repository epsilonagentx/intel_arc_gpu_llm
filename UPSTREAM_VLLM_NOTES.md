# Upstream vLLM XPU engine — notes

Companion to [`vllm_openai_xpu/compose.yaml`](vllm_openai_xpu/compose.yaml), the
way [`SCALER_NOTES.md`](SCALER_NOTES.md) is to `scaler/compose.yaml`.

**Status: BOOTED AND MEASURED on the B60.** Current pin `v0.29.0` with the V2
model runner — 83.5 tok/s, a 338,928-token KV pool (2.59x @128k), `smoke.sh` ALL
PASS, measured 2026-09-09 (§10). Sections 1–9 were written against `v0.28.0`;
where 0.29.0 changed an answer, §10 says so and wins.

---

## 1. Why this engine exists

Until now the assumption was that a B60 needs Intel's fork, because upstream
vLLM did not cover Arc Pro B-series or MXFP4 gpt-oss. **Both halves of that are
now false.**

`docs/models/hardware_supported_models/xpu.md` at `v0.28.0`:

> **Validated Hardware:** Intel® Arc™ Pro B-Series Graphics

…and in the same file `openai/gpt-oss-20b` is marked ✅ under **MXFP4**, where ✅
means "Runs and optimized". Corroborated in code: `vllm/platforms/xpu.py`
`supported_quantization` lists both `mxfp4` and `gpt_oss_mxfp4`.

⚠️ **Do not over-read that table.** `gpt-oss-120b` is also ✅ MXFP4, and its
60.7 GiB of weights fit neither one B60 nor 2×B70. The ✅ certifies architecture
and quant support, **not** that a model fits this card.

## 2. Image pin provenance

Upstream's XPU builds live in a **separate Docker Hub repo** from the CUDA ones:
`vllm/vllm-openai-xpu`, not `vllm/vllm-openai`. Checking the CUDA repo shows
zero XPU tags and wrongly suggests no image exists.

| Tag | Pushed | Compressed |
|---|---|---|
| `v0.29.0` (current pin) | 2026-09-09 | 4.17 GB |
| `latest` | 2026-09-09 | → v0.29.0 |
| `nightly` | daily | 4.16 GB |
| `v0.28.0` | 2026-08-26 | 4.14 GB |
| `v0.27.1` / `v0.27.0` | Aug 11 / Aug 10 | 5.00 GB |
| `v0.26.0` (oldest stable XPU) | 2026-07-25 | 9.44 GB |

Two contrasts with the Intel images: `latest` here **does** track newest stable
(see [`SCALER_NOTES.md`](SCALER_NOTES.md) §2 for why it does not there), and
4.17 GB is roughly **half** the 7.78 GB of the `intel/llm-scaler-vllm` 0.26.0
images. Image size fell every release up to 0.28.0 and has been flat since.

Version position — upstream is ahead of both engines in this repo:

| Engine | vLLM base | Image last moved |
|---|---|---|
| `intel/vllm:0.21.0-ubuntu24.04` | 0.21.0 | 2026-08-06 |
| `intel/llm-scaler-vllm:0.26.0-b2` | 0.26.0 | 2026-09-08 |
| `vllm/vllm-openai-xpu:v0.29.0` | **0.29.0** | 2026-09-09 |

The XPU image lands the same day as the GitHub release (2026-09-09, ~3 h before
the release notes were published), so this repo is one `docker pull` behind
mainline rather than a fork's release cycle.

## 3. Graph mode — the one lever, and it is single-GPU-only

From `check_and_update_config` in `vllm/platforms/xpu.py`:

> "XPU Graph support is experimental and currently only supports **single-GPU
> execution**."

So the single-B60 layout is the *only* one where this is offered — going dual
would forfeit it. The gate chain:

```
supports_xpu_graph() false        -> cudagraph_mode = NONE
VLLM_XPU_ENABLE_XPU_GRAPH unset/0 -> cudagraph_mode = NONE   <- default
else                              -> experimental, single-GPU only
```

`VLLM_XPU_ENABLE_XPU_GRAPH` defaults to `False` (`vllm/envs.py:310`).

The 2026-09-03 finding that `VLLM_XPU_ENABLE_XPU_GRAPH=1` was a no-op was
measured **on the scaler, under forced eager**. Upstream's
`support_static_graph_mode()` returns `True` and graphs are supported
single-GPU, so the flag may actually engage here. Untested.

### ⚠ `cudagraph_mode: NONE` is NOT the same as eager — MEASURED 2026-09-04

The first boot attempt was configured without `--enforce-eager` on the reasoning
that graphs default off, so the default must already be eager-equivalent.
**That was wrong, and it cost a failed boot.** The two settings are independent:

| Setting | First boot | What it controls |
|---|---|---|
| `cudagraph_mode` | `NONE` | graph capture — off, as the code predicted |
| `mode` | `CompilationMode.VLLM_COMPILE`, `backend: inductor` | **torch.compile — RAN** |
| `enforce_eager` | `False` | disables *both* of the above |

`torch.compile took 11.58 s` in that boot, and it saved an AOT-compiled
function. Inductor's kernel and workspace buffers are **not** capped by
`gpu-memory-utilization` — the gotcha already documented at
`vllm_xpu/compose.yaml:45` — so they consumed the space the KV pool needed:

```
Model loading took 12.87 GiB
Available KV cache memory: 2.89 GiB
ValueError: max seq len (131072) needs 3.1 GiB KV cache > available 2.89 GiB
          estimated maximum model length is 122048
```

Missed 128k by 0.21 GiB. Against the scaler's ~4.33 GiB pool at the same
128k/0.80, the ~1.4 GiB gap is the compile-buffer overhead the scaler never pays
because eager is forced there.

**So `--enforce-eager` is required on this engine too — for a different reason
than the scaler.** On the scaler it prevents silently-empty responses; here it
prevents Inductor from starving the KV pool. Rejected alternatives:

- **raising `gpu-memory-utilization`** — the documented anti-pattern; util does
  not cap compile buffers, so this pushes toward the 0.86 ceiling that OOM'd
- **`--max-model-len 122048`** — hides the cause and breaks 128k parity, which
  is what makes the benchmark against 85.6 tok/s comparable

Compiled mode may still be faster *if* it is correct. Test it as its own
experiment at reduced context — never as a silent default, given the scaler's
compiled-mode failure was silently empty content.

## 4. What differs from `scaler/compose.yaml`

| | scaler | upstream |
|---|---|---|
| Entrypoint | drops to a shell → needs `/bin/bash -lc` + `vllm serve` | `["vllm","serve"]` → **command is args only** |
| `--enforce-eager` | mandatory (silently-empty content otherwise) | **also mandatory**, different reason — Inductor buffers starve the KV pool (§3) |
| `shm_size` | 32g (llm-scaler docs) | 16g (no documented requirement) |
| `restart` | `unless-stopped` | **`"no"`** while unvalidated |
| Online-quant env | `VLLM_QUANTIZE_Q40_LIB` etc. | not present upstream |

Project and volume shape is deliberately **identical** to the other engines — see §7.

`restart: "no"` is deliberate: an unvalidated engine that auto-resurrects would
compete for the B60 and :8000, which is the failure mode already documented for
the `~/llama_cpp` stack.

## 5. Device mapping — upstream confirms the finding here

The documented XPU run recipe passes the **whole `/dev/dri` device plus a
`/dev/dri/by-path` bind mount**, unconditionally, at any world size. That is
exactly the pairing reverse-engineered locally from the
`oneCCL: ze_fd_manager.cpp:144 … opendir failed` crash
([`SCALER_NOTES.md`](SCALER_NOTES.md) §5) — so it is upstream's contract, not a
local workaround.

Upstream's example also uses `--privileged` and `--network=host`. This repo uses
`group_add` (render 992 / video 44) and an explicit port map instead, which is
less privileged and already proven on this host against the Intel images.
**Untested against this image** — if boot fails on device access, `--privileged`
is the first thing to try.

Three things the platform sets automatically, no config needed:
`UCX_MEMTYPE_CACHE=n`, `VLLM_WORKER_MULTIPROC_METHOD=spawn` (if unset), and
`shutdown_timeout=5`. That last one is worth knowing — the in-code reason is
that XPU needs a graceful shutdown to release oneCCL/Level Zero resources,
*"without this, subsequent server startups on the same devices may hang during
CCL initialization."*

## 6. What upstream does NOT provide

Intel's fork keeps some B60-specific surface that upstream lacks:

- the online int4 path and `VLLM_QUANTIZE_Q40_LIB`
- extra arch registrations (Gemma 4, Kimi-VL, ERNIE-VL)
- Battlematrix multi-GPU tuning

Upstream's XPU model table was narrower at 0.28.0. **0.29.0 restructured it**
(✅/🟨 columns replaced by a single Dtype column) and widened it considerably:
Qwen3-Next-80B-A3B, DeepSeek-V4-Flash, Qwen3-VL-32B, Qwen3.5-35B-A3B,
gemma-3-27b, gemma-4-31B and gemma-4-26B-A4B are all now listed as
BF16/Online FP8, plus `Qwen3-30B-A3B-FP8` alongside the existing
`Qwen3-30B-A3B-GPTQ-Int4`. Being listed is an architecture claim, not a fit
claim — the host-RAM and VRAM gates are unchanged, and none of this revisits a
route already decided against.

Requirements worth noting: **Python 3.12 exactly** (the `vllm-xpu-kernels`
wheels are 3.12-only and upstream flags this as a MUST), `torch==2.13.0` XPU,
`triton==3.7.2+xpu`. All satisfied inside the image; the host needs only the
in-kernel `xe` driver. The image also bundles compute-runtime 26.27.39122.11
(past the 26.18 the docs recommend), level-zero 1.32.0, and **`xpu-smi` 2.1.0**.

## 7. Project and volume layout

**Decision 2026-09-04: match the existing engines — shared project, plain
volumes.** Consistency first while the engine is unproven; revisit after testing.

All three engine folders pin `name: llm`, and Compose prefixes declared volumes
with the project name, so a plain `hf-cache:` in each file resolves to the *same*
real volume. Whichever engine comes up first creates `llm_hf-cache`; the others
find it, match its `com.docker.compose.project=llm` label, and attach. No
`external:` needed, and no owner — the three are peers in one project.

```yaml
volumes:
  hf-cache:                # -> llm_hf-cache, shared
  vllm-openai-cache:       # -> llm_vllm-openai-cache, this engine's own
```

| | Value |
|---|---|
| Project | `llm` (shared with `vllm_xpu/`, `scaler/`) |
| Container | `vllm-openai-xpu` |
| Weights (shared) | `llm_hf-cache` |
| Compile cache (own) | `llm_vllm-openai-cache` |

Compile caches stay per-engine — kernels are image-specific — which is why the
siblings are `llm_vllm-cache` and `llm_vllm-scaler-cache`.

Container name checked free of collisions 2026-09-04: in use are `vllm-xpu`,
`vllm-scaler`, `open-webui`, `llama-arc-b60`.

Two consequences of the shared project, both pre-existing on the other engines:

- **Orphan-container warnings are expected** when another engine's container
  lingers. The fix is to `down` that folder. **Never `--remove-orphans`** — in
  one shared project it would delete the other engines, possibly killing prod.
- **`down -v` from ANY engine folder deletes `llm_hf-cache`**, taking ~13 GB of
  weights with it. Plain project volumes are removable by any member of the
  project. Use bare `down`.

### Alternatives, if testing argues for a change

| | Project | `hf-cache` as | Trade |
|---|---|---|---|
| **A — chosen** | `llm` shared | plain | zero config, implicit share; orphan noise; `down -v` hazard |
| **B** | own | `external: llm_hf-cache` | isolation + shared weights; `down -v` safe; volume must pre-exist |
| **C** | own | plain | fully self-contained; costs a 13.8 GB re-download and a second copy |

## 8. Swap procedure

One GPU, ~22.7 GiB, and gpt-oss-20b needs ~17 GiB — so exactly **one** engine
runs at a time. From the repo root:

```bash
docker compose -f vllm_xpu/compose.yaml down          # if the stock engine is up
docker compose -f scaler/compose.yaml down            # if the scaler is up
docker compose -f vllm_openai_xpu/compose.yaml up -d
```

Also stop the `~/llama_cpp` stack first if it is running — it shares the B60 and
:8000 and is `restart: unless-stopped`.

Swap back:

```bash
docker compose -f vllm_openai_xpu/compose.yaml down
docker compose -f scaler/compose.yaml up -d
```

All three files share `name: llm`, so the ~13 GB of gpt-oss-20b weights in
`llm_hf-cache` are **not** re-downloaded. Expect the orphan-container warning
when another engine's container lingers; the fix is to `down` that folder.
**Never `--remove-orphans`** — it would delete the other engines. See §7.

## 9. Open — what to measure

### ✅ Answered on the first real boot, 2026-09-04

| Question | Result |
|---|---|
| Boots on the B60? | **Yes**, at 128k/0.80 **with `--enforce-eager`** (§3) |
| `group_add` enough, or `--privileged`? | **`group_add` suffices** — no `ze_fd_manager` failure, despite upstream's docs using `--privileged` |
| MXFP4 native or dequantized? | **Native** — engine config reports `quantization=gpt_oss_mxfp4`, weights load at 12.87 GiB (bf16 would be ~42 GB) |
| Which engine is serving? | `curl :8000/version` → `{"version":"0.29.0"}` since the bump (§10) |
| Weights re-downloaded? | **No** — `llm_hf-cache` attached, load took 2.3 s warm |

Memory breakdown at 128k/0.80 eager, from `gpu_worker.py:804`:

```
Free memory on device (21.83/22.71 GiB) on startup.
Desired GPU memory utilization is (0.8, 18.17 GiB).
Actual usage is 14.37 GiB consumed (weights + non-torch),
  0.69 GiB peak activation, 0.0 GiB CUDAGraph memory.
Current kv cache memory in use is 3.12 GiB.
```

| | |
|---|---|
| Weights | 12.87 GiB |
| **Non-torch** | **1.50 GiB** |
| Peak activation | 0.69 GiB |
| CUDAGraph | 0.0 GiB ← confirms eager took effect |
| KV pool | 3.12 GiB → **131,815 tokens, 1.01× @128k** |

**Concurrency is a regression against the scaler** (183,314 tok / 1.40×), and
the boot margin is only 0.02 GiB over the 3.1 GiB requirement. The cause is not
compile buffers — eager recovered just 0.23 GiB (2.89 → 3.12). It is upstream's
larger **1.50 GiB non-torch** footprint.

### ⭐ `--kv-cache-memory` — the engine prints the exact value

The same log line says:

```
Replace gpu_memory_utilization config with
  `--kv-cache-memory=3188837581` (2.97 GiB) to fit into requested memory, or
  `--kv-cache-memory=7117934592` (6.63 GiB) to fully utilize gpu memory.
```

`--gpu-memory-utilization 0.80` caps the engine at 18.17 GiB of a 22.71 GiB
card, stranding ~4.5 GiB. Setting `--kv-cache-memory=7117934592` should take the
pool from 3.12 → **6.63 GiB** (~2.1× concurrency), which would beat the scaler.
This is the lever already flagged as the best-value untested change; here it
arrives with an exact byte value. **Test it as its own change**, not bundled
with anything else.

### ✅ Correctness — `smoke.sh` ALL PASS, 2026-09-04

```
served: ['gpt-oss-20b']                              PASS
content: 'pong'                                      PASS
reasoning len: 917                                   PASS
tool_calls: get_weather {"city": "Paris"}             PASS
```

`content: 'pong'` is the important one — that is exactly where the scaler's
compiled mode failed silently. Both parser flags (`openai_gptoss`, `openai`)
transferred from the scaler unchanged, as the registry lookup predicted.

### 📉 Throughput — `bench.sh`, 128k/0.80 eager, 2026-09-04

| Run | tok/s | TTFT |
|---|---|---|
| 1 — warmup, discarded (cold) | 82.8 | 108 ms |
| 2 — `max_tokens=400` | **82.4** | 76 ms |
| 3 — `max_tokens=200` | **82.8** | 76 ms |

Against `intel/llm-scaler-vllm:0.26.0-b1` (85.6 @400 / 86.0 @200, TTFT ~73 ms):

> **Note on the scaler baseline.** Every `0.26.0-b1` column in this file is still
> valid for the current scaler pin. That pin moved to `0.26.0-b2` on 2026-09-10
> and was re-measured on the same box with the same scripts: 85.6 @400, 86.1
> @200, ~72 ms, KV pool 183,314 tok / 1.40× — identical within noise
> ([`SCALER_NOTES.md`](SCALER_NOTES.md) §3). b1 is kept as the column label
> because that is when the numbers were taken.

| | scaler `0.26.0-b1` | upstream `v0.28.0` | Δ |
|---|---|---|---|
| tok/s @400 | 85.6 | 82.4 | **−3.7%** |
| tok/s @200 | 86.0 | 82.8 | **−3.7%** |
| TTFT | ~73 ms | 76 ms | ~equal |
| KV pool @128k | 183,314 tok / 1.40× | 131,815 tok / 1.01× | **−28%** |

⚠️ **Methodology note:** every run reported `Content chunks = 0` — "model never
left reasoning phase". The 200/400 token budget was consumed entirely by
reasoning tokens, so these figures measure *reasoning-token* decode throughput.
That is consistent with the scaler baseline (same script, prompt and budgets),
so the comparison holds, but neither number is end-to-end content throughput.

### Verdict as of 2026-09-04

Upstream is **correct but slower and tighter**: −3.7% decode and a 28% smaller
KV pool at identical settings. It is not currently a reason to move off the
scaler pin. What it does offer is non-performance: two vLLM minors newer, half
the image size (4.14 vs 7.78 GB), mainline rather than an undocumented beta, a
trustworthy `latest`, and `xpu-smi` in the image.

### ⭐ `--kv-cache-memory=7117934592` — MEASURED 2026-09-04, big win on capacity

Booted clean, `smoke.sh` **ALL PASS** again, and:

```
GPU KV cache size: 280,425 tokens
Maximum concurrency for 131,072 tokens per request: 2.14x
```

| | tok/s @400 | tok/s @200 | TTFT | KV tokens | Concurrency |
|---|---|---|---|---|---|
| scaler `0.26.0-b1` | **85.6** | **86.0** | ~73 ms | 183,314 | 1.40× |
| upstream, util 0.80 only | 82.4 | 82.8 | 76 ms | 131,815 | 1.01× |
| upstream + `--kv-cache-memory` | 82.3 | 82.7 | 76 ms | **280,425** | **2.14×** |

**Pool 2.13× larger, tok/s unchanged** (82.4 → 82.3, within noise) — exactly as
expected, since decode is VRAM-bandwidth-bound. The ~4.5 GiB that
`--gpu-memory-utilization 0.80` strands is real, recoverable memory, and the
engine's printed byte value was accurate.

⚠️ **Two honest limits on this result:**

1. **2.14× is allocated capacity, not measured concurrency.** `bench.sh` is
   single-stream. The pool provably allocates and single-stream is correct —
   *not* that two concurrent 128k requests actually run. This config sits on
   ~0.14 GiB of card headroom, which is the territory where OOM-on-the-edge and
   504s appeared before. Only `--enforce-eager` (no growing compile buffers)
   makes it defensible. A real concurrent-load test is still owed.
2. **Setting `--kv-cache-memory` costs observability.** vLLM skips memory
   profiling when the pool is pinned, so the `gpu_worker.py:804` breakdown
   (`Actual usage` / non-torch / activation) **disappears from the log**. Unset
   the flag when you need that diagnostic.

### 🔑 The real finding: this lever is engine-agnostic and UNTESTED on the scaler

The util-based sizing strands memory on *both* engines. The scaler's own log
offered **8.05 GiB** (vs upstream's 6.63). If 6.63 GiB takes upstream from 1.01×
to 2.14×, then 8.05 GiB should take the scaler well past that **while keeping its
85.6 tok/s** — dominating upstream on both axes.

**So the highest-value next experiment is `--kv-cache-memory` on `scaler/`, not
anything further on upstream.** The engine choice turned out to matter less than
the pool sizing.

### Verdict as of 2026-09-04 (superseded by §10)

Upstream `v0.28.0` is **correct, capacity-competitive, and ~4% slower**. With
the pool pinned it beats the scaler on concurrency (2.14× vs 1.40×) but loses
decode (−3.9%). Not a reason to migrate — the scaler keeps the speed crown and
has not yet been given the same lever.

---

## 10. The 0.29.0 bump — MEASURED 2026-09-09

`v0.29.0` released 2026-09-09 (GitHub release 08:54 UTC; the XPU image was
pushed 05:43 UTC, ~3 h earlier). Pin moved `v0.28.0` → `v0.29.0`.

**Verified unchanged, so the compose stayed valid:** `vllm/platforms/xpu.py` is
**byte-identical** between the tags (graph gating, `supported_quantization`
mxfp4/gpt_oss_mxfp4, spawn, shutdown_timeout), `ENTRYPOINT ["vllm","serve"]`,
Python 3.12, `torch==2.13.0`, `triton==3.7.2+xpu`, and the `--enforce-eager`,
`--kv-cache-memory`, `openai_gptoss`, `openai` surfaces. XPU deps moved only
`vllm_xpu_kernels` 0.1.13.2 → 0.1.14.1, plus UCX `v1.21.x` / NIXL 1.3.2. Nothing
in the release's Breaking Changes list touches this engine.

### 10.1 Model Runner V2 became the default — the whole story of this upgrade

At `v0.28.0`, `VllmConfig.use_v2_model_runner` required
`arch ∈ DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES or not is_moe`.
`GptOssForCausalLM` is in neither set, so **0.28.0 silently ran V1** all along.
0.29.0 (#53183) deletes that gate: V2 is now default unless the arch is on the
ROCm MRV1 list, Triton is missing, or a config feature is unsupported. None
apply here → **0.29.0 selects V2 for this model**.

Confirming which runner is live, two ways:

| | V1 | V2 |
|---|---|---|
| Log line | *(none — absence is the signal)* | `xpu_worker.py:113 Using V2 Model Runner` |
| Module in log paths | `gpu_model_runner.py` | `worker/gpu/model_runner.py` |

`Initializing a V1 LLM engine` is the **engine** V1, a different axis — do not
read it as the runner. Override with `VLLM_USE_V2_MODEL_RUNNER` (`0`/`1`,
integer only; blank crashes `bool(int(""))`).

### 10.2 ⚠️ V2 SEGFAULTS unless `SYCL_CACHE_PERSISTENT=0`

First V2 boot died after a clean memory profile, with no Python traceback:

```
!!!!!!! Segfault encountered !!!!!!!
  at::native::xpu::topk_kernel → sbtopk_try_launch → single_wg_topk_try_launch
  → single_wg_launch_impl<c10::BFloat16, 8, 32, int>
  → sycl::handler::finalize() → ProgramManager::getBuiltURProgram
  → PersistentDeviceCodeCache::getItemFromDisc → getSortedImages   ← segfault
```

Traced to source: `gpu_worker.py:864` runs `warmup_kernels(...)` **for V2 only**
(V1 takes the `elif` at :867 → `_dummy_sampler_run`). That warmup builds its
batch with `SamplingParams.for_sampler_warmup()`, which hardcodes `logprobs=5,
prompt_logprobs=1` to "exercise all sampler logic" — and logprobs reach
`sampler.py:335 torch.topk(...)`, whose XPU kernel segfaults during SYCL program
build while reading the **persistent** device-code cache.

**Fix: `SYCL_CACHE_PERSISTENT=0`** (that variable is set by this repo, not a
vLLM default).
Verified as a single-variable change against the failing run: V2 then boots
clean and `smoke.sh` is ALL PASS. Cost is a device-kernel rebuild each boot;
total boot still ~70 s.

Notes on the failure mode, for whoever meets it next:

- The crashing feature is **logprobs, which this deployment never requests** —
  V2 died proving a path this deployment never uses. Normal serving never
  reaches it.
- Not memory: `OOMKilled=false`, the profile printed, the pool allocated.
- No config knob avoids the warmup. `for_sampler_warmup()` hardcodes its params,
  `--max-logprobs` does not reach it (warmup builds `SamplingParams` directly,
  unvalidated), and `envs.py` has no skip for `warmup_kernels` —
  `KernelConfig.enable_jit_warmup` governs the *other* warmup, which succeeds.
- **Unreported upstream** as of 2026-09-09: zero issues match `getSortedImages`
  or `PersistentDeviceCodeCache`. Nearest neighbours are #55231 (open, XPU MoE
  topk, but in `vllm-xpu-kernels`, not aten's `topk_kernel`) and #46179 (MRV2
  failing on ROCm — same shape, different platform). Worth filing.

### 10.3 Four configs measured, same box, same scripts

| | scaler `0.26.0-b1` | 0.29.0 + V1, pinned 6.63 GiB | 0.29.0 + V2, util only | **0.29.0 + V2, pinned 8.01 GiB** |
|---|---|---|---|---|
| tok/s @400 | **85.6** | 82.4 | 83.3 | 83.1 |
| tok/s @200 | **86.0** | 82.8 | 83.6 | 83.5 |
| TTFT | ~73 ms | 76 ms | 76 ms | 76 ms |
| KV pool | 183,314 tok | 280,425 tok | 169,123 tok | **338,928 tok** |
| Concurrency @128k | 1.40× | 2.14× | 1.29× | **2.59×** |
| `smoke.sh` | PASS | PASS | PASS | **ALL PASS** |

**The image bump alone is a no-op.** 0.29.0 + V1 reproduced 0.28.0 + V1 exactly
— 82.4 / 82.8 tok/s, 76 ms, 280,425 tokens, 12.87 GiB weights — which is what
made the runner the only variable in every later comparison.

**V2 is faster and leaner than V1:**

| | V1 | V2 |
|---|---|---|
| Weights | 12.87 GiB | 12.87 GiB |
| Non-torch | ~1.50 GiB | **~1.03 GiB** |
| Peak activation | 0.69 GiB | **0.27 GiB** |
| Advised "fully utilize" | 6.63 GiB (`7117934592`) | **8.01 GiB (`8603448832`)** |

⚠️ **`--kv-cache-memory` is runner-specific.** The byte value comes from that
runner's memory profile; carrying V1's 6.63 GiB onto V2 leaves ~1.4 GiB
unclaimed. Re-derive it after any runner or image change by commenting the flag
out for one boot and reading `gpu_worker.py:860`.

Same two limits as before: **2.59× is allocated capacity, not measured
concurrency** (`bench.sh` is single-stream), and the pinned pool leaves ~0.15 GiB
of card headroom — defensible only because `--enforce-eager` means no growing
compile buffers.

### 10.4 Verdict as of 2026-09-09

Upstream `v0.29.0` + V2 + the pinned pool is **the best-measured config this
engine has had**: `+85%` KV pool against the scaler for `−2.9%` decode, and
`+0.9%` speed with `+21%` pool against where the engine started the day. The
scaler still holds the decode crown at 85.6 tok/s — and still has not been given
the `--kv-cache-memory` lever, which remains the highest-value open experiment.

### Still open

1. **`--kv-cache-memory` on the scaler** (8.05 GiB) — still the priority, and now
   better motivated: the same lever took this engine 1.29× → 2.59×.
2. A genuine **concurrent-load** test — 2.59× is unverified capacity.
3. `VLLM_XPU_ENABLE_XPU_GRAPH=1` — only available single-GPU (§3), and compiled
   mode needs its own correctness pass before any speed claim.
4. The stock `intel/vllm:0.21.0` baseline, still unmeasured.
5. **File the MRV2 segfault upstream** (§10.2) — clean repro, no existing issue.
6. Cold-boot cost of `SYCL_CACHE_PERSISTENT=0` measured properly; and whether a
   later image lets the persistent cache be re-enabled.
7. Whether `/dev/dri/by-path` is still needed — 0.29.0 skips the oneCCL warm-up
   at world_size=1 (#52389), which may make it redundant (§5).
8. `Auto-initialization of reasoning token IDs failed` appears in the 0.29.0 boot
   log. `smoke.sh` reasoning passes, so it is cosmetic; unattributed to a runner.

The follow-on doc work is a sweep — `README.md`, `DEVELOPER.md` and
`INTEL_ARC_B60.md` all still describe a **two**-engine repo and a two-way swap.

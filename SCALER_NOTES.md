# llm-scaler engine notes

Rationale, measurements, and **tested-then-rejected** configurations for the
`llm-scaler` engine in [`scaler/compose.yaml`](scaler/compose.yaml). These notes
used to live as ~270 lines of comments inside that file, which made the actual
configuration hard to read. The stock engine (`vllm_xpu/`) documents itself
inline and is only referenced here for comparison.

How this file differs from the other docs:

| Doc | Audience | Answers |
|-----|----------|---------|
| [README.md](README.md) | operator | How do I run, swap, and verify the stack? |
| [DEVELOPER.md](DEVELOPER.md) | developer | How is it put together, and why these values? |
| **SCALER_NOTES.md** (this file) | whoever changes a scaler flag | What has already been tried, measured, and ruled out? |

Read this before "improving" a flag — most of the obvious ideas have been tested
on real hardware and rejected for recorded reasons. All measurements are on a
single **Intel Arc Pro B60 (24 GB)** serving `gpt-oss-20b` unless stated.

**Two conventions in this file:**

1. **Intel's documentation is linked, never copied.** Anything upstream already
   documents — supported models, reference commands, env-var meanings — is a URL
   below, so it cannot go stale here. What this file records is what *we*
   measured on *this* hardware, and where our findings differ from the docs.
2. **Every tested configuration names the full image tag and the date tested,**
   and says explicitly whether it has been re-tested on the current pin. A number
   without an image tag is not a result.

## Upstream documentation

Primary reference, pinned to the git tag matching our image so it cannot drift:
**[llm-scaler vLLM README @ `vllm-0.26.0-b1`](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md)**

| Topic | Link |
|-------|------|
| Supported models table | [§3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#3-supported-models) |
| INT4 / FP8 online quantisation | [§2.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#22-int4-and-fp8-quantized-online-serving) |
| Pulling / running the container | [§1.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#13-pulling-and-running-the-vllm-docker-container) |
| Finding maximum context length | [§2.7](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#27-finding-maximum-context-length) |
| gemma-4 / diffusiongemma reference commands | [§3.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#33-reference-commands-for-running-gemma-4-models-and-diffusiongemma) |
| FP8 KV cache | [§3.5](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#35-fp8-kv-cache) |
| MTP (speculative decoding) | [§3.6](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#36-mtp-enable) |
| OOM during online quantisation | [§4.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#42-out-of-memory-while-online-quantization) |
| Performance tuning / NUMA | [§5](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#5-performance-tuning) |

Other upstream sources:

| | |
|---|---|
| Latest docs (may be **ahead** of our pin) | <https://github.com/intel/llm-scaler/blob/main/vllm/README.md> |
| Image release list | <https://github.com/intel/llm-scaler/blob/main/Releases.md> |
| FAQ | <https://github.com/intel/llm-scaler/blob/main/vllm/FAQ.md> |
| Known issues | <https://github.com/intel/llm-scaler/blob/main/vllm/KNOWN_ISSUES.md> |
| Published image tags | <https://hub.docker.com/r/intel/llm-scaler-vllm/tags> |
| vLLM on KV-cache quantisation | <https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html> |

---

## 1. Current pin

| | |
|---|---|
| Image | `intel/llm-scaler-vllm:0.26.0-b1` |
| Base | vLLM 0.26.0 (engine reports `0.26.1.dev0+g568afb3a1.d20260831`) |
| Model | `openai/gpt-oss-20b`, served as `gpt-oss-20b` |
| Context / util | `--max-model-len 131072` (128k) / `--gpu-memory-utilization 0.80` |
| Boot mode | `--enforce-eager` — **mandatory**, see §4 |
| Status | Boot-tested and validated 2026-09-02 |

**Which engine owns `:8000`?** `curl -s localhost:8000/version`. The scaler
reports `0.26.1.dev0…`; the stock `vllm_xpu` image reports `0.21.0`. Useful
because three different stacks on this host can claim that port, and a benchmark
against the wrong one is worse than no benchmark.

---

## 2. Image pin provenance

Upstream says **do not use `:latest`**. On this project you additionally cannot
trust `latest` *or* the release notes to tell you what is newest — verified
2026-09-02 against the Docker Hub tag API and `git ls-remote --tags`:

| | |
|---|---|
| `intel/llm-scaler-vllm:0.26.0-b1` pushed | 2026-09-02, 7.78 GB compressed (b3 was 5.18 GB) |
| `Releases.md` says "Latest Release" | `0.21.0-b3.1` — on both `main` **and** the `vllm-0.26.0-b1` git tag |
| `latest` resolves to | the `0.21.0-b3.1`-era image |
| Skipped | `0.21.0-b3.1` (2026-08-13) |

So our pin is an **undocumented beta**. Check
[the tag list](https://hub.docker.com/r/intel/llm-scaler-vllm/tags) for ground
truth, not [`Releases.md`](https://github.com/intel/llm-scaler/blob/main/Releases.md).

The jump from `0.21.0-b3` is the vLLM **base**, not a patch level: 0.21.0 →
0.26.0. **Side effect:** the stock engine is still on 0.21.0, so the two engines
no longer share a base and a base-vs-scaler A/B is confounded by a five-minor
engine gap again — the opposite of why the 0.21.0 bump was originally taken.

No release in this line contains a gpt-oss fix (the notes are all Qwen3.6 /
gemma-4 / Muse-Glimmer / diffusiongemma work), yet gpt-oss-20b got ~19–21%
faster — so that win comes from the newer base, not the fork's own commits.

**Supported models:** read the live table at
[§3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#3-supported-models)
rather than a copy here. What our own diff of that table found (b3 →
`0.26.0-b1`): only **3 rows added** (`Muse-Glimmer-30B`, `Qwen3.8-27B`,
`Qwen3.8-27B-FP8`), none removed or changed, and none of the three fits a single
B60. gpt-oss-20b/120b remain in the MXFP4 column, so our model stays supported.

---

## 3. Performance record

All figures are **eager** (compiled is not an option — §4), single-stream via
`./bench.sh`, on one Arc Pro B60 serving `openai/gpt-oss-20b`.

### Decode throughput

| Image (full tag) | Date tested | Context / util | `bench.sh 400` | `bench.sh 200` | TTFT |
|------------------|-------------|----------------|----------------|----------------|------|
| `intel/llm-scaler-vllm:0.14.0-b8.3.2` | 2026-07-03 | 64k / 0.75 | 80.8 tok/s | — | 109 ms |
| `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | 128k / 0.80 | 70.6 tok/s | 72.0 tok/s | ~81–92 ms |
| **`intel/llm-scaler-vllm:0.26.0-b1`** | **2026-09-02** | 128k / 0.80 | **85.6 tok/s** | **86.0 tok/s** | **~73 ms** |

`0.26.0-b1` is **+21.2% over b3** and **+5.9% over b8.3.2** (the previous best),
so the ~10–12.6% regression that made b3 a hard sell is **gone**. Raw runs:
85.6 / 84.3 / 85.6 @400 and 86.0 / 86.0 / 86.0 @200.

⚠ The b8.3.2 row was measured at a **different context/util profile** (64k/0.75
vs 128k/0.80), so treat b8.3.2-vs-later as indicative, not matched. The
b3-vs-`0.26.0-b1` comparison *is* matched on every axis.

### Boot-time memory figures

| Image (full tag) | Date tested | Weights | KV memory | KV pool | Max concurrency @131,072 |
|------------------|-------------|---------|-----------|---------|--------------------------|
| `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | 12.87 GiB / 7.68 s | 5.46 GiB | 234,645 tok | 1.79× |
| `intel/llm-scaler-vllm:0.26.0-b1` | **not captured** | — | — | — | — |

For context, the stock `intel/vllm:0.21.0-ubuntu24.04` engine reports a 221k pool
/ 1.69× at the same profile, so the fork sized its KV pool slightly *larger* on
the same base. The b3 boot also confirmed 1 XPU enumerated,
`Intel(R) Arc(TM) Pro B60 Graphics`, 23.9 GiB.

`GET /metrics` does **not** expose pool size on this build (only
`kv_cache_usage_perc`), so filling in the `0.26.0-b1` row needs a boot-log read:

```bash
docker compose -f scaler/compose.yaml logs \
  | grep -E "KV cache size|Maximum concurrency|Model loading took"
```

### Correctness

| Image (full tag) | Date tested | `smoke.sh` | Notes |
|------------------|-------------|------------|-------|
| `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | ALL PASS | `message.reasoning` 835 chars |
| `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-02 | ALL PASS | `message.reasoning` 1082 chars; confirms the inherited `--reasoning-parser` / `--tool-call-parser` names survived the base jump |

`smoke.sh` checks four things: served model name, non-empty content, a populated
`message.reasoning`, and a `get_weather` tool_call.

### Bench hygiene — this has bitten us twice

- **`bench.sh` reports `total_chunks / decode_time`, not true tokens/s.** A
  longer budget can therefore score *lower* on the same engine: b3 gave 72.0
  @200 but 70.6 @400. **Always compare at equal `max_tokens`.** The b8.3.2 80.8
  figure was `./bench.sh 400`, which is why the matched b3 gap is ~12.6%, not
  the ~10% first quoted.
- **Always discard run 1 after a cold start.** On b3 the first request after
  startup measured 57.5 tok/s — warm-up, not the engine's speed.
- `0.26.0-b1` is nearly budget-flat (85.6 @400 vs 86.0 @200), so this trap
  matters mostly when comparing *across* images.

---

## 4. Boot mode: `--enforce-eager` is mandatory, not a safe default

| Configuration | Image (full tag) | Date tested | Result |
|---------------|------------------|-------------|--------|
| `--enforce-eager` (current) | `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-02 | Correct output, 85.6 tok/s |
| **no** `--enforce-eager` (torch.compile ON) | `intel/llm-scaler-vllm:0.14.0-b8.3.2` | 2026-07-03 | **Rejected — silently empty output** |
| no `--enforce-eager` | `:0.21.0-b3`, `:0.26.0-b1` | **not re-tested** | — |

Dropping `--enforce-eager` boots and compiles fine (~80 s) but then generates
**empty output** — `completion_tokens` increment while both `content` and
`reasoning` come back null/blank. A silent correctness failure with no crash or
NaN in the logs. Inductor's compiled graph is broken for this MXFP4 MoE on XPU;
the fork's custom kernels only produce valid output in eager.

Consequences:

- Every tok/s figure in §3 is an eager number and is the engine's **real** speed
  — not an under-statement, because compiled is not an option here.
- Eager disables torch.compile, so there is no compile-buffer growth and util is
  **not** constrained by Inductor buffers on this engine. 0.85 would be safe; we
  use 0.80 to match the stock engine (raised from 0.75 on 2026-07-08, once the
  B60 stopped driving displays).
- `0.21.0-b1` added "experimental XPU graph" support and the base is now much
  newer — either *might* change this. Re-test explicitly as a
  correctness-verified experiment; never flip it blind.

gpt-oss-20b is MXFP4 (pre-quantised) — **do not pass `--quantization`**.

---

## 5. Device mapping and oneCCL

The whole `/dev/dri` device mapping **plus** the `/dev/dri/by-path` bind-mount
are both required, matching `vllm_xpu/compose.yaml`.

A narrow `renderD128` + `card1` mapping worked only while this fork sat on an
older vLLM base. From 0.21.0 onward the base enumerates `/dev/dri/by-path` for
its warm-up `all_reduce` and dies without it:

```
oneCCL: ze_fd_manager.cpp:144 ... opendir failed
```

Docker's `devices:` never recreates that symlink directory, hence the separate
read-only bind-mount. The AMD iGPU nodes the whole-`/dev/dri` mapping also
exposes are harmless — they are not SYCL XPUs, so there is no wrong-GPU risk.

---

## 6. Volumes

- **`hf-cache` is shared with `vllm_xpu`** via the deliberate `name: llm`
  project name, so gpt-oss-20b's ~13 GB of weights are reused, not
  re-downloaded. `docker volume rm llm_hf-cache` would delete the gpt-oss
  weights too — prune selectively if reclaiming space.
- **`vllm-scaler-cache` is separate** because compiled kernels are
  image-version-specific. **Clear it once after any image bump** — from
  `scaler/`: `docker compose down`, then
  `docker volume rm llm_vllm-scaler-cache`. Running eager means little is cached
  there anyway, but stale kernels from a different vLLM base are not worth
  debugging.

---

## 7. Environment variables

`ZES_ENABLE_SYSMAN`, `SYCL_CACHE_PERSISTENT`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn` and `shm_size: 32g` follow upstream's
recommendations — see [§1.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#13-pulling-and-running-the-vllm-docker-container).
`VLLM_QUANTIZE_Q40_LIB`, `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` and
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` are the online-quantisation set, documented at
[§2.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#22-int4-and-fp8-quantized-online-serving)
and [§4.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#42-out-of-memory-while-online-quantization);
all three are **inert for gpt-oss-20b**, which is pre-quantised MXFP4.

Only the parts where our experience **differs from the docs** are recorded here:

| Variable | Our finding | Verified |
|----------|-------------|----------|
| `VLLM_QUANTIZE_Q40_LIB` | The path in Intel's README (`/usr/local/lib/python3.12/dist-packages/…`) is **wrong for this image line** and crash-loops the engine with "cannot open shared object file". The working path is under `/opt/venv/…`, found by `find` inside the image. | `:0.21.0-b3`, 2026-08-12. ⚠ **Unverified on `:0.26.0-b1`**, which grew ~50% — re-check before using `sym_int4`. |
| `HF_TOKEN` | Not needed for current models: `google/gemma-4-26B-A4B-it` is `gated: false`, apache-2.0 (unlike gemma 2/3). Kept as value-only passthrough so the file carries no secret and empty means anonymous. | HF API, 2026-08-12 |

---

## 8. Tested and rejected

### 8.1 `gemma-4-26B-A4B-it` on a single B60 — blocked

| Route attempted | Image (full tag) | Date tested | Result |
|-----------------|------------------|-------------|--------|
| `Intel/gemma-4-26B-A4B-it-int4-AutoRound` + `--quantization gptq` / `moe_wna16` | `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | ❌ `NotImplementedError: No Unquantized MoE backend…` — wall 1 |
| `google/gemma-4-26B-A4B-it` + `--quantization sym_int4` (upstream's documented route) | `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | ❌ `RuntimeError: unable to mmap 49907246508 bytes` — wall 2, host RAM |
| either route | `intel/llm-scaler-vllm:0.26.0-b1` | **not attempted** | — |

Two independent walls, and the second is the real one.

**Wall 1 — MoE construction.** With both `--quantization gptq` and
`--quantization moe_wna16` against a fully-int4 checkpoint:

```
NotImplementedError: No Unquantized MoE backend supports the deployment configuration.
```

Raised in `model_loader/utils.py` `initialize_model` — at model *construction*,
before a single weight is read. The Gemma4 MoE module is built unquantised
whatever `--quantization` says, then asks for an XPU backend for an unquantised
fused-MoE, and there is none. No `--quantization` value fixes it.

**Wall 2 — host RAM, and this one is decisive.** Upstream's documented route
([§2.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#22-int4-and-fp8-quantized-online-serving))
is `--quantization sym_int4` on Google's original checkpoint, which fails before
touching the GPU:

```
RuntimeError: unable to mmap 49907246508 bytes ... Cannot allocate memory
```

Google ships that checkpoint as **one 49.9 GB shard**, and online quantisation
must open the full-precision file. With 30 GiB RAM + 8 GiB swap under
`vm.overcommit_memory=0`, the kernel refuses a single 49.9 GB mapping outright.
**Online quantisation on this box is capped by host RAM, not VRAM** — check the
*largest shard*, not the total, before reaching for util/context knobs.
Workarounds if ever needed: `vm.overcommit_memory=1`, ~32 GB more swap, or best,
a pre-quantised normally-sharded checkpoint.

**Checkpoint findings** (the checkpoint was never the problem):

- `Intel/gemma-4-26B-A4B-it-int4-AutoRound` is fully int4 — 11,520 expert
  `qweight` tensors, zero unquantised expert weights, 8 shards, 15.4 GB,
  **largest shard 2.1 GB**. Verified via its HF `index.json`. That sharding
  would sidestep Wall 2 entirely.
- Its `quant_method` is `auto-round`, which b3's registry did not contain, but
  `packing_format` is `auto_round:auto_gptq` — the tensors are in GPTQ layout,
  so `gptq`/`moe_wna16` accept them. The label was the only mismatch; loading
  was never what failed.

**Status on `0.26.0-b1`:** the upstream support row for this model is
byte-identical to b3 — online fp8/int4 only, no offline column, and still no
reference command. Upstream publishes gemma-4 commands only for
`gemma-4-12B-it` (TP=1) and `diffusiongemma-26B-A4B-it` (TP=2), never for
`gemma-4-26B-A4B-it`; see
[§3.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#33-reference-commands-for-running-gemma-4-models-and-diffusiongemma)
for the current commands rather than trusting a copy here.

One new lever, **low confidence**: vLLM 0.26.0's registry adds `auto_gptq` and
`auto_awq` as first-class methods (`AutoGPTQConfig`/`AutoAWQConfig`), neither of
which existed in 0.21.0. There is still no `auto_round.py` upstream, and the b3
failure was in MoE *construction*, so this only helps if the fork now ships a
quantised-Gemma4 MoE XPU kernel. Not pursued.

`gemma-4-12B-it` remains the plausible single-B60 gemma-4 path — dense and TP=1
in upstream's own reference command ([§3.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#33-reference-commands-for-running-gemma-4-models-and-diffusiongemma)).
Untested here.

### 8.2 `--kv-cache-dtype fp8` — worked, then reverted by decision

Tested on `intel/llm-scaler-vllm:0.21.0-b3`, **2026-08-21**. It worked and cost
nothing measurable — but the flag is **not currently set**, and has **not** been
re-tested on the current pin.

| Metric | b3 without the flag | b3 **with** `--kv-cache-dtype fp8` | `0.26.0-b1` |
|--------|---------------------|------------------------------------|-------------|
| GPU KV cache size | 234,645 tok | **469,354 tok** (exactly 2.0×) | not re-tested |
| Max concurrency @131,072 | 1.79× | **3.58×** | not re-tested |
| KV memory | 5.46 GiB | 5.46 GiB | not re-tested |
| `bench.sh 200` | 72.0 tok/s | 72.0 tok/s (**unchanged**) | not re-tested |
| `bench.sh 400` | 70.6 tok/s | 70.6 tok/s | not re-tested |
| `smoke.sh` | ALL PASS | ALL PASS | not re-tested |
| Attention backend | `FLASH_ATTN` | `FLASH_ATTN` (no silent Triton fallback) | not re-tested |

Both b3 columns are from the same image, days apart; the flag was added,
measured, then removed. Removed because it did not touch the ~10% decode
regression being chased at the time — it buys **concurrency, not speed**. That
regression is gone as of `0.26.0-b1`, so the only remaining motive for fp8 KV is
long-context concurrency.

If re-added, validate on a **real long prompt** first: vLLM warns that fp8 KV may
cause accuracy loss without a proper scaling factor, and `smoke.sh`'s four
shallow checks do not clear that. Upstream guidance:
[vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)
and llm-scaler [§3.5](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#35-fp8-kv-cache).

### 8.3 Two dead ends, so nobody re-treads them

| Idea | Verdict | Basis |
|------|---------|-------|
| Enable "flash attention" | **Already on — not a lever.** There is no `--flash-attn` flag in vLLM (that is llama.cpp's `-fa`). `platforms/xpu.py` already selects `FLASH_ATTN` (FlashAttention 2, KV block size 64, NHD layout) with no flag from us. | Read from `platforms/xpu.py` in the running b3 container, 2026-08-21 |
| Switch to `TRITON_ATTN` | **Downgrade.** The only other non-MLA option, and that same source file calls it the old fallback; FA2 measured ~19% faster decode TPOT than Triton on B-series. | same |
| `--kv-cache-dtype turboquant_*` (`k8v4` / `4bit_nc` / `k3v4_nc` / `3bit_nc`) | **Ruled out.** Routes to a separate TURBOQUANT backend and costs 20–60% throughput, in exchange for capacity we do not need. | [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) |

---

## 9. Open items

- Bench the **stock** engine to close the base-vs-scaler comparison — the
  original point of keeping both folders, and now the only missing piece.
- Capture KV-pool size / max concurrency for `0.26.0-b1` from the boot log (§3).
- Re-verify the `VLLM_QUANTIZE_Q40_LIB` path on this image (§7) if `sym_int4` is
  ever used.
- `healthcheck.start_period` is still `7200s`, which was raised for the gemma-4
  experiment's ~52 GB download-and-quantise first boot. gpt-oss-20b only needs
  the ~1800s that covered its silent XPU cold start.

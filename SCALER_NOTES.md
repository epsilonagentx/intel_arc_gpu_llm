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
| Status | Boot-tested and validated 2026-09-02; re-validated after the 2026-09-03 XPU-graph experiment (§4) — config unchanged |

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
| `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-03 | 12.87 GiB / 7.28 s | **4.33 GiB** | **183,314 tok** | **1.40×** |

For context, the stock `intel/vllm:0.21.0-ubuntu24.04` engine reports a 221k pool
/ 1.69× at the same profile, so the fork sized its KV pool slightly *larger* on
the same base. The b3 boot also confirmed 1 XPU enumerated,
`Intel(R) Arc(TM) Pro B60 Graphics`, 23.9 GiB.

The `0.26.0-b1` VRAM budget, read from its boot log (`gpu_worker.py:857`):

| Item | Value |
|------|-------|
| Free on device at startup | 22.99 of 23.91 GiB |
| Budget at `--gpu-memory-utilization 0.80` | 19.12 GiB |
| Weights | 12.87 GiB |
| Peak activation | 0.80 GiB |
| Non-torch memory | 1.12 GiB |
| CUDAGraph memory | **0.00 GiB** (eager — §4) |
| **KV cache, the remainder** | **4.33 GiB** |

Two findings.

**1. KV memory *shrank* 5.46 → 4.33 GiB at an identical 128k/0.80 profile**, and
the pool with it: **234,645 → 183,314 tokens, 1.79× → 1.40×** concurrency at
131,072 tokens per request. Weights are byte-identical (12.87 GiB), so the newer
base spends ~1.1 GiB more outside the KV pool. **This is the real cost of the
`0.26.0-b1` upgrade:** decode got 21% faster and the KV pool got 22% smaller.
Single-stream users trade nothing; anything relying on concurrency at long
context lost a fifth of its headroom. Finding 2 below buys it back.

**2. There is ~1.86× of unused KV headroom, and the engine says so itself.** The
same log line ends:

```
Replace gpu_memory_utilization config with `--kv-cache-memory=4496187392`
(4.19 GiB) to fit into requested memory, or `--kv-cache-memory=8647532544`
(8.05 GiB) to fully utilize gpu memory.
```

8.05 GiB of KV is 1.86× today's 4.33 GiB — the same order of capacity gain as
`--kv-cache-dtype fp8` (§8.2), without fp8's accuracy question. Reaching it needs
util ≈ 0.955, or better, `--kv-cache-memory` set explicitly, which sizes the pool
directly instead of leaving it as whatever the util budget does not spend.
**Untested.** It would leave only ~0.15 GiB of the card unallocated, so an
intermediate value is the sane first attempt. Note that the 0.86 util ceiling
carried over from the stock engine does **not** bind here: eager reserves no
Inductor or CUDAGraph buffers, and the 0.00 GiB row above is the proof (§4).

`GET /metrics` does **not** expose pool size on this build (only
`kv_cache_usage_perc`), so the two missing cells still need a boot-log read:

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
  B60 stopped driving displays). The `0.26.0-b1` boot log now quantifies that
  headroom — §3.
- The base is five minors newer than the image that failed, which *might* change
  the verdict — but re-test it as a deliberate, correctness-verified experiment,
  never flip it blind. The fork's "experimental XPU graph" support, added in
  `0.21.0-b1`, is a **different** switch and is covered below.

gpt-oss-20b is MXFP4 (pre-quantised) — **do not pass `--quantization`**.

### XPU Graph — tested 2026-09-03, a **no-op** under eager. Do not set it.

| Configuration | Image (full tag) | Date tested | Result |
|---------------|------------------|-------------|--------|
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `--enforce-eager` | `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-03 | **No effect. Accepted, changed nothing, reverted.** |
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` without `--enforce-eager` | — | **not tested** | Would reintroduce the empty-output risk above |

The variable **is** read — setting it removes the `xpu.py:285` warning — but
nothing downstream changes. Measured against a same-boot baseline on the same
image:

| Metric | Baseline | `VLLM_XPU_ENABLE_XPU_GRAPH=1` |
|--------|----------|-------------------------------|
| `bench.sh 400` | 84.3 / 85.6 / 85.6 tok/s | 85.6 / 85.6 / 85.6 (run 1 cold, 85.6, discarded) |
| `bench.sh 200` | 86.0 / 86.0 / 86.0 tok/s | 86.0 / 86.0 / 86.0 |
| TTFT | ~73 ms | ~73–74 ms |
| KV cache / pool | 4.33 GiB / 183,314 tok / 1.40× | **byte-identical** |
| CUDAGraph memory | 0.0 GiB | **0.0 GiB** |
| `smoke.sh` | ALL PASS, reasoning 1082 chars | ALL PASS, reasoning 1082 chars |
| Greedy reference generation | — | **byte-for-byte identical** (`temperature: 0`) |

Three independent signals say the graph path never engaged: **zero** VRAM
allocated for graphs, **zero** capture/replay lines in the boot log, and
identical throughput at both budgets. `--enforce-eager` still logs
`mode: CompilationMode.NONE` / `cudagraph_mode: NONE`, so the most likely
explanation is that the fork's graph path is gated on the same compiled-mode
machinery eager switches off, and the env var only controls whether it *would* be
used. Setting it while eager is on is therefore inert — the warning is telling you
about a knob that cannot do anything in this configuration.

**Conclusion: not a lever, and the warning in the log is safe to ignore.**
Reaching it would mean dropping `--enforce-eager`, which is the configuration
that returns empty output (table above) — so XPU Graph is gated behind a known
correctness failure, not merely untested. Nothing to pursue unless a future image
either fixes compiled mode or decouples the two.

The remaining notes below are why it looked promising, kept so the reasoning is
auditable if a later image changes the picture.

#### Background

Read from the `0.26.0-b1` boot log, 2026-09-03. Three gates are logged, not one:

```
WARNING [vllm.py:1172] Enforce eager set, disabling torch.compile and CUDAGraphs.
                       This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
INFO    [vllm.py:1401] Cudagraph is disabled under eager mode
WARNING [xpu.py:285]   XPU Graph is disabled by environment variable,
                       please set VLLM_XPU_ENABLE_XPU_GRAPH=1 to enable it.
```

The first two are stock vLLM reacting to `--enforce-eager`. The third is the
fork's own code in `platforms/xpu.py`, and it attributes the disable to **the
environment variable**, not to eager — a parallel implementation to upstream's
CUDAGraph path, with its own switch. The same boot confirms the upstream path is
fully off: `'mode': CompilationMode.NONE`, `cudagraph_mode: CUDAGraphMode.NONE`,
`0.0 GiB for CUDAGraph memory`.

**What it does.** Level Zero / SYCL command-graph capture-and-replay, the XPU
analogue of CUDA graphs: the decode loop's kernel submissions are captured once
per shape and replayed as a single graph, removing per-launch host overhead. It
targets *launch-bound* decode, which is exactly single-stream gpt-oss-20b (3.6B
active params, many small kernels). It does nothing for prefill or for a
saturated batch, and captured graphs cost VRAM this config currently does not
spend.

**Why it looked worth testing.** The `xpu.py:285` wording implied the variable
could be set **while keeping `--enforce-eager`** — correctness guarantee retained,
only graph replay added — which would have made it the one remaining
decode-*speed* lever besides MTP (fp8 KV and `--kv-cache-memory` buy concurrency,
not speed). The measurement above shows the wording was misleading: the variable
is accepted under eager and does nothing.

**What the test cost, for calibration:** two container recreates, ~40 s boot each,
about 20 minutes end to end including baselines. Cheap enough to be worth
resolving even at low odds, which is the only reason it was run.

**Upstream documents none of it.** The variable appears neither in
[the pinned README](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md)
nor in [`main`'s](https://github.com/intel/llm-scaler/blob/main/vllm/README.md) —
both checked 2026-09-03. The only published mention is the `0.21.0-b1` release
note ("experimentally support XPU graph"); the string itself lives in the image,
in `platforms/xpu.py`. So there is nothing to link, and no documented values
beyond `=1`.

**The procedure used, reusable for any scaler flag on a live engine** —
correctness before speed, because this engine's known failure mode is silent:

1. **Baseline on the running container first, with no downtime:** boot-log pool
   line, `bench.sh 400` ×3, `bench.sh 200` ×3, `smoke.sh`, plus one
   `temperature: 0` reference generation saved to a file. Cross-boot comparisons
   are worth much less — the 2026-08-21 fp8 KV test lost ~2% to exactly that.
2. Change `environment:`, then `docker compose up -d` — it recreates in place, so
   no separate `stop` and less downtime.
3. Confirm the flag was actually *read* (here: the `xpu.py:285` warning
   disappearing), and re-read the `gpu_worker.py:857` budget line. A flag that
   changes no memory figure and adds no log line has probably not engaged.
4. `./smoke.sh`, then `diff` the reference generation. A tok/s number from a run
   returning empty or subtly wrong text looks excellent and means nothing.
5. `./bench.sh 400` and `./bench.sh 200`, discarding run 1 — the recreate is a
   **cold** start (§3 bench hygiene).
6. **Decide against a threshold fixed in advance** (this test used: keep only if
   ≥5% at both budgets with smoke passing), then either record the win here or
   revert the same session. An inert undocumented env var left set is a liability
   at the next image bump, when its semantics may change silently.
7. Record the outcome here with the image tag and date, pass or fail.

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
| `VLLM_XPU_ENABLE_XPU_GRAPH` | **Deliberately not set — tested and it does nothing under `--enforce-eager`** (§4). Undocumented upstream in both the pinned and `main` READMEs; exists only in the image and one release note. The `xpu.py:285` warning recommending it is **safe to ignore** and will appear on every boot. | Measured on `:0.26.0-b1`, 2026-09-03 |

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
- ~~Capture KV-pool size / max concurrency for `0.26.0-b1`~~ **done 2026-09-03**:
  183,314 tok / 1.40×, down from b3's 234,645 / 1.79× (§3).
- ~~Test `VLLM_XPU_ENABLE_XPU_GRAPH=1` under `--enforce-eager`~~ **done
  2026-09-03: no-op, reverted** (§4). MTP is now the only untried decode-*speed*
  lever.
- Consider `--kv-cache-memory` instead of util for pool sizing (§3): the engine
  reports 8.05 GiB available against 4.33 GiB in use, i.e. ~1.86× concurrency for
  free, no fp8 accuracy question. **This is now the best-value untested change on
  this engine** — it recovers the pool the `0.26.0-b1` upgrade cost and more.
  Leaves little slack at the top end, so try an intermediate value first.
- Re-verify the `VLLM_QUANTIZE_Q40_LIB` path on this image (§7) if `sym_int4` is
  ever used.
- `healthcheck.start_period` is still `7200s`, which was raised for the gemma-4
  experiment's ~52 GB download-and-quantise first boot. gpt-oss-20b only needs
  the ~1800s that covered its silent XPU cold start.

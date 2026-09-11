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
   below, so it cannot go stale here. What this file records is what was
   **measured on this hardware**, and where those findings differ from the docs.
2. **Every tested configuration names the full image tag and the date tested,**
   and says explicitly whether it has been re-tested on the current pin. A number
   without an image tag is not a result.

## Upstream documentation

Primary reference, pinned to the git tag matching the running image so it
cannot drift:
**[llm-scaler vLLM README @ `vllm-0.26.0-b2`](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md)**

⚠ **The b2 README is 116 lines *shorter* than b1's.** Diffed 2026-09-10: 116
lines removed, **zero added**. Deleted upstream: §2.11 Load Balancer,
§4.2 "Out-of-memory while online quantization", the oneAPI-sourcing note, and all
nine mentions of `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` (§7). b2's README is
byte-identical to `main`'s, so this is current upstream intent, not a stale tag
cut. Nothing was added — the supported-models table is unchanged from b1.

| Topic | Link |
|-------|------|
| Supported models table | [§3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#3-supported-models) |
| INT4 / FP8 online quantisation | [§2.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#22-int4-and-fp8-quantized-online-serving) |
| Pulling / running the container | [§1.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#13-pulling-and-running-the-vllm-docker-container) |
| Finding maximum context length | [§2.7](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#27-finding-maximum-context-length) |
| gemma-4 / diffusiongemma reference commands | [§3.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#33-reference-commands-for-running-gemma-4-models-and-diffusiongemma) |
| FP8 KV cache | [§3.5](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#35-fp8-kv-cache) |
| MTP (speculative decoding) | [§3.6](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#36-mtp-enable) |
| Performance tuning / NUMA | [§5](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#5-performance-tuning) |
| ~~OOM during online quantisation~~ | **§4.2 deleted at b2.** Last version that had it: [`vllm-0.26.0-b1` §4.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b1/vllm/README.md#42-out-of-memory-while-online-quantization) |

Other upstream sources:

| | |
|---|---|
| Latest docs (may be **ahead** of the pin here) | <https://github.com/intel/llm-scaler/blob/main/vllm/README.md> |
| Image release list | <https://github.com/intel/llm-scaler/blob/main/Releases.md> |
| FAQ | <https://github.com/intel/llm-scaler/blob/main/vllm/FAQ.md> |
| Known issues | <https://github.com/intel/llm-scaler/blob/main/vllm/KNOWN_ISSUES.md> |
| Published image tags | <https://hub.docker.com/r/intel/llm-scaler-vllm/tags> |
| vLLM on KV-cache quantisation | <https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html> |

---

## 1. Current pin

| | |
|---|---|
| Image | `intel/llm-scaler-vllm:0.26.0-b2` |
| Base | vLLM 0.26.0 (engine reports `0.26.1.dev0+g568afb3a1.d20260907`) |
| Model | `openai/gpt-oss-20b`, served as `gpt-oss-20b` |
| Context / util | `--max-model-len 131072` (128k) / `--gpu-memory-utilization 0.80` |
| Boot mode | `--enforce-eager` — **mandatory**, see §4 |
| Status | Boot-tested and validated 2026-09-10: `smoke.sh` ALL PASS, 85.6 tok/s, KV pool and VRAM budget byte-identical to b1 (§3) |

**Which engine owns `:8000`?** `curl -s localhost:8000/version`. The scaler
reports `0.26.1.dev0…`; the stock `vllm_xpu` image reports `0.21.0`. Useful
because three different stacks on this host can claim that port, and a benchmark
against the wrong one is worse than no benchmark.

⚠ **That check does not separate b1 from b2.** Both report the same fork commit
`g568afb3a1` — only the trailing build date differs (`d20260831` = b1,
`d20260907` = b2). To confirm *which build* is running, read the image tag:
`docker inspect -f '{{.Config.Image}}' vllm-scaler`.

---

## 2. Image pin provenance

Upstream says **do not use `:latest`**, and on this project `latest` is badly
stale — verified against the Docker Hub tag API, 2026-09-10:

| Tag | Pushed | Compressed | Digest |
|---|---|---|---|
| **`0.26.0-b2`** (current pin) | **2026-09-08** | **7.78 GB** | `sha256:52218ad8…` |
| `0.26.0-b1` (previous pin) | 2026-09-02 | 7.78 GB | `sha256:70e7eafd…` |
| `0.21.0-b3.1` | 2026-08-13 | 5.18 GB | `sha256:032916bd…` |
| `latest` | 2026-08-13 | 5.18 GB | `sha256:032916bd…` — **= `0.21.0-b3.1`** |

`latest` has not moved in two releases and still resolves to the b3.1-era image,
digest for digest. Check
[the tag list](https://hub.docker.com/r/intel/llm-scaler-vllm/tags) for ground
truth.

**`Releases.md` caught up at b2, but only on `main`.** At b1 it named
`0.21.0-b3.1` as "Latest Release" on both `main` and the release's own git tag.
Now `main`'s copy names `0.26.0-b2` and there is a real
[GitHub release page](https://github.com/intel/llm-scaler/releases/tag/vllm-0.26.0-b2)
(published 2026-09-09, marked pre-release) — but the copy *inside* the
`vllm-0.26.0-b2` tag still says `0.26.0-b1`. A tag's own `Releases.md` is always
one release behind; read [`main`'s](https://github.com/intel/llm-scaler/blob/main/Releases.md).

### What b2 changes — and why none of it reaches this config

b2 is a **bug-fix release on the same vLLM 0.26.0 base**, not a base jump. Its
[release notes](https://github.com/intel/llm-scaler/releases/tag/vllm-0.26.0-b2)
list five items; all four specific ones are in paths gpt-oss-20b does not use:

| Upstream fix | Reaches this config? |
|---|---|
| Fix prefix caching with MTP | **No** — prefix caching is on, but MTP is not (and cannot be, §9) |
| Improve MTP decoding performance | **No** — same reason |
| Fix incorrect output for `sym_int4` models | **No** — gpt-oss-20b is pre-quantised MXFP4, so `--quantization` is never passed |
| Fix block-FP8 output corruption during concurrent decoding | **No** — not a block-FP8 checkpoint, and `--kv-cache-dtype` is unset (§8.2) |
| "Bug fixes" | Unspecified |

Measurement agrees: b2 is **neutral on every axis tracked here** — same tok/s,
same TTFT, same KV pool, same VRAM budget, `smoke.sh` still ALL PASS (§3). So
take b2 as cheap insurance, not an upgrade, and **expect no measurable change**. It
does matter if this engine ever serves `sym_int4` or a block-FP8 checkpoint, both
of which had silent-wrong-output bugs until now.

The earlier jump from `0.21.0-b3` *was* a vLLM **base** change, 0.21.0 → 0.26.0.
**Side effect, still live:** the stock engine is on 0.21.0, so the two engines do
not share a base and a base-vs-scaler A/B is confounded by a five-minor engine
gap — the opposite of why the 0.21.0 bump was originally taken.

No release in this line contains a gpt-oss fix (the notes are all Qwen3.6 /
gemma-4 / Muse-Glimmer / diffusiongemma work), yet gpt-oss-20b got ~19–21%
faster at b1 — so that win came from the newer base, not the fork's own commits.

**Supported models:** read the live table at
[§3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#3-supported-models)
rather than a copy here. b2 adds **nothing** to it — the whole b1 → b2 README
diff is removals. The earlier b3 → b1 diff added only **3 rows**
(`Muse-Glimmer-30B`, `Qwen3.8-27B`, `Qwen3.8-27B-FP8`), none removed or changed,
and none of the three fits a single B60. gpt-oss-20b/120b remain in the MXFP4
column, so the served model stays supported.

---

## 3. Performance record

All figures are **eager** (compiled is not an option — §4), single-stream via
`./bench.sh`, on one Arc Pro B60 serving `openai/gpt-oss-20b`.

### Decode throughput

| Image (full tag) | Date tested | Context / util | `bench.sh 400` | `bench.sh 200` | TTFT |
|------------------|-------------|----------------|----------------|----------------|------|
| `intel/llm-scaler-vllm:0.14.0-b8.3.2` | 2026-07-03 | 64k / 0.75 | 80.8 tok/s | — | 109 ms |
| `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | 128k / 0.80 | 70.6 tok/s | 72.0 tok/s | ~81–92 ms |
| `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-02 | 128k / 0.80 | 85.6 tok/s | 86.0 tok/s | ~73 ms |
| **`intel/llm-scaler-vllm:0.26.0-b2`** | **2026-09-10** | 128k / 0.80 | **85.6 tok/s** | **86.1 tok/s** | **~72 ms** |

`0.26.0-b1` was **+21.2% over b3** and **+5.9% over b8.3.2** (the previous best),
so the ~10–12.6% regression that made b3 a hard sell is **gone**. Raw runs:
85.6 / 84.3 / 85.6 @400 and 86.0 / 86.0 / 86.0 @200.

**b2 = b1, to the last digit.** Raw runs 85.6 / 85.6 / 85.6 @400 (run 1 cold at
84.3, discarded) and 86.1 / 86.1 @200 (run 1 cold at 86.0). The +0.1 @200 is one
extra chunk on a 197-chunk stream — noise, not a win. Zero change is the
*expected* result given b2's release notes (§2), and it is the useful kind of
result: it says the bug-fix release did not quietly cost anything either.

⚠ The b8.3.2 row was measured at a **different context/util profile** (64k/0.75
vs 128k/0.80), so treat b8.3.2-vs-later as indicative, not matched. The
b3-vs-`0.26.0-b1` comparison *is* matched on every axis.

### Boot-time memory figures

| Image (full tag) | Date tested | Weights | KV memory | KV pool | Max concurrency @131,072 |
|------------------|-------------|---------|-----------|---------|--------------------------|
| `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | 12.87 GiB / 7.68 s | 5.46 GiB | 234,645 tok | 1.79× |
| `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-03 | 12.87 GiB / 7.28 s | **4.33 GiB** | **183,314 tok** | **1.40×** |
| `intel/llm-scaler-vllm:0.26.0-b2` | 2026-09-10 | 12.87 GiB / 7.65 s | **4.33 GiB** | **183,314 tok** | **1.40×** |

b2's boot log reproduces b1's budget line for line — same weights, same peak
activation (0.80 GiB), same non-torch (1.12 GiB), same 0.00 GiB CUDAGraph, same
pool to the token. The only digits that moved are load time (7.65 s vs 7.28 s,
disk noise) and the advised "fully utilize" byte value, `8647520256` vs b1's
`8647532544` — 12 KiB apart, i.e. 8.05 GiB either way.

For context, the stock `intel/vllm:0.21.0-ubuntu24.04` engine reports a 221k pool
/ 1.69× at the same profile, so the fork sized its KV pool slightly *larger* on
the same base. The b3 boot also confirmed 1 XPU enumerated,
`Intel(R) Arc(TM) Pro B60 Graphics`, 23.9 GiB.

The VRAM budget, read from the boot log (`gpu_worker.py:857`) — identical on b1
(2026-09-03) and b2 (2026-09-10):

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
(4.19 GiB) to fit into requested memory, or `--kv-cache-memory=8647520256`
(8.05 GiB) to fully utilize gpu memory.
```

(b2's byte value, read 2026-09-10; b1 printed `8647532544`. Re-read this line
after any image bump rather than carrying a byte value forward — it comes from
*that* build's memory profile.)

8.05 GiB of KV is 1.86× today's 4.33 GiB — the same order of capacity gain as
`--kv-cache-dtype fp8` (§8.2), without fp8's accuracy question. Reaching it needs
util ≈ 0.955, or better, `--kv-cache-memory` set explicitly, which sizes the pool
directly instead of leaving it as whatever the util budget does not spend.
**Untested.** It would leave only ~0.15 GiB of the card unallocated, so an
intermediate value is the sane first attempt. Note that the 0.86 util ceiling
carried over from the stock engine does **not** bind here: eager reserves no
Inductor or CUDAGraph buffers, and the 0.00 GiB row above is the proof (§4).

`GET /metrics` does **not** expose pool size on this build (only
`kv_cache_usage_perc`), so those figures always need a boot-log read — run this
from the repo root after any image bump:

```bash
docker compose -f scaler/compose.yaml logs \
  | grep -E "KV cache size|Maximum concurrency|Model loading took|gpu_worker.py:857"
```

### Correctness

| Image (full tag) | Date tested | `smoke.sh` | Notes |
|------------------|-------------|------------|-------|
| `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | ALL PASS | `message.reasoning` 835 chars |
| `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-02 | ALL PASS | `message.reasoning` 1082 chars; confirms the inherited `--reasoning-parser` / `--tool-call-parser` names survived the base jump |
| `intel/llm-scaler-vllm:0.26.0-b2` | 2026-09-10 | ALL PASS | `message.reasoning` 342 chars — a *third* of b1's on the identical `temperature: 0` prompt |

`smoke.sh` checks four things: served model name, non-empty content, a populated
`message.reasoning`, and a `get_weather` tool_call.

⚠ **Reasoning length is not a stable fingerprint across builds.** b3 gave 835
chars, b1 1082, b2 342 — same prompt, same `temperature: 0`. Determinism
*within* a build is evidenced (b1 reproduced 1082 exactly across the XPU-graph
test's two boots, §4); across builds it does not hold, most likely because one
differing kernel result flips an argmax and reroutes the whole chain of thought —
that mechanism is inferred, not measured. Either way, treat the number as
"populated / not populated". For a real regression check, `diff` a saved
reference generation on the **same** image (§4, step 4).

### Bench hygiene — each of these has already produced a wrong number

- **`bench.sh` reports `total_chunks / decode_time`, not true tokens/s.** A
  longer budget can therefore score *lower* on the same engine: b3 gave 72.0
  @200 but 70.6 @400. **Always compare at equal `max_tokens`.** The b8.3.2 80.8
  figure was `./bench.sh 400`, which is why the matched b3 gap is ~12.6%, not
  the ~10% first quoted.
- **Always discard run 1 after a cold start.** On b3 the first request after
  startup measured 57.5 tok/s — warm-up, not the engine's speed.
- `0.26.0-b1`/`-b2` are nearly budget-flat (85.6 @400 vs 86.1 @200), so this trap
  matters mostly when comparing *across* images.
- **A long-idle container is still a cold first run.** b2 had been up ~25 min,
  healthy and idle, and its first `bench.sh 400` still came in low (84.3, TTFT
  90 ms) before settling at 85.6 / 72 ms. Discard run 1 after *idleness*, not
  just after a boot.

---

## 4. Boot mode: `--enforce-eager` is mandatory, not a safe default

| Configuration | Image (full tag) | Date tested | Result |
|---------------|------------------|-------------|--------|
| `--enforce-eager` (current) | `intel/llm-scaler-vllm:0.26.0-b2` | 2026-09-10 | Correct output, 85.6 tok/s |
| `--enforce-eager` | `intel/llm-scaler-vllm:0.26.0-b1` | 2026-09-02 | Correct output, 85.6 tok/s |
| **no** `--enforce-eager` (torch.compile ON) | `intel/llm-scaler-vllm:0.14.0-b8.3.2` | 2026-07-03 | **Rejected — silently empty output** |
| no `--enforce-eager` | `:0.21.0-b3`, `:0.26.0-b1`, `:0.26.0-b2` | **not re-tested** | — |

Dropping `--enforce-eager` boots and compiles fine (~80 s) but then generates
**empty output** — `completion_tokens` increment while both `content` and
`reasoning` come back null/blank. A silent correctness failure with no crash or
NaN in the logs. Inductor's compiled graph is broken for this MXFP4 MoE on XPU;
the fork's custom kernels only produce valid output in eager.

Consequences:

- Every tok/s figure in §3 is an eager number and is the engine's **real** speed
  — not an under-statement, because compiled is not an option here.
- Eager disables torch.compile, so there is no compile-buffer growth and util is
  **not** constrained by Inductor buffers on this engine. 0.85 would be safe;
  0.80 is kept to match the stock engine (raised from 0.75 on 2026-07-08, once the
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

**Carried over to `0.26.0-b2` unretested, and that is defensible:** b2's boot log
prints the same three gate lines verbatim (below), the variable is still absent
from b2's README, and b2 changed nothing measurable (§2/§3). Nothing in the
release notes touches the graph path.

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

Read from the `0.26.0-b1` boot log, 2026-09-03, and unchanged in b2's
(re-read 2026-09-10). Three gates are logged, not one:

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
only graph replay added — which would have made it the **last** decode-*speed*
lever on this engine (fp8 KV and `--kv-cache-memory` buy concurrency, not speed;
MTP turned out not to be available for gpt-oss-20b at all, §9). The measurement
above shows the wording was misleading: the variable is accepted under eager and
does nothing.

**What the test cost, for calibration:** two container recreates, ~40 s boot each,
about 20 minutes end to end including baselines. Cheap enough to be worth
resolving even at low odds, which is the only reason it was run.

**Upstream documents none of it.** The variable appears neither in
[the pinned README](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md)
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
- **`vllm-scaler-cache` is separate** from `vllm_xpu`'s `vllm-cache` because
  compiled kernels are image-version-specific.

  **Earlier advice to clear it on every image bump is over-cautious — measured
  2026-09-10 and corrected.** Under `--enforce-eager` nothing is ever compiled,
  so the volume holds **16 KB**: two `modelinfos/*.json` files (gpt-oss and
  Gemma4 architecture metadata) and no kernels at all. Each carries a `hash` of
  the model source file, so a changed image invalidates its own entries —
  proven by b2 booting correctly on b1-era files. Inventory it yourself rather
  than trusting either claim:

  ```bash
  docker run --rm -v llm_vllm-scaler-cache:/v:ro alpine sh -c 'du -sh /v; find /v -type f'
  ```

  If `--enforce-eager` is ever dropped (§4), this reasoning expires: compiled
  artifacts *would* land here and *would* need clearing on an image bump.

---

## 7. Environment variables

`ZES_ENABLE_SYSMAN`, `SYCL_CACHE_PERSISTENT`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn` and `shm_size: 32g` follow upstream's
recommendations — see [§1.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#13-pulling-and-running-the-vllm-docker-container).
`VLLM_QUANTIZE_Q40_LIB` and `VLLM_ALLOW_LONG_MAX_MODEL_LEN` are the
online-quantisation set, documented at
[§2.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#22-int4-and-fp8-quantized-online-serving);
both are **inert for gpt-oss-20b**, which is pre-quantised MXFP4.

Only the parts where **behaviour here differs from the docs** are recorded:

| Variable | Finding | Verified |
|----------|-------------|----------|
| `VLLM_QUANTIZE_Q40_LIB` | The path in Intel's README (`/usr/local/lib/python3.12/dist-packages/…`) is **wrong for this image line** and crash-loops the engine with "cannot open shared object file". The working path, which the compose file sets, is `/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so`. | `:0.21.0-b3`, 2026-08-12. ✅ **Re-verified on `:0.26.0-b1` and `:0.26.0-b2`, 2026-09-10** — same path, 15,216 bytes, and the variable is still live in `vllm/envs.py` + `quantization/sym_int4.py`. b2's README still prints the wrong `dist-packages` path. |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | ☠ **Dead — removed from the compose file 2026-09-10.** Zero references anywhere in either 0.26.0 image (searched `/opt/venv`, `/llm`, `/root/.bashrc`, `/usr/local/bin`), and upstream deleted all nine mentions plus §4.2 from the README at b2. It was a live knob on the 0.21.0 line; the 0.26.0 base dropped it, so it was already a silent no-op under b1. | Both images searched 2026-09-10 |
| `HF_TOKEN` | Not needed for current models: `google/gemma-4-26B-A4B-it` is `gated: false`, apache-2.0 (unlike gemma 2/3). Kept as value-only passthrough so the file carries no secret and empty means anonymous. | HF API, 2026-08-12 |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | **Deliberately not set — tested and it does nothing under `--enforce-eager`** (§4). Undocumented upstream in both the pinned and `main` READMEs; exists only in the image and one release note. The `xpu.py:285` warning recommending it is **safe to ignore** and will appear on every boot. | Measured on `:0.26.0-b1`, 2026-09-03; warning re-confirmed on b2 |

**The re-check that found the dead variable, reusable on any image bump** — an
env var set on the wrong image line is either a crash or, worse, a silent no-op:

```bash
docker run --rm --entrypoint /bin/bash intel/llm-scaler-vllm:0.26.0-b2 -lc '
  ls -la /opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so
  grep -rl VLLM_QUANTIZE_Q40_LIB /opt/venv/lib/python3.12/site-packages/vllm | head'
```

Swap in each variable the compose file sets. No hit in the image means no code
reads it.

---

## 8. Tested and rejected

### 8.1 `gemma-4-26B-A4B-it` on a single B60 — blocked

| Route attempted | Image (full tag) | Date tested | Result |
|-----------------|------------------|-------------|--------|
| `Intel/gemma-4-26B-A4B-it-int4-AutoRound` + `--quantization gptq` / `moe_wna16` | `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | ❌ `NotImplementedError: No Unquantized MoE backend…` — wall 1 |
| `google/gemma-4-26B-A4B-it` + `--quantization sym_int4` (upstream's documented route) | `intel/llm-scaler-vllm:0.21.0-b3` | 2026-08-12 | ❌ `RuntimeError: unable to mmap 49907246508 bytes` — wall 2, host RAM |
| either route | `intel/llm-scaler-vllm:0.26.0-b1`, `:0.26.0-b2` | **not attempted** | — |

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
([§2.2](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#22-int4-and-fp8-quantized-online-serving))
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

**Status on `0.26.0-b1` and `-b2`:** the upstream support row for this model is
byte-identical to b3 — online fp8/int4 only, no offline column, and still no
reference command (b2 added a "MTP supported" note to the row, nothing else).
Upstream publishes gemma-4 commands only for
`gemma-4-12B-it` (TP=1) and `diffusiongemma-26B-A4B-it` (TP=2), never for
`gemma-4-26B-A4B-it`; see
[§3.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#33-reference-commands-for-running-gemma-4-models-and-diffusiongemma)
for the current commands rather than trusting a copy here.

One new lever, **low confidence**: vLLM 0.26.0's registry adds `auto_gptq` and
`auto_awq` as first-class methods (`AutoGPTQConfig`/`AutoAWQConfig`), neither of
which existed in 0.21.0. There is still no `auto_round.py` upstream, and the b3
failure was in MoE *construction*, so this only helps if the fork now ships a
quantised-Gemma4 MoE XPU kernel. Not pursued.

`gemma-4-12B-it` remains the plausible single-B60 gemma-4 path — dense and TP=1
in upstream's own reference command ([§3.3](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#33-reference-commands-for-running-gemma-4-models-and-diffusiongemma)).
Untested here.

### 8.2 `--kv-cache-dtype fp8` — worked, then reverted by decision

Tested on `intel/llm-scaler-vllm:0.21.0-b3`, **2026-08-21**. It worked and cost
nothing measurable — but the flag is **not currently set**, and has **not** been
re-tested on the current pin.

| Metric | b3 without the flag | b3 **with** `--kv-cache-dtype fp8` | `0.26.0-b1` / `-b2` |
|--------|---------------------|------------------------------------|---------------------|
| GPU KV cache size | 234,645 tok | **469,354 tok** (exactly 2.0×) | not re-tested |
| Max concurrency @131,072 | 1.79× | **3.58×** | not re-tested |
| KV memory | 5.46 GiB | 5.46 GiB | not re-tested |
| `bench.sh 200` | 72.0 tok/s | 72.0 tok/s (**unchanged**) | not re-tested |
| `bench.sh 400` | 70.6 tok/s | 70.6 tok/s | not re-tested |
| `smoke.sh` | ALL PASS | ALL PASS | not re-tested |
| Attention backend | `FLASH_ATTN` | `FLASH_ATTN` (no silent Triton fallback) | `FLASH_ATTN`, FA2 — re-confirmed in b2's boot log |

Both b3 columns are from the same image, days apart; the flag was added,
measured, then removed. Removed because it did not touch the ~10% decode
regression being chased at the time — it buys **concurrency, not speed**. That
regression is gone as of `0.26.0-b1`, so the only remaining motive for fp8 KV is
long-context concurrency.

⚠ **b2 changes the risk calculus slightly:** its release notes fix "block-FP8
output corruption during concurrent decoding". That is weight quantisation, not
`--kv-cache-dtype`, so it is not a fix *for* this flag — but it is a reminder
that the fork's fp8 paths were carrying concurrency bugs through the b1 pin. If
fp8 KV is ever re-added, validate it under actual concurrent load, not just
`bench.sh`'s single stream.

If re-added, validate on a **real long prompt** first: vLLM warns that fp8 KV may
cause accuracy loss without a proper scaling factor, and `smoke.sh`'s four
shallow checks do not clear that. Upstream guidance:
[vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)
and llm-scaler [§3.5](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#35-fp8-kv-cache).

### 8.3 Three dead ends, so nobody re-treads them

| Idea | Verdict | Basis |
|------|---------|-------|
| Enable "flash attention" | **Already on — not a lever.** There is no `--flash-attn` flag in vLLM (that is llama.cpp's `-fa`). `platforms/xpu.py` already selects `FLASH_ATTN` (FlashAttention 2, KV block size 64, NHD layout) with no flag set here. | Read from `platforms/xpu.py` in the running b3 container, 2026-08-21 |
| Switch to `TRITON_ATTN` | **Downgrade.** The only other non-MLA option, and that same source file calls it the old fallback; FA2 measured ~19% faster decode TPOT than Triton on B-series. | same |
| `--kv-cache-dtype turboquant_*` (`k8v4` / `4bit_nc` / `k3v4_nc` / `3bit_nc`) | **Ruled out.** Routes to a separate TURBOQUANT backend and costs 20–60% throughput, in exchange for capacity this deployment does not need. | [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) |

---

## 9. Open items

- **`--kv-cache-memory` instead of util for pool sizing (§3) — the priority.**
  The engine reports 8.05 GiB available against 4.33 GiB in use, i.e. ~1.86×
  concurrency for free, no fp8 accuracy question. The same lever is now
  **measured on the sibling engine**: it took upstream's XPU image from 1.29× to
  2.59× with decode unchanged
  ([`UPSTREAM_VLLM_NOTES.md`](UPSTREAM_VLLM_NOTES.md) §10.3), so this is no
  longer a speculative idea — it is a proven lever this engine has not been
  given. It leaves little slack at the top end, so try an intermediate value
  first, and re-read the advised byte value from *this* build's boot log (§3).
- Bench the **stock** engine to close the base-vs-scaler comparison — the
  original point of keeping both folders.
- `healthcheck.start_period` is still `7200s`, which was raised for the gemma-4
  experiment's ~52 GB download-and-quantise first boot. gpt-oss-20b only needs
  the ~1800s that covered its silent XPU cold start.

**Closed:**

- ~~Capture KV-pool size / max concurrency for `0.26.0-b1`~~ **done 2026-09-03**:
  183,314 tok / 1.40×, down from b3's 234,645 / 1.79× (§3). Unchanged on b2.
- ~~Test `VLLM_XPU_ENABLE_XPU_GRAPH=1` under `--enforce-eager`~~ **done
  2026-09-03: no-op, reverted** (§4).
- ~~Re-verify the `VLLM_QUANTIZE_Q40_LIB` path~~ **done 2026-09-10**: correct on
  both 0.26.0 images (§7). The same sweep found
  `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` dead and removed it from the compose file.
- ~~MTP as the remaining decode-*speed* lever~~ **not available on this engine
  — the earlier note was wrong.** Upstream's
  [§3.6](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md#36-mtp-enable)
  offers exactly two methods, `qwen3_5_mtp` (models with native MTP layers) and
  `gemma4_mtp` (needs a matching assistant checkpoint), verified only on
  `Qwen3.6-27B`, `Qwen3.6-35B-A3B`, `gemma-4-26B-A4B-it` and `gemma-4-31B-it`.
  gpt-oss-20b has neither native MTP nor a published assistant checkpoint, and
  none of those four models fits one B60 (§8.1 for gemma-4). **So b2's two MTP
  fixes are inert here, and there is no untried decode-speed lever left on this
  engine for this model** — the remaining levers all buy concurrency.

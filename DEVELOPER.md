# Local LLM stack — developer notes (the *why*)

Why the config in the engine compose files (`vllm_xpu/compose.yaml` and
`scaler/compose.yaml`) is the way it is. For how to *operate* the
stack see [README.md](README.md); for a configuration overview see [INTEL_ARC_B60.md](INTEL_ARC_B60.md).

All values here are empirical on the **Intel Arc Pro B60 (22.71 GiB usable)**. The
stack upgraded from `intel/vllm:0.17.0-xpu` to **`intel/vllm:0.21.0-ubuntu24.04`**;
the gpt-oss-20b boot and the `0.75` util ceiling were re-validated on 0.21.0, but
the other 0.17.0-era measurements below (Qwen3 context caps, the 0.86-OOM edge, the
reasoning-effort latencies) have **not** been re-run on 0.21.0. Nothing here is
portable to other cards or images without re-checking. The host is **Linux only** —
the Intel `xe` GPU driver is Linux-specific, so Windows and macOS are out of scope.

---

## Why `--gpu-memory-utilization 0.75` (not 0.95)

On this XPU build, `--gpu-memory-utilization` sizes the **weights + KV pool** but
does **NOT** cap torch.compile/Inductor kernel + workspace buffers, which keep
growing as new request shapes get compiled.

At **0.86** the card filled to 22.67 / 22.71 GiB (~0.04 GiB free) → OOM-on-the-edge,
instability, and 504s. **0.75** (~17 GiB: ~13.7 GiB weights + ~3.3 GiB KV pool)
leaves ~2.5 GiB of real headroom for that uncapped compile growth.

Re-validated on `0.21.0-ubuntu24.04` (compiled, production flags): clean boot, KV
pool ~3.96 GiB, no OOM at 0.75 — the ceiling carries over unchanged. The 0.86-OOM
edge above was characterised on `0.17.0-xpu` and not re-tested on 0.21.0.

**The trap:** util looks like a headroom dial but it doesn't account for the
compile buffers. To grow capacity, raise `--max-model-len` and re-check real
VRAM — **never** just bump util, or you'll OOM on the edge again.

## Why 64k context fits

gpt-oss-20b is an MoE with ~13.7 GiB MXFP4 weights (~3.6B active params). It's
natively 128k (YaRN, `max_position_embeddings=131072`), but `--max-model-len
65536` keeps the reserved KV pool + activation buffers small. gpt-oss's
alternating sliding-window(128) + full-attention layers halve per-request KV
cost, so the ~3.3 GiB pool holds 64k with concurrency to spare.

## Sizing `--max-model-len`

vLLM does a KV-cache pre-check at startup. If `max-model-len × KV-per-token`
doesn't fit in the VRAM left after weights + compile artifacts, startup fails
with an explicit error ("*the model's max seq len … is larger than the maximum
number of tokens that can be stored in KV cache*").

Methodology: pick an ambitious target, drop to the next round value if the
pre-check rejects. Don't compute it analytically — compile overhead isn't
predictable from outside.

Known-good empirical values on the B60:

| Model | Weights (loaded) | Working `--max-model-len` | Notes |
|-------|------------------|----------------------------|-------|
| gpt-oss-20b | ~13.7 GiB | **65536** (64k) | At 0.75 util; the value shipped in `vllm_xpu/compose.yaml` |
| Qwen3-32B-AWQ | 18.14 GiB | **7168** | 12k and 10k both failed the pre-check |

*Weights here are the loaded figure vLLM reports at startup (GiB); the ≈GB
on-disk cache sizes in README/INTEL_ARC_B60 are the same weights in GB units
(18.14 GiB ≈ 19 GB).*

**To go bigger later:** raise `--max-model-len` AND re-measure real VRAM
headroom. Drop to 32k/16k if a future swap's pre-check rejects at startup.

---

## Quantisation on the B60

- **MXFP4 is the only viable format for gpt-oss.** Its weights are natively
  MXFP4; loading as BF16 inflates to ~40 GB and won't fit 24 GB. Intel's
  container ships MXFP4 kernels for gpt-oss specifically. If MXFP4 ever fails to
  load on a newer image, fall back to `intel/vllm:0.10.2-xpu` (the version Intel
  publicly benchmarked) — do **not** try BF16, it doesn't fit.
- **Qwen: AWQ is the working path.** The official `Qwen/*-FP8` weights are
  blocked by an upstream vLLM XPU bug (`RMSNormQuantFusionPass` NameError). Each
  Qwen swap-back also means switching `--reasoning-parser` to `qwen3` (hybrid
  thinking; `/no_think` disables) and lowering `--max-model-len`.

## Reasoning-effort lever (gpt-oss)

Effort is a top-level request field, `reasoning_effort: low|medium|high`
(default `medium`). It's a **quality/latency** lever, not a throughput lever:

- `low` ≈ 307 ms TTFT-to-content — fastest to a visible answer.
- `high` can **starve content** if `max_tokens` is too low (reasoning consumes
  the budget before any content is emitted). Push `max_tokens` up for high
  effort on non-trivial prompts.

## Image / version notes

- The stack runs **`intel/vllm:0.21.0-ubuntu24.04`** (reports vLLM
  `v0.21.1.dev17+g0a4756bb5`; the `dev` suffix is an scm artifact). The tag scheme
  dropped the `-xpu` suffix of older images, but it **is** the Intel Arc/XPU build
  — `device_config=xpu` and torch.compile runs on the B60, verified by booting
  gpt-oss-20b on it.
- **Device passthrough differs from `0.17.0-xpu`:** 0.21.0 requires the whole
  `/dev/dri` **plus** a `/dev/dri/by-path:ro` mount (oneCCL enumerates via
  `by-path` on warm-up) or it won't boot. Details in `vllm_xpu/compose.yaml` and the
  README's *Upgrading the vLLM image*.
- **Gemma 4 arches are now registered** (`gemma4` / `gemma4_mm`) — unlike
  `0.17.0-xpu`, which topped out at Gemma3n. That clears the *architecture* gate,
  but running Gemma 4 on the B60 is still unproven here (XPU quant-kernel gaps), so
  this stack stays on gpt-oss-20b. The `qwen3` and `openai_gptoss` reasoning parsers
  are present as before.
- Reasoning trace field is still `message.reasoning`, not `reasoning_content`
  (re-verified on 0.21.0) — see [README.md](README.md) for the consumer-parsing
  implication.
- Predecessor: `0.17.0-xpu` was a frozen release-tag build (reported vLLM
  `0.1.dev14456`) that topped out at Gemma3n — kept here for upgrade context.

---

## Choosing the inference engine: base vs llm-scaler

Two interchangeable engine images serve the same gpt-oss-20b on the same
`:8000`, so either can be production — one at a time (single GPU). Each has its
own folder: `vllm_xpu/compose.yaml` (stock `intel/vllm`, the default)
and `scaler/compose.yaml` (Intel's B-series-optimised `llm-scaler-vllm`
fork). The operator swap/run procedure is in the README.

**Why compare:** measure whether the `llm-scaler` fork decodes gpt-oss-20b
faster than the stock image. The single-stream decode baseline of **~60 tok/s**
on the B60 (via `bench.sh`) was measured on `0.17.0-xpu`; re-baseline on the
current `0.21.0-ubuntu24.04` stock image before comparing — that's the yardstick.
Note the stock image has since jumped `0.17`→`0.21`, so the fork (built on an
older vLLM base) is no longer strictly newer than what it's being compared to.

**Why two folders, not a compose profile:** one GPU (~22.7 GiB) and gpt-oss-20b
needs ~17 GiB, so the two engines can't coexist (~31 GiB = OOM). A separate
folder per engine means every `up` must target an engine's folder (cd into it,
or `-f` its `compose.yaml`), so you can't start both by accident and "which
engine is prod" is always explicit.

**Image:** pinned to `intel/llm-scaler-vllm:0.14.0-b8.3.2` (the current build;
the fork's docs warn against `:latest`). b8.3.2 vs the prior b8.3.1 is only a
Qwen3.5/3.6-27B accuracy fix — no gpt-oss-20b impact — but it's the right base
for a first benchmark.

**`--enforce-eager` caveat:** the staged config boots with `--enforce-eager`,
which (a) disables torch.compile — removing the uncapped Inductor buffer growth
that forced 0.75 util on the stock image, so a higher util would be safe here
(we keep 0.75 to match the base vLLM engine) — and (b) gives a clean first
boot. **But eager mode is slower than compiled**, so it
under-states the scaler's real speed. Once it boots clean, drop
`--enforce-eager` and re-bench for the true number (watch VRAM; back off util if
it edges toward OOM). gpt-oss-20b is MXFP4 (pre-quantised) — do **not** pass
`--quantization`. The fork inherits upstream's parser flag names; if it renamed
them the server fails fast at startup with a clear arg error.

---

## Hardware & model rationale

Context for future hardware or model swaps:

- **B70 vs B60** — the Arc Pro B70 (32 GB) gives roughly 1.3× decode / 1.85×
  prefill plus context headroom over the B60, but does **not** unlock Gemma 4
  (that's software-gated, not a VRAM limit).
- **Multi-GPU** — on a consumer board a second card typically only gets a
  chipset x4 link, so don't tensor-/pipeline-parallel across cards; run each card
  as an independent engine instead.
- **Model freshness** — gpt-oss-20b's knowledge cutoff is mid-2024. Fresher fast
  MoEs (e.g. Qwen3.5/3.6-35B-A3B AWQ ≈ 24 GB) don't fit the B60's ~22.7 GiB
  usable, so freshness is better addressed with RAG than with a model swap on
  this card.

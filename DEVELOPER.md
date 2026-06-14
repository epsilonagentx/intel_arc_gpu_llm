# Local LLM stack — developer notes (the *why*)

Why the config in `docker-compose.yml` is the way it is. For how to *operate* the
stack see [README.md](README.md); for a configuration overview see [INTEL_ARC_B60.md](INTEL_ARC_B60.md).

All values here are empirical on the **Intel Arc Pro B60 (22.71 GiB usable)** with
`intel/vllm:0.17.0-xpu`. They are not portable to other cards or images without
re-checking.

---

## Why `--gpu-memory-utilization 0.75` (not 0.95)

On this XPU build, `--gpu-memory-utilization` sizes the **weights + KV pool** but
does **NOT** cap torch.compile/Inductor kernel + workspace buffers, which keep
growing as new request shapes get compiled.

At **0.86** the card filled to 22.67 / 22.71 GiB (~0.04 GiB free) → OOM-on-the-edge,
instability, and 504s. **0.75** (~17 GiB: ~13.7 GiB weights + ~3.3 GiB KV pool)
leaves ~2.5 GiB of real headroom for that uncapped compile growth.

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
| gpt-oss-20b | ~13.7 GiB | **65536** (64k) | At 0.75 util; the value shipped in `docker-compose.yml` |
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

- `intel/vllm:0.17.0-xpu` reports vLLM `0.1.dev14456`, but that dev string is an
  scm artifact — it's a **frozen release-tag build**, not rolling `main`.
- Model support tops out at **Gemma3n**; **Gemma 4 is not supported** on this
  image. The `qwen3` and `openai_gptoss` reasoning parsers are both present.
- Reasoning trace field is `message.reasoning`, not `reasoning_content` — see
  [README.md](README.md) for the consumer-parsing implication.

---

## llm-scaler A/B benchmark (design)

**Goal:** measure whether Intel's B-series-optimised `llm-scaler-vllm` fork
decodes gpt-oss-20b faster than the stock `intel/vllm:0.17.0-xpu`. Single-stream
decode baseline on stock is **~60 tok/s** on the B60 (measured single-stream via
`bench.sh`) — that's the yardstick.

**Why it's gated behind the `scaler` compose profile:** there is one GPU
(~22.7 GiB) and gpt-oss-20b needs ~17 GiB, so the scaler and the production
`vllm` service can't coexist (~31 GiB = OOM). The profile guarantees the scaler
never starts on a bare `docker compose up` and never disturbs the running
service. The operator run procedure is in the README.

**Image:** use the pinned beta tag `intel/llm-scaler-vllm:0.14.0-b8.3.1` — the
fork's docs warn against `:latest`.

**`--enforce-eager` caveat:** the staged config boots with `--enforce-eager`,
which (a) disables torch.compile — removing the uncapped Inductor buffer growth
that forced 0.75 util on the stock image, so 0.85 util is safe there — and (b)
gives a clean first boot. **But eager mode is slower than compiled**, so it
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

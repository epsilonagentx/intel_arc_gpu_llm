# Local LLM stack — why it is configured this way

Why the config in the engine compose files is the way it is. This file covers
the two **gpt-oss-20b** engines, `vllm_xpu/compose.yaml` and
`scaler/compose.yaml`; the third engine keeps its rationale in its own README,
[vllm_openai_xpu/README.md](vllm_openai_xpu/README.md). For how to *operate* the stack
see [README.md](README.md); for a configuration overview see
[INTEL_ARC_B60.md](INTEL_ARC_B60.md).

All values here are empirical on the **Intel Arc Pro B60 (22.71 GiB usable)**. The
stack upgraded from `intel/vllm:0.17.0-xpu` to **`intel/vllm:0.21.0-ubuntu24.04`**;
the gpt-oss-20b boot and the util ceiling were re-validated on 0.21.0, but
the other 0.17.0-era measurements below (Qwen3 context caps, the 0.86-OOM edge, the
reasoning-effort latencies) have **not** been re-run on 0.21.0. Nothing here is
portable to other cards or images without re-checking. The host is **Linux only** —
the Intel `xe` GPU driver is Linux-specific, so Windows and macOS are out of scope.

---

## Why `--gpu-memory-utilization 0.80` (not 0.95)

On this XPU build, `--gpu-memory-utilization` sizes the **weights + KV pool** but
does **NOT** cap torch.compile/Inductor kernel + workspace buffers, which keep
growing as new request shapes get compiled.

At **0.86** the card filled to 22.67 / 22.71 GiB (~0.04 GiB free) → OOM-on-the-edge,
instability, and 504s. That remains the hard ceiling. The shipped value is
**0.80**, which became safe once the displays were moved off the B60 onto the
iGPU — before that, 0.75 was the limit, because a desktop session was also
holding VRAM and a freeze was possible.

**The trap:** util looks like a headroom dial but it doesn't account for the
compile buffers. To grow capacity, raise `--max-model-len` and re-check real
VRAM — **never** just bump util, or you'll OOM on the edge again.

**`scaler/` and `vllm_openai_xpu/` size the KV pool explicitly** with
`--kv-cache-memory-bytes` instead of letting util decide it. That value is
absolute and **OOMs rather than shrinking**, so it is model- and runner-specific;
see [scaler/README.md](scaler/README.md) and
[vllm_openai_xpu/README.md](vllm_openai_xpu/README.md), *--kv-cache-memory-bytes*.

## Why 128k context fits

gpt-oss-20b is an MoE with 12.87 GiB of MXFP4 weights (~3.6B active params), natively
128k (YaRN, `max_position_embeddings=131072`), and the engines now ship the full
**131072**. What makes it affordable is the attention geometry: gpt-oss alternates
sliding-window(128) and full-attention layers, so per-request KV is roughly halved
against a dense model — ~24.8 KiB/token. With the explicit ~8 GiB pool that gives
~340k tokens of KV, i.e. **2.6× concurrency at 128k**.

The 64k/0.75 profile documented here previously was the pre-2026-07 baseline,
from when the B60 still drove displays.

---

## Quantization on the B60

- **MXFP4 is the only viable format for gpt-oss.** Its weights are natively
  MXFP4; loading as BF16 inflates to ~40 GB and won't fit 24 GB. Intel's
  container ships MXFP4 kernels for gpt-oss specifically. If MXFP4 ever fails to
  load on a newer image, fall back to `intel/vllm:0.10.2-xpu` (the version Intel
  publicly benchmarked) — do **not** try BF16, it doesn't fit.
- **Qwen: AWQ is the working path.** The official `Qwen/*-FP8` weights are
  blocked by an upstream vLLM XPU bug (`RMSNormQuantFusionPass` NameError). Each
  Qwen swap-back also means switching `--reasoning-parser` to `qwen3` (hybrid
  thinking; `/no_think` disables) and lowering `--max-model-len`.
- **gemma-4: offline int4, group-32 only.** It runs on the `vllm_openai_xpu`
  engine from a W4A16 compressed-tensors checkpoint. Group-64 builds are
  rejected at load. Checkpoints and numbers are in
  [Models that have run on the B60](#models-that-have-run-on-the-b60).

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
  [vllm_xpu/README.md](vllm_xpu/README.md)'s *Upgrading the image*.
- **Gemma 4 arches are now registered** (`gemma4` / `gemma4_mm`) — unlike
  `0.17.0-xpu`, which topped out at Gemma3n. That clears the *architecture* gate.
  The `qwen3` and `openai_gptoss` reasoning parsers are present as before.
  > **No longer unproven.** Gemma-4-26B-A4B runs on the B60 on the *upstream*
  > engine (`vllm_openai_xpu/`, v0.29.0 onward) from an offline int4 **group-32**
  > checkpoint, at 131,072 context. The XPU expert kernel accepts only group-32
  > or channelwise int4 — that narrowness, not a kernel gap, was the real
  > constraint. See [vllm_openai_xpu/README.md](vllm_openai_xpu/README.md),
  > *gemma-4 in depth*. This *scaler /
  > `vllm_xpu` stack* still stays on gpt-oss-20b.
- Reasoning trace field is still `message.reasoning`, not `reasoning_content`
  (re-verified on 0.21.0) — see [README.md](README.md) for the consumer-parsing
  implication.
- Predecessor: `0.17.0-xpu` was a frozen release-tag build (reported vLLM
  `0.1.dev14456`) that topped out at Gemma3n — kept here for upgrade context.

---

## Choosing the inference engine

There are three engines, each in its own folder, and only one runs at a time.
The top-level [README](README.md) has the table for picking one. The reasons
behind it:

- **Why three.** `vllm_xpu/` (stock `intel/vllm`) is the conservative baseline
  for gpt-oss-20b. `scaler/` (Intel's B-series fork, `llm-scaler-vllm`) is the
  fastest for gpt-oss-20b, at 85.6 tok/s against 83.1 on upstream.
  `vllm_openai_xpu/` (upstream's own XPU image) is the only engine that loads
  gemma-4. Switching to it changes the served model name, so gateway mappings
  have to change with it.
- **Why a folder per engine, not a compose profile.** Weights plus KV pool take
  17–21 GiB of the ~22.7 GiB card, so no two engines fit at once. Because every
  `up` has to name a folder, you can't start two by accident, and it's always
  explicit which engine is live.
- **The scaler's lead isn't the fork's work alone.** The fork is built on
  vLLM 0.26.0 and the stock image is on 0.21.0. A measured difference mixes
  Intel's Arc-specific changes with five minor versions of engine changes.
- **`--enforce-eager` is mandatory on the scaler.** In compiled mode the engine
  boots, then returns empty `content` and `reasoning` for gpt-oss-20b. Nothing
  crashes, so the logs give no warning. This was tested on 2026-07-03. Re-testing
  it on a newer base is an experiment that needs a correctness check, not a
  default you can simply flip.

Image pins, the image-by-image throughput table, release notes and the
env-var re-checks are all in [scaler/README.md](scaler/README.md).

---

## Hardware & model rationale

Context for future hardware or model swaps:

- **B70 vs B60** — the Arc Pro B70 (32 GB) gives roughly 1.3× decode / 1.85×
  prefill plus context headroom over the B60. It isn't needed for gemma-4,
  which already runs on the B60 (table below).
- **Multi-GPU** — on a consumer board a second card typically only gets a
  chipset x4 link, so don't tensor-/pipeline-parallel across cards; run each card
  as an independent engine instead.

### Models that have run on the B60

A model is listed only once it has actually booted and served requests on this
card. Being in an engine's supported-model table, or fitting on paper, doesn't
count. The numbers are the ones measured at the time.

| Model | Checkpoint | Quantization | Weights loaded | Engine | Context booted | Decode |
|---|---|---|---|---|---|---|
| gpt-oss-20b | `openai/gpt-oss-20b` | MXFP4 (native) | 12.87 GiB | •&nbsp;`vllm_xpu`<br>•&nbsp;`scaler`<br>•&nbsp;`vllm_openai_xpu` | 131,072 | • `scaler`: 85.6 tok/s ¹<br>• `vllm_openai_xpu`: 83.1 tok/s ¹ |
| gemma-4-26B-A4B-it | `adeepv/gemma-4-26B-A4B-it-W4A16-vLLM` | int4 W4A16, group-32 | 15.76 GiB | •&nbsp;`vllm_openai_xpu` | 45,056 | 56.5 tok/s |
| gemma-4-26B-A4B-it | `reinforce20001/gemma4-26b-a4b-it-qat-w4a16-ct` ² | int4 W4A16, group-32 | 16.93 GiB | •&nbsp;`vllm_openai_xpu` | 131,072 | 56.2 tok/s |
| Qwen3-32B-AWQ | `Qwen/Qwen3-32B-AWQ` | AWQ int4 | 18.14 GiB | •&nbsp;`vllm_xpu` ³ | 7,168 | not recorded |

¹ `bench.sh` counts streamed chunks, so both gpt-oss figures slightly
under-state the true token rate. The gemma-4 figures are true tokens from the
usage chunk.
² This repo stopped being downloadable (HTTP 401) on 2026-09-21. It still
works from a local cache, but a fresh install should use the `adeepv` row.
³ Measured on the predecessor image, `intel/vllm:0.17.0-xpu`, and not re-run
since. Treat it as historical until it's booted on the current image.

What the table says about quantization on this card:

- **gpt-oss has one format, MXFP4**, and it's native. It ships pre-quantized,
  so every engine here loads it as shipped. Never pass `--quantization`.
- **gemma-4 needs an offline int4 checkpoint with group size 32** (channelwise
  would also pass), in compressed-tensors format. The XPU expert kernel rejects
  the more common group-64 builds at load. Check
  `quantization_config.config_groups.*.weights.group_size` before downloading
  any MoE checkpoint. Both checkpoints above are built from Google's own QAT
  weights: `adeepv` repacks them and `reinforce20001` re-quantizes them.
- **Qwen3 dense ran as AWQ.** The official FP8 weights hit an XPU bug (see
  *Quantization on the B60*).
- **Fresher models are a swap, not only a RAG problem.** gpt-oss-20b's
  knowledge cutoff is mid-2024. gemma-4-26B-A4B is a 2026 model that fits in
  about 16 GiB with room for 131,072 tokens of context, so RAG is no longer the
  only way to get fresher knowledge. Some fresher MoEs still don't fit — e.g.
  Qwen3.5/3.6-35B-A3B AWQ is ≈ 24 GB against ~22.7 GiB usable.

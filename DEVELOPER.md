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
**The engines no longer share a vLLM base.** They briefly did — both on 0.21.0,
which the fork reached in `0.21.0-b1` — but the scaler pin moved to the 0.26.0
line on 2026-09-02 (`0.26.0-b1`, then `-b2` on 2026-09-10), taking the fork to a
vLLM 0.26.0 base. A measured difference is again a mix of the fork's
Arc-specific work *and* a five-minor-version engine gap, so attribute any win
carefully.

**Why two folders, not a compose profile:** one GPU (~22.7 GiB) and gpt-oss-20b
needs ~17 GiB, so the two engines can't coexist (~31 GiB = OOM). A separate
folder per engine means every `up` must target an engine's folder (cd into it,
or `-f` its `compose.yaml`), so you can't start both by accident and "which
engine is prod" is always explicit.

**Image:** pinned to `intel/llm-scaler-vllm:0.26.0-b2` (its docs warn against
`:latest`) — **boot-tested and validated** on the B60 with gpt-oss-20b at
128k/0.80 eager on 2026-09-10. It is still an unannounced beta line: `latest`
has not moved since 2026-08-13 and still resolves, digest for digest, to the
`0.21.0-b3.1`-era image. `Releases.md` on `main` does now name `0.26.0-b2`, but
the copy inside the b2 git tag still says `b1` — a tag's own release list is
always one behind. The running engine reports vLLM
`0.26.1.dev0+g568afb3a1.d20260907` on `GET /version` — the quickest way to
confirm which engine owns `:8000` without docker, since the stock image reports
`0.21.0`. Note that b1 and b2 share the fork commit `g568afb3a1` and differ only
in that trailing build date, so use `docker inspect -f '{{.Config.Image}}'
vllm-scaler` to tell the two *builds* apart.

**This line reversed the throughput regression and is the fastest measured on the
B60** — **85.6 tok/s** on `./bench.sh 400` at ~72 ms TTFT, +21.2% over
`0.21.0-b3` and +5.9% over `0.14.0-b8.3.2`, the previous best. `smoke.sh` is ALL
PASS, so the inherited parser flag names survived the base jump. The
image-by-image table, the bench-hygiene rules that make those numbers comparable,
and the KV-pool figures are all in [SCALER_NOTES.md](SCALER_NOTES.md) §3.

**b1 → b2 is a bug-fix release, and it is measurably a no-op here**: identical
tok/s, TTFT, KV pool and VRAM budget, `smoke.sh` still ALL PASS. All four of its
named fixes land in paths gpt-oss-20b does not use — MTP (which this model cannot
run at all), `sym_int4`, and block-FP8. Take it as cheap insurance and expect no
change; the reasoning is in [SCALER_NOTES.md](SCALER_NOTES.md) §2.

**The pool is now sized explicitly, not left as the util remainder.** Adding
`--kv-cache-memory-bytes 8647520256` on 2026-09-10 claimed the 3.87 GiB that
`--gpu-memory-utilization 0.80` was stranding: KV 4.33 → **8.05 GiB**, pool
183,314 → **340,663 tokens**, concurrency at 128k 1.40× → **2.60×**, with decode,
TTFT and a byte-identical greedy generation all unchanged — capacity is free
because decode is bandwidth-bound. Three consequences worth carrying: the flag
**overrides util entirely** (util stays only as the fallback), its byte value is
**absolute and image-specific** so it must be re-derived on an image bump, and
the CLI flag is `--kv-cache-memory-bytes` even though the engine's own log advises
`--kv-cache-memory`. Full measurement and the re-derive procedure:
[SCALER_NOTES.md](SCALER_NOTES.md) §3.

The earlier `b3 → 0.26.0-b1` step *was* a vLLM *base* jump, 0.21.0 → 0.26.0. The
supported-model table is nearly unchanged from `b3`: three rows added
(`Muse-Glimmer-30B`, `Qwen3.8-27B`, `Qwen3.8-27B-FP8`), none removed or changed,
none of the three fitting a single B60, and b2 adds nothing further — its whole
README diff is *removals*. gpt-oss-20b/120b stay in the MXFP4 column, so support
is retained; **no release in this line touches gpt-oss**, so any change in
gpt-oss-20b behaviour comes from the newer base, not the fork's own commits.

Two version-coupled settings in `scaler/compose.yaml` were re-checked against
both 0.26.0 images on 2026-09-10 (method in [SCALER_NOTES.md](SCALER_NOTES.md)
§7): `VLLM_QUANTIZE_Q40_LIB`'s `.so` path is **correct and unchanged**, closing a
long-standing unknown, while `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` turned out to be
**dead** — absent from both images and deleted from upstream's README at b2 — and
has been removed from the file. Clearing `llm_vllm-scaler-cache` on an image bump
is **not** needed on this engine: eager compiles nothing, so the volume holds
16 KB of hash-guarded metadata. The remaining inherited risk, the
`--reasoning-parser` / `--tool-call-parser` flag names, stays **cleared** by the
passing `smoke.sh` above.

**`--enforce-eager` is mandatory, not a safe default:** the config boots with
`--enforce-eager`, which disables torch.compile — removing the uncapped Inductor
buffer growth that constrains util on the stock image, so a higher util would be
safe here (0.80 is kept to match the base vLLM engine). Eager is normally slower
than compiled, **but compiled mode was tested on 2026-07-03 and rejected**:
without `--enforce-eager` the engine boots and compiles fine, then returns empty
`content` *and* `reasoning` for gpt-oss-20b — a silent correctness failure with no
crash in the logs. So the eager figure is the fork's real number, not an
under-statement. The base is five minors newer than the image that failed, so the
verdict *might* have changed; treat re-testing it as a deliberate,
correctness-verified experiment, not a default to flip. The fork's experimental
XPU graph support (added in `0.21.0-b1`) has its own switch,
`VLLM_XPU_ENABLE_XPU_GRAPH` — **tested 2026-09-03 and it is a no-op while
`--enforce-eager` is set**, so the boot-log warning recommending it is safe to
ignore. Measurements in [SCALER_NOTES.md](SCALER_NOTES.md) §4.
gpt-oss-20b is MXFP4 (pre-quantised) — do **not** pass `--quantization`. The fork
inherits upstream's parser flag names; if it renamed them the server fails fast at
startup with a clear arg error.

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

# Intel Arc Pro B60

The card this stack was first built on. This file holds only what is specific to
it: the card itself, the settings sized for its 24 GB, and what was measured on
it. How to run things, and why they are configured the way they are, is shared
with the [Arc Pro B70](INTEL_ARC_B70.md) and lives in the shared docs; start at
the [README](README.md).

> The B60 was replaced by a B70 on 2026-09-26 in the machine these numbers come
> from, so none of them can be re-measured there. They stay as a record, and they
> still hold for anyone running a B60. Most measurements quoted in the engine
> READMEs were taken on this card as well.

A row marked "spec" comes from Intel's published specification; everything else
was measured.

## The card

| | |
|---|---|
| GPU | Battlemage G21, PCI ID `8086:e211` |
| Memory | 24 GB GDDR6, 192-bit, 456 GB/s (spec) |
| What vLLM sees | 23.9 GiB in total; 22.3–23.0 GiB free at boot, depending on the engine |
| Compute | 20 Xe cores, 160 XMX engines (spec) |
| PCIe | 5.0 x8, the card's native width |
| Power cap | 200 W |
| Clock under load | 2,400 MHz |
| Driver | the in-kernel `xe` driver, nothing extra to install |

Power, read with [`watt.sh`](watt.sh): about 38 W for the board and 23 W for the
chip at idle, and a flat ~113 W board / ~70 W chip while decoding gpt-oss-20b on
the scaler engine. At ~78 tok/s that is roughly 1.45 J per token.

The settings below assume the card drives no display. While it still drove the
desktop, the stock engine's utilization had to stay at 0.75 instead of 0.80.

## Settings for this card

These are the values that depend on the card's memory. Everything else in the
engine templates is the same on both cards.

| Engine | Model | Context | KV memory or utilization | Mode |
|---|---|---|---|---|
| `vllm_xpu/` | gpt-oss-20b | 131,072 | util 0.80; 0.86 filled the card and is the OOM edge | compiled |
| `scaler/` | gpt-oss-20b | 131,072 | `--kv-cache-memory-bytes 8647520256` (8.05 GiB) | eager, mandatory |
| `vllm_openai_xpu/` | gpt-oss-20b | 131,072 | `VLLM_KV_CACHE_MEMORY=8603448832` (8.0 GiB) | eager, needed with this value |
| `vllm_openai_xpu/` | gemma-4-26B-A4B-it, `reinforce20001/gemma4-26b-a4b-it-qat-w4a16-ct` | 131,072 | `VLLM_KV_CACHE_MEMORY=4563402752` (4.25 GiB) | compiled |
| `vllm_openai_xpu/` | gemma-4-26B-A4B-it, `adeepv/gemma-4-26B-A4B-it-W4A16-vLLM` | 45,056 | `VLLM_KV_CACHE_MEMORY=5637144576` (5.25 GiB) | compiled |

**The two gemma-4 rows.** 131,072 was boot-tested with `reinforce20001` at
4.25 GiB. `adeepv` is about 1.2 GiB lighter, so it got 5.25 GiB, but it only ran
here at a 45,056-token cap. In compiled mode both measured about 56 tok/s, in
different sessions; an earlier eager-mode comparison had `reinforce20001` 7%
faster. Its repository was unavailable for about a week in September 2026, and
`adeepv` is the fallback if that happens again.

**Don't carry a value from one model to the other.** gemma-4's weights
(15.76 GiB) plus gpt-oss's 8.0 GiB exceed the 22.33 GiB free, and the boot
fails. The other way round boots but strands about 4 GiB.

**Why eager goes with gpt-oss's value on the upstream engine.** With 8.0 GiB
pinned, compiled mode's buffers starve the KV pool and the boot dies 0.21 GiB
short of 131,072. On the scaler, eager is mandatory for a different reason:
compiled mode boots and then returns empty answers.

**vLLM 0.31.0 hasn't been booted on this card.** It captures XPU graphs in
compiled mode and has no switch to turn them off apart from eager mode. On
0.30.0, graphs took 1.54 GiB here and didn't fit next to gemma-4's 4.25 GiB
KV setting at 131,072 context. If a 0.31.0 boot fails at graph capture, use
`VLLM_EAGER_FLAG=--enforce-eager` (about 7% slower decode for gemma-4) or stay
on the `v0.30.0` image.

## Measured results

| Engine | Model | Decode | First token | KV pool |
|---|---|---|---|---|
| `scaler/` 0.26.0-b2 | gpt-oss-20b | 85.6 tok/s ¹ | ~72 ms | 340,663 tokens (2.60×) |
| `vllm_openai_xpu/` 0.29.0 | gpt-oss-20b | 83.1 tok/s ¹ | ~76 ms | 338,928 tokens (2.59×) |
| `vllm_openai_xpu/` 0.30.0 | gemma-4, `adeepv`, 45,056 context | 56.46 tok/s | 40 ms | 117,681 tokens (2.61×) |
| `vllm_openai_xpu/` 0.29.0 | gemma-4, `reinforce20001`, 131,072 context | 56.2 tok/s thinking off, 53.7 on | ~89 ms off, ~128 ms on | 152,592 tokens (1.16×) |
| `vllm_xpu/` 0.21.0 | gpt-oss-20b | not measured on this image | — | — |

¹ `bench.sh` at 400 tokens. It counts streamed chunks, so it slightly
understates the true rate. The gemma-4 figures are true token counts.

**More than one request at a time** (gemma-4, `adeepv`, 0.29.0, 45,056 cap): one
request decoded at 54.8 tok/s, and six at once reached 201.6 tok/s in total,
3.68× as much, with every answer the same length.

**Prefill on Triton is quadratic on this card:** 9,643 tokens took 26.3 s,
31,957 took 305.3 s and 64,708 took 21.6 minutes. The fitted curve, prefix
caching and the rest of gemma-4's detail are in
[vllm_openai_xpu/GEMMA_4_26B_A4B.md](vllm_openai_xpu/GEMMA_4_26B_A4B.md), which was
measured on this card.

## Models that have run on the B60

A model is listed only once it has actually booted and served requests on this
card. Being in an engine's supported-model table, or fitting on paper, doesn't
count. The numbers are the ones measured at the time.

| Model | Checkpoint | Quantization | Weights loaded | Engine | Context booted | Decode |
|---|---|---|---|---|---|---|
| gpt-oss-20b | `openai/gpt-oss-20b` | MXFP4 (native) | 12.87 GiB | •&nbsp;`vllm_xpu`<br>•&nbsp;`scaler`<br>•&nbsp;`vllm_openai_xpu` | 131,072 | • `scaler`: 85.6 tok/s ¹<br>• `vllm_openai_xpu`: 83.1 tok/s ¹ |
| gemma-4-26B-A4B-it | `reinforce20001/gemma4-26b-a4b-it-qat-w4a16-ct` | int4 W4A16, group-32 | 16.93 GiB | •&nbsp;`vllm_openai_xpu` | 131,072 | 56.2 tok/s |
| gemma-4-26B-A4B-it | `adeepv/gemma-4-26B-A4B-it-W4A16-vLLM` | int4 W4A16, group-32 | 15.76 GiB | •&nbsp;`vllm_openai_xpu` | 45,056 | 56.5 tok/s |

¹ `bench.sh` counts streamed chunks, so both gpt-oss figures slightly
understate the true token rate. The gemma-4 figures are true tokens from the
usage chunk.

## What doesn't fit

- **gemma-4-26B-A4B on the scaler engine.** Two separate walls, both recorded
  with their exact errors in [scaler/README.md](scaler/README.md) §8.1. It runs
  on the upstream engine instead.
- **Qwen3.5/3.6-35B-A3B in AWQ**, about 24 GB of weights against ~22.7 GiB
  usable.
- **gpt-oss-120b**, 60.7 GiB of weights.
- **Most 27–30B int4 models fit as weights but not as context.** They land at
  14–17 GiB, and their per-token KV cost trades 128k of context down to roughly
  20–32k.

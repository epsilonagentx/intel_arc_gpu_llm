# Intel Arc Pro B70

This is the card the stack runs on now. This file holds only what is specific to
it: the card itself, the settings sized for its 32 GB, and what has been
measured on it. How to run things, and why they are configured the way they
are, is the same on both cards, so that stays in the shared docs; start at the
[README](README.md). The [Arc Pro B60](INTEL_ARC_B60.md) has its own file.

Everything here was measured on one B70 in a single-GPU machine. A row marked
"spec" comes from Intel's published specification, and anything that hasn't
been tried says "not measured".

## The card

| | |
|---|---|
| GPU | Battlemage G31, PCI ID `8086:e223` |
| Memory | 32 GB GDDR6, 256-bit, 608 GB/s (spec) |
| What vLLM sees | 30.3 GiB in total, 29.92 GiB free at boot |
| Compute | 32 Xe cores, 256 XMX engines (spec) |
| PCIe | 5.0 x16, and the link trains to the full 32 GT/s |
| Power cap | 275 W on the card tested; Intel's reference is 230 W (spec), so partner cards vary |
| Clock under load | 2,800 MHz |
| Driver | the in-kernel `xe` driver, nothing extra to install |

Against the B60, the gains come from memory bandwidth (608 against 456 GB/s),
more Xe cores and 8 GB more memory. The wider PCIe link adds next to nothing: a
single card decodes from its own memory, and the B60's link idled at x1 while
decoding.

Power, read with [`watt.sh`](watt.sh), which works on this card unchanged: 47.7 W
for the board and 26.2 W for the chip while idle with gemma-4 loaded. The most
seen under load is about 183 W, during a long prefill on Triton, well under the
cap. Power while decoding with Intel attention is not measured.

## Settings for this card

These are the values that depend on the card's memory. Everything else in the
engine templates is the same on both cards.

| Engine | Model | Context | KV memory | Mode |
|---|---|---|---|---|
| `vllm_openai_xpu/` | gemma-4-26B-A4B-it, `adeepv/gemma-4-26B-A4B-it-W4A16-vLLM`, with the draft model | 131,072 | `VLLM_KV_CACHE_MEMORY=10200547328` (9.5 GiB); 10.5 GiB without the draft | compiled |
| `vllm_openai_xpu/` | Qwen3.8-27B, `RedHatAI/Qwen3.8-27B-INT4` | 131,072 | `VLLM_KV_CACHE_MEMORY=8603448832` (8.0 GiB) | compiled |
| `vllm_openai_xpu/` | Qwen3.8-27B with MTP and an FP8 output layer | 131,072 | `VLLM_KV_CACHE_MEMORY=8603448832` (8.0 GiB) | compiled |
| `scaler/` | Qwen3.8-27B, int4 at load | 98,304 | `--gpu-memory-utilization 0.90`, no byte value | eager |
| `vllm_openai_xpu/` | gpt-oss-20b | — | not measured | — |
| `scaler/` | gpt-oss-20b | — | not measured | — |
| `vllm_xpu/` | gpt-oss-20b | — | not measured | — |

**How the gemma-4 value was found.** Boot once with the KV line commented out of
`compose.yaml`, so vLLM profiles the card itself. It then prints the byte value
that would use all of it, here `--kv-cache-memory=11916317184` (11.1 GiB). That
leaves only about 0.5 GiB spare, so the setting is 10.5 GiB, which keeps about
1.1 GiB free for the compile buffers. Taking the line out of `.env` isn't
enough for this: Compose then falls back to gpt-oss's B60 value. The draft
model adds 0.78 GiB of weights, so with it the setting drops by 1 GiB to
9.5 GiB.

**gpt-oss has no B70 value yet.** The B60 values should boot here, because they
are smaller than the room this card has, but they would leave memory unused.
That is inferred, not booted. The same one-boot check gives the right number.

**Why `adeepv` rather than `reinforce20001`.** In a same-day comparison with the
same settings, `adeepv` decoded at 74.3–74.9 tok/s and `reinforce20001` at
68.4–69.9, so `adeepv` is about 8% faster here. It is also lighter (15.76
against 16.82 GiB), which leaves more room for the KV cache:
`reinforce20001` got 341,080 tokens (2.60×) from a 9.5 GiB setting.

**Qwen3.8-27B on the scaler needs util 0.90.** At 128k it needs 8.09 GiB of KV
cache, util 0.80 left only 4.13 GiB, and that boot failed. 98,304 tokens fits.
On the upstream engine the 8.0 GiB value gives a full 131,072. It's the
gpt-oss B60 value reused. The boot log suggests about 4 GiB is still free, so
a larger value would probably fit (inferred, not tried).

## Measured: gemma-4 on the upstream engine

`vllm/vllm-openai-xpu:v0.30.0`, the `adeepv` checkpoint, 131,072 context, the
10.5 GiB setting, compiled mode, true token counts:

| | Triton (default) | Intel attention |
|---|---|---|
| decode, 512 tokens, thinking off / on | 73.7–74.3 / 74.4–74.5 tok/s | 85.7–86.3 / 86.5–86.7 tok/s |
| decode, 3,000 / 5,000-token answers | not measured | 81.0 / 80.3 tok/s |
| first token, short prompt | 79 ms | 57–59 ms |
| cold prefill, ~11.8k tokens | 27.7 s | 1.7 s |
| cold prefill, ~24.2k tokens | 116.6 s | 4.2 s |
| 16,049-token prompt: first token | 51.1 s | 2.5 s |
| 16,049-token prompt: decode after it | 38.1 tok/s | 74.8 tok/s |
| longest prompt run | 60,924 tokens in about 13 min ¹ | 129,331 tokens in 51.7 s |
| KV pool | 376,999 tokens (2.88×) | 394,408 tokens (3.01×) |
| weights on the card | 15.76 GiB | 14.69 GiB ² |
| image input | works | off |
| `smoke.sh` | ALL PASS | ALL PASS, reasoning on and off |

¹ One run, with a second request overlapping it.
² With image input off vLLM loads less, most likely because the image encoder is
skipped (inferred).

Intel attention is the plugin explained in
[vllm_openai_xpu/PLUGIN.md](vllm_openai_xpu/PLUGIN.md), and image input is what
it costs.

**With the draft model on top** (2026-10-03, Intel attention, 3 draft tokens,
9.5 GiB setting), decode went from 82.1 to 137–146 tok/s with default sampling
and from 86.6 to 156 at temperature 0. 6 requests at a time went from 323 to
386 tok/s in total. After a 16,853-token prompt, decode went from 75.0 to
109.5 tok/s, with the first token 0.4 s later. The KV pool is 356,893 tokens
(2.72×). How it works, and the 4- and 5-token runs, are in
[*Faster decode with a draft model*](vllm_openai_xpu/GEMMA_4_26B_A4B.md#faster-decode-with-a-draft-model).

**Against the B60, on the same Triton settings:** decode is about 1.30× faster
(73.2 against 56.1–56.5 tok/s) and prefill 1.43–1.50× faster (11,782 tokens in
27.7 s, 23,307 in 107.2 s). That's in line with the memory-bandwidth ratio.
Prefill on Triton is still quadratic here, just quicker.

**The KV formula holds on this card too.** The formula in the engine README
predicts 2.877× concurrency at the 10.5 GiB setting and 131,072 context, and the
log says 2.88×. See
[*Context is nearly free*](vllm_openai_xpu/GEMMA_4_26B_A4B.md#context-is-nearly-free--the-kv-formula).

**XPU graphs are on from vLLM 0.31.0, and they fit.** Capture takes 0.74 GiB
for gemma-4 and 0.96 GiB for Qwen3.8, next to the KV settings above, and
makes gemma-4 decode about 8% faster after a long prompt. Qwen3.8 needs
`VLLM_MAX_NUM_SEQS=128` for the capture. On 0.30.0, with graphs off, the GPU
was already 99–100% busy during decode, which is why short prompts barely
change. The 0.30.0 → 0.31.0 comparison is in
[vllm_openai_xpu/README.md](vllm_openai_xpu/README.md#0300--0310).

## Measured: Qwen3.8-27B on the upstream engine

`vllm/vllm-openai-xpu:v0.30.0`, `RedHatAI/Qwen3.8-27B-INT4`, 131,072 context,
the 8.0 GiB setting, compiled mode, true token counts, thinking off
(2026-10-02):

| | Triton | Intel attention |
|---|---|---|
| weights on the card | 17.56 GiB | 17.56 GiB |
| KV pool | 244,270 tokens (1.86×) | 244,270 tokens (1.86×) |
| decode, 512 tokens | 29.7 tok/s | **32.9 tok/s** |
| 9,411-token prompt: first token | 73.6 s | **5.9 s** |
| 18,786-token prompt: first token | 283.0 s | **12.8 s** |
| 18,786-token prompt: decode after it | 5.3 tok/s | **30.2 tok/s** |
| image input | works | works |
| `smoke.sh` | ALL PASS | ALL PASS |

Intel attention is about 11% faster at decode than Triton, and about 15% faster
than the scaler below. It also fits the full 131,072 context where the scaler
fits 98,304. How to run it is in
[vllm_openai_xpu/QWEN3_8_27B.md](vllm_openai_xpu/QWEN3_8_27B.md).

## Measured: Qwen3.8-27B with MTP and an FP8 output layer

Same engine, checkpoint and context, with MTP speculative decoding (3 draft
tokens) and the output layer converted to FP8 (2026-10-03). The MTP rows with
a 16-bit or FP8 output layer ran at a 7.0 GiB KV setting, the int4 row at
8.0 GiB; the FP8 set measured the same at 8.0 GiB within noise. Three
different prompts, each sent once, 512 tokens, thinking off, median per run:

| | default sampling | greedy |
|---|---|---|
| no MTP | 32.2 tok/s | 32.9 tok/s |
| MTP, 16-bit output layer | 53.9, 53.4 tok/s | 54.8, 56.5 tok/s |
| **MTP, FP8 output layer** | **58.9, 53.1, 60.3 tok/s** | **67.3, 64.8, 65.3 tok/s** |
| MTP, int4 output layer and draft head | 71.5, 72.2 tok/s | 73.9, 73.5 tok/s |

Weights take 17.18 GiB. At the 8.0 GiB setting the KV pool is 208,093 tokens
(1.59×) and the card sits at about 31,000 of 32,656 MiB, steady through a
56,620-token prompt and four requests at once. At 7.0 GiB it's 181,068 tokens
(1.38×) with about 1 GiB more headroom. Greedy answers are byte-identical to
the 16-bit output layer, and `smoke.sh` passes. The int4 output layer is
faster but shifts the next-token ranking measurably, with the same GSM8K
score; the quality numbers are in
[vllm_openai_xpu/tools/README.md](vllm_openai_xpu/tools/README.md#measured).
How to set it up is in
[vllm_openai_xpu/QWEN3_8_27B.md](vllm_openai_xpu/QWEN3_8_27B.md#faster-decode-mtp-and-an-fp8-output-layer).

## Measured: Qwen3.8-27B on the scaler

`intel/llm-scaler-vllm:0.26.0-b2`, the official BF16 checkpoint quantized to
int4 as it loads (`--quantization sym_int4`):

| | |
|---|---|
| weights on the card | 17.83 GiB |
| context | 98,304 at util 0.90 (128k failed at 0.80) |
| KV pool | 123,006 tokens (1.25×) |
| decode, 512 tokens, thinking off | 28.7–28.9 tok/s |
| `smoke.sh` | ALL PASS |

It's a dense 27B model, so every token reads all of its weights, which is why it
decodes at well under half gemma-4's rate (inferred). The command, including the
extra flags it needs, is the commented-out block in
[`scaler/compose.yaml`](scaler/compose.yaml).

## Models that have run on the B70

A model is listed once it has actually booted and served requests on this card.

| Model | Checkpoint | Quantization | Weights loaded | Engine | Context booted | Decode |
|---|---|---|---|---|---|---|
| gemma-4-26B-A4B-it | `adeepv/gemma-4-26B-A4B-it-W4A16-vLLM` | int4 W4A16, group-32 | 15.76 GiB | `vllm_openai_xpu` | 131,072 | 74 tok/s on Triton, 86 with Intel attention, 137–156 with the draft model as well |
| gemma-4-26B-A4B-it | `reinforce20001/gemma4-26b-a4b-it-qat-w4a16-ct` | int4 W4A16, group-32 | 16.82 GiB | `vllm_openai_xpu` | 131,072 | 68.4–69.9 tok/s on Triton |
| Qwen3.8-27B | `RedHatAI/Qwen3.8-27B-INT4` | int4 W4A16, group-128 | 17.56 GiB | `vllm_openai_xpu` | 131,072 | 32.9 tok/s with Intel attention, 29.7 on Triton |
| Qwen3.8-27B | the same, output layer converted to FP8 by `vllm_openai_xpu/tools/quantize_heads.py` | int4 W4A16, FP8 output layer | 17.18 GiB | `vllm_openai_xpu` | 131,072 | 65.8 tok/s greedy with MTP (3 drafts) |
| Qwen3.8-27B | the same, output layer and draft head converted to int4 (`--head int4 --mtp-int4`) | int4 W4A16 throughout | 15.8 GiB | `vllm_openai_xpu` | 131,072 | 73.7 tok/s greedy with MTP (3 drafts) |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | int4 at load (`sym_int4`) | 17.83 GiB | `scaler` | 98,304 | 28.7 tok/s |

## Not measured yet

- gpt-oss-20b on any of the three engines, and so a B70 KV value for it.
- The stock `vllm_xpu/` engine at all.
- More than one request at a time on Qwen3.8, or more than 6 on gemma-4. Every
  other figure above is a single request. Speculative decoding gains less when
  several requests share the card: about 20% for gemma-4 at 6 requests, against
  1.7× for one.
- Power while decoding with Intel attention.
- Models that fit 32 GB but not 24 GB, such as Qwen3.5/3.6-35B-A3B in AWQ
  (about 24 GB of weights). None has been tried.

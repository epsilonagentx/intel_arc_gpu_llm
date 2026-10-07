# Qwen3.8-27B on the upstream XPU engine

How Qwen3.8-27B runs on this engine, and how MTP and an FP8 output layer double
its decode. Starting, switching models and the settings every model shares are
in [README.md](README.md).

| | |
|---|---|
| Checkpoint | `RedHatAI/Qwen3.8-27B-INT4`, int4 group-128, 17.56 GiB |
| Attention | Intel's kernel, with image input still working |
| Speculative decoding | its own MTP head, 3 tokens: 65.8 tok/s greedy on the B70 with an FP8 output layer |
| Context | 131,072 |
| Reasoning | on by default |

It's Block C in `.env.example`. It's a dense 27B model: 48 of its 64 layers are
linear attention (Gated DeltaNet) and 16 are full attention, so the KV cache
grows with only those 16 layers. Every token reads all of its weights, which
is why it decodes at well under half gemma-4's rate. MTP and an FP8 output
layer double it (see
[*Faster decode*](#faster-decode-mtp-and-an-fp8-output-layer) below).

## Contents

- [Checkpoint](#checkpoint)
- [Intel attention, with images](#intel-attention-with-images)
- [Measured against Triton](#measured-against-triton)
- [Leave room for thinking](#leave-room-for-thinking)
- [Faster decode: MTP and an FP8 output layer](#faster-decode-mtp-and-an-fp8-output-layer)

---

## Checkpoint

`RedHatAI/Qwen3.8-27B-INT4`: int4 symmetric, group size 128,
in compressed-tensors format, with the vision encoder kept at full precision.
It loads as 17.56 GiB and runs on `XPUwNa16LinearKernel`, the same int4
kernel gemma-4's non-expert layers use. This engine can't quantize a
full-precision checkpoint to int4 as it loads (the scaler can), so it needs a
pre-quantized one. For a dense model the XPU kernel takes symmetric or
asymmetric int4 with any group size that's a multiple of 32, which most int4
builds on Hugging Face meet. Check `quantization_config` before trying another.

Compose doesn't pin a revision, so a new upload to that repo loads on the next
boot. The measured one is `91bd022d5b49442a868bc35008f6c21e1860edfa`.

## Intel attention, with images

The full-attention heads are size 256, which
Intel's kernel takes without the head-512 plugin. Qwen also doesn't use
gemma-4's bidirectional attention over image tokens, so `auto` already picks
FLASH_ATTN and image input keeps working. The block sets `FLASH_ATTN`
explicitly anyway, so the choice is visible. In the log you should see
`Using Flash Attention backend.`, and `Setting attention block size to 1600
tokens`: vLLM makes the attention blocks large enough to hold the
linear-attention state, and this is normal.

The log also mentions Triton (`Warming up Qwen GDN Triton kernels`), and that
doesn't mean attention fell back. On XPU the linear-attention layers run their
XPU path, which calls Intel's own kernel (`torch.ops._xpu_C.gdn_attention`),
and 0.31.0 says so with `GDN decode kernel: XPU` (0.30.0 printed "cuda" there
even on XPU). `Warmed M-RoPE Triton kernels` is the position encoding, a small
step outside attention. This is read from the v0.31.0 source.

## Measured against Triton

Measured on the B70 on 2026-10-02, same checkpoint, 131,072 context, the
8.0 GiB KV value and compiled mode, true token counts, thinking off:

| | Triton | Intel attention |
|---|---|---|
| decode, 512 tokens | 29.7 tok/s | **32.9 tok/s** |
| 9,411-token prompt: first token | 73.6 s | **5.9 s** |
| 18,786-token prompt: first token | 283.0 s | **12.8 s** |
| 18,786-token prompt: decode after it | 5.3 tok/s | **30.2 tok/s** |
| code hidden 10% into those prompts | found | found |
| image input | works | works |
| KV pool | 244,270 tokens (1.86×) | 244,270 tokens (1.86×) |
| `smoke.sh` | ALL PASS | ALL PASS |

On Triton the first-token time roughly quadruples when the prompt doubles, so
long prompts get slow fast. With Intel attention it grows in line with the
prompt. For comparison, the scaler runs the official BF16 checkpoint
quantized to int4 as it loads, at 28.7 tok/s, and fits only 98,304 tokens of
context on the same card ([INTEL_ARC_B70.md](../INTEL_ARC_B70.md)).

## Leave room for thinking

Qwen3.8 thinks by default and at length. On
`bench.sh`'s default question it reasoned for 2,477 tokens, about 80 s, before
the first word of the answer, so `./bench.sh 400` and `./bench.sh 2000` both
end with no answer at all. Use `THINKING=0 ./bench.sh 400` to measure speed
(32.4 tok/s, measured), and give clients a `max_tokens` of several thousand,
or have them turn thinking off for quick replies.

## Faster decode: MTP and an FP8 output layer

Two changes together double Qwen3.8's decode, from 32.9 to 65.8 tok/s, with
the same answers.

**MTP speculative decoding.** The checkpoint ships a small draft head (MTP,
0.79 GiB) that guesses the next tokens; the main model then checks several
guesses in one step. vLLM supports it for this model out of the box:
`VLLM_SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}`.
Each extra draft token is accepted less often (about 73%, 46% and 28% for
the first three), so 3 is the sweet spot: 2 was slower, and 4 was no faster
with greedy and slower with sampling.

**An FP8 output layer.** The checkpoint keeps `lm_head`, 248,320 rows by
5,120, at 16-bit (2.37 GiB), and with MTP it's read for the main step and
again for every draft, because the draft head shares it. vLLM can't quantize
it as it loads (its online quantization skips the output layer), but it does
load a checkpoint that declares an FP8 `lm_head`, and runs it on XPU's
native FP8 W8A16 kernel. [`tools/quantize_heads.py`](tools/quantize_heads.py)
makes such a copy next to the original, which stays untouched. How to run it,
what it changes in the Red Hat checkpoint, and its two faster variants (an
int4 draft head, and an int4 output layer) are in
[tools/README.md](tools/README.md). Then point the engine at the copy:

```dotenv
VLLM_MODEL=/cache/huggingface/local/Qwen3.8-27B-INT4-fp8head
VLLM_SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}
VLLM_KV_CACHE_MEMORY=8603448832
VLLM_MAX_NUM_SEQS=128
```

`VLLM_MAX_NUM_SEQS=128` is needed from vLLM 0.31.0 on, with or without MTP:
0.31.0 captures XPU graphs, and the capture needs one linear-attention state
block per request vLLM allows at once. The 8.0 GiB KV setting holds 154, so
the default of 256 stops the boot.

The KV value stays at 8.0 GiB, the same as without MTP: the pool is 208,093
tokens (1.59× at 131,072) and the card sits at about 31,000 of 32,656 MiB.
A 56,620-token prompt and four requests at once didn't move that by more
than a few MiB. 7.0 GiB (`7516192768`) also works, with 181,068 tokens and
about 1 GiB more headroom.

**Measured** on the B70 on 2026-10-03, same benchmark throughout: three
different prompts, each sent once, 512 tokens, thinking off, decode timed
from the first token to the last, median per run. The int4-draft-head rows
ran at 8.0 GiB, the other MTP rows at 7.0 GiB; at 8.0 GiB the FP8 set
measured 62.1 and 64.9 tok/s, the same within noise:

| | default sampling | greedy |
|---|---|---|
| no MTP | 32.2 | 32.9 |
| MTP, 3 drafts, 16-bit output layer | 53.9, 53.4 | 54.8, 56.5 |
| **MTP, 3 drafts, FP8 output layer** | **58.9, 53.1, 60.3** | **67.3, 64.8, 65.3** |
| MTP, 2 drafts, 16-bit output layer | 49.1 | 53.6 |
| MTP, 4 drafts, 16-bit output layer | 46.6 | 54.0 |
| MTP, 3 drafts, FP8 output layer, int4 draft head | 60.8, 63.3 | 67.6, 68.1 |
| MTP, 3 drafts, int4 output layer, int4 draft head | 71.5, 72.2 | 73.9, 73.5 |

- Speed depends on the text: code is easiest to guess (up to 80 tok/s),
  free prose the hardest (around 50).
- With sampling the FP8 gain is smaller and noisier than with greedy (+7%
  against +18%), because the drafts that get accepted change from run to run.
- On the first 100 GSM8K problems all three output layers (16-bit, FP8, int4)
  scored 96. Ranked by next token, FP8 stays close to two identical runs of
  the same setup; int4 changes the top token at about 3% more positions, so
  it's the fastest choice but not a free one. Details in
  [tools/README.md](tools/README.md#measured).
- `smoke.sh` passes, image input still works, and the prefix cache still hits
  on follow-up turns.

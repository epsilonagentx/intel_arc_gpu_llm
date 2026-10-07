# gemma-4-26B-A4B on the upstream XPU engine

How gemma-4-26B-A4B-it runs on this engine, and why it's set up the way it is.
It's Block B in `.env.example` and the model this engine currently serves.
Starting, switching models and the settings every model shares are in
[README.md](README.md).

| | |
|---|---|
| Checkpoint | `adeepv/gemma-4-26B-A4B-it-W4A16-vLLM`, int4 group-32, 15.76 GiB |
| Attention | Intel's kernel through [`head512_plugin/`](PLUGIN.md), with image input |
| Speculative decoding | Google's draft model, 3 tokens: 137–156 tok/s on the B70 |
| Context | 131,072; extra context costs almost no memory |
| Reasoning | on by default in the `.env.example` block |

> ⚠ **Known issue:** in long coding-agent sessions gemma-4 can get stuck
> thinking and repeat itself until the token limit. See
> [*Known issue: thinking loops in agent sessions*](#known-issue-thinking-loops-in-agent-sessions).

## Contents

- [Checkpoint constraint](#checkpoint-constraint)
- [Context is nearly free — the KV formula](#context-is-nearly-free--the-kv-formula)
- [The cap is a guardrail, not a speed setting](#the-cap-is-a-guardrail-not-a-speed-setting)
- [Prefill on Triton is quadratic](#-prefill-on-triton-is-quadratic)
- [Intel attention for gemma-4](#intel-attention-for-gemma-4)
- [Faster decode with a draft model](#faster-decode-with-a-draft-model)
- [Prefix caching](#prefix-caching)
- [Why it decodes slower than gpt-oss per pass](#why-it-decodes-slower-than-gpt-oss-per-pass)
- [Known issue: thinking loops in agent sessions](#known-issue-thinking-loops-in-agent-sessions)
- [Open questions](#open-questions)

---

## Checkpoint constraint

The checkpoint must be **int4-symmetric with `group_size` 32 or channelwise**.
The XPU expert kernel (`XPUExpertsWNA16`) accepts only those two schemes, so the
smaller and far more popular **group-64** builds are rejected at load despite
being ~0.7 GiB lighter. Check
`quantization_config.config_groups.*.weights.group_size` before trying any other
MoE checkpoint.

## Context is nearly free — the KV formula

gemma-4 has **25 sliding-window layers (window 1024) and 5 full-attention
layers**. vLLM sizes the sliding layers by the window, not by
`--max-model-len`, so only the 5 full-attention layers grow with context, and
they're the cheap ones (2 KV heads × 512, against 8 × 256):

```
fixed  (25 sliding layers, window-bounded) = 1.235 GB      # context-independent
linear (5 full layers)                     = 20 KiB/token
```

So **raising `--max-model-len` costs no VRAM**; it only lowers the concurrency
figure. On the B60's 4.25 GiB setting the formula matched the logged pool to
0.003% at 32k, 64k and 131k, with 162,496 as the ceiling. On the B70's 10.5 GiB
setting it predicted 2.877× at 131,072 and the log said 2.88×; there the
checkpoint's own 262,144 limit comes before memory runs out (computed, not
booted). A 64,708-token prompt answered a question about its last record
correctly, so the long context really works.

> **⚠ Raising `--max-num-batched-tokens` lowers the context ceiling.** It
> enlarges the sliding layers' reservation: at 8192 the fixed block grows from
> 1.235 to 3.568 GB and the B60's ceiling falls to 48,576. Leave it at the
> default: 2496 with image input on, which vLLM raises from its 2048 so one
> video input fits a batch, and 2048 with `--language-model-only`.

When sizing any hybrid model like this, per-token KV figures mislead. Read
`layer_types`, `sliding_window`, `num_key_value_heads` **and**
`num_global_key_value_heads` from its config.

## The cap is a guardrail, not a speed setting

**The cap doesn't change decode speed** (identical at 32k, 64k and 128k on the
B60). The KV pool is set in bytes, so a given prompt costs the same wherever
the cap sits. What the cap does control is the longest prompt a client can
send. A lower cap turns an oversized prompt into a quick `400` instead of tens
of minutes of prefill on Triton. If you keep 131,072, make sure client
timeouts along the whole path allow for it.

## ⚠ Prefill on Triton is quadratic

On the default Triton backend, prefill time, not VRAM, limits usable context.
[Intel attention](#intel-attention-for-gemma-4) removes the problem. Measured on
the B60 (the B70 is 1.4–1.5× faster):

| cold prompt | wall |
|---|---|
| 2,421 tok | 2.2 s |
| 9,621 tok | 19.4 s |
| 19,221 tok | 80.8 s |
| **64,708 tok** | **21.6 min** |

That's almost exactly quadratic, `T ≈ 26.3 s × (n / 9643)^2.048`. Beyond 64,708
tokens it's extrapolation: about 51 min at 96k and 92 min at 131k, never run.
Prefill here is even slower per token than decode, which a healthy kernel would
never be.

### Why the slow kernel gets chosen

The boot log says it:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}.
FA4 not available, forcing TRITON_ATTN backend.
```

vLLM lets FLASH_ATTN take head size 512 only on FlashAttention 4, which is
CUDA-only, so it puts every gemma-4 layer on Triton. Intel's head-512 kernel is
compiled into this image but sits behind that check. No setting gets past it;
the small plugin in `head512_plugin/` does. [PLUGIN.md](PLUGIN.md) walks
through the check and the source.

## Intel attention for gemma-4

One line in `.env` puts gemma-4 on Intel's head-512 kernel instead of Triton:

```dotenv
VLLM_ATTN_BACKEND=FLASH_ATTN
```

Then `docker compose up -d`. To go back, set it to `auto` and run `up -d` again.
Image input keeps working from vLLM 0.31.0 on; see
[*Images with Intel attention*](#images-with-intel-attention) below.

**How it works.** `head512_plugin/` makes FLASH_ATTN accept head size 512 on
XPU, and changes nothing else: with the switch off gemma-4 still gets Triton.
Compose mounts it and puts it on `PYTHONPATH`, so every vLLM process loads it.
[PLUGIN.md](PLUGIN.md) walks through it file by file.

**Measured** on the B70 on 2026-09-30, on vLLM 0.30.0. Both columns use the same
checkpoint, the 131,072 context cap, the 10.5 GiB KV pin and the same scripts,
with true token counts. On 0.30.0 Intel attention ran only with
`--language-model-only`, so that column is text-only:

| | Triton (switch off) | Intel attention |
|---|---|---|
| decode, 512 tokens, thinking off / on | 73.7–74.3 / 74.4–74.5 tok/s | **85.7–86.3 / 86.5–86.7 tok/s** |
| cold prefill, ~11.8k tokens | 27.7 s | **1.7 s** |
| cold prefill, ~24.2k tokens | 116.6 s | **4.2 s** |
| 16,049-token prompt: first token | 51.1 s | **2.5 s** |
| 16,049-token prompt: decode after it | 38.1 tok/s | **74.8 tok/s** |
| KV pool | 376,999 tokens (2.88×) | 394,408 tokens (3.01×) |
| image input | yes | no (0.30.0) |

The larger pool on the right comes from running text-only, not from the
kernel; see the next section.

Longer prompts, with Intel attention: 67,956 tokens in 17.8 s and 129,331 in
51.7 s. Triton took about 13 minutes for 60,924 tokens on the same card.

The answers held up: the same answers at temperature 0, `smoke.sh` passed with
thinking on and off, and a code hidden in documents of 11,991, 67,956 and
129,331 tokens was found every time. It sat far outside the 1024-token sliding
window, so only the head-512 layers could have found it.

**After an image upgrade, check the boot log before trusting it:**

- `xpu_head512: FLASH_ATTN on XPU now accepts head size 512` means the plugin
  loaded;
- `Using Flash Attention backend.` and `Setting kv cache block size to 64` mean
  it took effect;
- there must be no `[vllm_xpu_kernels] XPU kernel not compiled … falling back to
  PyTorch reference attention`. Intel's wrapper silently switches to a slow
  reference path when a kernel variant is missing, and that line is the only
  sign.

### Images with Intel attention

gemma-4 lets the tokens of one image look at each other in both directions,
and vLLM applies that rule only on Triton or FlashAttention 4.
[PLUGIN.md](PLUGIN.md#images-and-the-left-to-right-rule) explains the rule. On
0.30.0 asking for FLASH_ATTN with images on stopped the boot, so Intel
attention needed `--language-model-only` and cost image input.

From 0.31.0, vLLM boots it anyway when FLASH_ATTN is asked for by name, reads
the image tokens left to right like text, and says so in the log:

```
Using Flash Attention on XPU for a multimodal prefix-LM model because it was
explicitly requested. The prefix-LM bidirectional mask cannot be applied, so
image/video inputs will produce incorrect results
```

**Measured, the answers didn't change.** On the B70 on 2026-10-06, nine test
images went to both backends with images on, the draft model on, temperature 0,
thinking off, each sent twice: an invoice, serial numbers, a 3×3 and a 5×5 grid
of coloured shapes, 13 and 27 dots, a bar chart, an 8×8 grid of letters and a
small-print log. Each answer was checked for the expected values:

| | Triton | Intel attention |
|---|---|---|
| easy five images | 48/48 | 48/48 |
| hard four images | 18/22 | 18/22 |
| missed on both | one column of the letter grid, 29 dots | one column of the letter grid, 26 dots |
| KV pool (9.5 GiB setting) | 341,080 tokens | 340,227 tokens |

A likely reason, not proven: the image encoder already lets the parts of a
picture see each other before its tokens reach the language model, so reading
them left to right there loses little. Nine synthetic images are a small test.
Real photos haven't been tried, and many images per request were tested only
for memory (below), not for the answers.

**What image input costs.** The image encoder's weights take about 1 GiB of
card memory. That comes out of the free memory, not the KV setting, which
was 9.5 GiB in both of these boots. The KV pool still holds 4.7% fewer tokens, 340,227 instead
of 356,893 with the draft model. The likely reason, inferred: with images on
vLLM raises the per-step budget from 2048 to 2496 tokens, and that enlarges
the sliding layers' reservation, as in the warning under
[*Context is nearly free*](#context-is-nearly-free--the-kv-formula).

Reading images needs almost no memory beyond that. vLLM encodes at most 2,496
tokens of images per step (about 9 images) and sets that aside at boot. On the
B70 the engine grew by 0.10 GiB on the first image request and then stayed flat
through 480 images in one request (128,671 tokens, read in 113.8 s), 6
requests at once with 40 images each, and 4 at once with 20 images and long
text. nvtop showed 1.92 GiB of the card still free afterwards.

**The KV setting is 11.0 GiB because of that free memory.** With images on and
the draft model, `VLLM_KV_CACHE_MEMORY=11811160064` gives 393,915 tokens
(3.01×). It was set by a load test, not by vLLM's own profiling. At that value
the engine sets aside less working memory at boot and claims the rest under
load, and PyTorch keeps it until the next restart:

| load, 11.0 GiB setting | engine memory | free on the card (nvtop) |
|---|---|---|
| idle after boot | 29.89 GiB | 2.10 GiB |
| 480 images in one request | 31.03 GiB | about 0.85 GiB (computed) |
| 64 requests at once | 31.40 GiB | about 0.5 GiB (computed) |
| 256 requests at once, the most the engine takes | 31.61 GiB | **0.24 GiB** |

Every request succeeded, but 0.24 GiB is a thin margin. A heavier mix, for
example many requests at once that all carry images, hasn't been tried, and if
the engine runs out of memory it stays down. `11274289152` (10.5 GiB) leaves
about 0.5 GiB more (inferred) for about 4% fewer tokens.

Decode speed didn't change (draft model on, both 0.31.0):

| | images on | text-only (10-05) |
|---|---|---|
| decode, default sampling | 106–152 tok/s, average 130 | 128–141 |
| decode, temperature 0 | 117–183 tok/s, average 147 | 149 |
| decode right after an image | 171–178 tok/s | — |
| 6 requests at once, total | 393 tok/s | 389–397 |

The text-only column is from the day before with other prompts, so the single
rows are only comparable as ranges. With the draft model the rate depends on
the text, which is why each prompt lands somewhere else. 6 requests at once is
the cleanest comparison.

To go text-only for the larger pool, add
`VLLM_TEXT_ONLY_FLAG=--language-model-only` and run `up -d`. Images then get
`400 At most 0 image(s) may be provided in one prompt`.

## Faster decode with a draft model

Google publishes a small "assistant" model for gemma-4,
[`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant):
4 layers, 0.42B parameters, 0.78 GiB. It guesses the next few tokens, and
gemma-4 checks all of them in one pass instead of producing one token per pass.
Wrong guesses are thrown away, so the answers come from gemma-4 exactly as
before. Only the speed changes. Take the QAT variant: the `adeepv` checkpoint
is built from Google's QAT weights, so the matching draft should guess better.

Two lines in the gemma-4 block of `.env` switch it on:

```dotenv
VLLM_SPECULATIVE_CONFIG={"model":"google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant","num_speculative_tokens":3}
VLLM_KV_CACHE_MEMORY=11811160064
```

Then `docker compose up -d`. The first boot downloads the draft. vLLM sees
`model_type: gemma4_assistant` in its config and picks the MTP method itself,
so there's no `"method"` key. The draft's weights take 0.78 GiB. The runs
below used a 9.5 GiB KV setting to make room for them; a later load test
showed 11.0 GiB fits too, with a thin margin (see
[*Images with Intel attention*](#images-with-intel-attention)). The draft adds no KV cache of its own: its
layers have only query projections and read gemma-4's cache. The boot log
shows it as four `Gemma4 MTP: draft layer N … -> language_model.model.layers.28`
(and `.29`) lines.

**Measured** on the B70 on 2026-10-03. Same checkpoint, Intel attention, 131,072
context, compiled. Decode is the median of three 512-token answers (a story,
code, an explanation) with thinking off, each set run twice:

| | without | 3 draft tokens | 4 | 5 |
|---|---|---|---|---|
| decode, default sampling | 82.1 tok/s | **137–146** | 136 | 125 |
| decode, temperature 0 | 86.6 tok/s | **156** | 155 | 152 |
| 6 requests at once, total | 323 tok/s | **386** | — | — |
| 16,853-token prompt: first token | 2.65 s | 3.05 s | — | — |
| 16,853-token prompt: decode after it | 75.0 tok/s | **109.5** | — | — |
| KV pool, text-only | 394,408 (3.01×) | 356,893 (2.72×) | same | same |
| weights on the card | 14.69 GiB | 15.47 GiB | same | same |

The gain depends on the text. Code is easiest to guess (167–175 tok/s) and the
story hardest (109–117). The draft guesses well: about 2.75–3.05 tokens are
accepted per pass, and its first, second and third guesses are right about 78%,
55% and 43% of the time. With 4 or 5 tokens the extra guesses are wrong too
often to pay for themselves, so 3 it is.

With several requests at a time the gain shrinks to about 20%, because the GPU
is already busy with the other requests. A long prompt's first token arrives
about 0.4 s later. `smoke.sh` passed, and a code hidden in the 16,853-token
prompt was found with and without the draft.

Google [warns](https://ai.google.dev/gemma/docs/mtp/overview) that on the
26B-A4B mixture-of-experts model the draft may not speed up a single request,
since checking several tokens pulls in more experts. That didn't happen on
this card.

This run was text-only. With image input on, the pool with the draft model is
340,227 tokens (2.60×) at 9.5 GiB and 393,915 (3.01×) at the 11.0 GiB default;
see
[*Images with Intel attention*](#images-with-intel-attention).

To turn it off, set `VLLM_SPECULATIVE_CONFIG=null`, put the KV setting back to
`11274289152` and run `up -d`.

**Tried and not worth it: a smaller output layer.** gemma-4 shares its 16-bit
output layer with its embeddings, `[262144, 2816]`, 1.47 GB read for every
token. A copy stored as FP8 made no measurable difference on top of the draft
model (143–145 and 158 tok/s). As int4 with group size 32 it added about 5% (147–159 and 164)
but slightly changes the model's output, and that wasn't measured.

## Prefix caching

vLLM reuses the KV of a prompt prefix it has already seen. The same prompt sent
twice on the B60, on Triton:

| n | cold | warm | speedup |
|---|---|---|---|
| 9,643 | 26.3 s | 0.31 s | **84×** |
| 31,957 | 305.3 s | 0.63 s | **481×** |

It works on gemma-4's sliding-window layers too. A growing conversation only
pays for the new turn, though on Triton each turn gets slower as the
conversation grows (a 2k-token turn takes about 40 s at 32k of context and
2.2 min at 100k, on the B60). What hurts most is a large prompt the engine hasn't seen
(a fresh document, a big paste), once per engine lifetime, since the cache
lives in VRAM and a recreate empties it. **Put stable content first:** matching
starts from token 0, so a document placed after a varying question is never
reused. With Intel attention a cold prompt is 16–28× cheaper, so this matters
much less.

---

## Why it decodes slower than gpt-oss per pass

Not memory bandwidth: gemma-4 reads less per token (about 3.23 GB against
3.71 GB). It runs 240 small expert matrix multiplies per token (30 layers × 8
of 128 experts, each 704 wide) where gpt-oss runs 96 that are four times
wider, and small ones use the GPU poorly. The draft model gets around that by
checking several tokens per pass — see
[*Faster decode with a draft model*](#faster-decode-with-a-draft-model).

---

## Known issue: thinking loops in agent sessions

**Not fixed yet.** In long sessions with a coding agent, gemma-4 sometimes
starts thinking and never stops, so the answer never comes. It goes on until
the client gives up or the request's token limit runs out. The next request in
the same session usually works fine. It has been seen on the B70 with Intel
attention, with and without the draft model.

**What the captured loops look like.** Five were caught on 3 and 4 October
2026, with a coding agent (opencode) talking to the engine through a gateway:

- All five were stuck in the thinking. None reached the answer or a tool call.
- They started 25,000–55,000 tokens into the conversation, so it isn't only a
  problem near the context limit.
- Most came straight after a failed or broken file edit, at a point where the
  model had to choose what to do next.
- The model usually had the right idea in its first few hundred characters.
  Then it kept reopening the decision ("Actually, I'll use `write`." / "Wait,
  I'll check…") until it settled into a block of 54 to 357 characters that it
  repeated word for word, 150–235 times.
- While it loops, the draft model's guesses are accepted 98–100% of the time,
  against 50–65% for normal text. The engine log's `SpecDecoding metrics` lines
  show it. Copying a file word for word also gives about 100%, so this hints at
  a loop but doesn't prove one.

**Capping the thinking isn't enough on its own.** A temporary plugin gave every
request a default `thinking_token_budget` of 4,096 (see
[*What reasoning actually costs*](README.md#what-reasoning-actually-costs)). It
did cut the loop, and the model then made a valid tool call.
But the agent sends the model's reasoning back with each tool step, the chat
template puts it into the next prompt, and the next step looped again within
its first 1,000 characters. The cut-off loop seeded the next one. The plugin
was taken out again, and nothing from it is in this repository.

**The likely cause: the chat template feeds old reasoning back in.** gemma-4's
template keeps the reasoning of every assistant step after the last user
message and renders it as a thought block in the next prompt. In this
checkpoint's `chat_template.jinja` that's lines 239–242:

```jinja
{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}
{%- if thinking_text and thinking_gate -%}
    {{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}
```

The first half of the gate is the one that fires here. The second half,
`preserve_thinking`, is a template option that defaults to off. During a chain
of tool calls there's no new user message, so each step sees all of its own
earlier thinking again, and a loop can feed on itself. The design looks
deliberate: it lets the model carry its reasoning through a tool chain.

A Hugging Face discussion on Google's model page describes exactly this, and
proposes a fix:

- [*Chat template may re-inject prior-turn reasoning during multi-turn tool use → repetition loops*](https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/48),
  google/gemma-4-26B-A4B-it discussion #48, by ManniX-ITA. With the stock
  template, 4 of 12 seeds of a multi-turn agent test looped. With the
  re-injection switched off (`{%- if false and thinking_text … %}`), none did,
  and code benchmarks held up (HumanEval+ 92.07%). That was measured on a
  pruned derivative of the 26B-A4B, not on this checkpoint.
- [The fixed template](https://huggingface.co/ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF/blob/main/chat_template.fixed.jinja)
  and [its unit test](https://huggingface.co/ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF/blob/main/template_loop_unittest.py),
  linked from that discussion.

**Not tested here yet.** In this checkpoint's template, the same fix means
changing line 241 to `{%- if false and thinking_text and thinking_gate -%}`.
The edited copy would be passed to the server with `--chat-template`, so no
client has to change. What it would cost is unknown: the model would no longer
see its own reasoning from earlier steps in the same tool chain.

**Until then:** stop the looping request; the next one normally works. Don't
reach for repetition or frequency penalties as a workaround: they also push
the model away from copying code exactly, which a coding agent depends on.

---

## Open questions

1. **A ~131k prompt on Triton** has never been run to the end; 64.7k is the
   largest. With Intel attention, 129,331 tokens took 51.7 s.
2. **fp8 KV cache, untested.** Triton accepts it, and the check that would
   reject it applies only to CUDA. It would halve the KV memory; the accuracy
   cost is unknown. It needs a `--kv-cache-dtype` setting in compose.
3. **The draft model on the B60.** Its 0.78 GiB would most likely come out of
   the 4.25 GiB KV setting, which means a lower context cap (inferred, not
   booted).
4. **Report the head-512 check upstream.** The kernel is compiled in, and only
   `head512_plugin/` makes it reachable. A fix upstream would let the plugin go.
5. **Images on Intel attention, beyond nine test images.** The image tokens
   are read left to right, and nine synthetic images gave the same answers as
   Triton. Real photos, several images per request and video are untested.
   vLLM 0.31.0 has a backend that sends image batches to Triton and the rest
   to Flash Attention (`TRITON_FLASH_ATTN`), but only for NVIDIA. See
   [PLUGIN.md](PLUGIN.md#images-and-the-left-to-right-rule).
6. **gemma-4's thinking loops.** Does the template fix from Hugging Face
   discussion #48 stop them on this checkpoint, and what does it cost? Do the
   loops happen on Triton attention too? See *Known issue: thinking loops in
   agent sessions*.

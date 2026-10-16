# Quantizing the output layer and the draft head

`quantize_heads.py` makes a faster local copy of an int4 checkpoint. Most int4
builds quantize the body of the model but keep two parts at 16-bit:

- **the output layer** (`lm_head`), which turns the model's last state into a
  score for every token in the vocabulary;
- **the MTP draft head**, a small extra layer that guesses the next few
  tokens for speculative decoding.

With MTP on, both are read for the main step and again for every draft
token, so on a dense model they make up a large share of what the card reads
per step. vLLM can't quantize them as it loads (its online quantization skips
the output layer), but it loads a checkpoint that declares them quantized.
This tool writes such a checkpoint next to the original.

It was built and measured with `RedHatAI/Qwen3.8-27B-INT4` on an Arc Pro B70,
and the steps below use that model. Why MTP helps at all, and how it's
switched on, is in
[../QWEN3_8_27B.md](../QWEN3_8_27B.md#faster-decode-mtp-and-an-fp8-output-layer).

## The three variants

| | Output layer | Draft head | Folder it creates | Greedy decode with MTP | Quality vs the original |
|---|---|---|---|---|---|
| **default** | FP8 | 16-bit | `Qwen3.8-27B-INT4-fp8head` | 64.8–67.3 tok/s | same as the original within noise |
| `--mtp-int4` | FP8 | int4 | `Qwen3.8-27B-INT4-fp8head-int4mtp` | 67.6–68.1 tok/s | the same |
| `--head int4 --mtp-int4` | int4 | int4 | `Qwen3.8-27B-INT4-int4head-int4mtp` | 73.5–73.9 tok/s | small, measurable shift |

For comparison, the original checkpoint decodes at 32.9 tok/s without MTP and
54.8–56.5 with it. The default is the recommended one; the last row is the
fastest, and *Measured* below says what it costs.

## Step by step

All commands run from `vllm_openai_xpu/`, inside the engine image, as your
own user. They assume the models live in `~/models/hf` (`HF_CACHE` in
`.env`). With the default named volume instead, use
`-v llm_hf-cache:/cache/huggingface` and leave out `--user` and `-e HOME`.

**1. Get the checkpoint into the cache.** Skip this if the engine has
already served it. Pinning the revision keeps the result the same as the one
measured here:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e HF_HOME=/cache/huggingface -v ~/models/hf:/cache/huggingface \
  --entrypoint hf vllm/vllm-openai-xpu:v0.31.0 \
  download RedHatAI/Qwen3.8-27B-INT4 \
  --revision 91bd022d5b49442a868bc35008f6c21e1860edfa
```

The tool needs exactly one cached snapshot of the repo, and stops otherwise.

**2. Run the tool.** Add `--mtp-int4`, or `--head int4 --mtp-int4`, for the
other variants. It takes a few minutes and about 16 GB of disk:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v ~/models/hf:/cache/huggingface -v "$PWD":/work:ro \
  --entrypoint python3 vllm/vllm-openai-xpu:v0.31.0 \
  /work/tools/quantize_heads.py RedHatAI/Qwen3.8-27B-INT4
```

It prints the rounding error of every layer it changes, then `done:` and the
folder. For this checkpoint: 2.25% for the FP8 output layer, 13.3% for the
int4 one, and 13–15% for each of the draft head's seven layers.

**3. Point the engine at the copy.** In `.env`, in the Qwen3.8 block, with
the folder from step 2 (seen from inside the container):

```dotenv
VLLM_MODEL=/cache/huggingface/local/Qwen3.8-27B-INT4-fp8head
VLLM_SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}
VLLM_KV_CACHE_MEMORY=8603448832
```

The served name doesn't change, so clients keep asking for `qwen3.8-27b`.

**4. Start it and check.**

```bash
docker compose up -d
cd .. && ./smoke.sh
```

The log should show `Model loading took 17.18 GiB` (16.6 with `--mtp-int4`,
15.8 with `--head int4 --mtp-int4`; the original is 17.56) and
`KV cache size: 208,093 tokens`. `smoke.sh` should report `qwen3.8-27b ->
/cache/huggingface/local/…` and ALL PASS.

**To undo it,** point `VLLM_MODEL` back at `RedHatAI/Qwen3.8-27B-INT4` and
delete the folder. Nothing else was changed.

## What happens to the Red Hat checkpoint

The original snapshot is only read. The copy is a folder of links back to it,
plus the few files the tool rewrites:

| File in the copy | What it is |
|---|---|
| `tokenizer.json`, `chat_template.jinja`, `generation_config.json`, … | links to the original, unchanged |
| `config.json` | rewritten: `lm_head` (and with `--mtp-int4` the `re:^mtp.*` entry) leave the `ignore` list, and a config group declares the output layer's new format |
| `model.safetensors.index.json` | rewritten: points the changed weights at their new files |
| `lm_head-fp8.safetensors` or `lm_head-int4.safetensors` | new: the output layer, 248,320 × 5,120. 2.37 GiB at 16-bit becomes 1.19 GiB in FP8 or 0.61 GiB in int4 |
| `model.safetensors` | rewritten without the output layer, 15 GB. This checkpoint keeps almost all its weights in this one file, so taking the layer out means writing the whole file again; that's where the 16 GB goes |
| `model_mtp.safetensors` | with `--mtp-int4` only: the draft head, 0.79 GiB at 16-bit, 0.20 GiB in int4. Otherwise a link |

The formats are the ones vLLM already handles for this checkpoint:

- **FP8:** one scale per row, `float8_e4m3fn` values. It runs on Intel's
  native FP8 kernel (`fp8_gemm_w8a16`).
- **int4:** exactly the checkpoint's own int4 format, symmetric, one scale
  per group of 128 weights, packed eight to an int32, so it runs on the same
  kernel as the rest of the model. Unpacking one of the checkpoint's own
  layers and packing it again with this code gives back the same bytes.

The values are plain rounding to the nearest step. Red Hat quantized the rest
of the model with a calibrated process instead, AWQ smoothing followed by
GPTQ (see the checkpoint's `recipe.yaml`). That's why the int4 rounding error
here is high, and why the output layer in particular shows it (*Measured*).

## Measured

On the B70 on 2026-10-03.

**Speed,** with MTP and 3 draft tokens: three different prompts, each sent
once, 512 tokens, thinking off, decode timed from the first token to the last,
median per run:

| | default sampling | greedy |
|---|---|---|
| original, no MTP | 32.2 | 32.9 |
| original, MTP | 53.4–53.9 | 54.8–56.5 |
| default (FP8 output layer) | 53.1–62.1 | 64.8–67.3 |
| `--mtp-int4` | 60.8–63.3 | 67.6–68.1 |
| `--head int4 --mtp-int4` | 71.5–72.2 | 73.5–73.9 |

The int4 draft head adds only about 3%, less than its size suggests. The
draft steps are small, so fixed overhead per step matters more there than
bytes.

**Quality,** with MTP off, so the output layer is the only difference: the
first 100 GSM8K math problems (greedy), and the next-token ranking at 17,911
positions of the same texts, each compared with the original:

| | same top-ranked next token | average score drift | GSM8K |
|---|---|---|---|
| the same setup run twice (the noise floor) | 99.62% | 0.007 | 96/100 |
| FP8 output layer | 98.83% | 0.022 | 96/100 |
| int4 output layer | 96.76% | 0.063 | 96/100 |

- The FP8 output layer stays close to the noise floor.
- The int4 one changes the top-ranked token at about 3% more positions, with
  three times FP8's drift. These are mostly near-ties between two good words,
  which is why the math score didn't move. But 100 problems can only show a
  large loss, so a small one, or a change in style or recall, isn't ruled out.
- The draft head can't change answers on its own: the main model checks every
  draft. A near-tie can still tip the other way when different drafts change
  how many tokens are checked at once.
- Two identical runs already word half of the longer answers differently,
  because requests running side by side change the arithmetic slightly. So
  whether answers come out word for word the same isn't a useful test here.

## Other checkpoints

The tool checks what it needs and stops with a message otherwise:

- compressed-tensors format, with `lm_head` in the `ignore` list;
- an output layer of its own, not tied to the embeddings;
- for anything int4: the checkpoint's own `Linear` layers quantized as
  symmetric, grouped, pack-quantized int4, whose scheme it copies;
- for `--mtp-int4`: `mtp.*` weights in the checkpoint.

Only `RedHatAI/Qwen3.8-27B-INT4` has been run through it. Another checkpoint
may pass these checks and still need its own boot and quality test before
it's trusted.

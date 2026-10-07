# The head512 plugin

`head512_plugin/` is a small vLLM plugin: about 35 lines of Python and two short
text files. It removes one check that stops gemma-4 from using Intel's own
attention kernel, and that is all it does. It doesn't change any file in the
image, it doesn't install anything, and on its own it doesn't switch anything
on. The `.env` line in
[*Intel attention for gemma-4*](GEMMA_4_26B_A4B.md#intel-attention-for-gemma-4) does the
switching; the plugin is what makes that switch possible.

## The problem it solves

### Two kinds of layer in gemma-4

gemma-4 works through text in 30 stages, called layers. In 25 of them, each
token can only look back at the 1,024 tokens just before it. In the other five,
at positions 6, 12, 18, 24 and 30, each token can look back at the whole
conversation:

```
layer:  1  2  3  4  5  [6]  7  8  9 10 11 [12] ... 25 26 27 28 29 [30]
        short-range    full  short-range   full       short-range  full
```

Think of reading a long book. Most of the time you only keep the last page or
two in mind; that's the 1,024-token window, roughly 700–800 English words. Every
sixth layer, the model flips back through the whole book to connect what it's
reading now with something from chapter one. That's what the five full layers
do.

Head size is how detailed each of those lookups is. In every layer, each token
is turned into lists of numbers that say what it's looking for and what it
contains, and tokens are matched by comparing those lists. The head size is the
length of one list. A layer runs 16 of these heads side by side, each looking
for a different kind of connection. The short-range layers use 256 numbers per
head and the full layers use 512, so each comparison in a full layer costs about
twice as much.

Those five full layers matter far more than their number suggests. They compare
every token with every earlier token, so their work grows with the square of the
prompt length: on a 131k-token prompt they do about 93% of all the attention
work. They're also the only way the model can find anything more than 1,024
tokens back. So whichever kernel runs them decides how fast long prompts are.

### The check that gets in the way

vLLM's Flash Attention backend decides which head sizes it can run in a function
called `supports_head_size()`. Up to 256 the answer is yes, so the short-range
layers were never the problem. Above 256 it's yes only if FlashAttention 4 is
available, and FA4 only exists for NVIDIA GPUs. On an Intel GPU the check tries
to import the CUDA flash-attention module, the import fails, and the answer is
no. vLLM won't run some layers on one backend and the rest on another, so it puts
all 30 layers of gemma-4 on Triton, which is slow on long prompts: on this card
prefill time grows roughly with the square of the prompt length.

The odd part is that the kernels were already there. Intel's kernel library in
the image, `vllm_xpu_kernels/libattn_kernels_xe_2.so`, contains head-512 prefill
kernels (`chunk_policy_head512`) and a head-512 decode kernel
(`paged_decode_kernel_template_q8_h512_p64`) built for exactly gemma-4's layout:
8 query heads per KV head and 64-token KV blocks. vLLM never used them because
the check asks "are you FlashAttention 4?" when the question that matters is
"can you run head size 512?".

No setting gets past it. `--attention-backend=FLASH_ATTN` is accepted and then
refused at that same check, and vLLM fixes Intel GPUs at FlashAttention version 2
before it looks at any version setting. Changing the code was the only way in.

It was worth it. On the B70, prefill got 16–28× faster and decode about 16%
faster. The full numbers are in
[GEMMA_4_26B_A4B.md](GEMMA_4_26B_A4B.md#intel-attention-for-gemma-4).

## What's in the folder

```
head512_plugin/
├── vllm_xpu_head512.py
└── vllm_xpu_head512-1.0.dist-info/
    ├── METADATA
    └── entry_points.txt
```

**`vllm_xpu_head512.py`** is the only real code. Its `register()` function wraps
`FlashAttentionBackend.supports_head_size()` so that 512 is accepted, while
every other size is still decided by vLLM's original function:

```python
def supports_head_size(cls, head_size: int) -> bool:
    return head_size == 512 or original(cls, head_size)
```

It allows exactly 512, not everything up to 512, because 512 is the only size we
have confirmed Intel kernels for. It returns straight away on a machine without
an Intel GPU, and it marks the function it installs so it never wraps it twice in
the same process. When it runs, it logs:

```
xpu_head512: FLASH_ATTN on XPU now accepts head size 512
```

**`METADATA`** is four lines: a name, a version and a one-line summary. A
`.dist-info` folder is how Python records an installed package, so with this
folder on the path Python treats the plugin as installed, and nobody has to run
`pip install` inside the image.

**`entry_points.txt`** is two lines. It registers `register()` in the
`vllm.general_plugins` group, which vLLM runs at startup:

```ini
[vllm.general_plugins]
xpu_head512 = vllm_xpu_head512:register
```

## How vLLM finds it

[`compose.yaml`](compose.yaml) does two things for the plugin: it mounts
`head512_plugin/` read-only at `/opt/vllm-plugins`, and it sets
`PYTHONPATH=/opt/vllm-plugins`.

When vLLM starts, every process it launches calls `load_general_plugins()`. That
asks Python for every entry point in the `vllm.general_plugins` group, finds
ours through the `.dist-info` folder, and calls `register()`. With one GPU vLLM
runs two processes, the API server and the engine core, so the log line shows up
twice. If it's missing, the plugin didn't load.

One thing to watch: vLLM loads every general plugin unless the `VLLM_PLUGINS`
environment variable is set. If you ever set it, put `xpu_head512` in the list.
Otherwise the plugin is skipped without a word, and a boot with Intel attention
switched on fails at the head-size check.

## Turning it on and off

The plugin is always loaded, and it's harmless when nothing asks for Flash
Attention. With the compose defaults, gemma-4 still runs on Triton. gpt-oss
isn't affected either way, because its head size is 64.

This `.env` line turns Intel attention on:

```dotenv
VLLM_ATTN_BACKEND=FLASH_ATTN
```

Then run `docker compose up -d`. To go back, set it to `auto` and run it again.

Image input keeps working, with one difference in how the image is read. The
last section,
[*Images and the left-to-right rule*](#images-and-the-left-to-right-rule),
explains it.

## Why a plugin and not a patched file

The first test, the A/B comparison in the README, mounted a patched copy of the
whole `flash_attn.py` over the original. That works, but it freezes 2,056 lines
of vLLM at one version. After the next image upgrade the old copy would quietly
replace the new file, and whatever upstream changed in it would be lost without
any warning.

The plugin changes one function while vLLM runs and leaves everything else to
the image. It also goes through vLLM's own plugin system, the same mechanism
vLLM's documentation uses to add an out-of-tree model, so it isn't fighting
the code it extends. If a future vLLM renames that class or function, the
plugin's import should fail at startup and stop the boot, which is loud rather
than silent. (That last part is from reading the source; it hasn't been tried.)

## After an image upgrade

Check the boot log before trusting the numbers. GEMMA_4_26B_A4B.md lists what
to look for under [*Intel attention for gemma-4*](GEMMA_4_26B_A4B.md#intel-attention-for-gemma-4).
In short: the `xpu_head512` line means the plugin loaded,
`Using Flash Attention backend.` means it took effect, and there must be no
`XPU kernel not compiled … falling back` warning. That warning means a kernel
variant is missing from the new image and you're on a slow reference path.

## When to delete it

The plugin is only needed because of how vLLM asks the question. Once vLLM asks
Intel's kernels what they support instead of asking about FA4,
`--attention-backend=FLASH_ATTN` will work on its own. At that point delete
`head512_plugin/`, plus the mount and the `PYTHONPATH` line in `compose.yaml`.

## How this came about

It started with a plain question: why does gemma-4 run on Triton when gpt-oss,
on the same engine, gets Flash Attention? The boot log answers half of it:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}.
FA4 not available, forcing TRITON_ATTN backend.
```

Following that message through vLLM's source inside the image led to the
head-size check above. The surprise came from looking at Intel's kernel library
next. The head-512 kernels were already compiled for this GPU, with no way to
reach them.

The first try was the patched-file test. It showed the kernel works and gives
the same answers: the same answers at temperature 0, `smoke.sh` passing with
reasoning on and off, and a code hidden in documents up to 129k tokens long found
every time. After that the change was rewritten as this plugin so it could stay.

## Read the source

In this repo:

- [`head512_plugin/vllm_xpu_head512.py`](head512_plugin/vllm_xpu_head512.py)
- [`head512_plugin/vllm_xpu_head512-1.0.dist-info/entry_points.txt`](head512_plugin/vllm_xpu_head512-1.0.dist-info/entry_points.txt)
- [`compose.yaml`](compose.yaml), the plugin mount and `PYTHONPATH`
- [*Why the slow kernel gets chosen*](GEMMA_4_26B_A4B.md#why-the-slow-kernel-gets-chosen)
  in the README, the full chain of checks

In vLLM v0.31.0, the version this image is built from:

- The head-size check the plugin wraps:
  [`supports_head_size()`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/flash_attn.py#L420-L428)
- The image-input check it leaves alone:
  [`supports_mm_prefix()`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/flash_attn.py#L440-L442)
- Where vLLM lets Flash Attention take images anyway when it's asked for by
  name: [`get_attn_backend_cls()`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/platforms/xpu.py#L184-L202)
- Why FA4 always comes back "no" on Intel, the failed import:
  [`is_fa_version_supported()`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/fa_utils.py#L271-L279)
- Why no setting can ask for another version, Intel is fixed at 2:
  [`get_flash_attn_version()`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/fa_utils.py#L83-L84)
- The plugin loader:
  [`load_general_plugins()`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/plugins/__init__.py#L77-L90)
- vLLM's own guide to plugins:
  [Plugin System](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/design/plugin_system.md)

The head-512 kernels themselves aren't on GitHub in readable form here; they
are compiled into `libattn_kernels_xe_2.so` inside the image.

## Images and the left-to-right rule

gemma-4 reads a picture differently from text. An image arrives as a block of
tokens in the prompt, 258 of them in this setup whatever the image size. Text is
read strictly left to right: each token can only look at the tokens before it,
never ahead, because when the model writes it can't see words it hasn't written
yet. That's called causal attention, and it's the one pattern every attention
kernel supports.

A picture has no left-to-right order, though. The top-left corner means more once
you've seen the bottom-right. So gemma-4 lets the tokens of one image look at
each other in both directions, while everything else stays causal. Its config
says so: `use_bidirectional_attention: vision`.

```
prompt:  text text text | img img img img img | text text
         each looks     | all image tokens    | each looks back at
         back only      | see each other      | everything before
```

That mix needs a kernel that can be told where each image starts and ends, and
that drops the left-to-right rule inside it. vLLM calls this "mm_prefix" and asks
each backend whether it can do it, through `supports_mm_prefix()`. Triton can.
Flash Attention says yes only with FA4, the same NVIDIA-only question as the
head-size check, so on an Intel GPU the answer is no. The plugin leaves this
check alone on purpose: nobody has found a way to make Intel's kernel apply the
image rule.

**On vLLM 0.30.0** that meant a choice. Asking for Flash Attention with images
on stopped the boot with *"mm_prefix (PrefixLM bidirectional attention)
requires FlashAttention v4"*. `--language-model-only` was the way out: it tells
vLLM to treat gemma-4 as a text model, so the image rule is never needed, and
images get `400 At most 0 image(s) may be provided in one prompt`.

**From vLLM 0.31.0** the choice is gone. When Flash Attention is asked for by
name, vLLM uses it for gemma-4 with images on, reads the image tokens left to
right like text, and warns in the log:

```
Using Flash Attention on XPU for a multimodal prefix-LM model because it was
explicitly requested. The prefix-LM bidirectional mask cannot be applied, so
image/video inputs will produce incorrect results
```

The warning sounds worse than what was measured. Nine test images gave the same
answers on Intel attention as on Triton, which applies the rule correctly; the
numbers are in
[*Images with Intel attention*](GEMMA_4_26B_A4B.md#images-with-intel-attention).
A likely reason, not proven: the image encoder already lets the parts of a
picture see each other before the tokens reach the language model.

The proper fix exists in 0.31.0 but only for NVIDIA. A backend called
`TRITON_FLASH_ATTN` sends every batch that holds image tokens to Triton and the
rest to Flash Attention, so images get the rule and text keeps the fast
kernel. It asks for FlashAttention 3 or 4 and an NVIDIA Hopper GPU, and the
Intel backend list never offers it. Making it work here would take a second
plugin, and a check that both backends can share one KV cache layout on XPU.
That hasn't been tried.

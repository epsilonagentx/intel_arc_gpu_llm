# Stock Intel vLLM engine

This folder runs Intel's own published vLLM image, `intel/vllm:0.21.0-ubuntu24.04`.
No fork, no release candidate. It serves `gpt-oss-20b` on an OpenAI-compatible
API at `:8000`, the same as the other two engines in this repo, so anything
sitting in front of that port doesn't care which one is running.

Think of this as the baseline you fall back to. It's the oldest vLLM of the three
and the least interesting, which is exactly the point: when something breaks on
one of the other engines, this is the one you bring up to find out whether the
problem was the engine or the setup around it. It's also the only one configured
with `restart: unless-stopped`, so it comes back on its own after a reboot.

| Doc | What it covers |
|-----|----------------|
| [README.md](../README.md) | The stack as a whole — which engine to pick, monitoring, firewall |
| [DEVELOPER.md](../DEVELOPER.md) | Why the numbers are what they are |
| **This file** | Running *this* engine: start it, upgrade it, swap its model, what bites |

> **One GPU, one engine.** `gpt-oss-20b` needs roughly 17 GiB of the card's
> ~22.7 GiB, so this service and the ones in `scaler/` or `vllm_openai_xpu/`
> cannot run together. Bring the other one down first. All three deliberately
> share the Compose project name `llm` so they reuse the same weight cache —
> which also means **never pass `--remove-orphans`**, because it will cheerfully
> delete the other engine's container.

---

## Running it

```bash
cd vllm_xpu
docker compose up -d vllm                    # start
docker compose logs -f vllm                  # follow startup
docker compose stop vllm                     # stop
docker compose up -d --force-recreate vllm   # apply a config edit
```

Or from the repo root with `-f vllm_xpu/compose.yaml` in place of the `cd`. Either
way Compose reads `vllm_xpu/.env` if present — the project directory is the
compose file's folder, not wherever you ran the command from.

**Name the `vllm` service explicitly on `stop`.** All three engines share the
project name `llm`, so a bare `docker compose stop` is a wider blast radius than
it looks.

Then verify from the repo root, where the scripts live:

```bash
cd ..
./smoke.sh            # served? content? reasoning? tool calls?
./bench.sh 400        # TTFT and tok/s
```

### What the startup signals actually mean

The healthcheck flips to healthy once `/health` returns 200. That means the model
is **served** — it does not mean compilation is finished. The first request after
any (re)start triggers ~30–60 s of torch.compile work; everything after that is
fast.

**First run on a fresh cache is silent for 10–15 minutes.** No log output at all
while oneAPI/SYCL initialises, which is why the healthcheck allows a 30-minute
start period. It has not hung — the root README's *"is it stuck or working?"*
section has the `/proc` reads that prove it. Resist restarting; you'd throw away
the compile work and start the silence over. After the first run,
`SYCL_CACHE_PERSISTENT=1` plus the `vllm-cache` volume cut restarts to ~30 s.

Because the service is `restart: unless-stopped`, it auto-starts when the Docker
daemon does. An explicit `docker compose stop vllm` is what keeps it down.

---

## What it serves, and why

`gpt-oss-20b` is a mixture-of-experts model shipping pre-quantised in MXFP4. Two
things follow from that, and together they're why this model fits so comfortably
on a 24 GB card:

- **No quantisation step at load.** The weights are already 4-bit, about 13 GB to
  fetch and 12.87 GiB resident once loaded. Nothing has to be converted in host
  RAM on the way in.
- **Cheap KV cache.** Its layers alternate between sliding-window and full
  attention, which roughly halves the per-request cost of a long context compared
  with an all-full-attention model of the same size.

So the full 131,072-token context — the model's native maximum — fits without any
tuning gymnastics, and there's still pool left over for prefix-cache reuse across
multi-turn chats.

The engine is started with `--reasoning-parser openai_gptoss`, which lifts
gpt-oss's internal analysis channel out into `message.reasoning`. Clients that
know about that field, Open WebUI among them, render it as a collapsible panel
instead of dumping the model's scratch work into the reply.

## What it won't do

It won't load gemma-4. That's an engine-version limit, not a memory or
quantisation problem — this image is built on a vLLM old enough that the
architecture isn't registered in it. If you want gemma-4, use
[`vllm_openai_xpu/`](../vllm_openai_xpu/README.md), which runs a much newer
upstream vLLM.

It's also not the fastest way to serve `gpt-oss-20b`. Intel's `llm-scaler` fork
in [`scaler/`](../scaler/README.md) is tuned for Arc B-series and measurably beats
it on the same model and the same card. The two are drop-in replacements for each
other — identical model name, identical port — so swapping between them needs no
change downstream.

---

## Configuration — `.env`

Everything configurable is a value in `.env`, interpolated into `compose.yaml`.
The flag names stay in the compose file; only values come from the environment.
With no `.env` present the defaults render exactly the validated command, so the
file runs as-is. Start from `.env.example`, which documents every variable with
its default.

**`VLLM_GPU_MEMORY_UTILIZATION` (default `0.80`) — the one with teeth.** This
sizes the weights plus the KV pool, but on this XPU build it does **not** cap
torch.compile's kernel and workspace buffers, and those keep growing as new
request shapes get compiled. At `0.86` the card was measured filled to 22.67 of
22.71 GiB, leaving 40 MiB of headroom, and the result was OOM-on-the-edge
behaviour and 504s under load. Treat `0.86` as a wall you never walk up to.
`0.80` is comfortable now that the displays are driven by the integrated GPU and
nothing else competes for VRAM; if you're sharing the card with a desktop
session, come down to `0.75`.

**`VLLM_MAX_MODEL_LEN` (default `131072`).** The model's native ceiling. Raising
context is close to free here for the reasons above, so there's rarely a reason to
lower it — but note a cap is a guardrail on the longest prompt a client can
submit, not a speed setting. Decode rate doesn't change with it.

---

## Swapping the served model

Two steps: set the model-specific values in `.env`, then force-recreate. The
compose file is never edited.

**Step 1 — the model-specific variables:**

| Variable | What it sets |
|----------|--------------|
| `VLLM_MODEL` | Hugging Face repo ID (e.g. `openai/gpt-oss-20b`) |
| `VLLM_SERVED_MODEL_NAME` | The name clients call it by; what a gateway's model mapping points at |
| `VLLM_REASONING_PARSER` | Model-family specific. Wrong parser = empty reasoning field, **not** a crash |
| `VLLM_TOOL_CALL_PARSER` | Model-family specific, same quiet failure mode |
| `VLLM_MAX_MODEL_LEN` | Context window — must fit VRAM after weights and compile buffers |
| `VLLM_GPU_MEMORY_UTILIZATION` | See the warning above before raising it |

**Step 2 — recreate:**

```bash
docker compose up -d --force-recreate vllm
```

`--force-recreate` is not optional. vLLM bakes its CLI arguments into the
container at creation, so a plain `up -d` finds a running container, leaves it
alone, and **your `.env` edit appears to do nothing.**

**If the model is already cached**, that's the whole procedure — no re-download.
Compile artifacts in `vllm-cache` are model-specific, so the first request after
a swap still recompiles; the volume just stops it being completely cold.

**If it isn't cached**, the first `up -d` downloads the weights into `hf-cache`.
Budget ~10–30 min of download, plus the silent 10–15 min cold start, plus ~30–60 s
of first-request compile.

### Models validated on this engine

| HF repo | On-disk | `--reasoning-parser` | `--tool-call-parser` | Reasoning |
|---------|---------|----------------------|----------------------|-----------|
| `openai/gpt-oss-20b` | ~13 GB MXFP4 | `openai_gptoss` | `openai` | Always on; effort via `reasoning_effort` |
| `Qwen/Qwen3-32B-AWQ` | ~19 GB | `qwen3` | `hermes` | Hybrid; `/no_think` disables |

Sizes are on-disk cache footprint, not loaded weights.

### Worked example: gpt-oss-20b ↔ Qwen3-32B-AWQ

With no `.env` overrides the compose defaults serve gpt-oss-20b. To serve
Qwen3-32B-AWQ instead:

```dotenv
VLLM_MODEL=Qwen/Qwen3-32B-AWQ
VLLM_SERVED_MODEL_NAME=qwen3-32b
VLLM_MAX_MODEL_LEN=7168
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_REASONING_PARSER=qwen3
VLLM_TOOL_CALL_PARSER=hermes
```

…then force-recreate. What changed, and why each one matters:

- **Context 131,072 → 7,168.** The empirical B60 cap for this model; 10k and 12k
  both fail vLLM's KV pre-check at startup. Qwen3-32B is **dense**, so unlike
  gpt-oss it gets no sliding-window discount on KV — that one architectural
  difference is the whole reason the context collapses by 18×.
- **Reasoning parser → `qwen3`.** Qwen3 is hybrid-thinking (`/no_think` in the
  prompt turns it off). The `openai_gptoss` parser would leave the reasoning field
  silently empty.
- **Tool parser → `hermes`.** Qwen3 emits Hermes-style tool calls, not gpt-oss's
  `openai` format. The image also ships `qwen3_xml` and `qwen3_coder`; the latter
  is only for Qwen3-**Coder**.
- **Utilisation 0.80 → 0.9.** Qwen3-32B-AWQ's weights are ~18 GiB, so 0.80 leaves
  too little for a usable pool. ⚠ **This is over the 0.86 OOM edge described
  above** — it is a tight fit on 22.7 GiB and the reason the context is only
  7,168. Watch real VRAM and size it empirically rather than trusting this
  number. You must also unset any `--kv-cache-memory-bytes` pin, which is
  absolute and OOMs rather than shrinking.
- **AWQ, not FP8.** The official `Qwen/*-FP8` weights hit an XPU bug on this
  image. AWQ is the working path.

Swapping back is commenting those lines out again — the compose defaults *are*
the gpt-oss-20b config — and force-recreating.

---

## Upgrading the image

Pinned to `intel/vllm:0.21.0-ubuntu24.04`. Two things changed from the earlier
`0.17.0-xpu` and both are already baked into `compose.yaml`, but they'll bite if
you bump the image yourself.

**Device passthrough.** 0.21.0 needs the **whole `/dev/dri`** plus a read-only
bind of `/dev/dri/by-path`. oneCCL's device manager calls `opendir()` on
`/dev/dri/by-path` during its start-up collective even with a single GPU, and
Docker's `devices:` directive never recreates that symlink directory. Leave it out
and boot dies with `oneCCL: ze_fd_manager … opendir failed`. `0.17.0-xpu` booted
with just the `renderD128`/`card1` nodes.

Exposing the whole directory also hands the container any other render nodes on
the box, such as an integrated GPU. Harmless — they aren't SYCL devices and vLLM
ignores them.

**Compile cache.** torch.compile kernels are image-version-specific, so clear the
old cache once on upgrade:

```bash
docker compose down
docker volume rm llm_vllm-cache
docker compose up -d vllm
```

The first request then runs the usual ~30–60 s compile and the cache repopulates.
In steady state it refills itself and speeds up restarts, so there's no reason to
clear it otherwise.

---

## Things that are specific to the host, not to you

Two values in `compose.yaml` are properties of the machine it was written on and
will very likely be wrong on yours:

```yaml
group_add:
  - "992"   # host 'render' GID
  - "44"    # host 'video' GID
```

Check yours with `getent group render video` and edit accordingly. Getting these
wrong shows up as permission errors opening the GPU device nodes, which is not an
obvious symptom.

Unlike the upstream engine, this one runs happily with
`SYCL_CACHE_PERSISTENT=1`, which is why the setting is enabled here.

---

## Volumes

- **`hf-cache`** holds the downloaded model weights and is shared with the other
  two engines on purpose, so swapping engines doesn't re-download anything. It's
  also why `docker compose down -v` from *any* engine folder is a bad idea: it
  takes the shared weights with it. Use a bare `down`.
- **`vllm-cache`** holds compiled kernels and belongs to this engine alone. See
  *Upgrading the image* for when to clear it.

---

## When it misbehaves

**Compose warns about orphan containers.** Expected, and not a problem — it's the
side effect of three engines sharing one project name. Stop the other engine from
*its* folder; don't take Compose's suggestion to prune orphans.

**`.env` edit had no effect.** Missing `--force-recreate`.

**Port 8000 is already taken.** Something else is serving on it. A local
`llama.cpp` setup is the usual culprit, and if it's under a process supervisor it
will come back by itself after you kill it — stop it properly rather than killing
the process.

**`HTTP 404 ... model does not exist`.** The served name doesn't match what the
client is asking for. `./bench.sh` prints what *is* served.

**Reasoning field is empty.** Either the wrong `--reasoning-parser` for the model
family, or a client reading `reasoning_content` — the field is **`reasoning`**.

**Silence on first boot.** Covered above. Thirty minutes is budgeted for a reason.

**The card looks idle while the model is clearly working.** Container GPU
utilisation doesn't show up in the usual host-side monitors. Read the sysfs
frequency nodes instead; the root [README](../README.md) covers this.

---

The compose file is commented in full and is the authoritative record of why each
value is what it is — this file is the orientation, that file is the detail. For
the stack as a whole, monitoring, firewall and the optional chat UI, see the root
[README](../README.md); for the design rationale behind the numbers, see
[DEVELOPER.md](../DEVELOPER.md).

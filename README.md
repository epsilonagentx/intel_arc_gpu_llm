# Local LLM stack

> ⚠️ The only official source is [github.com/epsilonagentx/intel_arc_gpu_llm](https://github.com/epsilonagentx/intel_arc_gpu_llm); copies elsewhere are not maintained by me.
>
> 💡 Using it? **Fork** the repo (don't just download a copy) and work on your own branch — that keeps you linked to upstream for updates and makes contributing back easy. See [how to fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo), or [fork this one directly](https://github.com/epsilonagentx/intel_arc_gpu_llm/fork).

Hardware: Intel Arc Pro B60 (24 GB VRAM, `xe` driver). **Host OS: Linux only** —
any modern distribution with Docker and the Intel `xe` GPU driver. Windows and
macOS are not supported: the `xe` kernel driver and the sysfs/hwmon helper
scripts (`watt.sh`, the troubleshooting `/proc` reads) are Linux-specific.
Three interchangeable engine containers ship here; *Choosing and running an
engine* below covers which to pick and where each one's own guide lives. This
is the **how-to** for running and operating the stack. The *why* behind the
config (VRAM sizing, the util decision, quantisation choices) is in
[DEVELOPER.md](DEVELOPER.md); measurements and the record of what has already
been tried and rejected on the scaler engine are in
[scaler/README.md](scaler/README.md); a configuration overview is
in [INTEL_ARC_B60.md](INTEL_ARC_B60.md).

The stack runs **one** vLLM engine at a time on port 8000 (LAN-exposed), chosen
from **three** interchangeable images — stock `intel/vllm`, Intel's `llm-scaler`
fork, or upstream's own `vllm-openai-xpu`. An optional chat UI (Open WebUI) ships
as a **separate** Compose project you can bring up alongside it — see *Running
Open WebUI (optional)* below.

**Currently served: `gemma-4-26b-a4b`** on the upstream engine. The other two
engines are configured for `gpt-oss-20b`; gemma-4 runs only on upstream, which
is the sole engine here whose kernels load it. Model names are kept truthful —
one served id per model — so a swap is visible to downstream clients rather than
silently changing the model behind a fixed label.

---

## Choosing and running an engine

Three interchangeable vLLM engines, each in its own folder, all publishing
`:8000`. **Each folder has its own README, and that is where the operator detail
for that engine lives** — how to run it, upgrade it, swap its model, and what
bites. Pick one and start there:

| folder | image | serves | pick it for |
|---|---|---|---|
| [`vllm_xpu/`](vllm_xpu/README.md) | stock `intel/vllm` 0.21.0 | `gpt-oss-20b` | the conservative baseline |
| [`scaler/`](scaler/README.md) | `intel/llm-scaler-vllm` 0.26.0-b2 | `gpt-oss-20b` | **fastest** for gpt-oss (85.6 tok/s) |
| [`vllm_openai_xpu/`](vllm_openai_xpu/README.md) | `vllm/vllm-openai-xpu` v0.30.0 | `gemma-4-26b-a4b` | **currently live**; the only one that loads gemma-4 |

**One GPU → exactly one engine at a time.** Each needs 13–17 GiB of weights plus
its KV pool; together they OOM. Every command targets a folder, so you can't
start two by accident. All three share the Compose project name `llm` on purpose
so they reuse one weights cache — which is also why you must **never pass
`--remove-orphans`**: in a shared project it deletes the other engines.

From the repo root, with `<folder>` being one of the three above:

```bash
docker compose -f <folder>/compose.yaml up -d      # start
docker compose -f <folder>/compose.yaml logs -f    # follow startup
docker compose -f <folder>/compose.yaml down       # stop
```

Never `down -v` — the weights volume is shared, so it takes every engine's cached
models with it.

Swapping engines is a `down` in one folder and an `up -d` in another. **Any of
the three can follow any other** — they are peers, not a base plus alternatives.

First find out which one is up, rather than assuming:

```bash
docker ps --filter name=vllm --format '{{.Names}}\t{{.Image}}'
curl -s localhost:8000/v1/models        # served id, and `root` = the real checkpoint
```

Then stop it and start the one you want:

```bash
docker compose -f <current>/compose.yaml down     # only one is ever up
docker compose -f <wanted>/compose.yaml up -d     # any of the three folders above
# first boot on a cold cache is 10-15 min of silence, no logs — wait it out
./bench.sh 400     # speed
./smoke.sh         # reasoning + tool-calling still work
```

**Which models an engine can serve, and how to point it at a different one, is
documented in that engine's own README** — and the three do not work the same
way: `vllm_xpu/` and `vllm_openai_xpu/` read their model from a `.env` file,
while `scaler/` has its model written into the `command:` block of its compose
file. Go to the folder before changing a model.

`vllm_xpu/` and `scaler/` both serve gpt-oss-20b, so they are drop-in
replacements for each other and downstream clients need no change when you swap
between them. **Switching to or from `vllm_openai_xpu/` changes the served model
name**, so a gateway's model mapping has to move with it — that visibility is
deliberate, since the models differ in far more than their name.

### Which one, and why

`vllm_xpu/` is the stock Intel image and the conservative baseline: oldest vLLM
of the three, and the one to bring up when you need to know whether a problem is
the engine or the setup around it.

`scaler/` is Intel's fork, tuned for Arc B-series. On this hardware it is the
faster of the two gpt-oss engines — `0.26.0-b2` measures **85.6 tok/s**
single-stream on `./bench.sh 400` at ~72 ms TTFT, the best figure recorded here —
and its image unlocks quantised-MoE paths the stock one lacks. It has
engine-specific rules that matter before you run it in anger (**do not use
`:latest`**, why it must stay `--enforce-eager`, and how to re-derive its KV-pool
byte value); all of them, with the measurement log, are in
[scaler/README.md](scaler/README.md).

`vllm_openai_xpu/` runs upstream's own image and is what's live today. It is
several vLLM minors newer than the others and the only engine here whose kernels
load gemma-4. Its guide covers both models it's validated for, and the quadratic
prefill behaviour that gemma-4 brings with it —
[vllm_openai_xpu/README.md](vllm_openai_xpu/README.md).

**Benchmark before adopting any of them.** The win has to be measured on your own
box, and the three sit on **different vLLM bases** (scaler → 0.26.0, stock →
0.21.0, upstream → 0.29.0), so any difference mixes a fork's optimisations with an
engine-version gap. Compare at equal `max_tokens`, and discard the first run after
a cold start or a long idle — otherwise the numbers lie. The hygiene rules are in
[scaler/README.md](scaler/README.md).

---

## Benchmarking — `bench.sh`

Runs one streamed chat-completion request and reports TTFT, decode rate, and
token counts. It splits the reasoning stream from the content stream, so
reasoning-native (gpt-oss) and hybrid-thinking (Qwen3) models are measured
fairly.

```bash
MODEL=gpt-oss-20b ./bench.sh 600                                   # 600 max tokens, default prompt
MODEL=gpt-oss-20b ./bench.sh 600 "Summarize the French Revolution." # custom prompt
MODEL=gpt-oss-20b VLLM_ENDPOINT=http://192.168.x.x:8000 ./bench.sh 600  # remote target
```

- **`MODEL=`** must match `--served-model-name`.
- **First positional arg** = `max_tokens` (default 200). Bump to **600+** for
  reasoning models — reasoning eats most of a small budget before any content
  appears.
- **Second positional arg** = custom prompt.
- **`VLLM_ENDPOINT=`** overrides the endpoint (default `http://localhost:8000`).
  All three engines publish `:8000`, so the same command benchmarks whichever is up.

Two TTFT numbers are printed: `TTFT (any)` = first token of any kind ("is it
alive"), `TTFT (content)` = first user-visible token after reasoning finishes
("how long until the answer appears"). `Decode tok/s (all)` counts reasoning +
content — the right single-stream number for a reasoning-native model. Aggregate
throughput under concurrent load is much higher; this bench is one-user only.

`bench.sh` has no reasoning-effort knob, so it runs at the model default
(`medium`). To compare effort levels, hit `/v1/chat/completions` directly with a
`reasoning_effort` field — see DEVELOPER.md.

---

## Smoke test — `smoke.sh`

A fast end-to-end **correctness** check of the running service (where `bench.sh`
measures *speed*): confirms the model is served, plain chat generates content, the
reasoning trace comes through (`message.reasoning`), and tool-calling emits a
`tool_call`. Handy right after a (re)start, a model swap, an image upgrade, or a
switch between the base and scaler engines.

```bash
./smoke.sh                                          # localhost:8000, model gpt-oss-20b
MODEL=qwen3-32b ./smoke.sh                          # after a model swap
VLLM_ENDPOINT=http://192.168.x.x:8000 ./smoke.sh    # remote target
```

- **`MODEL=`** must match `--served-model-name` (default `gpt-oss-20b`).
- **`VLLM_ENDPOINT=`** overrides the endpoint (default `http://localhost:8000`).
- Exits non-zero if any check fails, so it drops into scripts/CI. The tool-calling
  check assumes the served model supports tools (gpt-oss and Qwen3 both do).

---

## Power & live monitoring

**Power — `watt.sh`** reads the B60's `xe` hwmon energy counters straight from
sysfs (no root, no packages):

```bash
./watt.sh            # 1s samples
./watt.sh 2          # 2s samples
PCI=0000:03:00.0 ./watt.sh   # example BDF — find yours: lspci | grep -i display
```

Ctrl-C prints min/avg/max for the run — handy running alongside `bench.sh`. The
`xe` driver exposes only cumulative energy (µJ), so the script derives watts from
the delta between samples.

**Live utilisation/VRAM — `nvtop`** (v3.0.x or newer) is the working TUI monitor
for the `xe` B60. `intel_gpu_top` does **not** work here (it's i915-only);
Intel's `xpu-smi` is an alternative if you install it.

---

## Reasoning / thinking output

vLLM emits the reasoning trace into **`message.reasoning`** (and
`delta.reasoning` in streams), **not** `reasoning_content` as some vLLM docs
suggest — verified across every engine here, on both older and current Intel
images. Any consumer parsing for `reasoning_content`
sees empty strings while thinking tokens are silently consumed.

Per-family behaviour:

- **gpt-oss** — always reasoning, no off switch. Effort is a request field
  (`reasoning_effort: low|medium|high`, default `medium`) — see
  [DEVELOPER.md](DEVELOPER.md) for its latency behaviour. Reasoning tokens count
  against `--max-model-len`.
- **Qwen3** — hybrid; thinking on by default, `/no_think` in the user message
  disables it.
- **gemma-4** — **opt-in, off by default.** Its chat template defaults
  `enable_thinking` to false, so the reasoning field comes back empty until a
  request asks for it: `"chat_template_kwargs": {"enable_thinking": true}`. An
  empty trace on gemma-4 is this flag being unset, **not** a broken parser.
  Server-side always-on is `--default-chat-template-kwargs '{"enable_thinking":true}'`,
  and a request can still override it either way. Reasoning does not cost decode
  speed here, but it consumes `max_tokens` first — allow 1500+ or you get a trace
  and no answer. See
  [vllm_openai_xpu/README.md](vllm_openai_xpu/README.md).

---

## Clients

The endpoint is OpenAI-compatible, so any OpenAI-style client works. A consumer is
usually one of two kinds — an AI gateway in front of it, or a containerized
tool/UI that talks to it directly:

- **AI gateways / proxies** (e.g. LiteLLM, Bifrost — any gateway works) — front
  the endpoint to add routing, key management, or multiple backends. Point them at
  `http://<host>:8000/v1` using the `--served-model-name`; `api_key` can be any
  value (vLLM needs no auth). When swapping the model, update the gateway's model
  mapping to match `--served-model-name`. A plain proxy has **no tool executor** —
  to use tool-calling, route through the gateway's tool-call path
  (`--enable-auto-tool-choice --tool-call-parser openai` are already set on vLLM
  for this).
- **Any containerized tool / UI** that speaks the OpenAI API — for example
  **Open WebUI**, a self-hosted chat UI provided as an optional **separate**
  Compose project (`open_web_ui/compose.yaml`); run it per *Running Open WebUI
  (optional)* below. (Open WebUI renders `message.reasoning` as a collapsible
  panel.)

---

## Firewall (recommended)

The vLLM endpoint has **no authentication**, so don't expose port 8000 to
untrusted networks. Restrict it with a host firewall — UFW is shown here, but any
firewall (firewalld, nftables, iptables) does the same job:

- **Port 8000 (vLLM):** allow only your LAN subnet — or bind it to localhost if
  you only consume it on the host. With UFW, for example:
  `sudo ufw allow from 192.168.x.0/24 to any port 8000 proto tcp`
- **Port 3000 (Open WebUI), if you run it:** the mapping in
  `open_web_ui/compose.yaml` is `3000:8080`, which binds **all** interfaces —
  so either change it to `127.0.0.1:3000:8080` to keep the auth-disabled UI on
  localhost, or firewall it to your LAN subnet the same way as port 8000.

---

## Volumes

Host path: `/var/lib/docker/volumes/llm_<name>/_data`

| Volume | Contents | Notes |
|--------|----------|-------|
| `hf-cache` | HF model weights | Survives compose changes; **shared by all three engines** (no re-download when you swap) |
| `vllm-cache` | torch.compile + AOT artifacts (`vllm_xpu/`, the stock engine) | Critical — without it the first-request torch.compile (~30–60 s) re-runs cold on every restart |
| `vllm-scaler-cache` | llm-scaler engine compile cache | Separate from `vllm-cache` (kernels are image-specific); only created when the scaler engine (`scaler/compose.yaml`) first boots. Stays near-empty — that engine runs `--enforce-eager`, so nothing is compiled, and it needs no clearing on an image bump |
| `vllm-openai-cache` | upstream engine compile cache | Own volume again (kernels are image-specific). This engine runs **compiled**, so unlike the scaler's it does fill — clear it once after an image bump, per [vllm_openai_xpu/README.md](vllm_openai_xpu/README.md) |

Open WebUI's data lives in its own project, so its volume is
`open-webui_open-webui-data` (not `llm_*`) — see *Running Open WebUI (optional)*.

---

## Running Open WebUI (optional)

Open WebUI is an optional, self-hosted chat UI kept in its **own** Compose
project (`open_web_ui/compose.yaml`) so it deploys and updates independently of
the inference engine. It's just one example of an OpenAI-compatible client — swap in
any UI you prefer.

> ⚠️ **Local testing only — not production-hardened.** This config runs with auth
> off (`WEBUI_AUTH=false`), open CORS (Open WebUI defaults `CORS_ALLOW_ORIGIN` to
> `*` and logs `WARNING: CORS_ALLOW_ORIGIN IS SET TO '*' - NOT RECOMMENDED FOR
> PRODUCTION DEPLOYMENTS`), and binds `:3000` on all interfaces. Before any real or
> shared use: set `WEBUI_AUTH=true`, pin `CORS_ALLOW_ORIGIN` to your actual origin,
> and keep the port off untrusted networks (see *Firewall*).

```bash
docker compose -f open_web_ui/compose.yaml up -d      # start the UI
docker compose -f open_web_ui/compose.yaml logs -f    # follow
docker compose -f open_web_ui/compose.yaml down       # stop
```

**Start a vLLM engine first.** These are independent Compose projects, so there's
no automatic `depends_on` linking them. Order isn't fatal, though — if you start the
UI first it runs fine but shows no models until vLLM is reachable, then they
appear on refresh. Both services are `restart: unless-stopped`, so after a host
reboot they self-start and the UI populates once vLLM is healthy.

Open it at `http://localhost:3000` (auth disabled). It runs as its own Compose
project (`open-webui`), so it sits on a separate Docker network and reaches vLLM
through the **host's published port**, not by Docker service name:

- **Same host (default):** `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1`
  — the file maps `host.docker.internal` to the host gateway (Linux).
- **Different host:** set `OPENAI_API_BASE_URL=http://<vllm-host>:8000/v1`.
- **Scaler engine:** no change needed — `scaler/compose.yaml` also serves
  on `:8000` (only one engine runs at a time on the single GPU).

Chats/users/settings persist in the `open-webui_open-webui-data` volume across
restarts. It renders `message.reasoning` as a collapsible panel out of the box.

---

## Troubleshooting: "is it stuck or working?"

The first run on a new HF cache has a long silent phase (no logs) while
oneAPI/SYCL initialises. From the host:

- `cat /proc/<pid>/status` — `nonvoluntary_ctxt_switches` should be incrementing
- `cat /proc/<pid>/io` — `read_bytes` growing means weight load has begun

`cat /proc/<pid>/stack` is blocked by `ptrace_scope` inside the container, so
live stack samples won't work.

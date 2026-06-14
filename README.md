# Local LLM stack — operator guide (devops)

Hardware: Intel Arc Pro B60 (24 GB VRAM, `xe` driver). Host: any modern Linux
(needs Docker and the Intel `xe` GPU driver). Container:
`intel/vllm:0.17.0-xpu`. This is the **how-to** for running and operating the
stack. The *why* behind the config (VRAM sizing, the 0.75-util decision,
quantisation choices) is in [DEVELOPER.md](DEVELOPER.md); a configuration
overview is in [INTEL_ARC_B60.md](INTEL_ARC_B60.md).

The stack is a single vLLM service (port 8000, LAN-exposed) serving
`gpt-oss-20b`. Open WebUI is **commented out** in `docker-compose.yml` but can be
re-enabled (see below).

---

## Running the stack

```bash
docker compose up -d vllm            # start
docker compose logs -f vllm          # follow startup
docker compose stop vllm             # stop
docker compose up -d --force-recreate vllm   # apply a compose edit
```

`docker compose logs -f vllm` follows startup. The healthcheck flips to healthy
once `/health` returns 200 — that's the signal the model is **served**, not that
compile is done. The first request after a (re)start triggers ~30–60 s of
torch.compile work; subsequent requests are fast. The `vllm` service is
`restart: unless-stopped`, so it auto-starts on a Docker daemon restart — an
explicit `docker compose stop vllm` is what keeps it down.

> **First run on a fresh cache is silent for 10–15 min** (oneAPI/SYCL cold
> start, no logs). See *Troubleshooting* below to confirm it's working, not
> stuck. After the first run, `SYCL_CACHE_PERSISTENT=1` + the `vllm-cache`
> volume cut restarts to ~30 s.

---

## Swapping the served model

Four knobs in `docker-compose.yml` under `services.vllm.command`:

| Flag | What to change |
|------|----------------|
| `vllm serve <REPO_ID>` | Hugging Face repo ID (e.g. `openai/gpt-oss-20b`) |
| `--served-model-name <ID>` | Name clients call it by; what a downstream gateway's model mapping points to |
| `--reasoning-parser <NAME>` | Model-family specific. Wrong parser = empty reasoning field, **not** a crash |
| `--max-model-len <N>` | Context window — must fit VRAM after weights + compile buffers (see DEVELOPER.md) |

Then recreate the container:

```bash
docker compose up -d --force-recreate vllm
```

`--force-recreate` is required: vLLM caches its CLI args in the container, so a
compose edit alone won't relaunch with new arguments.

**Quick swap (model already cached):** one compose edit + `up -d
--force-recreate`. No re-download. Compile artifacts in `vllm-cache` are
model-specific, so the first request after a swap still re-compiles — the volume
just stops it from being completely cold.

**New model (not yet cached):** the first `up -d` after editing the repo ID
downloads weights into `hf-cache`. Plan for ~10–30 min download + the silent
10–15 min XPU cold start + ~30–60 s first-request compile.

### Cached models and their parsers

| HF repo | On-disk size | `--reasoning-parser` | Reasoning |
|---------|--------------|----------------------|-----------|
| `openai/gpt-oss-20b` | ~13 GB MXFP4 | `openai_gptoss` | Always on; effort via `reasoning_effort` |
| `Qwen/Qwen3-32B-AWQ` | ~19 GB | `qwen3` | Hybrid; `/no_think` disables |

Sizes above are on-disk cache footprint; loaded-weight (GiB) figures and context
caps live in [DEVELOPER.md](DEVELOPER.md)'s sizing table. Swapping back to Qwen
also means lowering `--max-model-len` (7168 was the empirical cap for 32B-AWQ on
the B60) — details in DEVELOPER.md.

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

Two TTFT numbers are printed: `TTFT (any)` = first token of any kind ("is it
alive"), `TTFT (content)` = first user-visible token after reasoning finishes
("how long until the answer appears"). `Decode tok/s (all)` counts reasoning +
content — the right single-stream number for a reasoning-native model. Aggregate
throughput under concurrent load is much higher; this bench is one-user only.

`bench.sh` has no reasoning-effort knob, so it runs at the model default
(`medium`). To compare effort levels, hit `/v1/chat/completions` directly with a
`reasoning_effort` field — see DEVELOPER.md.

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
suggest. The `intel/vllm:0.17.0-xpu` build uses the shorter name. Any consumer
parsing for `reasoning_content` sees empty strings while thinking tokens are
silently consumed.

Per-family behaviour:

- **gpt-oss** — always reasoning, no off switch. Effort is a request field
  (`reasoning_effort: low|medium|high`, default `medium`) — see
  [DEVELOPER.md](DEVELOPER.md) for its latency behaviour. Reasoning tokens count
  against `--max-model-len`.
- **Qwen3** — hybrid; thinking on by default, `/no_think` in the user message
  disables it.

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
  **Open WebUI**, a self-hosted chat UI bundled (commented out) in
  `docker-compose.yml`; enable it per *Re-enabling Open WebUI* below. (Open WebUI
  renders `message.reasoning` as a collapsible panel.)

---

## Firewall (recommended)

The vLLM endpoint has **no authentication**, so don't expose port 8000 to
untrusted networks. Restrict it with a host firewall — UFW is shown here, but any
firewall (firewalld, nftables, iptables) does the same job:

- **Port 8000 (vLLM):** allow only your LAN subnet — or bind it to localhost if
  you only consume it on the host. With UFW, for example:
  `sudo ufw allow from 192.168.x.0/24 to any port 8000 proto tcp`
- **Port 3000 (Open WebUI), if you enable it:** the bundled mapping is
  `3000:8080`, which binds **all** interfaces — so either change it to
  `127.0.0.1:3000:8080` to keep the auth-disabled UI on localhost, or firewall it
  to your LAN subnet the same way as port 8000.

---

## Volumes

Host path: `/var/lib/docker/volumes/llm_<name>/_data`

| Volume | Contents | Notes |
|--------|----------|-------|
| `hf-cache` | HF model weights | Survives compose changes |
| `vllm-cache` | torch.compile + AOT artifacts | Critical — without it the first-request torch.compile (~30–60 s) re-runs cold on every restart |
| `open-webui-data` | WebUI users / chats / settings | Created once Open WebUI is enabled and run; then persists across restarts (even if the service is commented back out) |
| `vllm-scaler-cache` | A/B service compile cache | Only created when the `scaler` profile first boots |

---

## Re-enabling Open WebUI

Uncomment the `open-webui` service **and** the `open-webui-data` volume in
`docker-compose.yml`, then `docker compose up -d open-webui`. It points at
`http://vllm:8000/v1`, runs on `http://localhost:3000` with auth disabled, and
persists chats/users/settings in the `open-webui-data` volume (so they survive
restarts, and even commenting the service back out). It renders
`message.reasoning` as a collapsible panel out of the box.

---

## A/B testing the llm-scaler image

A second vLLM image (Intel's B-series-optimised `llm-scaler` fork) is wired up
behind the `scaler` compose profile so it never starts on a bare `up`. **One
GPU** — the scaler and the main `vllm` service **cannot run at the same time**
(the VRAM math is in [DEVELOPER.md](DEVELOPER.md)). Run them one at a time:

```bash
docker compose stop vllm
docker compose --profile scaler up -d vllm-scaler
# first boot = the silent 10–15 min XPU cold start (no logs) — wait
VLLM_ENDPOINT=http://localhost:8001 ./bench.sh 400
docker compose stop vllm-scaler && docker compose start vllm   # restore
```

The staged config boots with `--enforce-eager` (safe first boot). Why — and the
"drop eager and re-bench for the true number" follow-up — are in
[DEVELOPER.md](DEVELOPER.md).

---

## Troubleshooting: "is it stuck or working?"

The first run on a new HF cache has a long silent phase (no logs) while
oneAPI/SYCL initialises. From the host:

- `cat /proc/<pid>/status` — `nonvoluntary_ctxt_switches` should be incrementing
- `cat /proc/<pid>/io` — `read_bytes` growing means weight load has begun

`cat /proc/<pid>/stack` is blocked by `ptrace_scope` inside the container, so
live stack samples won't work.

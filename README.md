# Local LLM stack — operator guide (devops)

> ⚠️ The only official source is [github.com/epsilonagentx/intel_arc_gpu_llm](https://github.com/epsilonagentx/intel_arc_gpu_llm); copies elsewhere are not maintained by me.
>
> 💡 Using it? **Fork** the repo (don't just download a copy) and work on your own branch — that keeps you linked to upstream for updates and makes contributing back easy. See [how to fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo), or [fork this one directly](https://github.com/epsilonagentx/intel_arc_gpu_llm/fork).

Hardware: Intel Arc Pro B60 (24 GB VRAM, `xe` driver). **Host OS: Linux only** —
any modern distribution with Docker and the Intel `xe` GPU driver. Windows and
macOS are not supported: the `xe` kernel driver and the sysfs/hwmon helper
scripts (`watt.sh`, the troubleshooting `/proc` reads) are Linux-specific.
Default engine container: `intel/vllm:0.21.0-ubuntu24.04` (an `llm-scaler`
alternative is also provided — see *Choosing the inference engine* below). This
is the **how-to** for running and operating the stack. The *why* behind the
config (VRAM sizing, the 0.75-util decision, quantisation choices) is in
[DEVELOPER.md](DEVELOPER.md); a configuration overview is in
[INTEL_ARC_B60.md](INTEL_ARC_B60.md).

The stack runs one vLLM engine (port 8000, LAN-exposed) serving `gpt-oss-20b`,
from one of two **interchangeable** engine images — stock `intel/vllm` or
Intel's `llm-scaler` fork. An optional chat UI (Open WebUI) ships as a
**separate** Compose file you can bring up alongside it — see *Running Open WebUI
(optional)* below.

---

## Running the stack

The engine ships as two interchangeable images, each in its own Compose file —
pick one; they serve the same `gpt-oss-20b` on the same port `:8000`:

- **`docker-compose.vllm.yml`** — stock `intel/vllm` (the default, recommended).
- **`docker-compose.scaler.yml`** — Intel's B-series-optimised `llm-scaler` fork
  (see *Choosing the inference engine* below).

**One GPU → run exactly one engine at a time** (each needs ~17 GiB; together they
OOM). Neither file uses the default `docker-compose.yml` name, so every command
names its file with `-f` — which also means you can't start both by accident.

```bash
docker compose -f docker-compose.vllm.yml up -d vllm                    # start
docker compose -f docker-compose.vllm.yml logs -f vllm                  # follow startup
docker compose -f docker-compose.vllm.yml stop vllm                     # stop
docker compose -f docker-compose.vllm.yml up -d --force-recreate vllm   # apply a compose edit
```

Every command needs `-f docker-compose.vllm.yml` because neither engine file
uses the default `docker-compose.yml` name — a bare `docker compose up -d` finds
no config and errors out.

The `logs -f vllm` command above follows startup. The healthcheck flips to healthy
once `/health` returns 200 — that's the signal the model is **served**, not that
compile is done. The first request after a (re)start triggers ~30–60 s of
torch.compile work; subsequent requests are fast. The `vllm` service is
`restart: unless-stopped`, so it auto-starts on a Docker daemon restart — an
explicit `docker compose -f docker-compose.vllm.yml stop vllm` is what keeps it down.

> **First run on a fresh cache is silent for 10–15 min** (oneAPI/SYCL cold
> start, no logs). See *Troubleshooting* below to confirm it's working, not
> stuck. After the first run, `SYCL_CACHE_PERSISTENT=1` + the `vllm-cache`
> volume cut restarts to ~30 s.

---

## Upgrading the vLLM image

The stack is pinned to `intel/vllm:0.21.0-ubuntu24.04`. Two things differ from the
earlier `0.17.0-xpu` — both already baked into `docker-compose.vllm.yml`, but they bite
if you bump the image yourself:

- **Device passthrough.** 0.21.0 needs the **whole `/dev/dri`** plus a
  `/dev/dri/by-path:ro` bind-mount. oneCCL enumerates the GPU through `by-path`
  during warm-up, and Docker's `devices:` never recreates that symlink dir — miss
  it and the container dies at boot (`oneCCL: ze_fd_manager … opendir failed`).
  `0.17.0-xpu` booted with just the `renderD128`/`card1` nodes.
- **Compile cache.** torch.compile kernels are image-version-specific, so clear the
  old cache once on upgrade:
  ```bash
  docker compose -f docker-compose.vllm.yml down
  docker volume rm llm_vllm-cache
  docker compose -f docker-compose.vllm.yml up -d vllm
  ```
  The first request then runs the usual ~30–60 s compile and the cache repopulates.

---

## Swapping the served model

A swap is two steps — edit the launch command, then force-recreate.

**Step 1 — edit `services.vllm.command` in `docker-compose.vllm.yml`.** Change the model-specific knobs:

| Flag | What to change |
|------|----------------|
| `vllm serve <REPO_ID>` | Hugging Face repo ID (e.g. `openai/gpt-oss-20b`) |
| `--served-model-name <ID>` | Name clients call it by; what a downstream gateway's model mapping points to |
| `--reasoning-parser <NAME>` | Model-family specific. Wrong parser = empty reasoning field, **not** a crash |
| `--max-model-len <N>` | Context window — must fit VRAM after weights + compile buffers (see DEVELOPER.md) |

(To swap the model on the **scaler** engine instead, edit the `command:` block in
`docker-compose.scaler.yml` the same way — its `llm-scaler` image is what unlocks
quantised MoE such as `Qwen3-30B-A3B-GPTQ-Int4`; see *Choosing the inference
engine*.)

**Step 2 — recreate the container:**

```bash
docker compose -f docker-compose.vllm.yml up -d --force-recreate vllm
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

### Worked example: gpt-oss-20b ↔ Qwen3-32B-AWQ

The shipped `command:` serves gpt-oss-20b:

```yaml
    command: >
      vllm serve openai/gpt-oss-20b
        --host 0.0.0.0
        --port 8000
        --max-model-len 65536
        --gpu-memory-utilization 0.75
        --reasoning-parser openai_gptoss
        --enable-auto-tool-choice
        --tool-call-parser openai
        --served-model-name gpt-oss-20b
```

To serve **Qwen3-32B-AWQ** instead, edit that block to:

```yaml
    command: >
      vllm serve Qwen/Qwen3-32B-AWQ
        --host 0.0.0.0
        --port 8000
        --max-model-len 7168
        --reasoning-parser qwen3
        --enable-auto-tool-choice
        --tool-call-parser hermes
        --served-model-name qwen3-32b
```

…then `docker compose -f docker-compose.vllm.yml up -d --force-recreate vllm`. What changed and why:

- **Repo + served name** → `Qwen/Qwen3-32B-AWQ`, called `qwen3-32b` by clients (update any gateway's model mapping to match).
- **`--max-model-len` 65536 → 7168** — the empirical B60 cap for this model; 10k and 12k both fail vLLM's KV pre-check at startup.
- **`--reasoning-parser` → `qwen3`** — Qwen3 is hybrid-thinking (`/no_think` in the prompt turns it off); the `openai_gptoss` parser would leave the reasoning field empty.
- **`--tool-call-parser` → `hermes`** — Qwen3 emits Hermes-style tool calls, not gpt-oss's `openai` format. (The image also ships `qwen3_xml` and `qwen3_coder`; the latter is only for Qwen3-**Coder**.) Drop both tool flags if you don't need tool-calling.
- **Dropped `--gpu-memory-utilization 0.75`** — Qwen3-32B-AWQ's weights are ~18 GiB, which won't fit the ~17 GiB that 0.75 reserves, so it falls back to vLLM's 0.9 default. It's a tight fit on 22.7 GiB (the reason `--max-model-len` is only 7168) — watch real VRAM and size it empirically per [DEVELOPER.md](DEVELOPER.md).
- **AWQ, not FP8** — the official `Qwen/*-FP8` weights hit an XPU bug on this image; AWQ is the working path.

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
  Both engines publish `:8000`, so the same command benchmarks whichever is up.

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
suggest — verified on both the older `intel/vllm:0.17.0-xpu` and the current
`intel/vllm:0.21.0-ubuntu24.04`. Any consumer parsing for `reasoning_content`
sees empty strings while thinking tokens are silently consumed.

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
  **Open WebUI**, a self-hosted chat UI provided as an optional **separate**
  Compose file (`docker-compose.openwebui.yml`); run it per *Running Open WebUI
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
  `docker-compose.openwebui.yml` is `3000:8080`, which binds **all** interfaces —
  so either change it to `127.0.0.1:3000:8080` to keep the auth-disabled UI on
  localhost, or firewall it to your LAN subnet the same way as port 8000.

---

## Volumes

Host path: `/var/lib/docker/volumes/llm_<name>/_data`

| Volume | Contents | Notes |
|--------|----------|-------|
| `hf-cache` | HF model weights | Survives compose changes; **shared by both engines** (no re-download when you swap) |
| `vllm-cache` | torch.compile + AOT artifacts (base engine) | Critical — without it the first-request torch.compile (~30–60 s) re-runs cold on every restart |
| `vllm-scaler-cache` | llm-scaler engine compile cache | Separate from `vllm-cache` (kernels are image-specific); only created when the scaler engine (`docker-compose.scaler.yml`) first boots |

Open WebUI's data lives in its own project, so its volume is
`open-webui_open-webui-data` (not `llm_*`) — see *Running Open WebUI (optional)*.

---

## Running Open WebUI (optional)

Open WebUI is an optional, self-hosted chat UI kept in its **own** Compose file
(`docker-compose.openwebui.yml`) so it deploys and updates independently of the
inference engine. It's just one example of an OpenAI-compatible client — swap in
any UI you prefer.

> ⚠️ **Local testing only — not production-hardened.** This config runs with auth
> off (`WEBUI_AUTH=false`), open CORS (Open WebUI defaults `CORS_ALLOW_ORIGIN` to
> `*` and logs `WARNING: CORS_ALLOW_ORIGIN IS SET TO '*' - NOT RECOMMENDED FOR
> PRODUCTION DEPLOYMENTS`), and binds `:3000` on all interfaces. Before any real or
> shared use: set `WEBUI_AUTH=true`, pin `CORS_ALLOW_ORIGIN` to your actual origin,
> and keep the port off untrusted networks (see *Firewall*).

```bash
docker compose -f docker-compose.openwebui.yml up -d      # start the UI
docker compose -f docker-compose.openwebui.yml logs -f    # follow
docker compose -f docker-compose.openwebui.yml down       # stop
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
- **Scaler engine:** no change needed — `docker-compose.scaler.yml` also serves
  on `:8000` (only one engine runs at a time on the single GPU).

Chats/users/settings persist in the `open-webui_open-webui-data` volume across
restarts. It renders `message.reasoning` as a collapsible panel out of the box.

---

## Choosing the inference engine (base vs llm-scaler)

The stack ships two interchangeable vLLM engines — the stock `intel/vllm`
(`docker-compose.vllm.yml`, the default) and Intel's B-series-optimised
`llm-scaler` fork (`docker-compose.scaler.yml`). Both serve `gpt-oss-20b` on
`:8000`, so whichever is up, downstream clients need no change. **One GPU — they
cannot run at the same time** (the VRAM math is in [DEVELOPER.md](DEVELOPER.md)).
Swap one for the other:

```bash
docker compose -f docker-compose.vllm.yml down            # stop the base engine
docker compose -f docker-compose.scaler.yml up -d         # start the scaler engine on :8000
# first boot = the silent 10–15 min XPU cold start (no logs) — wait
./bench.sh 400                                            # benchmark it (default endpoint :8000)
./smoke.sh                                               # verify reasoning + tool-calling still work
#   …and back to the base engine:
docker compose -f docker-compose.scaler.yml down
docker compose -f docker-compose.vllm.yml up -d
```

Why you might switch: the `llm-scaler` fork is tuned for Arc B-series and its
image unlocks quantised-MoE paths that the stock image doesn't (e.g.
`Qwen3-30B-A3B-GPTQ-Int4`). Benchmark it against the stock engine before adopting
it in prod — the stock image has since jumped `0.17`→`0.21`, so the fork is no
longer strictly newer, and the win (if any) has to be measured, not assumed.

The scaler config boots with `--enforce-eager` (safe first boot, but slower — so
it under-states the engine's real speed). Dropping eager for the true number,
plus the VRAM re-tune it then needs, is covered in [DEVELOPER.md](DEVELOPER.md).
The fork inherits upstream's parser flag names but that's unverified here, so run
`./smoke.sh` to confirm the reasoning channel and tool-calling work before
trusting it in prod.

---

## Troubleshooting: "is it stuck or working?"

The first run on a new HF cache has a long silent phase (no logs) while
oneAPI/SYCL initialises. From the host:

- `cat /proc/<pid>/status` — `nonvoluntary_ctxt_switches` should be incrementing
- `cat /proc/<pid>/io` — `read_bytes` growing means weight load has begun

`cat /proc/<pid>/stack` is blocked by `ptrace_scope` inside the container, so
live stack samples won't work.

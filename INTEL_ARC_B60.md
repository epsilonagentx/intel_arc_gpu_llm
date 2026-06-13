# Intel Arc B60 LLM stack — work hub

The hub for the Arc B60 LLM stack: what's running, what's pending, and where the
real docs live. This page is the **living state**; the stable how/why lives in
the two role docs:

- **[README.md](README.md)** — devops / operator: how to run, swap, monitor, troubleshoot.
- **[DEVELOPER.md](DEVELOPER.md)** — developer: why the config is what it is.

> This repo lives at `/home/epsilonagentx/github_projects/intel_arc_gpu_llm`
> (GitHub remote: `epsilonagentx/intel_arc_gpu_llm`). The Compose project name is
> pinned to `llm` (`name: llm` in `docker-compose.yml`), so the cache volumes
> stay `llm_*` regardless of the folder name.

---

## Current config

Source of truth is `docker-compose.yml`; this table is a snapshot for quick orientation.

| | |
|---|---|
| Service | `vllm` (container `vllm-xpu`) |
| Image | `intel/vllm:0.17.0-xpu` |
| Model | `openai/gpt-oss-20b`, served as **`gpt-oss-20b`** |
| Endpoint | `http://localhost:8000/v1` (LAN-exposed on port 8000) |
| Context | `--max-model-len 65536` (64k) |
| VRAM | `--gpu-memory-utilization 0.75` |
| Reasoning | `--reasoning-parser openai_gptoss` → trace in `message.reasoning` |
| Tools | `--enable-auto-tool-choice --tool-call-parser openai` (OpenAI format, **on**) |

**Live state (2026-06-13):** the stack is **down** — only `portainer` is
running, nothing is bound to host `:8000`. Bring it up with the start procedure
in the README. (The `vllm` service is `restart: unless-stopped`, so it will
auto-start on a Docker daemon restart unless explicitly stopped.)

## Cached models in `hf-cache` (last known)

Swap-back is one compose edit, no re-download — see the README's swap procedure.
Sizes below are on-disk footprint; loaded-weight GiB and context caps are in
[DEVELOPER.md](DEVELOPER.md).

- `openai/gpt-oss-20b` (~13 GB on disk) — currently configured to serve
- `Qwen/Qwen3-32B-AWQ` (~19 GB on disk) — prior model (was served as `qwen3-32b`, capped at 7168 ctx on the B60)
- `Qwen3-14B-AWQ` and `Qwen3-4B-Thinking-2507` were deleted 2026-05-25 to reclaim ~17 GB; re-pullable from HF if needed

**Present on host (2026-06-13):** images `intel/vllm:0.17.0-xpu` +
`intel/llm-scaler-vllm:0.14.0-b8.3.1` (plus `open-webui:main`, `portainer`);
volumes `hf-cache`, `vllm-cache`, `open-webui-data` — no `vllm-scaler-cache` yet.

## Consumers

- **LiteLLM** proxy on `192.168.x.x:4000` — model name `openai/gpt-oss-20b`,
  `api_base=http://<this-host-LAN-ip>:8000/v1`, `api_key="EMPTY"`. No tool
  executor on the LiteLLM side.
- **Bifrost** gateway on `192.168.x.x:4010` — fronts gpt-oss-20b for coding
  CLIs (model id `arc-b60-vllm-xpu/gpt-oss-20b`); uses the tool-calling path,
  verified end-to-end through the firewall.

---

## Open questions / decisions pending

- **ComfyUI** (`yanwk/comfyui-boot:xpu`, confirmed B60-supported) was scoped and
  authored but is on hold pending a VRAM-overlap decision. **Do not add it
  without an explicit go.** Two paths were offered:
  1. add as a service, stop vLLM manually when generating images;
  2. coexist via a smaller Qwen-Image quant (Q3 ~8 GB) alongside the LLM.

## Verify next session (still unconfirmed)

- **UFW**: is `allow from 192.168.x.0/24 to any port 8000 proto tcp` actually in
  `sudo ufw status`? Couldn't check this session — needs root.
- **LiteLLM rename**: has the config on `192.168.x.x` actually been renamed to
  `openai/gpt-oss-20b` (from `openai/qwen3-32b`)? Remote host — not reachable
  from here.

## Recent changes

- **2026-06-13** — docs reorganised by role (this wiki + README + DEVELOPER);
  `CLAUDE.md` reduced to a pointer that imports this file.
- **IPEX-LLM cleanup done** — `intelanalytics/ipex-llm-inference-cpp-xpu` is no
  longer present (confirmed via `docker images`); the dead ~28 GB image is gone.
- **2026-06-11** — llm-scaler A/B service staged (compose profile `scaler`, port
  8001). Image `intel/llm-scaler-vllm:0.14.0-b8.3.1` is pulled but the test has
  **not been run** (no `vllm-scaler-cache` volume exists yet). Design and
  procedure live in DEVELOPER.md / README.md.
- **2026-06-08** — `watt.sh` added (B60 power from sysfs `xe` energy counters,
  no root, no packages).
- Tool-calling enabled on vLLM (`--enable-auto-tool-choice --tool-call-parser
  openai`) for the Bifrost coding CLIs.

## Decisions on record (hardware / model, not stack config)

Context for future swaps, kept out of the operator docs deliberately:

- **B70 vs B60** — Arc Pro B70 32 GB gives ~1.3× decode / ~1.85× prefill plus
  context headroom, but does **not** unlock Gemma 4 (software-gated).
- **Host GPU expansion (X870E)** — second card only gets chipset x4; don't
  TP/PP across cards. Plan is B70 = LLM engine, B60 = independent engine.
- **Model freshness** — stayed on gpt-oss-20b (cutoff Jun 2024); fresher fast
  MoEs (Qwen3.5/3.6-35B-A3B AWQ ≈ 24 GB) won't fit 22.7 GiB. Freshness → RAG,
  not a model swap.

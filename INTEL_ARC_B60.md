# Intel Arc B60 LLM stack — configuration overview

A quick-reference snapshot of how the stack is configured. The how-to and the
why live elsewhere:

- **[README.md](README.md)** — how to run, swap, monitor and troubleshoot.
- **[DEVELOPER.md](DEVELOPER.md)** — why the config is what it is.

> The Compose project name is pinned to `llm` (`name: llm` in **all three**
> engine compose files), so the cache volumes stay `llm_*` regardless of the
> folder the repo is checked out into — and so all three engines share one
> weights cache. Never pass `--remove-orphans`: in one shared project it deletes
> the other engines.

---

## Current config

Each engine's compose file is the source of truth, and each folder's README
explains that engine:

| folder | image | its README |
|---|---|---|
| `vllm_xpu/` | stock `intel/vllm` | [vllm_xpu/README.md](vllm_xpu/README.md) |
| `scaler/` | Intel's `llm-scaler` fork | [scaler/README.md](scaler/README.md) |
| `vllm_openai_xpu/` | upstream's XPU image — **currently live** | [vllm_openai_xpu/README.md](vllm_openai_xpu/README.md) |

⚠ The table below describes the **stock `vllm_xpu/` engine only**, as a quick
orientation snapshot. It is not what is running today.

| | |
|---|---|
| Host OS | **Linux only** (Intel `xe` GPU driver + Docker; no Windows/macOS) |
| Service | `vllm` (container `vllm-xpu`) |
| Image | `intel/vllm:0.21.0-ubuntu24.04` |
| Devices | whole `/dev/dri` + `/dev/dri/by-path` bind-mount (oneCCL enumeration) |
| Model | `openai/gpt-oss-20b`, served as **`gpt-oss-20b`** |
| Endpoint | `http://localhost:8000/v1` (LAN-exposed on port 8000) |
| Context | `--max-model-len 131072` (128k) |
| VRAM | `--gpu-memory-utilization 0.80` (this engine does **not** pin the KV pool; `scaler/` and `vllm_openai_xpu/` do) |
| Reasoning | `--reasoning-parser openai_gptoss` → trace in `message.reasoning` |
| Tools | `--enable-auto-tool-choice --tool-call-parser openai` (OpenAI format, **on**) |

## Cached models

Already in the shared weights cache, so swapping to either needs no download.
Sizes are on-disk footprint:

- `openai/gpt-oss-20b` — ~13 GB
- `Qwen/Qwen3-32B-AWQ` — ~19 GB

How to point an engine at a different model differs per engine, so it is
documented in each folder's README — for these two, in
[vllm_xpu/README.md](vllm_xpu/README.md), which also has a worked example.

## Clients

The endpoint is OpenAI-compatible at `http://<host>:8000/v1` and needs no auth,
so any OpenAI-style client works — typically an AI gateway in front of it, or a
tool that speaks the API directly. The root [README](README.md) covers both,
along with the optional Open WebUI project.

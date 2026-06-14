# Intel Arc B60 LLM stack — configuration overview

A quick-reference snapshot of how the stack is configured. The how-to and the
why live in the two role docs:

- **[README.md](README.md)** — devops / operator: how to run, swap, monitor, troubleshoot.
- **[DEVELOPER.md](DEVELOPER.md)** — developer: why the config is what it is.

> The Compose project name is pinned to `llm` (`name: llm` in
> `docker-compose.yml`), so the cache volumes stay `llm_*` regardless of the
> folder the repo is checked out into.

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

## Cached models

Swapping the served model is one compose edit, with no re-download for a model
that's already cached — see the README's swap procedure. Sizes below are on-disk
footprint; loaded-weight GiB and context caps are in [DEVELOPER.md](DEVELOPER.md).

- `openai/gpt-oss-20b` (~13 GB on disk) — the configured/served model
- `Qwen/Qwen3-32B-AWQ` (~19 GB on disk) — alternate (parser `qwen3`; capped at ~7168 ctx on the B60, see [DEVELOPER.md](DEVELOPER.md))

## Clients

The endpoint is OpenAI-compatible (`/v1`), so any OpenAI-style client works — a
consumer is generally either an AI gateway in front of it or a containerized
tool/UI that talks to it directly:

- **AI gateways / proxies** (e.g. LiteLLM, Bifrost — any gateway works) — front
  the endpoint at `http://<host>:8000/v1` with the served model name to add
  routing, key management, or multiple backends (`api_key` can be any value;
  vLLM needs no auth).
- **Any containerized tool / UI** that speaks the OpenAI API — such as Open
  WebUI, a self-hosted chat UI included (commented out) in `docker-compose.yml`.

See the README's *Clients* and *Re-enabling Open WebUI* sections for details.

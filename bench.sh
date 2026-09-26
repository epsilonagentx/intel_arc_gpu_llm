#!/usr/bin/env bash
# Measure TTFT and tok/s against the running vLLM service.
#
# Usage:
#   ./bench.sh                                       # default prompt, 200 tokens
#   ./bench.sh 400                                   # max_tokens=400
#   ./bench.sh 400 "Summarize the French Revolution." # custom prompt
#   THINKING=1 ./bench.sh 1500                       # force reasoning ON
#   THINKING=0 ./bench.sh 400                        # force reasoning OFF
#
# THINKING is tri-state and controls `chat_template_kwargs.enable_thinking`:
#   auto (default) — send nothing; the server's own default decides
#   1              — force on   (models whose reasoning is opt-in: gemma-4, Qwen3)
#   0              — force off  (suppress reasoning even if the server defaults it on)
# Both directions are per-request overrides and beat the server-side
# --default-chat-template-kwargs, so `auto` is the only value that reports what
# the deployed default actually does. No-op for gpt-oss-20b, which always reasons.
# NOTE: with reasoning on, a low --max-tok is consumed entirely by the reasoning
# phase, leaving zero content chunks — budget 1500+ to also get an answer.
set -euo pipefail

MAX_TOK="${1:-200}"
PROMPT="${2:-Explain in detail how the human immune system identifies and destroys cancer cells.}"
ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000}"
THINKING="${THINKING:-auto}"

# Default to whatever the engine actually serves, so the script keeps working
# across a model swap. A hardcoded default would 404 the moment `.env` changes.
MODEL="${MODEL:-}"
MODEL_SRC="explicit"
if [[ -z "$MODEL" ]]; then
    MODEL=$(curl -s --max-time 10 "$ENDPOINT/v1/models" 2>/dev/null \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) || true
    MODEL_SRC="auto-detected"
fi
if [[ -z "$MODEL" ]]; then
    echo "ERROR: no model id readable from $ENDPOINT/v1/models — is the engine up?" >&2
    echo "       Start it, or pass one explicitly: MODEL=<id> $0" >&2
    exit 1
fi

echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL ($MODEL_SRC)"
echo "Prompt:   $PROMPT"
echo "Max tok:  $MAX_TOK"
echo "Thinking: $THINKING"
echo

python3 - "$ENDPOINT" "$MODEL" "$MAX_TOK" "$PROMPT" "$THINKING" <<'PY'
import json, sys, time, urllib.request

endpoint, model, max_tok, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
thinking = sys.argv[5]          # "auto" | "1" | "0"

payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tok,
    "stream": True,
}
# "auto" deliberately sends no chat_template_kwargs, so the server-side
# --default-chat-template-kwargs decides. "1"/"0" override it either way.
if thinking in ("1", "0"):
    payload["chat_template_kwargs"] = {"enable_thinking": thinking == "1"}
elif thinking != "auto":
    print(f"ERROR: THINKING must be auto, 1 or 0 (got {thinking!r})", file=sys.stderr)
    sys.exit(1)

req = urllib.request.Request(
    f"{endpoint}/v1/chat/completions",
    method="POST",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload).encode(),
)

def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

t0 = time.perf_counter()
ttft_any = None        # first token of any kind (reasoning or content)
ttft_content = None    # first delta.content specifically
chunks_reasoning = 0
chunks_content = 0
reasoning_out, content_out = [], []
try:
    stream = urllib.request.urlopen(req, timeout=300)
except urllib.error.HTTPError as e:
    detail = e.read().decode(errors="replace")[:300]
    if e.code == 404:
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10) as r:
                served = [m["id"] for m in json.load(r).get("data", [])]
        except Exception:
            served = []
        fail(f"HTTP 404: model {model!r} is not served.\n"
             f"  Served now: {served}\n"
             f"  Re-run with MODEL=<one of those>, or unset MODEL to auto-detect.")
    fail(f"HTTP {e.code} from /v1/chat/completions: {detail}")
except urllib.error.URLError as e:
    fail(f"Cannot reach {endpoint}: {e.reason}")

with stream as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            ch = json.loads(payload)
        except Exception:
            continue
        delta = ch["choices"][0].get("delta", {})
        r_delta = delta.get("reasoning")
        c_delta = delta.get("content")
        now = time.perf_counter() - t0
        if r_delta:
            if ttft_any is None:
                ttft_any = now
            chunks_reasoning += 1
            reasoning_out.append(r_delta)
        if c_delta:
            if ttft_any is None:
                ttft_any = now
            if ttft_content is None:
                ttft_content = now
            chunks_content += 1
            content_out.append(c_delta)
elapsed = time.perf_counter() - t0
total_chunks = chunks_reasoning + chunks_content
decode_time = elapsed - (ttft_any or 0)

reasoning_text = "".join(reasoning_out)
content_text = "".join(content_out)
if reasoning_text:
    print("--- reasoning (first 300 chars) ---")
    print(reasoning_text[:300] + ("..." if len(reasoning_text) > 300 else ""))
    print()
print("--- content (first 400 chars) ---")
print(content_text[:400] + ("..." if len(content_text) > 400 else ""))
print()
print(f"TTFT (any)          = {ttft_any*1000:.0f} ms" if ttft_any else "TTFT (any)          = n/a")
if ttft_content is not None:
    print(f"TTFT (content)      = {ttft_content*1000:.0f} ms")
else:
    print(f"TTFT (content)      = n/a (model never left reasoning phase)")
print(f"Decode time         = {decode_time:.2f} s")
print(f"Reasoning chunks    = {chunks_reasoning}")
print(f"Content chunks      = {chunks_content}")
print(f"Total chunks        = {total_chunks}")
print(f"Decode tok/s (all)  ~ {total_chunks/decode_time:.1f}" if decode_time > 0 else "")
print(f"Wall time           = {elapsed:.2f} s")
PY

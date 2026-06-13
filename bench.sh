#!/usr/bin/env bash
# Measure TTFT and tok/s against the running vLLM service.
#
# Usage:
#   ./bench.sh                                       # default prompt, 200 tokens
#   ./bench.sh 400                                   # max_tokens=400
#   ./bench.sh 400 "Summarize the French Revolution." # custom prompt
set -euo pipefail

MAX_TOK="${1:-200}"
PROMPT="${2:-Explain in detail how the human immune system identifies and destroys cancer cells.}"
ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-gpt-oss-20b}"

echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL"
echo "Prompt:   $PROMPT"
echo "Max tok:  $MAX_TOK"
echo

python3 - "$ENDPOINT" "$MODEL" "$MAX_TOK" "$PROMPT" <<'PY'
import json, sys, time, urllib.request

endpoint, model, max_tok, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

req = urllib.request.Request(
    f"{endpoint}/v1/chat/completions",
    method="POST",
    headers={"Content-Type": "application/json"},
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "stream": True,
    }).encode(),
)

t0 = time.perf_counter()
ttft_any = None        # first token of any kind (reasoning or content)
ttft_content = None    # first delta.content specifically
chunks_reasoning = 0
chunks_content = 0
reasoning_out, content_out = [], []
with urllib.request.urlopen(req, timeout=300) as r:
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

#!/usr/bin/env bash
# Smoke-test the running vLLM service end-to-end: the model is served, plain chat
# generates content, the reasoning trace comes through, and tool-calling emits a
# tool_call. A fast "did the boot/upgrade actually work?" check — complements
# bench.sh (which measures speed, not correctness).
#
# Usage:
#   ./smoke.sh                                   # localhost:8000, model gpt-oss-20b
#   VLLM_ENDPOINT=http://192.168.x.x:8000 ./smoke.sh
#   MODEL=qwen3-32b ./smoke.sh                   # after a model swap
set -uo pipefail

ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-gpt-oss-20b}"

echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL"
echo

python3 - "$ENDPOINT" "$MODEL" <<'PY'
import json, sys, urllib.request

endpoint, model = sys.argv[1], sys.argv[2]
fails = 0

def get(path):
    with urllib.request.urlopen(f"{endpoint}{path}", timeout=30) as r:
        return json.load(r)

def chat(payload):
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode(),
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]

def check(ok, label):
    global fails
    print(("PASS" if ok else "FAIL") + f": {label}")
    if not ok:
        fails += 1

# 1. model is served
try:
    ids = [m["id"] for m in get("/v1/models").get("data", [])]
    print("served:", ids)
    check(model in ids, f"{model} is served")
except Exception as e:
    check(False, f"/v1/models reachable ({e})")

# 2. plain chat generates content
try:
    m = chat({"model": model, "temperature": 0, "max_tokens": 2048,
              "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}]})
    content = (m.get("content") or "").strip()
    print("content:", repr(content[:80]))
    check(bool(content), "non-empty content returned")
except Exception as e:
    check(False, f"chat completion ({e})")

# 3. reasoning trace present — vLLM puts it in message.reasoning, NOT reasoning_content
try:
    m = chat({"model": model, "temperature": 0, "max_tokens": 4096,
              "messages": [{"role": "user", "content":
                            "A farmer has 17 sheep; all but 9 run away. How many are left? Think step by step."}]})
    reasoning = m.get("reasoning") or m.get("reasoning_content")
    print("reasoning len:", len(reasoning or ""))
    check(bool(reasoning), "reasoning trace populated (message.reasoning)")
except Exception as e:
    check(False, f"reasoning check ({e})")

# 4. tool-calling emits a tool_call (skip if the served model has no tool support)
try:
    m = chat({"model": model, "temperature": 0, "max_tokens": 4096,
              "tool_choice": "auto",
              "tools": [{"type": "function", "function": {
                  "name": "get_weather", "description": "Get current weather for a city",
                  "parameters": {"type": "object",
                                 "properties": {"city": {"type": "string"}},
                                 "required": ["city"]}}}],
              "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]})
    tcs = m.get("tool_calls") or []
    print("tool_calls:", json.dumps(tcs)[:200])
    check(bool(tcs) and tcs[0]["function"]["name"] == "get_weather", "get_weather tool_call emitted")
except Exception as e:
    check(False, f"tool-calling check ({e})")

print()
print("ALL PASS" if fails == 0 else f"{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
PY

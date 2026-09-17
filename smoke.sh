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
#   THINKING=1 ./smoke.sh                        # force reasoning ON
#   THINKING=0 ./smoke.sh                        # assert reasoning can be turned OFF
#
# THINKING is tri-state and drives check 3 (`chat_template_kwargs.enable_thinking`):
#   auto (default) — send nothing; asserts the DEPLOYED default reasons. Passes on
#                    gpt-oss (always reasons) and on gemma-4 only when the server
#                    was started with --default-chat-template-kwargs enabling it.
#   1              — force on;  asserts a reasoning trace comes back
#   0              — force off; asserts reasoning is SUPPRESSED (the check inverts),
#                    which proves a request can override the server-side default
# MODEL is auto-detected from /v1/models when unset, so this keeps working across
# a model swap without a stale hardcoded name.
set -uo pipefail

ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000}"
THINKING="${THINKING:-auto}"

# Default to whatever the engine actually serves, so the script keeps working
# across a model swap. A hardcoded default would fail every check after `.env`
# changes, and report it as a model fault rather than a stale script default.
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
echo "Thinking: $THINKING"
echo

python3 - "$ENDPOINT" "$MODEL" "$THINKING" <<'PY'
import json, sys, urllib.request

endpoint, model, thinking = sys.argv[1], sys.argv[2], sys.argv[3]   # "auto" | "1" | "0"
if thinking not in ("auto", "1", "0"):
    print(f"ERROR: THINKING must be auto, 1 or 0 (got {thinking!r})", file=sys.stderr)
    sys.exit(1)
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
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        # Surface the server's own message; a bare HTTPError hides a 404 on the
        # model name behind what looks like a generation failure.
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from None

def check(ok, label, fail_hint=""):
    """fail_hint is appended only when the check fails, so a PASS stays clean."""
    global fails
    print(("PASS" if ok else "FAIL") + f": {label}" + ("" if ok else fail_hint))
    if not ok:
        fails += 1

# 1. model is served
try:
    cards = get("/v1/models").get("data", [])
    ids = [m["id"] for m in cards]
    print("served:", ids)
    # --served-model-name is a label and need not match the checkpoint, so print
    # `root` too: it is the only place the actually-loaded repo is visible here.
    for c in cards:
        print(f"  {c['id']} -> {c.get('root')}  (max_model_len={c.get('max_model_len')})")
    if len(ids) != len(set(ids)):
        print("WARNING: duplicate ids in /v1/models — a served-model-name is repeated")
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
    payload = {"model": model, "temperature": 0, "max_tokens": 4096,
               "messages": [{"role": "user", "content":
                             "A farmer has 17 sheep; all but 9 run away. How many are left? Think step by step."}]}
    if thinking in ("1", "0"):
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "1"}
    m = chat(payload)
    reasoning = m.get("reasoning") or m.get("reasoning_content")
    print("reasoning len:", len(reasoning or ""))
    if thinking == "0":
        # Inverted on purpose: THINKING=0 tests that a request can SUPPRESS
        # reasoning, including overriding a server-side enable_thinking default.
        check(not reasoning, "reasoning suppressed by request (enable_thinking=false)",
              " — request-level override was ignored; the server default won")
    else:
        hint = ("" if thinking == "1" else
                " — reasoning is opt-in on this model: retry with THINKING=1, or set"
                " VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS='{\"enable_thinking\":true}' to default it on")
        check(bool(reasoning), "reasoning trace populated (message.reasoning)", hint)
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

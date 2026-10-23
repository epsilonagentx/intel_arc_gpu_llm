# Edited chat templates

A model's chat template turns the messages of a request into the prompt the
model reads. It ships with the checkpoint, and a few of them reject requests
that real clients send. The templates here are copies of a checkpoint's own
template with one change each, so the fix lives on the server and no client
has to change.

Compose mounts this folder read-only at `/opt/chat-templates`.
`VLLM_CHAT_TEMPLATE` in `.env` picks one by its path in the container; empty
keeps the model's own template:

```dotenv
VLLM_CHAT_TEMPLATE=/opt/chat-templates/qwen3_8_merge_system.jinja
```

Apply it with `docker compose up -d`. A template belongs to one model, so
switch it together with the rest of that model's block in `.env.example`.

| Template | Model | What it changes |
|---|---|---|
| `qwen3_8_merge_system.jinja` | Qwen3.8-27B | merges every system message into one |

## Qwen3.8: several system messages

Qwen3.8's template accepts one system message, and only as the first message.
Any other `system` message fails the request with a 400:

```text
Chat template rejected the request: System message must be at the beginning.
```

A client that sends one system prompt never sees this. It shows up behind a
gateway that adds its own system message in front of the client's, for
example the instructions of the MCP servers it connects. The client's system
prompt is then the second message, and every request fails. A `developer`
message doesn't trigger it, only a second `system` one.

`qwen3_8_merge_system.jinja` joins the text of every system message, in order
and separated by a blank line, into the one system block at the top, and no
longer rejects the extra ones. The rest of the template is unchanged.

**Tested offline**, by rendering both templates with jinja2 on 2026-10-19:

- Every request the original accepts gives exactly the same prompt with the
  copy: a plain chat, tools, a tool call and its result, several turns,
  reasoning effort `medium` and `low`, and thinking off. So it costs nothing
  for clients that send one system message, and their prefix cache still hits.
- Two system messages give exactly the prompt the original builds from one
  system message holding both texts.
- It still rejects what the original rejects otherwise: an unknown reasoning
  effort, a request with no user message, an image in a system message.

**On the engine**, vLLM 0.32.0 on the B70 the same day: a request with two
system messages, "Always answer in French." and "Answer in one word.", came
back 200 with `Bleu`, so the model followed both.

The copy comes from `RedHatAI/Qwen3.8-27B-INT4`, revision
`91bd022d5b49442a868bc35008f6c21e1860edfa`, under Apache-2.0. The FP8 copy
from [`../tools/`](../tools/VLLM_OPENAI_XPU_TOOLS.md) links the same file. If
a new upload changes the checkpoint's template, make the copy again from the
new one.

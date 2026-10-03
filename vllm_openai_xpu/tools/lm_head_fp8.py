"""Make a local copy of a compressed-tensors checkpoint with an FP8 output layer.

Many int4 checkpoints keep `lm_head` at 16-bit. On a dense model with MTP it is
read for the main step and again for every draft token, so halving it is worth
~18% decode (README "Qwen3.8-27B"). vLLM can't quantize `lm_head` as it loads,
but it does load a checkpoint that declares an FP8 `lm_head`, and XPU runs that
on its native FP8 W8A16 kernel.

Run it inside the engine image, as your own user, with the model folder
mounted where the engine sees it (from vllm_openai_xpu/):

    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \\
      -v ~/models/hf:/cache/huggingface -v "$PWD":/work:ro \\
      --entrypoint python3 vllm/vllm-openai-xpu:v0.30.0 \\
      /work/tools/lm_head_fp8.py RedHatAI/Qwen3.8-27B-INT4

The copy lands in /cache/huggingface/local/<name>-fp8head, which is what
VLLM_MODEL then points at. Every file links back to the original snapshot
except config.json, the index, the shard that held lm_head (rewritten without
it) and lm_head-fp8.safetensors. The original is only read.
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CACHE = "/cache/huggingface"

ap = argparse.ArgumentParser()
ap.add_argument("model", help="Hugging Face repo id already in the cache")
ap.add_argument("--name", help="output folder name (default: <repo>-fp8head)")
args = ap.parse_args()

snaps = glob.glob(f"{CACHE}/hub/models--{args.model.replace('/', '--')}/snapshots/*/")
if len(snaps) != 1:
    sys.exit(f"expected one cached snapshot of {args.model}, found {len(snaps)}")
src = snaps[0]
dst = f"{CACHE}/local/{args.name or args.model.split('/')[-1] + '-fp8head'}/"

cfg = json.load(open(src + "config.json"))
qc = cfg.get("quantization_config", {})
text_cfg = cfg.get("text_config", {})
if qc.get("quant_method") != "compressed-tensors":
    sys.exit("only compressed-tensors checkpoints are supported")
if "lm_head" not in qc.get("ignore", []):
    sys.exit("lm_head is not in the ignore list; it is already quantized")
if cfg.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings")):
    sys.exit("lm_head is tied to the embeddings; nothing separate to convert")

idx = json.load(open(src + "model.safetensors.index.json"))
shard = idx["weight_map"]["lm_head.weight"]
os.makedirs(dst, exist_ok=True)

# Link everything we don't rewrite back to the snapshot, relatively, so the
# links resolve both on the host and inside the container.
for f in os.listdir(src):
    if f not in {"config.json", "model.safetensors.index.json", shard}:
        link = dst + f
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(src + f, dst), link)

# One scale per output row, float8_e4m3fn values.
with safe_open(src + shard, "pt") as fh:
    rest = {k: fh.get_tensor(k) for k in fh.keys() if k != "lm_head.weight"}
    w = fh.get_tensor("lm_head.weight").float()
fmax = torch.finfo(torch.float8_e4m3fn).max
scale = (w.abs().amax(dim=1, keepdim=True) / fmax).clamp(min=1e-12)
q = (w / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
err = ((q.float() * scale - w).abs().mean() / w.abs().mean()).item()
print(f"lm_head {tuple(w.shape)}: mean relative error {err:.2%}")

save_file({"lm_head.weight": q.contiguous(), "lm_head.weight_scale": scale.contiguous()},
          dst + "lm_head-fp8.safetensors", metadata={"format": "pt"})
save_file(rest, dst + shard, metadata={"format": "pt"})

idx["weight_map"]["lm_head.weight"] = "lm_head-fp8.safetensors"
idx["weight_map"]["lm_head.weight_scale"] = "lm_head-fp8.safetensors"
json.dump(idx, open(dst + "model.safetensors.index.json", "w"), indent=2)

qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]
qc["config_groups"]["group_lm_head_fp8"] = {
    "targets": ["re:.*lm_head$"],
    "format": "float-quantized",
    "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                "symmetric": True, "dynamic": False, "group_size": None,
                "observer": "minmax", "actorder": None, "block_structure": None},
    "input_activations": None,
    "output_activations": None,
}
json.dump(cfg, open(dst + "config.json", "w"), indent=2)
print(f"done: {dst}")

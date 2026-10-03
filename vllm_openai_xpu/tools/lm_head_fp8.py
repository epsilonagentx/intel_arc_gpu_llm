"""Make a local copy of a compressed-tensors checkpoint with an FP8 output layer.

Many int4 checkpoints keep `lm_head` at 16-bit. On a dense model with MTP it is
read for the main step and again for every draft token, so halving it is worth
~18% decode (README "Qwen3.8-27B"). vLLM can't quantize `lm_head` as it loads,
but it does load a checkpoint that declares an FP8 `lm_head`, and XPU runs that
on its native FP8 W8A16 kernel.

--mtp-int4 also packs the MTP draft head's linear layers to int4, in the same
format and group size as the checkpoint's own int4 layers. That's another ~3%.
The draft head only proposes tokens and the main model checks every one, so
answers stay the same apart from rare rewordings at near-ties.

Run it inside the engine image, as your own user, with the model folder
mounted where the engine sees it (from vllm_openai_xpu/):

    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \\
      -v ~/models/hf:/cache/huggingface -v "$PWD":/work:ro \\
      --entrypoint python3 vllm/vllm-openai-xpu:v0.30.0 \\
      /work/tools/lm_head_fp8.py RedHatAI/Qwen3.8-27B-INT4

The copy lands in /cache/huggingface/local/<repo>-fp8head (or
<repo>-fp8head-int4mtp), which is what VLLM_MODEL then points at. Every file
links back to the original snapshot except config.json, the index,
lm_head-fp8.safetensors and the shards this rewrites. The original is only
read.
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
ap.add_argument("--mtp-int4", action="store_true",
                help="also pack the MTP draft head to int4")
ap.add_argument("--name", help="output folder name (default: <repo>-fp8head"
                               "[-int4mtp])")
args = ap.parse_args()

snaps = glob.glob(f"{CACHE}/hub/models--{args.model.replace('/', '--')}/snapshots/*/")
if len(snaps) != 1:
    sys.exit(f"expected one cached snapshot of {args.model}, found {len(snaps)}")
src = snaps[0]
default_name = args.model.split("/")[-1] + "-fp8head" + ("-int4mtp" if args.mtp_int4 else "")
dst = f"{CACHE}/local/{args.name or default_name}/"

cfg = json.load(open(src + "config.json"))
qc = cfg.get("quantization_config", {})
text_cfg = cfg.get("text_config", {})
if qc.get("quant_method") != "compressed-tensors":
    sys.exit("only compressed-tensors checkpoints are supported")
if "lm_head" not in qc.get("ignore", []):
    sys.exit("lm_head is not in the ignore list; it is already quantized")
if cfg.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings")):
    sys.exit("lm_head is tied to the embeddings; nothing separate to convert")

group_size = None
if args.mtp_int4:
    # The draft head reuses the checkpoint's own int4 scheme for `Linear`, so
    # that scheme has to be one we can reproduce exactly.
    for g in qc["config_groups"].values():
        w = g.get("weights") or {}
        if "Linear" in g.get("targets", []) and w.get("num_bits") == 4 \
                and w.get("type") == "int" and w.get("symmetric") \
                and w.get("strategy") == "group" \
                and (g.get("format") or qc.get("format")) == "pack-quantized":
            group_size = w["group_size"]
    if group_size is None:
        sys.exit("--mtp-int4 needs a symmetric int4 group scheme for Linear")

idx = json.load(open(src + "model.safetensors.index.json"))
wm = idx["weight_map"]
lm_shard = wm["lm_head.weight"]
mtp_shards = {f for k, f in wm.items() if k.startswith("mtp.")} if args.mtp_int4 else set()
if args.mtp_int4 and not mtp_shards:
    sys.exit("no mtp.* weights in this checkpoint")
rewrite = {lm_shard} | mtp_shards
os.makedirs(dst, exist_ok=True)

# Link everything we don't rewrite back to the snapshot, relatively, so the
# links resolve both on the host and inside the container.
for f in os.listdir(src):
    if f not in {"config.json", "model.safetensors.index.json"} | rewrite:
        link = dst + f
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(src + f, dst), link)


def lm_head_to_fp8(w):
    """One scale per output row, float8_e4m3fn values."""
    w = w.float()
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale = (w.abs().amax(dim=1, keepdim=True) / fmax).clamp(min=1e-12)
    q = (w / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    err = ((q.float() * scale - w).abs().mean() / w.abs().mean()).item()
    print(f"lm_head {tuple(w.shape)}: mean relative error {err:.2%}")
    return {"lm_head.weight": q.contiguous(), "lm_head.weight_scale": scale.contiguous()}


def linear_to_int4(name, w):
    """Symmetric int4, absmax/7 per group, packed like the checkpoint's own."""
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    w = w.float()
    n, k = w.shape
    g = w.view(n, k // group_size, group_size)
    scale = (g.abs().amax(-1) / 7).clamp(min=1e-8)
    q = torch.round(g / scale.unsqueeze(-1)).clamp(-8, 7).view(n, k).to(torch.int8)
    err = ((q.float().view_as(g) * scale.unsqueeze(-1) - g).abs().mean() / g.abs().mean()).item()
    print(f"{name} {n}x{k}: int4 mean relative error {err:.1%}")
    return {f"{name}.weight_packed": pack_to_int32(q, 4),
            f"{name}.weight_scale": scale.to(torch.bfloat16),
            f"{name}.weight_shape": torch.tensor([n, k], dtype=torch.int64)}


for shard in sorted(rewrite):
    out = {}
    with safe_open(src + shard, "pt") as fh:
        for key in fh.keys():
            t = fh.get_tensor(key)
            if key == "lm_head.weight":
                fp8 = lm_head_to_fp8(t)
                save_file(fp8, dst + "lm_head-fp8.safetensors", metadata={"format": "pt"})
                for k in fp8:
                    wm[k] = "lm_head-fp8.safetensors"
            elif key.startswith("mtp.") and key.endswith(".weight") and t.dim() == 2 \
                    and shard in mtp_shards:
                del wm[key]
                for k, v in linear_to_int4(key[: -len(".weight")], t).items():
                    out[k] = v
                    wm[k] = shard
            else:
                out[key] = t
    save_file(out, dst + shard, metadata={"format": "pt"})

json.dump(idx, open(dst + "model.safetensors.index.json", "w"), indent=2)

qc["ignore"] = [i for i in qc["ignore"]
                if i != "lm_head" and not (args.mtp_int4 and "mtp" in i)]
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

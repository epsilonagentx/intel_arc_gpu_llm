"""Quantize the output layer (and optionally the MTP draft head) of a checkpoint.

Makes a local copy of a compressed-tensors checkpoint whose 16-bit `lm_head`
is stored as FP8 (default) or int4, and with --mtp-int4 whose MTP draft head
is int4 too. vLLM can't quantize these layers as it loads, but it loads a
checkpoint that declares them quantized. How to run it, what it changes and
what it measured: ./README.md.
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
ap.add_argument("--head", choices=("fp8", "int4"), default="fp8",
                help="precision for lm_head (default: fp8)")
ap.add_argument("--mtp-int4", action="store_true",
                help="also pack the MTP draft head to int4")
ap.add_argument("--name", help="output folder name (default: <repo>-<head>head"
                               "[-int4mtp])")
args = ap.parse_args()

snaps = glob.glob(f"{CACHE}/hub/models--{args.model.replace('/', '--')}/snapshots/*/")
if len(snaps) != 1:
    sys.exit(f"expected one cached snapshot of {args.model}, found {len(snaps)}")
src = snaps[0]
default_name = (args.model.split("/")[-1] + f"-{args.head}head"
                + ("-int4mtp" if args.mtp_int4 else ""))
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

# int4 reuses the checkpoint's own int4 scheme for `Linear`, so that scheme
# has to be one we can reproduce exactly: symmetric, grouped, pack-quantized.
int4_group = None
if args.head == "int4" or args.mtp_int4:
    for g in qc["config_groups"].values():
        w = g.get("weights") or {}
        if "Linear" in g.get("targets", []) and w.get("num_bits") == 4 \
                and w.get("type") == "int" and w.get("symmetric") \
                and w.get("strategy") == "group" \
                and (g.get("format") or qc.get("format")) == "pack-quantized":
            int4_group = g
    if int4_group is None:
        sys.exit("int4 needs the checkpoint to use symmetric int4 groups for Linear")
    group_size = int4_group["weights"]["group_size"]

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


def to_fp8(name, w):
    """One scale per output row, float8_e4m3fn values."""
    w = w.float()
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale = (w.abs().amax(dim=1, keepdim=True) / fmax).clamp(min=1e-12)
    q = (w / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    err = ((q.float() * scale - w).abs().mean() / w.abs().mean()).item()
    print(f"{name} {tuple(w.shape)}: FP8 mean relative error {err:.2%}")
    return {f"{name}.weight": q.contiguous(), f"{name}.weight_scale": scale.contiguous()}


def to_int4(name, w):
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


head_file = f"lm_head-{args.head}.safetensors"
for shard in sorted(rewrite):
    out = {}
    with safe_open(src + shard, "pt") as fh:
        for key in fh.keys():
            t = fh.get_tensor(key)
            if key == "lm_head.weight":
                del wm[key]
                head = to_fp8("lm_head", t) if args.head == "fp8" else to_int4("lm_head", t)
                save_file(head, dst + head_file, metadata={"format": "pt"})
                for k in head:
                    wm[k] = head_file
            elif key.startswith("mtp.") and key.endswith(".weight") and t.dim() == 2 \
                    and shard in mtp_shards:
                del wm[key]
                for k, v in to_int4(key[: -len(".weight")], t).items():
                    out[k] = v
                    wm[k] = shard
            else:
                out[key] = t
    save_file(out, dst + shard, metadata={"format": "pt"})

json.dump(idx, open(dst + "model.safetensors.index.json", "w"), indent=2)

qc["ignore"] = [i for i in qc["ignore"]
                if i != "lm_head" and not (args.mtp_int4 and "mtp" in i)]
if args.head == "fp8":
    qc["config_groups"]["group_lm_head_fp8"] = {
        "targets": ["re:.*lm_head$"],
        "format": "float-quantized",
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                    "symmetric": True, "dynamic": False, "group_size": None,
                    "observer": "minmax", "actorder": None, "block_structure": None},
        "input_activations": None,
        "output_activations": None,
    }
else:
    qc["config_groups"]["group_lm_head_int4"] = {
        **json.loads(json.dumps(int4_group)),
        "targets": ["re:.*lm_head$"],
        "format": "pack-quantized",
    }
json.dump(cfg, open(dst + "config.json", "w"), indent=2)
print(f"done: {dst}")

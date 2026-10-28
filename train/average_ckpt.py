"""Average HF checkpoints of the same architecture (weight averaging / model soup). CPU, numpy only.

    python train/average_ckpt.py dirA dirB [dirC ...] --out merged [--weights 0.5 0.5]

On Kaggle, run it as a script with CONFIG {"models": [HF ids or dirs], "weights": [...], "out_repo": ...}
and ophyx-keys attached. Every tensor is averaged in float32 and saved as float32. Config,
tokenizer and generation config come from the first checkpoint. Shapes and tensor names must match
exactly, or it stops.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
from safetensors.numpy import load_file, save_file


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        out.update(load_file(f))
    if not out:
        raise SystemExit(f"no safetensors in {d}")
    return out


def average(dirs, weights, out):
    if weights and len(weights) != len(dirs):
        raise SystemExit(f"{len(dirs)} checkpoints but {len(weights)} weights")
    w = np.array(weights if weights else [1.0] * len(dirs), dtype=np.float64)
    w = w / w.sum()
    acc, dtypes = None, {}
    for d, wi in zip(dirs, w):
        sd = load_dir(d)
        if acc is None:
            acc = {k: v.astype(np.float32) * np.float32(wi) for k, v in sd.items()}
            dtypes = {k: str(v.dtype) for k, v in sd.items()}
            continue
        if set(sd) != set(acc):
            raise SystemExit(f"{d}: tensor names differ ({len(set(sd) ^ set(acc))} differ)")
        for k, v in sd.items():
            if v.shape != acc[k].shape:
                raise SystemExit(f"{d}: {k} has shape {v.shape}, expected {acc[k].shape}")
            acc[k] += v.astype(np.float32) * np.float32(wi)
        del sd
    os.makedirs(out, exist_ok=True)
    save_file(acc, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(dirs[0]):
        if f.endswith((".json", ".txt", ".model")) and not f.endswith("index.json"):
            shutil.copy(os.path.join(dirs[0], f), os.path.join(out, f))
    cfg = os.path.join(out, "config.json")
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        c["torch_dtype"] = "float32"
        if "rope_theta" not in c and "rope_theta" in (c.get("rope_parameters") or {}):
            c["rope_theta"] = c["rope_parameters"]["rope_theta"]  # transformers < 5 reads only this
        json.dump(c, open(cfg, "w"), indent=1)
    json.dump({"sources": dirs, "weights": w.tolist(), "source_dtypes": sorted(set(dtypes.values()))},
              open(os.path.join(out, "average.json"), "w"), indent=1)
    print(f"averaged {len(dirs)} checkpoints, {len(acc)} tensors -> {out}", flush=True)


def resolve(m, token, cache):
    """A dir, a glob (kernel outputs mount at varying depths under /kaggle/input), or an HF id."""
    if m in cache:
        return cache[m]
    from huggingface_hub import snapshot_download
    if os.path.isdir(m):
        d = m
    elif any(c in m for c in "*?["):
        hits = sorted(glob.glob(m, recursive=True))
        if len(hits) != 1:
            raise SystemExit(f"{m}: {len(hits)} matches, need exactly 1: {hits[:5]}")
        d = hits[0]
    else:
        d = snapshot_download(m, local_dir=f"/kaggle/tmp/avg/{len(cache)}", token=token,
                              allow_patterns=["*.json", "*.safetensors", "*.txt"])
    cache[m] = d
    return d


def kaggle_main(cfg):
    """CONFIG {"models": [...], "weights": [...], "out_repo": ...} for one average, or
    {"soups": {name: {"models": [...], "weights": [...]}}} for several, saved under /kaggle/working/<name>/."""
    from huggingface_hub import HfApi
    keys = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    token = json.load(open(keys[0]))["HF_TOKEN"] if keys else None
    soups = cfg.get("soups") or {"averaged": {"models": cfg["models"], "weights": cfg.get("weights"),
                                              "out_repo": cfg.get("out_repo")}}
    cache = {}
    for name, s in soups.items():
        dirs = [resolve(m, token, cache) for m in s["models"]]
        out = f"/kaggle/working/{name}"
        average(dirs, s.get("weights"), out)
        if s.get("out_repo"):
            api = HfApi(token=token)
            api.create_repo(s["out_repo"], repo_type="model", private=True, exist_ok=True)
            api.upload_folder(repo_id=s["out_repo"], folder_path=out, commit_message=f"Average of {len(dirs)} checkpoints")
            print("uploaded to", s["out_repo"], flush=True)


if __name__ == "__main__":
    if "CONFIG" in globals():
        kaggle_main(CONFIG)  # noqa: F821  (set by kaggle/launch.py)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("dirs", nargs="+")
        ap.add_argument("--out", required=True)
        ap.add_argument("--weights", nargs="*", type=float)
        a = ap.parse_args()
        average(a.dirs, a.weights, a.out)

"""Turn a trainer checkpoint (local dir or the HF backup) into a Hugging Face Qwen3 model, on CPU.

  python train/export_ckpt.py hf://abhash-chakraborty/pycoder-runs/nano125m-v1/latest out/nano125m-v1-hf
  python train/export_ckpt.py run/nano125m-v1/ckpt-00007629 out/nano125m-v1-hf --push abhash-chakraborty/pycoder-nano125m

Model size, vocab and seq come from the checkpoint's meta.json. Needs jax (CPU is fine), numpy,
safetensors and huggingface_hub. The tokenizer comes from the data repo unless --tokenizer is given.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--tokenizer", default=None)
ap.add_argument("--data-repo", default="abhash-chakraborty/pycoder-data")
ap.add_argument("--data-version", default="v1")
ap.add_argument("--push", default=None, help="private HF model repo to upload the export to")
args = ap.parse_args()

env = {}
for line in open(os.path.join(ROOT, ".env")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        env[k] = v.strip("'\"")
token = env.get("HF_TOKEN")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

src = args.src
if src.startswith("hf://"):
    repo, prefix = "/".join(src[5:].split("/")[:2]), "/".join(src[5:].split("/")[2:])
    d = snapshot_download(repo, allow_patterns=[f"{prefix}/params.npz", f"{prefix}/meta.json"], token=token,
                          local_dir=os.path.join(args.out + ".src"))
    src = os.path.join(d, prefix)
meta = json.load(open(os.path.join(src, "meta.json")))
run = meta["config"]
tok = args.tokenizer or hf_hub_download(args.data_repo, f"{args.data_version}/tokenizer/tokenizer.json",
                                        repo_type="dataset", token=token, local_dir=args.out + ".tok")

# The trainer as a library, with the run's own model settings, so the parameter tree matches exactly.
g = {"__name__": "pretrain", "CONFIG": {**{k: run[k] for k in ("model", "vocab", "seq", "eos_id") if k in run},
                                        "library": True, "tokenizer_path": tok}}
exec(compile(open(os.path.join(ROOT, "train", "pretrain.py")).read(), "pretrain.py", "exec"), g)
jax = g["jax"]
dims, vocab, seq = g["model_dims"](), run.get("vocab", 32768), run.get("seq", 2048)
like = jax.eval_shape(lambda: g["init_params"](dims, vocab, jax.random.PRNGKey(0)))
params = g["load_params"](src, like)
g["export_hf"](params, dims, vocab, seq, args.out, eos_id=run.get("eos_id", 0))
json.dump({"source": args.src, "step": meta["step"], "run_name": run.get("run_name")},
          open(os.path.join(args.out, "pycoder_export.json"), "w"), indent=1)
print("exported step", meta["step"], "to", args.out, sorted(os.listdir(args.out)))

if args.push:
    api = HfApi(token=token)
    api.create_repo(args.push, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=args.push, folder_path=args.out, commit_message=f"{run.get('run_name')} step {meta['step']}")
    print("uploaded to", args.push)
sys.exit(0)

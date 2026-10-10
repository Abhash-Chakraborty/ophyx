"""Copy token files from the HF dataset into this kernel's output, so training sessions attach them
instantly (kernel_sources) instead of spending TPU time on downloads. Kaggle CPU, with keys.

Output: /kaggle/working/tokens/<bucket>/{train,val}/*.bin, tokenizer.json, manifest.json.
Kernel outputs are capped at ~20 GB; the job stops before copying if the total would not fit.
"""

import collections
import glob
import json
import os
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
REPO = CONFIG.get("repo", "abhash-chakraborty/pycoder-data")
V = CONFIG.get("version", "v1")
OUT = "/kaggle/working" if os.path.isdir("/kaggle") else os.path.abspath("pack-out")
LIMIT = CONFIG.get("limit_gb", 19.0) * 1e9
T0 = time.time()

keys = json.load(open(glob.glob("/kaggle/input/**/keys.json", recursive=True)[0]))
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

api = HfApi(token=keys["HF_TOKEN"])
info = api.dataset_info(REPO, files_metadata=True)
files = [s for s in info.siblings if s.rfilename.startswith(f"{V}/tokens/") and s.rfilename.endswith(".bin")]
total = sum(s.size or 0 for s in files)
print(f"{len(files)} token files, {total / 1e9:.2f} GB", flush=True)
if total > LIMIT:
    raise SystemExit(f"{total / 1e9:.1f} GB does not fit in the kernel output; split the pack by bucket")
snapshot_download(REPO, repo_type="dataset", token=keys["HF_TOKEN"], local_dir=os.path.join(OUT, "_dl"),
                  allow_patterns=[f"{V}/tokens/*/*/*.bin", f"{V}/tokenizer/*"], max_workers=16)
os.rename(os.path.join(OUT, "_dl", V, "tokens"), os.path.join(OUT, "tokens"))
os.rename(os.path.join(OUT, "_dl", V, "tokenizer", "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))
manifest = collections.defaultdict(lambda: collections.Counter())
for p in glob.glob(os.path.join(OUT, "tokens", "*", "*", "*.bin")):
    bucket, split = p.split(os.sep)[-3:-1]
    manifest[bucket][split] += os.path.getsize(p) // 2
manifest = {b: dict(c) for b, c in sorted(manifest.items())}
json.dump({"version": V, "tokens": manifest, "total_train_tokens": sum(c.get("train", 0) for c in manifest.values())},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
import shutil  # noqa: E402

shutil.rmtree(os.path.join(OUT, "_dl"), ignore_errors=True)
for b, c in manifest.items():
    print(f"{b:<36} train {c.get('train', 0) / 1e6:9.1f}M  val {c.get('val', 0) / 1e6:6.2f}M", flush=True)
print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)

"""Bring models and Python wheels into a Kaggle output, for sessions that must run with internet off.

Kaggle CPU, internet on, keys attached (pycoder-keys); it runs no model or generated code. Attach this
kernel's output to an offline session (GRPO, sandboxed scoring) as a kernel source.

CONFIG: models [{"repo": "abhash-chakraborty/ophyx-350m-sft", "dir": "ophyx-350m-sft"}],
pip ["trl"] (requirement strings). Wheels are the packages pip would add or upgrade on top of this
image (pip install --dry-run --report), downloaded without their dependencies, so the offline
session installs them with `pip install --no-index --find-links wheels/ ...`.

Output under /kaggle/working: model/<dir>/ (config, tokenizer, safetensors), wheels/*.whl, relay.json.
"""

import glob
import json
import os
import subprocess
import sys

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
OUT = CONFIG.get("out", "/kaggle/working")
KEYS = json.load(open(glob.glob("/kaggle/input/**/keys.json", recursive=True)[0]))


def main():
    from huggingface_hub import snapshot_download
    info = {"models": {}, "wheels": []}
    for m in CONFIG.get("models", []):
        d = os.path.join(OUT, "model", m["dir"])
        snapshot_download(m["repo"], local_dir=d, token=KEYS["HF_TOKEN"],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        info["models"][m["dir"]] = {"repo": m["repo"], "files": sorted(os.listdir(d))}
        print("model", m["repo"], "->", d, flush=True)
    reqs = CONFIG.get("pip", [])
    if reqs:
        wd = os.path.join(OUT, "wheels")
        os.makedirs(wd, exist_ok=True)
        rep = os.path.join(OUT, "pip-report.json")
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", "--report", rep, *reqs], check=True)
        pins = [f"{x['metadata']['name']}=={x['metadata']['version']}" for x in json.load(open(rep))["install"]]
        print("wheels needed:", pins, flush=True)
        if pins:
            subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", wd, *pins], check=True)
        info["wheels"] = sorted(os.listdir(wd))
        info["requirements"] = reqs
    json.dump(info, open(os.path.join(OUT, "relay.json"), "w"), indent=1)
    print("DONE", json.dumps(info)[:2000], flush=True)


main()

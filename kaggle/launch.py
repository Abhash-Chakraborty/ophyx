"""Package a job script as a Kaggle kernel and push it.

The script gets a `CONFIG = {...}` preamble, so one source file serves every shard.
Runs on the orchestration box (light: writes two small files and calls the Kaggle CLI).

  python kaggle/launch.py data/jobs/ingest.py pycoder-data-stack-edu-0 \
      --config '{"job": "stack_edu", "shard": 0, "nshards": 3}' --datasets pycoder-keys --timeout 28800
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

USER = "abhashchakraborty"
KAGGLE = shutil.which("kaggle") or os.path.expanduser("~/.local/bin/kaggle")
ACCEL = {"cpu": None, "t4": "NvidiaTeslaT4", "tpu": "TpuV5E8"}


def load_env(path):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v.strip().strip("'\"")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("name", help="kernel slug, e.g. pycoder-data-stack-edu-0")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--accel", choices=ACCEL, default="cpu")
    ap.add_argument("--timeout", type=int, default=12 * 3600)
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--kernel-sources", nargs="*", default=[])
    ap.add_argument("--no-internet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.loads(a.config)
    build = tempfile.mkdtemp(prefix=f"{a.name}-")
    with open(os.path.join(root, a.script)) as f:
        src = f.read()
    with open(os.path.join(build, "main.py"), "w") as f:
        f.write(f"CONFIG = {json.dumps(cfg)}\n" + src)
    meta = {
        "id": f"{USER}/{a.name}",
        "title": a.name,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": a.accel == "t4",
        "enable_tpu": a.accel == "tpu",
        "enable_internet": not a.no_internet,
        "dataset_sources": [d if "/" in d else f"{USER}/{d}" for d in a.datasets],
        "competition_sources": [],
        "kernel_sources": [k if "/" in k else f"{USER}/{k}" for k in a.kernel_sources],
    }
    with open(os.path.join(build, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if a.dry_run:
        print("built", build)
        return
    env = {**os.environ, **{k: v for k, v in load_env(os.path.join(root, ".env")).items() if k == "KAGGLE_API_TOKEN"}}
    cmd = [KAGGLE, "kernels", "push", "-p", build, "-t", str(a.timeout)]
    if ACCEL[a.accel]:
        cmd += ["--accelerator", ACCEL[a.accel]]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "no output")
    shutil.rmtree(build, ignore_errors=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

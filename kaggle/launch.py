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
ACCEL = {"cpu": None, "t4": "NvidiaTeslaT4", "tpu": "TpuV5E8", "tpu6": "TpuV6E8"}


def load_env(path):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v.strip().strip("'\"")
    return env


def lint(path):
    """Undefined names, use-after-del, unused locals on the fully assembled script (includes and all).
    Two relaunches failed on bugs this catches, after costing a Kaggle round trip each."""
    ruff = shutil.which("ruff") or os.path.expanduser("~/.cache/ophyx-smoke/venv/bin/ruff")
    if not os.path.exists(ruff):
        print("warning: ruff not found; skipping lint")
        return True
    r = subprocess.run([ruff, "check", "--select", "F821,F823,F841", "--output-format", "concise", "--quiet",
                        "--no-cache", path], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout.strip()[-3000:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("name", help="kernel slug, e.g. pycoder-data-stack-edu-0")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--accel", choices=ACCEL, default="cpu")
    ap.add_argument("--timeout", type=int, default=12 * 3600)
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--kernel-sources", nargs="*", default=[])
    ap.add_argument("--include", nargs="*", default=[], help="files prepended before the script, in order")
    ap.add_argument("--no-internet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-lint", action="store_true", help="skip the pre-push ruff check (not recommended)")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.loads(a.config)
    build = tempfile.mkdtemp(prefix=f"{a.name}-")
    src = ""
    for inc in a.include:
        with open(os.path.join(root, inc)) as f:
            src += f.read() + "\n\n"
    with open(os.path.join(root, a.script)) as f:
        src += f.read()
    with open(os.path.join(build, "main.py"), "w") as f:
        f.write(f"CONFIG = {cfg!r}\n" + src)
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
    if not a.no_lint and not lint(os.path.join(build, "main.py")):
        shutil.rmtree(build, ignore_errors=True)
        sys.exit("lint failed: not pushing (fix it, or --no-lint)")
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

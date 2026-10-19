"""Keeps multi-session Kaggle training chains going. Runs hourly on GitHub Actions (or by hand).

A chain (kaggle/chains/<name>.json) alternates two kernels, A and B: each session attaches the
other's output (a notebook can't attach its own), resumes from the newest checkpoint (attached
output or the HF backup), trains until its budget, saves and exits. This script:

  - does nothing while either kernel is queued or running;
  - stops, with a phone notification, when the run has finished, hit a NaN, or failed twice in a row;
  - otherwise pushes the next session.

A chain with "after": "<run>" waits until that run's HF status says finished (SFT after pretraining).

State lives next to the checkpoint backup on HF: <run>/status.json (written by the trainer) and
<run>/chain_state.json (written here). Needs KAGGLE_API_TOKEN, HF_TOKEN and NTFY_TOPIC in the environment.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "abhashchakraborty"
KAGGLE = os.environ.get("KAGGLE_BIN", "kaggle")


def env_or_dotenv(key):
    if os.environ.get(key):
        return os.environ[key]
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith(key + "="):
                return line.strip().split("=", 1)[1].strip("'\"")
    return None


def notify(title, msg, tags="robot"):
    topic = env_or_dotenv("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode()[:3000],
                                                          headers={"Title": title, "Tags": tags}, method="POST"), timeout=15)
        except Exception as e:
            print("ntfy failed:", e)


def kernel_status(slug):
    e = {**os.environ, "KAGGLE_API_TOKEN": env_or_dotenv("KAGGLE_API_TOKEN") or ""}
    r = subprocess.run([KAGGLE, "kernels", "status", f"{USER}/{slug}"], capture_output=True, text=True, env=e)
    out = (r.stdout + r.stderr).strip()
    for s in ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCEL"):
        if s in out.upper():
            return s
    return "NONE"  # never pushed, or not found


class HF:
    def __init__(self, repo, run):
        from huggingface_hub import HfApi
        self.api, self.repo, self.run = HfApi(token=env_or_dotenv("HF_TOKEN")), repo, run
        self.api.create_repo(repo, repo_type="model", private=True, exist_ok=True)

    def read(self, name):
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(self.repo, f"{self.run}/{name}", token=env_or_dotenv("HF_TOKEN"),
                                force_download=True, cache_dir=tempfile.mkdtemp())
            return json.load(open(p))
        except Exception:
            return None

    def write(self, name, obj):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(obj, f, indent=1)
        self.api.upload_file(path_or_fileobj=f.name, path_in_repo=f"{self.run}/{name}", repo_id=self.repo,
                             repo_type="model", commit_message=f"watchdog: {name}")


def push(chain, which, previous):
    c = chain
    cmd = [sys.executable, os.path.join(ROOT, "kaggle", "launch.py"), c["script"], c["kernels"][which],
           "--accel", c["accel"], "--timeout", str(c["timeout"]), "--config", json.dumps(c["config"]),
           "--datasets", *c.get("datasets", [])]
    sources = list(c.get("kernel_sources", [])) + ([previous] if previous else [])
    if sources:
        cmd += ["--kernel-sources", *sources]
    if c.get("include"):
        cmd += ["--include", *c["include"]]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "KAGGLE_API_TOKEN": env_or_dotenv("KAGGLE_API_TOKEN") or ""})
    print("push:", (r.stdout + r.stderr).strip()[-300:])
    return r.returncode == 0


def check(chain_path):
    chain = json.load(open(chain_path))
    name = os.path.basename(chain_path).replace(".json", "")
    if not chain.get("enabled", False):
        print(f"{name}: disabled")
        return
    run = chain["config"]["run_name"]
    hf = HF(chain["config"]["hf_ckpt_repo"], run)
    state = hf.read("chain_state.json") or {"last": None, "failures": 0, "sessions": 0, "stopped": False}
    if state.get("stopped"):
        print(f"{name}: stopped ({state.get('why')})")
        return
    if chain.get("after") and not (HF(chain["config"]["hf_ckpt_repo"], chain["after"]).read("status.json") or {}).get("finished"):
        print(f"{name}: waiting for {chain['after']} to finish")  # e.g. SFT on a base run that is still training
        return
    statuses = {k: kernel_status(k) for k in chain["kernels"]}
    print(f"{name}: {statuses}, state {state}")
    if any(s in ("QUEUED", "RUNNING") for s in statuses.values()):
        return
    status = hf.read("status.json")
    if status and status.get("finished"):
        state.update(stopped=True, why="finished")
        hf.write("chain_state.json", state)
        notify(f"{run}: training finished", json.dumps(status), "tada")
        return
    if status and status.get("reason") == "nan":
        state.update(stopped=True, why="nan")
        hf.write("chain_state.json", state)
        notify(f"{run}: stopped on NaN loss", json.dumps(status), "warning")
        return
    if state["last"] is not None:
        last_status = statuses[chain["kernels"][state["last"]]]
        progressed = status is not None and status.get("step", -1) > state.get("last_step", -1)
        if last_status != "COMPLETE" or not progressed:
            state["failures"] += 1
        else:
            state["failures"] = 0
        state["last_step"] = status.get("step", -1) if status else -1
        if state["failures"] >= 2:
            state.update(stopped=True, why=f"two failed sessions in a row ({last_status})")
            hf.write("chain_state.json", state)
            notify(f"{run}: chain stopped", f"Two sessions in a row failed or made no progress. Last kernel status: {last_status}. "
                   f"Status: {json.dumps(status)}", "x")
            return
    which = 0 if state["last"] is None else 1 - state["last"]
    previous = chain["kernels"][state["last"]] if state["last"] is not None else None
    if push(chain, which, previous):
        state.update(last=which, sessions=state["sessions"] + 1, pushed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        hf.write("chain_state.json", state)
        notify(f"{run}: session {state['sessions']} queued", f"{chain['kernels'][which]} (step {status and status.get('step')})", "rocket")
    else:
        state["failures"] += 1
        hf.write("chain_state.json", state)
        notify(f"{run}: push failed", "kaggle kernels push returned an error; will retry next hour", "warning")


if __name__ == "__main__":
    for p in sorted(glob.glob(os.path.join(ROOT, "kaggle", "chains", "*.json"))):
        try:
            check(p)
        except Exception as e:
            print(f"{p}: {type(e).__name__}: {e}")
            notify("watchdog error", f"{os.path.basename(p)}: {type(e).__name__}: {str(e)[:300]}", "warning")

"""Wait for a session's heartbeat on HF (sessions/<name>-*.json) and print its latest content.

  python kaggle/tools/poll_heartbeat.py seq1 [timeout_s] [after_utc_iso]
"""

import json
import os
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env = dict(os.environ)
if os.path.exists(os.path.join(ROOT, ".env")):
    for line in open(os.path.join(ROOT, ".env")):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env.setdefault(k, v.strip("'\""))
if not env.get("HF_TOKEN"):
    sys.exit("HF_TOKEN is not set (environment or .env)")
REPO = env.get("OPHYX_RUNS_REPO", "abhash-chakraborty/ophyx-runs")
name = sys.argv[1]
deadline = time.time() + (int(sys.argv[2]) if len(sys.argv) > 2 else 3500)
api = HfApi(token=env["HF_TOKEN"])
while time.time() < deadline:
    try:
        fs = sorted(f for f in api.list_repo_files(REPO) if f.startswith(f"sessions/{name}-"))
        if fs:
            b = json.load(open(hf_hub_download(REPO, fs[-1], token=env["HF_TOKEN"], force_download=True)))
            print(fs[-1], json.dumps({k: b.get(k) for k in ("utc", "uptime_s", "phase", "run", "step", "loss", "tokens_per_s")}))
            print("\n".join(b.get("tail", [])[-12:]))
            sys.exit(0)
    except Exception:
        pass
    time.sleep(180)
print(time.strftime("%H:%M UTC", time.gmtime()), "no heartbeat yet for", name)
sys.exit(1)

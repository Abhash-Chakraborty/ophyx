"""Wait until a run's hourly HF checkpoint backup appears (or advances), then print its step.

  python kaggle/tools/poll_backup.py nano125m-v1 [min_step] [timeout_s]
"""

import json
import os
import sys
import time

from huggingface_hub import hf_hub_download

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
run = sys.argv[1]
min_step = int(sys.argv[2]) if len(sys.argv) > 2 else 0
deadline = time.time() + (int(sys.argv[3]) if len(sys.argv) > 3 else 3500)
while time.time() < deadline:
    try:
        m = json.load(open(hf_hub_download(REPO, f"{run}/latest/meta.json",
                                           token=env["HF_TOKEN"], force_download=True)))
        if m["step"] > min_step:
            print(time.strftime("%H:%M UTC", time.gmtime()), "backup at step", m["step"])
            sys.exit(0)
    except Exception:
        pass
    time.sleep(120)
print(time.strftime("%H:%M UTC", time.gmtime()), "no new backup yet")
sys.exit(1)

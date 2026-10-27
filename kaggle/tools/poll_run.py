"""Wait for any sign of a run on HF: its status.json or a checkpoint backup. Prints what it finds.

  python kaggle/tools/poll_run.py nano125m-v1 [timeout_s]
"""

import json
import os
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env = {}
for line in open(os.path.join(ROOT, ".env")):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        env[k] = v.strip("'\"")
run = sys.argv[1]
deadline = time.time() + (int(sys.argv[2]) if len(sys.argv) > 2 else 3500)
api = HfApi(token=env["HF_TOKEN"])
while time.time() < deadline:
    try:
        files = api.list_repo_files("abhash-chakraborty/ophyx-runs")
        found = {}
        for name in ("status.json", "latest/meta.json"):
            if f"{run}/{name}" in files:
                found[name] = json.load(open(hf_hub_download("abhash-chakraborty/ophyx-runs", f"{run}/{name}",
                                                             token=env["HF_TOKEN"], force_download=True)))
        if found:
            print(time.strftime("%H:%M UTC", time.gmtime()), json.dumps(found)[:800])
            sys.exit(0)
    except Exception:
        pass  # repo not created yet
    time.sleep(120)
print(time.strftime("%H:%M UTC", time.gmtime()), "nothing on HF yet for", run)

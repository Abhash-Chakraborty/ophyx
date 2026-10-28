"""Wait until a specific kernel version's output (its log) can be downloaded, then print the useful lines.

  python kaggle/tools/poll_version_log.py ophyx-train-seq1 1 [timeout_s]
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env = dict(os.environ)
if os.path.exists(os.path.join(ROOT, ".env")):
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("KAGGLE_API_TOKEN=") and not env.get("KAGGLE_API_TOKEN"):
            env["KAGGLE_API_TOKEN"] = line.strip().split("=", 1)[1].strip("'\"")
USER = env.get("KAGGLE_USERNAME", "abhashchakraborty")
kernel, version = sys.argv[1], sys.argv[2]
deadline = time.time() + (int(sys.argv[3]) if len(sys.argv) > 3 else 3600)
kaggle = env.get("KAGGLE_BIN", "kaggle")
cache = os.path.expanduser("~/.cache/ophyx-ops")
os.makedirs(cache, exist_ok=True)
while time.time() < deadline:
    d = tempfile.mkdtemp(dir=cache)
    subprocess.run([kaggle, "kernels", "output", f"{USER}/{kernel}/{version}", "-p", d,
                    "--file-pattern", r".*\.(log|json)$"], env=env, capture_output=True, text=True)
    logs = glob.glob(os.path.join(d, "*.log"))
    if logs:
        txt = "".join(e.get("data", "") for e in json.load(open(logs[0])))
        lines = [l for l in txt.splitlines() if l.strip() and "%|" not in l]
        print(time.strftime("%H:%M UTC", time.gmtime()), f"log for {kernel} v{version}: {len(lines)} lines; saved in {d}")
        print("\n".join(x[:240] for x in lines[:25]))
        print("...")
        print("\n".join(x[:240] for x in lines[-25:]))
        sys.exit(0)
    os.rmdir(d) if not os.listdir(d) else None
    time.sleep(180)
print(time.strftime("%H:%M UTC", time.gmtime()), "no log yet")
sys.exit(1)

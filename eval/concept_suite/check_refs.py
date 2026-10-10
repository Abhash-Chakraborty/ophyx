"""Run every task's reference solution against its hidden tests. Every task must pass.

For each task: write solution.py (the reference) and test_task.py (the tests) into a fresh temp
dir, run `python -m pytest -q -x test_task.py` under `nice -n 19` with a timeout, with an empty
environment (HOME and TMPDIR point into the temp dir, so pytest's tmp_path lives there too).
At most 2 tasks run at a time; this machine hosts other services.

    python eval/concept_suite/check_refs.py [--filter SUBSTR] [--repeat N] [--jobs 2] [--slow 3.0]

--repeat runs each task N times (to catch flaky concurrency tests). Tasks slower than --slow
seconds are listed. Exits non-zero if anything fails.
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/.cache/pycoder-smoke/venv/bin/python")
TMP_ROOT = os.path.expanduser("~/.cache/pycoder-smoke/concept-tmp")


def run_one(task, timeout):
    d = tempfile.mkdtemp(prefix="ref-", dir=TMP_ROOT)
    try:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(task["reference"])
        with open(os.path.join(d, "test_task.py"), "w") as f:
            f.write(task["tests"])
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "TMPDIR": d, "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = ["nice", "-n", "19", PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_task.py"]
        t = time.time()
        try:
            p = subprocess.run(cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, timeout, "timeout"
        out = p.stdout.decode(errors="replace")
        return p.returncode == 0, time.time() - t, out[-2500:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.jsonl"))
    ap.add_argument("--filter", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--slow", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(TMP_ROOT, exist_ok=True)
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    tasks = [t for t in tasks if a.filter in t["task_id"]]
    work = [t for t in tasks for _ in range(a.repeat)]
    t0 = time.time()
    fails, times = {}, {}
    with cf.ThreadPoolExecutor(min(a.jobs, 2)) as ex:
        futs = {ex.submit(run_one, t, a.timeout): t for t in work}
        for fut in cf.as_completed(futs):
            t = futs[fut]
            ok, sec, out = fut.result()
            times[t["task_id"]] = max(sec, times.get(t["task_id"], 0))
            if not ok:
                fails.setdefault(t["task_id"], out)
    for tid, out in sorted(fails.items()):
        print(f"FAIL {tid}\n{out}\n")
    slow = sorted(((s, tid) for tid, s in times.items() if s > a.slow), reverse=True)
    for s, tid in slow:
        print(f"SLOW {s:5.2f}s {tid}")
    n = len(tasks)
    print(f"check_refs: {n - len(fails)}/{n} passed ({100 * (n - len(fails)) / max(n, 1):.1f}%), "
          f"repeat={a.repeat}, {len(slow)} slower than {a.slow}s, max {max(times.values(), default=0):.2f}s, "
          f"wall {time.time() - t0:.0f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

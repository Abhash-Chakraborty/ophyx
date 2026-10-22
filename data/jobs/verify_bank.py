"""Run the problem bank's reference solutions against their tests. Kaggle CPU, internet off, no keys.

Appended after data/jobs/judge.py by the launcher. Attach the problem-bank kernel output (bank/bank-*.jsonl).
CONFIG: shard, nshards (problems split by a hash of pid), workers, per_test_s, max_passing (stop
after this many passing solutions per problem, default 5: all of them), budget_s.

A problem is kept when at least one solution passes every kept test. Output, under /kaggle/working:
  verified/bank-<shard>.jsonl   the bank record plus "passing" (indices into solutions), "status" per
                                tried solution (pass, or the first failure's kind) and "inconsistent":
                                true when another solution gives a wrong answer. Those are usually
                                problems that accept several answers (a special judge), which exact
                                output comparison can't check: pass-rate and RL jobs should skip them.
  sft_rows/bank-<shard>.jsonl   one SFT row per kept problem: the statement -> a passing solution
  verified/stats-<shard>.json   counts by source, level and failure kind, and the time taken
"""

import collections
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
SHARD, NSHARDS = CONFIG.get("shard", 0), CONFIG.get("nshards", 1)
SRC = CONFIG.get("src", "/kaggle/input")
OUTW = CONFIG.get("out", "/kaggle/working")
WORKERS = CONFIG.get("workers", (os.cpu_count() or 4) * 2)
PER_TEST = CONFIG.get("per_test_s", 4)
MAX_PASSING = CONFIG.get("max_passing", 5)
BUDGET = CONFIG.get("budget_s", 11 * 3600)
T0 = time.time()

STDIN_ASK = ["Write a Python 3 program for this problem. Read the input from standard input and print the answer.",
             "Solve this in Python. The program reads stdin and writes stdout.",
             "Write a complete Python solution that reads from standard input."]
CALL_ASK = ["Write the function in Python.", "Complete this code in Python:", "Implement it in Python, keeping this signature:"]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def mine(pid):
    return int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) % NSHARDS == SHARD


def verify(rec):
    status, passing = [], []
    for i, sol in enumerate(rec["solutions"]):
        if time.time() - T0 > BUDGET:
            status.append("budget")
            break
        r = judge(sol, rec["tests"], rec["fn_name"], per_test_s=PER_TEST)
        if r["status"] == "pass":
            status.append("pass")
            passing.append(i)
            if len(passing) >= MAX_PASSING:
                break
        else:
            status.append((r.get("first_fail") or {}).get("kind", "fail"))
    return {**rec, "passing": passing, "status": status, "inconsistent": bool(passing) and "wrong_answer" in status}


def sft_row(rec):
    """Statement -> the shortest passing solution, as the user would ask for it."""
    h = int(hashlib.sha1(rec["pid"].encode()).hexdigest()[:8], 16)
    sol = min((rec["solutions"][i] for i in rec["passing"]), key=len)
    q = rec["question"].strip()
    if rec["fn_name"]:
        starter = (rec.get("starter_code") or "").strip()
        ask = CALL_ASK[h % len(CALL_ASK)]
        tail = f"{ask}\n\n```python\n{starter}\n```" if starter else f"Write a Python function `{rec['fn_name']}` for this."
        prompt = f"{q}\n\n{tail}"
    else:
        prompt = f"{q}\n\n{STDIN_ASK[h % len(STDIN_ASK)]}"
    return {"instruction": prompt, "output": sol, "tag": "bank", "seq_id": rec["pid"]}


def main():
    files = sorted(glob.glob(os.path.join(SRC, "**", "bank", "bank-*.jsonl"), recursive=True))
    recs = [r for f in files for r in map(json.loads, open(f)) if mine(r["pid"])]
    log(f"shard {SHARD}/{NSHARDS}: {len(recs)} problems from {len(files)} files, {WORKERS} workers, per test {PER_TEST}s")
    os.makedirs(os.path.join(OUTW, "verified"), exist_ok=True)
    os.makedirs(os.path.join(OUTW, "sft_rows"), exist_ok=True)
    st = collections.Counter()
    kinds = collections.Counter()
    out_v = open(os.path.join(OUTW, "verified", f"bank-{SHARD}.jsonl"), "w")
    out_s = open(os.path.join(OUTW, "sft_rows", f"bank-{SHARD}.jsonl"), "w")
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for n, rec in enumerate(ex.map(verify, recs), 1):
            kinds.update(s for s in rec["status"] if s != "pass")
            key = rec["source"]
            st[f"{key}/problems"] += 1
            st[f"{key}/solutions_tried"] += len(rec["status"])
            st[f"{key}/solutions_passed"] += len(rec["passing"])
            if rec["passing"]:
                st[f"{key}/kept"] += 1
                st[f"{key}/inconsistent"] += rec["inconsistent"]
                st[f"level{rec.get('level')}/kept"] += 1
                out_v.write(json.dumps(rec) + "\n")
                out_s.write(json.dumps(sft_row(rec)) + "\n")
            if n % 500 == 0:
                log(f"{n}/{len(recs)}: kept {sum(v for k, v in st.items() if k.endswith('/kept') and not k.startswith('level'))}")
    out_v.close()
    out_s.close()
    stats = {"shard": SHARD, "counts": dict(sorted(st.items())), "failure_kinds": dict(kinds),
             "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(stats, open(os.path.join(OUTW, "verified", f"stats-{SHARD}.json"), "w"), indent=1)
    log("STATS", json.dumps(stats))


if __name__ == "__main__" and not CONFIG.get("library"):
    main()

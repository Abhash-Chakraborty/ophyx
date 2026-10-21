"""LiveCodeBench leaderboard scores recomputed on exactly the problems we run.

    python eval/lcb_window.py perf.json --ids samples/<model>/lcb.jsonl [--md out.md] [--top 15]
    python eval/lcb_window.py perf.json --from 2024-10-01 --to 2025-05-01

perf.json is https://livecodebench.github.io/performances_generation.json (per-problem pass@1 for
every leaderboard model). The leaderboard page averages it over a date window; here the window is
our own problem set (task ids from any of our lcb sample or result files) or a date range. Only
models with a result on every one of those problems are listed, so the numbers are comparable.
"""

import argparse
import collections
import datetime
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perf")
    ap.add_argument("--ids", nargs="*", help="jsonl files with task_id or question_id per line")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--md")
    a = ap.parse_args()
    d = json.load(open(a.perf))
    want = None
    if a.ids:
        want = set()
        for f in a.ids:
            for line in open(f):
                r = json.loads(line)
                qid = str(r.get("task_id") or r.get("question_id"))
                for plat in ("atcoder_", "leetcode_", "codeforces_"):  # our task ids are <platform>_<question_id>
                    qid = qid[len(plat):] if qid.startswith(plat) else qid
                want.add(qid)
    day = lambda ms: datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    by_model = collections.defaultdict(dict)
    probs = set()
    for p in d["performances"]:
        if want is not None and p["question_id"] not in want:
            continue
        if a.start and day(p["date"]) < a.start or a.end and day(p["date"]) >= a.end:
            continue
        probs.add(p["question_id"])
        by_model[p["model"]][p["question_id"]] = p["pass@1"]
    n = len(want) if want is not None else len(probs)
    full = {m: v for m, v in by_model.items() if len(v) == len(probs)}
    rows = sorted(((sum(v.values()) / len(v), m) for m, v in full.items()), reverse=True)[:a.top]
    lines = [f"LiveCodeBench leaderboard data on {len(probs)} problems"
             + (f" ({n - len(probs)} of our {n} are not in the leaderboard file)" if want is not None and n != len(probs) else ""),
             "", "| Model | pass@1 |", "|---|---|"] + [f"| {m} | {s:.1f} |" for s, m in rows]
    text = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()

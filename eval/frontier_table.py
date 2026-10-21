"""Published scores of large and frontier models, in the columns of eval/R1_RESULTS.md.

    python eval/frontier_table.py [--md out.md] [--top 20]

Reads eval/published/published_scores.json (every row has a source URL). Each column takes one
variant from one leaderboard, so a column is internally comparable. Across columns, and against our
own runs, the differences below remain: they are printed under the table.
"""

import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = [  # label, benchmark, variant (exact), source note
    ("HumanEval", "EvalPlus", "HumanEval (base tests) pass@1 greedy"),
    ("HumanEval+", "EvalPlus", "HumanEval+ pass@1 greedy"),
    ("MBPP", "EvalPlus", "MBPP (EvalPlus 378-task sanitized subset, base tests) pass@1 greedy"),
    ("MBPP+", "EvalPlus", "MBPP+ (378 tasks) pass@1 greedy"),
    ("BCB-H complete", "BigCodeBench", "Hard (148 tasks), Complete split, pass@1"),
    ("BCB-H instruct", "BigCodeBench", "Hard (148 tasks), Instruct split, pass@1"),
    ("LCB v6 Feb-May 25", "LiveCodeBench", "Code generation pass@1, release_v6 data, slider window (2025-02-01 to 2025-05-01)"),
    ("CRUX-I", "CRUXEval", "CRUXEval-I pass@1"),
    ("CRUX-O", "CRUXEval", "CRUXEval-O pass@1"),
]
NOTES = [
    "EvalPlus: leaderboard results.json, last modified 2024-12-26, greedy, mostly instruct prompts.",
    "BigCodeBench-Hard: leaderboard, last modified 2025-04-16, all 148 tasks. Our own runs also report the 131 tasks "
    "whose reference solution passes in our offline sandbox; compare against our all-148 number.",
    "LiveCodeBench: leaderboard per-problem data (2025-08-01), recomputed for the Feb-May 2025 window.",
    "CRUXEval: leaderboard data.csv (2024-12-06), T=0.2 with n=10; our runs are greedy, so they are close but not identical.",
    "Rows from model cards and papers are left out here: they use their own prompts and windows.",
]


def norm(name):
    return " ".join(name.replace("_", " ").split()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--groups", nargs="*", default=["frontier", "open-large"])
    a = ap.parse_args()
    rows = json.load(open(os.path.join(HERE, "published", "published_scores.json")))
    out = []
    for bench in dict.fromkeys(c[1] for c in COLS):  # one table per leaderboard: model names differ between them
        cols = [c for c in COLS if c[1] == bench]
        table, meta = collections.defaultdict(dict), {}
        for r in rows:
            if r["group"] not in a.groups or "REPORTED" in (r.get("note") or ""):
                continue
            if "+cot" in r["model"]:  # chain-of-thought runs; ours use the direct prompt
                continue
            for label, b, variant in cols:
                if r["benchmark"] == b and r["variant"] == variant:
                    key = norm(r["model"])
                    table[key][label] = max(table[key].get(label, r["score"]), r["score"])
                    meta.setdefault(key, (r["model"], r.get("size_b"), r["group"]))
        main_col = cols[1][0] if len(cols) == 4 else cols[0][0]  # HumanEval+ for EvalPlus
        keep = sorted((k for k in table if main_col in table[k]), key=lambda k: -table[k][main_col])[:a.top]
        out.append(f"### {bench}\n")
        out.append("| Model | Size | " + " | ".join(c[0] for c in cols) + " |")
        out.append("|" + "---|" * (2 + len(cols)))
        for k in keep:
            name, size, group = meta[k]
            cells = [name, f"{size:g}B" if size else "closed"]
            cells += ["" if table[k].get(c[0]) is None else f"{table[k][c[0]]:.1f}" for c in cols]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    text = "\n".join(out) + "\n".join(f"- {n}" for n in NOTES) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()

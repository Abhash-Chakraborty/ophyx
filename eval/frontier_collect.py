"""Turn frontier_task.py replies into bench_gen-style samples that bench_exec.py scores.

    python eval/frontier_collect.py <downloaded run dir(s)> --prompts prompts.jsonl --out samples_dir

Each reply becomes {task_id, prefill: "", completion, finish, prompt_tokens, ...extra} in
samples/<model>/<bench>.jsonl, with meta.json kind "instruct". For the benchmarks where bench_gen
prefilled "```python" for open instruct models (EvalPlus, BigCodeBench), the completion is the first
fenced Python block of the reply, cut at the bench's stop strings, as the prefill would have given.
Other benchmarks keep the whole reply; bench_exec extracts code there for every instruct model alike.
Replies missing because of the budget or an API error are written as empty answers (finish "missing"),
so they score as failures; the count is printed.
"""

import argparse
import collections
import glob
import json
import os
import re

FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def first_block(text):
    m = FENCE.search(text)
    return m.group(1) if m else text


def cut(text, stops):
    idx = [i for i in (text.find(s) for s in stops) if i >= 0]
    return text[:min(idx)] if idx else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="dirs from `kaggle b t download` (searched recursively)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = {(r["bench"], r["task_id"]): r for r in map(json.loads, open(a.prompts))}
    got = collections.defaultdict(dict)  # (model, bench) -> task_id -> reply; later runs fill gaps of earlier ones
    for d in a.runs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            model, bench = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-6]
            for r in map(json.loads, open(f)):
                if "response" in r and (bench, r["task_id"]) in prompts:
                    got[(model, bench)].setdefault(r["task_id"], r)
    missing = collections.Counter()
    for (model, bench), replies in sorted(got.items()):
        mdir = os.path.join(a.out, "samples", model)
        os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, f"{bench}.jsonl"), "w") as f:
            for (b, tid), p in prompts.items():
                if b != bench:
                    continue
                r = replies.get(tid)
                if r is None:  # an empty answer, so every task is present (EvalPlus requires that) and it scores as a fail
                    missing[(model, bench)] += 1
                    r = {"response": "", "finish": "missing"}
                text = r["response"]
                if p["code_block"]:
                    text = cut(first_block(text), p["stops"])
                elif p["stops"]:
                    text = cut(text, p["stops"])
                f.write(json.dumps({"task_id": tid, "prefill": "", "completion": text, "finish": r.get("finish"),
                                    "prompt_tokens": r.get("in") or 0, **p["extra"]}) + "\n")
        meta_p = os.path.join(mdir, "meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {"model": f"api:{model}", "name": model, "kind": "instruct",
                                                                         "engine": "kaggle-model-proxy", "benches": {}}
        usd = sum(r.get("usd") or 0 for r in replies.values())
        meta["benches"][bench] = {"n": len(replies), "usd": round(usd, 4),
                                  "gen_tokens": sum(r.get("out") or 0 for r in replies.values())}
        json.dump(meta, open(meta_p, "w"), indent=1)
        print(f"{model:34s} {bench:14s} {len(replies):4d} replies, ${usd:.3f}")
    for k, v in sorted(missing.items()):
        print(f"missing {k[0]} {k[1]}: {v}")


if __name__ == "__main__":
    main()

"""DPO pairs from the SFT model's own answers to the chat prompts. Runs anywhere, light.

    python data/jobs/chat_pairs.py <chat-check output dir> <chat_sft.jsonl> <out.jsonl> [--model sft]

Inputs: samples-<model>.jsonl (eval/chat_gen.py on chat_prompts.jsonl: greedy plus samples) and
checked.jsonl (data/jobs/chat_check.py) from the same directory, plus the SFT rows for teacher answers.

For each prompt with a real check (program or asserts):
  chosen    a sample that passed and has the house-style docstrings; if none did, the teacher's answer
  rejected  a sample that failed (wrong output, crash, timeout), or one without docstrings
Up to two pairs per prompt, each with a different rejected answer. Prompts the model always gets
right teach nothing and are skipped.

Rows: {"id", "prompt_text" (rendered like serve/modal_app.py), "chosen", "rejected", "why"}, for train/dpo_t4.py.
"""

import argparse
import ast
import json
import os
import random


def documented(code):
    """The house style: a module docstring, and a docstring on every function and class."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return ast.get_docstring(tree) is not None and all(ast.get_docstring(n) is not None for n in defs)


def render(turns):
    text = ""
    for prompt, answer in turns[:-1]:
        text += f"<|user|>\n{prompt}\n<|python|>\n{answer.strip(chr(10))}\n<|end|>\n"
    return text + f"<|user|>\n{turns[-1][0]}\n<|python|>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("sft")
    ap.add_argument("out")
    ap.add_argument("--model", default="sft")
    ap.add_argument("--per_prompt", type=int, default=2)
    a = ap.parse_args()
    name = f"samples-{a.model}"
    res = {}
    for line in open(os.path.join(a.src, "checked.jsonl")):
        r = json.loads(line)
        if r["file"] == name:
            res[(r["id"], r["i"])] = r
    teacher = {r["id"]: r["turns"][-1][1] for r in map(json.loads, open(a.sft))}
    rng = random.Random(0)
    pairs, stats = [], {"prompts": 0, "all_pass": 0, "no_signal": 0, "teacher_chosen": 0, "sample_chosen": 0}
    for line in open(os.path.join(a.src, f"{name}.jsonl")):
        r = json.loads(line)
        if (r.get("check") or {}).get("kind") not in ("program", "asserts"):
            continue
        stats["prompts"] += 1
        good, bad = [], []
        for i, ans in enumerate(r["answers"]):
            c = res.get((r["id"], i))
            if c is None or not ans.strip():
                continue
            if c["ok"] and documented(ans) and r["finish"][i] == "stop":
                good.append(ans)
            elif not c["ok"] or r["finish"][i] == "length":
                bad.append((ans, c["why"][:80] if not c["ok"] else "too long"))
            else:
                bad.append((ans, "no docstring"))
        if not bad:
            stats["all_pass"] += 1
            continue
        if not good and r["id"] in teacher:
            good = [teacher[r["id"]]]
            stats["teacher_chosen"] += 1
        elif good:
            stats["sample_chosen"] += 1
        if not good:
            stats["no_signal"] += 1
            continue
        uniq = list({ans: why for ans, why in bad}.items())
        rng.shuffle(uniq)
        for ans, why in uniq[:a.per_prompt]:
            pairs.append({"id": r["id"], "prompt_text": render(r["turns"]), "chosen": rng.choice(good), "rejected": ans,
                          "why": why})
    rng.shuffle(pairs)
    with open(a.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    stats["pairs"] = len(pairs)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

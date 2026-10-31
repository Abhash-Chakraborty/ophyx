"""Build round three's SFT set, DPO prompts and chat eval from the chat-check output. Runs anywhere, light.

    python data/jobs/chat_build.py <chat dir with *.jsonl and checked.jsonl> <out dir>

Reads the files data/synth/chat_data.py and data/jobs/chat_check.py wrote, and keeps:
  - teacher answers, follow-ups and complaint fixes whose check passed (unchecked ones too, when the
    code only has to parse), and only if the answer has the docstrings the house style asks for (module, functions, classes)
  - nothing whose request shares a 10-gram (in words) with a task in eval/ (hands-on, real-world,
    concept suite), so the scores stay clean

Splits by base request, so a request and its follow-ups land on the same side:
  eval   200 requests (with checks) and their follow-ups: the held-out chat eval
  train  everything else: SFT rows, and the prompts DPO samples from later

Writes:
  chat_sft.jsonl     {"id", "source", "turns": [[prompt, answer, train], ...]}
  chat_prompts.jsonl {"id", "turns": [[prompt, answer-or-null], ...], "check"}: the train prompts with checks
  chat_eval.jsonl    same shape, the held-out side, plus the requests from Abhash's own chat log
  stats.json
"""

import ast
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_FILES = ["eval/manual/manual.jsonl", "eval/realworld/realworld.jsonl", "eval/concept_suite/tasks.jsonl"]
N_EVAL = 200
# the requests from the chat that started round three; always in the eval, never trained on
LOG_REQUESTS = [
    ("write an python code to print my name Saksham in capital with color red in terminal.",
     {"kind": "program", "stdin": "", "expect": ["SAKSHAM", "\u001b[31m"]}),
    ("write an python code to print my name Saksham in capital with color red in terminal. it should display in red "
     "color dont print red", {"kind": "program", "stdin": "", "expect": ["SAKSHAM", "\u001b[31m"]}),
]


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(text, n=10):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def eval_grams():
    out = set()
    for f in EVAL_FILES:
        for line in open(os.path.join(ROOT, f)):
            out |= grams(json.loads(line).get("prompt", ""))
    return out


def documented(code):
    """The house style: a module docstring, and a docstring on every function and class."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return ast.get_docstring(tree) is not None and all(ast.get_docstring(n) is not None for n in defs)


def read(d, name):
    p = os.path.join(d, name)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def main(src, out):
    os.makedirs(out, exist_ok=True)
    answers, followups, complaints = (read(src, n) for n in ("answers.jsonl", "followups.jsonl", "complaints.jsonl"))
    ok = {(r["file"], r["id"]): r["ok"] for r in read(src, "checked.jsonl") if r["i"] == 0}
    banned = eval_grams()
    stats = {"answers": len(answers), "followups": len(followups), "complaints": len(complaints)}

    def clean(text):
        return not (grams(text) & banned)

    good = {}
    for r in answers:
        if not clean(r["request"]):
            stats["dropped_eval_overlap"] = stats.get("dropped_eval_overlap", 0) + 1
        elif ok.get(("answers", r["id"])) and documented(r["answer"]):
            good[r["id"]] = r
    check_of = {r["id"]: r.get("check") for r in answers}
    conv_ok = lambda f, c: ok.get((f, c["id"])) and documented(c["turns"][-1][1]) and clean(c["turns"][-1][0])
    fol = [c for c in followups if c["base"] in good and conv_ok("followups", c)]
    com = [c for c in complaints if conv_ok("complaints", c)]  # the base may be a request the teacher failed

    rng = random.Random(0)
    checked = sorted(i for i, r in good.items() if (r.get("check") or {}).get("kind") in ("program", "asserts"))
    rng.shuffle(checked)
    held = set(checked[:N_EVAL])
    stats.update(kept_answers=len(good), kept_followups=len(fol), kept_complaints=len(com), eval_requests=len(held))

    sft, prompts, ev = [], [], []
    for i, r in good.items():
        row = {"id": i, "turns": [[r["request"], None]], "check": r.get("check")}
        if i in held:
            ev.append({**row, "source": "request"})
        else:
            sft.append({"id": i, "source": "answer", "turns": [[r["request"], r["answer"], True]]})
            prompts.append(row)
    for c in fol:
        row = {"id": c["id"], "turns": [c["turns"][0], [c["turns"][1][0], None]], "check": c.get("check")}
        if c["base"] in held:
            ev.append({**row, "source": "followup"})
        else:
            sft.append({"id": c["id"], "source": "followup", "turns": [c["turns"][0] + [True], c["turns"][1] + [True]]})
            prompts.append(row)
    for c in com:
        # the first answer is the old model's mistake: in the context, not in the loss
        row = {"id": c["id"], "turns": [c["turns"][0], [c["turns"][1][0], None]], "check": check_of.get(c["base"])}
        if c["base"] in held:
            ev.append({**row, "source": "complaint"})
        else:
            sft.append({"id": c["id"], "source": "complaint", "turns": [c["turns"][0] + [False], c["turns"][1] + [True]]})
            prompts.append(row)
    for k, (req, check) in enumerate(LOG_REQUESTS):
        ev.append({"id": f"log{k}", "source": "log", "turns": [[req, None]], "check": check})
    rng.shuffle(sft)
    for name, rows in (("chat_sft.jsonl", sft), ("chat_prompts.jsonl", prompts), ("chat_eval.jsonl", ev)):
        with open(os.path.join(out, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    stats.update(sft=len(sft), prompts=len(prompts), eval=len(ev),
                 sft_by_source={s: sum(r["source"] == s for r in sft) for s in ("answer", "followup", "complaint")})
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

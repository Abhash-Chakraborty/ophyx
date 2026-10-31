"""Build round three's SFT set, DPO prompts and chat eval from the chat-check output. Runs anywhere, light.

    python data/jobs/chat_build.py <chat dir with *.jsonl and checked.jsonl> <out dir>

Reads the files data/synth/chat_data.py and data/jobs/chat_check.py wrote (docs.jsonl, the docs phase's
docstring-only rewrites, replaces the answers it covers), and keeps:
  - teacher answers, follow-ups and complaint fixes whose check passed (unchecked ones too, when the
    code only has to parse), and only if the answer has the docstrings the house style asks for (module, functions, classes).
    A program that ran cleanly but missed an expected string is kept too: the teacher wrote the check
    without seeing the answer, and in a sample of these misses most were the check's fault (a wrong
    average, a placeholder like "[content of file1.txt]", a regex where literal text belonged)
  - nothing whose request shares a 10-gram (in words) with a task in eval/ (hands-on, real-world,
    concept suite), so the scores stay clean

Splits by base request, so a request and its follow-ups land on the same side:
  eval   200 requests whose teacher answer passed its check outright, and their follow-ups: the held-out chat eval
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


def script_name(request, code):
    """A file name for a Usage line: one the request names, else the first function or class, snake_cased."""
    m = re.search(r"\b(\w+)\.py\b", request)
    if m:
        return m.group(1)
    try:
        defs = [n.name for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    except (SyntaxError, ValueError, RecursionError):
        defs = []
    defs = [d for d in defs if d != "main"]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", defs[0]).lower().strip("_") if defs else "main"


PLACEHOLDERS = ["todo", "script_name", "your_script", "your_script_name", "script"]


def fix_docs(code, request):
    """Give Usage lines a real file name. The docs prompt's example, todo.py, leaked into a sixth of the
    rewrites, and the teacher often wrote script.py or script_name.py on its own."""
    names = [n for n in PLACEHOLDERS if f"{n}.py" not in request.lower() and re.search(rf"\b{n}(?=\.py\b| import)", code)]
    if not names:
        return code
    name = script_name(request, code)
    return re.sub(rf"\b(?:{'|'.join(names)})(?=\.py\b| import)", name, code)


def read(d, name):
    p = os.path.join(d, name)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def main(src, out):
    os.makedirs(out, exist_ok=True)
    answers, followups, complaints = (read(src, n) for n in ("answers.jsonl", "followups.jsonl", "complaints.jsonl"))
    docs = {(r["file"], r["id"]): r["answer"] for r in read(src, "docs.jsonl")}
    for r in answers:
        r["answer"] = fix_docs(docs.get(("answers", r["id"]), r["answer"]), r["request"])
    for name, convs in (("followups", followups), ("complaints", complaints)):
        for c in convs:
            c["turns"][-1][1] = fix_docs(docs.get((name, c["id"]), c["turns"][-1][1]), c["turns"][0][0])
    for c in followups:  # the first turn is the base request's answer, trained on too
        c["turns"][0][1] = fix_docs(docs.get(("answers", c["base"]), c["turns"][0][1]), c["turns"][0][0])
    why = {(r["file"], r["id"]): r["why"] if r["ok"] else "ran" if r["why"].startswith("missing") else None
           for r in read(src, "checked.jsonl") if r["i"] == 0}
    ok = {k: w is not None for k, w in why.items()}
    banned = eval_grams()
    stats = {"answers": len(answers), "followups": len(followups), "complaints": len(complaints), "docs_fixed": len(docs)}

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
    checked = sorted(i for i, r in good.items() if why.get(("answers", i)) == "ok")
    rng.shuffle(checked)
    held = set(checked[:N_EVAL])
    stats.update(kept_answers=len(good), kept_ran_only=sum(why.get(("answers", i)) == "ran" for i in good),
                 kept_followups=len(fol), kept_complaints=len(com), eval_requests=len(held))

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

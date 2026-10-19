"""Short verified exercises from a Tinker-hosted teacher (paid, hard dollar cap). Runs on the orchestration box.

Each sample is one small module: a docstring stating the task the way a user would ask it, a solution of one
to three functions (or one small class), and 4-8 pytest tests. Short on purpose: it matches HumanEval/MBPP-style
work and fits the 2,048-token SFT window. Generated code is never executed here; batches go to the private
Kaggle dataset `pycoder-synth-tinker`, where data/synth/verify.py runs them offline with
{"keep_failures": true}.

  python data/synth/tinker_short.py gen --n 2000 --budget 1.0        # stops at $1.00 total spend
  python data/synth/tinker_short.py repair --failed DIR --budget 10  # fix failures from a verify run

Repair: the teacher sees its module and the pytest output and returns the whole fixed module. A repaired
sample that then passes also gives a self-repair SFT example (broken code + error -> fixed code).
Spend is computed from token counts and the price table below and kept in OUT/state.json.
"""

import argparse
import ast
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_g = {"__name__": "lib", "__file__": os.path.join(ROOT, "data", "synth", "generate.py"), "CONFIG": {"library": True}}
exec(compile(open(_g["__file__"]).read(), "generate.py", "exec"), _g)
LEAVES, DOMAINS, FENCE, load_env, upload = (_g[k] for k in ("LEAVES", "DOMAINS", "FENCE", "load_env", "upload"))

PROMPT_VERSION = 1
MODEL = "Qwen/Qwen3.6-35B-A3B"
PRICES = {"Qwen/Qwen3.6-35B-A3B": (0.45, 1.335), "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (0.184, 0.495)}  # $/M in, out

SYSTEM = (
    "You write short, self-contained Python 3.12 exercises with their solutions and tests. Output exactly one Python "
    "module and nothing else: no Markdown, no code fences, no prose outside the code. Structure:\n"
    "1. A module docstring that states the task the way a user would ask for it: what to write, the function or class "
    "names and signatures, the expected inputs and outputs, and how edge cases and invalid input must behave. Do not "
    "describe the solution.\n"
    "2. The solution: one to three functions, or one small class, with type hints and short docstrings. Standard "
    "library only. Correct, idiomatic, PEP 8.\n"
    "3. Four to eight pytest test functions named test_*, at the top level of the module (not in a class, not under "
    "__main__), covering normal cases, edge cases and error cases. Work out every expected value by hand.\n"
    "Keep the whole module between 30 and 90 lines. No input(), no network, no files outside tmp_path, no sleeping."
)
FLAVOURS = ["a single pure function", "two cooperating functions", "one small class with a few methods",
            "a parsing or validation function", "a function over lists or dicts", "a string-processing function",
            "a numeric or math function", "a function using a standard-library module well"]
DIFFICULTY = {1: "easy", 2: "medium", 3: "medium-hard", 4: "hard"}


def spec(i, seed=0):
    r = random.Random(f"tinker-short-{seed}-{i}")
    area, leaf = LEAVES[i % len(LEAVES)] if i < 4 * len(LEAVES) else r.choice(LEAVES)
    return {"i": i, "area": area, "leaf": leaf, "kind": "exercise", "flavour": r.choice(FLAVOURS),
            "difficulty": r.choice([1, 2, 2, 3, 3, 4]), "domain": r.choice(DOMAINS), "teacher": MODEL,
            "temperature": r.choice([0.6, 0.8, 1.0])}


def prompt(s):
    return (f"Topic: {s['leaf']} (area: {s['area'].replace('_', ' ')}). Write {s['flavour']}. "
            f"Difficulty: {DIFFICULTY[s['difficulty']]}. If it fits, use an example from {s['domain']}; otherwise ignore "
            f"the domain. Make it a concrete task, not a generic textbook example. Output one module in this order: the task "
            f"docstring, the solution, then 4-8 top-level pytest functions named test_*. The module is not complete "
            f"without the tests.")


def repair_prompt(r):
    return ("This module's tests fail. Here is the module:\n\n" + r["code"] + "\n\nHere is the pytest output:\n\n"
            + r["tail"][-1500:] + "\n\nFind the bug and return the whole corrected module. Fix the solution if it is "
            "wrong; fix a test only if its expected value is wrong. Keep the same task, names and structure.")


def extract_code(text):
    """All fenced blocks joined (the teacher sometimes splits solution and tests); else the bare text."""
    blocks = FENCE.findall(text or "")
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks) + "\n"
    return _g["extract_code"](text)


def shape_ok(code):
    """>= 4 top-level tests, a task docstring and at least one non-test definition."""
    t = ast.parse(code)
    tests = sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test") for x in t.body)
    defs = sum(isinstance(x, (ast.FunctionDef, ast.ClassDef)) and not x.name.startswith("test") for x in t.body)
    return tests >= 4 and defs >= 1 and bool(ast.get_docstring(t)) and code.count("\n") <= 200


class Teacher:
    def __init__(self, model):
        import tinker
        os.environ["TINKER_API_KEY"] = load_env()["TINKER_API_KEY"]
        self.tinker, self.model = tinker, model
        self.client = tinker.ServiceClient().create_sampling_client(base_model=model)
        self.tok = self.client.get_tokenizer()

    def ask(self, user, temperature, max_tokens=2500):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        ids = self.tok.encode(self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                           enable_thinking=False), add_special_tokens=False)
        params = self.tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95,
                                            stop=[self.tok.eos_token] if self.tok.eos_token else None)
        for attempt in range(4):
            try:
                seq = self.client.sample(self.tinker.ModelInput.from_ints(ids), 1, params).result().sequences[0]
                break
            except Exception as e:  # transient API errors: back off, then give up on this sample
                print(f"  sample error ({type(e).__name__}: {str(e)[:120]}); retry {attempt + 1}", flush=True)
                time.sleep(10 * 2 ** attempt)
        else:
            return None, len(ids), 0, "error"
        text = re.sub(r"<think>.*?</think>", "", self.tok.decode(seq.tokens, skip_special_tokens=True), flags=re.S)
        return text, len(ids), len(seq.tokens), str(seq.stop_reason)

    def cost(self, n_in, n_out):
        p_in, p_out = PRICES[self.model]
        return n_in / 1e6 * p_in + n_out / 1e6 * p_out


REJECTS = []  # for prompt tuning; written next to the batches
TESTS_ONLY = ("Write the pytest tests for this module: 4 to 8 top-level functions named test_*, covering normal cases, "
              "edge cases and error cases, with every expected value worked out by hand. Output only the test functions "
              "and any imports they need, as Python code. Do not repeat the module.\n\n")


def n_tests(code):
    return sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test") for x in ast.parse(code).body)


def one(teacher, job):
    """One sample -> (record or None, reason, cost). A module with a task and a solution but no tests gets a second,
    cheaper call that writes only the tests (the teacher skips them about a fifth of the time)."""
    base, user, temperature = job
    text, n_in, n_out, stop = teacher.ask(user, temperature)
    cost, usage = teacher.cost(n_in, n_out), {"prompt_tokens": n_in, "completion_tokens": n_out}
    if text is None:
        return None, "error", cost
    if "length" in stop.lower():
        return None, "truncated", cost
    code = extract_code(text)
    try:
        if not shape_ok(code) and n_tests(code) == 0 and ast.get_docstring(ast.parse(code)) and code.count("\n") <= 160:
            t2, i2, o2, stop2 = teacher.ask(TESTS_ONLY + code, 0.6, max_tokens=1500)
            cost += teacher.cost(i2, o2)
            usage = {"prompt_tokens": n_in + i2, "completion_tokens": n_out + o2, "tests_added": True}
            if t2 is not None and "length" not in stop2.lower():
                code = code.rstrip("\n") + "\n\n\n" + extract_code(t2)
        ok = shape_ok(code)
    except (SyntaxError, ValueError, RecursionError):
        return None, "no_parse", cost
    if not ok:
        REJECTS.append({"id": base["id"], "why": "bad_shape", "code": code[:4000]})
        return None, "bad_shape", cost
    return {**base, "code": code, "usage": usage}, "ok", cost


def run_batch(teacher, jobs, workers):
    """jobs: [(record_base, user_prompt, temperature)] -> (records, stats, cost)"""
    recs, stats, cost = [], {"ok": 0, "no_parse": 0, "truncated": 0, "bad_shape": 0, "error": 0, "tests_added": 0}, 0.0
    with cf.ThreadPoolExecutor(workers) as ex:
        for rec, why, c in ex.map(lambda j: one(teacher, j), jobs):
            cost += c
            stats[why] += 1
            if rec:
                recs.append(rec)
                stats["tests_added"] += bool(rec["usage"].get("tests_added"))
    return recs, stats, cost


def write_batch(out, name, recs):
    path = os.path.join(out, f"batch-{name}.jsonl")
    with open(path + ".tmp", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "repair"])
    ap.add_argument("--n", type=int, default=0, help="gen: generate specs with index < n")
    ap.add_argument("--failed", default=None, help="repair: directory with *.failed.jsonl from verify.py")
    ap.add_argument("--budget", type=float, required=True, help="stop once total spend (all modes) reaches this many USD")
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/pycoder-synth-tinker/raw"))
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dataset", default="abhashchakraborty/pycoder-synth-tinker")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    state_p = os.path.join(a.out, "state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"next": 0, "spent": 0.0, "repaired": []}
    teacher = Teacher(MODEL)
    if a.mode == "gen":
        work = [(start, min(a.n, start + a.batch)) for start in range(state["next"], a.n, a.batch)]
        jobs_for = lambda se: [({"id": hashlib.sha1(f"ts{PROMPT_VERSION}-{i}".encode()).hexdigest()[:16], **spec(i),
                                  "prompt_version": PROMPT_VERSION}, prompt(spec(i)), spec(i)["temperature"]) for i in range(*se)]
        name_for = lambda se: f"t{se[0]:07d}"
    else:
        failed = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.failed, "*.failed.jsonl"))) for l in open(p) if l.strip()]
        failed = [r for r in failed if r["id"] not in set(state["repaired"]) and r.get("why") == "tests_failed" and not r.get("repair_of")]  # one repair round per sample
        work = [(k, min(len(failed), k + a.batch)) for k in range(0, len(failed), a.batch)]
        jobs_for = lambda se: [({k: v for k, v in r.items() if k not in ("code", "tail", "why", "usage")} | {
            "id": r["id"] + "-r1", "repair_of": r["id"], "broken_code": r["code"], "error": r["tail"][-1500:]},
            repair_prompt(r), 0.4) for r in failed[se[0]:se[1]]]
        name_for = lambda se: f"r{int(time.time())}-{se[0]:05d}"
    print(f"{a.mode}: {len(work)} batches; spent so far ${state['spent']:.3f} of ${a.budget:.2f}", flush=True)
    for se in work:
        if state["spent"] >= a.budget:
            print(f"budget reached (${state['spent']:.3f}); stopping", flush=True)
            break
        jobs = jobs_for(se)
        t0 = time.time()
        recs, stats, cost = run_batch(teacher, jobs, a.workers)
        write_batch(a.out, name_for(se), recs)
        with open(os.path.join(a.out, "rejects.jsonl"), "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in REJECTS)
        REJECTS.clear()
        state["spent"] += cost
        if a.mode == "gen":
            state["next"] = se[1]
        else:
            state["repaired"] += [j[0]["repair_of"] for j in jobs]
        json.dump(state, open(state_p, "w"))
        print(f"[{time.strftime('%H:%M:%S')}] {name_for(se)}: {stats}, ${cost:.3f} in {time.time() - t0:.0f}s; "
              f"total ${state['spent']:.3f}", flush=True)
    if not a.no_upload:
        upload(a.out, load_env(), a.dataset)


if __name__ == "__main__":
    sys.exit(main())

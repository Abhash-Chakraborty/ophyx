"""Synthetic data generator: curriculum leaf x kind x difficulty x domain x style -> one Python module.

Runs on the orchestration box. It is network-bound (one HTTP call at a time per worker, a few per
second in total) and needs no GPU. Raw generations are written as gzipped JSONL batches and pushed
to the private Kaggle dataset `ophyx-synth-raw`, where a keyless, offline Kaggle session executes
them (data/synth/verify.py). Generated code is never executed here.

  python data/synth/generate.py --n 2000 --out ~/.cache/ophyx-synth            # pilot
  python data/synth/generate.py --n 200000 --out ~/.cache/ophyx-synth --upload-every 10

Resumable: the next index is stored in OUT/state.json, and sample i always gets the same spec.
"""

import argparse
import ast
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if "__file__" in globals() else os.getcwd()
PROMPT_VERSION = 3  # v2: length limit. v3: tests mandatory and checked; Groq teachers (Cerebras free quota ran out)

# rpm keeps each teacher under its free-tier limits (Groq: 8k tokens/min and 1,000 requests/day per model).
TEACHERS = {
    "gpt-oss-120b@groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "key": "GROQ_TOKEN",
                          "model": "openai/gpt-oss-120b", "extra": {"reasoning_effort": "low"}, "rpm": 0.65, "max_tokens": 6000},
    "qwen3.8-27b@groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "key": "GROQ_TOKEN",
                         "model": "qwen/qwen3.8-27b", "extra": {"reasoning_effort": "none"}, "rpm": 0.65, "max_tokens": 6000},
    "qwen3.8-27b@cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions", "key": "CEREBRAS_TOKEN",
                             "model": "qwen-3.8-27b", "extra": {"disable_reasoning": True}, "rpm": 35},
}
ACTIVE_TEACHERS = ["gpt-oss-120b@groq", "qwen3.8-27b@groq"]

SYSTEM = (
    "You write Python source files for a high-quality programming textbook and test suite. Output exactly one complete "
    "Python 3.12 module and nothing else: no Markdown, no code fences, no prose outside the code. Every explanation goes "
    "in docstrings and comments. The code must be correct, idiomatic, PEP 8 formatted, fully type-hinted, and runnable "
    "with only the Python standard library (plus pytest for tests). No network access, no reading or writing files "
    "outside a temporary directory, no input(), no sleeping longer than 0.1 seconds. Keep the whole module between 120 "
    "and 300 lines: complete and focused beats long. Never refuse; if the topic is broad, pick one concrete, useful angle."
)

TESTS_RULE = (" The tests are required: put them at the end of the module as top-level functions named test_* "
              "(not inside a class or under __main__), and work out every expected value by hand before writing it.")

KINDS = {
    "textbook": (0.30, (
        "Kind: textbook module. Teach the topic the way a senior engineer would: what it is, why it matters, how it works "
        "underneath, trade-offs, and common mistakes, all in the module docstring and in docstrings and comments. Include "
        "several worked examples as functions or classes. End with an `if __name__ == \"__main__\":` block that runs every "
        "example in under two seconds, prints short results, and uses assert statements to check each claim it makes.")),
    "exercise": (0.40, (
        "Kind: exercise. The module docstring states the task as a user would ask for it. Then the solution, with type "
        "hints and docstrings. Then at least six pytest test functions (named test_*) covering normal cases, edge cases "
        "and error cases. Tests must be deterministic and fast; use tmp_path or monkeypatch where needed." + TESTS_RULE)),
    "design": (0.20, (
        "Kind: design-as-code. The module docstring gives the requirements, constraints, the design chosen, trade-offs and "
        "the alternatives considered. Then a clean implementation with small classes and clear interfaces. Then at least "
        "six pytest test functions (named test_*) that check the behaviour and the invariants." + TESTS_RULE)),
    "errors": (0.10, (
        "Kind: error-handling scenario. Write code that handles a specific failure mode realistically (for example "
        "timeouts, partial writes, malformed input, transient failures with retries, resource cleanup). Explain the "
        "failure modes in the docstrings. Then at least six pytest test functions (named test_*) that trigger each "
        "failure path using fakes, mocks or monkeypatch, never real network or long sleeps." + TESTS_RULE)),
}

DOMAINS = ["finance", "games", "devops", "science", "web backend", "data engineering", "education", "e-commerce",
           "logistics", "IoT sensors", "music", "text processing", "security tooling", "robotics simulation",
           "command-line tools", "social network", "travel booking", "energy grid", "sports statistics",
           "healthcare scheduling (synthetic data only)", "library management", "chat applications", "maps and geometry"]
STYLES = ["object-oriented with small focused classes", "functional style with pure functions", "dataclass-centric",
          "protocol-oriented with typing.Protocol", "minimal and direct", "written for a beginner, extra explanation",
          "concise, written for an expert", "generator-based streaming", "layered: pure core with a thin I/O shell"]
DIFFICULTY = {1: "beginner", 2: "easy", 3: "intermediate", 4: "advanced", 5: "expert, production-grade"}


def leaves():
    cfg = globals().get("CONFIG") or {}
    cur = cfg.get("curriculum") or json.load(open(os.path.join(ROOT, "data", "curriculum.json")))
    return [(area, leaf) for area, ls in cur["areas"].items() for leaf in ls]


LEAVES = leaves()


def spec(i, seed=0):
    r = random.Random(f"{seed}-{i}")
    kinds = list(KINDS)
    kind = r.choices(kinds, weights=[KINDS[k][0] for k in kinds])[0]
    area, leaf = LEAVES[i % len(LEAVES)] if i < 10 * len(LEAVES) else r.choice(LEAVES)  # even coverage first
    teacher = ACTIVE_TEACHERS[i % len(ACTIVE_TEACHERS)]
    return {"i": i, "area": area, "leaf": leaf, "kind": kind, "difficulty": r.choice([1, 2, 3, 3, 4, 4, 5]),
            "domain": r.choice(DOMAINS), "style": r.choice(STYLES), "teacher": teacher, "temperature": r.choice([0.6, 0.8, 1.0])}


def prompt(s):
    return (f"Topic: {s['leaf']} (area: {s['area'].replace('_', ' ')}).\n{KINDS[s['kind']][1]}\n"
            f"Difficulty: {s['difficulty']}/5 ({DIFFICULTY[s['difficulty']]}). Application domain for examples: {s['domain']}. "
            f"Style: {s['style']}. Make it substantially different from a generic textbook example.")


FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(text):
    """Largest fenced block if any; otherwise the text with a stray opening/closing fence removed."""
    text = text or ""
    blocks = FENCE.findall(text)
    if blocks:
        code = max(blocks, key=len)
    else:
        lines = text.strip("\n").split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip("\n") + "\n"


def load_env():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v.strip().strip("'\"")
    return env


class Pacer:
    """At most `rpm` request starts per minute across threads, plus a pause after a 429."""

    def __init__(self, rpm):
        self.gap, self.next, self.lock = 60.0 / rpm, time.time(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))

    def backoff(self, seconds):
        with self.lock:  # capped: a stray multi-hour retry-after once stalled the whole service
            self.next = max(self.next, time.time() + min(seconds, 900))


def shape_ok(code, kind):
    """Cheap structural check before upload: test kinds need >= 4 top-level tests, textbooks a __main__ block."""
    t = ast.parse(code)
    if kind == "textbook":
        return any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in t.body)
    n = sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test") for x in t.body)
    return n >= 4


def generate_one(s, env, pacers):
    pacer = pacers[s["teacher"]]
    t = TEACHERS[s["teacher"]]
    body = {"model": t["model"], "temperature": s["temperature"], "max_tokens": t.get("max_tokens", 8000),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(s)}], **t["extra"]}
    for attempt in range(6):
        pacer.wait()
        req = urllib.request.Request(t["url"], data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {env[t['key']]}", "Content-Type": "application/json", "User-Agent": "ophyx-synth/1"})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=300) as r:
                j = json.load(r)
            content = j["choices"][0]["message"].get("content") or ""
            print(f"  req {s['i']} {s['teacher']}: {time.time() - t0:.0f}s, {j.get('usage', {}).get('total_tokens')} tok", flush=True)
            return content, j.get("usage", {}), j["choices"][0].get("finish_reason")
        except urllib.error.HTTPError as e:
            wait = float(e.headers.get("retry-after") or 0) or min(300, 10 * 2 ** attempt)
            print(f"  req {s['i']} {s['teacher']}: HTTP {e.code}, retry-after {e.headers.get('retry-after')}, "
                  f"waiting {min(wait, 900):.0f}s", flush=True)
            if e.code == 429:
                pacer.backoff(wait)
            elif e.code < 500:
                return None, {"error": f"HTTP {e.code}: {e.read()[:200]!r}"}, None
            time.sleep(wait)
        except Exception:  # network hiccup
            time.sleep(min(120, 5 * 2 ** attempt))
    return None, {"error": "gave up"}, None


def upload(out_dir, env, dataset):
    """Push all batch files as a new version of the private Kaggle dataset (only when --dataset is given)."""
    if not dataset:
        return
    up = os.path.join(out_dir, "upload")
    os.makedirs(up, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith("batch-") and f.endswith((".jsonl", ".jsonl.gz")) and not os.path.exists(os.path.join(up, f)):
            os.link(os.path.join(out_dir, f), os.path.join(up, f))
    meta = {"title": dataset.split("/")[-1], "id": dataset, "licenses": [{"name": "other"}]}
    json.dump(meta, open(os.path.join(up, "dataset-metadata.json"), "w"))
    kaggle = shutil.which("kaggle") or os.path.expanduser("~/.local/bin/kaggle")
    e = {**os.environ, "KAGGLE_API_TOKEN": env["KAGGLE_API_TOKEN"]}
    exists = subprocess.run([kaggle, "datasets", "status", meta["id"]], env=e, capture_output=True, text=True)
    if "ready" in (exists.stdout + exists.stderr).lower():
        cmd = [kaggle, "datasets", "version", "-p", up, "-m", f"{len(os.listdir(up)) - 1} batches", "-q"]
    else:
        cmd = [kaggle, "datasets", "create", "-p", up, "-q"]  # private by default
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    print("upload:", (r.stdout + r.stderr).strip().splitlines()[-1:] or "ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="generate samples with index < n")
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/ophyx-synth"))
    ap.add_argument("--rpm", type=float, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--daily-token-cap", type=float, default=150e6)
    ap.add_argument("--upload-every", type=int, default=0, help="upload after every k batches (0 = only at the end)")
    ap.add_argument("--dataset", default=None, help="Kaggle dataset to push batches to, e.g. abhashchakraborty/ophyx-synth-raw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = load_env()
    state_p = os.path.join(a.out, "state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"next": 0, "day": time.strftime("%Y-%m-%d"), "day_tokens": 0}
    pacers = {t: Pacer(TEACHERS[t]["rpm"]) for t in TEACHERS}
    batches_since_upload = 0
    print(f"{len(LEAVES)} leaves; starting at {state['next']} of {a.n}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        while state["next"] < a.n:
            today = time.strftime("%Y-%m-%d")
            if state["day"] != today:
                state["day"], state["day_tokens"] = today, 0
            if state["day_tokens"] >= a.daily_token_cap:
                print("daily token cap reached; sleeping 30 min", flush=True)
                time.sleep(1800)
                continue
            start, end = state["next"], min(a.n, state["next"] + a.batch)
            specs = [spec(i) for i in range(start, end)]
            t0 = time.time()
            results = list(ex.map(lambda s: (s, *generate_one(s, env, pacers)), specs))
            recs, rejects, stats = [], [], {"ok": 0, "no_parse": 0, "error": 0, "truncated": 0, "bad_shape": 0}
            tokens = 0
            for s, content, usage, finish in results:
                tokens += int((usage or {}).get("total_tokens") or 0)
                if content is None:
                    stats["error"] += 1
                    rejects.append({"i": s["i"], "kind": s["kind"], "error": str((usage or {}).get("error"))[:300]})
                    continue
                code = extract_code(content)
                try:
                    ast.parse(code)
                except (SyntaxError, ValueError, RecursionError) as e:
                    stats["no_parse"] += 1
                    rejects.append({"i": s["i"], "kind": s["kind"], "error": str(e)[:200], "head": content[:400], "tail": content[-400:]})
                    continue
                if finish == "length":
                    stats["truncated"] += 1
                    continue
                if not shape_ok(code, s["kind"]):
                    stats["bad_shape"] += 1
                    continue
                sid = hashlib.sha1(f"{PROMPT_VERSION}-{s['i']}".encode()).hexdigest()[:16]
                recs.append({"id": sid, **s, "prompt_version": PROMPT_VERSION, "code": code,
                             "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}})
                stats["ok"] += 1
            path = os.path.join(a.out, f"batch-{start:07d}.jsonl")  # plain: Kaggle rewrites .gz uploads
            with open(path + ".tmp", "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            os.replace(path + ".tmp", path)
            with open(os.path.join(a.out, "rejects.jsonl"), "a") as f:  # small; for prompt tuning
                for r in rejects:
                    f.write(json.dumps(r) + "\n")
            state["next"] = end
            state["day_tokens"] += tokens
            json.dump(state, open(state_p, "w"))
            dt = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] batch {start}-{end}: {stats}, {tokens / 1e6:.2f}M tokens in {dt:.0f}s "
                  f"({tokens / dt / 1e3:.1f}k tok/s); today {state['day_tokens'] / 1e6:.1f}M", flush=True)
            batches_since_upload += 1
            if a.upload_every and batches_since_upload >= a.upload_every:
                upload(a.out, env, a.dataset)
                batches_since_upload = 0
    upload(a.out, env, a.dataset)


if __name__ == "__main__" and not (globals().get("CONFIG") or {}).get("library"):
    main()

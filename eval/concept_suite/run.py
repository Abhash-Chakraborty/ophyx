"""Score a model on the private concept suite: greedy pass@1 against each task's hidden pytest tests.

Generates with Hugging Face transformers on a GPU (as eval/evalplus_run.py does), or scores a JSONL
of precomputed completions, or (self-test) scores the reference solutions, which must give 100%.

Prompt: the project chat template `<|user|>\n{prompt}\n<|python|>\n`, generation stopping at
`<|end|>`. A tokenizer without those special tokens (a base model) gets a plain prompt instead: the
task as a module docstring, with the completion cut at the first new top-level block that is not
part of the solution. The Python module is extracted from the output (``` fences are unwrapped),
written to solution.py next to the hidden tests as test_task.py, and pytest runs in a subprocess
with a wall-clock timeout, prlimit CPU/memory/file-size/file-count limits, nice 19, an empty
environment and its own temp dir. A task passes when pytest exits 0.

CONFIG keys (CLI flags with the same names, dashes for underscores, override them):
  model / models   HF id or local path (or a list) to generate with
  completions      JSONL of precomputed outputs: {"task_id", "completion"} is raw model text, cut and
                   extracted like generated text; {"task_id", "solution"} is a finished module
  self_test        score the reference solutions instead (checks the harness and the tasks)
  template         "auto" (default), "chat" or "plain"
  tasks, filter, limit, batch_size, max_new_tokens, jobs, timeout, python, nice, tmp, out

    python eval/concept_suite/run.py --self-test --out /tmp/concept
    OPHYX_CONFIG='{"model": "/kaggle/working/hf-ckpt"}' python eval/concept_suite/run.py
    python eval/concept_suite/run.py --completions samples.jsonl --tag mymodel

Writes <out>/concept-<tag>.jsonl (one record per task) and merges the summary into
<out>/concept_results.json.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
HERE = os.path.dirname(os.path.abspath(__file__))
_SMOKE_TMP = os.path.expanduser("~/.cache/ophyx-smoke/concept-tmp")
DEFAULTS = {
    "model": None, "models": None, "completions": None, "self_test": False, "template": "auto",
    "tasks": os.path.join(HERE, "tasks.jsonl"), "filter": "", "limit": 0, "tag": None,
    "batch_size": 16, "max_new_tokens": 1024, "jobs": 2, "timeout": 30, "python": sys.executable,
    "nice": True, "tmp": _SMOKE_TMP if os.path.isdir(os.path.dirname(_SMOKE_TMP)) else None,
    "out": "/kaggle/working" if os.path.isdir("/kaggle") else os.getcwd(),
}
CHAT_TOKENS = ("<|user|>", "<|python|>", "<|end|>")
CHAT_STOPS = ["<|end|>", "<|user|>", "<|python|>"]
# A base model keeps writing after the module; cut at the first top-level block that starts a demo,
# a test or another task. Function and class definitions are not stops: a module may need several.
PLAIN_STOPS = ["\nif __name__", '\n"""', "\n'''", "\n```", "\n# Test", "\n# Example", "\nassert ", "\nprint("]
FENCE_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)(?:\n```|\Z)", re.S)
PRLIMIT = shutil.which("prlimit")
SKIPPED_RE = re.compile(r"(\d+) skipped")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def settings(argv=None):
    cfg = {**DEFAULTS, **CONFIG}
    if argv is not None:
        ap = argparse.ArgumentParser(description="concept suite pass@1")
        for key, default in DEFAULTS.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(default, bool):
                ap.add_argument(flag, action="store_true", default=None)
                if default:
                    ap.add_argument("--no-" + key.replace("_", "-"), dest=key, action="store_false")
            else:
                ap.add_argument(flag, type=type(default) if isinstance(default, (int, float)) else str)
        known, _ = ap.parse_known_args(argv)  # notebooks pass their own argv; ignore what we don't know
        cfg.update({k: v for k, v in vars(known).items() if v is not None})
    return cfg


# ---------------------------------------------------------------- prompts and extraction


def chat_prompt(task):
    return f"<|user|>\n{task['prompt']}\n<|python|>\n"


def plain_prompt(task):
    return '"""' + task["prompt"].replace('"""', "'''") + '\n"""\n'


def truncate(text, stops):
    cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
    return text if cut < 0 else text[:cut]


def compiles(code):
    try:
        compile(code, "solution.py", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def extract_module(text, stops=CHAT_STOPS):
    """The Python module in a model output: cut at the stop strings, then unwrap ``` fences if any
    (preferring the longest block that compiles)."""
    text = truncate(text, stops)
    if "```" in text:
        blocks = FENCE_RE.findall(text)
        if blocks:
            good = [b for b in blocks if compiles(b)]
            text = max(good or blocks, key=len)
    return text.strip("\n") + "\n"


# ---------------------------------------------------------------- generation


def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def has_chat_tokens(tok):
    vocab = tok.get_vocab()
    return all(t in vocab for t in CHAT_TOKENS)


def generate(model_id, tok, prompts, chat, cfg):
    import torch
    from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
    stops = CHAT_STOPS if chat else PLAIN_STOPS
    eos = [tok.convert_tokens_to_ids("<|end|>")] if chat else []
    eos += [tok.eos_token_id] if tok.eos_token_id is not None else []

    def decode(ids):  # chat mode keeps special tokens so <|end|> is visible to truncate()
        return tok.batch_decode(ids, skip_special_tokens=not chat)

    class StopOnStrings(StoppingCriteria):
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kw):
            tails = decode(input_ids[:, self.start:])
            return torch.tensor([any(s in t for s in stops) for t in tails], device=input_ids.device)

    outs = []
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths batch well
    for b in range(0, len(order), cfg["batch_size"]):
        idx = order[b:b + cfg["batch_size"]]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=not chat).to("cuda")
        start = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos or None,
                                 stopping_criteria=StoppingCriteriaList([StopOnStrings(start)]))
        texts = decode(gen[:, start:])
        if chat:  # drop padding after the end of each sequence
            texts = [truncate(t, [s for s in (tok.pad_token, tok.eos_token) if s]) for t in texts]
        outs.extend(zip(idx, texts))
        log(f"generated {min(b + cfg['batch_size'], len(order))}/{len(order)}")
    del model
    torch.cuda.empty_cache()
    return [t for _, t in sorted(outs)]


# ---------------------------------------------------------------- sandboxed scoring


def run_tests(code, tests, cfg):
    d = tempfile.mkdtemp(prefix="cs-", dir=cfg["tmp"])
    try:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "test_task.py"), "w") as f:
            f.write(tests)
        timeout = cfg["timeout"]
        cmd = [cfg["python"], "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-o", "addopts=",
               "test_task.py"]
        if PRLIMIT:
            cmd = [PRLIMIT, f"--cpu={int(timeout)}", f"--as={2 << 30}", f"--fsize={50 << 20}", "--nofile=256",
                   "--"] + cmd
        if cfg["nice"]:
            cmd = ["nice", "-n", "19"] + cmd
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "TMPDIR": d, "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        t = time.time()
        p = subprocess.Popen(cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return "timeout", timeout, "", 0
        out = out.decode(errors="replace")
        status = "pass" if p.returncode == 0 else "fail"
        skips = int(m.group(1)) if (m := SKIPPED_RE.search(out[-500:])) else 0
        return status, time.time() - t, "" if status == "pass" else out[-1500:], skips
    finally:
        shutil.rmtree(d, ignore_errors=True)


def python_version(python):
    out = subprocess.run([python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                         stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    return tuple(int(x) for x in out.split("."))


def score(tasks, solutions, cfg):
    pyver = python_version(cfg["python"])
    recs = [None] * len(tasks)

    def one(i):
        t, code = tasks[i], solutions[i]
        rec = {"task_id": t["task_id"], "area": t["area"], "difficulty": t["difficulty"], "leaf": t["leaf"],
               "parses": bool(code.strip()) and compiles(code), "solution": code}
        need = tuple(int(x) for x in t.get("min_python", "3.0").split("."))
        if need > pyver:
            rec.update(status="skipped", seconds=0, tail=f"needs Python {t['min_python']}")
        elif not rec["parses"]:
            rec.update(status="syntax_error" if code.strip() else "empty", seconds=0, tail="")
        else:
            status, sec, tail, skips = run_tests(code, t["tests"], cfg)
            rec.update(status=status, seconds=round(sec, 2), tail=tail, pytest_skips=skips)
        recs[i] = rec

    with cf.ThreadPoolExecutor(max(1, cfg["jobs"])) as ex:
        list(ex.map(one, range(len(tasks))))
    return recs, pyver


def summarise(recs):
    scored = [r for r in recs if r["status"] != "skipped"]

    def rate(rs):
        return round(100 * sum(r["status"] == "pass" for r in rs) / max(len(rs), 1), 1)

    groups = {}
    for key in ("area", "difficulty"):
        g = {}
        for r in scored:
            g.setdefault(r[key], []).append(r)
        groups[key] = {str(k): {"n": len(v), "pass@1": rate(v)} for k, v in sorted(g.items())}
    return {"n": len(scored), "skipped": len(recs) - len(scored), "pass@1": rate(scored),
            "passed": sum(r["status"] == "pass" for r in scored),
            "parses": round(100 * sum(r["parses"] for r in scored) / max(len(scored), 1), 1),
            "timeouts": sum(r["status"] == "timeout" for r in scored),
            # pytest exits 0 when tests skip (e.g. no tzdata in the sandbox), so such passes are flagged
            "passes_with_skips": sum(r["status"] == "pass" and r.get("pytest_skips", 0) > 0 for r in scored),
            "by_area": groups["area"], "by_difficulty": groups["difficulty"]}


def report(tag, s, template, pyver):
    lines = [f"concept suite: {tag} (template {template}, python {'.'.join(map(str, pyver))})",
             f"pass@1 {s['pass@1']:5.1f}%  ({s['passed']}/{s['n']})   parses {s['parses']:.1f}%   "
             f"timeouts {s['timeouts']}   passes with skipped tests {s['passes_with_skips']}" + (f"   skipped {s['skipped']} (need newer Python)" if s["skipped"] else "")]
    lines.append(f"  {'area':22} {'n':>4} {'pass@1':>7}")
    lines += [f"  {a:22} {v['n']:4} {v['pass@1']:6.1f}%" for a, v in s["by_area"].items()]
    lines.append(f"  {'difficulty':22} {'n':>4} {'pass@1':>7}")
    lines += [f"  {d:22} {v['n']:4} {v['pass@1']:6.1f}%" for d, v in s["by_difficulty"].items()]
    return "\n".join(lines)


# ---------------------------------------------------------------- main


def load_tasks(cfg):
    tasks = [json.loads(l) for l in open(cfg["tasks"]) if l.strip()]
    tasks = [t for t in tasks if cfg["filter"] in t["task_id"]]
    return tasks[:cfg["limit"]] if cfg["limit"] else tasks


def runs(cfg, tasks):
    """Yield (tag, template, solutions) for each thing to score."""
    if cfg["self_test"]:
        yield cfg["tag"] or "reference", "reference", [t["reference"] for t in tasks]
        return
    if cfg["completions"]:
        rows = {}
        for line in open(cfg["completions"]):
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
        missing = [t["task_id"] for t in tasks if t["task_id"] not in rows]
        if missing:
            log(f"warning: {len(missing)} tasks have no completion and count as failures")
        stops = PLAIN_STOPS if cfg["template"] == "plain" else CHAT_STOPS
        sols = []
        for t in tasks:
            r = rows.get(t["task_id"], {})
            sols.append(r["solution"] if "solution" in r else extract_module(r.get("completion", ""), stops))
        tag = cfg["tag"] or os.path.splitext(os.path.basename(cfg["completions"]))[0]
        yield tag, cfg["template"] if cfg["template"] != "auto" else "completions", sols
        return
    models = cfg["models"] or ([cfg["model"]] if cfg["model"] else [])
    if isinstance(models, str):
        models = [models]
    if not models:
        raise SystemExit("nothing to score: give model, models, completions or self_test")
    for model_id in models:
        tok = load_tokenizer(model_id)
        chat = cfg["template"] == "chat" or (cfg["template"] == "auto" and has_chat_tokens(tok))
        prompts = [chat_prompt(t) if chat else plain_prompt(t) for t in tasks]
        log(f"{model_id}: {len(prompts)} prompts, template {'chat' if chat else 'plain'}")
        comps = generate(model_id, tok, prompts, chat, cfg)
        sols = [extract_module(c, CHAT_STOPS if chat else PLAIN_STOPS) for c in comps]
        tag = cfg["tag"] if cfg["tag"] and len(models) == 1 else model_id.rstrip("/").replace("/", "__")
        yield tag, "chat" if chat else "plain", sols


def main(argv=None):
    cfg = settings(argv)
    os.makedirs(cfg["out"], exist_ok=True)
    if cfg["tmp"]:
        os.makedirs(cfg["tmp"], exist_ok=True)
    tasks = load_tasks(cfg)
    log(f"{len(tasks)} tasks from {cfg['tasks']}, {cfg['jobs']} test processes, timeout {cfg['timeout']}s, "
        f"prlimit {'on' if PRLIMIT else 'unavailable'}")
    res_path = os.path.join(cfg["out"], "concept_results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for tag, template, sols in runs(cfg, tasks):
        t = time.time()
        recs, pyver = score(tasks, sols, cfg)
        s = summarise(recs)
        s.update(template=template, python=".".join(map(str, pyver)), seconds=round(time.time() - t))
        with open(os.path.join(cfg["out"], f"concept-{tag}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        results[tag] = s
        json.dump(results, open(res_path, "w"), indent=1)
        log(report(tag, s, template, pyver))
        fails = [r for r in recs if r["status"] not in ("pass", "skipped")]
        if cfg["self_test"] and fails:
            for r in fails:
                log(f"SELF-TEST FAIL {r['task_id']} {r['status']}\n{r['tail']}")
            raise SystemExit(1)
    return results


if __name__ == "__main__":  # also true when a notebook cell exec()s this file with CONFIG set
    main(sys.argv[1:])

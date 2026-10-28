"""Score bench_gen.py answers offline: a Kaggle CPU kernel with no internet and no keys.

Inputs (attached): the kit (eval/bench_kit.py output), the private eval data (ophyx-evaldata: the
concept suite, its run.py and the hands-on prompts) and one or more bench_gen kernels' outputs.
Every model answer runs in a child process with a wall-clock timeout, prlimit limits, nice 19, an
empty environment and its own temp dir.

  humaneval, mbpp   EvalPlus 0.3.1 (installed from the kit's wheels), base and plus pass@1
  concept, manual,  eval/concept_suite/run.py on precomputed completions
  realworld
  crux_o, crux_i    the CRUXEval check: run `assert <gold> == <prediction>` after the function
  lcb               LiveCodeBench: stdin/stdout programs, or Solution().<fn>(*args) for LeetCode;
                    all public + private tests must pass; output compared line by line, numbers as Decimal
  bcb_*, bcbf_*     BigCodeBench-Hard (bcb_*) and full BigCodeBench (bcbf_*), scored as BigCodeBench's calibrated
                    default: code_prompt with a stub task_func, then the answer (sanitized by BigCodeBench's rule,
                    which keeps functions without a return), with the task's unittest tests. "canonical" scores the reference solutions, and
                    pass@1 is also reported over the tasks whose reference passes here, since some tasks need
                    the network or a missing library
  mple_js, mple_cpp MultiPL-E HumanEval: prompt + completion + tests as MultiPL-E builds them (a chat answer's
  mple_sh           whole function, after the prompt's #includes); node from ophyx-evaldata, g++ -std=c++17
                    (compiled with more memory than the program gets), bash
  ds1000            DS-1000: the answer cut as its postprocess does, put into code_context's test_execution
                    (and test_string); pass@1 is also reported over the problems whose reference passes here

CONFIG: models (names to score; default all found), skip_models, benches, shard / nshards (split
the models across kernels), jobs (parallel children, default 4), canonical (default true: the runnable
sets of BigCodeBench-Hard, full BigCodeBench and DS-1000, each computed the first time a model's answers
need it; an attached bcb_runnable.json, bcbf_runnable.json or ds1000_runnable.json from an earlier run is
reused instead, so every shard scores the same tasks, unless recompute_canonical is set), sample_roots
(more directories to search for samples/<model>/meta.json, used by eval/bench_smoke.py).
"""

import concurrent.futures as cf
import glob
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
OUT = CONFIG.get("out", "/kaggle/working" if os.path.isdir("/kaggle") else os.path.abspath("bench_exec"))
JOBS = CONFIG.get("jobs", 4)
INPUT = CONFIG.get("input_root", "/kaggle/input")
PRLIMIT = shutil.which("prlimit")
T0 = time.time()
os.makedirs(OUT, exist_ok=True)
TMP = tempfile.mkdtemp(prefix="bx-")
FENCE_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)(?:\n```|\Z)", re.S)
LCB_HEADER = ("from typing import *\nfrom functools import *\nfrom collections import *\nfrom itertools import *\n"
              "from heapq import *\nfrom bisect import *\nfrom string import *\nfrom operator import *\nfrom math import *\n"
              "import math, sys, re, collections, itertools, heapq, bisect, functools, string, random, copy\n"
              "sys.setrecursionlimit(50000)\n")


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def _locate(marker):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", marker), recursive=True))
    if not hits:
        raise FileNotFoundError(marker)
    return os.path.dirname(hits[0])


def find(name):
    """A kit file (next to manifest.json) or an eval-data file (next to run.py), never a same-named sample file."""
    for marker in ("manifest.json", "run.py"):
        try:
            p = os.path.join(_locate(marker), name)
        except FileNotFoundError:
            continue
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def rows_of(path):
    return [json.loads(line) for line in open(path) if line.strip()]


# Repo-format markers some base models write as plain text (starcoder2's <file_sep>, stable-code's
# <repo_continuation>): vLLM keeps them, since the tokenizer doesn't mark them special. Plain words like
# <filename> also turn up in ordinary code ("usage: wc.py <filename>"), so those count only at a line start.
SPECIAL_RE = re.compile(r"<(?:file_sep|repo_continuation|\|endoftext\||\|fim_\w+\||fim_\w+|issue_\w+|jupyter_\w+"
                        r"|empty_output|commit_\w+)>|(?m:^<(?:filename|gh_stars|reponame)>)")


def trim_special(samples):
    """Cut each completion at the first marker; returns (samples, how many were cut)."""
    cut = 0
    for s in samples:
        m = SPECIAL_RE.search(s.get("completion") or "")
        if m:
            s["completion"] = s["completion"][: m.start()]
            cut += 1
    return samples, cut


def sandbox(cmd, cwd, timeout, stdin=None, mem_gb=4, extra_env=None, stdout_only=False):
    """Run cmd in its own session; returns (status, seconds, output). stdout_only: drop stderr and keep all
    of stdout (for answers compared against expected output), instead of the last 2,000 chars of both."""
    """Run cmd in cwd with limits; returns (status, seconds, output tail)."""
    if PRLIMIT:
        cmd = [PRLIMIT, f"--cpu={int(timeout) + 1}", f"--as={mem_gb << 30}", f"--fsize={100 << 20}", "--nofile=512",
               "--"] + cmd
    cmd = ["nice", "-n", "19"] + cmd
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": cwd, "TMPDIR": cwd, "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MPLBACKEND": "Agg", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **(extra_env or {})}
    t = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL if stdout_only else subprocess.STDOUT,
                         stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, start_new_session=True)

    def kill():  # the whole group: background children must not outlive the answer or hold the pipe open
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    try:
        out, _ = p.communicate(input=stdin.encode() if stdin is not None else None, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill()
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # a child that left the group still holds the pipe
            p.stdout.close()
        return "timeout", timeout, ""
    except Exception as e:
        kill()
        return "error", time.time() - t, str(e)[:300]
    kill()
    out = out.decode(errors="replace")
    return ("pass" if p.returncode == 0 else "fail"), time.time() - t, out if stdout_only else out[-2000:]


def workdir():
    return tempfile.mkdtemp(dir=TMP)


def parallel(fn, items):
    with cf.ThreadPoolExecutor(JOBS) as ex:
        return list(ex.map(fn, items))


def sanitize(code, entry):
    from evalplus.sanitize import sanitize as sz
    try:
        return sz(code, entry)
    except Exception:
        return code


def bcb_sanitize(code):
    """BigCodeBench's sanitizer: EvalPlus's, without the rule that drops a function with no return statement
    (a task_func that only plots or writes files would vanish)."""
    import types

    import evalplus.sanitize as es
    g = {**vars(es), "has_return_statement": lambda node: True}
    for name in ("extract_target_code_or_empty", "sanitize"):
        f = getattr(es, name)
        g[name] = types.FunctionType(f.__code__, g, name, f.__defaults__)
        g[name].__kwdefaults__ = f.__kwdefaults__
    try:
        return g["sanitize"](code, "task_func")
    except Exception:
        return code


def compiles(code):
    try:
        compile(code, "solution.py", "exec", dont_inherit=True)
        return True
    except Exception:
        return False


def extract_code(text):
    if "```" in text:
        blocks = FENCE_RE.findall(text)
        if blocks:
            good = [b for b in blocks if compiles(b)]
            return max(good or blocks, key=len)
    return text


# ---------------------------------------------------------------- EvalPlus


def evalplus_score(name, kind, ds, samples):
    sols = []
    for r in samples:
        code = r["completion"] if kind == "instruct" else r["prefill"] + r["completion"]
        sols.append({"task_id": r["task_id"], "solution": sanitize(code, r["entry_point"])})
    d = os.path.join(OUT, "evalplus", name)
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, f"{ds}.jsonl")
    with open(sp, "w") as f:
        for s in sols:
            f.write(json.dumps(s) + "\n")
    # EvalPlus runs each solution in its own limited process; offline, with no keys in this session
    r = subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", ds, "--samples", sp, "--parallel",
                        str(JOBS), "--i_just_wanna_run"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res_path = sp.replace(".jsonl", "_eval_results.json")
    if r.returncode != 0 or not os.path.exists(res_path):
        return {"error": r.stdout.decode(errors="replace")[-800:]}
    res = json.load(open(res_path))["eval"]
    n = len(res)
    base = sum(v[0]["base_status"] == "pass" for v in res.values())
    plus = sum(v[0]["base_status"] == "pass" and v[0]["plus_status"] == "pass" for v in res.values())
    return {"n": n, "base": round(100 * base / n, 1), "plus": round(100 * plus / n, 1),
            "parses": round(100 * sum(compiles(s["solution"]) for s in sols) / n, 1)}


# ---------------------------------------------------------------- concept suite and hands-on prompts (run.py)


def runpy_score(name, kind, bench, samples):
    tasks = find("tasks.jsonl" if bench == "concept" else f"{bench}.jsonl")
    d = os.path.join(OUT, bench)
    os.makedirs(d, exist_ok=True)
    cp = os.path.join(d, f"{name}.completions.jsonl")
    with open(cp, "w") as f:
        for r in samples:
            f.write(json.dumps({"task_id": r["task_id"], "completion": r["prefill"] + r["completion"]}) + "\n")
    template = "plain" if kind == "base" else "chat"
    r = subprocess.run([sys.executable, find("run.py"), "--completions", cp, "--tag", name, "--template", template,
                        "--tasks", tasks, "--jobs", str(JOBS), "--out", d, "--tmp", TMP],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res_path = os.path.join(d, "concept_results.json")
    if r.returncode != 0 or not os.path.exists(res_path):
        return {"error": r.stdout.decode(errors="replace")[-800:]}
    s = json.load(open(res_path)).get(name, {})
    return {k: s.get(k) for k in ("n", "pass@1", "passed", "parses", "timeouts", "by_area", "by_difficulty")}


# ---------------------------------------------------------------- CRUXEval

CRUX_CHILD = r'''
import json, signal, sys
items = json.load(open("items.json"))
res = {}
def alarm(*a):
    raise TimeoutError()
signal.signal(signal.SIGALRM, alarm)
for it in items:
    ok = False
    try:
        signal.alarm(3)
        ns = {}
        exec(it["program"], ns)
        ok = True
    except BaseException:
        ok = False
    finally:
        signal.alarm(0)
    res[it["id"]] = ok
    with open("res.json", "w") as f:
        json.dump(res, f)
'''


def crux_prediction(mode, text):
    gen = text.split("[/ANSWER]")[0]
    lines = [ln for ln in gen.split("\n") if "assert" in ln]
    gen = lines[0].strip() if lines else gen.strip()
    if mode == "o":
        return gen.split("==", 1)[1].strip() if "==" in gen else gen
    if "assert f" in gen:
        gen = "f" + gen.split("assert f", 1)[1].strip()
    return gen.split("==")[0].strip()


def crux_score(name, mode, samples):
    data = {r["id"]: r for r in rows_of(find("cruxeval.jsonl"))}
    items = []
    for s in samples:
        r = data[s["task_id"]]
        pred = crux_prediction(mode, s["completion"])
        check = f"assert {r['output']} == {pred}"  # output: pred is a literal; input: pred is the call f(...)
        if mode == "i" and not pred.startswith("f("):
            check = "assert False"
        items.append({"id": s["task_id"], "program": r["code"] + "\n" + check})
    return {"n": len(items), "pass@1": round(100 * sum(batched_asserts(items).values()) / max(len(items), 1), 1)}


def batched_asserts(items, size=100):
    out = {}

    def one(chunk):
        d = workdir()
        json.dump(chunk, open(os.path.join(d, "items.json"), "w"))
        open(os.path.join(d, "child.py"), "w").write(CRUX_CHILD)
        sandbox([sys.executable, "-I", "child.py"], d, timeout=4 * len(chunk) + 10, mem_gb=2)
        try:
            got = json.load(open(os.path.join(d, "res.json")))
        except Exception:
            got = {}
        shutil.rmtree(d, ignore_errors=True)
        return chunk, got

    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    retry = []
    for chunk, got in parallel(one, chunks):
        for it in chunk:
            if it["id"] in got:
                out[it["id"]] = bool(got[it["id"]])
            else:
                retry.append(it)
    for chunk, got in parallel(one, [[it] for it in retry]):  # a crash or a hang in a batch: run its rest one by one
        out[chunk[0]["id"]] = bool(got.get(chunk[0]["id"], False))
    return out


# ---------------------------------------------------------------- LiveCodeBench

LCB_CHILD = r'''
import io, json, signal, sys, time
from decimal import Decimal, InvalidOperation
spec = json.load(open("spec.json"))
code = open("solution.py").read()
def alarm(*a):
    raise TimeoutError()
signal.signal(signal.SIGALRM, alarm)
def same(out, exp):
    a = [l.strip() for l in out.strip().split("\n")]
    b = [l.strip() for l in exp.strip().split("\n")]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        xs, ys = x.split(), y.split()
        if len(xs) != len(ys):
            return False
        for u, v in zip(xs, ys):
            if u == v:
                continue
            try:
                if Decimal(u) != Decimal(v):
                    return False
            except (InvalidOperation, ValueError):
                return False
    return True
def norm(x):
    try:
        return json.loads(json.dumps(x))
    except Exception:
        return x
status = "pass"
detail = ""
compiled = compile(code, "solution.py", "exec")
if spec["mode"] == "call":
    ns = {"__name__": "solution"}
    try:
        signal.alarm(10)
        exec(compiled, ns)
        signal.alarm(0)
        sol = ns["Solution"]()
        fn = getattr(sol, spec["fn"])
    except BaseException as e:
        signal.alarm(0)
        status, detail = "fail", f"load: {type(e).__name__}: {e}"[:300]
    if status == "pass":
        for i, t in enumerate(spec["tests"]):
            try:
                args = [json.loads(x) for x in t["input"].split("\n") if x.strip()]
                exp = json.loads(t["output"])
                signal.alarm(spec["per_test"])
                got = fn(*args)
                signal.alarm(0)
                if norm(got) != exp:
                    status, detail = "fail", f"test {i}: wrong answer"
                    break
            except BaseException as e:
                signal.alarm(0)
                status, detail = "fail", f"test {i}: {type(e).__name__}: {str(e)[:200]}"
                break
else:
    real_stdout = sys.stdout
    for i, t in enumerate(spec["tests"]):
        buf = io.StringIO()
        sys.stdin = io.TextIOWrapper(io.BytesIO(t["input"].encode()))
        sys.stdout = buf
        err = None
        try:
            signal.alarm(spec["per_test"])
            exec(compiled, {"__name__": "__main__"})
        except SystemExit:
            pass
        except BaseException as e:
            err = e
        finally:
            signal.alarm(0)
            sys.stdout = real_stdout
        if err is not None:
            status, detail = "fail", f"test {i}: {type(err).__name__}: {str(err)[:200]}"
            break
        if not same(buf.getvalue(), t["output"]):
            status, detail = "fail", f"test {i}: wrong answer"
            break
json.dump({"status": status, "detail": detail}, open("res.json", "w"))
'''
LCB_SUBPROC = ("open(0", "fileno()", "os.read(0", "sys.stdin.buffer.raw")


def lcb_one(code, meta, tests):
    d = workdir()
    try:
        if not compiles(code):
            return "syntax_error", ""
        call = meta["testtype"] == "functional"
        if call:
            code = LCB_HEADER + code
        open(os.path.join(d, "solution.py"), "w").write(code)
        if not call and any(s in code for s in LCB_SUBPROC):  # reads fd 0 directly: one process per test
            for i, t in enumerate(tests):
                st, _, out = sandbox([sys.executable, "-I", "solution.py"], d, timeout=8, stdin=t["input"],
                                     stdout_only=True)
                if st != "pass" or not _same_lines(out, t["output"]):
                    return "fail", f"test {i}"
            return "pass", ""
        json.dump({"mode": "call" if call else "stdin", "fn": meta.get("fn_name"), "tests": tests, "per_test": 6},
                  open(os.path.join(d, "spec.json"), "w"))
        open(os.path.join(d, "child.py"), "w").write(LCB_CHILD)
        st, _, out = sandbox([sys.executable, "-I", "child.py"], d, timeout=min(240, 10 + 6 * len(tests)), mem_gb=4)
        try:
            r = json.load(open(os.path.join(d, "res.json")))
            return r["status"], r["detail"]
        except Exception:
            return ("timeout" if st == "timeout" else "fail"), out[-200:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _same_lines(out, exp):
    """The same comparison as same() in LCB_CHILD: line by line, numbers compared as decimals."""
    from decimal import Decimal, InvalidOperation
    a = [x.strip() for x in out.strip().split("\n")]
    b = [x.strip() for x in exp.strip().split("\n")]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        xs, ys = x.split(), y.split()
        if len(xs) != len(ys):
            return False
        for u, v in zip(xs, ys):
            if u == v:
                continue
            try:
                if Decimal(u) != Decimal(v):
                    return False
            except (InvalidOperation, ValueError):
                return False
    return True


def lcb_score(name, kind, samples):
    meta = {r["task_id"]: r for r in rows_of(find("lcb.jsonl"))}
    tdir = os.path.dirname(find("lcb.jsonl"))

    def one(s):
        m = meta[s["task_id"]]
        text = s["prefill"] + s["completion"] if kind != "instruct" else s["completion"]
        code = extract_code(text)
        tests = json.load(open(os.path.join(tdir, "lcb_tests", f"{s['task_id']}.json")))
        st, detail = lcb_one(code, m, tests)
        return {"task_id": s["task_id"], "difficulty": m["difficulty"], "date": m["date"], "status": st, "detail": detail}

    recs = parallel(one, samples)
    _dump(f"lcb/{name}.jsonl", recs)
    by = {}
    for r in recs:
        by.setdefault(r["difficulty"], []).append(r["status"] == "pass")
    n = len(recs)
    return {"n": n, "pass@1": round(100 * sum(r["status"] == "pass" for r in recs) / max(n, 1), 1),
            "by_difficulty": {k: {"n": len(v), "pass@1": round(100 * sum(v) / len(v), 1)} for k, v in sorted(by.items())}}


# ---------------------------------------------------------------- BigCodeBench-Hard


def bcb_code(kind, s, task):
    text = s["completion"] if kind == "instruct" else s["prefill"] + s["completion"]
    code = bcb_sanitize(extract_code(text) if kind == "instruct" else text)
    # BigCodeBench's calibrated score: the whole prompt (imports, constants, classes the tests use) with a stub
    # task_func, then the answer, whose task_func replaces it
    return task["code_prompt"] + "\n    pass\n" + code


ROOT_AT_START = set(os.listdir("/"))


def clear_strays():
    """Remove top-level directories a test created and left behind: BigCodeBench/832 writes under
    /non/existent/path/, which 288, 350 and 1101 expect not to exist."""
    for name in set(os.listdir("/")) - ROOT_AT_START:
        shutil.rmtree(os.path.join("/", name), ignore_errors=True)


def bcb_run(code, task):
    clear_strays()
    d = workdir()
    try:
        open(os.path.join(d, "test_bcb.py"), "w").write(code + "\n\n" + task["test"] + "\n")
        st, sec, out = sandbox([sys.executable, "-m", "unittest", "-q", "test_bcb"], d, timeout=120, mem_gb=8)
        return st, out[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bcb_score(name, kind, split, samples, ok_ids, kit="bcb_hard.jsonl", tag="bcb"):
    tasks = {t["task_id"]: t for t in rows_of(find(kit))}

    def one(s):
        t = tasks[s["task_id"]]
        st, tail = bcb_run(bcb_code(kind, s, t), t)
        return {"task_id": s["task_id"], "status": st, "tail": tail}

    recs = parallel(one, samples)
    _dump(f"{tag}/{name}-{split}.jsonl", recs)
    return pass_rates(recs, ok_ids)


def pass_rates(recs, ok_ids):
    n = len(recs)
    passed = {r["task_id"] for r in recs if r["status"] == "pass"}
    res = {"n": n, "pass@1": round(100 * len(passed) / max(n, 1), 1)}
    if ok_ids:
        res["pass@1_runnable"] = round(100 * len(passed & ok_ids) / len(ok_ids), 1)
        res["n_runnable"] = len(ok_ids)
    return res


def bcb_canonical(kit="bcb_hard.jsonl", tag="bcb"):
    tasks = rows_of(find(kit))

    def one(t):
        code = t["code_prompt"] + "\n" + t["canonical_solution"]
        st, tail = bcb_run(code, t)
        return t["task_id"], st, tail

    res = parallel(one, tasks)
    _dump(f"{tag}/canonical.jsonl", [{"task_id": a, "status": b, "tail": c} for a, b, c in res])
    ok = {a for a, b, _ in res if b == "pass"}
    log(f"{tag} canonical: {len(ok)}/{len(tasks)} reference solutions pass in this sandbox")
    return ok, len(tasks)


# ---------------------------------------------------------------- MultiPL-E

MPLE_TAGS = {"js": ("js", "javascript", "node"), "cpp": ("cpp", "c++", "cc"), "sh": ("sh", "bash", "shell")}
MPLE_FILE = {"js": "prog.js", "cpp": "prog.cpp", "sh": "prog.sh"}
ANY_FENCE_RE = re.compile(r"```[ \t]*([\w+#-]*)[ \t]*\n(.*?)(?:\n```|\Z)", re.S)


def node():
    """node, or the official binary that ophyx-evaldata carries: Kaggle has no node, and datasets drop the
    executable bit, so it starts through the loader."""
    found = shutil.which("node", path="/usr/bin:/bin:/usr/local/bin")
    if found:
        return [found]
    shipped = sorted(glob.glob(os.path.join(INPUT, "**", "node-v*-linux-x64"), recursive=True))
    if not shipped:
        raise FileNotFoundError("node")
    return [shipped[-1]] if os.access(shipped[-1], os.X_OK) else ["/lib64/ld-linux-x86-64.so.2", shipped[-1]]


def mple_entry(tests):
    """The function the tests call: `candidate = name;` (JavaScript, C++) or the call inside candidate() (Bash)."""
    m = re.search(r"candidate = (\w+)", tests) or re.search(r"candidate\(\) \{\s*(\w+)", tests)
    return m.group(1) if m else None


def mple_extract(lang, text, entry):
    """A chat answer's code: the longest block in the language (or untagged) that names the function, else the
    longest such block, else any block; with no fence, the text itself (the prompt opened the block)."""
    blocks = [(t.lower(), b) for t, b in ANY_FENCE_RE.findall(text)]
    if not blocks:
        return text
    mine = [b for t, b in blocks if t in MPLE_TAGS[lang] or not t]
    named = [b for b in mine if entry and entry in b]
    return max(named or mine or [b for _, b in blocks], key=len)


def drop_main(code):
    """A C++ answer's own main(): the tests bring theirs."""
    m = re.search(r"^[ \t]*int\s+main\s*\([^)]*\)\s*\{", code, re.M)
    if not m:
        return code
    depth = 0
    for i in range(m.end() - 1, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            return code[:m.start()] + code[i + 1:]
    return code[:m.start()]


def mple_program(lang, kind, s, task):
    """The file to run. A base model continues the prompt, which MultiPL-E's tests then close (C++, Bash) and call.
    A chat answer is a whole function: it goes after the prompt's #include lines, and the tests lose the brace
    that would have closed the prompt's function."""
    if kind == "base":
        return s["prefill"] + s["completion"] + "\n" + task["tests"]
    code = mple_extract(lang, s["completion"], mple_entry(task["tests"]))
    tests = task["tests"]
    if lang in ("cpp", "sh") and tests.lstrip().startswith("}"):
        tests = tests.lstrip()[1:]
    if lang == "cpp":
        code = "".join(ln + "\n" for ln in task["prompt"].split("\n") if ln.startswith("#include")) + drop_main(code)
    return code + "\n" + tests


def mple_run(lang, program):
    d = workdir()
    try:
        src = MPLE_FILE[lang]
        open(os.path.join(d, src), "w").write(program)
        if lang == "cpp":
            st, _, out = sandbox(["g++", "-std=c++17", "-o", "prog", src], d, timeout=60, mem_gb=4)
            if st != "pass":
                return ("compile_error" if st == "fail" else st), out[-300:]
            cmd = ["./prog"]
        else:
            cmd = node() + [src] if lang == "js" else ["bash", src]
        st, _, out = sandbox(cmd, d, timeout=15, mem_gb=2)
        return st, out[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def mple_score(name, kind, lang, samples):
    tasks = {t["task_id"]: t for t in rows_of(find(f"mple_{lang}.jsonl"))}

    def one(s):
        st, tail = mple_run(lang, mple_program(lang, kind, s, tasks[s["task_id"]]))
        return {"task_id": s["task_id"], "status": st, "tail": tail}

    recs = parallel(one, samples)
    _dump(f"mple/{name}-{lang}.jsonl", recs)
    res = pass_rates(recs, None)
    if lang == "cpp":
        res["compile_errors"] = sum(r["status"] == "compile_error" for r in recs)
    return res


# ---------------------------------------------------------------- DS-1000


def ds1000_code(text):
    """DS-1000's postprocess (cut at </code>, drop <code>, cut at END SOLUTION), taking the fenced block when a
    chat model writes prose around it."""
    text = text.split("</code>")[0]
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1)
    return text.split("\nEND SOLUTION")[0].split("# SOLUTION END")[0].replace("<code>", "")


def ds1000_run(code, task):
    ctx = task["code_context"]
    program = ctx + f"\ncode = {code!r}\ntest_execution(code)\n" + ("test_string(code)\n" if "test_string(" in ctx else "")
    d = workdir()
    try:
        open(os.path.join(d, "test_ds.py"), "w").write(program)
        st, _, out = sandbox([sys.executable, "-I", "test_ds.py"], d, timeout=120, mem_gb=8)
        return st, out[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def ds1000_score(name, kind, samples, ok_ids):
    tasks = {t["task_id"]: t for t in rows_of(find("ds1000.jsonl"))}

    def one(s):
        t = tasks[s["task_id"]]
        text = s["completion"] if kind == "instruct" else s["prefill"] + s["completion"]
        st, tail = ds1000_run(ds1000_code(text), t)
        return {"task_id": s["task_id"], "library": t["library"], "status": st, "tail": tail}

    recs = parallel(one, samples)
    _dump(f"ds1000/{name}.jsonl", recs)
    res = pass_rates(recs, ok_ids)
    by = {}
    for r in recs:
        by.setdefault(r["library"], []).append(r["status"] == "pass")
    res["by_library"] = {k: {"n": len(v), "pass@1": round(100 * sum(v) / len(v), 1)} for k, v in sorted(by.items())}
    return res


def ds1000_canonical():
    tasks = rows_of(find("ds1000.jsonl"))
    res = parallel(lambda t: (t["task_id"], *ds1000_run(t["reference_code"], t)), tasks)
    _dump("ds1000/canonical.jsonl", [{"task_id": a, "status": b, "tail": c} for a, b, c in res])
    ok = {a for a, b, _ in res if b == "pass"}
    log(f"ds1000 canonical: {len(ok)}/{len(tasks)} reference solutions pass in this sandbox")
    return ok, len(tasks)


# family: (runnable-set file, the key in results["canonical"], how to compute it)
CANONICAL = {"bcb": ("bcb_runnable.json", "bcb_hard", bcb_canonical),
             "bcbf": ("bcbf_runnable.json", "bcb_full", lambda: bcb_canonical("bcb_full.jsonl", "bcbf")),
             "ds1000": ("ds1000_runnable.json", "ds1000", ds1000_canonical)}
RUNNABLE = {}


def runnable(family, results):
    """The tasks whose reference passes in this sandbox, for one family; None when canonical is off and no
    earlier set is attached. Computed once per kernel, the first time a model's answers need it."""
    if family in RUNNABLE:
        return RUNNABLE[family]
    fname, key, compute = CANONICAL[family]
    ok_ids = None
    prior = sorted(glob.glob(os.path.join(INPUT, "**", fname), recursive=True))
    if prior and not CONFIG.get("recompute_canonical"):  # one runnable set for every shard and every rerun
        ok_ids = set(json.load(open(prior[0])))
        log(f"{family}: {len(ok_ids)} runnable tasks from {prior[0]}")
    elif CONFIG.get("canonical", True):
        ok_ids, n = compute()
        results.setdefault("canonical", {})[key] = {"n": n, "pass@1": round(100 * len(ok_ids) / n, 1),
                                                    "n_runnable": len(ok_ids)}
    if ok_ids is not None:
        json.dump(sorted(ok_ids), open(os.path.join(OUT, fname), "w"))
    RUNNABLE[family] = ok_ids
    return ok_ids


def _dump(rel, recs):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- main


def setup():
    kit = os.path.dirname(find("manifest.json"))
    wheels = os.path.join(kit, "wheels")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheels, "--no-deps",
                    "--no-build-isolation", *glob.glob(os.path.join(wheels, "*"))], check=False)
    cache = os.path.expanduser("~/.cache/evalplus")
    os.makedirs(cache, exist_ok=True)
    for f in glob.glob(os.path.join(kit, "evalplus_cache", "*")):
        shutil.copy(f, cache)
    try:  # the first start reads node off the dataset mount, which once took longer than a 20 s test allows
        subprocess.run(node() + ["-e", "0"], timeout=300, capture_output=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def sample_dirs():
    found = {}
    metas = [m for root in [INPUT, *CONFIG.get("sample_roots", [])]
             for m in sorted(glob.glob(os.path.join(root, "**", "samples", "*", "meta.json"), recursive=True))]
    for meta in metas:
        m = json.load(open(meta))
        d = os.path.dirname(meta)
        if m["name"] in found:  # the same model from two generation runs: keep the one with more benches
            old = found[m["name"]][0]
            n_old, n_new = len(glob.glob(os.path.join(old, "*.jsonl"))), len(glob.glob(os.path.join(d, "*.jsonl")))
            log(f"{m['name']}: samples in {old} and {d}; using the one with more benches")
            if n_new <= n_old:
                continue
        found[m["name"]] = (d, m)
    return found


def main():
    setup()
    models = sample_dirs()
    names = sorted(models)
    if CONFIG.get("models"):
        names = [n for n in names if n in CONFIG["models"]]
    names = [n for n in names if n not in CONFIG.get("skip_models", [])]
    shard, nshards = CONFIG.get("shard", 0), CONFIG.get("nshards", 1)
    names = names[shard::nshards]
    log(f"scoring {len(names)} models: {names}")
    res_path = os.path.join(OUT, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    benches = CONFIG.get("benches")
    for name in names:
        d, meta = models[name]
        kind = meta["kind"]
        results[name] = {"_kind": kind, "_model": meta["model"]}
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            bench = os.path.basename(f)[:-6]
            if benches and bench not in benches:
                continue
            samples, cut = trim_special(rows_of(f))
            if cut:
                print(f"{name} {bench}: cut {cut} completions at a repo-format marker", flush=True)
            t = time.time()
            try:
                if bench in ("humaneval", "mbpp"):
                    r = evalplus_score(name, kind, bench, samples)
                elif bench in ("concept", "manual", "realworld"):
                    r = runpy_score(name, kind, bench, samples)
                elif bench in ("crux_o", "crux_i"):
                    r = crux_score(name, bench[-1], samples)
                elif bench == "lcb":
                    r = lcb_score(name, kind, samples)
                elif bench.startswith("bcb_"):
                    r = bcb_score(name, kind, bench[4:], samples, runnable("bcb", results))
                elif bench.startswith("bcbf_"):
                    r = bcb_score(name, kind, bench[5:], samples, runnable("bcbf", results), "bcb_full.jsonl", "bcbf")
                elif bench.startswith("mple_"):
                    r = mple_score(name, kind, bench[5:], samples)
                elif bench == "ds1000":
                    r = ds1000_score(name, kind, samples, runnable("ds1000", results))
                else:
                    continue
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = {"error": f"{type(e).__name__}: {e}"[:500]}
            r["seconds"] = round(time.time() - t)
            results[name][bench] = r
            log(f"{name} {bench}: " + json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}))
            json.dump(results, open(res_path, "w"), indent=1)
    log("EXEC DONE")


if __name__ == "__main__":
    main()

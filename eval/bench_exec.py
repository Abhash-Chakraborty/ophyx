"""Score bench_gen.py answers offline: a Kaggle CPU kernel with no internet and no keys.

Inputs (attached): the kit (eval/bench_kit.py output), the private eval data (pycoder-evaldata: the
concept suite, its run.py and the hands-on prompts) and one or more bench_gen kernels' outputs.
Every model answer runs in a child process with a wall-clock timeout, prlimit limits, nice 19, an
empty environment and its own temp dir.

  humaneval, mbpp   EvalPlus 0.3.1 (installed from the kit's wheels), base and plus pass@1
  concept, manual   eval/concept_suite/run.py on precomputed completions
  crux_o, crux_i    the CRUXEval check: run `assert <gold> == <prediction>` after the function
  lcb               LiveCodeBench: stdin/stdout programs, or Solution().<fn>(*args) for LeetCode;
                    all public + private tests must pass; output compared line by line, numbers as Decimal
  bcb_*             BigCodeBench-Hard: the solution (plus code_prompt imports) with the task's unittest
                    tests. "canonical" scores the reference solutions, and pass@1 is also reported over
                    the tasks whose reference passes here, since some tasks need the network or a missing library

CONFIG: models (names to score; default all found), skip_models, benches, shard / nshards (split
the models across kernels), jobs (parallel children, default 4), canonical (default true on shard 0).
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


def sandbox(cmd, cwd, timeout, stdin=None, mem_gb=4, extra_env=None):
    """Run cmd in cwd with limits; returns (status, seconds, output tail)."""
    if PRLIMIT:
        cmd = [PRLIMIT, f"--cpu={int(timeout) + 1}", f"--as={mem_gb << 30}", f"--fsize={100 << 20}", "--nofile=512",
               "--"] + cmd
    cmd = ["nice", "-n", "19"] + cmd
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": cwd, "TMPDIR": cwd, "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MPLBACKEND": "Agg", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **(extra_env or {})}
    t = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, start_new_session=True)
    try:
        out, _ = p.communicate(input=stdin.encode() if stdin is not None else None, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return "timeout", timeout, ""
    except Exception as e:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        return "error", time.time() - t, str(e)[:300]
    return ("pass" if p.returncode == 0 else "fail"), time.time() - t, out.decode(errors="replace")[-2000:]


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
    tasks = find("tasks.jsonl" if bench == "concept" else "manual.jsonl")
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
                st, _, out = sandbox([sys.executable, "-I", "solution.py"], d, timeout=8, stdin=t["input"])
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
    return [x.strip() for x in out.strip().split("\n")] == [x.strip() for x in exp.strip().split("\n")]


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
    code = sanitize(extract_code(text) if kind == "instruct" else text, "task_func")
    imports = [ln for ln in task["code_prompt"].split("\n") if ln.startswith(("import ", "from "))]
    return "\n".join(imports) + "\n\n" + code  # the task's own imports, as BigCodeBench's calibrated score does


def bcb_run(code, task):
    d = workdir()
    try:
        open(os.path.join(d, "test_bcb.py"), "w").write(code + "\n\n" + task["test"] + "\n")
        st, sec, out = sandbox([sys.executable, "-m", "unittest", "-q", "test_bcb"], d, timeout=120, mem_gb=8)
        return st, out[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bcb_score(name, kind, split, samples, ok_ids):
    tasks = {t["task_id"]: t for t in rows_of(find("bcb_hard.jsonl"))}

    def one(s):
        t = tasks[s["task_id"]]
        st, tail = bcb_run(bcb_code(kind, s, t), t)
        return {"task_id": s["task_id"], "status": st, "tail": tail}

    recs = parallel(one, samples)
    _dump(f"bcb/{name}-{split}.jsonl", recs)
    n = len(recs)
    passed = {r["task_id"] for r in recs if r["status"] == "pass"}
    res = {"n": n, "pass@1": round(100 * len(passed) / max(n, 1), 1)}
    if ok_ids:
        res["pass@1_runnable"] = round(100 * len(passed & ok_ids) / len(ok_ids), 1)
        res["n_runnable"] = len(ok_ids)
    return res


def bcb_canonical():
    tasks = rows_of(find("bcb_hard.jsonl"))

    def one(t):
        code = t["code_prompt"] + "\n" + t["canonical_solution"]
        st, tail = bcb_run(code, t)
        return t["task_id"], st, tail

    res = parallel(one, tasks)
    _dump("bcb/canonical.jsonl", [{"task_id": a, "status": b, "tail": c} for a, b, c in res])
    ok = {a for a, b, _ in res if b == "pass"}
    log(f"bcb canonical: {len(ok)}/{len(tasks)} reference solutions pass in this sandbox")
    return ok


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


def sample_dirs():
    found = {}
    for meta in sorted(glob.glob(os.path.join(INPUT, "**", "samples", "*", "meta.json"), recursive=True)):
        m = json.load(open(meta))
        found[m["name"]] = (os.path.dirname(meta), m)
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
    ok_ids = None
    if CONFIG.get("canonical", shard == 0):
        ok_ids = bcb_canonical()
        results["canonical"] = {"bcb_hard": {"n": 148, "pass@1": round(100 * len(ok_ids) / 148, 1)}}
        json.dump(sorted(ok_ids), open(os.path.join(OUT, "bcb_runnable.json"), "w"))
    for name in names:
        d, meta = models[name]
        kind = meta["kind"]
        results[name] = {"_kind": kind, "_model": meta["model"]}
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            bench = os.path.basename(f)[:-6]
            if benches and bench not in benches:
                continue
            samples = rows_of(f)
            t = time.time()
            try:
                if bench in ("humaneval", "mbpp"):
                    r = evalplus_score(name, kind, bench, samples)
                elif bench in ("concept", "manual"):
                    r = runpy_score(name, kind, bench, samples)
                elif bench in ("crux_o", "crux_i"):
                    r = crux_score(name, bench[-1], samples)
                elif bench == "lcb":
                    r = lcb_score(name, kind, samples)
                elif bench.startswith("bcb_"):
                    r = bcb_score(name, kind, bench[4:], samples, ok_ids)
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

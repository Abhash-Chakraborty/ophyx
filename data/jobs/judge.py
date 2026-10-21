"""Sandboxed judge for contest-style programs and pytest modules (included before a job script by the launcher).

Only for Kaggle CPU sessions with internet off and no keys attached: the code it runs is untrusted.

  judge(code, tests, fn_name=None)   stdin/stdout program (fn_name None) or call-based function, against
                                     tests [{"input", "output"}]. Call-based inputs are a JSON list of
                                     arguments and outputs a JSON value (APPS/TACO convention).
  judge_pytest(module)               `pytest module.py`, as data/synth/verify.py runs it

One subprocess per call: `python -I runner.py` under prlimit (CPU, address space, file size, open
files), an empty environment, its own temp dir and process group. Inside it, every test runs in a
forked child (no interpreter start-up per test) with its own CPU limit, stdin and stdout redirected
to files. The first failing test is reported with its kind (wrong_answer, runtime_error, timeout),
its input, the expected and actual output, and the traceback.
"""

import json as _json
import os as _os
import shutil as _shutil
import signal as _signal
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile

_PRLIMIT = _shutil.which("prlimit")

JUDGE_RUNNER = r'''
import io, json, math, os, resource, sys, time, traceback
if hasattr(sys, "set_int_max_str_digits"):  # contest answers can be huge integers
    sys.set_int_max_str_digits(0)
P = json.load(open(sys.argv[1]))
code, tests, fn_name = P["code"], P["tests"], P.get("fn_name")
per = P.get("per_test_s", 4)
res = {"status": "pass", "n_run": 0, "n_pass": 0, "first_fail": None, "fails": 0}


def done():
    json.dump(res, open("result.json", "w"))
    sys.exit(0)


with open("solution.py", "w") as f:  # tracebacks then show the source lines
    f.write(code)
try:
    compiled = compile(code, "solution.py", "exec", dont_inherit=True)
except BaseException as e:
    res.update(status="fail", first_fail={"i": -1, "kind": "syntax_error", "stderr": f"{type(e).__name__}: {e}"[-1500:]})
    done()


def jsonable(x):
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return sorted((jsonable(v) for v in x), key=repr)
    return x


def child(i, t):
    resource.setrlimit(resource.RLIMIT_CPU, (per, per + 1))
    os.dup2(os.open(f"in{i}", os.O_RDONLY), 0)
    os.dup2(os.open(f"out{i}", os.O_WRONLY | os.O_CREAT | os.O_TRUNC), 1)
    os.dup2(os.open(f"err{i}", os.O_WRONLY | os.O_CREAT | os.O_TRUNC), 2)
    sys.stdin = io.TextIOWrapper(io.BufferedReader(io.FileIO(0, "r", closefd=False)))
    sys.stdout = io.TextIOWrapper(io.BufferedWriter(io.FileIO(1, "w", closefd=False)))
    sys.stderr = io.TextIOWrapper(io.FileIO(2, "w", closefd=False), write_through=True)
    status = 0
    try:
        if fn_name is None:
            exec(compiled, {"__name__": "__main__", "__builtins__": __builtins__})
        else:
            g = {"__name__": "solution", "__builtins__": __builtins__}
            exec(compiled, g)
            fn = g.get(fn_name)
            if not callable(fn):
                cls = g.get("Solution") or next((v for v in g.values() if isinstance(v, type) and hasattr(v, fn_name)), None)
                if cls is None:
                    raise NameError(f"no function or method named {fn_name!r}")
                fn = getattr(cls(), fn_name)
            out = fn(*json.loads(t["input"]))
            with open(f"ret{i}", "w") as f:
                try:
                    json.dump(jsonable(out), f)
                except (TypeError, ValueError):
                    json.dump(repr(out), f)
    except SystemExit as e:
        status = 0 if e.code in (None, 0) else 1
        if status:
            print(f"SystemExit: {e.code}", file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        status = 1
    try:
        sys.stdout.flush()
    except BaseException:
        pass
    os._exit(status)


def same_text(got, want):
    norm = lambda s: [l.rstrip() for l in s.replace("\r\n", "\n").rstrip().split("\n")]
    if norm(got) == norm(want):
        return True
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if not math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6):
            return False
    return True


def same_value(got, want):
    if got == want:
        return True
    if isinstance(want, list) and len(want) == 1 and got == want[0]:
        return True
    if isinstance(got, (int, float)) and isinstance(want, (int, float)) and not isinstance(got, bool):
        return math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(want, list) and len(want) == 1 and isinstance(want[0], (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, want[0], rel_tol=1e-6, abs_tol=1e-6)
    return False


def tail(path, n=1500):
    try:
        with open(path, errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


for i, t in enumerate(tests):
    with open(f"in{i}", "w") as f:
        f.write(t["input"] if fn_name is None else "")
    pid = os.fork()
    if pid == 0:
        child(i, t)
    t0, wait, status, timed_out = time.time(), 0.0005, None, False
    while True:
        p, st = os.waitpid(pid, os.WNOHANG)
        if p:
            status = st
            break
        if time.time() - t0 > per * 2 + 1:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            timed_out = True
            break
        time.sleep(wait)
        wait = min(wait * 2, 0.02)
    res["n_run"] += 1
    kind = None
    if timed_out or (os.WIFSIGNALED(status) and os.WTERMSIG(status) in (9, 24)):
        kind = "timeout"
    elif not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        kind = "runtime_error"
    else:
        if fn_name is None:
            got = tail(f"out{i}", 1 << 20)
            ok = same_text(got, t["output"])
        else:
            try:
                got = json.load(open(f"ret{i}"))
                ok = same_value(got, json.loads(t["output"]))
            except (OSError, ValueError):
                got, ok = None, False
        if not ok:
            kind = "wrong_answer"
    if kind is None:
        res["n_pass"] += 1
    else:
        res["fails"] += 1
        if res["first_fail"] is None:
            got_s = tail(f"out{i}", 1500) if fn_name is None else (json.dumps(got)[:1500] if kind == "wrong_answer" else None)
            res["first_fail"] = {"i": i, "kind": kind, "input": t["input"][:4000], "expected": t["output"][:4000],
                                 "got": got_s, "stderr": tail(f"err{i}")}
        res["status"] = "fail"
        if P.get("stop_on_fail", True):
            break
    for name in (f"in{i}", f"out{i}", f"err{i}", f"ret{i}"):
        try:
            os.remove(name)
        except OSError:
            pass
done()
'''


def _sandbox_run(cmd, d, wall_s, cpu_s, mem=2 << 30):
    env = {"PATH": "/usr/bin:/bin", "HOME": d, "TMPDIR": d, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if _PRLIMIT:
        cmd = [_PRLIMIT, f"--cpu={cpu_s}", f"--as={mem}", f"--fsize={50 << 20}", "--nofile=256", "--"] + cmd
    p = _subprocess.Popen(cmd, cwd=d, env=env, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                          stdin=_subprocess.DEVNULL, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=wall_s)
        return p.returncode, out.decode(errors="replace"), False
    except _subprocess.TimeoutExpired:
        try:
            _os.killpg(p.pid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        return p.returncode, out.decode(errors="replace"), True


def judge(code, tests, fn_name=None, per_test_s=4, stop_on_fail=True, total_s=None):
    d = _tempfile.mkdtemp(prefix="jx-")
    try:
        with open(_os.path.join(d, "runner.py"), "w") as f:
            f.write(JUDGE_RUNNER)
        with open(_os.path.join(d, "payload.json"), "w") as f:
            _json.dump({"code": code, "tests": tests, "fn_name": fn_name, "per_test_s": per_test_s,
                        "stop_on_fail": stop_on_fail}, f)
        total = total_s or (per_test_s * 2 + 2) * len(tests) + 10
        rc, out, timed_out = _sandbox_run([_sys.executable, "-I", "runner.py", "payload.json"], d, total,
                                          per_test_s * len(tests) + 20)
        try:
            with open(_os.path.join(d, "result.json")) as f:
                return _json.load(f)
        except (OSError, ValueError):
            return {"status": "fail", "n_run": 0, "n_pass": 0, "fails": 1,
                    "first_fail": {"i": -1, "kind": "timeout" if timed_out else "judge_error", "stderr": out[-1500:]}}
    finally:
        _shutil.rmtree(d, ignore_errors=True)


def judge_pytest(module, timeout=40):
    """-> (ok, output tail). ok means exit 0 with at least one test passed and none failed."""
    d = _tempfile.mkdtemp(prefix="jp-")
    try:
        with open(_os.path.join(d, "mod.py"), "w") as f:
            f.write(module)
        cmd = [_sys.executable, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-o", "addopts=", "mod.py"]
        rc, out, timed_out = _sandbox_run(cmd, d, timeout, timeout)
        if timed_out:
            return False, "timeout"
        return rc == 0 and " passed" in out and " failed" not in out, out[-4000:]
    finally:
        _shutil.rmtree(d, ignore_errors=True)

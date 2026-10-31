"""Run the code from data/synth/chat_data.py against its checks. Kaggle CPU, internet off, no keys.

The launcher includes data/jobs/judge.py first. Attach the chat-data kernel output (chat/*.jsonl).

Each answer runs once as a program (__name__ == "__main__") in the judge's sandbox, with time.sleep
made instant so countdowns and timers finish. A check passes when:
  program  the run exits cleanly on the check's stdin and the output contains every expected string
           (an expected ANSI colour code matches any escape sequence carrying that colour number; runs of
           spaces don't matter, and a number matches the same value written another way: 77 and 77.0)
  asserts  the answer with the assert lines appended exits cleanly
  none     the code only has to parse; it may need files or a network, so it isn't run
A program that stops because the check didn't give it what it needs (a missing file, command-line
arguments, more input) counts as unchecked, like "none". A missing third-party module is a failure:
the house style is standard library only.
Teacher answers with no check, or a check that failed to parse, count as "none".

What runs: teacher answers (answers.jsonl), follow-up and complaint answers (the check of the request
they continue: their own for follow-ups, the original request's for complaints) and every Ophyx answer
(ophyx.jsonl). Ophyx answers are judged by the original request's check. Samples from eval/chat_gen.py
(samples-<model>.jsonl) carry their own check and are judged the same way.

CONFIG: workers (4), src (/kaggle/input).
Output in /kaggle/working/chat/: checked.jsonl, one row per run {"file", "id", "i", "ok", "why", "kind"},
plus the input files copied over, so later jobs need only this kernel.
"""

import glob as _glob
import json as _json
import os as _os
import re as _re
import shutil as _shutil
import time as _time
from concurrent.futures import ThreadPoolExecutor as _Pool

C_SRC = CONFIG.get("src", "/kaggle/input")
C_OUT = _os.path.join(CONFIG.get("out", "/kaggle/working"), "chat")
C_T0 = _time.time()
C_DROP = CONFIG.get("drop", True)  # run each answer as its own user (needs root, as on Kaggle)
SENTINEL = "\x00no output matches this\x00"
PRELUDE = "import time as _time_mod\n_time_mod.sleep = lambda *a, **k: None\n"
_ANSI = _re.compile(r"^\x1b\[([0-9;]+)m$")
_NUM = _re.compile(r"-?\d+(?:\.\d+)?")
_ENV = ("FileNotFoundError", "the following arguments are required", "EOFError", "SystemExit: 1", "SystemExit: 2",
        "IsADirectoryError", "PermissionError")


def clog(*a):
    print(f"[{_time.time() - C_T0:7.0f}s]", *a, flush=True)


def contains(out, want):
    m = _ANSI.match(want)
    if m:  # \x1b[31m also matches \x1b[1;31m or \x1b[0;31;40m
        codes = [c for c in m.group(1).split(";") if c not in ("0", "1")]
        return all(_re.search(rf"\x1b\[(?:[0-9]+;)*{c}(?:;[0-9]+)*m", out) for c in codes)
    if want in out or " ".join(want.split()) in " ".join(out.split()):
        return True
    if _NUM.fullmatch(want.strip()):
        return any(abs(float(x) - float(want)) < 1e-9 for x in _NUM.findall(out))
    return False


def failed(ff):
    """-> (ok, why) for a run that didn't exit cleanly."""
    err = ff.get("stderr") or ""
    if ff.get("kind") == "runtime_error" and "ModuleNotFoundError" not in err and any(e in err for e in _ENV):
        return True, "unchecked_env"
    return False, ff.get("kind", "pass?") + ": " + err[-300:]


def run_check(code, check):
    """-> (ok, why)"""
    kind = (check or {}).get("kind", "none")
    if kind == "program" and isinstance(check.get("expect"), list):
        stdin = check.get("stdin") or ""
        stdin = stdin if stdin.endswith("\n") or not stdin else stdin + "\n"
        r = judge(PRELUDE + code, [{"input": stdin, "output": SENTINEL}], per_test_s=6, guard=True, drop=C_DROP)
        ff = r.get("first_fail") or {}
        if ff.get("kind") != "wrong_answer":
            return failed(ff)
        out = ff.get("got") or ""
        missing = [w for w in check["expect"] if isinstance(w, str) and w and not contains(out, w)]
        return (not missing), ("missing " + _json.dumps(missing)[:300] if missing else "ok")
    lines = check.get("code") if kind == "asserts" else None
    lines = "\n".join(x for x in lines if isinstance(x, str)) if isinstance(lines, list) else lines
    if isinstance(lines, str) and "assert" in lines:
        r = judge(PRELUDE + code + "\n\n" + lines + "\n", [{"input": "", "output": SENTINEL}], per_test_s=6,
                  guard=True, drop=C_DROP)
        ff = r.get("first_fail") or {}
        if ff.get("kind") != "wrong_answer":
            return False, ff.get("kind", "pass?") + ": " + (ff.get("stderr") or "")[-300:]
        return True, "ok"
    try:
        compile(code, "answer.py", "exec")
        return True, "unchecked"
    except (SyntaxError, ValueError):
        return False, "syntax_error"


def load(name):
    hits = sorted(_glob.glob(_os.path.join(C_SRC, "**", "chat", name), recursive=True))
    return [_json.loads(x) for x in open(hits[0])] if hits else []


def main():
    _os.makedirs(C_OUT, exist_ok=True)
    for f in _glob.glob(_os.path.join(C_SRC, "**", "chat", "*.jsonl"), recursive=True):
        if _os.path.basename(f) != "checked.jsonl":
            _shutil.copy(f, C_OUT)
    answers, followups, complaints, ophyx = (load(n) for n in ("answers.jsonl", "followups.jsonl", "complaints.jsonl",
                                                                "ophyx.jsonl"))
    check_of = {r["id"]: r.get("check") for r in answers}
    jobs = [("answers", r["id"], 0, r["answer"], r.get("check")) for r in answers]
    jobs += [("followups", r["id"], 0, r["turns"][-1][1], r.get("check")) for r in followups]
    jobs += [("complaints", r["id"], 0, r["turns"][-1][1], check_of.get(r["base"])) for r in complaints]
    jobs += [("ophyx", r["id"], i, a, check_of.get(r["id"])) for r in ophyx for i, a in enumerate(r["answers"])]
    for f in sorted(_glob.glob(_os.path.join(C_SRC, "**", "chat", "samples-*.jsonl"), recursive=True)):
        name = _os.path.basename(f)[:-6]  # eval/chat_gen.py output: the check travels with each row
        rows_f = [_json.loads(x) for x in open(f)]
        jobs += [(name, r["id"], i, a, r.get("check")) for r in rows_f for i, a in enumerate(r["answers"])]
    clog(f"{len(jobs)} runs: {len(answers)} answers, {len(followups)} follow-ups, {len(complaints)} complaints, "
         f"{len(ophyx)} ophyx requests")

    def one(j):
        f, i, k, code, check = j
        try:
            ok, why = run_check(code, check)
        except Exception as e:  # noqa: BLE001 - one broken row must not stop the job
            ok, why = False, f"judge_error: {e}"[:300]
        return {"file": f, "id": i, "i": k, "ok": ok, "why": why, "kind": (check or {}).get("kind", "none")}

    rows, stats = [], {}
    with _Pool(CONFIG.get("workers", 4)) as pool:
        for n, r in enumerate(pool.map(one, jobs)):
            rows.append(r)
            s = stats.setdefault(r["file"], {"runs": 0, "ok": 0, "unchecked": 0})
            s["runs"] += 1
            s["ok"] += r["ok"]
            s["unchecked"] += r["why"] == "unchecked"
            if n % 1000 == 0:
                clog(f"{n}/{len(jobs)}")
    with open(_os.path.join(C_OUT, "checked.jsonl"), "w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")
    _json.dump(stats, open(_os.path.join(C_OUT, "check_stats.json"), "w"), indent=1)
    clog(_json.dumps(stats))


main()

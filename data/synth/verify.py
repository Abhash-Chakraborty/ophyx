"""Execute synthetic samples in a sandbox and keep only the ones that pass. Kaggle CPU, offline, no keys.

Attach the `pycoder-synth-raw` dataset. Each sample runs in its own temporary directory as a
subprocess with CPU, memory, file-size and process limits, a wall-clock timeout, an empty
environment, and no network (the session has internet off and no secrets attached).

  textbook          `python module.py` must exit 0 (its asserts check its own claims)
  exercise/design/  `pytest module.py` must pass with at least 4 tests and no failures
  errors

Writes /kaggle/working/verified/<batch>.jsonl and a stats file. Restart-safe per batch.
"""

import concurrent.futures as cf
import glob
import gzip
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
SHARD, NSHARDS = CONFIG.get("shard", 0), CONFIG.get("nshards", 1)
SRC = CONFIG.get("src", "/kaggle/input")
OUT = os.path.join(CONFIG.get("out", "/kaggle/working"), "verified")
os.makedirs(OUT, exist_ok=True)
WORKERS = CONFIG.get("workers", (os.cpu_count() or 4) * 2)
T0 = time.time()
PRLIMIT = shutil.which("prlimit")
BATCH_RE = re.compile(r"(batch-.+?)(\.jsonl)?(\.gz)?(\.tmp)?$")  # any batch name, minus the extensions
PASSED_RE = re.compile(r"(\d+) passed")
FAILED_RE = re.compile(r"(\d+) (failed|error)")


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def limits(cpu_s):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 2))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    return apply


def run(code, kind):
    d = tempfile.mkdtemp(prefix="sx-")
    try:
        with open(os.path.join(d, "mod.py"), "w") as f:
            f.write(code)
        if kind == "textbook":
            cmd, timeout = [sys.executable, "-I", "mod.py"], 15
        else:
            cmd, timeout = [sys.executable, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "-o", "addopts=", "mod.py"], 40
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "TMPDIR": d, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        t = time.time()
        if PRLIMIT:  # preexec_fn is not safe with many threads; prlimit sets the limits from outside
            cmd = [PRLIMIT, f"--cpu={timeout}", f"--as={2 << 30}", f"--fsize={50 << 20}", "--nofile=256", "--"] + cmd
            extra = {"start_new_session": True}
        else:
            extra = {"preexec_fn": limits(timeout)}
        p = subprocess.Popen(cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, **extra)
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return {"ok": False, "why": "timeout"}
        out = out.decode(errors="replace")[-3000:]
        sec = round(time.time() - t, 2)
        if kind == "textbook":
            return {"ok": p.returncode == 0, "why": "ok" if p.returncode == 0 else "nonzero_exit", "sec": sec,
                    "tail": "" if p.returncode == 0 else out[-600:]}
        passed = int(m.group(1)) if (m := PASSED_RE.search(out)) else 0
        failed = bool(FAILED_RE.search(out))
        ok = p.returncode == 0 and passed >= 4 and not failed
        why = "ok" if ok else ("few_tests" if p.returncode == 0 and not failed else "tests_failed")
        return {"ok": ok, "why": why, "passed": passed, "sec": sec, "tail": "" if ok else out[-600:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def batch_files(root):
    """Batch files by name, whatever Kaggle did to them (it decompresses .gz uploads into
    batch-N.jsonl/batch-N.jsonl.gz.tmp). Returns {batch name: path}."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "batch-*"), recursive=True):
        m = BATCH_RE.match(os.path.basename(p))
        if m and os.path.isfile(p) and not p.endswith(".stats.json"):
            out.setdefault(m.group(1), p)
    return dict(sorted(out.items()))


def read_jsonl(path):
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    fh = gzip.open(path, "rt") if gz else open(path)
    with fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    try:
        import pytest  # noqa: F401
        log("pytest", pytest.__version__)
    except ImportError:
        raise SystemExit("pytest is not installed in this image")
    found = batch_files(SRC)
    files = [(n, f) for i, (n, f) in enumerate(found.items()) if i % NSHARDS == SHARD]
    log(f"{len(files)} batch files for shard {SHARD}/{NSHARDS}, {WORKERS} workers")
    totals = {}
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for name, path in files:
            dest = os.path.join(OUT, name + ".jsonl")
            if os.path.exists(dest):
                continue
            recs = read_jsonl(path)
            t = time.time()
            results = list(ex.map(lambda r: run(r["code"], r["kind"]), recs))
            stats = {}
            with open(dest + ".tmp", "w") as f:
                for r, res in zip(recs, results):
                    key = f"{r['kind']}/{res['why']}"
                    stats[key] = stats.get(key, 0) + 1
                    if res["ok"]:
                        f.write(json.dumps({**r, "verify": {k: res.get(k) for k in ("passed", "sec")}}) + "\n")
            os.replace(dest + ".tmp", dest)
            fails = [(r["kind"], r["leaf"], res["why"], res.get("tail", "")[-300:]) for r, res in zip(recs, results) if not res["ok"]][:20]
            json.dump({"stats": stats, "sample_failures": fails}, open(os.path.join(OUT, name + ".stats.json"), "w"), indent=1)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v
            ok = sum(v for k, v in stats.items() if k.endswith("/ok"))
            log(f"{name}: {ok}/{len(recs)} passed in {time.time() - t:.0f}s")
    json.dump(totals, open(os.path.join(OUT, f"totals-{SHARD}.json"), "w"), indent=1)
    by_kind = {}
    for k, v in totals.items():
        kind, why = k.split("/")
        by_kind.setdefault(kind, {})[why] = v
    log("TOTALS", json.dumps(by_kind))


if __name__ == "__main__" and not CONFIG.get("library"):
    main()

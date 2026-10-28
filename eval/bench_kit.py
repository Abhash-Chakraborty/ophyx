"""Benchmark kit: download every public test set once, in the shape bench_gen.py and bench_exec.py read.

Runs as a Kaggle CPU kernel with internet and no keys (it downloads and reshapes text; it runs no
code from the benchmarks). Its output is attached to the generation kernels (prompts) and to the
offline execution kernels (tests, plus the pip wheels EvalPlus needs, since those run with no
internet).

  kit/humaneval.jsonl, kit/mbpp.jsonl   EvalPlus problems (prompt, entry_point) for prompting
  kit/evalplus_cache/                   EvalPlus's own data files, copied into ~/.cache/evalplus offline
  kit/cruxeval.jsonl                    800 functions with an input and an output
  kit/lcb.jsonl                         LiveCodeBench problems from test5 + test6 (Oct 2024 to Apr 2025)
  kit/lcb_tests/<qid>.json              their public + private tests
  kit/bcb_hard.jsonl                    BigCodeBench-Hard v0.1.4 (148 tasks)
  kit/bcb_full.jsonl                    BigCodeBench v0.1.4, every task (1,140), same fields
  kit/mple_{js,cpp,sh}.jsonl            MultiPL-E HumanEval in JavaScript, C++ and Bash (prompt, tests,
                                        stop_tokens); smoke_ref is the first passing StarCoder2-15B or
                                        DeepSeek-Coder-7B completion from nuprl/MultiPL-E-completions, kept
                                        only to smoke-test the scorer, since MultiPL-E has no reference solutions
  kit/ds1000.jsonl                      DS-1000 (1,000 problems: prompt, reference_code, code_context, metadata)
  kit/wheels/                           evalplus and its pure-Python dependencies
  kit/manifest.json                     counts and versions

CONFIG: out (default /kaggle/working/kit), limit (rows per benchmark, for a local smoke test),
lcb_files (default ["test5.jsonl", "test6.jsonl"]), skip (list of benchmark names).
"""

import base64
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zlib

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
OUT = CONFIG.get("out", "/kaggle/working/kit" if os.path.isdir("/kaggle") else os.path.abspath("kit"))
LIMIT = CONFIG.get("limit", 0)
SKIP = set(CONFIG.get("skip", []))
EVALPLUS_DEPS = ["wget", "tempdir", "multipledispatch", "appdirs", "termcolor", "fire", "tree-sitter",
                 "tree-sitter-python", "rich", "psutil", "stop-sequencer"]
T0 = time.time()
os.makedirs(OUT, exist_ok=True)
manifest = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def write_jsonl(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest[name] = len(rows)
    log(f"{name}: {len(rows)} rows")


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def evalplus():
    pip("install", "-q", "--no-deps", "evalplus==0.3.1")
    pip("install", "-q", *EVALPLUS_DEPS)
    wheels = os.path.join(OUT, "wheels")
    os.makedirs(wheels, exist_ok=True)
    pip("download", "-q", "--no-deps", "-d", wheels, "evalplus==0.3.1", *EVALPLUS_DEPS)
    manifest["wheels"] = sorted(os.listdir(wheels))
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    he, mb = get_human_eval_plus(), get_mbpp_plus()
    cache = os.path.join(OUT, "evalplus_cache")
    os.makedirs(cache, exist_ok=True)
    for f in glob.glob(os.path.expanduser("~/.cache/evalplus/*.jsonl")):
        shutil.copy(f, cache)
    manifest["evalplus_cache"] = sorted(os.listdir(cache))
    for name, probs in (("humaneval.jsonl", he), ("mbpp.jsonl", mb)):
        rows = [{"task_id": k, "prompt": v["prompt"], "entry_point": v["entry_point"]} for k, v in probs.items()]
        write_jsonl(name, rows[:LIMIT] if LIMIT else rows)


def hf_lines(repo, path):
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"datasets/{repo}/{path}", "r") as f:
        for i, line in enumerate(f):
            if LIMIT and i >= LIMIT:
                return
            if line.strip():
                yield json.loads(line)


def cruxeval():
    rows = [{"id": r["id"], "code": r["code"], "input": r["input"], "output": r["output"]}
            for r in hf_lines("cruxeval-org/cruxeval", "test.jsonl")]
    write_jsonl("cruxeval.jsonl", rows)


def decode_private(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def lcb():
    tdir = os.path.join(OUT, "lcb_tests")
    os.makedirs(tdir, exist_ok=True)
    rows = []
    for fname in CONFIG.get("lcb_files", ["test5.jsonl", "test6.jsonl"]):
        for r in hf_lines("livecodebench/code_generation_lite", fname):
            meta = json.loads(r.get("metadata") or "{}")
            tests = json.loads(r["public_test_cases"]) + decode_private(r["private_test_cases"])
            qid = f"{r['platform']}_{r['question_id']}"
            with open(os.path.join(tdir, f"{qid}.json"), "w") as f:
                json.dump([{"input": t["input"], "output": t["output"]} for t in tests], f)
            rows.append({"task_id": qid, "title": r["question_title"], "question": r["question_content"],
                         "platform": r["platform"], "date": r["contest_date"][:10], "difficulty": r["difficulty"],
                         "starter_code": r["starter_code"], "fn_name": meta.get("func_name"),
                         "testtype": tests[0]["testtype"] if tests else "stdin", "n_tests": len(tests),
                         "source_file": fname})
    rows.sort(key=lambda r: (r["date"], r["task_id"]))
    manifest["lcb_window"] = [rows[0]["date"], rows[-1]["date"]] if rows else None
    write_jsonl("lcb.jsonl", rows)


def hf_parquet(repo, path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset")).to_dict("records")


def bcb(repo, name):
    keep = ["task_id", "complete_prompt", "instruct_prompt", "canonical_solution", "code_prompt", "test",
            "entry_point", "libs"]
    rows = [{k: (r[k] if k != "libs" else str(r[k])) for k in keep}
            for r in hf_parquet(repo, "data/v0.1.4-00000-of-00001.parquet")]
    write_jsonl(name, rows[:LIMIT] if LIMIT else rows)


def bcb_hard():
    bcb("bigcode/bigcodebench-hard", "bcb_hard.jsonl")


def bcb_full():
    bcb("bigcode/bigcodebench", "bcb_full.jsonl")


MPLE_LANGS = ["js", "cpp", "sh"]
MPLE_REFS = ["starcoder2_15b", "deepseekcoder1.5_7b_base"]  # base-model runs in nuprl/MultiPL-E-completions


def mple():
    for lang in MPLE_LANGS:
        refs = {}
        for model in MPLE_REFS:
            exp = f"humaneval.{lang}.{model}.0.2.reworded"
            for r in hf_parquet("nuprl/MultiPL-E-completions", f"{exp}/test-00000-of-00001.parquet"):
                ok = [c for c, st in zip(r["completions"], r["statuses"]) if st == "OK"]
                if ok and r["problem"] not in refs:
                    refs[r["problem"]] = ok[0]
        rows = [{"task_id": r["name"], "language": r["language"], "prompt": r["prompt"], "tests": r["tests"],
                 "stop_tokens": list(r["stop_tokens"]), "smoke_ref": refs.get(r["name"])}
                for r in hf_parquet("nuprl/MultiPL-E", f"humaneval-{lang}/test-00000-of-00001.parquet")]
        manifest[f"mple_{lang}_smoke_refs"] = sum(r["smoke_ref"] is not None for r in rows)
        write_jsonl(f"mple_{lang}.jsonl", rows[:LIMIT] if LIMIT else rows)


def ds1000():
    rows = [{"task_id": str(r["metadata"]["problem_id"]), "prompt": r["prompt"], "reference_code": r["reference_code"],
             "code_context": r["code_context"], "library": r["metadata"]["library"],
             "perturbation": r["metadata"]["perturbation_type"]}
            for r in hf_lines("xlangai/DS-1000", "test.jsonl")]
    write_jsonl("ds1000.jsonl", rows)


for name, fn in (("evalplus", evalplus), ("cruxeval", cruxeval), ("lcb", lcb), ("bcb_hard", bcb_hard),
                 ("bcb_full", bcb_full), ("mple", mple), ("ds1000", ds1000)):
    if name in SKIP:
        continue
    t = time.time()
    fn()
    log(f"{name} done in {time.time() - t:.0f}s")
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
log("KIT DONE", json.dumps({k: v for k, v in manifest.items() if isinstance(v, (int, list)) and k != "wheels"}))

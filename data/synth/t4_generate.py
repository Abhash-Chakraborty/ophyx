"""Production synthetic generation on Kaggle 2x T4 (appended after data/synth/generate.py and
data/synth/verify.py by the launcher). No keys are attached to these sessions: the teacher is a
public model, and generated code runs in the same session, so there is nothing to leak.

Per chunk of specs:
  1. generate with Qwen2.5-Coder-14B-Instruct-AWQ (vLLM, tensor parallel over both T4s)
  2. shape check, then execute in the sandbox (verify.run)
  3. failures get one repair round: the teacher sees its module and the error output and returns a
     fixed module, which is executed again
  4. passing samples go to /kaggle/working/verified/batch-t4-<start>.jsonl; every repaired one also
     yields a bug-fix pair (broken code, error, fixed code) for the "bugfix" bucket

CONFIG: start (first spec index), count, chunk, budget_s, model, tp.
"""

import concurrent.futures as _cf
import subprocess as _sp
import sys

_sp.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
_sp.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)

from vllm import LLM, SamplingParams  # noqa: E402

T_BEGIN = time.time()
START = CONFIG.get("start", 1_000_000)
COUNT = CONFIG.get("count", 20_000)
CHUNK = CONFIG.get("chunk", 1000)
BUDGET = CONFIG.get("budget_s", 11 * 3600)
MODEL = CONFIG.get("model", "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
TAG = MODEL.split("/")[-1] + "@kaggle-t4"
VDIR = "/kaggle/working/verified"
os.makedirs(VDIR, exist_ok=True)

REPAIR = ("Your module failed when it was run. Here is the module, then the error output. Fix the module so that it "
          "runs and every test passes, keeping the same purpose, names and style. If a test's expected value was "
          "wrong, fix the test. Output the complete corrected module and nothing else.")

llm = LLM(model=MODEL, tensor_parallel_size=CONFIG.get("tp", 2), dtype="float16", max_model_len=8192,
          gpu_memory_utilization=0.92)
pool = _cf.ThreadPoolExecutor((os.cpu_count() or 4) * 2)
totals = {}


def bump(k, n=1):
    totals[k] = totals.get(k, 0) + n


def chat(convs, temps):
    params = [SamplingParams(temperature=t, top_p=0.95, max_tokens=6000) for t in temps]
    return [o.outputs[0] for o in llm.chat(convs, params, use_tqdm=False)]


for c0 in range(START, START + COUNT, CHUNK):
    if time.time() - T_BEGIN > BUDGET - 1800:
        print("budget nearly spent; stopping", flush=True)
        break
    dest = os.path.join(VDIR, f"batch-t4-{c0:07d}.jsonl")
    if os.path.exists(dest):
        continue
    t = time.time()
    specs = [spec(i) for i in range(c0, min(c0 + CHUNK, START + COUNT))]
    for sp in specs:
        sp["teacher"] = TAG
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(sp)}] for sp in specs]
    outs = chat(convs, [sp["temperature"] for sp in specs])
    cand = []
    for sp, conv, o in zip(specs, convs, outs):
        code = extract_code(o.text)
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            bump("no_parse")
            continue
        if o.finish_reason == "length" or not shape_ok(code, sp["kind"]):
            bump("bad_shape_or_length")
            continue
        cand.append((sp, conv, code))
    res = list(pool.map(lambda x: run(x[2], x[0]["kind"]), cand))
    kept, bugfix, retry = [], [], []
    for (sp, conv, code), r in zip(cand, res):
        if r["ok"]:
            kept.append({**sp, "code": code, "repaired": False, "verify": {"passed": r.get("passed")}})
        elif r["why"] in ("tests_failed", "nonzero_exit") and r.get("tail"):
            retry.append((sp, conv, code, r["tail"]))
        else:
            bump(f"fail_{r['why']}")
    bump("first_pass_ok", len(kept))
    if retry:
        rconvs = [conv + [{"role": "assistant", "content": code},
                          {"role": "user", "content": f"{REPAIR}\n\nError output:\n{tail[-1500:]}"}]
                  for sp, conv, code, tail in retry]
        routs = chat(rconvs, [0.4] * len(retry))
        fixed = []
        for (sp, conv, code, tail), o in zip(retry, routs):
            new = extract_code(o.text)
            try:
                ast.parse(new)
                if shape_ok(new, sp["kind"]):
                    fixed.append((sp, code, tail, new))
                    continue
            except (SyntaxError, ValueError, RecursionError):
                pass
            bump("repair_unusable")
        rres = list(pool.map(lambda x: run(x[3], x[0]["kind"]), fixed))
        for (sp, code, tail, new), r in zip(fixed, rres):
            if r["ok"]:
                kept.append({**sp, "code": new, "repaired": True, "verify": {"passed": r.get("passed")}})
                bugfix.append({**sp, "kind": "bugfix", "broken": code, "error": tail[-1500:], "code": new})
            else:
                bump("repair_failed")
        bump("repaired_ok", sum(1 for r in rres if r["ok"]))
    with open(dest + ".tmp", "w") as f:
        for r in kept:
            r["id"] = hashlib.sha1(f"t4-{TAG}-{r['i']}".encode()).hexdigest()[:16]
            r["prompt_version"] = PROMPT_VERSION
            f.write(json.dumps(r) + "\n")
    os.replace(dest + ".tmp", dest)
    with open(os.path.join(VDIR, f"bugfix-t4-{c0:07d}.jsonl"), "w") as f:
        for r in bugfix:
            f.write(json.dumps(r) + "\n")
    bump("generated", len(specs))
    bump("kept", len(kept))
    json.dump(totals, open(os.path.join(VDIR, "t4_totals.json"), "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] chunk {c0}: {len(specs)} -> {len(kept)} verified "
          f"({len(bugfix)} after repair) in {time.time() - t:.0f}s; totals {totals}", flush=True)
print("TOTALS", json.dumps(totals), flush=True)

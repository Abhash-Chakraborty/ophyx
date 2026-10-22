"""Kaggle Benchmarks task: frontier and open API models answer ophyx's benchmark prompts.

Pushed with `kaggle b t push <name> -f frontier_task.py -d abhashchakraborty/pycoder-frontier-prompts`,
which runs it once on Kaggle's infrastructure (the model proxy and the AI quota only work there).
It writes every reply to /kaggle/working/<model>/<bench>.jsonl; `kaggle b t download` brings them
back, eval/frontier_collect.py turns them into bench_gen-style samples, and bench_exec.py scores
them offline like every other model. Nothing is executed here.

Edit the constants before a push: MODELS (CLI slugs from `kaggle b t models`), BENCHES (priority order:
a budget stop cuts the last ones), BUDGET_USD (this run stops sending new requests once the
proxy-reported cost reaches it), ONLY (a top-up of the replies an earlier run missed), EXTRA (output tokens
added for every request, for re-asking replies that stopped at the length limit), WORKERS.
Reasoning models get reasoning_effort "low" (OpenAI and xAI) and 4,096 extra output tokens for their thinking;
Gemma 4 and Claude Haiku are listed too, since they used the whole budget thinking and returned no text.
"""

import collections
import concurrent.futures as cf
import glob
import json
import os
import random
import threading
import time

import kaggle_benchmarks as kbench

MODELS = __MODELS__
BENCHES = __BENCHES__
BUDGET_USD = __BUDGET__
ONLY = __ONLY__  # None, or {slug: ["bench|task_id", ...]}: a top-up run that asks only these
EXTRA = __EXTRA__
WORKERS = 4
REASONING = ("gpt-5", "gpt-6", "gpt-oss", "gemma-4", "claude-haiku", "gemini-2.5", "gemini-3", "gemini-4", "deepseek-r1", "thinking", "grok", "glm-5")
ROOT = os.environ.get("OPHYX_FRONTIER_ROOT", "/kaggle")  # changed only by the local mock test
FALLBACK_PRICE = (5e-6, 25e-6)  # $/token in, out: used only when the proxy reports no cost


def norm(s):
    s = s.lower().split("/")[-1].replace("@", "-")
    return s[:-3] if s.endswith("-it") else s


@kbench.task(name="ophyx-frontier-run", description="Frontier and open API models on ophyx's benchmark prompts "
             "(EvalPlus, LiveCodeBench, BigCodeBench-Hard, CRUXEval, concept suite, hands-on tasks). Replies are "
             "saved for offline scoring. Author: Abhash Chakraborty.")
def frontier_run(llm) -> tuple[int, int]:
    from openai import OpenAI
    available = [m for m in os.environ.get("LLMS_AVAILABLE", "").split(",") if m]
    by_norm = {norm(m): m for m in available}
    ids = {}
    for slug in MODELS:
        mid = by_norm.get(norm(slug))
        print(f"model {slug} -> {mid}", flush=True)
        if mid:
            ids[slug] = mid
    files = glob.glob(f"{ROOT}/input/**/prompts.jsonl", recursive=True)
    prompts = sorted((r for r in map(json.loads, open(files[0])) if r["bench"] in BENCHES), key=lambda r: BENCHES.index(r["bench"]))
    base = os.environ["MODEL_PROXY_URL"].rstrip("/")
    for suffix in ("/openapi", "/genai"):  # the variable may already end in one; kaggle_benchmarks strips it the same way
        base = base[: -len(suffix)] if base.endswith(suffix) else base
    client = OpenAI(api_key=os.environ["MODEL_PROXY_API_KEY"], base_url=base + "/openapi", max_retries=0, timeout=300)
    lock = threading.Lock()
    spent = [0.0]
    no_temp = set()
    stats = collections.Counter()
    outs = {}
    for slug in ids:
        os.makedirs(f"{ROOT}/working/{slug}", exist_ok=True)
        for b in BENCHES:
            outs[(slug, b)] = open(f"{ROOT}/working/{slug}/{b}.jsonl", "w")

    def call(slug, p):
        if spent[0] >= BUDGET_USD:
            return {"error": "budget"}
        mid = ids[slug]
        reasoning = any(k in slug for k in REASONING)
        kw = {"model": mid, "messages": p["messages"], "max_completion_tokens": p["max_tokens"] + (4096 if reasoning else 256) + EXTRA}
        if reasoning and ("gpt" in slug or "grok" in slug):
            kw["reasoning_effort"] = "low"
        if slug not in no_temp:
            kw["temperature"] = 0
        for attempt in range(6):
            try:
                r = client.chat.completions.create(**kw)
                u = r.usage
                cost = (getattr(u, "cost", None) or (u.model_extra or {}).get("cost") or {}) if u else {}
                nd = (cost.get("input_tokens_cost_nanodollars") or 0) + (cost.get("output_tokens_cost_nanodollars") or 0)
                usd = nd / 1e9 if nd else ((u.prompt_tokens or 0) * FALLBACK_PRICE[0] + (u.completion_tokens or 0) * FALLBACK_PRICE[1] if u else 0)
                with lock:
                    spent[0] += usd
                c = r.choices[0] if r.choices else None
                if c is None or c.message is None:  # an empty reply (seen from Gemma and Haiku): ask again, then give up
                    if attempt < 2:
                        continue
                    return {"error": f"no message (finish {getattr(c, 'finish_reason', None)})"}
                return {"response": c.message.content or "", "finish": c.finish_reason, "usd": round(usd, 6),
                        "in": u.prompt_tokens if u else None, "out": u.completion_tokens if u else None}
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"[:400]
                if "temperature" in msg.lower() and "temperature" in kw:
                    no_temp.add(slug)
                    kw.pop("temperature")
                    continue
                if "reasoning_effort" in msg and "reasoning_effort" in kw:
                    kw.pop("reasoning_effort")
                    continue
                if "429" in msg or "rate" in msg.lower() or "timeout" in msg.lower() or "503" in msg or "502" in msg:
                    time.sleep(min(60, 4 * 2 ** attempt) + random.random())
                    continue
                return {"error": msg}
        return {"error": "retries exhausted"}

    for slug in list(ids):  # one tiny call per model first, so a broken setup costs one request, not a whole run
        probe = call(slug, {"messages": [{"role": "user", "content": "Reply with the word ok."}], "max_tokens": 16})
        print(f"preflight {slug}: {probe.get('error') or 'ok'}", flush=True)
        if "error" in probe:
            ids.pop(slug)
    if not ids:
        json.dump({"error": "every model failed the preflight", "spent_usd": spent[0]}, open(f"{ROOT}/working/summary.json", "w"))
        return 0, 1
    jobs = [(slug, p) for p in prompts for slug in ids
            if ONLY is None or f"{p['bench']}|{p['task_id']}" in ONLY.get(slug, ())]  # interleaved, so one model's rate limit doesn't stall the rest
    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(call, slug, p): (slug, p) for slug, p in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            slug, p = futs[fut]
            res = fut.result()
            stats[(slug, "ok" if "response" in res else res["error"][:40])] += 1
            outs[(slug, p["bench"])].write(json.dumps({"task_id": p["task_id"], **res}) + "\n")
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} done, ${spent[0]:.2f} spent, {time.time() - t0:.0f}s", flush=True)
    for f in outs.values():
        f.close()
    summary = {"spent_usd": round(spent[0], 4), "models": ids, "benches": BENCHES, "counts": {f"{k[0]}|{k[1]}": v for k, v in stats.items()},
               "seconds": round(time.time() - t0)}
    json.dump(summary, open(f"{ROOT}/working/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    ok = sum(v for k, v in stats.items() if k[1] == "ok")
    return ok, len(jobs)


frontier_run.run(kbench.llm)

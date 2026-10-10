"""Local teacher on Kaggle 2x T4 with vLLM (appended after data/synth/generate.py by the launcher).

Phase 1 (this version): measure throughput and output quality for candidate teachers, so we can
decide how much synthetic data the T4 quota can buy. Outputs samples in the same batch format as
the API generator, for the offline verifier.
"""

import subprocess as _sp

T_START = time.time()
OUTD = "/kaggle/working"
import sys
_sp.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
import sys  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

N = CONFIG.get("n_per_model", 96)
results = {}
for model_id, tp in CONFIG.get("models", [["Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", 1], ["Qwen/Qwen2.5-Coder-14B-Instruct-AWQ", 2]]):
    tag = model_id.split("/")[-1]
    try:
        t = time.time()
        llm = LLM(model=model_id, tensor_parallel_size=tp, dtype="float16", max_model_len=8192,
                  gpu_memory_utilization=0.90, enforce_eager=False)
        load_s = time.time() - t
        specs = [spec(500_000 + i) for i in range(N)]
        for sp in specs:
            sp["teacher"] = f"{tag}@kaggle-t4"
        msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(sp)}] for sp in specs]
        params = [SamplingParams(temperature=sp["temperature"], top_p=0.95, max_tokens=6000) for sp in specs]
        t = time.time()
        outs = llm.chat(msgs, params, use_tqdm=False)
        gen_s = time.time() - t
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        stats = {"ok": 0, "no_parse": 0, "truncated": 0, "bad_shape": 0}
        recs = []
        for sp, o in zip(specs, outs):
            code = extract_code(o.outputs[0].text)
            if o.outputs[0].finish_reason == "length":
                stats["truncated"] += 1
                continue
            try:
                ast.parse(code)
            except (SyntaxError, ValueError, RecursionError):
                stats["no_parse"] += 1
                continue
            if not shape_ok(code, sp["kind"]):
                stats["bad_shape"] += 1
                continue
            stats["ok"] += 1
            recs.append({"id": hashlib.sha1(f"t4-{tag}-{sp['i']}".encode()).hexdigest()[:16], **sp,
                         "prompt_version": PROMPT_VERSION, "code": code})
        with open(os.path.join(OUTD, f"batch-{tag}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        results[tag] = {"load_s": round(load_s), "gen_s": round(gen_s), "tokens": n_tok,
                        "tokens_per_s": round(n_tok / gen_s), "M_tokens_per_gpu_hour": round(n_tok / gen_s * 3600 / 1e6, 1),
                        **stats}
        print(tag, results[tag], flush=True)
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc()
        results[tag] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    json.dump(results, open(os.path.join(OUTD, "t4_teacher.json"), "w"), indent=1)
print("RESULTS", json.dumps(results), flush=True)

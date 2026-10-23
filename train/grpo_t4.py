"""GRPO on the problem bank with test-based rewards. Kaggle T4 x2, internet off, no keys.

The reward runs model-written programs, so this session never has keys or internet. Everything it
needs comes in as kernel sources:
  relay_in.py output     model/<dir>/ (the starting checkpoint) and wheels/ (TRL and its deps)
  verify-bank outputs    verified/bank-*.jsonl (problems and tests)
  bank-judge outputs     judged/<name>/bank-*.jsonl (the starting model's pass rate per problem), optional
Prepended by the launcher: data/jobs/judge.py and data/jobs/verify_bank.py (CONFIG "library": true),
for judge() and the bank prompt wording, so prompts match the SFT data exactly.

Problems: consistent bank problems whose sampled pass rate lies in `band` (default 1/8 to 6/8: the
model sometimes solves them, so a group of samples has a learning signal), minus `holdout` problems
kept for the before/after check. The reward is 1 when every test passes and 0 otherwise. There is
no partial credit, because partial credit pays for special-casing the visible tests.

CONFIG: model_dir (default: the first model/* in the inputs), judged_name, band, holdout (200),
budget_s (per run, default 9.5 h), runs: a list of overrides, one per GPU, run in parallel by forked
processes, e.g. [{"lr": 1e-6}, {"lr": 3e-6}]. Keys a run may set: lr, beta, num_generations,
prompts_per_step, max_completion, temperature, seed, name.

Output under /kaggle/working/grpo/<name>/: final/ (fp32 safetensors + tokenizer), log.jsonl (TRL
logs), holdout.json (greedy pass@1 on the holdout problems, this model vs the starting model), meta.json.
"""

import glob as _glob
import json as _json
import multiprocessing as _mp
import os as _os
import random as _random
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import time as _time

G_BUDGET = CONFIG.get("budget_s", int(9.5 * 3600))
G_BAND = CONFIG.get("band", [0.125, 0.75])
G_HOLDOUT = CONFIG.get("holdout", 200)
G_OUT = _os.path.join(OUTW, "grpo")
G_PER_TEST = CONFIG.get("per_test_s", 2)
G_MAX_PROMPT = CONFIG.get("max_prompt", 768)
DEFAULT_RUN = {"lr": 1e-6, "beta": 0.0, "num_generations": 8, "prompts_per_step": 8, "max_completion": 512,
               "temperature": 0.8, "seed": 0}


def g_model_dir():
    if CONFIG.get("model_dir"):
        return CONFIG["model_dir"]
    hits = sorted(_glob.glob(_os.path.join(SRC, "**", "model", "*", "config.json"), recursive=True))
    if not hits:
        raise SystemExit("no model/*/config.json in the inputs: attach the relay_in output")
    return _os.path.dirname(hits[0])


def g_install():
    wheels = sorted(_glob.glob(_os.path.join(SRC, "**", "wheels"), recursive=True))
    if not wheels:
        return
    rel = _glob.glob(_os.path.join(SRC, "**", "relay.json"), recursive=True)
    reqs = _json.load(open(rel[0])).get("requirements", ["trl"]) if rel else ["trl"]
    r = _subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheels[0], *reqs],
                        capture_output=True, text=True)
    log("pip offline install:", r.returncode, (r.stdout + r.stderr)[-1500:])


def g_problems():
    files = sorted(_glob.glob(_os.path.join(SRC, "**", "verified", "bank-*.jsonl"), recursive=True))
    recs = [r for f in files for r in map(_json.loads, open(f)) if not r.get("inconsistent")]
    rates = {}
    name = CONFIG.get("judged_name", "ophyx-350m-sft")
    files = _glob.glob(_os.path.join(SRC, "**", "judged", name, "bank-*.jsonl"), recursive=True)
    if not files:  # outputs judged before the rename sit under judged/pycoder-base350m-sft/
        names = {f.split(_os.sep)[-2] for f in _glob.glob(_os.path.join(SRC, "**", "judged", "*", "bank-*.jsonl"), recursive=True)}
        if len(names) > 1:
            raise SystemExit(f"no judged/{name}/ but several others {sorted(names)}: set judged_name")
        if names:
            name = names.pop()
            log(f"judged_name: using judged/{name}/")
            files = _glob.glob(_os.path.join(SRC, "**", "judged", name, "bank-*.jsonl"), recursive=True)
        elif not CONFIG.get("allow_no_rates"):  # without rates there is no band, and most groups carry no signal
            raise SystemExit("no judged/*/bank-*.jsonl in the inputs: attach the bank-judge outputs or set allow_no_rates")
    for f in files:
        for r in map(_json.loads, open(f)):
            rates[r["pid"]] = (r["passed"] / max(1, r["n"]), r.get("greedy_pass"))
    rng = _random.Random(0)
    rng.shuffle(recs)
    if rates:
        recs = [r for r in recs if r["pid"] in rates]
        hold = recs[:G_HOLDOUT]
        train = [r for r in recs[G_HOLDOUT:] if G_BAND[0] <= rates[r["pid"]][0] <= G_BAND[1]]
    else:
        hold, train = recs[:G_HOLDOUT], recs[G_HOLDOUT:]
    log(f"{len(recs)} problems, {len(rates)} with pass rates; train {len(train)} in band {G_BAND}, holdout {len(hold)}")
    return train, hold, rates


def g_prompt(rec):
    return f"<|user|>\n{sft_row(rec)['instruction']}\n<|python|>\n"


def g_clean(text):
    t = text.split("<|end|>")[0]
    t = t.strip("\n")
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if "```" in t:
            t = t[:t.index("```")]
    return t.rstrip() + "\n"


def g_rewards(recs_by_pid, pids, completions):
    import concurrent.futures as cf

    def one(args):
        pid, c = args
        code = g_clean(c if isinstance(c, str) else c[0]["content"])
        if not code.strip():
            return 0.0
        rec = recs_by_pid[pid]
        tests = sorted(rec["tests"], key=lambda t: len(t["input"]) + len(t["output"]))
        # 512 MB and nobody: two runs judge in parallel next to two trainers, and a pilot container died silently
        return 1.0 if judge(code, tests, rec["fn_name"], per_test_s=G_PER_TEST, guard=True, mem=512 << 20,
                            drop=True)["status"] == "pass" else 0.0

    with cf.ThreadPoolExecutor(max(2, (_os.cpu_count() or 4))) as ex:
        return list(ex.map(one, zip(pids, completions)))


def g_low_resources():
    """Why to stop now (host memory or /tmp nearly full), or None."""
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except ImportError:
        avail = None
    free = _shutil.disk_usage("/tmp").free
    if avail is not None and avail < 2 << 30:
        return f"host memory {avail / 2**30:.1f} GB free"
    if free < 3 << 30:
        return f"/tmp {free / 2**30:.1f} GB free"
    return None


def g_greedy_pass(model, tok, recs, end_id, max_new):
    import torch
    out = {}
    model.eval()
    tok.padding_side = "left"
    for b in range(0, len(recs), 16):
        chunk = recs[b:b + 16]
        enc = tok([g_prompt(r) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad(), torch.autocast(model.device.type, dtype=torch.float16, enabled=model.device.type == "cuda"):
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=end_id, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        for r, ok in zip(chunk, g_rewards({r["pid"]: r for r in chunk}, [r["pid"] for r in chunk], texts)):
            out[r["pid"]] = ok
    return out


def g_run(spec, gpu, train, hold, rates, logf):
    fd = _os.open(logf, _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC)
    _os.dup2(fd, 1)
    _os.dup2(fd, 2)
    _os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    _os.environ["JUDGE_UID_BASE"] = str(61000 + 1000 * int(gpu))  # each run reaps only its own sandbox uids
    code = 0
    try:
        g_train(spec, train, hold, rates)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(code)


def g_train(spec, train, hold, rates):
    import dataclasses

    import torch
    import transformers
    import trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

    t0 = _time.time()
    name = spec["name"]
    out = _os.path.join(G_OUT, name)
    _os.makedirs(out, exist_ok=True)
    print(f"{name}: trl {trl.__version__}, transformers {transformers.__version__}, torch {torch.__version__}, "
          f"cuda {torch.cuda.is_available()}", flush=True)
    mdir = g_model_dir()
    tok = AutoTokenizer.from_pretrained(mdir)
    tok.eos_token = "<|end|>"  # TRL masks everything after the first eos in a completion
    if tok.pad_token is None or tok.pad_token == tok.eos_token:
        tok.pad_token = "<|pad|>"
    end_id = tok.convert_tokens_to_ids("<|end|>")
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32)
    model.generation_config.eos_token_id = end_id
    model.generation_config.pad_token_id = tok.pad_token_id

    recs_by_pid = {r["pid"]: r for r in train + hold}
    rows = []
    for r in train:
        p = g_prompt(r)
        if len(tok(p, add_special_tokens=False)["input_ids"]) <= G_MAX_PROMPT:
            rows.append({"prompt": p, "pid": r["pid"]})
    ds = Dataset.from_list(rows).shuffle(seed=spec["seed"])
    print(f"{name}: {len(rows)} training prompts", flush=True)

    def reward_tests(prompts, completions, pid, **kw):
        return g_rewards(recs_by_pid, pid, completions)

    n_gen = spec["num_generations"]
    want = dict(output_dir=_os.path.join(out, "ckpt"), learning_rate=spec["lr"], beta=spec["beta"],
                num_generations=n_gen, per_device_train_batch_size=n_gen,
                gradient_accumulation_steps=spec["prompts_per_step"], generation_batch_size=n_gen * spec["prompts_per_step"],
                max_prompt_length=G_MAX_PROMPT, max_completion_length=spec["max_completion"],
                temperature=spec["temperature"], top_p=0.95, mask_truncated_completions=spec.get("mask_truncated", True),
                lr_scheduler_type="constant_with_warmup", warmup_steps=5, max_grad_norm=1.0, weight_decay=0.0,
                fp16=torch.cuda.is_available(), bf16=False, gradient_checkpointing=True, logging_steps=1, save_strategy="no",
                report_to="none", seed=spec["seed"], num_train_epochs=1, dataloader_num_workers=0,
                remove_unused_columns=False, log_completions=False)
    fields = {f.name for f in dataclasses.fields(trl.GRPOConfig)}
    dropped = sorted(k for k in want if k not in fields)
    args = trl.GRPOConfig(**{k: v for k, v in want.items() if k in fields})
    print(f"{name}: GRPOConfig ignores {dropped}", flush=True)

    logf = open(_os.path.join(out, "log.jsonl"), "w")

    class Budget(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            logf.write(_json.dumps({**(logs or {}), "step": state.global_step, "t": round(_time.time() - t0)}) + "\n")
            logf.flush()

        def on_step_end(self, a, state, control, **kw):
            if _time.time() - t0 > G_BUDGET - 1800:  # keep 30 min for the holdout check and the save
                control.should_training_stop = True
            low = g_low_resources()
            if low:  # stop and save while we still can: a killed container keeps no output at all
                print(f"{name}: stopping at step {state.global_step}: {low}", flush=True)
                control.should_training_stop = True
            if state.global_step % 10 == 0:
                print(f"{name}: step {state.global_step}, {(_time.time() - t0) / 60:.0f} min", flush=True)

    kw = {"processing_class": tok} if "processing_class" in trl.GRPOTrainer.__init__.__code__.co_varnames else {"tokenizer": tok}
    trainer = trl.GRPOTrainer(model=model, reward_funcs=[reward_tests], args=args, train_dataset=ds,
                              callbacks=[Budget()], **kw)
    trainer.train()
    logf.close()
    model = trainer.model
    model.config.use_cache = True
    final = _os.path.join(out, "final")
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    after = g_greedy_pass(model, tok, hold, end_id, spec["max_completion"])
    before = {r["pid"]: rates.get(r["pid"], (None, None))[1] for r in hold}
    both = [p for p in after if before.get(p) is not None]
    res = {"holdout": len(hold), "after_greedy_pass@1": sum(after.values()) / max(1, len(after)),
           "before_greedy_pass@1": (sum(bool(before[p]) for p in both) / len(both)) if both else None,
           "compared": len(both), "per_pid": {p: [before.get(p), after[p]] for p in after}}
    _json.dump(res, open(_os.path.join(out, "holdout.json"), "w"), indent=1)
    _json.dump({"spec": spec, "steps": trainer.state.global_step, "minutes": round((_time.time() - t0) / 60),
                "train_prompts": len(rows), "trl": trl.__version__, "ignored_config": dropped},
               open(_os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"{name}: done, holdout greedy {res['before_greedy_pass@1']} -> {res['after_greedy_pass@1']:.3f}", flush=True)


def g_main():
    g_install()
    train, hold, rates = g_problems()
    runs = [{**DEFAULT_RUN, **r} for r in CONFIG.get("runs", [{}])]
    for i, r in enumerate(runs):
        r.setdefault("name", f"run{i}-lr{r['lr']:g}")
    ctx = _mp.get_context("fork")
    _os.makedirs(G_OUT, exist_ok=True)
    procs = []
    for i, r in enumerate(runs):
        p = ctx.Process(target=g_run, args=(r, str(i), train, hold, rates, _os.path.join(G_OUT, f"log-{r['name']}.txt")))
        p.start()
        procs.append(p)
    for r, p in zip(runs, procs):
        p.join()
        log(f"{r['name']} exit {p.exitcode}; log tail:\n" + open(_os.path.join(G_OUT, f"log-{r['name']}.txt")).read()[-3000:])


if __name__ == "__main__":
    g_main()

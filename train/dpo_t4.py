"""DPO on pass/fail pairs from bank_judge.py. Kaggle T4, keys attached; it runs no generated code.

Attach ophyx-keys and the bank-judge outputs (judged/<name>/pairs-*.jsonl). The pairs are the
model's own samples for one problem: chosen passed every test, rejected failed one. Training uses
the SFT template, so prompts end with "<|python|>\\n" and both answers end with <|end|>.

CONFIG: model (HF id of the starting checkpoint), out_repo (private HF model repo for the result,
none: keep it in /kaggle/working only), pairs_glob (a list of globs is fine), lr (5e-7), beta (0.1), epochs (1),
batch (prompts per step, 32), max_len (2048), holdout (300 pairs for the eval loss), budget_s, seed.

Output under /kaggle/working/dpo/: final/ (fp32 safetensors + tokenizer), log.jsonl, meta.json.
"""

import dataclasses
import glob
import json
import os
import random
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
SRC = CONFIG.get("src", "/kaggle/input")
OUT = os.path.join(CONFIG.get("out", "/kaggle/working"), "dpo")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def pairs():
    pats = CONFIG.get("pairs_glob", os.path.join(SRC, "**", "judged", "*", "pairs-*.jsonl"))
    files = sorted({f for p in ([pats] if isinstance(pats, str) else pats) for f in glob.glob(p, recursive=True)})
    rows = [r for f in files for r in map(json.loads, open(f))]
    random.Random(CONFIG.get("seed", 0)).shuffle(rows)
    log(f"{len(rows)} pairs from {len(files)} files")
    # "prompt_text" is a conversation already rendered up to the last "<|python|>\n" (round three's
    # multi-turn pairs); "prompt" is a single task
    return [{"prompt": r.get("prompt_text") or f"<|user|>\n{r['prompt']}\n<|python|>\n",
             "chosen": r["chosen"].rstrip("\n") + "\n<|end|>",
             "rejected": r["rejected"].rstrip("\n") + "\n<|end|>"} for r in rows if r["chosen"] != r["rejected"]]


def train_one():
    # Kaggle's T4 session shows two GPUs; Trainer would wrap the model in DataParallel, which DPOTrainer refuses
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CONFIG.get("gpu", 0)))
    import torch
    import trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

    model_id = CONFIG.get("model", "abhash-chakraborty/ophyx-350m-sft")
    if model_id.startswith("/") and "*" in model_id:  # a kernel output: Kaggle mounts them at varying depths
        model_id = sorted(glob.glob(model_id, recursive=True))[0]
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.eos_token = "<|end|>"
    if tok.pad_token is None or tok.pad_token == tok.eos_token:
        tok.pad_token = "<|pad|>"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    rows = pairs()
    hold = CONFIG.get("holdout", 300)
    train_ds, eval_ds = Dataset.from_list(rows[hold:]), Dataset.from_list(rows[:hold])
    batch = CONFIG.get("batch", 32)
    micro = CONFIG.get("micro_batch", 4)
    os.makedirs(OUT, exist_ok=True)
    want = dict(output_dir=os.path.join(OUT, "ckpt"), learning_rate=CONFIG.get("lr", 5e-7), beta=CONFIG.get("beta", 0.1),
                num_train_epochs=CONFIG.get("epochs", 1), per_device_train_batch_size=micro, per_device_eval_batch_size=micro,
                gradient_accumulation_steps=max(1, batch // micro), max_length=CONFIG.get("max_len", 2048),
                lr_scheduler_type="cosine", warmup_steps=CONFIG.get("warmup_steps", 10), fp16=torch.cuda.is_available(), bf16=False,
                gradient_checkpointing=True, logging_steps=5, eval_strategy="steps", eval_steps=CONFIG.get("eval_steps", 50),
                save_strategy="no", report_to="none", seed=CONFIG.get("seed", 0), remove_unused_columns=False,
                precompute_ref_log_probs=False)
    fields = {f.name for f in dataclasses.fields(trl.DPOConfig)}
    dropped = sorted(k for k in want if k not in fields)
    args = trl.DPOConfig(**{k: v for k, v in want.items() if k in fields})
    log(f"trl {trl.__version__}; DPOConfig ignores {dropped}; {len(train_ds)} train / {len(eval_ds)} eval pairs")
    logf = open(os.path.join(OUT, "log.jsonl"), "w")
    budget = CONFIG.get("budget_s", int(10.5 * 3600))

    class Budget(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            logf.write(json.dumps({**(logs or {}), "step": state.global_step, "t": round(time.time() - T0)}) + "\n")
            logf.flush()

        def on_step_end(self, a, state, control, **kw):
            if time.time() - T0 > budget - 1800:
                control.should_training_stop = True

    trainer = trl.DPOTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tok,
                             callbacks=[Budget()])
    trainer.train()
    final_eval = trainer.evaluate()
    logf.close()
    final = os.path.join(OUT, "final")
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # newer transformers keep rope_theta only inside rope_parameters; older ones (and the GGUF converter)
    # then fall back to Qwen3's 1e6 and the model breaks
    cfg = json.load(open(os.path.join(final, "config.json")))
    cfg.setdefault("rope_theta", (cfg.get("rope_parameters") or {}).get("rope_theta", 100000.0))
    json.dump(cfg, open(os.path.join(final, "config.json"), "w"), indent=2)
    meta = {"config": CONFIG, "steps": trainer.state.global_step, "minutes": round((time.time() - T0) / 60),
            "final_eval": final_eval, "trl": trl.__version__, "ignored_config": dropped}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1, default=str)
    log("final eval", json.dumps(final_eval, default=str))
    if CONFIG.get("out_repo") and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(CONFIG["out_repo"], repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=CONFIG["out_repo"], folder_path=final, commit_message="DPO on bank pass/fail pairs")
        log("uploaded to", CONFIG["out_repo"])


def main():
    """CONFIG["runs"] (optional): [{"name", "gpu", ...overrides such as lr or beta}], one forked child per
    run on its own GPU, outputs under dpo/<name>/. Without it, one run in dpo/."""
    global OUT
    keys = glob.glob(os.path.join(SRC, "**", "keys.json"), recursive=True)
    if keys and json.load(open(keys[0])).get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = json.load(open(keys[0]))["HF_TOKEN"]
    if os.path.isdir("/kaggle"):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl"], capture_output=True, text=True)
        log("pip install trl:", r.returncode, r.stderr[-800:])
    runs = CONFIG.get("runs")
    if not runs:
        train_one()
        return
    import multiprocessing as mp
    ctx = mp.get_context("fork")  # the parent never imports torch, so forking is safe
    procs = []
    for k, run in enumerate(runs):
        def child(run=run, k=k):
            global OUT
            OUT = os.path.join(OUT, run["name"])
            os.makedirs(OUT, exist_ok=True)
            fd = os.open(os.path.join(OUT, "stdout.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            base_repo = CONFIG.get("out_repo")
            CONFIG.update({kk: v for kk, v in run.items() if kk != "name"})
            if base_repo and not run.get("out_repo"):
                CONFIG["out_repo"] = f"{base_repo}-{run['name']}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(run.get("gpu", k % 2))
            code = 0
            try:
                train_one()
            except BaseException:
                import traceback
                traceback.print_exc()
                code = 1
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        p = ctx.Process(target=child)
        p.start()
        procs.append((run, p))
    for run, p in procs:
        p.join()
        out = os.path.join(OUT, run["name"], "stdout.txt")
        log(f"{run['name']}: exit {p.exitcode}\n" + (open(out).read()[-3000:] if os.path.exists(out) else ""))


main()

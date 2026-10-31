"""Supervised fine-tuning on Kaggle T4s: everyday chat data plus replay of the earlier SFT mix.

Round three: the released model writes good functions for benchmark-style prompts, but fails how
people actually ask (scripts, their own names and details, follow-ups). This trains it on
conversations from data/synth/chat_data.py that passed data/jobs/chat_check.py, mixed with replayed
examples from the token/mask files of the earlier SFT stages, so it keeps what it had.

Runs on T4s with plain PyTorch, so it doesn't wait in the TPU queue. Each run in CONFIG["runs"] gets
its own GPU in a forked child (two at a time on a 2x T4 session), for example two learning rates.
Attach ophyx-keys (HF token for the data) and the chat SFT dataset. It runs no generated code.

Data
  chat   jsonl rows {"turns": [[prompt, answer, train], ...]}; a conversation renders as
         <|user|>\\n p \\n<|python|>\\n a \\n<|end|> per turn, joined by "\\n" (as serve/modal_app.py builds
         history). Loss counts only on answers with train true (a complaint's first answer is the old
         model's mistake: it stays in the context, unlearned).
  replay examples cut from the earlier SFT token files at each <|user|>, with their masks.

CONFIG: model (HF id), chat_glob, replay {"repo", "patterns", "ratio": replay tokens per chat token},
runs [{"name", "lr", "gpu"}], epochs (2), max_len (2048), batch_tokens (32768), micro_tokens (4096),
warmup (30 steps), weight_decay (0.0), budget_s, out_repo (optional private HF repo prefix), seed.

Output under /kaggle/working/sft/<run>/: final/ (safetensors + tokenizer), log.jsonl, meta.json.
"""

import glob
import json
import math
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
SRC = CONFIG.get("src", "/kaggle/input")
OUT = os.path.join(CONFIG.get("out", "/kaggle/working"), "sft")
T0 = time.time()
BUDGET = CONFIG.get("budget_s", 10 * 3600)
USER_ID = 1
REPLAY_DIR = None  # set by main() before the runs fork, so they don't download the same files at once


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def render(tok, turns):
    """-> (ids, mask) for one conversation"""
    end, user, py = (tok.convert_tokens_to_ids(t) for t in ("<|end|>", "<|user|>", "<|python|>"))
    ids, mask = [], []
    for k, (prompt, answer, train) in enumerate(turns):
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        p = ([] if k == 0 else nl) + [user] + tok(f"\n{prompt}\n", add_special_tokens=False)["input_ids"] + [py] + nl
        a = tok(f"{answer.strip(chr(10))}\n", add_special_tokens=False)["input_ids"] + [end]
        ids += p + a
        mask += [0] * len(p) + [1 if train else 0] * len(a)
    return ids, mask


def replay_examples(rng, want_tokens, max_len):
    import numpy as np
    if not REPLAY_DIR or want_tokens <= 0:
        return []
    d = REPLAY_DIR
    pool = []
    for f in sorted(glob.glob(os.path.join(d, "**", "tokens", "train", "*.bin"), recursive=True)):
        t = np.fromfile(f, dtype=np.uint16)
        m = np.fromfile(f.replace("/tokens/", "/mask/"), dtype=np.uint8)
        starts = np.flatnonzero(t == USER_ID)
        bounds = list(zip(starts, list(starts[1:]) + [len(t)]))
        pool += [(t, m, a, b) for a, b in bounds if 16 < b - a <= max_len]
    rng.shuffle(pool)
    out, n = [], 0
    for t, m, a, b in pool:
        if n >= want_tokens:
            break
        out.append((t[a:b].astype("int64").tolist(), m[a:b].astype("int64").tolist()))
        n += b - a
    log(f"replay: {len(out)} examples, {n / 1e6:.2f}M tokens from {len(pool)} candidates")
    return out


def batches(examples, rng, batch_tokens, micro_tokens):
    """Length-bucketed micro-batches (padding stays small), grouped into steps of ~batch_tokens."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    micro, cur, cur_max = [], [], 0
    for i in order:
        n = len(examples[i][0])
        if cur and max(cur_max, n) * (len(cur) + 1) > micro_tokens:
            micro.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        micro.append(cur)
    rng.shuffle(micro)
    steps, step, n = [], [], 0
    for mb in micro:
        step.append(mb)
        n += sum(len(examples[i][0]) for i in mb)
        if n >= batch_tokens:
            steps.append(step)
            step, n = [], 0
    if step:
        steps.append(step)
    return steps


def train(run, logf):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = random.Random(CONFIG.get("seed", 0))
    model_id = CONFIG.get("model", "abhash-chakraborty/ophyx-350m-v2")
    if model_id.startswith("/") and "*" in model_id:
        model_id = sorted(glob.glob(model_id, recursive=True))[0]
    tok = AutoTokenizer.from_pretrained(model_id)
    max_len = CONFIG.get("max_len", 2048)
    chat = []
    for f in sorted(glob.glob(CONFIG.get("chat_glob", os.path.join(SRC, "**", "chat_sft*.jsonl")), recursive=True)):
        for line in open(f):
            ids, mask = render(tok, json.loads(line)["turns"])
            if len(ids) <= max_len and sum(mask):
                chat.append((ids, mask))
    chat_tokens = sum(len(x[0]) for x in chat)
    log(f"chat: {len(chat)} conversations, {chat_tokens / 1e6:.2f}M tokens")
    replay = replay_examples(rng, int(chat_tokens * (CONFIG.get("replay") or {}).get("ratio", 1.0)), max_len)
    epochs = CONFIG.get("epochs", 2)
    plan = []
    for e in range(epochs):  # chat repeats each epoch; the replay is split across epochs, so no example repeats
        data = chat + replay[e::epochs]
        plan += [[[data[i] for i in mb] for mb in step]
                 for step in batches(data, rng, CONFIG.get("batch_tokens", 32768), CONFIG.get("micro_tokens", 4096))]
    log(f"{len(plan)} steps over {epochs} epochs")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=run["lr"], betas=(0.9, 0.95), weight_decay=CONFIG.get("weight_decay", 0.0))
    scaler = torch.amp.GradScaler("cuda")
    warm = CONFIG.get("warmup", 30)
    pad = tok.convert_tokens_to_ids("<|end|>")

    def lr_at(s):
        if s < warm:
            return run["lr"] * (s + 1) / warm
        return run["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, len(plan) - warm))))

    model.train()
    t1 = time.time()
    seen = 0
    for s, step in enumerate(plan):
        if time.time() - T0 > BUDGET - 1200:
            log(f"{run['name']}: time budget reached at step {s}/{len(plan)}")
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        n_loss = sum(sum(m) for mb in step for _, m in mb)
        total = 0.0
        for mb in step:
            L = max(len(x) for x, _ in mb)
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in mb]).cuda()
            msk = torch.tensor([m + [0] * (L - len(m)) for _, m in mb]).cuda()
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in mb]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                                                     reduction="none")
            loss = (loss * msk[:, 1:].reshape(-1)).sum() / n_loss
            scaler.scale(loss).backward()
            total += loss.item()
            seen += int(att.sum())
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == len(plan) - 1:
            rec = {"step": s, "loss": round(total, 4), "lr": lr_at(s), "grad_norm": round(float(gn), 3),
                   "tok_s": round(seen / (time.time() - t1)), "t": round(time.time() - T0)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
    return model, tok, s + 1, len(plan)


def child(run, gpu):
    os.setsid()
    d = os.path.join(OUT, run["name"])
    os.makedirs(d, exist_ok=True)
    fd = os.open(os.path.join(d, "stdout.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = 0
    try:
        with open(os.path.join(d, "log.jsonl"), "w") as logf:
            model, tok, steps, planned = train(run, logf)
        final = os.path.join(d, "final")
        model.config.use_cache = True
        model.save_pretrained(final, safe_serialization=True)
        tok.save_pretrained(final)
        # newer transformers keep rope_theta only inside rope_parameters; older ones (and the GGUF
        # converter) then fall back to Qwen3's 1e6 and the model breaks
        cfg_path = os.path.join(final, "config.json")
        cfg = json.load(open(cfg_path))
        cfg.setdefault("rope_theta", (cfg.get("rope_parameters") or {}).get("rope_theta", 100000.0))
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        json.dump({"run": run, "config": CONFIG, "steps": steps, "planned": planned,
                   "minutes": round((time.time() - T0) / 60)}, open(os.path.join(d, "meta.json"), "w"), indent=1)
        if CONFIG.get("out_repo") and os.environ.get("HF_TOKEN"):
            from huggingface_hub import HfApi
            api = HfApi()
            repo = f"{CONFIG['out_repo']}-{run['name']}"
            api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=final, commit_message="SFT on everyday chat data")
            print("uploaded to", repo, flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    keys = glob.glob(os.path.join(SRC, "**", "keys.json"), recursive=True)
    if keys and json.load(open(keys[0])).get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = json.load(open(keys[0]))["HF_TOKEN"]
    os.makedirs(OUT, exist_ok=True)
    global REPLAY_DIR
    spec = CONFIG.get("replay") or {}
    if spec:
        from huggingface_hub import snapshot_download
        t = time.time()
        REPLAY_DIR = snapshot_download(spec["repo"], repo_type="dataset", allow_patterns=spec["patterns"],
                                       local_dir="/kaggle/tmp/replay" if os.path.isdir("/kaggle") else os.path.join(OUT, "replay"))
        log(f"replay files in {time.time() - t:.0f}s")
    ctx = mp.get_context("fork")  # the parent never touches CUDA, so forking is safe
    runs = CONFIG.get("runs", [{"name": "lr2e5", "lr": 2e-5}])
    procs = []
    for k, run in enumerate(runs):
        p = ctx.Process(target=child, args=(run, run.get("gpu", k % 2)))
        p.start()
        procs.append((run, p))
    for run, p in procs:
        p.join(timeout=max(60, BUDGET + 600 - (time.time() - T0)))
        if p.is_alive():
            p.kill()
            p.join()
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        out = os.path.join(OUT, run["name"], "stdout.txt")
        log(f"{run['name']}: exit {p.exitcode}\n" + (open(out).read()[-3000:] if os.path.exists(out) else ""))
    subprocess.run(["rm", "-rf", "/kaggle/tmp/replay"], check=False)


if __name__ == "__main__":
    main()

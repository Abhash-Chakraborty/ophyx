"""Trainer validation for one TPU session (appended after train/pretrain.py by the launcher).

1. Resume is exact: 60 steps straight vs 30 + restart + 30 must give the same weights.
2. Export: our JAX logits vs Hugging Face Qwen3ForCausalLM logits on the exported checkpoint.
3. Optimizer sanity: AdamW vs Muon on the debug model (stdlib corpus, so a smoke check, not an ablation).
4. Speed matrix for the 350M base model: attention (xla, splash) x loss chunking x recompute.

Data is the Python source already installed on the image, as in the speed test.
SMOKE shrinks everything for a CPU dry run.
"""

import hashlib
import sys

SMOKE_V = bool(CONFIG.get("smoke"))
VAL_OUT = out_dir()
WORKDIR = "/kaggle/tmp/validate" if KAGGLE else os.path.join(VAL_OUT, "work")
os.makedirs(WORKDIR, exist_ok=True)
RES = {"smoke": SMOKE_V, "devices": [str(d) for d in jax.devices()]}
SEQ_V = 128 if SMOKE_V else 2048
VOCAB_V = 1024 if SMOKE_V else 32768
SPECIALS = ["<|end|>", "<|user|>", "<|python|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|file_sep|>", "<|repo|>", "<|pad|>"]


def save_res():
    json.dump(RES, open(os.path.join(VAL_OUT, "validate.json"), "w"), indent=1)


def build_corpus():
    roots = sorted({p for p in sys.path if p and os.path.isdir(p) and "lib" in p})
    seen, docs, total, cap = set(), [], 0, 3e6 if SMOKE_V else 200e6
    for root in roots:
        for path in glob.iglob(os.path.join(root, "**", "*.py"), recursive=True):
            try:
                if os.path.getsize(path) > 500_000:
                    continue
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            h = hashlib.md5(text.encode()).digest()
            if len(text) < 200 or h in seen:
                continue
            seen.add(h)
            docs.append(text)
            total += len(text)
            if total >= cap:
                break
        if total >= cap:
            break
    rng = np.random.default_rng(0)
    rng.shuffle(docs)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(docs[: max(1, len(docs) // 3)], trainer=trainers.BpeTrainer(
        vocab_size=VOCAB_V, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    tok.save(os.path.join(VAL_OUT, "tokenizer.json"))
    n_val = max(1, len(docs) // 100)

    def enc(ds, path):
        ids = []
        for i in range(0, len(ds), 2000):
            for e in tok.encode_batch(ds[i:i + 2000]):
                ids.extend(e.ids)
                ids.append(0)
        np.asarray(ids, dtype=np.uint16).tofile(path)
        return len(ids)

    n_tr = enc(docs[n_val:], os.path.join(WORKDIR, "train.bin"))
    n_va = enc(docs[:n_val], os.path.join(WORKDIR, "val.bin"))
    RES["corpus"] = {"files": len(docs), "train_tokens_m": round(n_tr / 1e6, 1), "val_tokens_m": round(n_va / 1e6, 2)}
    log("corpus", RES["corpus"])
    return tok


def run(overrides):
    base = {"library": True, "notify": False, "resume_inputs": False, "vocab": VOCAB_V, "seq": SEQ_V,
            "save": False, "log_every": 10, "eval_every": 10 ** 9, "lr": 3e-3, "warmup_tokens": 0,
            "data": {"buckets": {"stdlib": {"glob": os.path.join(WORKDIR, "train.bin"), "weight": 1.0}},
                     "val_glob": os.path.join(WORKDIR, "val.bin")}}
    CONFIG.clear()
    CONFIG.update(base)
    CONFIG.update(overrides)
    return train()


def tree_maxdiff(a, b):
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_resume():
    m = "tiny" if SMOKE_V else "debug"
    micro = 2 if SMOKE_V else 8
    common = {"model": m, "micro_batch": micro, "batch_tokens": 2 * micro * len(jax.devices()) * SEQ_V,
              "total_tokens": 60 * 2 * micro * len(jax.devices()) * SEQ_V, "save": True, "ckpt_every_s": 10 ** 9,
              "warmup_tokens": 10 * 2 * micro * len(jax.devices()) * SEQ_V}
    pa, sa = run({**common, "out": os.path.join(WORKDIR, "A")})
    run({**common, "out": os.path.join(WORKDIR, "B"), "max_steps_this_session": 30})
    pb, sb = run({**common, "out": os.path.join(WORKDIR, "B")})
    d = tree_maxdiff(pa, pb)
    RES["resume"] = {"straight_loss": sa["final_loss"], "resumed_loss": sb["final_loss"], "max_param_diff": d,
                     "exact": d == 0.0, "pass": d < 1e-5}
    log("resume test:", RES["resume"])
    return pa


def test_export(params, tok):
    m = "tiny" if SMOKE_V else "debug"
    CONFIG["model"] = m
    dims = model_dims()
    hfdir = os.path.join(VAL_OUT, "hf_debug")
    export_hf(params, dims, VOCAB_V, SEQ_V, hfdir, eos_id=0)
    tok.save(os.path.join(hfdir, "tokenizer.json"))
    T = 64 if SMOKE_V else 256
    x = np.fromfile(os.path.join(WORKDIR, "val.bin"), dtype=np.uint16)[:T].astype(np.int32)[None]
    CONFIG["dtype"] = "float32"
    CONFIG["attention"] = "xla"
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    hidden = make_forward(dims, mesh, T)
    ours = np.asarray(jax.jit(lambda p, x: hidden(p, x) @ p["embed"].T)(params, jnp.asarray(x)))[0]
    import torch
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(hfdir, dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = hf(torch.tensor(x, dtype=torch.long)).logits[0].numpy()
    diff = float(np.max(np.abs(ours - theirs)))
    agree = float(np.mean(ours.argmax(-1) == theirs.argmax(-1)))
    RES["export"] = {"max_logit_diff": diff, "argmax_agreement": agree, "pass": diff < 1e-2 and agree > 0.99,
                     "hf_class": type(hf).__name__}
    log("export test:", RES["export"])


def test_optimizers():
    m = "tiny" if SMOKE_V else "debug"
    micro = 2 if SMOKE_V else 8
    steps = 20 if SMOKE_V else 400
    st = micro * len(jax.devices()) * SEQ_V
    out = {}
    for name, ov in [("adamw_3e-3", {"optimizer": "adamw", "lr": 3e-3}),
                     ("muon_3e-3", {"optimizer": "muon", "lr": 3e-3}),
                     ("muon_6e-3", {"optimizer": "muon", "lr": 6e-3})]:
        try:
            _, s = run({"model": m, "micro_batch": micro, "batch_tokens": st, "total_tokens": steps * st,
                        "warmup_tokens": (steps // 10) * st, "eval_every": steps, "log_every": steps // 4, **ov})
            rows = [json.loads(l) for l in open(os.path.join(out_dir(), "metrics.jsonl"))]
            vals = [r["val_loss"] for r in rows if "val_loss" in r]
            out[name] = {"final_loss": s["final_loss"], "val_loss": vals[-1] if vals else None,
                         "tokens_per_s": s["tokens_per_s"]}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        log("optimizer", name, out[name])
        os.remove(os.path.join(out_dir(), "metrics.jsonl"))
        RES["optimizers"] = out
        save_res()


def test_speed():
    m = "tiny" if SMOKE_V else "base"
    grid = [("xla", 0, "full", 8), ("xla", 512, "full", 8), ("xla", 512, "dots", 8),
            ("splash", 0, "full", 8), ("splash", 512, "full", 8), ("splash", 512, "dots", 8),
            ("splash", 512, "dots", 16), ("splash", 512, "none", 4)]
    if SMOKE_V:
        grid = [("xla", 0, "full", 2), ("xla", 64, "dots", 2)]
    out = []
    for attn, chunk, remat, micro in grid:
        st = micro * len(jax.devices()) * SEQ_V
        row = {"attention": attn, "loss_chunk": chunk, "remat": remat, "micro_batch": micro}
        try:
            _, s = run({"model": m, "micro_batch": micro, "batch_tokens": st, "total_tokens": 30 * st,
                        "attention": attn, "loss_chunk": chunk, "remat": remat, "log_every": 10})
            row.update(tokens_per_s=s["tokens_per_s"], n_params=s["n_params"])
            row["hours_10b"] = round(10e9 / s["tokens_per_s"] / 3600, 1) if s["tokens_per_s"] else None
        except Exception as e:
            msg = str(e)
            row["error"] = "OOM" if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() else f"{type(e).__name__}: {msg[:300]}"
        log("speed", row)
        out.append(row)
        RES["speed_base"] = out
        save_res()
        jax.clear_caches()


tok = build_corpus()
save_res()
params_debug = None
for name, fn in [("resume", lambda: test_resume()), ("export", None), ("optimizers", test_optimizers), ("speed", test_speed)]:
    try:
        if name == "resume":
            params_debug = fn()
        elif name == "export":
            if params_debug is not None:
                test_export(params_debug, tok)
        else:
            fn()
    except Exception as e:
        import traceback
        traceback.print_exc()
        RES[name] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
    save_res()
    jax.clear_caches()

ok = {k: RES.get(k, {}).get("pass") for k in ("resume", "export")}
best = max((r for r in RES.get("speed_base", []) if r.get("tokens_per_s")), key=lambda r: r["tokens_per_s"], default=None)
summary = f"resume {ok['resume']}, export {ok['export']}; best 350M: " + (
    f"{best['tokens_per_s']:,} tok/s ({best['attention']}, chunk {best['loss_chunk']}, {best['remat']}, b={best['micro_batch']})" if best else "n/a")
log("SUMMARY", summary)
RES["total_minutes"] = round((time.time() - T0) / 60, 1)
save_res()
notify("trainer validation done", summary, "white_check_mark")

"""TPU speed test: train the 20M debug model on real Python, then time the nano and base configs.

Runs as a Kaggle script on TPU v5e-8. Corpus is the .py files already installed on the image, so
nothing is downloaded. Results go to stdout and /kaggle/working/results.json.

SMOKE=1 shrinks everything so the script can be checked on a CPU box first.
"""

import glob
import hashlib
import json
import math
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)

VOCAB = 1024 if SMOKE else 32768
SEQ = 128 if SMOKE else 2048
CORPUS_CAP = 4e6 if SMOKE else 300e6  # bytes of source text
TRAIN_SECONDS = 20 if SMOKE else 480  # debug model training budget
BENCH_STEPS = 3 if SMOKE else 20  # timed steps per throughput config
PEAK_TFLOPS_PER_CHIP = 197.0  # v5e bf16

CONFIGS = {
    # name: d_model, layers, heads, kv_heads, ffn
    "debug": (384, 6, 6, 2, 1024),
    "nano": (768, 16, 12, 4, 2048),
    "base": (1024, 28, 16, 4, 2816),
}
if SMOKE:
    CONFIGS = {"debug": (64, 2, 2, 1, 128), "nano": (64, 2, 2, 1, 128), "base": (96, 2, 2, 1, 192)}

T0 = time.time()
RESULTS = {"smoke": SMOKE}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save_results():
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)


# ---------------------------------------------------------------- environment

devices = jax.devices()
log(f"jax {jax.__version__}, {len(devices)} devices: {devices[0].device_kind}")
RESULTS["devices"] = [str(d) for d in devices]
RESULTS["device_kind"] = devices[0].device_kind
mesh = Mesh(np.array(devices), ("data",))
REPL = NamedSharding(mesh, P())
BATCH_SHARD = NamedSharding(mesh, P("data", None))
N_DEV = len(devices)

# ---------------------------------------------------------------- corpus + tokenizer


def collect_corpus():
    roots = sorted({p for p in sys.path if p and os.path.isdir(p) and "lib" in p})
    seen, docs, total = set(), [], 0
    for root in roots:
        for path in glob.iglob(os.path.join(root, "**", "*.py"), recursive=True):
            try:
                if os.path.getsize(path) > 500_000:
                    continue
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            h = hashlib.md5(text.encode()).digest()
            if len(text) < 200 or h in seen:
                continue
            seen.add(h)
            docs.append(text)
            total += len(text)
            if total >= CORPUS_CAP:
                return docs, total
    return docs, total


log("collecting .py files")
docs, nbytes = collect_corpus()
rng = np.random.default_rng(0)
rng.shuffle(docs)
n_val = max(1, len(docs) // 100)
val_docs, train_docs = docs[:n_val], docs[n_val:]
log(f"corpus: {len(docs)} files, {nbytes / 1e6:.0f} MB (val {n_val} files)")
RESULTS["corpus"] = {"files": len(docs), "mb": round(nbytes / 1e6, 1)}

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=["<|end|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
t = time.time()
tok.train_from_iterator(train_docs[: max(1, len(train_docs) // 3)], trainer=trainer)
EOT = tok.token_to_id("<|end|>")
log(f"tokenizer trained in {time.time() - t:.0f}s, vocab {tok.get_vocab_size()}")


def encode(ds):
    ids = []
    for i in range(0, len(ds), 2000):
        for e in tok.encode_batch(ds[i : i + 2000]):
            ids.extend(e.ids)
            ids.append(EOT)
    return np.asarray(ids, dtype=np.uint16)


t = time.time()
train_ids, val_ids = encode(train_docs), encode(val_docs)
chars_per_tok = sum(map(len, val_docs)) / len(val_ids)
log(f"encoded in {time.time() - t:.0f}s: {len(train_ids) / 1e6:.1f}M train tokens, "
    f"{len(val_ids) / 1e6:.2f}M val, {chars_per_tok:.2f} chars/token")
RESULTS["tokens"] = {"train_m": round(len(train_ids) / 1e6, 1), "chars_per_token": round(chars_per_tok, 2)}
save_results()


def get_batch(ids, batch, rng):
    starts = rng.integers(0, len(ids) - SEQ - 1, size=batch)
    idx = starts[:, None] + np.arange(SEQ + 1)[None, :]
    return ids[idx].astype(np.int32)


# ---------------------------------------------------------------- model (Llama/Qwen3-style)


def init_params(cfg, key):
    d, L, H, KV, F = cfg
    hd = d // H
    ks = jax.random.split(key, 8)
    std = 0.02
    out_std = std / math.sqrt(2 * L)

    def n(k, shape, s):
        return (jax.random.normal(k, shape, jnp.float32) * s).astype(jnp.float32)

    return {
        "embed": n(ks[0], (VOCAB, d), std),
        "final_norm": jnp.ones((d,)),
        "layers": {
            "attn_norm": jnp.ones((L, d)),
            "mlp_norm": jnp.ones((L, d)),
            "q_norm": jnp.ones((L, hd)),
            "k_norm": jnp.ones((L, hd)),
            "wq": n(ks[1], (L, d, H * hd), std),
            "wk": n(ks[2], (L, d, KV * hd), std),
            "wv": n(ks[3], (L, d, KV * hd), std),
            "wo": n(ks[4], (L, H * hd, d), out_std),
            "w_gate": n(ks[5], (L, d, F), std),
            "w_up": n(ks[6], (L, d, F), std),
            "w_down": n(ks[7], (L, F, d), out_std),
        },
    }


def rms_norm(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (y * w).astype(x.dtype)


def rope_tables(T, hd, base=100_000.0):
    inv = 1.0 / (base ** (np.arange(0, hd, 2) / hd))
    ang = np.arange(T)[:, None] * inv[None, :]
    return jnp.asarray(np.cos(ang), jnp.float32), jnp.asarray(np.sin(ang), jnp.float32)


def apply_rope(x, cos, sin):  # x: [B, T, heads, hd]
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(x.dtype)


def make_forward(cfg, remat_policy):
    d, L, H, KV, F = cfg
    hd = d // H
    dt = jnp.bfloat16

    def layer(h, p, cos, sin):
        B, T, _ = h.shape
        x = rms_norm(h, p["attn_norm"])
        q = (x @ p["wq"].astype(dt)).reshape(B, T, H, hd)
        k = (x @ p["wk"].astype(dt)).reshape(B, T, KV, hd)
        v = (x @ p["wv"].astype(dt)).reshape(B, T, KV, hd)
        q = apply_rope(rms_norm(q, p["q_norm"]), cos, sin)
        k = apply_rope(rms_norm(k, p["k_norm"]), cos, sin)
        a = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        h = h + a.reshape(B, T, H * hd) @ p["wo"].astype(dt)
        x = rms_norm(h, p["mlp_norm"])
        g = jax.nn.silu(x @ p["w_gate"].astype(dt)) * (x @ p["w_up"].astype(dt))
        return h + g @ p["w_down"].astype(dt)

    if remat_policy == "full":
        layer = jax.checkpoint(layer)
    elif remat_policy == "dots":
        layer = jax.checkpoint(layer, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)

    def forward(params, x):
        T = x.shape[1]
        cos, sin = rope_tables(T, hd)
        h = params["embed"].astype(dt)[x]

        def body(h, p):
            return layer(h, p, cos, sin), None

        h, _ = jax.lax.scan(body, h, params["layers"])
        h = rms_norm(h, params["final_norm"])
        return h @ params["embed"].astype(dt).T  # tied output head

    return forward


def loss_fn(forward, params, batch):
    x, y = batch[:, :-1], batch[:, 1:]
    logits = forward(params, x).astype(jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    tgt = jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
    ce = jnp.mean(lse - tgt)
    return ce + 1e-4 * jnp.mean(lse**2), ce


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def flops_per_token(cfg, n_params):
    d, L, H, KV, F = cfg
    return 6 * n_params + 12 * L * d * SEQ  # PaLM-style: dense + attention


def build(cfg, remat_policy, lr_sched):
    forward = make_forward(cfg, remat_policy)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_sched, b1=0.9, b2=0.95, weight_decay=0.1),
    )

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, batch):
        (_, ce), grads = jax.value_and_grad(partial(loss_fn, forward), has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, ce

    @jax.jit
    def eval_ce(params, batch):
        return loss_fn(forward, params, batch)[1]

    return forward, opt, step, eval_ce


def is_oom(e):
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s or "Out of memory" in s or "out of memory" in s


# ---------------------------------------------------------------- throughput benchmark


def bench(name, cfg, per_dev_batch, remat_policy):
    params = jax.device_put(init_params(cfg, jax.random.PRNGKey(0)), REPL)
    n_params = count_params(params)
    _, opt, step, _ = build(cfg, remat_policy, 1e-4)
    opt_state = jax.device_put(opt.init(params), REPL)
    gb = per_dev_batch * N_DEV
    brng = np.random.default_rng(1)
    batches = [jax.device_put(get_batch(train_ids, gb, brng), BATCH_SHARD) for _ in range(4)]
    t = time.time()
    params, opt_state, ce = step(params, opt_state, batches[0])
    ce.block_until_ready()
    compile_s = time.time() - t
    for i in range(2):  # warm-up
        params, opt_state, ce = step(params, opt_state, batches[i % 4])
    ce.block_until_ready()
    t = time.time()
    for i in range(BENCH_STEPS):
        params, opt_state, ce = step(params, opt_state, batches[i % 4])
    ce.block_until_ready()
    dt_ = (time.time() - t) / BENCH_STEPS
    tok_s = gb * SEQ / dt_
    mfu = tok_s * flops_per_token(cfg, n_params) / (PEAK_TFLOPS_PER_CHIP * 1e12 * N_DEV)
    del params, opt_state, batches
    return {
        "params_m": round(n_params / 1e6, 1),
        "per_device_batch": per_dev_batch,
        "global_batch_tokens": gb * SEQ,
        "remat": remat_policy,
        "step_s": round(dt_, 3),
        "tokens_per_s": round(tok_s),
        "mfu": round(mfu, 3),
        "compile_s": round(compile_s, 1),
    }


def bench_config(name, cfg, attempts):
    best, tried = None, []
    for per_dev_batch, remat in attempts:
        try:
            r = bench(name, cfg, per_dev_batch, remat)
            log(f"{name} b={per_dev_batch} remat={remat}: {r['tokens_per_s']:,} tok/s, "
                f"MFU {r['mfu']:.1%}, step {r['step_s']}s, compile {r['compile_s']}s")
            tried.append(r)
            if best is None or r["tokens_per_s"] > best["tokens_per_s"]:
                best = r
        except Exception as e:  # OOM or compile failure: record it and try the next setting
            msg = "OOM" if is_oom(e) else f"{type(e).__name__}: {str(e)[:300]}"
            log(f"{name} b={per_dev_batch} remat={remat}: failed ({msg})")
            tried.append({"per_device_batch": per_dev_batch, "remat": remat, "error": msg})
        jax.clear_caches()
    return {"best": best, "tried": tried}


# ---------------------------------------------------------------- 1. train the debug model


def train_debug(per_dev_batch):
    cfg = CONFIGS["debug"]
    gb = per_dev_batch * N_DEV
    est_steps = 30 if SMOKE else 2000
    sched = optax.warmup_cosine_decay_schedule(0.0, 3e-3, min(200, est_steps // 10), est_steps, 3e-4)
    forward, opt, step, eval_ce = build(cfg, "none", sched)
    params = jax.device_put(init_params(cfg, jax.random.PRNGKey(0)), REPL)
    n_params = count_params(params)
    opt_state = jax.device_put(opt.init(params), REPL)
    log(f"debug model: {n_params / 1e6:.1f}M params, global batch {gb} x {SEQ} = {gb * SEQ:,} tokens")

    brng, vrng = np.random.default_rng(2), np.random.default_rng(3)
    val_batches = [jax.device_put(get_batch(val_ids, gb, vrng), BATCH_SHARD) for _ in range(4)]

    def val_loss():
        return float(np.mean([float(eval_ce(params, b)) for b in val_batches]))

    curve, s, t_start, tokens = [], 0, None, 0
    log_every = 5 if SMOKE else 50
    while s < est_steps:
        batch = jax.device_put(get_batch(train_ids, gb, brng), BATCH_SHARD)
        params, opt_state, ce = step(params, opt_state, batch)
        s += 1
        if s == 1:
            ce.block_until_ready()
            t_start = time.time()
            log(f"first step compiled, train loss {float(ce):.3f}")
            continue
        tokens += gb * SEQ
        if s % log_every == 0:
            el = time.time() - t_start
            row = {"step": s, "train_loss": round(float(ce), 4), "tokens_m": round(tokens / 1e6, 1),
                   "elapsed_s": round(el, 1), "tokens_per_s": round(tokens / el)}
            if s % (log_every * 4) == 0:
                row["val_loss"] = round(val_loss(), 4)
            curve.append(row)
            log(json.dumps(row))
            if el > TRAIN_SECONDS:
                break
    el = time.time() - t_start
    final_val = val_loss()
    log(f"debug done: {s} steps, {tokens / 1e6:.0f}M tokens in {el:.0f}s "
        f"({tokens / el:,.0f} tok/s), val loss {final_val:.3f}")

    # Greedy-ish sample to see that it writes Python.
    prompt = 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
    ids = tok.encode(prompt).ids
    W = 256
    fwd = jax.jit(forward)
    srng = np.random.default_rng(4)
    while len(ids) < W:
        ctx = np.zeros((1, W), np.int32)
        ctx[0, : len(ids)] = ids
        logits = np.asarray(fwd(params, jnp.asarray(ctx))[0, len(ids) - 1], np.float32) / 0.6
        top = np.argpartition(logits, -20)[-20:]
        p = np.exp(logits[top] - logits[top].max())
        nxt = int(top[srng.choice(len(top), p=p / p.sum())])
        if nxt == EOT:
            break
        ids.append(nxt)
    sample = tok.decode(ids)
    log("sample:\n" + sample)
    with open(os.path.join(OUT, "sample.txt"), "w") as f:
        f.write(sample)

    tf = flops_per_token(cfg, n_params)
    return {
        "params_m": round(n_params / 1e6, 1),
        "steps": s,
        "tokens_m": round(tokens / 1e6, 1),
        "seconds": round(el),
        "tokens_per_s": round(tokens / el),
        "mfu": round(tokens / el * tf / (PEAK_TFLOPS_PER_CHIP * 1e12 * N_DEV), 3),
        "final_val_loss": round(final_val, 4),
        "curve": curve,
    }


for pdb in ([2] if SMOKE else [8, 4]):
    try:
        RESULTS["debug_training"] = train_debug(pdb)
        break
    except Exception as e:
        log(f"debug training b={pdb} failed: {'OOM' if is_oom(e) else type(e).__name__}: {str(e)[:300]}")
        RESULTS["debug_training"] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
        jax.clear_caches()
save_results()
jax.clear_caches()

# ---------------------------------------------------------------- 2. throughput for nano and base

ATTEMPTS = {
    "nano": [(8, "dots"), (16, "full"), (8, "full"), (4, "dots")],
    "base": [(8, "full"), (8, "dots"), (4, "dots"), (16, "full")],
}
if SMOKE:
    ATTEMPTS = {"nano": [(2, "dots"), (2, "full")], "base": [(2, "full")]}

RESULTS["throughput"] = {}
for name in ("nano", "base"):
    RESULTS["throughput"][name] = bench_config(name, CONFIGS[name], ATTEMPTS[name])
    save_results()

# ---------------------------------------------------------------- summary

log("=== SUMMARY ===")
dt_ = RESULTS["debug_training"]
if "tokens_per_s" in dt_:
    log(f"debug {dt_['params_m']}M: {dt_['tokens_per_s']:,} tok/s, MFU {dt_['mfu']:.1%}, "
        f"val loss {dt_['final_val_loss']}")
for name, r in RESULTS["throughput"].items():
    b = r["best"]
    if b:
        hours_10b = 10e9 / b["tokens_per_s"] / 3600
        b["hours_for_10b_tokens"] = round(hours_10b, 1)
        log(f"{name} {b['params_m']}M: {b['tokens_per_s']:,} tok/s, MFU {b['mfu']:.1%} "
            f"(b={b['per_device_batch']}, remat={b['remat']}) -> 10B tokens in {hours_10b:.1f} h")
    else:
        log(f"{name}: no setting fitted")
RESULTS["total_seconds"] = round(time.time() - T0)
save_results()
log("done")

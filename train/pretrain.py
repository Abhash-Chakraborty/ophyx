"""Pretraining on TPU (or any JAX backend): Qwen3-style decoder, data-parallel, restart-safe.

One Kaggle session runs until its time budget is nearly spent, saves a checkpoint and exits; the
next session attaches that output and continues. The batch for step s is a pure function of
(seed, s), so a resumed run sees exactly the data the uninterrupted run would have.

CONFIG keys (all optional except data):
  model           "debug" | "nano" | "base" | [d, layers, heads, kv_heads, ffn]
  vocab, seq      vocab size (from the tokenizer) and context length
  batch_tokens    tokens per optimizer step (gradient accumulation fills it)
  micro_batch     sequences per device per micro-step
  total_tokens    run length; sets the schedule
  lr, warmup_tokens, decay_frac, min_lr_frac     WSD schedule
  optimizer       "adamw" | "muon" (Muon on matrices, AdamW on embeddings and norms)
  attention       "xla" | "splash"
  remat           "none" | "dots" | "full"
  loss_chunk      sequence chunk for the cross-entropy (0 = whole sequence at once)
  data            {"buckets": {name: {"glob": "...", "weight": w}}, "val_glob": "..."}
  budget_s        wall-clock budget for this session; stops ~margin_s before it
  ckpt_every_s    checkpoint interval
  export_hf       write a Hugging Face Qwen3 checkpoint when the run completes
  wandb           project name, or null
"""

import glob
import json
import math
import os
import shutil
import time
import urllib.request
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
T0 = time.time()
KAGGLE = os.path.isdir("/kaggle")


def out_dir():
    d = CONFIG.get("out", "/kaggle/working" if KAGGLE else os.path.abspath("run"))
    os.makedirs(d, exist_ok=True)
    return d


MODELS = {
    "tiny": (64, 2, 4, 2, 192),
    "debug": (384, 6, 6, 2, 1024),
    "nano": (768, 16, 12, 4, 2048),
    "base": (1024, 28, 16, 4, 2816),
}


def cfg(key, default):
    return CONFIG.get(key, default)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_keys():
    paths = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    return json.load(open(paths[0])) if paths else {}


KEYS = load_keys()


def notify(title, msg, tags="robot"):
    topic = KEYS.get("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode()[:3000],
                                                          headers={"Title": title, "Tags": tags}, method="POST"), timeout=15)
        except Exception as e:
            log("ntfy failed:", e)


# ---------------------------------------------------------------- model


def model_dims():
    m = cfg("model", "debug")
    return tuple(MODELS[m]) if isinstance(m, str) else tuple(m)


def init_params(dims, vocab, key):
    d, L, H, KV, F = dims
    hd = d // H
    ks = jax.random.split(key, 8)
    std, out_std = 0.02, 0.02 / math.sqrt(2 * L)

    def n(k, shape, s):
        return jax.random.normal(k, shape, jnp.float32) * s

    return {
        "embed": n(ks[0], (vocab, d), std),
        "final_norm": jnp.ones((d,)),
        "layers": {
            "attn_norm": jnp.ones((L, d)), "mlp_norm": jnp.ones((L, d)),
            "q_norm": jnp.ones((L, hd)), "k_norm": jnp.ones((L, hd)),
            "wq": n(ks[1], (L, d, H * hd), std), "wk": n(ks[2], (L, d, KV * hd), std),
            "wv": n(ks[3], (L, d, KV * hd), std), "wo": n(ks[4], (L, H * hd, d), out_std),
            "w_gate": n(ks[5], (L, d, F), std), "w_up": n(ks[6], (L, d, F), std),
            "w_down": n(ks[7], (L, F, d), out_std),
        },
    }


def rms_norm(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (y * w).astype(x.dtype)


def rope_tables(T, hd, base=100_000.0):
    inv = 1.0 / (base ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.arange(T)[:, None] * inv[None, :]
    return jnp.asarray(np.cos(ang), jnp.float32), jnp.asarray(np.sin(ang), jnp.float32)


def apply_rope(x, cos, sin):  # x [B, T, heads, hd]; HF rotate_half convention
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)


def make_attention(kind, mesh, H, KV, hd, T):
    if kind == "splash":
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

        G = H // KV
        mask = sm.MultiHeadMask([sm.CausalMask(shape=(T, T)) for _ in range(G)])
        kernel = sk.make_splash_mqa_single_device(mask=mask)

        def local(q, k, v):  # q [b, T, H, hd], k/v [b, T, KV, hd]
            b = q.shape[0]
            qg = q.reshape(b, T, KV, G, hd).transpose(0, 2, 3, 1, 4) * (hd ** -0.5)  # [b, KV, G, T, hd]
            kg = k.transpose(0, 2, 1, 3)  # [b, KV, T, hd]
            vg = v.transpose(0, 2, 1, 3)
            o = jax.vmap(jax.vmap(kernel))(qg, kg, vg)  # [b, KV, G, T, hd]
            return o.transpose(0, 3, 1, 2, 4).reshape(b, T, H, hd)

        return jax.shard_map(local, mesh=mesh, in_specs=(P("data"),) * 3, out_specs=P("data"), check_vma=False)

    def xla(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True)

    return xla


def make_forward(dims, mesh, T):
    d, L, H, KV, F = dims
    hd = d // H
    dt = jnp.dtype(cfg("dtype", "bfloat16"))
    attn = make_attention(cfg("attention", "xla"), mesh, H, KV, hd, T)

    def layer(h, p, cos, sin):
        B = h.shape[0]
        x = rms_norm(h, p["attn_norm"])
        q = (x @ p["wq"].astype(dt)).reshape(B, T, H, hd)
        k = (x @ p["wk"].astype(dt)).reshape(B, T, KV, hd)
        v = (x @ p["wv"].astype(dt)).reshape(B, T, KV, hd)
        q = apply_rope(rms_norm(q, p["q_norm"]), cos, sin)
        k = apply_rope(rms_norm(k, p["k_norm"]), cos, sin)
        h = h + attn(q, k, v).reshape(B, T, H * hd) @ p["wo"].astype(dt)
        x = rms_norm(h, p["mlp_norm"])
        return h + (jax.nn.silu(x @ p["w_gate"].astype(dt)) * (x @ p["w_up"].astype(dt))) @ p["w_down"].astype(dt)

    remat = cfg("remat", "dots")
    if remat == "full":
        layer = jax.checkpoint(layer)
    elif remat == "dots":
        layer = jax.checkpoint(layer, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)

    def hidden(params, x):
        cos, sin = rope_tables(T, hd)
        h = params["embed"].astype(dt)[x]
        h, _ = jax.lax.scan(lambda h, p: (layer(h, p, cos, sin), None), h, params["layers"])
        return rms_norm(h, params["final_norm"])

    return hidden


def make_loss(hidden, T):
    chunk = cfg("loss_chunk", 0) or T

    def ce_chunk(emb, h, y):  # h [B, c, d], y [B, c]
        logits = (h @ emb.T).astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        return jnp.sum(lse - tgt), jnp.sum(lse ** 2)

    ce_chunk = jax.checkpoint(ce_chunk)

    def loss_fn(params, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        h = hidden(params, x)
        emb = params["embed"].astype(jnp.dtype(cfg("dtype", "bfloat16")))
        B = h.shape[0]
        n = T // chunk
        hc = h.reshape(B, n, chunk, -1).transpose(1, 0, 2, 3)
        yc = y.reshape(B, n, chunk).transpose(1, 0, 2)
        ce_sum, z_sum = jax.lax.map(lambda a: ce_chunk(emb, a[0], a[1]), (hc, yc))
        ntok = B * T
        ce = jnp.sum(ce_sum) / ntok
        return ce + 1e-4 * jnp.sum(z_sum) / ntok, ce

    return loss_fn


# ---------------------------------------------------------------- optimizer


def newton_schulz(g, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(jnp.bfloat16)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (jnp.linalg.norm(x) + 1e-7)
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return (x.T if transpose else x).astype(jnp.float32)


def muon(lr, beta=0.95, weight_decay=0.1):
    """Muon for the stacked [L, m, n] matrices; update RMS matched to AdamW (0.2·sqrt(max(m, n)))."""

    def init(params):
        return {"mom": jax.tree.map(jnp.zeros_like, params), "count": jnp.zeros([], jnp.int32)}

    def update(grads, state, params):
        mom = jax.tree.map(lambda m, g: beta * m + g, state["mom"], grads)
        nesterov = jax.tree.map(lambda m, g: beta * m + g, mom, grads)
        rate = lr(state["count"]) if callable(lr) else lr

        def one(u, p):
            o = jax.vmap(newton_schulz)(u)
            scale = 0.2 * math.sqrt(max(u.shape[-2], u.shape[-1]))
            return -rate * (o * scale + weight_decay * p)

        updates = jax.tree.map(one, nesterov, params)
        return updates, {"mom": mom, "count": state["count"] + 1}

    return optax.GradientTransformation(init, update)


def build_optimizer(schedule):
    adamw = optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1)
    if cfg("optimizer", "adamw") == "adamw":
        return optax.chain(optax.clip_by_global_norm(1.0), adamw)
    matrix_keys = {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}

    def label(params):
        return {
            "embed": "adamw", "final_norm": "adamw",
            "layers": {k: ("muon" if k in matrix_keys else "adamw") for k in params["layers"]},
        }

    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.multi_transform({"adamw": adamw, "muon": muon(schedule)}, label))


def wsd_schedule(peak, warmup, total, decay_frac, min_frac):
    decay_start = int(total * (1 - decay_frac))

    def f(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * jnp.minimum(1.0, (step + 1) / max(1, warmup))
        frac = jnp.clip((step - decay_start) / max(1, total - decay_start), 0.0, 1.0)
        decay = peak * (min_frac + (1 - min_frac) * (1 - jnp.sqrt(frac)))  # 1-sqrt cooldown
        return jnp.where(step < decay_start, warm, decay)

    return f


# ---------------------------------------------------------------- data


class Data:
    """Random windows from uint16 token shards, mixed by bucket weight. batch(step) is deterministic."""

    def __init__(self, spec, seq, seed):
        self.seq, self.seed = seq, seed
        self.buckets = []
        for name, b in spec["buckets"].items():
            files = sorted(glob.glob(b["glob"], recursive=True))
            if not files:
                raise FileNotFoundError(f"bucket {name}: nothing matches {b['glob']}")
            maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
            sizes = np.array([len(m) - seq - 1 for m in maps], dtype=np.int64)
            keep = sizes > 0
            self.buckets.append((name, float(b["weight"]), [m for m, k in zip(maps, keep) if k], sizes[keep]))
            log(f"data bucket {name}: {len(files)} shards, {sum(len(m) for m in maps) / 1e6:.1f}M tokens, weight {b['weight']}")
        w = np.array([b[1] for b in self.buckets])
        self.weights = w / w.sum()
        self.val = None
        if spec.get("val_glob"):
            vf = sorted(glob.glob(spec["val_glob"], recursive=True))
            self.val = [np.memmap(f, dtype=np.uint16, mode="r") for f in vf]

    def windows(self, rng, n, maps, sizes):
        probs = sizes / sizes.sum()
        which = rng.choice(len(maps), size=n, p=probs)
        out = np.empty((n, self.seq + 1), np.int32)
        for i, s in enumerate(which):
            off = rng.integers(0, sizes[s])
            out[i] = maps[s][off: off + self.seq + 1]
        return out

    def batch(self, step, n):
        rng = np.random.default_rng([self.seed, step])
        counts = rng.multinomial(n, self.weights)
        parts = [self.windows(rng, c, b[2], b[3]) for c, b in zip(counts, self.buckets) if c]
        x = np.concatenate(parts)
        return x[rng.permutation(n)]

    def val_batches(self, n, k=4):
        if not self.val:
            return []
        rng = np.random.default_rng([self.seed, 10 ** 9])
        sizes = np.array([len(m) - self.seq - 1 for m in self.val])
        return [self.windows(rng, n, self.val, sizes) for _ in range(k)]


# ---------------------------------------------------------------- checkpoints


def flat(tree):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return {f"a{i:05d}": np.asarray(jax.device_get(x)) for i, x in enumerate(leaves)}


def save_ckpt(state, step, extra):
    d = os.path.join(out_dir(), f"ckpt-{step:08d}")
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    np.savez(os.path.join(tmp, "params.npz"), **flat(state[0]))
    np.savez(os.path.join(tmp, "opt.npz"), **flat(state[1]))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"step": step, "config": CONFIG, **extra}, f, indent=1)
    os.replace(tmp, d)
    for old in sorted(glob.glob(os.path.join(out_dir(), "ckpt-*")))[:-2]:  # keep the latest two
        if not old.endswith(".tmp"):
            shutil.rmtree(old, ignore_errors=True)
    log(f"checkpoint saved: step {step}")
    return d


def find_latest_ckpt():
    cands = glob.glob(os.path.join(out_dir(), "ckpt-*/meta.json"))
    if cfg("resume_inputs", True):
        cands += glob.glob("/kaggle/input/**/ckpt-*/meta.json", recursive=True)
    cands += [p for p in glob.glob(os.path.join(cfg("resume_dir", "/nonexistent"), "**/ckpt-*/meta.json"), recursive=True)]
    best = None
    for m in cands:
        try:
            step = json.load(open(m))["step"]
        except Exception:
            continue
        if best is None or step > best[0]:
            best = (step, os.path.dirname(m))
    return best


def load_ckpt(path, like):
    def restore(fname, tree):
        z = np.load(os.path.join(path, fname))
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        new = [z[f"a{i:05d}"] for i in range(len(leaves))]
        for a, b in zip(new, leaves):
            assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
        return jax.tree_util.tree_unflatten(treedef, new)

    return restore("params.npz", like[0]), restore("opt.npz", like[1])


# ---------------------------------------------------------------- export to Hugging Face (Qwen3)


def export_hf(params, dims, vocab, seq, out_dir, eos_id=0):
    from safetensors.numpy import save_file

    d, L, H, KV, F = dims
    hd = d // H
    p = jax.device_get(params)
    lay = p["layers"]
    t = {"model.embed_tokens.weight": p["embed"], "model.norm.weight": p["final_norm"]}
    for i in range(L):
        pre = f"model.layers.{i}."
        t[pre + "input_layernorm.weight"] = lay["attn_norm"][i]
        t[pre + "post_attention_layernorm.weight"] = lay["mlp_norm"][i]
        t[pre + "self_attn.q_proj.weight"] = lay["wq"][i].T
        t[pre + "self_attn.k_proj.weight"] = lay["wk"][i].T
        t[pre + "self_attn.v_proj.weight"] = lay["wv"][i].T
        t[pre + "self_attn.o_proj.weight"] = lay["wo"][i].T
        t[pre + "self_attn.q_norm.weight"] = lay["q_norm"][i]
        t[pre + "self_attn.k_norm.weight"] = lay["k_norm"][i]
        t[pre + "mlp.gate_proj.weight"] = lay["w_gate"][i].T
        t[pre + "mlp.up_proj.weight"] = lay["w_up"][i].T
        t[pre + "mlp.down_proj.weight"] = lay["w_down"][i].T
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in t.items()},
              os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    config = {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": vocab,
        "hidden_size": d, "intermediate_size": F, "num_hidden_layers": L, "num_attention_heads": H,
        "num_key_value_heads": KV, "head_dim": hd, "hidden_act": "silu", "max_position_embeddings": seq,
        "rms_norm_eps": 1e-6, "rope_theta": 100000.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 100000.0},
        "tie_word_embeddings": True, "attention_bias": False, "attention_dropout": 0.0,
        "use_sliding_window": False, "bos_token_id": eos_id, "eos_token_id": eos_id, "torch_dtype": "float32",
    }
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    log("exported HF checkpoint to", out_dir)


# ---------------------------------------------------------------- training loop


def train(data=None):
    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",))
    repl, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("data", None))
    ndev = len(devices)
    dims = model_dims()
    vocab, seq = cfg("vocab", 32768), cfg("seq", 2048)
    micro = cfg("micro_batch", 8) * ndev
    accum = max(1, cfg("batch_tokens", micro * seq) // (micro * seq))
    step_tokens = accum * micro * seq
    total_steps = max(1, cfg("total_tokens", 100 * step_tokens) // step_tokens)
    warmup = max(1, cfg("warmup_tokens", 200 * step_tokens) // step_tokens)
    sched = wsd_schedule(cfg("lr", 3e-3), warmup, total_steps, cfg("decay_frac", 0.15), cfg("min_lr_frac", 0.05))
    opt = build_optimizer(sched)
    data = data or Data(cfg("data", {}), seq, cfg("seed", 0))
    log(f"model {dims}, vocab {vocab}, seq {seq}, {ndev} devices, micro {micro} x accum {accum} = "
        f"{step_tokens:,} tokens/step, {total_steps} steps, warmup {warmup}")

    hidden = make_forward(dims, mesh, seq)
    loss_fn = make_loss(hidden, seq)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, batches):  # batches [accum, micro, seq+1]
        def micro_step(carry, b):
            g_acc, ce_acc = carry
            (_, ce), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
            return (jax.tree.map(jnp.add, g_acc, g), ce_acc + ce), None

        zeros = jax.tree.map(jnp.zeros_like, params)
        (g, ce), _ = jax.lax.scan(micro_step, (zeros, jnp.zeros([], jnp.float32)), batches)
        g = jax.tree.map(lambda x: x / accum, g)
        gnorm = optax.global_norm(g)
        updates, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, ce / accum, gnorm

    @jax.jit
    def eval_ce(params, b):
        return loss_fn(params, b)[1]

    params = jax.jit(lambda k: init_params(dims, vocab, k), out_shardings=repl)(jax.random.PRNGKey(cfg("seed", 0)))
    opt_state = jax.jit(opt.init, out_shardings=repl)(params)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    start = 0
    found = find_latest_ckpt()
    if found:
        start, path = found
        p, o = load_ckpt(path, (params, opt_state))
        params, opt_state = jax.device_put(p, repl), jax.device_put(o, repl)
        log(f"resumed from {path} at step {start}")
    log(f"{n_params / 1e6:.1f}M parameters")

    use_wandb = bool(cfg("wandb", None)) and KEYS.get("WANDB_API_KEY")
    if use_wandb:
        try:
            os.environ["WANDB_API_KEY"] = KEYS["WANDB_API_KEY"]
            import wandb
            wandb.init(project=cfg("wandb", "pycoder"), name=cfg("run_name", None), id=cfg("run_id", None),
                       resume="allow", config=CONFIG)
        except Exception as e:
            log("wandb disabled:", e)
            use_wandb = False

    deadline = T0 + cfg("budget_s", 11 * 3600) - cfg("margin_s", 1500)
    last_ckpt, metrics_f = time.time(), open(os.path.join(out_dir(), "metrics.jsonl"), "a")
    session_steps = cfg("max_steps_this_session", None)
    last_rate = None
    val = [jax.device_put(v, shard) for v in data.val_batches(micro)]
    step, t_log, tok_log = start, time.time(), 0
    nxt = data.batch(step, accum * micro).reshape(accum, micro, seq + 1)
    stop_reason = "done"
    while step < total_steps:
        batches = jax.device_put(nxt, NamedSharding(mesh, P(None, "data", None)))
        params, opt_state, ce, gnorm = train_step(params, opt_state, batches)
        step += 1
        if step < total_steps:
            nxt = data.batch(step, accum * micro).reshape(accum, micro, seq + 1)  # overlaps device compute
        tok_log += step_tokens
        if step % cfg("log_every", 20) == 0 or step == total_steps:
            ce_v, gn = float(ce), float(gnorm)
            dt = time.time() - t_log
            row = {"step": step, "loss": round(ce_v, 4), "gnorm": round(gn, 3), "lr": float(sched(step)),
                   "tokens_per_s": round(tok_log / dt), "tokens_seen": step * step_tokens}
            if not math.isfinite(ce_v):
                stop_reason = "nan"
                log("non-finite loss, stopping without saving", row)
                break
            if val and step % cfg("eval_every", 200) == 0:
                row["val_loss"] = round(float(np.mean([float(eval_ce(params, v)) for v in val])), 4)
            log(json.dumps(row))
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            if use_wandb:
                wandb.log(row, step=step)
            last_rate = row["tokens_per_s"]
            t_log, tok_log = time.time(), 0
        if session_steps and step - start >= session_steps:
            stop_reason = "session_steps"
            break
        if cfg("save", True) and time.time() - last_ckpt > cfg("ckpt_every_s", 1800):
            save_ckpt((params, opt_state), step, {"n_params": n_params})
            last_ckpt = time.time()
        if time.time() > deadline:
            stop_reason = "budget"
            break
    if stop_reason != "nan" and cfg("save", True):
        save_ckpt((params, opt_state), step, {"n_params": n_params})
    status = {"step": step, "total_steps": total_steps, "finished": step >= total_steps, "reason": stop_reason,
              "hours": round((time.time() - T0) / 3600, 2), "tokens_per_s": last_rate, "n_params": n_params,
              "final_loss": round(float(ce), 4)}
    json.dump(status, open(os.path.join(out_dir(), "status.json"), "w"))
    if status["finished"] and cfg("export_hf", False):
        export_hf(params, dims, vocab, seq, os.path.join(out_dir(), "hf"), eos_id=cfg("eos_id", 0))
    if cfg("notify", True):
        notify(f"train {cfg('run_name', 'run')}: {stop_reason}", json.dumps(status),
               "white_check_mark" if stop_reason != "nan" else "warning")
    return params, status


if __name__ == "__main__" and not CONFIG.get("library"):
    train()

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
  hf_data         {"repo", "patterns", "dir"}: download data files from an HF dataset before reading globs
  init_from       weights to start from (SFT): checkpoint dir/glob or "hf://<repo>/<run>/latest"
"""

import glob
import json
import math
import os
import shutil
import time
import urllib.request
from functools import partial

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
T0 = time.time()

SPLASH_PROBE = r"""
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk, splash_attention_mask as sm
T, G, hd = 2048, 4, 64
k = sk.make_splash_mqa_single_device(mask=sm.MultiHeadMask([sm.CausalMask(shape=(T, T)) for _ in range(G)]))
q = jnp.ones((G, T, hd), jnp.bfloat16); kv = jnp.ones((T, hd), jnp.bfloat16)
out = jax.jit(k)(q, kv, kv).block_until_ready()
g = jax.jit(jax.grad(lambda q: k(q, kv, kv).astype(jnp.float32).sum()))(q).block_until_ready()
print("SPLASH_OK", out.shape, float(g.astype(jnp.float32).sum()) == float(g.astype(jnp.float32).sum()))
"""


def choose_attention():
    """attention="auto": Kaggle's TPU image ships a libtpu too old for Pallas kernels. Install the libtpu that
    matches the installed jax into a side directory, test splash attention (forward + backward) in a separate
    process, and only if that passes point this process at the new libtpu. Otherwise plain XLA attention.
    Must run before `import jax`; never raises."""
    if CONFIG.get("attention") != "auto" and not any(r.get("attention") == "auto" for r in CONFIG.get("runs", [])):
        return
    os.environ["OPHYX_ATTENTION"] = "xla"
    if not os.path.isdir("/kaggle") or not os.environ.get("TPU_ACCELERATOR_TYPE", os.environ.get("ISTPUVM")):
        return
    import importlib.metadata as md
    import subprocess
    import sys
    try:
        tgt = "/kaggle/tmp/libtpu_new"
        v = md.version("jax")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", tgt, f"jax[tpu]=={v}"],
                           capture_output=True, text=True, timeout=900)
        libs = sorted(glob.glob(os.path.join(tgt, "libtpu*", "**", "libtpu*.so"), recursive=True))
        if not libs:
            print(f"[attention] no libtpu installed (pip rc {r.returncode}: {r.stderr[-300:]}); using xla", flush=True)
            return
        env = {**os.environ, "TPU_LIBRARY_PATH": libs[0]}
        t = subprocess.run([sys.executable, "-c", SPLASH_PROBE], capture_output=True, text=True, timeout=900, env=env)
        if "SPLASH_OK" in t.stdout:
            os.environ["TPU_LIBRARY_PATH"] = libs[0]
            os.environ["OPHYX_ATTENTION"] = "splash"
            print(f"[attention] splash works with {libs[0]}", flush=True)
        else:
            print(f"[attention] splash probe failed; using xla: {(t.stdout + t.stderr)[-400:]}", flush=True)
    except Exception as e:  # anything goes wrong: stay on the image's libtpu and xla attention
        print(f"[attention] probe error {type(e).__name__}: {e}; using xla", flush=True)


choose_attention()

import jax  # noqa: E402  (after choose_attention, which may switch libtpu)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402
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


_LOG_TAIL = []  # last log lines, shipped with the heartbeat
PHASE = {"phase": "start", "run": None, "step": None}


def log(*a):
    line = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    _LOG_TAIL.append(line[:300])
    del _LOG_TAIL[:-60]


def load_keys():
    paths = glob.glob("/kaggle/input/**/keys.json", recursive=True) + glob.glob(CONFIG.get("keys_path", "/nonexistent"))
    return json.load(open(paths[0])) if paths else {}


KEYS = load_keys()
_HEARTBEAT = {"started": False}


def set_phase(phase, **kw):
    PHASE.update(phase=phase, **kw)
    log(f"phase: {phase}" + (f" {kw}" if kw else ""))


def start_heartbeat(every_s=600):
    """Kaggle shows no logs while a session runs; a 3-hour session once looked alive but produced nothing.
    Every few minutes, write phase / step / recent log lines to HF so a session is never a black box."""
    repo = CONFIG.get("heartbeat_repo", "abhash-chakraborty/ophyx-runs")
    if _HEARTBEAT["started"] or not KEYS.get("HF_TOKEN") or not (KAGGLE or CONFIG.get("heartbeat_local")):
        return
    every_s = CONFIG.get("heartbeat_every_s", every_s)
    _HEARTBEAT["started"] = True
    import threading
    name = f"sessions/{CONFIG.get('session_name', 'session')}-{time.strftime('%Y%m%d-%H%M', time.gmtime(T0))}.json"

    def beat():
        from huggingface_hub import HfApi
        api = HfApi(token=KEYS["HF_TOKEN"])
        notify(f"session {CONFIG.get('session_name', 'session')} started", f"script up after {time.time() - T0:.0f}s")
        warned = False
        while True:
            body = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "uptime_s": round(time.time() - T0),
                    **PHASE, "tail": _LOG_TAIL[-40:]}
            try:  # copy in the session output, readable after the session even if HF uploads never work
                with open(os.path.join("/kaggle/working" if KAGGLE else out_dir(), "heartbeat.json"), "w") as f:
                    json.dump(body, f, indent=1)
            except Exception:
                pass
            try:
                api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
                api.upload_file(path_or_fileobj=json.dumps(body, indent=1).encode(), path_in_repo=name,
                                repo_id=repo, repo_type="model",
                                commit_message=f"heartbeat: {PHASE.get('phase')} step {PHASE.get('step')}")
            except Exception as e:
                if not warned:
                    log(f"heartbeat failed: {type(e).__name__}: {str(e)[:200]}")
                    notify("heartbeat failed", f"{type(e).__name__}: {str(e)[:300]}", "warning")
                    warned = True
            time.sleep(every_s)

    threading.Thread(target=beat, daemon=True).start()


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
    if kind == "auto":
        kind = os.environ.get("OPHYX_ATTENTION", "xla")
        log(f"attention: {kind}")
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

    def ce_chunk_w(emb, h, y, w):  # SFT: loss only where w > 0 (the answer)
        logits = (h @ emb.T).astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        return jnp.sum(w * (lse - tgt)), jnp.sum(w * lse ** 2), jnp.sum(w)

    ce_chunk_w = jax.checkpoint(ce_chunk_w)

    def loss_fn(params, batch, w=None):
        if w is not None:
            return weighted(params, batch, w)
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

    def weighted(params, batch, w):
        x, y = batch[:, :-1], batch[:, 1:]
        h = hidden(params, x)
        emb = params["embed"].astype(jnp.dtype(cfg("dtype", "bfloat16")))
        B = h.shape[0]
        n = T // chunk
        hc = h.reshape(B, n, chunk, -1).transpose(1, 0, 2, 3)
        yc = y.reshape(B, n, chunk).transpose(1, 0, 2)
        wc = w.reshape(B, n, chunk).transpose(1, 0, 2)
        ce_sum, z_sum, w_sum = jax.lax.map(lambda a: ce_chunk_w(emb, a[0], a[1], a[2]), (hc, yc, wc))
        ntok = jnp.maximum(1.0, jnp.sum(w_sum))
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


FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = 3, 4, 5  # ids fixed by the tokenizer's special-token order
USER_ID = 1  # <|user|>: every SFT example starts with it


class Data:
    """Random windows from uint16 token shards, mixed by bucket weight. batch(step) is deterministic.

    spec["phases"] (optional): [{"until": 0.85, "weights": {bucket: w}}, {"until": 1.0, "weights": {...}}]
    switches the mix by training progress (e.g. tier A heavy during the LR decay). spec["fim_rate"]
    rearranges that share of windows from buckets listed in spec["fim_buckets"] into
    prefix/suffix/middle order (PSM) with the FIM sentinel tokens.

    SFT buckets add "mask_glob" (uint8 files parallel to the token files: 1 = answer token, loss counts) and
    "aligned": true (windows start where an example starts, at a <|user|> token, so no prompt is cut off).
    With any masked bucket, batch() returns (tokens, weights); otherwise just tokens, exactly as before.
    """

    def __init__(self, spec, seq, seed, total_steps=None):
        self.seq, self.seed, self.total_steps = seq, seed, total_steps
        # copies: a numpy array written back into CONFIG makes the checkpoint meta.json unserializable
        self.phases = [dict(ph) for ph in spec["phases"]] if spec.get("phases") else None
        self.fim_rate = float(spec.get("fim_rate", 0.0))
        self.fim_buckets = set(spec.get("fim_buckets", []))
        self.buckets = []
        self.extra = {}  # bucket name -> (mask maps, start positions per map) for SFT buckets
        for name, b in spec["buckets"].items():
            files = sorted(glob.glob(b["glob"], recursive=True))
            if not files:
                raise FileNotFoundError(f"bucket {name}: nothing matches {b['glob']}")
            maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
            sizes = np.array([len(m) - seq - 1 for m in maps], dtype=np.int64)
            keep = sizes > 0
            if b.get("mask_glob"):
                mfiles = sorted(glob.glob(b["mask_glob"], recursive=True))
                by_name = {os.path.basename(f): f for f in mfiles}
                masks = [np.memmap(by_name[os.path.basename(f)], dtype=np.uint8, mode="r") for f in files]
                assert all(len(m) == len(t) for m, t in zip(masks, maps)), f"bucket {name}: mask/token length mismatch"
                starts = None
                if b.get("aligned"):
                    starts = [np.flatnonzero(np.asarray(t) == USER_ID) for t in maps]
                    starts = [st[st < len(t) - seq - 1] for st, t in zip(starts, maps)]
                if starts:  # sample files by number of usable example starts
                    sizes = np.array([len(st) for st in starts], dtype=np.int64)
                    keep = keep & (sizes > 0)
                kept = [i for i, k in enumerate(keep) if k]  # the same files as the token maps below, in order
                self.extra[name] = ([masks[i] for i in kept], [starts[i] for i in kept] if starts else None)
            self.buckets.append((name, float(b["weight"]), [m for m, k in zip(maps, keep) if k], sizes[keep]))
            log(f"data bucket {name}: {len(files)} shards, {sum(len(m) for m in maps) / 1e6:.1f}M tokens, weight {b['weight']}"
                + (" (masked" + (", example-aligned)" if b.get("aligned") else ")") if b.get("mask_glob") else ""))
        self.has_mask = bool(self.extra)
        w = np.array([b[1] for b in self.buckets])
        self.weights = w / w.sum()
        self.names = [b[0] for b in self.buckets]
        if self.phases:
            for ph in self.phases:
                pw = np.array([float(ph["weights"].get(n, 0.0)) for n in self.names])
                ph["_w"] = pw / pw.sum()
                log(f"data phase until {ph['until']}: " + ", ".join(f"{n} {x:.3f}" for n, x in zip(self.names, ph["_w"])))
        self.val = None
        self.val_masks = None
        if spec.get("val_glob"):
            vf = sorted(glob.glob(spec["val_glob"], recursive=True))
            self.val = [np.memmap(f, dtype=np.uint16, mode="r") for f in vf]
            keep = [len(m) > seq + 1 for m in self.val]  # like the buckets: a shard shorter than one window can't be sampled
            vf, self.val = [f for f, k in zip(vf, keep) if k], [m for m, k in zip(self.val, keep) if k]
            if not all(keep):
                log(f"val: dropped {keep.count(False)} shards shorter than {seq + 1} tokens, {len(vf)} left")
            if spec.get("val_mask_glob"):
                vm = {os.path.basename(f): f for f in glob.glob(spec["val_mask_glob"], recursive=True)}
                self.val_masks = [np.memmap(vm[os.path.basename(f)], dtype=np.uint8, mode="r") for f in vf]

    def windows(self, rng, n, maps, sizes):
        probs = sizes / sizes.sum()
        which = rng.choice(len(maps), size=n, p=probs)
        out = np.empty((n, self.seq + 1), np.int32)
        for i, s in enumerate(which):
            off = rng.integers(0, sizes[s])
            out[i] = maps[s][off: off + self.seq + 1]
        return out

    def weights_at(self, step):
        if not self.phases or not self.total_steps:
            return self.weights
        frac = step / self.total_steps
        for ph in self.phases:
            if frac < ph["until"]:
                return ph["_w"]
        return self.phases[-1]["_w"]

    def fim(self, rng, w):
        """PSM on one window of seq+1 tokens: drop 3 tokens from the end to make room for the sentinels."""
        body = w[: self.seq + 1 - 3]
        a, b = sorted(rng.integers(0, len(body) + 1, size=2))
        return np.concatenate([[FIM_PREFIX], body[:a], [FIM_SUFFIX], body[b:], [FIM_MIDDLE], body[a:b]]).astype(np.int32)

    def masked_windows(self, rng, n, name, maps, sizes):
        masks, starts = self.extra[name]
        probs = sizes / sizes.sum()
        which = rng.choice(len(maps), size=n, p=probs)
        out = np.empty((n, self.seq + 1), np.int32)
        wts = np.empty((n, self.seq + 1), np.float32)
        for i, s in enumerate(which):
            off = starts[s][rng.integers(0, len(starts[s]))] if starts else rng.integers(0, sizes[s])
            out[i] = maps[s][off: off + self.seq + 1]
            wts[i] = masks[s][off: off + self.seq + 1]
        return out, wts

    def batch(self, step, n):
        rng = np.random.default_rng([self.seed, step])
        counts = rng.multinomial(n, self.weights_at(step))
        parts, wparts = [], []
        for c, b in zip(counts, self.buckets):
            if not c:
                continue
            if b[0] in self.extra:
                ws, wt = self.masked_windows(rng, c, b[0], b[2], b[3])
            else:
                ws = self.windows(rng, c, b[2], b[3])
                wt = np.ones(ws.shape, np.float32) if self.has_mask else None
                if self.fim_rate and b[0] in self.fim_buckets:
                    for i in np.where(rng.random(c) < self.fim_rate)[0]:
                        ws[i] = self.fim(rng, ws[i])
            parts.append(ws)
            wparts.append(wt)
        x = np.concatenate(parts)
        perm = rng.permutation(n)
        if not self.has_mask:
            return x[perm]
        return x[perm], np.concatenate(wparts)[perm][:, 1:]  # weight for predicting token t+1

    def val_batches(self, n, k=4):
        if not self.val:
            return []
        rng = np.random.default_rng([self.seed, 10 ** 9])
        sizes = np.array([len(m) - self.seq - 1 for m in self.val])
        if self.val_masks is None:
            return [self.windows(rng, n, self.val, sizes) for _ in range(k)]
        out = []
        for _ in range(k):
            which = rng.choice(len(self.val), size=n, p=sizes / sizes.sum())
            x = np.empty((n, self.seq + 1), np.int32)
            w = np.empty((n, self.seq + 1), np.float32)
            for i, s in enumerate(which):
                off = rng.integers(0, sizes[s])
                x[i] = self.val[s][off: off + self.seq + 1]
                w[i] = self.val_masks[s][off: off + self.seq + 1]
            out.append((x, w[:, 1:]))
        return out


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
        json.dump({"step": step, "config": CONFIG, **extra}, f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))  # never lose a save to meta
    if os.path.isdir(d):  # same step saved before (e.g. a finished run re-launched): replace it
        shutil.rmtree(d)
    os.replace(tmp, d)
    for old in sorted(glob.glob(os.path.join(out_dir(), "ckpt-*")))[:-2]:  # keep the latest two
        if not old.endswith(".tmp"):
            shutil.rmtree(old, ignore_errors=True)
    log(f"checkpoint saved: step {step}")
    return d


# Off-Kaggle copy of the latest checkpoint, so a session that dies before writing its output
# can't send the chain back to an older step. CONFIG["hf_ckpt_repo"] enables it.
_backup_thread = None


def hf_backup(ckpt_dir, step):
    repo = cfg("hf_ckpt_repo", None)
    if not repo or not KEYS.get("HF_TOKEN"):
        return
    import threading

    def push():
        from huggingface_hub import HfApi
        api = HfApi(token=KEYS["HF_TOKEN"])
        prefix = f"{cfg('run_name', 'run')}/latest"
        for attempt in range(5):
            try:
                api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
                api.upload_folder(repo_id=repo, repo_type="model", folder_path=ckpt_dir, path_in_repo=prefix,
                                  commit_message=f"checkpoint step {step}", delete_patterns=["*"])
                try:  # keep only the latest checkpoint in history, or storage grows ~4 GB per backup
                    api.super_squash_history(repo_id=repo, repo_type="model")
                except Exception as e:
                    log("history squash failed:", e)
                log(f"checkpoint step {step} backed up to HF")
                return
            except Exception as e:
                log(f"HF backup failed ({type(e).__name__}: {str(e)[:150]}), retry")
                time.sleep(60 * (attempt + 1))

    global _backup_thread
    if not wait_backup(1800):  # never two uploads at once, and a hung upload must not stall training
        log(f"previous HF backup still running after 30 min; skipping the backup of step {step}")
        notify("HF backup stuck", f"skipped the backup of step {step}", "warning")
        return
    _backup_thread = threading.Thread(target=push, daemon=True)
    _backup_thread.start()


def wait_backup(timeout):
    if _backup_thread is not None and _backup_thread.is_alive():
        _backup_thread.join(timeout)
    return _backup_thread is None or not _backup_thread.is_alive()


def hf_restore_if_newer(found):
    repo = cfg("hf_ckpt_repo", None)
    if not repo or not KEYS.get("HF_TOKEN"):
        return found
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        prefix = f"{cfg('run_name', 'run')}/latest"
        meta = json.load(open(hf_hub_download(repo, f"{prefix}/meta.json", token=KEYS["HF_TOKEN"])))
        if found is None or meta["step"] > found[0]:
            d = snapshot_download(repo, allow_patterns=[f"{prefix}/*"], token=KEYS["HF_TOKEN"],
                                  local_dir=os.path.join("/kaggle/tmp" if KAGGLE else out_dir(), "hf_restore"))
            log(f"HF backup (step {meta['step']}) is newer than attached ({found and found[0]}); using it")
            return meta["step"], os.path.join(d, prefix)
    except Exception as e:
        log(f"no HF checkpoint used ({type(e).__name__}: {str(e)[:120]})")
    return found


def publish_status(status):
    """Small status file the watchdog reads (it can't download multi-GB kernel outputs)."""
    repo = cfg("hf_ckpt_repo", None)
    if not repo or not KEYS.get("HF_TOKEN"):
        return
    try:
        from huggingface_hub import HfApi
        p = os.path.join(out_dir(), "status.json")
        HfApi(token=KEYS["HF_TOKEN"]).upload_file(path_or_fileobj=p, path_in_repo=f"{cfg('run_name', 'run')}/status.json",
                                                  repo_id=repo, repo_type="model", commit_message=f"status: {status['reason']} at {status['step']}")
    except Exception as e:
        log("status publish failed:", e)


def find_latest_ckpt():
    cands = glob.glob(os.path.join(out_dir(), "ckpt-*/meta.json"))
    if cfg("resume_inputs", True):
        cands += glob.glob("/kaggle/input/**/ckpt-*/meta.json", recursive=True)
    cands += [p for p in glob.glob(os.path.join(cfg("resume_dir", "/nonexistent"), "**/ckpt-*/meta.json"), recursive=True)]
    best = None
    for m in cands:
        try:
            meta = json.load(open(m))
            step = meta["step"]
        except Exception:
            continue
        if meta.get("config", {}).get("run_name", cfg("run_name", None)) != cfg("run_name", None):
            continue  # an attached session output can hold several runs' checkpoints
        if best is None or step > best[0]:
            best = (step, os.path.dirname(m))
    return best


def fetch_hf_data():
    """SFT packs (a few GB) come straight from the HF dataset: quicker than attaching them as Kaggle inputs,
    which can add hours of setup before the script starts."""
    spec = cfg("hf_data", None)
    if not spec:
        return
    from huggingface_hub import snapshot_download
    t = time.time()
    d = snapshot_download(spec["repo"], repo_type="dataset", allow_patterns=spec["patterns"], token=KEYS.get("HF_TOKEN"),
                          local_dir=spec.get("dir") or ("/kaggle/tmp/hf_data" if KAGGLE else os.path.join(out_dir(), "hf_data")),
                          max_workers=16)
    n = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(f))
    log(f"hf_data: {n / 1e9:.2f} GB from {spec['repo']} into {d} in {time.time() - t:.0f}s")


def load_params(src, like_params):
    """Weights only from a checkpoint dir (or glob), or "hf://<repo>/<run>/latest" (the trainer's HF backup)."""
    if src.startswith("hf://"):
        from huggingface_hub import snapshot_download
        repo, prefix = src[5:].split("/", 2)[:2], src[5:].split("/", 2)[2]
        repo = "/".join(repo)
        d = snapshot_download(repo, allow_patterns=[f"{prefix}/params.npz", f"{prefix}/meta.json"], token=KEYS.get("HF_TOKEN"),
                              local_dir=os.path.join("/kaggle/tmp" if KAGGLE else out_dir(), "init"))
        path = os.path.join(d, prefix)
    else:
        cands = sorted(glob.glob(src, recursive=True))
        if not cands:
            raise FileNotFoundError(f"init_from: nothing matches {src}")
        path = cands[-1]
    z = np.load(os.path.join(path, "params.npz"))
    leaves, treedef = jax.tree_util.tree_flatten(like_params)
    new = [z[f"a{i:05d}"] for i in range(len(leaves))]
    for a, b in zip(new, leaves):
        assert a.shape == b.shape, f"init_from shape mismatch {a.shape} vs {b.shape}"
    return jax.tree_util.tree_unflatten(treedef, new)


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
    toks = glob.glob("/kaggle/input/**/tokenizer.json", recursive=True) + glob.glob(cfg("tokenizer_path", "/nonexistent"))
    if toks:  # so AutoTokenizer / evalplus can load the export directly
        shutil.copy(toks[0], os.path.join(out_dir, "tokenizer.json"))
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|end|>", "pad_token": "<|pad|>",
                   "bos_token": None, "model_max_length": seq, "clean_up_tokenization_spaces": False},
                  open(os.path.join(out_dir, "tokenizer_config.json"), "w"), indent=1)
    log("exported HF checkpoint to", out_dir)


# ---------------------------------------------------------------- training loop


def train(data=None):
    start_heartbeat()
    set_phase("devices", run=cfg("run_name", None), step=None)
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
    set_phase("data")
    if data is None:
        fetch_hf_data()
    data = data or Data(cfg("data", {}), seq, cfg("seed", 0), total_steps)
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

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step_w(params, opt_state, batches, weights):  # SFT: + weights [accum, micro, seq]
        def micro_step(carry, bw):
            g_acc, ce_acc = carry
            (_, ce), g = jax.value_and_grad(loss_fn, has_aux=True)(params, bw[0], bw[1])
            return (jax.tree.map(jnp.add, g_acc, g), ce_acc + ce), None

        zeros = jax.tree.map(jnp.zeros_like, params)
        (g, ce), _ = jax.lax.scan(micro_step, (zeros, jnp.zeros([], jnp.float32)), (batches, weights))
        g = jax.tree.map(lambda x: x / accum, g)
        gnorm = optax.global_norm(g)
        updates, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, ce / accum, gnorm

    @jax.jit
    def eval_ce(params, b):
        return loss_fn(params, b)[1]

    @jax.jit
    def eval_ce_w(params, b, w):
        return loss_fn(params, b, w)[1]

    set_phase("init_params")
    params = jax.jit(lambda k: init_params(dims, vocab, k), out_shardings=repl)(jax.random.PRNGKey(cfg("seed", 0)))
    opt_state = jax.jit(opt.init, out_shardings=repl)(params)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    start = 0
    set_phase("restore")
    found = hf_restore_if_newer(find_latest_ckpt())
    if found:
        start, path = found
        p, o = load_ckpt(path, (params, opt_state))
        params, opt_state = jax.device_put(p, repl), jax.device_put(o, repl)
        log(f"resumed from {path} at step {start}")
    elif cfg("init_from", None):  # SFT and later stages: weights from another run, fresh optimizer and schedule
        params = jax.device_put(load_params(cfg("init_from", None), params), repl)
        log(f"initialised weights from {cfg('init_from', None)}")
    log(f"{n_params / 1e6:.1f}M parameters")
    if start >= total_steps:  # already finished: nothing to train, don't re-save or re-export
        log(f"run already finished at step {start}; skipping")
        status = {"step": start, "total_steps": total_steps, "finished": True, "reason": "already_done",
                  "hours": 0.0, "tokens_per_s": None, "n_params": n_params, "final_loss": None}
        json.dump(status, open(os.path.join(out_dir(), "status.json"), "w"))
        publish_status(status)  # the session that finished may have failed to publish; the watchdog waits on this
        return params, status

    set_phase("training", step=start)
    use_wandb = bool(cfg("wandb", None)) and KEYS.get("WANDB_API_KEY")
    if use_wandb:
        try:
            os.environ["WANDB_API_KEY"] = KEYS["WANDB_API_KEY"]
            import wandb
            wandb.init(project=cfg("wandb", "ophyx"), name=cfg("run_name", None), id=cfg("run_id", None),
                       resume="allow", config=CONFIG)
        except Exception as e:
            log("wandb disabled:", e)
            use_wandb = False

    deadline = T0 + cfg("budget_s", 11 * 3600) - cfg("margin_s", 1500)
    last_ckpt, metrics_f = time.time(), open(os.path.join(out_dir(), "metrics.jsonl"), "a")
    session_steps = cfg("max_steps_this_session", None)
    last_rate = None
    masked = getattr(data, "has_mask", False)
    if masked:
        val = [(jax.device_put(x, shard), jax.device_put(w, shard)) for x, w in data.val_batches(micro)]
    else:
        val = [jax.device_put(v, shard) for v in data.val_batches(micro)]
    seq_shard = NamedSharding(mesh, P(None, "data", None))

    def next_batch(s):
        b = data.batch(s, accum * micro)
        if masked:
            return b[0].reshape(accum, micro, seq + 1), b[1].reshape(accum, micro, seq)
        return b.reshape(accum, micro, seq + 1)

    step, t_log, tok_log = start, time.time(), 0
    nxt = next_batch(step)
    stop_reason = "done"
    while step < total_steps:
        if masked:
            params, opt_state, ce, gnorm = train_step_w(params, opt_state, jax.device_put(nxt[0], seq_shard),
                                                        jax.device_put(nxt[1], seq_shard))
        else:
            batches = jax.device_put(nxt, seq_shard)
            params, opt_state, ce, gnorm = train_step(params, opt_state, batches)
        step += 1
        if step < total_steps:
            nxt = next_batch(step)  # overlaps device compute
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
                row["val_loss"] = round(float(np.mean([float(eval_ce_w(params, *v)) if masked else float(eval_ce(params, v))
                                                       for v in val])), 4)
            log(json.dumps(row))
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            if use_wandb:
                wandb.log(row, step=step)
            last_rate = row["tokens_per_s"]
            PHASE.update(step=step, loss=row["loss"], tokens_per_s=last_rate)
            t_log, tok_log = time.time(), 0
        if session_steps and step - start >= session_steps:
            stop_reason = "session_steps"
            break
        if cfg("save", True) and time.time() - last_ckpt > cfg("ckpt_every_s", 1800):
            if not math.isfinite(float(ce)):  # loss is only logged every log_every steps; never back up a NaN
                stop_reason = "nan"
                log("non-finite loss before a checkpoint, stopping without saving", {"step": step})
                break
            hf_backup(save_ckpt((params, opt_state), step, {"n_params": n_params}), step)
            last_ckpt = time.time()
        if time.time() > deadline:
            stop_reason = "budget"
            break
    set_phase("saving", step=step)
    if stop_reason != "nan" and cfg("save", True):
        hf_backup(save_ckpt((params, opt_state), step, {"n_params": n_params}), step)
    if not wait_backup(1800):
        log("final HF backup did not finish within 30 min; the checkpoint is still in the session output")
    status = {"step": step, "total_steps": total_steps, "finished": step >= total_steps, "reason": stop_reason,
              "hours": round((time.time() - T0) / 3600, 2), "tokens_per_s": last_rate, "n_params": n_params,
              "final_loss": round(float(ce), 4)}
    json.dump(status, open(os.path.join(out_dir(), "status.json"), "w"))
    publish_status(status)
    if status["finished"] and cfg("export_hf", False):
        export_hf(params, dims, vocab, seq, os.path.join(out_dir(), "hf"), eos_id=cfg("eos_id", 0))
    if cfg("notify", True):
        notify(f"train {cfg('run_name', 'run')}: {stop_reason}", json.dumps(status),
               "white_check_mark" if stop_reason != "nan" else "warning")
    return params, status


if __name__ == "__main__" and not CONFIG.get("library"):
    train()

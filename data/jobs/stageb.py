"""Data v1, stage B: global dedup, decontamination, tokenizer, tokenisation. Kaggle CPU sessions.

Jobs (CONFIG["job"]), run in this order:
  sig        per part: MinHash signature (128 perms, word 5-grams) + 10-gram benchmark overlap
  dedup      exact + near-dup (LSH 16x8, Jaccard >= 0.7) across all sources, best source wins;
             drop benchmark matches (n-gram hits or MinHash near-dups of benchmark items);
             assign buckets; write keep-list and report
  tokenizer  train the 32k byte-level BPE on a sample of the kept mix; compare chars/token
  tokenize   per part: kept docs -> uint16 token files per bucket (+ 0.5% held out)

Inputs and outputs live in the private HF dataset (v1/raw -> v1/sig, v1/dedup, v1/tokenizer, v1/tokens).
"""

import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
import traceback
import urllib.request
import zlib

import numpy as np

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
JOB = CONFIG["job"]
SHARD, NSHARDS = CONFIG.get("shard", 0), CONFIG.get("nshards", 1)
REPO = CONFIG.get("repo", "abhash-chakraborty/ophyx-data")
V = CONFIG.get("version", "v1")  # input version (raw parts)
OV = CONFIG.get("out_version", V)  # where this run writes; "smoke" for test runs
SMOKE = OV == "smoke"
WORK = CONFIG.get("work", "/kaggle/tmp/stageb" if os.path.isdir("/kaggle") else os.path.abspath("stageb-work"))
OUTW = "/kaggle/working" if os.path.isdir("/kaggle") else WORK
os.makedirs(WORK, exist_ok=True)
T0 = time.time()

PERMS, BANDS, ROWS = 128, 16, 8
SHINGLE, NGRAM = 5, 10
PRIME = np.uint64((1 << 61) - 1)
_rng = np.random.default_rng(42)
A = _rng.integers(1, (1 << 61) - 1, PERMS, dtype=np.uint64)
B = _rng.integers(0, (1 << 61) - 1, PERMS, dtype=np.uint64)
WORD = re.compile(r"\w+")
SPECIALS = ["<|end|>", "<|user|>", "<|python|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|file_sep|>", "<|repo|>", "<|pad|>"]
SOURCE_RANK = {"cpython_lib": 0, "cpython_docs": 0, "peps_docs": 0, "opencoder_algorithmic_corpus": 1,
               "opencoder_synthetic_code_snippet": 1, "opencoder_synthetic_qa": 1, "stack_edu": 2, "fineweb_edu": 3}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_keys():
    p = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    if p:
        return json.load(open(p[0]))
    return {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}


KEYS = load_keys()


def notify(title, msg, tags="robot"):
    topic = KEYS.get("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode()[:3000],
                                                          headers={"Title": title, "Tags": tags}, method="POST"), timeout=15)
        except Exception as e:
            log("ntfy failed:", e)


def hf():
    from huggingface_hub import HfApi
    return HfApi(token=KEYS["HF_TOKEN"])


def download(path_in_repo):
    """Plain files under WORK/dl (no cache symlinks), so os.remove really frees the disk."""
    local = os.path.join(WORK, "dl", path_in_repo)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path_in_repo, repo_type="dataset", token=KEYS["HF_TOKEN"],
                           local_dir=os.path.join(WORK, "dl"))


def commit(files, message):
    """files: list of (local_path, path_in_repo). Retries with backoff."""
    from huggingface_hub import CommitOperationAdd
    for i in range(0, len(files), 40):
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=l) for l, p in files[i:i + 40]]
        for attempt in range(8):
            try:
                hf().create_commit(REPO, repo_type="dataset", operations=ops, commit_message=message)
                break
            except Exception as e:
                wait = min(600, 30 * 2 ** attempt)
                log(f"commit failed ({type(e).__name__}: {str(e)[:200]}), retry in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("HF commit kept failing")


def raw_parts():
    fs = hf().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in fs if f.startswith(f"{V}/raw/") and f.endswith(".parquet") and "/_manifests/" not in f), set(fs)


def my_parts(parts):
    return [p for p in parts if int(hashlib.md5(p.encode()).hexdigest(), 16) % NSHARDS == SHARD]


# ---------------------------------------------------------------- MinHash and n-grams


def minhash(text):
    toks = WORD.findall(text.lower())
    if len(toks) < SHINGLE:
        toks = toks + [""] * (SHINGLE - len(toks))
    sh = [" ".join(toks[i:i + SHINGLE]) for i in range(0, len(toks) - SHINGLE + 1)]
    if len(sh) > 8000:  # long files: an even sample keeps the estimate unbiased enough
        sh = sh[:: len(sh) // 8000 + 1]
    h = np.fromiter((zlib.crc32(s.encode()) for s in set(sh)), dtype=np.uint64)
    v = (h[:, None] * A[None, :] + B[None, :]) % PRIME
    return np.min(v & np.uint64(0xFFFFFFFF), axis=0).astype(np.uint32)


def ngrams(text):
    toks = WORD.findall(text.lower())
    return {" ".join(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1)}


# ---------------------------------------------------------------- benchmarks

BENCH = None  # {ngram: problem_id}, built once in the parent before forking


def bench_items():
    """Yields (problem_id, text) for every eval problem: prompts plus reference solutions."""
    import gzip
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    def gh_jsonl(url):
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ophyx"}), timeout=120).read()
        return [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()]

    def hf_records(repo, keep=lambda f: True):
        out = []
        for f in list_repo_files(repo, repo_type="dataset", token=KEYS["HF_TOKEN"]):
            if not keep(f) or not f.endswith((".parquet", ".jsonl", ".json")):
                continue
            p = hf_hub_download(repo, f, repo_type="dataset", token=KEYS["HF_TOKEN"], local_dir=os.path.join(WORK, "bench", repo))
            if p.endswith(".parquet"):
                out += pq.read_table(p).to_pylist()
            else:
                with open(p) as fh:
                    txt = fh.read()
                try:
                    obj = json.loads(txt)
                    out += obj if isinstance(obj, list) else [obj]
                except json.JSONDecodeError:
                    out += [json.loads(l) for l in txt.splitlines() if l.strip()]
        return out

    specs = [
        ("humaneval+", lambda: gh_jsonl("https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"),
         ["prompt", "canonical_solution"], "task_id"),
        ("mbpp+", lambda: gh_jsonl("https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"),
         ["prompt", "canonical_solution"], "task_id"),
        ("mbpp", lambda: hf_records("google-research-datasets/mbpp", lambda f: f.startswith("full/")), ["text", "code"], "task_id"),
        ("bigcodebench", lambda: hf_records("bigcode/bigcodebench"), ["complete_prompt", "canonical_solution"], "task_id"),
        ("cruxeval", lambda: hf_records("cruxeval-org/cruxeval"), ["code"], "id"),
        ("ds1000", lambda: hf_records("xlangai/DS-1000"), ["prompt", "reference_code"], None),
        ("livecodebench", lambda: hf_records("livecodebench/code_generation_lite", lambda f: f.endswith(".jsonl")),
         ["question_content", "starter_code"], "question_id"),
        ("apps_test", lambda: hf_records("codeparrot/apps", lambda f: "test" in f), ["question", "solutions"], "problem_id"),
        # our private concept suite (eval/concept_suite), uploaded to the private data repo
        ("concept_suite", lambda: [json.loads(l) for l in open(hf_hub_download(REPO, "eval/concept_suite/tasks.jsonl",
                                   repo_type="dataset", token=KEYS["HF_TOKEN"], local_dir=os.path.join(WORK, "bench", "concept")))],
         ["prompt", "reference"], "task_id"),
    ]
    report = {}
    if SMOKE:
        specs = specs[:2]
    for name, fetch, fields, idf in specs:
        try:
            recs = fetch()
            n = 0
            for i, r in enumerate(recs):
                parts = []
                for f in fields:
                    v = r.get(f)
                    if isinstance(v, str) and v.startswith("[") and f == "solutions":
                        try:
                            v = "\n".join(json.loads(v)[:3])  # APPS: first few solutions
                        except Exception:
                            pass
                    if isinstance(v, str):
                        parts.append(v)
                if parts:
                    n += 1
                    yield f"{name}/{r.get(idf, i) if idf else i}", "\n".join(parts)
            report[name] = n
        except Exception as e:
            report[name] = f"failed: {type(e).__name__}: {str(e)[:200]}"
        log("benchmark", name, report[name])
    bench_items.report = report


def build_bench_index():
    owners = collections.defaultdict(set)
    items = []
    for pid, text in bench_items():
        items.append((pid, text))
        for g in ngrams(text):
            owners[g].add(pid)
    # n-grams shared by more than 3 problems are generic code, not evidence of copying
    index = {g: next(iter(p)) for g, p in owners.items() if len(p) <= 3}
    log(f"benchmark index: {len(items)} problems, {len(index):,} distinctive 10-grams")
    return index, items


def contamination(text):
    hits = collections.Counter(BENCH[g] for g in ngrams(text) if g in BENCH)
    if not hits:
        return None, 0
    pid, n = hits.most_common(1)[0]
    return (pid if n >= 3 else None), n


def _sig_row(text):
    pid, n = contamination(text)
    return minhash(text).tobytes(), pid, n


# ---------------------------------------------------------------- job: sig


def job_sig():
    global BENCH
    import pyarrow as pa
    import pyarrow.parquet as pq

    BENCH, _ = build_bench_index()
    parts, existing = raw_parts()
    mine = my_parts(parts)
    if SMOKE:
        mine = CONFIG.get("smoke_parts") or parts[:2]
    log(f"{len(mine)} of {len(parts)} parts for shard {SHARD}")
    pending, done_rows = [], 0
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for k, part in enumerate(mine):
            out = part.replace(f"{V}/raw/", f"{OV}/sig/")
            if out in existing:
                continue
            t = time.time()
            tab = pq.read_table(download(part), columns=["id", "source", "kind", "text", "quality", "repo"]).to_pylist()
            res = pool.map(_sig_row, [r["text"] for r in tab], chunksize=64)
            table = pa.table({
                "id": [r["id"] for r in tab], "source": [r["source"] for r in tab], "kind": [r["kind"] for r in tab],
                "quality": [r["quality"] for r in tab], "n_chars": [len(r["text"]) for r in tab],
                "repo": [r["repo"] for r in tab], "sig": [x[0] for x in res],
                "contam": [x[1] for x in res], "contam_hits": [x[2] for x in res], "part": [part] * len(tab)})
            local = os.path.join(WORK, out.replace("/", "__"))
            pq.write_table(table, local, compression="zstd")
            pending.append((local, out))
            done_rows += len(tab)
            log(f"[{k + 1}/{len(mine)}] {part}: {len(tab)} docs, {sum(1 for x in res if x[1])} flagged, {time.time() - t:.0f}s")
            os.remove(download(part))
            if len(pending) >= 20:
                commit(pending, f"sig shard {SHARD}")
                pending = []
    if pending:
        commit(pending, f"sig shard {SHARD}")
    return {"docs": done_rows, "parts": len(mine)}


# ---------------------------------------------------------------- job: dedup


def bucket_of(source, kind, quality, stack_cut, github_cut=99.0):
    if source == "stack_edu":
        return "stack_edu_a" if (quality or 0) >= stack_cut else "stack_edu_b"
    if source == "github":  # repo-level quality score (tests, CI, typing, docstrings, lint)
        return "github_a" if (quality or 0) >= github_cut else "github_b"
    if source in ("cpython_docs", "peps_docs"):
        return "python_docs"
    if source.startswith("opencoder_"):
        return source.replace("opencoder_", "oc_")
    return source


def _load_sigs(paths):
    """Signature parts -> numpy arrays, no per-document Python objects (10.9M docs as Python lists OOM'd 31 GB)."""
    import pyarrow.parquet as pq

    src_names, part_names = [], []
    code_src, code_part = {}, {}
    acc = collections.defaultdict(list)
    contam = {}
    base = 0
    for p in paths:
        t = pq.read_table(p, columns=["id", "source", "kind", "quality", "n_chars", "sig", "contam", "part"])
        n = t.num_rows
        acc["id"].append(np.array(t["id"].to_pylist(), dtype="S40"))
        srcs = t["source"].to_pylist()
        for x in set(srcs):
            if x not in code_src:
                code_src[x] = len(src_names)
                src_names.append(x)
        acc["src"].append(np.array([code_src[x] for x in srcs], dtype=np.int16))
        acc["mixed"].append(np.array([k == "mixed" for k in t["kind"].to_pylist()], dtype=bool))
        acc["qual"].append(np.nan_to_num(t["quality"].to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0))
        acc["nch"].append(t["n_chars"].to_numpy().astype(np.int64))
        sig = t["sig"].combine_chunks()
        offs = np.frombuffer(sig.buffers()[1], dtype=np.int32)[sig.offset: sig.offset + n + 1]
        data = np.frombuffer(sig.buffers()[2], dtype=np.uint8)
        assert np.all(np.diff(offs) == PERMS * 4), "unexpected signature length"
        acc["sig"].append(data[offs[0]: offs[-1]].view(np.uint32).reshape(n, PERMS).copy())
        for i, c in enumerate(t["contam"].to_pylist()):
            if c:
                contam[base + i] = c
        parts = t["part"].to_pylist()
        for x in set(parts):
            if x not in code_part:
                code_part[x] = len(part_names)
                part_names.append(x)
        acc["part"].append(np.array([code_part[x] for x in parts], dtype=np.int32))
        base += n
        os.remove(p)
    arr = {k: np.concatenate(v) for k, v in acc.items()}
    return arr, contam, src_names, part_names


def job_dedup():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    fs = hf().list_repo_files(REPO, repo_type="dataset")
    sig_from = CONFIG.get("sig_from", OV)
    sig_parts = sorted(f for f in fs if f.startswith(f"{sig_from}/sig/") and f.endswith(".parquet"))
    raw = [f for f in fs if f.startswith(f"{V}/raw/") and f.endswith(".parquet") and "/_manifests/" not in f]
    if len(sig_parts) < len(raw) and not SMOKE:
        raise RuntimeError(f"only {len(sig_parts)} of {len(raw)} signature parts exist")
    if SMOKE:
        sig_parts = sig_parts[: CONFIG.get("smoke_sig_parts", 3)]
    t0 = time.time()
    snapshot_download(REPO, repo_type="dataset", token=KEYS["HF_TOKEN"], local_dir=os.path.join(WORK, "dl"),
                      allow_patterns=sig_parts, max_workers=16)  # parallel; one by one took 43 min
    arr, contam, src_names, part_names = _load_sigs([os.path.join(WORK, "dl", p) for p in sig_parts])
    n = len(arr["id"])
    log(f"loaded {n:,} signatures from {len(sig_parts)} parts in {time.time() - t0:.0f}s; sig matrix {arr['sig'].nbytes / 1e9:.1f} GB")

    # benchmark items join the pool with the best rank, so documents that near-duplicate them are caught
    global BENCH
    BENCH, items = build_bench_index()
    nb = len(items)
    bench_code = len(src_names)
    src_names.append("benchmark")
    ids = np.concatenate([arr.pop("id"), np.array([f"bench{i:011d}".encode() for i in range(nb)], dtype="S40")])
    src = np.concatenate([arr.pop("src"), np.full(nb, bench_code, np.int16)])
    mixed = np.concatenate([arr.pop("mixed"), np.zeros(nb, bool)])
    qual = np.concatenate([arr.pop("qual"), np.zeros(nb)])
    nch = np.concatenate([arr.pop("nch"), np.array([len(t) for _, t in items], np.int64)])
    part = np.concatenate([arr.pop("part"), np.full(nb, -1, np.int32)])
    sig = np.concatenate([arr.pop("sig"), np.stack([minhash(t) for _, t in items]) if nb else np.zeros((0, PERMS), np.uint32)])
    del arr
    N = len(ids)
    src_rank = np.array([-1 if x == "benchmark" else (1 if x.startswith("synth_") else SOURCE_RANK.get(x, 5)) for x in src_names])
    rank = src_rank[src]
    order = np.lexsort((-nch, -qual, rank))  # best first
    inv = np.empty(N, np.int64)
    inv[order] = np.arange(N)  # position in priority order; lower is better

    status = np.zeros(N, np.int8)  # 0 keep, 1 exact dup, 2 near dup, 3 bench n-gram, 4 bench near-dup
    o = np.lexsort((inv, ids))
    same = np.r_[False, ids[o][1:] == ids[o][:-1]]
    status[o[same]] = 1
    for i in contam:
        if status[i] == 0:
            status[i] = 3
    best = np.full(N, np.iinfo(np.int64).max, np.int64)
    mult = np.random.default_rng(7).integers(1, 2 ** 63 - 1, ROWS, dtype=np.uint64) | np.uint64(1)
    for b in range(BANDS):
        key = (sig[:, b * ROWS:(b + 1) * ROWS].astype(np.uint64) * mult).sum(axis=1)
        o = np.lexsort((inv, key))
        k = key[o]
        start_ = np.r_[True, k[1:] != k[:-1]]
        grp_first = o[np.maximum.accumulate(np.where(start_, np.arange(N), 0))]
        cand = ~start_
        a, f = o[cand], grp_first[cand]
        hit = np.zeros(len(a), bool)
        for c0 in range(0, len(a), 2_000_000):  # bounded memory for the comparison
            sl = slice(c0, c0 + 2_000_000)
            hit[sl] = (sig[a[sl]] == sig[f[sl]]).mean(axis=1) >= 0.7
        np.minimum.at(best, a[hit], inv[f[hit]])
        log(f"band {b + 1}/{BANDS}: {int(hit.sum()):,} near-dup links")
        del key, o, k, start_, grp_first, cand, a, f, hit
    has = best < np.iinfo(np.int64).max
    leader = np.where(has, order[np.minimum(best, N - 1)], -1)
    nd = has & (status == 0)
    bench_lead = nd & (src[np.maximum(leader, 0)] == bench_code)
    status[bench_lead] = 4
    status[nd & ~bench_lead] = 2
    status[src == bench_code] = 9

    code = {x: i for i, x in enumerate(src_names)}
    m_stack = (src == code.get("stack_edu", -9)) & (status == 0)
    stack_cut = float(np.quantile(qual[m_stack], 0.75)) if m_stack.any() else 99.0
    m_gh = (src == code.get("github", -9)) & (status == 0)
    github_cut = float(np.quantile(qual[m_gh], 0.70)) if m_gh.any() else 99.0
    keep = np.where(status == 0)[0]
    bucket_names, bcode = [], {}
    kb = np.empty(len(keep), np.int16)
    for j, i in enumerate(keep):
        b = bucket_of(src_names[src[i]], "mixed" if mixed[i] else "code", qual[i], stack_cut, github_cut)
        if b not in bcode:
            bcode[b] = len(bucket_names)
            bucket_names.append(b)
        kb[j] = bcode[b]
    part_arr = np.array(part_names + [None], dtype=object)
    keep_tab = pa.table({"id": [x.decode() for x in ids[keep]], "bucket": [bucket_names[c] for c in kb],
                         "part": part_arr[part[keep]].tolist()})
    kp = os.path.join(WORK, "keep.parquet")
    pq.write_table(keep_tab, kp, compression="zstd")

    names = {0: "kept", 1: "exact_dup", 2: "near_dup", 3: "bench_ngram", 4: "bench_minhash"}
    report = {"stack_edu_tier_a_quality_cut": stack_cut, "github_tier_a_quality_cut": github_cut,
              "benchmarks": getattr(bench_items, "report", {}), "by_source": {}}
    for x, c in code.items():
        if x == "benchmark":
            continue
        m = src == c
        r = {names[k]: int(((status == k) & m).sum()) for k in names}
        r["kept_chars"] = int(nch[m & (status == 0)].sum())
        report["by_source"][x] = r
    report["by_bucket"] = {b: {"docs": int((kb == c).sum()), "chars": int(nch[keep][kb == c].sum())} for b, c in bcode.items()}
    report["bench_ngram_hits_by_benchmark"] = dict(collections.Counter(c.split("/")[0] for c in contam.values()))
    pid_of = [pid for pid, _ in items]
    report["bench_minhash_hits_by_benchmark"] = dict(collections.Counter(
        pid_of[leader[i] - (N - nb)].split("/")[0] for i in np.where(status == 4)[0]))
    rp = os.path.join(WORK, "dedup_report.json")
    json.dump(report, open(rp, "w"), indent=1)
    if not SMOKE or CONFIG.get("smoke_upload"):
        commit([(kp, f"{OV}/dedup/keep.parquet"), (rp, f"{OV}/dedup/report.json")], "dedup keep-list and report")
    json.dump(report, open(os.path.join(OUTW, "dedup_report.json"), "w"), indent=1)
    log(json.dumps(report["by_bucket"]))
    return {"kept": int(len(keep)), "kept_chars": int(nch[keep].sum()), "of": int(N - nb)}


# ---------------------------------------------------------------- job: tokenizer

SAMPLE_SHARE = {"stack_edu_a": 0.25, "stack_edu_b": 0.30, "oc_algorithmic_corpus": 0.07, "oc_synthetic_code_snippet": 0.06,
                "oc_synthetic_qa": 0.07, "cpython_lib": 0.05, "python_docs": 0.05, "fineweb_edu": 0.15}


def load_keep():
    import pyarrow.parquet as pq
    t = pq.read_table(download(f"{OV}/dedup/keep.parquet")).to_pydict()
    return dict(zip(t["id"], t["bucket"])), collections.Counter(t["part"])


def job_tokenizer():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    keep, part_counts = load_keep()
    target = CONFIG.get("sample_chars", 1.2e9)
    rng = np.random.default_rng(0)
    parts = list(part_counts)
    rng.shuffle(parts)
    got = collections.Counter()
    texts, held = [], []
    for p in parts:
        if sum(got.values()) >= 0.9 * target:
            break
        tab = pq.read_table(download(p), columns=["id", "text"]).to_pydict()
        for i, t in zip(tab["id"], tab["text"]):
            b = keep.get(i)
            if b is None:
                continue
            if int(i[:4], 16) % 200 == 0:
                held.append((b, t))
                continue
            if got[b] < SAMPLE_SHARE.get(b, 0.02) * target:
                texts.append(t)
                got[b] += len(t)
        os.remove(download(p))
    log(f"tokenizer sample: {sum(got.values()) / 1e6:.0f}M chars", dict((k, round(v / 1e6)) for k, v in got.items()))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(
        vocab_size=CONFIG.get("vocab", 32768), special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2, show_progress=False))
    assert tok.token_to_id("<|end|>") == 0
    tp = os.path.join(WORK, "tokenizer.json")
    tok.save(tp)
    del texts

    # characters per token on held-out code, against GPT-2 and Qwen2.5-Coder
    from huggingface_hub import hf_hub_download
    others = {}
    for name, repo in [("gpt2", "openai-community/gpt2"), ("qwen2.5-coder", "Qwen/Qwen2.5-Coder-0.5B")]:
        try:
            others[name] = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", local_dir=os.path.join(WORK, "tok", name)))
        except Exception as e:
            log("baseline tokenizer", name, "failed:", e)
    code = [t for b, t in held if b.startswith(("stack_edu", "cpython", "oc_"))][:3000]
    eng = [t for b, t in held if b in ("fineweb_edu", "python_docs")][:1000]
    rep = {"vocab": tok.get_vocab_size(), "held_out_docs": {"code": len(code), "text": len(eng)}, "chars_per_token": {}}
    for name, t in [("ours", tok), *others.items()]:
        rep["chars_per_token"][name] = {
            "code": round(sum(map(len, code)) / max(1, sum(len(e.ids) for e in t.encode_batch(code))), 3),
            "text": round(sum(map(len, eng)) / max(1, sum(len(e.ids) for e in t.encode_batch(eng))), 3)}
    vocab = tok.get_vocab()
    rep["space_run_tokens"] = {n: ("Ġ" * n) in vocab for n in (3, 4, 7, 8, 11, 12, 15, 16)}
    rp = os.path.join(WORK, "tokenizer_report.json")
    json.dump(rep, open(rp, "w"), indent=1)
    json.dump(rep, open(os.path.join(OUTW, "tokenizer_report.json"), "w"), indent=1)
    commit([(tp, f"{OV}/tokenizer/tokenizer.json"), (rp, f"{OV}/tokenizer/report.json")], "tokenizer")
    log(json.dumps(rep["chars_per_token"]))
    return rep["chars_per_token"]


# ---------------------------------------------------------------- job: tokenize


def encode_docs(tok, docs, eot, batch=2000):
    """docs: [(bucket, id, text)] -> {bucket: (train uint16 arrays, val uint16 arrays)}.
    Encodes in small batches and keeps only uint16 ids: a 90M-token part encoded in one call held
    full Encoding objects (strings, offsets, masks) and an int list, and OOM-killed a 31 GB session."""
    by = collections.defaultdict(lambda: ([], []))
    end = np.array([eot], dtype=np.uint16)
    for j in range(0, len(docs), batch):
        part = docs[j:j + batch]
        enc = tok.encode_batch([x for _, _, x in part], add_special_tokens=False)
        for (b, i, _), e in zip(part, enc):
            dest = by[b][1] if int(i[:4], 16) % 200 == 0 else by[b][0]
            dest.append(np.asarray(e.ids, dtype=np.uint16))
            dest.append(end)
        del enc
    return by


def job_tokenize():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(download(f"{OV}/tokenizer/tokenizer.json"))
    eot = tok.token_to_id("<|end|>")
    keep, part_counts = load_keep()
    fs = set(hf().list_repo_files(REPO, repo_type="dataset"))
    mine = my_parts(sorted(part_counts))
    pending, totals = [], collections.Counter()
    for k, p in enumerate(mine):
        tag = p.replace(f"{V}/raw/", "").replace("/", "__").replace(".parquet", "")
        marker = f"{OV}/tokens/_done/{tag}.json"
        if marker in fs:
            continue
        t = time.time()
        tab = pq.read_table(download(p), columns=["id", "text"]).to_pydict()
        docs = [(keep[i], i, x) for i, x in zip(tab["id"], tab["text"]) if i in keep]
        del tab
        n_docs = len(docs)
        by = encode_docs(tok, docs, eot)
        del docs
        files, counts = [], {}
        for b, (tr, va) in by.items():
            for split, chunks in (("train", tr), ("val", va)):
                if not chunks:
                    continue
                ids = np.concatenate(chunks)
                local = os.path.join(WORK, f"{b}-{split}-{tag}.bin")
                ids.tofile(local)
                files.append((local, f"{OV}/tokens/{b}/{split}/{tag}.bin"))
                counts[f"{b}/{split}"] = len(ids)
                totals[f"{b}/{split}"] += len(ids)
        ml = os.path.join(WORK, f"{tag}.json")
        json.dump(counts, open(ml, "w"))
        files.append((ml, marker))
        pending += files
        log(f"[{k + 1}/{len(mine)}] {p}: {n_docs} docs -> {sum(counts.values()) / 1e6:.1f}M tokens, {time.time() - t:.0f}s")
        os.remove(download(p))
        if len(pending) >= 30:
            commit(pending, f"tokens shard {SHARD}")
            for l, _ in pending:
                os.remove(l)
            pending = []
    if pending:
        commit(pending, f"tokens shard {SHARD}")
    return dict(totals)


def main():
    log(f"stage B job={JOB} shard={SHARD}/{NSHARDS} cpus={os.cpu_count()}")
    out = {"sig": job_sig, "dedup": job_dedup, "tokenizer": job_tokenizer, "tokenize": job_tokenize}[JOB]()
    out["minutes"] = round((time.time() - T0) / 60, 1)
    json.dump(out, open(os.path.join(OUTW, f"stageb_{JOB}_{SHARD}.json"), "w"), indent=1)
    log("DONE", json.dumps(out)[:2000])
    notify(f"stage B {JOB} {SHARD} done", json.dumps(out)[:1500], "white_check_mark")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        notify(f"stage B {JOB} {SHARD} FAILED", f"{type(e).__name__}: {str(e)[:500]}", "x")
        raise

"""Problem bank: public contest problems with tests and Python 3 reference solutions. Kaggle CPU, keys + internet.

Appended after data/jobs/sft_build.py (CONFIG "library": true), which supplies the benchmark index for
decontamination, non_english() and the HF helpers. This job only downloads and reshapes text; nothing
here runs the solutions. data/jobs/verify_bank.py runs them, offline and without keys.

Sources (train splits only; licences from the dataset cards, 21 Oct 2026):
  deepmind/code_contests  CC-BY-4.0    stdin/stdout; public + private + generated tests; Python 3 = language 3
  BAAI/TACO               Apache-2.0   stdin/stdout and call-based (fn_name); includes APPS and CodeContests
  codeparrot/apps         MIT          the same format as TACO

One record per problem (jsonl, /kaggle/working/bank/bank-NN.jsonl):
  pid, source, licence, url, difficulty (source label), level (1-5 or null), tags, question,
  starter_code, fn_name (null for stdin/stdout), tests [{"input", "output"}], solutions [str]
Call-based tests store the argument list and the expected value as JSON text.

Filters, each counted in stats.json: no Python 3 solution that parses, no tests, images in the
statement, file input/output, statement too short or too long, non-English, decontamination (the
10-gram rule of stageb.py against HumanEval(+), MBPP(+), APPS test, CodeContests test/valid,
LiveCodeBench, CRUXEval, BigCodeBench, DS-1000 and the concept suite), and duplicates across sources
(exact normalised statement, then MinHash LSH on word 5-grams, Jaccard >= 0.8). A duplicate group
keeps the copy with the most tests and pools the distinct solutions.

Tests: at most `max_tests` (25), public ones first; a test whose input + output exceeds `max_test_bytes`
(10 KB) is dropped. Solutions: at most `max_solutions` (5) distinct ones that parse, 2-200 lines.
"""

BANK_OUT = os.path.join(OUTW, "bank")
MAX_TESTS = CONFIG.get("max_tests", 25)
MAX_TEST_BYTES = CONFIG.get("max_test_bytes", 10_000)
MAX_SOLUTIONS = CONFIG.get("max_solutions", 5)
LICENCE = {"code_contests": "CC-BY-4.0", "taco": "Apache-2.0", "apps": "MIT"}
BANK_STATS = collections.Counter()
LEVEL_TACO = {"EASY": 1, "MEDIUM": 2, "MEDIUM_HARD": 3, "HARD": 4, "VERY_HARD": 5}
LEVEL_APPS = {"introductory": 1, "interview": 2, "competition": 4}


def parses(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def pick_solutions(cands):
    out, seen = [], set()
    for c in cands:
        if not isinstance(c, str):
            continue
        c = c.replace("\r\n", "\n").strip("\n") + "\n"
        key = re.sub(r"\s+", "", c)
        n = c.count("\n")
        if key in seen or not 2 <= n <= 200 or not parses(c):
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= MAX_SOLUTIONS:
            break
    return out


def pick_tests(pairs):
    out, seen = [], set()
    for i, o in pairs:
        if not isinstance(i, str) or not isinstance(o, str):
            continue
        if len(i) + len(o) > MAX_TEST_BYTES:
            BANK_STATS["test_too_big"] += 1
            continue
        if (i, o) in seen:
            continue
        seen.add((i, o))
        out.append({"input": i, "output": o})
        if len(out) >= MAX_TESTS:
            break
    return out


def lines_text(x):
    """TACO/APPS stdin tests are a string or a list of lines."""
    if isinstance(x, list):
        return "\n".join(str(v) for v in x) + "\n"
    return x if isinstance(x, str) else None


def io_tests(raw):
    """input_output JSON (TACO/APPS) -> (fn_name, [(input, output)])."""
    try:
        io = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None, []
    if not isinstance(io, dict):
        return None, []
    ins, outs, fn = io.get("inputs") or [], io.get("outputs") or [], io.get("fn_name")
    if fn:
        return fn, [(json.dumps(a), json.dumps(b)) for a, b in zip(ins, outs) if isinstance(a, list)]
    return None, [(lines_text(a), lines_text(b)) for a, b in zip(ins, outs)]


def cc_level(rating):
    if not rating:
        return None
    return 1 if rating < 1200 else 2 if rating < 1600 else 3 if rating < 2000 else 4 if rating < 2400 else 5


def record(source, pid, question, solutions, fn_name, tests, **meta):
    BANK_STATS[f"{source}/rows"] += 1
    if not question or len(question) < 80:
        return BANK_STATS.update([f"{source}/question_too_short"])
    if len(question) > 12_000:
        return BANK_STATS.update([f"{source}/question_too_long"])
    if "<image>" in question or re.search(r"!\[[^\]]*\]\(", question):
        return BANK_STATS.update([f"{source}/has_image"])
    if non_english(question):
        return BANK_STATS.update([f"{source}/non_english"])
    sols = pick_solutions(solutions)
    if not sols:
        return BANK_STATS.update([f"{source}/no_python3_solution"])
    tests = pick_tests(tests)
    if not tests:
        return BANK_STATS.update([f"{source}/no_tests"])
    hit, _ = contamination(question + "\n" + sols[0])
    if hit:
        BANK_STATS[f"contaminated_by/{hit.split('/')[0]}"] += 1
        return BANK_STATS.update([f"{source}/contaminated"])
    BANK_STATS[f"{source}/kept"] += 1
    return {"pid": f"{source}/{pid}", "source": source, "licence": LICENCE[source], "question": question.strip() + "\n",
            "fn_name": fn_name, "tests": tests, "solutions": sols, **meta}


def parquet_rows(repo, path, cols, batch=64):
    import pyarrow.parquet as pq
    if SMOKE:  # range reads of the first row group only
        from huggingface_hub import HfFileSystem
        pf = pq.ParquetFile(HfFileSystem(token=KEYS.get("HF_TOKEN")).open(f"datasets/{repo}/{path}"))
        names = [c for c in cols if c in pf.schema_arrow.names]
        yield from pf.read_row_group(0, columns=names).slice(0, CONFIG.get("smoke_rows", 60)).to_pylist()
        return
    local = download(path, repo=repo)
    try:
        pf = pq.ParquetFile(local)
        names = [c for c in cols if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=batch, columns=names):
            yield from b.to_pylist()
    finally:
        os.remove(local)


def code_contests():
    from huggingface_hub import list_repo_files
    files = sorted(f for f in list_repo_files("deepmind/code_contests", repo_type="dataset") if "/train-" in f)
    files = files[:1] if SMOKE else files
    cols = ["name", "description", "public_tests", "private_tests", "generated_tests", "solutions", "difficulty",
            "cf_rating", "cf_tags", "source", "input_file", "output_file", "cf_contest_id", "cf_index"]
    for f in files:
        for r in parquet_rows("deepmind/code_contests", f, cols):
            if (r.get("input_file") or "").strip() or (r.get("output_file") or "").strip():
                BANK_STATS["code_contests/file_io"] += 1
                continue
            s = r.get("solutions") or {}
            py3 = [c for lang, c in zip(s.get("language") or [], s.get("solution") or []) if lang == 3]
            tests = []
            for k in ("public_tests", "private_tests", "generated_tests"):
                t = r.get(k) or {}
                tests += list(zip(t.get("input") or [], t.get("output") or []))
            url = (f"https://codeforces.com/problemset/problem/{r['cf_contest_id']}/{r['cf_index']}"
                   if r.get("cf_contest_id") else None)
            rec = record("code_contests", r["name"], r.get("description"), py3, None, tests, url=url,
                         difficulty=str(r.get("difficulty")), level=cc_level(r.get("cf_rating")),
                         tags=list(r.get("cf_tags") or []), starter_code="")
            if rec:
                yield rec
        log(f"code_contests {f}: {BANK_STATS['code_contests/kept']} kept so far")


def taco_like(source, repo, files, reader):
    for f in files:
        for i, r in enumerate(reader(repo, f)):
            try:
                sols = json.loads(r.get("solutions") or "[]")
            except json.JSONDecodeError:
                sols = []
            fn, tests = io_tests(r.get("input_output"))
            if source == "taco" and str(r.get("picture_num") or "0") not in ("0", "", "None"):
                BANK_STATS["taco/has_image"] += 1
                continue
            diff = r.get("difficulty")
            level = LEVEL_TACO.get(diff) if source == "taco" else LEVEL_APPS.get(diff)
            try:
                tags = json.loads((r.get("tags") or "[]").replace("'", '"')) if source == "taco" else []
            except json.JSONDecodeError:
                tags = []
            pid = r.get("problem_id", r.get("id", f"{os.path.basename(f).split('.')[0]}-{i}"))
            rec = record(source, pid, r.get("question"), sols, fn, tests, url=r.get("url"), difficulty=diff,
                         level=level, tags=tags, starter_code=r.get("starter_code") or "")
            if rec:
                yield rec
        log(f"{source} {f}: {BANK_STATS[f'{source}/kept']} kept so far")


def taco():
    from huggingface_hub import list_repo_files
    files = sorted(f for f in list_repo_files("BAAI/TACO", repo_type="dataset") if f.startswith("ALL/train-"))
    files = files[:1] if SMOKE else files
    cols = ["question", "solutions", "starter_code", "input_output", "difficulty", "url", "tags", "picture_num", "source"]
    yield from taco_like("taco", "BAAI/TACO", files, lambda repo, f: parquet_rows(repo, f, cols))


def apps():
    def reader(repo, f):
        if SMOKE:  # the first few MB of the file
            from huggingface_hub import hf_hub_url
            req = urllib.request.Request(hf_hub_url(repo, f, repo_type="dataset"), headers={"Range": "bytes=0-3000000"})
            for line in urllib.request.urlopen(req, timeout=120).read().decode().split("\n")[:-1][:CONFIG.get("smoke_rows", 60)]:
                yield json.loads(line)
            return
        local = download(f, repo=repo)
        try:
            with open(local) as fh:
                for k, line in enumerate(fh):
                    if k >= CONFIG.get("smoke_rows_apps", 1 << 30):
                        break
                    yield json.loads(line)
        finally:
            os.remove(local)
    yield from taco_like("apps", "codeparrot/apps", ["train.jsonl"], reader)


# ---------------------------------------------------------------- dedup

def norm_question(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def minhash(text, k=64, n=5):
    words = norm_question(text).split()
    sh = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
    hs = np.array([int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) & 0xFFFFFFFF for s in sh], dtype=np.uint64)
    rng = np.random.default_rng(1234)
    a = rng.integers(1, (1 << 31) - 1, size=k, dtype=np.uint64)
    b = rng.integers(0, (1 << 31) - 1, size=k, dtype=np.uint64)
    return (((hs[None, :] * a[:, None] + b[:, None]) % ((1 << 31) - 1)).min(axis=1)).astype(np.uint32)


def dedup(recs, bands=16, rows=4):
    """Union-find over exact-statement and LSH-candidate pairs whose MinHash similarity is >= 0.8."""
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact = {}
    for i, r in enumerate(recs):
        key = hashlib.sha1(norm_question(r["question"]).encode()).hexdigest()
        if key in exact:
            union(exact[key], i)
        else:
            exact[key] = i
    sigs = [minhash(r["question"]) for r in recs]
    buckets = collections.defaultdict(list)
    for i, s in enumerate(sigs):
        for bnd in range(bands):
            buckets[(bnd, s[bnd * rows:(bnd + 1) * rows].tobytes())].append(i)
    for ids in buckets.values():
        if len(ids) < 2 or len(ids) > 200:
            continue
        for j in ids[1:]:
            if find(j) != find(ids[0]) and (sigs[ids[0]] == sigs[j]).mean() >= 0.8:
                union(ids[0], j)
    groups = collections.defaultdict(list)
    for i in range(len(recs)):
        groups[find(i)].append(i)
    out = []
    for ids in groups.values():
        best = max(ids, key=lambda i: (len(recs[i]["tests"]), -i))
        rec = dict(recs[best])
        # pool solutions only from copies with the same interface (stdin, or the same function name)
        same = [best] + [i for i in ids if i != best and recs[i]["fn_name"] == recs[best]["fn_name"]]
        pool = [s for i in same for s in recs[i]["solutions"]]
        rec["solutions"] = pick_solutions(pool)
        rec["also"] = [recs[i]["pid"] for i in ids if i != best]
        BANK_STATS["duplicates_merged"] += len(ids) - 1
        out.append(rec)
    return out


def bank_main():
    global BENCH
    os.makedirs(BANK_OUT, exist_ok=True)
    BENCH, _ = build_bench_index()
    recs = []
    for gen in (code_contests, taco, apps):
        recs += list(gen())
        log(f"{gen.__name__} done: {len(recs)} records in total, {dict(BANK_STATS)}")
    recs = dedup(recs)
    recs.sort(key=lambda r: r["pid"])
    per = CONFIG.get("per_shard", 2000)
    for k in range(0, len(recs), per):
        with open(os.path.join(BANK_OUT, f"bank-{k // per:02d}.jsonl"), "w") as f:
            for r in recs[k:k + per]:
                f.write(json.dumps(r) + "\n")
    by = collections.Counter(r["source"] for r in recs)
    stats = {"problems": len(recs), "by_source": dict(by),
             "call_based": sum(r["fn_name"] is not None for r in recs),
             "by_level": dict(collections.Counter(str(r.get("level")) for r in recs)),
             "tests_mean": round(float(np.mean([len(r["tests"]) for r in recs])), 1) if recs else 0,
             "solutions_mean": round(float(np.mean([len(r["solutions"]) for r in recs])), 2) if recs else 0,
             "licences": LICENCE, "counts": dict(sorted(BANK_STATS.items())), "benchmarks": getattr(bench_items, "report", {}),
             "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(stats, open(os.path.join(BANK_OUT, "stats.json"), "w"), indent=1)
    # wheels for the offline sessions that run the bank (ruff formats solutions before bug injection)
    if not SMOKE:
        subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-d", os.path.join(OUTW, "wheels"), "ruff"],
                       check=False)
    log("BANK", json.dumps({k: stats[k] for k in ("problems", "by_source", "call_based", "tests_mean", "solutions_mean")}))
    notify("problem bank built", f"{len(recs):,} problems: {dict(by)}", "white_check_mark")


bank_main()

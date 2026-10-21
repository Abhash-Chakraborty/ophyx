"""Judge bank_sample.py samples against the bank's tests. Kaggle CPU, internet off, no keys.

Appended after data/jobs/judge.py, data/jobs/verify_bank.py and data/jobs/bug_inject.py by the
launcher (CONFIG "library": true), for the sandbox, the bank prompt wording and the repair prompt
wording. Attach the verify-bank outputs (verified/bank-*.jsonl) and the bank-sample output
(samples/<name>/bank-*.jsonl). CONFIG: shard, nshards, workers, per_test_s, budget_s, name.

Each distinct sample is judged once, with every test (stop at the first failure). Output, under
/kaggle/working/judged/<name>/:
  bank-<shard>.jsonl     {pid, source, level, n, passed, greedy_pass, results: [{pass, kind}]}
  expert-<shard>.jsonl   SFT rows: problem -> a passing sample (the shortest; a second, different one
                         when fewer than half the samples pass). Expert iteration.
  pairs-<shard>.jsonl    {prompt, chosen, rejected, pid, rejected_kind}: up to 2 pass/fail pairs per problem
  selfrepair-<shard>.jsonl  SFT rows: the model's own failing sample with the failure it produced
                         (traceback, or input with expected and actual output) -> a passing sample
  stats-<shard>.json     pass@1 (sampled mean), pass@k, greedy pass@1, by source and level, and how
                         problems split by pass rate (none / some / all)
"""

J_NAME = CONFIG.get("name", "pycoder-base350m-sft")
J_OUT = os.path.join(OUTW, "judged", J_NAME)
J_PER_TEST = CONFIG.get("per_test_s", 4)


def load_samples():
    files = sorted(glob.glob(os.path.join(SRC, "**", "samples", J_NAME, "bank-*.jsonl"), recursive=True))
    return {r["pid"]: r for f in files for r in map(json.loads, open(f)) if mine(r["pid"])}, files


def judge_problem(args):
    rec, s = args
    if time.time() - T0 > BUDGET:
        return None
    tests = sorted(rec["tests"], key=lambda t: len(t["input"]) + len(t["output"]))
    cache = {}

    def run(code):
        if code not in cache:
            if not code.strip():
                cache[code] = {"status": "fail", "first_fail": {"kind": "empty"}}
            else:
                cache[code] = judge(code, tests, rec["fn_name"], per_test_s=J_PER_TEST)
        return cache[code]

    res = [run(c) for c in s["samples"]]
    g = run(s["greedy"]) if s.get("greedy") is not None else None
    return rec, s, res, g


def main_judge():
    samples, files = load_samples()
    banks = sorted(glob.glob(os.path.join(SRC, "**", "verified", "bank-*.jsonl"), recursive=True))
    recs = [r for f in banks for r in map(json.loads, open(f)) if r["pid"] in samples]
    log(f"shard {SHARD}/{NSHARDS}: {len(recs)} problems with samples ({len(files)} sample files), {WORKERS} workers")
    os.makedirs(J_OUT, exist_ok=True)
    outs = {k: open(os.path.join(J_OUT, f"{k}-{SHARD}.jsonl"), "w") for k in ("bank", "expert", "pairs", "selfrepair")}
    st = collections.Counter()
    kinds = collections.Counter()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for n_done, item in enumerate(ex.map(judge_problem, [(r, samples[r["pid"]]) for r in recs]), 1):
            if item is None:
                st["budget_skipped"] += 1
                continue
            rec, s, res, g = item
            ok = [r["status"] == "pass" for r in res]
            kinds.update((r.get("first_fail") or {}).get("kind", "fail") for r in res if r["status"] != "pass")
            n, k = len(ok), sum(ok)
            gp = None if g is None else g["status"] == "pass"
            outs["bank"].write(json.dumps({"pid": rec["pid"], "source": rec["source"], "level": rec.get("level"), "n": n,
                                           "passed": k, "greedy_pass": gp,
                                           "results": [{"pass": o, "kind": (r.get("first_fail") or {}).get("kind")}
                                                       for o, r in zip(ok, res)]}) + "\n")
            for key in ("all", f"src/{rec['source']}", f"level/{rec.get('level')}"):
                st[f"{key}/problems"] += 1
                st[f"{key}/samples"] += n
                st[f"{key}/samples_passed"] += k
                st[f"{key}/any_pass"] += k > 0
                st[f"{key}/all_pass"] += k == n
                st[f"{key}/greedy_pass"] += bool(gp)
            prompt = sft_row(rec)["instruction"]
            good = sorted({c for c, o in zip(s["samples"], ok) if o}, key=len)
            bad = [(c, r) for c, o, r in zip(s["samples"], ok, res) if not o and c.strip()]
            if good:
                picks = good[:2] if k < n / 2 else good[:1]
                for i, c in enumerate(picks):
                    outs["expert"].write(json.dumps({"instruction": prompt, "output": c, "tag": "expert",
                                                     "seq_id": f"{rec['pid']}-x{i}"}) + "\n")
                seen = set()
                for c, r in bad:
                    if c in seen or len(seen) >= 2:
                        continue
                    seen.add(c)
                    outs["pairs"].write(json.dumps({"prompt": prompt, "chosen": good[len(seen) % len(good)], "rejected": c,
                                                    "pid": rec["pid"],
                                                    "rejected_kind": (r.get("first_fail") or {}).get("kind")}) + "\n")
                rng = random.Random(rec["pid"])
                for c, r in bad:
                    fail = r.get("first_fail") or {}
                    if fail.get("kind") not in ("runtime_error", "wrong_answer", "timeout") or fail.get("i", -1) < 0:
                        continue
                    p = bank_prompt(rec, c, fail, rng)
                    if p:
                        h = hashlib.sha1(c.encode()).hexdigest()[:16]
                        outs["selfrepair"].write(json.dumps({"instruction": p, "output": good[0],
                                                             "tag": f"selfrepair_{fail['kind']}", "seq_id": h}) + "\n")
                        st["selfrepair_rows"] += 1
                        break
            if n_done % 500 == 0:
                a = st["all/problems"]
                log(f"{n_done}/{len(recs)}: pass@1 {st['all/samples_passed'] / max(1, st['all/samples']):.3f}, "
                    f"any {st['all/any_pass'] / max(1, a):.3f}, greedy {st['all/greedy_pass'] / max(1, a):.3f}")
    for f in outs.values():
        f.close()
    stats = {"shard": SHARD, "name": J_NAME, "counts": dict(sorted(st.items())), "failure_kinds": dict(kinds),
             "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(stats, open(os.path.join(J_OUT, f"stats-{SHARD}.json"), "w"), indent=1)
    log("STATS", json.dumps(stats))


if __name__ == "__main__":
    main_judge()

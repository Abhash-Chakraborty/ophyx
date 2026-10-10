"""GitHub discovery, gates 1-2 of DATA.md "Collecting from GitHub". Runs on the orchestration box:
API calls only (no cloning), well under the rate limits, results to a small JSONL.

Gate 1: search `language:Python fork:false archived:false` sliced by star ranges (each query
returns at most 1,000 results, so the ranges are narrow at the low end).
Gate 2: licence on the allowlist, size < 200 MB, pushed within ~3 years, not a template/tutorial/
course dump (name, description and topic heuristics), Python >= 60% of language bytes.

  python data/github/discover.py --out ~/.cache/pycoder-github
Resumable: finished star slices and checked repos are recorded in OUT.
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALLOW = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "psf-2.0", "unlicense", "cc0-1.0", "zlib", "0bsd", "mit-0"}
JUNK = re.compile(r"(?i)\b(tutorial|course|homework|assignment|leetcode|exercises?|bootcamp|awesome|cheat-?sheet|"
                  r"interview|100-?days|learn(ing)?-python|my-first|template|boilerplate|cookiecutter|dotfiles|"
                  r"crawler-collection|study|notes|demo|examples?)\b")
JUNK_DESC = re.compile(r"(?i)\b(tutorials?|course(work)?|homework|assignments?|leetcode|bootcamp|100 days|cheat ?sheet|"
                       r"interview (prep|questions)|my (first|personal) (project|python))\b")


def token():
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("GITHUB_TOKEN="):
            return line.strip().split("=", 1)[1].strip("'\"")
    raise SystemExit("GITHUB_TOKEN missing from .env")


TOKEN = token()


def get(url, min_gap=[0.0]):
    """GET with rate-limit handling; read-only."""
    for attempt in range(6):
        wait = min_gap[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
                                                   "User-Agent": "pycoder-discover/1", "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                remaining = int(r.headers.get("X-RateLimit-Remaining", "100"))
                if remaining < 5:
                    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    min_gap[0] = reset + 2
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = int(e.headers.get("X-RateLimit-Reset") or time.time() + 60)
                time.sleep(max(5, reset - time.time() + 2))
                continue
            if e.code == 422:
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def star_slices(lo=50, hi=400_000):
    """Narrow ranges at low star counts where repos are dense."""
    edges, s = [], lo
    while s < hi:
        step = 2 if s < 100 else 5 if s < 300 else 20 if s < 1000 else 200 if s < 5000 else 2000 if s < 20000 else hi
        edges.append((s, min(hi, s + step) - 1))
        s += step
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/pycoder-github"))
    ap.add_argument("--min-stars", type=int, default=50)
    ap.add_argument("--pushed-after", default="2023-01-01")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cand_p, done_p, keep_p = (os.path.join(a.out, f) for f in ("candidates.jsonl", "slices_done.json", "repos.jsonl"))
    done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
    seen = set()
    if os.path.exists(cand_p):
        seen = {json.loads(l)["full_name"] for l in open(cand_p)}

    # gate 1 + the metadata half of gate 2, from search results alone
    search_gap = [0.0]
    with open(cand_p, "a") as out:
        for lo, hi in star_slices(a.min_stars):
            key = f"{lo}-{hi}"
            if key in done:
                continue
            n_new = 0
            for page in range(1, 11):
                q = f"language:Python fork:false archived:false stars:{lo}..{hi} pushed:>={a.pushed_after}"
                url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                    {"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
                search_gap[0] = max(search_gap[0], time.time() + 2.1)  # search API: 30 requests/min
                j = get(url, search_gap)
                items = (j or {}).get("items", [])
                for r in items:
                    if r["full_name"] in seen:
                        continue
                    seen.add(r["full_name"])
                    lic = ((r.get("license") or {}).get("spdx_id") or "").lower()
                    name_topics = " ".join([r["name"], " ".join(r.get("topics") or [])])
                    rec = {"full_name": r["full_name"], "stars": r["stargazers_count"], "size_kb": r["size"],
                           "license": lic, "pushed_at": r["pushed_at"], "default_branch": r["default_branch"],
                           "is_template": r.get("is_template", False),
                           "junk": bool(JUNK.search(name_topics) or JUNK_DESC.search(r.get("description") or "")),
                           "description": (r.get("description") or "")[:200], "topics": r.get("topics") or []}
                    out.write(json.dumps(rec) + "\n")
                    n_new += 1
                if len(items) < 100:
                    break
            out.flush()
            done.add(key)
            json.dump(sorted(done), open(done_p, "w"))
            print(f"[{time.strftime('%H:%M:%S')}] stars {key}: +{n_new} (total {len(seen)})", flush=True)

    # rest of gate 2: licence, size, template/junk, then language share (one API call per survivor)
    checked = set()
    if os.path.exists(keep_p):
        checked = {json.loads(l)["full_name"] for l in open(keep_p)}
    cands = [json.loads(l) for l in open(cand_p)]
    survivors = [c for c in cands if c["license"] in ALLOW and c["size_kb"] < 200_000 and not c["is_template"]
                 and not c["junk"] and c["full_name"] not in checked]
    print(f"{len(cands)} candidates; {len(survivors)} pass licence/size/junk; checking language share", flush=True)
    with open(keep_p, "a") as out:
        for k, c in enumerate(survivors):
            langs = get(f"https://api.github.com/repos/{c['full_name']}/languages") or {}
            total = sum(langs.values()) or 1
            c["python_share"] = round(langs.get("Python", 0) / total, 3)
            c["keep"] = c["python_share"] >= 0.6
            out.write(json.dumps(c) + "\n")
            if k % 500 == 0:
                out.flush()
                print(f"[{time.strftime('%H:%M:%S')}] languages {k}/{len(survivors)}", flush=True)
    kept = [json.loads(l) for l in open(keep_p)]
    print(f"done: {sum(r['keep'] for r in kept)} repos kept of {len(kept)} checked", flush=True)


if __name__ == "__main__":
    main()

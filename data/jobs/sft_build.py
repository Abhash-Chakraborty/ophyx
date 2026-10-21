"""SFT data builder: OpenCoder SFT sets -> one-Python-module responses -> tokens + loss masks. Kaggle CPU.

CONFIG keys:
  stage       "stage1" (OpenCoder-LLM/opc-sft-stage1, our stage 3a) or "stage2" (opc-sft-stage2, stage 3b), or
              "synth": our verified short exercises (data/synth/tinker_short.py -> verify.py), read from
              synth_glob (default: verified batches attached from the pycoder-verify-tinker session), or
              "repair" / "bank": rows prepared by bug_inject.py / verify_bank.py, read from rows_glob
              (default /kaggle/input/**/sft_rows/<stage>-*.jsonl), parts of `chunk` rows, or
              "expert" / "selfrepair": rows from bank_judge.py (a passing sample of our own model, and the
              model's failing program plus its error -> a passing sample), handled the same way; their
              default rows_glob is /kaggle/input/**/judged/*/<stage>-*.jsonl
  shard, nshards   sharding by source file (default 1 shard: exact dedup is then global)
  version     input version for the tokenizer (default v1); out_version: where outputs go (default = version)
  smoke       true: ~smoke_rows rows from the datasets-server rows API, no uploads, outputs under WORK
  max_tokens  longest rendered example kept (default 4096)

Pipeline per example (drop reasons are counted at every step, per source, in the report):
  1. Python only. The sources have no language field, so each fenced block is classified by its tag:
     python/py -> Python; shell, console, output, text, json, yaml, ... -> "auxiliary" (kept as
     comments); any other language tag -> the example is dropped. Untagged blocks count as Python
     if they compile and look like code, as another language if they look like C/JS/HTML/SQL, and
     as auxiliary otherwise. At least one Python block is required.
  2. Response -> one module. Python blocks that are >= 80% contained in a larger block (step-by-step
     snippets followed by "the complete code") are dropped, as are earlier blocks whose top-level
     definitions are all redefined later (earlier versions); their imports are rescued. Blocks that
     do not compile are dropped if they are a small share of the code, else the example is dropped.
     Prose before the first kept block and after the last one becomes the module docstring (intro,
     then explanation); prose between kept blocks becomes `#` comments in place. Auxiliary blocks
     become indented literal blocks in the docstring, or indented comments between code. Markdown
     is cleaned (headings, bold, links, rules), chatty lead-ins ("Certainly! Here's the code:") are
     removed, prose is re-wrapped at 79 columns. `from __future__` imports are hoisted, repeated
     top-level imports removed, IPython `!`/`%` lines commented out.
     Dropped: fewer than 3 code lines, or prose far outweighing code ("mostly_prose").
     A response with no fences that compiles as a whole is taken as one Python block. Prompts or
     responses with > 5% non-Latin letters are dropped ("non_english": the tokenizer is English/Python).
  3. The module must compile (stricter than ast.parse: also catches misplaced __future__ imports,
     `return` outside a function). Secrets and personal data are redacted (as in ingest.py), then
     `ruff format`; a format that breaks compilation is reverted. Modules using a name that is never
     defined (ruff F821) are dropped: they are snippets, not complete modules.
  4. Decontamination: same 10-gram rule as stageb.py, on prompt + module (weaker for APPS and
     CodeContests test, see WEAK_BENCH).
  5. Exact dedup on sha1(prompt, module). Parts run in a fixed order and a restart reloads the ids
     of finished parts, so with one shard the result is global and deterministic. With several
     shards, each shard also sees parts other shards finished before it started; every duplicate
     group still keeps at least one copy.
  6. Template: <|user|> "\n" prompt "\n" <|python|> "\n" module <|end|>. The "\n" after <|python|>
     is part of the prompt, so inference prompts end with "<|python|>\n".
     Multi-turn: both OpenCoder sets are single-turn (instruction, output). Rows given as a message
     list are kept only when they have exactly one user and one assistant turn; longer
     conversations are dropped ("multi_turn") rather than silently losing context.
  7. Tokens: uint16 ids; mask uint8 = 1 on module tokens and the final <|end|>, 0 on the prompt.
     Examples start with <|user|> (id 1), which marks example boundaries for packing.
  8. Outputs under {out_version}/sft/{stage}/:
       text/{part}.parquet         id, source, prompt, response_module, n_tokens, split, meta
       tokens/{train,val}/{part}.bin   uint16, examples concatenated
       mask/{train,val}/{part}.bin     uint8, same length as the token file
       ids/{part}.bin              uint64 dedup hashes of kept examples (restart state)
       _done/{part}.json           per-part counts (written last; its presence marks the part done)
       _files/{file}.json          all parts of a source file are done
       report.json                 counts summed over every finished part
     1% validation split, chosen by id hash (independent of position).
"""

import ast
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.parse
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
STAGE = CONFIG.get("stage", "stage2")
SHARD, NSHARDS = CONFIG.get("shard", 0), CONFIG.get("nshards", 1)
REPO = CONFIG.get("repo", "abhash-chakraborty/pycoder-data")
V = CONFIG.get("version", "v1")
OV = CONFIG.get("out_version", V)
SMOKE = bool(CONFIG.get("smoke", False))
WORK = CONFIG.get("work", "/kaggle/tmp/sft" if os.path.isdir("/kaggle") else tempfile.mkdtemp(prefix="sft-"))
OUTW = "/kaggle/working" if os.path.isdir("/kaggle") else WORK
CHUNK = CONFIG.get("chunk", 50_000)
MAX_TOKENS = CONFIG.get("max_tokens", 4096)
PROCS = CONFIG.get("procs", os.cpu_count())
PREFIX = f"{OV}/sft/{STAGE}"
os.makedirs(WORK, exist_ok=True)
T0 = time.time()

SOURCES = {
    "stage1": ("OpenCoder-LLM/opc-sft-stage1", ["filtered_infinity_instruct", "largescale_diverse_instruct", "realuser_instruct"]),
    "stage2": ("OpenCoder-LLM/opc-sft-stage2", ["educational_instruct", "evol_instruct", "mceval_instruct", "package_instruct"]),
    "synth": (None, ["tinker_short"]),
    "repair": (None, ["repair"]),  # bug-injection repair pairs (data/jobs/bug_inject.py), rows from rows_glob
    "bank": (None, ["bank"]),  # verified problem-bank solutions (data/jobs/verify_bank.py), rows from rows_glob
    "expert": (None, ["expert"]),  # passing samples of our own model (data/jobs/bank_judge.py), rows from rows_glob
    "selfrepair": (None, ["selfrepair"]),  # the model's failing program + its error -> a passing sample (bank_judge.py)
}
ROW_STAGES = ("repair", "bank", "expert", "selfrepair")
SPECIALS = ["<|end|>", "<|user|>", "<|python|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|file_sep|>", "<|repo|>", "<|pad|>"]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_keys():
    p = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    if p:
        return json.load(open(p[0]))
    if not SMOKE:
        raise SystemExit("keys.json not found: attach the pycoder-keys dataset")
    return {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}


KEYS = load_keys()


def notify(title, msg, tags="robot"):
    topic = KEYS.get("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode()[:3000],
                                                          headers={"Title": title, "Tags": tags}, method="POST"), timeout=15)
        except Exception as e:  # notifications must never break the job
            log("ntfy failed:", e)


def hf():
    from huggingface_hub import HfApi
    return HfApi(token=KEYS["HF_TOKEN"])


def download(path_in_repo, repo=REPO):
    """Plain files under WORK/dl (no cache symlinks), so os.remove really frees the disk."""
    d = os.path.join(WORK, "dl", repo.replace("/", "__"))
    local = os.path.join(d, path_in_repo)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    for attempt in range(6):
        try:
            return hf_hub_download(repo, path_in_repo, repo_type="dataset", token=KEYS["HF_TOKEN"], local_dir=d)
        except Exception as e:
            if attempt == 5:
                raise
            log(f"download {path_in_repo} failed ({type(e).__name__}), retrying")
            time.sleep(20 * (attempt + 1))


class Uploader:
    """Buffers finished parts and commits them in batches; a part's files never straddle two commits
    except when one part alone exceeds a commit, and its _done marker always goes last."""

    def __init__(self):
        self.parts, self.done = [], set()
        if not SMOKE:
            self.done = set(hf().list_repo_files(REPO, repo_type="dataset"))

    def add_part(self, files):
        self.parts.append(files)
        if sum(map(len, self.parts)) >= 36:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        if SMOKE:
            for files in self.parts:
                for local, p in files:
                    log("smoke: would upload", p, os.path.getsize(local), "bytes")
            self.parts = []
            return
        batches, cur = [], []
        for files in self.parts:
            if cur and len(cur) + len(files) > 40:
                batches.append(cur)
                cur = []
            cur += files
        batches.append(cur)
        for b in batches:
            for i in range(0, len(b), 40):
                self._commit(b[i:i + 40])
        for files in self.parts:
            for local, p in files:
                self.done.add(p)
                os.remove(local)
        self.parts = []

    def _commit(self, files):
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=l) for l, p in files]
        for attempt in range(8):
            try:
                hf().create_commit(REPO, repo_type="dataset", operations=ops,
                                   commit_message=f"sft {STAGE} shard {SHARD}: {len(ops)} files")
                return
            except Exception as e:
                wait = min(600, 30 * 2 ** attempt)
                log(f"commit failed ({type(e).__name__}: {str(e)[:200]}), retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("HF commit kept failing")


# ---------------------------------------------------------------- benchmarks (copied from stageb.py)

NGRAM = 10
WORD = re.compile(r"\w+")
BENCH = None  # {ngram: problem_id}, built once in the parent before forking


def ngrams(text):
    toks = WORD.findall(text.lower())
    return {" ".join(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1)}


def bench_items():
    """Yields (problem_id, text) for every eval problem: prompts plus reference solutions."""
    import gzip
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    def gh_jsonl(url):
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pycoder"}), timeout=120).read()
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
        ("code_contests_test", lambda: hf_records("deepmind/code_contests", lambda f: "/test-" in f or "/valid-" in f),
         ["description"], "name"),
        ("concept_suite", lambda: hf_records(REPO, lambda f: f == "eval/concept_suite/tasks.jsonl"), ["prompt", "reference"],
         "task_id"),
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


# Competitive-programming statements share boilerplate ("The first line contains an integer n"), so for
# these benchmarks three shared 10-grams is not evidence: the bank's first run flagged 1,172 of the 5,000
# APPS train problems as APPS test copies, though the splits are disjoint. A match there has to cover a
# quarter of `ref`'s 10-grams (the statement, or the whole text).
WEAK_BENCH = {"apps_test", "code_contests_test"}


def contamination(text, ref=None):
    hits = collections.Counter(BENCH[g] for g in ngrams(text) if g in BENCH)
    if not hits:
        return None, 0
    bar = None
    for pid, n in hits.most_common():
        if pid.split("/")[0] in WEAK_BENCH:
            bar = bar or max(3, 0.25 * len(ngrams(text if ref is None else ref)))
            if n >= bar:
                return pid, n
        elif n >= 3:
            return pid, n
    return None, hits.most_common(1)[0][1]


# ---------------------------------------------------------------- redaction (copied from ingest.py)

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----[\s\S]*?-----END[^-]*-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{60,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{32,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
]
ASSIGNED_SECRET_RE = re.compile(
    r"""(?i)(\b\w*(?:api_?key|secret|token|passw(?:or)?d|access_?key)\s*[:=]\s*)(['"])([A-Za-z0-9_\-+=]{16,})\2""")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
SAFE_IPS = {"127.0.0.1", "0.0.0.0", "255.255.255.255", "255.255.255.0", "8.8.8.8", "1.1.1.1"}
SAFE_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "localhost")


def redact(text):
    """Returns (text, n_redactions). Placeholders keep strings valid Python."""
    n = 0
    for pat in SECRET_PATTERNS:
        text, k = pat.subn("<REDACTED>", text)
        n += k

    def assigned(m):
        nonlocal n
        v = m.group(3)
        if not (re.search(r"\d", v) and re.search(r"[A-Za-z]", v)):  # words like "password_field_name"
            return m.group(0)
        n += 1
        return f"{m.group(1)}{m.group(2)}<REDACTED>{m.group(2)}"

    text = ASSIGNED_SECRET_RE.sub(assigned, text)

    def email(m):
        nonlocal n
        e = m.group(0).lower()
        if e.endswith(SAFE_EMAIL_DOMAINS) or e.startswith(("git@", "noreply@", "no-reply@")):
            return m.group(0)
        n += 1
        return "user@example.com"

    def ip(m):
        nonlocal n
        s = m.group(0)
        if s in SAFE_IPS or s.startswith(("10.", "192.168.", "127.", "172.16.", "0.")):
            return s
        n += 1
        return "192.0.2.1"

    text = EMAIL_RE.sub(email, text)
    text = IPV4_RE.sub(ip, text)
    return text, n


# ---------------------------------------------------------------- response -> module

PY_TAGS = {"python", "py", "python3", "py3", "python2", "ipython", "ipython3", "sage", "pyx"}
AUX_TAGS = {"text", "txt", "plaintext", "plain", "output", "out", "result", "console", "shell", "sh", "bash", "zsh",
            "terminal", "cmd", "bat", "batch", "powershell", "ps", "ps1", "shell-session", "shellsession", "none",
            "log", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "csv", "tsv", "env", "dotenv",
            "pycon", "python-repl", "pip", "requirements", "markdown", "md", "math", "latex", "tex"}
FENCE_OPEN = re.compile(r"^([ \t]*)(`{3,}|~{3,})[ \t]*\{?\.?([\w+#.-]*)")
LOOKS_PY = re.compile(r"^\s*(def |class |import |from \S+ import |print\(|for .+ in .+:|while .+:|if .+:|return\b|@\w+|"
                      r"with .+:|try:|async def |\w+(\.\w+)*\s*=\s*\S|\w+(\.\w+)*\()", re.M)
OTHER_HINT = re.compile(r"(;\s*$|^\s*[{}]\s*$|#include|\bfunction\b|\bpublic\s+(static\s+)?\w+|console\.log|</?[a-zA-Z]+[^>]*>|"
                        r"\b(var|const|let)\s+\w+\s*=|^\s*SELECT\s.+\bFROM\b|\bfn\s+\w+\(|\bfunc\s+\w+\()", re.M | re.I)
INTERJECTION = re.compile(r"^(?:sure|certainly|absolutely|of course|okay|ok|yes|great|got it|no problem|alright|"
                          r"great question|good question|indeed)\b[!.,]*\s*", re.I)
LEADIN = re.compile(r"^(?:here(?:'s|’s| is| are)|below (?:is|are)|the following (?:is|are)|this is|"
                    r"you can (?:use|do|achieve|try)[^.]*?(?:with|using|by))\s+", re.I)
FILLER = set("""a an the python code program function functions script implementation solution snippet to solve this
problem that which accomplishes implements does task described above below for it of your you requested following
example complete full final updated modified revised version is are in how can could will using with be written here
simple sample possible one way approach and as given""".split())
ABOVE = re.compile(r"\b(code|function|program|script|implementation|solution|class|snippet|example)s? above\b|"
                   r"\babove (code|function|program|script|implementation|solution|class|snippet|example)", re.I)


NON_LATIN = re.compile(r"[\u0370-\u03ff\u0400-\u052f\u0590-\u06ff\u0900-\u0dff\u0e00-\u0eff\u1100-\u11ff"
                       r"\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


def non_english(text):
    """True when more than 5% of the letters are in a non-Latin script (CJK, Cyrillic, ...)."""
    n = len(NON_LATIN.findall(text))
    return n > 20 and n > 0.05 * sum(c.isalpha() for c in text)


def compiles(src):
    try:
        compile(src, "<module>", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError, TypeError):
        return False


def normalise(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip("\n") + "\n"


def split_markdown(text):
    """-> list of ("prose", text) and ("code", tag, text, closed)."""
    out, prose, lines, i = [], [], text.split("\n"), 0
    while i < len(lines):
        m = FENCE_OPEN.match(lines[i])
        if not m:
            prose.append(lines[i])
            i += 1
            continue
        if prose:
            out.append(("prose", "\n".join(prose)))
            prose = []
        indent, fence, tag = len(m.group(1).expandtabs()), m.group(2), m.group(3).lower()
        close = re.compile(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        body, i, closed = [], i + 1, False
        while i < len(lines):
            if close.match(lines[i]):
                closed, i = True, i + 1
                break
            ln = lines[i].expandtabs(8) if indent else lines[i]
            body.append(ln[indent:] if ln[:indent].strip() == "" else ln.lstrip())
            i += 1
        out.append(("code", tag, "\n".join(body).strip("\n"), closed))
    if prose:
        out.append(("prose", "\n".join(prose)))
    return out


def fix_python(code):
    """Returns compiling source or None. Tries: as is, dedented, IPython magics commented out."""
    cands = [code, textwrap.dedent(code)]
    fut = re.compile(r"^from __future__ import [^\n]+$", re.M)
    if fut.search(code):  # misplaced __future__ imports; to_module hoists them to the top later anyway
        d = textwrap.dedent(code)
        cands.append("\n".join(fut.findall(d)) + "\n" + fut.sub("", d))
    magic = re.compile(r"^(\s*)([!%].*)$", re.M)
    if magic.search(code):
        cands.append(magic.sub(r"\1# \2", textwrap.dedent(code)))
    for c in cands:
        if compiles(c):
            return c
    return None


def classify(tag, code):
    """-> "py", "aux", "repl" or "other"."""
    if tag in PY_TAGS:
        if re.match(r"\s*>>> ", code):
            return "repl"
        return "py"
    if tag in AUX_TAGS:
        return "repl" if re.match(r"\s*>>> ", code) else "aux"
    if tag:
        return "other"
    if re.match(r"\s*>>> ", code):
        return "repl"
    if fix_python(code) is not None and LOOKS_PY.search(code):
        return "py"
    if len(OTHER_HINT.findall(code)) >= 2:
        return "other"
    return "aux"


def code_lines(src):
    return [x.strip() for x in src.split("\n") if x.strip() and not x.strip().startswith("#")]


def top_level(src):
    tree = ast.parse(src)
    defs, other, imports = set(), 0, []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(src, n))
        elif not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)):
            other += 1
    return defs, other, imports


def clean_md(text):
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)
    text = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    lines = []
    for ln in text.split("\n"):
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", ln):
            continue
        ln = re.sub(r"^\s{0,3}#{1,6}\s+", "", ln)
        ln = re.sub(r"^(\s*)[*+]\s+", r"\1- ", ln)
        lines.append(ln.rstrip())
    return "\n".join(lines).strip("\n")


REFERS = re.compile(r"(?i)\b(here|below|following|as follows|(this|the|sample|example) (code|snippet|implementation|"
                    r"program|script|solution))\b")


def rewrite_leadin(sent):
    """"Here's a Python program that X." -> "A Python program that X."; None if nothing is left."""
    core = LEADIN.sub("", sent)
    if core == sent:
        return sent
    if len([w for w in re.findall(r"[a-z]+", core.lower()) if w not in FILLER]) < 3:
        return None
    return core[0].upper() + core[1:].rstrip(" :.") + "."


def strip_leadin(text, at_start, at_end):
    """Removes "Certainly!" openers, rewrites a "Here's ..." first sentence, and drops a final sentence
    that points at a code block ("... using the following code:")."""
    text = text.strip()
    if not text:
        return text
    paras = text.split("\n\n")
    if at_start:
        p0 = INTERJECTION.sub("", paras[0], count=1).lstrip()
        if "\n" not in p0.strip():
            sents = re.split(r"(?<=[.!?:])\s+", p0.strip())
            first = rewrite_leadin(sents[0])
            sents = ([first] if first else []) + sents[1:]
            p0 = " ".join(sents)
        paras[0] = p0
    if at_end:
        last = paras[-1].strip()
        if last.endswith(":") and "\n" not in last:
            sents = re.split(r"(?<=[.!?])\s+", last)
            if REFERS.search(sents[-1]) or LEADIN.match(sents[-1]):
                sents = sents[:-1]
            else:
                sents[-1] = sents[-1][:-1] + "."
            paras[-1] = " ".join(sents)
    return "\n\n".join(p for p in paras if p.strip()).strip()


def wrap_prose(text, width):
    """Re-wraps paragraphs; keeps list items (with hanging indent), tables and indented lines."""
    out, para, item = [], [], None  # item: [indent+marker, hanging indent, words]

    def flush():
        nonlocal item
        if para:
            out.extend(textwrap.wrap(" ".join(para), width, break_long_words=False, break_on_hyphens=False))
            para.clear()
        if item:
            out.extend(textwrap.wrap(item[2], width, initial_indent=item[0], subsequent_indent=item[1],
                                     break_long_words=False, break_on_hyphens=False) or [item[0].rstrip()])
            item = None

    for ln in text.split("\n"):
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", ln)
        if not ln.strip():
            flush()
            out.append("")
        elif m:
            flush()
            ind, mark, body = m.groups()
            item = [ind + mark + " ", ind + " " * (len(mark) + 1), body.strip()]
        elif item and ln.startswith(" ") and not ln.startswith("    " + item[1]):
            item[2] += " " + ln.strip()  # continuation of a list item
        elif "|" in ln or ln.startswith("    "):
            flush()
            out.append(ln)
        else:
            if item:
                flush()
            para.append(ln.strip())
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")


def as_docstring(text):
    text = text.replace('"""', "'''").rstrip()
    if text.endswith('"'):
        text += " "
    first = text.split("\n", 1)
    body = text if len(first) == 1 and len(text) < 70 else text + "\n"
    cands = []
    if "\\" in text:
        cands.append(f'r"""{body}"""')
        cands.append('"""' + body.replace("\\", "\\\\") + '"""')
    else:
        cands.append(f'"""{body}"""')
    for c in cands:
        if compiles(c):
            return c
    return None


def as_comment(text, indent=""):
    return "\n".join((indent + "# " + ln).rstrip() if ln.strip() else indent + "#" for ln in text.split("\n"))


def literal(code):
    return "\n".join(("    " + ln).rstrip() for ln in code.split("\n"))


def to_module(response):
    """Markdown response -> (module source, reason). reason is "ok" or why it was dropped."""
    segs = split_markdown(normalise(response))
    if not any(x[0] == "code" for x in segs):
        bare = fix_python(normalise(response))  # a response that is nothing but code, without fences
        if bare is not None and len(LOOKS_PY.findall(bare)) >= 2:
            segs = [("code", "python", bare.strip("\n"), True)]
    items = []  # dicts: kind prose|py|aux; for py: src (or None if it doesn't compile)
    for s in segs:
        if s[0] == "prose":
            if s[1].strip():
                items.append({"kind": "prose", "text": s[1]})
            continue
        _, tag, code, closed = s
        if not code.strip():
            continue
        c = classify(tag, code)
        if c == "other":
            return None, "other_language"
        if c in ("aux", "repl"):
            items.append({"kind": "aux", "text": code})
        else:
            items.append({"kind": "py", "raw": code, "src": fix_python(code)})
    py = [it for it in items if it["kind"] == "py"]
    if not py:
        return None, "no_python_block"
    aux_lines = [len(it["text"].split("\n")) for it in items if it["kind"] == "aux"]
    if aux_lines and (max(aux_lines) > 40 or sum(aux_lines) > 60):
        return None, "large_non_python_block"

    # redundant blocks: >= 80% of their code lines appear in a larger kept block
    lines = [set(code_lines(it["raw"])) for it in py]
    kept_idx = []
    for i in sorted(range(len(py)), key=lambda k: (-len(lines[k]), k)):
        if lines[i] and any(len(lines[i] & lines[j]) >= 0.8 * len(lines[i]) for j in kept_idx):
            py[i]["drop"] = "redundant"
        else:
            kept_idx.append(i)
    # blocks that do not compile
    total = sum(len(code_lines(it["raw"])) for it in py if "drop" not in it) or 1
    bad = sum(len(code_lines(it["raw"])) for it in py if "drop" not in it and it["src"] is None)
    if bad > 0.25 * total:
        return None, "no_parse"
    for it in py:
        if "drop" not in it and it["src"] is None:
            it["drop"] = "no_parse"
    live = [it for it in py if "drop" not in it]
    if not live:
        return None, "no_parse"
    # earlier versions: every top-level definition is redefined by a later block
    for it in live:
        it["defs"], it["other"], it["imports"] = top_level(it["src"])
    for k, it in enumerate(live):
        later = set().union(*[x["defs"] for x in live[k + 1:]]) if k + 1 < len(live) else set()
        if it["defs"] and it["defs"] <= later and it["other"] == 0:
            it["drop"] = "superseded"
    rescued = []
    for it in py:
        if it.get("drop") in ("redundant", "superseded"):
            src = it["src"] if it["src"] is not None else fix_python(it["raw"])
            if src is not None:
                rescued += [imp for imp in top_level(src)[2] if imp]
    live = [it for it in py if "drop" not in it]

    # assemble: prose before the first / after the last kept block -> docstring, between -> comments
    first = next(k for k, it in enumerate(items) if it is live[0])
    last = max(k for k, it in enumerate(items) if it is live[-1])
    intro, outro, body = [], [], []
    pending = []  # comment lines waiting for the next kept block

    for k, it in enumerate(items):
        kind = it["kind"]
        if kind == "py" and "drop" in it:
            continue
        if k < first or k > last:
            dest = intro if k < first else outro
            if kind == "prose":
                dest.append(("prose", it["text"]))
            elif kind == "aux":
                dest.append(("aux", it["text"]))
            continue
        if kind == "py":
            body.append("\n".join(pending + [it["src"].strip("\n")]))  # comments sit right above their code
            pending = []
        elif kind == "prose":
            t = wrap_prose(clean_md(it["text"]), 77)
            if t.strip():
                pending.append(as_comment(t))
        else:
            pending.append(as_comment(literal(it["text"])))
    # (anything pending here would sit between blocks; by construction the last item is a kept block)

    def render(parts, is_intro):
        chunks = []
        for n, (kind, t) in enumerate(parts):
            if kind == "prose":
                t = clean_md(t)
                t = strip_leadin(t, at_start=is_intro and n == 0, at_end=is_intro and n == len(parts) - 1)
                t = ABOVE.sub(lambda m: m.group(0).replace("above", "below").replace("Above", "Below"), t) if not is_intro else t
                t = wrap_prose(t, 79)
                if t.strip():
                    chunks.append(t)
            else:
                chunks.append(literal(t))
        return "\n\n".join(chunks).strip("\n")

    doc_text = "\n\n".join(x for x in (render(intro, True), render(outro, False)) if x)
    doc_text = re.sub(r"^(?:explanation|code explanation|how it works)\s*:?\s*\n+", "", doc_text, flags=re.I)  # no intro
    code = "\n\n\n".join(body)
    head = []
    if code.startswith("#!"):
        shebang, code = code.split("\n", 1) if "\n" in code else (code, "")
        head.append(shebang)
    future = re.findall(r"^from __future__ import [^\n]+$", code, re.M)
    if future:
        code = re.sub(r"^from __future__ import [^\n]+\n?", "", code, flags=re.M)
    prose_chars = len(doc_text) + sum(len(x) for x in code.split("\n") if x.lstrip().startswith("#"))
    if doc_text:
        ds = as_docstring(doc_text)
        if ds is None:
            return None, "docstring_failed"
        head.append(ds)
    head += list(dict.fromkeys(future))
    have = set(re.findall(r"^(?:import|from) [^\n]+$", code, re.M))
    rescued = [r for r in dict.fromkeys(rescued) if r not in have and "\n" not in r]
    if rescued:
        head.append("\n".join(rescued))
    module = normalise("\n\n".join(head + [code]))
    module = dedup_imports(module)
    if not compiles(module):
        return None, "assembled_no_parse"

    n_code = len(code_lines(code))
    code_chars = sum(len(x) for x in code_lines(code))
    if n_code < 3:
        return None, "trivial_code"
    if (prose_chars > 3 * code_chars and n_code < 5) or prose_chars > 6 * code_chars:
        return None, "mostly_prose"
    return module, "ok"


def dedup_imports(src):
    """Drops a top-level import statement that repeats an earlier top-level one exactly."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    seen, drop = set(), set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            key = ast.dump(n)
            if key in seen:
                drop.update(range(n.lineno - 1, n.end_lineno))
            seen.add(key)
    if not drop:
        return src
    lines = src.split("\n")
    out = normalise("\n".join(l for i, l in enumerate(lines) if i not in drop))
    return out if compiles(out) else src


# ---------------------------------------------------------------- per-row work (in pool workers)


def extract(row):
    """-> (prompt, response) or a drop reason string."""
    if row.get("instruction") is not None and row.get("output") is not None:
        return row["instruction"], row["output"]
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, list):
        role = lambda m: (m.get("role") or m.get("from") or "").lower()
        text = lambda m: m.get("content") or m.get("value") or ""
        turns = [m for m in msgs if role(m) not in ("system",)]
        if len(turns) == 2 and role(turns[0]) in ("user", "human") and role(turns[1]) in ("assistant", "gpt"):
            return text(turns[0]), text(turns[1])
        return "multi_turn"
    return "bad_schema"


# Requests for attack tooling are dropped: the model will be open-sourced, and the project doesn't want it to have
# been taught to write malware, exploits or detection evasion. Defensive and educational security code is kept;
# some benign rows that merely mention these terms are lost, which is an acceptable price.
OFFENSIVE = re.compile(
    r"(?i)\b(keylogger|ransomware|reverse[- ]shell|bind[- ]shell|shellcode|botnet|ddos|denial[- ]of[- ]service attack|"
    r"credential[- ](stealer|harvest\w*)|steal(ing)? (the )?(user'?s? )?(passwords?|credentials|cookies|tokens)|"
    r"phishing (page|kit|site|email)|(bypass|evade|evading|disable) (the )?(antivirus|anti-virus|av|edr|windows defender|"
    r"detection|amsi|uac)|undetectable|fud (payload|crypter)|crypter|rootkit|backdoor|trojan|"
    r"(write|create|build|generate|develop)\w* (a |an )?(\w+ )?(malware|virus|worm|exploit|payload)|"
    r"exploit (code|payload|for (cve|the vulnerability))|zero[- ]day|cryptojack\w*|"
    r"brute[- ]?forc\w* (the )?(login|password|ssh|rdp|ftp)|(sql injection|xss) payload)\b")


def _work(row):
    """-> (reason, prompt, module, contaminating_problem)"""
    try:
        x = extract(row)
        if isinstance(x, str):
            return x, None, None, None
        prompt, resp = x
        if not prompt.strip() or not resp.strip():
            return "empty", None, None, None
        if len(resp) > 60_000 or len(prompt) > 30_000:
            return "too_long", None, None, None
        if any(s in prompt or s in resp for s in SPECIALS):
            return "contains_special_token", None, None, None
        if non_english(prompt) or non_english(resp):
            return "non_english", None, None, None
        if OFFENSIVE.search(prompt):
            return "offensive_security", None, None, None
        prompt = normalise(prompt).strip("\n")
        module, reason = to_module(resp)
        if module is None:
            return reason, None, None, None
        module, _ = redact(module)
        prompt, _ = redact(prompt)
        if not compiles(module):
            return "redact_broke_parse", None, None, None
        pid, _ = contamination(prompt + "\n" + module)
        if pid:
            return "contaminated", None, None, pid
        return "ok", prompt, module, None
    except RecursionError:
        return "recursion", None, None, None


def _compiles(t):
    return compiles(t)


def ruff_format(texts):
    """Formats a batch with one ruff call. Returns the list with failures left unchanged."""
    if not shutil.which("ruff") or not texts:
        return texts
    d = tempfile.mkdtemp(dir=WORK)
    try:
        for i, t in enumerate(texts):
            with open(os.path.join(d, f"{i}.py"), "w") as f:
                f.write(t)
        subprocess.run(["ruff", "format", "--quiet", "--no-cache", "--line-length", "88", d],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
        out = []
        for i, t in enumerate(texts):
            with open(os.path.join(d, f"{i}.py")) as f:
                out.append(f.read())
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def ruff_undefined(texts):
    """Indices of modules that use a name that is never defined (ruff F821): not complete modules."""
    if not shutil.which("ruff") or not texts:
        return set()
    d = tempfile.mkdtemp(dir=WORK)
    try:
        for i, t in enumerate(texts):
            with open(os.path.join(d, f"{i}.py"), "w") as f:
                f.write(t)
        r = subprocess.run(["ruff", "check", "--select", "F821", "--output-format", "json", "--no-cache", "--exit-zero",
                            "--quiet", d], capture_output=True, text=True, timeout=1800)
        return {int(os.path.basename(x["filename"])[:-3]) for x in json.loads(r.stdout or "[]")}
    except (subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return set()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- parts


def sha16(prompt, module):
    return hashlib.sha1((prompt + "\x00" + module).encode("utf-8", "surrogatepass")).hexdigest()[:16]


class Tok:
    def __init__(self):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(download(f"{V}/tokenizer/tokenizer.json"))
        self.end, self.user, self.py = (self.t.token_to_id(s) for s in ("<|end|>", "<|user|>", "<|python|>"))
        assert (self.end, self.user, self.py) == (0, 1, 2), "special token ids changed"
        self.nl = self.t.encode("\n").ids

    def render(self, prompts, modules):
        """-> list of (ids, mask) as numpy arrays."""
        pe = self.t.encode_batch(["\n" + p + "\n" for p in prompts])
        me = self.t.encode_batch(modules)
        out = []
        for a, b in zip(pe, me):
            head = [self.user] + a.ids + [self.py] + self.nl
            tail = b.ids + [self.end]
            out.append((np.asarray(head + tail, dtype=np.uint16),
                        np.concatenate([np.zeros(len(head), np.uint8), np.ones(len(tail), np.uint8)])))
        return out


def process_part(tag, source, rows, pool, tok, seen, uploader):
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = time.time()
    st = collections.Counter(input_rows=len(rows))
    contam = collections.Counter()
    res = pool.map(_work, rows, chunksize=32)
    keep = []
    for row, (reason, prompt, module, pid) in zip(rows, res):
        st[reason] += 1
        if pid:
            contam[pid.split("/")[0]] += 1
        if reason == "ok":
            keep.append([row, prompt, module])
    st.pop("ok", None)
    st["converted"] = len(keep)
    formatted = ruff_format([k[2] for k in keep])
    for k, f, good in zip(keep, formatted, pool.map(_compiles, formatted, chunksize=64)):
        if good:
            k[2] = normalise(f)
        else:
            st["format_reverted"] += 1
    undefined = ruff_undefined([k[2] for k in keep])
    st["undefined_name"] = len(undefined)
    keep = [k for i, k in enumerate(keep) if i not in undefined]
    uniq = []
    for row, prompt, module in keep:
        h = sha16(prompt, module)
        if int(h, 16) in seen:
            st["duplicate"] += 1
            continue
        seen.add(int(h, 16))
        uniq.append((h, row, prompt, module))
    rendered = tok.render([u[2] for u in uniq], [u[3] for u in uniq])
    recs, arrays = [], {"train": ([], []), "val": ([], [])}
    for (h, row, prompt, module), (ids, mask) in zip(uniq, rendered):
        if len(ids) > MAX_TOKENS:
            st["too_many_tokens"] += 1
            seen.discard(int(h, 16))
            continue
        split = "val" if int(h[:4], 16) % 100 == 0 else "train"
        arrays[split][0].append(ids)
        arrays[split][1].append(mask)
        meta = {k: row[k] for k in ("tag", "seq_id", "entry_point", "testcase") if row.get(k) is not None}
        recs.append({"id": h, "source": source, "prompt": prompt, "response_module": module, "n_tokens": int(len(ids)),
                     "split": split, "meta": json.dumps(meta, default=str)})
    st["kept"] = len(recs)
    files = []
    local = lambda name: os.path.join(WORK, "out", name.replace("/", "__"))
    os.makedirs(os.path.join(WORK, "out"), exist_ok=True)
    schema = pa.schema([("id", pa.string()), ("source", pa.string()), ("prompt", pa.string()), ("response_module", pa.string()),
                        ("n_tokens", pa.int32()), ("split", pa.string()), ("meta", pa.string())])
    p = local(f"text/{tag}.parquet")
    pq.write_table(pa.table({f.name: [r[f.name] for r in recs] for f in schema}, schema=schema), p, compression="zstd")
    files.append((p, f"{PREFIX}/text/{tag}.parquet"))
    for split, (ids, masks) in arrays.items():
        st[f"examples_{split}"] = len(ids)
        if not ids:
            continue
        tid, msk = np.concatenate(ids), np.concatenate(masks)
        assert len(tid) == len(msk)
        st[f"tokens_{split}"] = int(len(tid))
        st[f"loss_tokens_{split}"] = int(msk.sum())
        for kind, arr in (("tokens", tid), ("mask", msk)):
            p = local(f"{kind}/{split}/{tag}.bin")
            arr.tofile(p)
            files.append((p, f"{PREFIX}/{kind}/{split}/{tag}.bin"))
    p = local(f"ids/{tag}.bin")
    np.asarray([int(r["id"], 16) for r in recs], dtype=np.uint64).tofile(p)
    files.append((p, f"{PREFIX}/ids/{tag}.bin"))
    out = {"part": tag, "source": source, "counts": dict(st), "contaminated_by": dict(contam),
           "n_tokens_hist": np.histogram([r["n_tokens"] for r in recs], bins=[0, 256, 512, 1024, 2048, 4096, 1 << 20])[0].tolist()}
    p = local(f"_done/{tag}.json")
    json.dump(out, open(p, "w"), indent=1)
    files.append((p, f"{PREFIX}/_done/{tag}.json"))
    uploader.add_part(files)
    log(f"{tag}: {len(rows)} rows -> {st['converted']} converted -> {len(recs)} kept "
        f"({st.get('tokens_train', 0) / 1e6:.2f}M train tokens) in {time.time() - t:.0f}s")
    return out, recs


def split_task(code):
    """Module with a task docstring -> (task text, the rest of the module), or None."""
    try:
        t = ast.parse(code)
    except SyntaxError:
        return None
    doc = ast.get_docstring(t)
    if not doc or not t.body[1:]:
        return None
    rest = "\n".join(code.split("\n")[t.body[0].end_lineno:]).strip("\n") + "\n"
    return doc.strip(), rest


REPAIR_PROMPT = "This Python module fails its tests.\n\n```python\n{code}\n```\n\npytest output:\n\n```\n{error}\n```\n\nFix it."


def synth_rows():
    """Verified exercises -> rows. Each gives "task -> solution and tests"; a repaired one also gives
    "broken module + pytest output -> fixed module" (self-repair)."""
    rows, seen_ids = [], set()
    paths = sorted(p for p in glob.glob(CONFIG.get("synth_glob", "/kaggle/input/**/verified/batch-*.jsonl"), recursive=True)
                   if not p.endswith(".failed.jsonl"))
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            st = split_task(r["code"])
            if st:
                rows.append({"instruction": st[0], "output": st[1], "tag": "task", "seq_id": r["id"]})
            if r.get("repair_of"):
                err = "\n".join(r["error"].strip().split("\n")[-25:])
                rows.append({"instruction": REPAIR_PROMPT.format(code=r["broken_code"].strip("\n"), error=err),
                             "output": r["code"], "tag": "repair", "seq_id": r["id"]})
    log(f"synth: {len(seen_ids)} verified samples from {len(paths)} files -> {len(rows)} rows")
    return rows


def file_rows():
    """Rows prepared by another job ({"instruction", "output", "tag", "seq_id"} per line), from rows_glob."""
    rows, seen_ids = [], set()
    default = f"/kaggle/input/**/judged/*/{STAGE}-*.jsonl" if STAGE in ("expert", "selfrepair") else \
        f"/kaggle/input/**/sft_rows/{STAGE}-*.jsonl"
    paths = sorted(glob.glob(CONFIG.get("rows_glob", default), recursive=True))
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["seq_id"] in seen_ids:
                continue
            seen_ids.add(r["seq_id"])
            rows.append(r)
    log(f"{STAGE}: {len(rows)} rows from {len(paths)} files")
    return rows


def chunks_of(path, cols):
    """Exact CHUNK-row slices by absolute row position, independent of Parquet row groups."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = [c for c in cols if c in pf.schema_arrow.names]
    buf = []
    for b in pf.iter_batches(batch_size=10_000, columns=names):
        buf += b.to_pylist()
        while len(buf) >= CHUNK:
            yield buf[:CHUNK]
            buf = buf[CHUNK:]
    if buf:
        yield buf


def smoke_rows(src_repo, config, n, offset):
    rows = []
    while len(rows) < n:
        q = urllib.parse.urlencode({"dataset": src_repo, "config": config, "split": "train",
                                    "offset": offset + len(rows), "length": min(100, n - len(rows))})
        req = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{q}",
                                     headers={"Authorization": f"Bearer {KEYS.get('HF_TOKEN', '')}", "User-Agent": "pycoder/0.1"})
        got = json.load(urllib.request.urlopen(req, timeout=120))["rows"]
        if not got:
            break
        rows += [dict(r["row"], _truncated=bool(r.get("truncated_cells"))) for r in got]
    return rows


def write_report(uploader, this_run):
    """Sums the per-part counts of every finished part of this stage (this run's and earlier ones)."""
    parts = dict(this_run)
    if not SMOKE:
        fs = hf().list_repo_files(REPO, repo_type="dataset")
        for f in fs:
            if f.startswith(f"{PREFIX}/_done/") and f.endswith(".json"):
                tag = f.rsplit("/", 1)[1][:-5]
                if tag not in parts:
                    parts[tag] = json.load(open(download(f)))
    rep = {"stage": STAGE, "parts": len(parts), "total": collections.Counter(), "by_source": {},
           "contaminated_by": collections.Counter(), "n_tokens_hist_bins": [0, 256, 512, 1024, 2048, 4096, "inf"],
           "n_tokens_hist": [0] * 6, "benchmarks": getattr(bench_items, "report", {})}
    for p in parts.values():
        rep["total"].update(p["counts"])
        rep["by_source"].setdefault(p["source"], collections.Counter()).update(p["counts"])
        rep["contaminated_by"].update(p["contaminated_by"])
        rep["n_tokens_hist"] = [a + b for a, b in zip(rep["n_tokens_hist"], p["n_tokens_hist"])]
    order = ["input_rows", "bad_schema", "multi_turn", "empty", "too_long", "contains_special_token", "non_english", "offensive_security", "other_language",
             "no_python_block", "large_non_python_block", "no_parse", "docstring_failed", "assembled_no_parse",
             "trivial_code", "mostly_prose", "redact_broke_parse", "recursion", "contaminated", "converted",
             "format_reverted", "undefined_name", "duplicate", "too_many_tokens", "kept", "examples_train", "examples_val",
             "tokens_train", "tokens_val", "loss_tokens_train", "loss_tokens_val"]
    sort = lambda c: {k: c[k] for k in order + sorted(set(c) - set(order)) if k in c}
    rep["total"] = sort(rep["total"])
    rep["by_source"] = {s: sort(c) for s, c in sorted(rep["by_source"].items())}
    rep["contaminated_by"] = dict(rep["contaminated_by"])
    p = os.path.join(WORK, "out", "report.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rep, open(p, "w"), indent=1)
    json.dump(rep, open(os.path.join(OUTW, f"sft_{STAGE}_report.json"), "w"), indent=1)
    uploader.add_part([(p, f"{PREFIX}/report.json")])
    uploader.flush()
    return rep


def main():
    global BENCH
    if not SMOKE and not shutil.which("ruff"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ruff"], check=False)
    log(f"sft build stage={STAGE} shard={SHARD}/{NSHARDS} smoke={SMOKE} cpus={os.cpu_count()} ruff={bool(shutil.which('ruff'))}")
    src_repo, configs = SOURCES[STAGE]
    uploader = Uploader()
    tok = Tok()
    BENCH, _ = build_bench_index()

    seen = set()
    for f in sorted(uploader.done):  # restart: dedup state from every finished part
        if f.startswith(f"{PREFIX}/ids/"):
            seen.update(np.fromfile(download(f), dtype=np.uint64).tolist())
    log(f"{len(seen):,} example ids from finished parts")
    this_run, samples = {}, []
    with mp.get_context("fork").Pool(PROCS) as pool:
        if STAGE == "synth":  # small: one part, also in smoke mode (which then just skips uploads)
            rows = synth_rows()
            tag = f"tinker-short-{CONFIG.get('synth_tag', 'v1')}"
            if f"{PREFIX}/_done/{tag}.json" not in uploader.done:
                out, recs = process_part(tag, "tinker_short", rows, pool, tok, seen, uploader)
                this_run[tag] = out
                if SMOKE:
                    with open(os.path.join(WORK, "samples.jsonl"), "w") as fh:
                        fh.writelines(json.dumps(r) + "\n" for r in recs)
        elif STAGE in ROW_STAGES:
            rows = file_rows()
            if SMOKE:
                rows = rows[:CONFIG.get("smoke_rows", 200)]
            for k in range(0, len(rows), CHUNK):
                tag = f"{STAGE}-{CONFIG.get('rows_tag', 'v1')}-c{k // CHUNK:03d}"
                if f"{PREFIX}/_done/{tag}.json" in uploader.done:
                    continue
                out, recs = process_part(tag, STAGE, rows[k:k + CHUNK], pool, tok, seen, uploader)
                this_run[tag] = out
                samples += recs[:50]
            if SMOKE:
                with open(os.path.join(WORK, "samples.jsonl"), "w") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in samples)
        elif SMOKE:
            n = CONFIG.get("smoke_rows", 200)
            for c in configs:
                rows = smoke_rows(src_repo, c, -(-n // len(configs)), CONFIG.get("smoke_offset", 0))
                trunc = [r for r in rows if r.pop("_truncated")]
                rows = [r for r in rows if r not in trunc]
                out, recs = process_part(f"smoke-{c}", f"opencoder_{c}", rows, pool, tok, seen, uploader)
                out["counts"]["smoke_truncated_rows"] = len(trunc)
                this_run[out["part"]] = out
                samples += recs
            with open(os.path.join(WORK, "samples.jsonl"), "w") as fh:
                for r in samples:
                    fh.write(json.dumps(r) + "\n")
        else:
            files = sorted(f for f in hf().list_repo_files(src_repo, repo_type="dataset")
                           if f.endswith(".parquet") and any(c in f for c in configs))
            mine = [f for i, f in enumerate(files) if i % NSHARDS == SHARD]
            log(f"{len(mine)} of {len(files)} source files for shard {SHARD}")
            for f in mine:
                config = next(c for c in configs if c in f)
                ftag = f.rsplit(".", 1)[0].replace("data/", "").replace("/", "-")
                if f"{PREFIX}/_files/{ftag}.json" in uploader.done:
                    continue
                local = download(f, repo=src_repo)
                tags = []
                for k, rows in enumerate(chunks_of(local, ["instruction", "output", "tag", "seq_id", "entry_point", "testcase",
                                                           "messages", "conversations"])):
                    tag = f"{ftag}-c{k:03d}"
                    tags.append(tag)
                    if f"{PREFIX}/_done/{tag}.json" in uploader.done:
                        continue
                    out, _ = process_part(tag, f"opencoder_{config}", rows, pool, tok, seen, uploader)
                    this_run[tag] = out
                os.remove(local)
                fp = os.path.join(WORK, "out", f"_files__{ftag}.json")
                json.dump({"file": f, "parts": tags}, open(fp, "w"))
                uploader.add_part([(fp, f"{PREFIX}/_files/{ftag}.json")])
    uploader.flush()
    rep = write_report(uploader, this_run)
    out = {"stage": STAGE, "shard": SHARD, "parts_this_run": len(this_run), **{k: rep["total"].get(k, 0) for k in
           ("input_rows", "kept", "examples_val", "tokens_train", "loss_tokens_train")}, "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(out, open(os.path.join(OUTW, f"sft_{STAGE}_{SHARD}.json"), "w"), indent=1)
    log("DONE", json.dumps(out))
    notify(f"sft {STAGE} shard {SHARD} done",
           f"{out['kept']:,} of {out['input_rows']:,} examples kept, {out['tokens_train'] / 1e6:,.0f}M train tokens, "
           f"{out['minutes']:.0f} min", "white_check_mark")


if __name__ == "__main__" and not CONFIG.get("library"):
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        notify(f"sft {STAGE} shard {SHARD} FAILED", f"{type(e).__name__}: {str(e)[:500]}", "x")
        raise

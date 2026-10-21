"""Bug injection: break verified code on purpose, run its tests, keep the real failure as a repair example.

Kaggle CPU, internet off, no keys. Appended after data/jobs/judge.py by the launcher. Attach the
verify_bank outputs (verified/bank-*.jsonl, plus the ruff wheel from the problem-bank output) and the
pycoder-verify-tinker output (verified/batch-*.jsonl: exercises whose module carries its own pytest tests).

Each source program is first formatted with ruff (line length 88, as sft_build.py does), so the broken
code in the prompt and the fixed module in the answer differ only by the bug. Then one AST node is
changed by splicing new text into its source span (the rest of the file keeps its formatting):

  comparison     < <= > >= == != in / not in, is / is not swapped
  arithmetic     + - * / // % ** swapped (also in augmented assignment)
  boolean        and <-> or, a dropped `not`
  off_by_one     an integer constant +-1, or +-1 on the last argument of range()
  index          a subscript index or slice bound +-1, [-1] -> [0]
  swap_args      the first two positional arguments of a call swapped
  wrong_variable a variable read replaced by another variable of the same scope
  drop_statement an assignment, call or return statement removed
  missing_import an import removed
  init_value     an initial value changed (0 <-> 1, True <-> False, inf -> 0, "" -> " ")
  name_typo      a misspelt variable name (NameError)
  missing_arg    the last argument dropped from a call to a function the module defines (TypeError)
  missing_conversion  int(x) / float(x) / str(x) / list(x) -> x
  wrong_method   a method or builtin swapped for a plausible wrong one (append/extend, min/max, ...)

A mutant is kept only when it compiles and its tests fail. Tests run shortest first, so the reported
failure is the smallest failing input. Mutation types are balanced across the run, with at most
`per_program` examples per source program. Prompts show the broken code with either the traceback
(runtime errors) or the input, actual and expected output (wrong answers, no traceback); some include
the problem statement. Timeouts are kept rarely (`timeout_share`).

Output under /kaggle/working:
  repair/repair-<shard>.jsonl    full records (source pid, mutation type, failure kind, broken, fixed, error)
  sft_rows/repair-<shard>.jsonl  {"instruction", "output", "tag", "seq_id"} rows for sft_build.py stage repair
  repair/stats-<shard>.json
"""

import ast
import collections
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
SHARD, NSHARDS = CONFIG.get("shard", 0), CONFIG.get("nshards", 1)
SRC = CONFIG.get("src", "/kaggle/input")
OUTW = CONFIG.get("out", "/kaggle/working")
WORKERS = CONFIG.get("workers", (os.cpu_count() or 4) * 2)
PER_PROGRAM = CONFIG.get("per_program", 2)
PER_EXERCISE = CONFIG.get("per_exercise", 3)
MAX_TRIES = CONFIG.get("max_tries", 10)
TIMEOUT_SHARE = CONFIG.get("timeout_share", 0.03)
BUDGET = CONFIG.get("budget_s", 11 * 3600)
SHOW = CONFIG.get("show_chars", 800)
T0 = time.time()
TYPE_COUNTS = collections.Counter()
KIND_COUNTS = collections.Counter()
LOCK = threading.Lock()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- ruff (offline wheel)

def ensure_ruff():
    if shutil.which("ruff"):
        return shutil.which("ruff")
    wheels = glob.glob(os.path.join(SRC, "**", "wheels", "ruff-*.whl"), recursive=True)
    if wheels:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", wheels[0]], check=False)
    return shutil.which("ruff") or next(iter(glob.glob(os.path.join(os.path.dirname(sys.executable), "ruff"))), None)


def ruff_format(texts, ruff):
    if not ruff or not texts:
        return texts
    d = tempfile.mkdtemp(prefix="rf-")
    try:
        for i, t in enumerate(texts):
            with open(os.path.join(d, f"{i}.py"), "w") as f:
                f.write(t)
        subprocess.run([ruff, "format", "--quiet", "--no-cache", "--line-length", "88", d],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
        out = []
        for i, t in enumerate(texts):
            with open(os.path.join(d, f"{i}.py")) as f:
                new = f.read()
            out.append(new if compiles(new) else t)
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def compiles(src):
    try:
        compile(src, "<m>", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError, TypeError):
        return False


# ---------------------------------------------------------------- mutations

CMP_SWAP = {ast.Lt: (ast.LtE, "<", "<="), ast.LtE: (ast.Lt, "<=", "<"), ast.Gt: (ast.GtE, ">", ">="),
            ast.GtE: (ast.Gt, ">=", ">"), ast.Eq: (ast.NotEq, "==", "!="), ast.NotEq: (ast.Eq, "!=", "=="),
            ast.In: (ast.NotIn, "in", "not in"), ast.NotIn: (ast.In, "not in", "in"),
            ast.Is: (ast.IsNot, "is", "is not"), ast.IsNot: (ast.Is, "is not", "is")}
BIN_SWAP = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "+"), ast.FloorDiv: ("//", "/"),
            ast.Div: ("/", "//"), ast.Mod: ("%", "//"), ast.Pow: ("**", "*")}
METHOD_SWAP = {"append": "extend", "extend": "append", "upper": "lower", "lower": "upper", "lstrip": "rstrip",
               "rstrip": "lstrip", "strip": "rstrip", "startswith": "endswith", "endswith": "startswith",
               "index": "find", "find": "index", "keys": "values", "values": "keys", "items": "keys",
               "add": "update", "insert": "append", "popleft": "pop", "appendleft": "append", "rfind": "find",
               "isdigit": "isalpha", "isalpha": "isdigit", "most_common": "items", "intersection": "union",
               "union": "intersection", "issubset": "issuperset", "get": "pop"}
BUILTIN_SWAP = {"max": "min", "min": "max", "any": "all", "all": "any", "sorted": "reversed", "sum": "len", "round": "int"}
CONVERSIONS = {"int", "float", "str", "list", "tuple", "set", "sorted"}
BUILTINS = set(dir(__builtins__)) if not isinstance(__builtins__, dict) else set(__builtins__)


class Src:
    def __init__(self, text):
        self.text = text
        self.b = text.encode()
        self.starts = [0]
        for line in self.b.split(b"\n")[:-1]:
            self.starts.append(self.starts[-1] + len(line) + 1)

    def off(self, line, col):
        return self.starts[line - 1] + col

    def span(self, node):
        return self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset)

    def seg(self, a, b):
        return self.b[a:b].decode()

    def node_text(self, node):
        return self.seg(*self.span(node))


def gap_replace(src, left, right, old, new):
    """Replace operator `old` with `new` in the text between two nodes (no comments allowed in between)."""
    a, b = src.span(left)[1], src.span(right)[0]
    if b <= a:
        return None
    gap = src.seg(a, b)
    if "#" in gap or "\n" in gap and "\\" in gap:
        return None
    pat = r"(?<![<>=!*/])" + re.escape(old) + r"(?![=*/])" if not old[0].isalpha() else r"\b" + re.escape(old) + r"\b"
    m = re.search(pat, gap)
    if not m:
        return None
    return a + len(gap[:m.start()].encode()), a + len(gap[:m.end()].encode()), new


def whole_lines(src, node):
    """Byte span of the full lines a statement occupies, or None if it shares a line with other code."""
    a, b = src.span(node)
    ls, le = src.starts[node.lineno - 1], (src.starts[node.end_lineno] if node.end_lineno < len(src.starts) else len(src.b))
    before = src.b[ls:a]
    after = src.b[b:le].split(b"#")[0]
    if before.strip() or after.strip() or b";" in src.b[ls:le]:
        return None
    return ls, le


def walk_scope(node):
    """Nodes of one scope: like ast.walk, but not into nested function, lambda or class bodies."""
    todo = list(ast.iter_child_nodes(node))
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(n))


def scopes(tree):
    """The module and every function, each with its own nodes and the names stored in it."""
    out = []
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        own = list(walk_scope(node))
        stored = {n.id for n in own if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if not isinstance(node, ast.Module):
            stored |= {a.arg for a in node.args.args + node.args.kwonlyargs}
        out.append((own, stored))
    return out


def typo(name, rng):
    if len(name) >= 4 and rng.random() < 0.5:
        i = rng.randrange(1, len(name))
        return name[:i] + name[i + 1:]
    if len(name) >= 3:
        i = rng.randrange(0, len(name) - 1)
        if name[i] != name[i + 1]:
            return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return name + "s"


def candidates(text, allowed, rng):
    """-> {type: [(start, end, replacement)]} for nodes inside the allowed top-level statements."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {}
    src = Src(text)
    out = collections.defaultdict(list)
    defined_funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(defined_funcs)
    roots = [s for s in tree.body if allowed(s)]
    nodes = [n for r in roots for n in ast.walk(r)]

    def add(kind, item):
        if item:
            out[kind].append(item)

    for n in nodes:
        if not hasattr(n, "lineno"):
            continue
        try:
            if isinstance(n, ast.Compare) and type(n.ops[0]) in CMP_SWAP:
                _, old, new = CMP_SWAP[type(n.ops[0])]
                add("comparison", gap_replace(src, n.left, n.comparators[0], old, new))
                for c in n.comparators:
                    if isinstance(c, ast.Constant) and type(c.value) is int:
                        a, b = src.span(c)
                        add("off_by_one", (a, b, str(c.value + rng.choice([-1, 1]) if c.value > 0 else c.value + 1)))
            elif isinstance(n, ast.BinOp) and type(n.op) in BIN_SWAP:
                old, new = BIN_SWAP[type(n.op)]
                add("arithmetic", gap_replace(src, n.left, n.right, old, new))
                for c in (n.left, n.right):
                    if isinstance(c, ast.Constant) and type(c.value) is int and abs(c.value) < 10 ** 6:
                        a, b = src.span(c)
                        add("off_by_one", (a, b, str(c.value + 1 if c.value <= 0 else c.value + rng.choice([-1, 1]))))
            elif isinstance(n, ast.AugAssign) and type(n.op) in BIN_SWAP:
                old, new = BIN_SWAP[type(n.op)]
                add("arithmetic", gap_replace(src, n.target, n.value, old + "=", new + "="))
            elif isinstance(n, ast.BoolOp) and len(n.values) >= 2:
                old, new = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
                add("boolean", gap_replace(src, n.values[0], n.values[1], old, new))
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
                a, b = src.span(n)
                add("boolean", (a, b, src.node_text(n.operand) if not isinstance(n.operand, (ast.BoolOp, ast.Compare))
                                else "(" + src.node_text(n.operand) + ")"))
            elif isinstance(n, ast.Subscript):
                s = n.slice
                if isinstance(s, ast.Constant) and type(s.value) is int:
                    a, b = src.span(s)
                    add("index", (a, b, "0" if s.value == -1 else str(s.value + rng.choice([-1, 1]) if s.value > 0 else s.value + 1)))
                elif isinstance(s, ast.UnaryOp) and isinstance(s.op, ast.USub) and isinstance(s.operand, ast.Constant) and s.operand.value == 1:
                    a, b = src.span(s)
                    add("index", (a, b, "0"))
                elif isinstance(s, ast.Slice):
                    for bound in (s.lower, s.upper):
                        if isinstance(bound, ast.Constant) and type(bound.value) is int:
                            a, b = src.span(bound)
                            add("index", (a, b, str(bound.value + 1)))
                        elif isinstance(bound, ast.Name):
                            a, b = src.span(bound)
                            add("index", (a, b, f"{bound.id} {rng.choice(['+', '-'])} 1"))
                elif isinstance(s, ast.Name) and rng.random() < 0.5:
                    a, b = src.span(s)
                    add("index", (a, b, f"{s.id} {rng.choice(['+', '-'])} 1"))
            elif isinstance(n, ast.Call):
                f = n.func
                pos = [x for x in n.args if not isinstance(x, ast.Starred)]
                if isinstance(f, ast.Name) and f.id == "range" and n.args and not n.keywords and len(pos) == len(n.args):
                    last = n.args[min(len(n.args), 2) - 1]
                    a, b = src.span(last)
                    t = src.node_text(last)
                    simple = isinstance(last, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.Constant))
                    add("off_by_one", (a, b, f"{t if simple else '(' + t + ')'} {rng.choice(['+', '-'])} 1"))
                if len(pos) >= 2 and len(pos) == len(n.args):
                    a0, b0 = src.span(n.args[0])
                    a1, b1 = src.span(n.args[1])
                    t0, t1 = src.seg(a0, b0), src.seg(a1, b1)
                    if t0 != t1 and "#" not in src.seg(b0, a1):
                        add("swap_args", (a0, b1, t1 + src.seg(b0, a1) + t0))
                if isinstance(f, ast.Name) and f.id in defined_funcs and pos and len(pos) == len(n.args) and not n.keywords:
                    d = defined_funcs[f.id]
                    if len(d.args.args) - len(d.args.defaults) - (1 if d.args.args and d.args.args[0].arg == "self" else 0) >= len(pos):
                        a, _ = src.span(n.args[-1])
                        cut = src.span(n.args[-2])[1] if len(pos) >= 2 else a
                        add("missing_arg", (cut, src.span(n.args[-1])[1], ""))
                if isinstance(f, ast.Name) and f.id in CONVERSIONS and len(n.args) == 1 and not n.keywords and \
                        not isinstance(n.args[0], (ast.Starred, ast.Constant)) and f.id not in defined_funcs:
                    a, b = src.span(n)
                    add("missing_conversion", (a, b, src.node_text(n.args[0])))
                if isinstance(f, ast.Name) and f.id in BUILTIN_SWAP and f.id not in defined_funcs:
                    a, b = src.span(f)
                    add("wrong_method", (a, b, BUILTIN_SWAP[f.id]))
                if isinstance(f, ast.Attribute) and f.attr in METHOD_SWAP:
                    a, b = src.span(f)
                    add("wrong_method", (b - len(f.attr.encode()), b, METHOD_SWAP[f.attr]))
            elif isinstance(n, ast.Assign) and len(n.targets) == 1:
                v = n.value
                rep = None
                if isinstance(v, ast.Constant):
                    if v.value is True or v.value is False:
                        rep = str(not v.value)
                    elif type(v.value) is int:
                        rep = "1" if v.value == 0 else "0" if v.value in (1, -1) else str(v.value + 1)
                    elif isinstance(v.value, str) and v.value == "":
                        rep = '" "'
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "float" and v.args and \
                        isinstance(v.args[0], ast.Constant) and str(v.args[0].value).lstrip("+-") == "inf":
                    rep = "0"
                elif isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
                    rep = "0"
                if rep:
                    a, b = src.span(v)
                    add("init_value", (a, b, rep))
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return)) and not (
                    isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)):
                add("drop_statement", ("STMT", n))
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                add("missing_import", ("STMT", n))
        except (AttributeError, IndexError, TypeError, ValueError):
            continue

    # statements: only when the parent body keeps at least one other statement
    for parent in [tree] + nodes:
        for field in ("body", "orelse", "finalbody"):
            body = getattr(parent, field, None)
            if isinstance(body, list) and len(body) >= 2:
                for st in body:
                    st._can_drop = True
    for kind in ("drop_statement", "missing_import"):
        items = []
        for _, st in out.pop(kind, []):
            if not getattr(st, "_can_drop", False) and kind == "drop_statement":
                continue
            if kind == "missing_import" and isinstance(st, ast.ImportFrom) and st.module == "__future__":
                continue
            span = whole_lines(src, st)
            if span:
                items.append((span[0], span[1], ""))
        if items:
            out[kind] = items

    # variable reads: a wrong variable of the same scope, or a typo
    in_region = {id(n) for n in nodes}
    for own, stored in scopes(tree):
        region = [n for n in own if isinstance(n, ast.Name) and id(n) in in_region]
        locs = sorted(v for v in stored if v not in BUILTINS and len(v) >= 1)
        for n in region:
            if not isinstance(n.ctx, ast.Load) or n.id in BUILTINS or n.id not in stored:
                continue
            a, b = src.span(n)
            others = [v for v in locs if v != n.id]
            if others:
                out["wrong_variable"].append((a, b, rng.choice(others)))
            t = typo(n.id, rng)
            if t not in all_names and t not in BUILTINS:
                out["name_typo"].append((a, b, t))
    for k in list(out):
        uniq = {(a, b, r) for a, b, r in out[k]}
        out[k] = sorted(uniq)
    return dict(out)


def apply(text, item):
    a, b, rep = item
    bt = text.encode()
    try:
        return (bt[:a] + rep.encode() + bt[b:]).decode()
    except UnicodeDecodeError:
        return None


def mutants(text, allowed, seed, n_tries):
    """Yields (type, mutant) in balanced order: rarest type so far first, one site per type per round."""
    rng = random.Random(seed)
    cands = candidates(text, allowed, rng)
    for v in cands.values():
        rng.shuffle(v)
    seen = {text}
    tries = 0
    while tries < n_tries and any(cands.values()):
        with LOCK:
            order = sorted((k for k in cands if cands[k]), key=lambda k: (TYPE_COUNTS[k], rng.random()))
        for kind in order:
            if not cands[kind]:
                continue
            m = apply(text, cands[kind].pop())
            if m is None or m in seen or not compiles(m):
                continue
            seen.add(m)
            tries += 1
            yield kind, m
            break


# ---------------------------------------------------------------- prompts

def clean_tb(tb):
    """Drop the judge's own frames; keep the frames in solution.py and the exception line."""
    out, skip = [], False
    for line in tb.rstrip().split("\n"):
        if line.startswith('  File "'):
            skip = "solution.py" not in line
            if skip:
                continue
            line = re.sub(r'File "[^"]*solution\.py"', 'File "solution.py"', line)
        elif skip and line.startswith("    "):
            continue
        else:
            skip = False
        out.append(line)
    out = out[-30:]
    return "\n".join(out)


def clip(s, n=SHOW):
    s = s if s is not None else ""
    return s if len(s) <= n else s[:n].rstrip("\n") + "\n... (truncated)"


def show_call(fn, args_json, n=SHOW):
    try:
        args = json.loads(args_json)
        return clip(f"{fn}({', '.join(repr(a) for a in args)})", n)
    except (ValueError, TypeError):
        return clip(args_json, n)


def show_value(js):
    try:
        return clip(repr(json.loads(js)))
    except (ValueError, TypeError):
        return clip(str(js))


def bank_prompt(rec, broken, fail, rng):
    q = rec["question"].strip()
    with_q = len(q) < 2500 and rng.random() < 0.5
    head = f"{q}\n\n" if with_q else ""
    fn = rec["fn_name"]
    code = f"```python\n{broken.strip()}\n```"
    inp = show_call(fn, fail["input"]) if fn else clip(fail["input"])
    exp = show_value(fail["expected"]) if fn else clip(fail["expected"])
    what = "this call" if fn else "this input"
    if fail["kind"] == "runtime_error":
        tb = clean_tb(fail.get("stderr") or "")
        if not tb.strip():
            return None
        lead = rng.choice(["My solution crashes.", "This program raises an error.", "I get an exception when I run this."])
        return (f"{head}{lead}\n\n{code}\n\nOn {what}:\n\n```\n{inp}\n```\n\nI get:\n\n```\n{tb}\n```\n\n"
                + rng.choice(["Fix the bug.", "Fix it and give me the whole program.", "What's wrong? Fix the code."]))
    if fail["kind"] == "wrong_answer":
        got = show_value(fail["got"]) if fn else clip(fail.get("got") or "")
        lead = rng.choice(["My solution gives the wrong answer.", "This code returns a wrong result.", "Something is off in this program."])
        verb = "it returns" if fn else "it prints"
        return (f"{head}{lead}\n\n{code}\n\nFor {what}:\n\n```\n{inp}\n```\n\n{verb}:\n\n```\n{got if got.strip() else '(nothing)'}\n```\n\n"
                f"but the expected {'result' if fn else 'output'} is:\n\n```\n{exp}\n```\n\n"
                + rng.choice(["Find the bug and fix it.", "Fix the code.", "Please fix it."]))
    if fail["kind"] == "timeout":
        return (f"{head}This program never finishes (or is far too slow) on {what}:\n\n```\n{inp}\n```\n\n{code}\n\n"
                "Fix it so it gives the right answer.")
    return None


def split_tests(module):
    """Tinker exercise -> (solution part, tests part) as source text, by top-level test_ functions."""
    tree = ast.parse(module)
    lines = module.split("\n")
    sol, tests = [], []
    for i, node in enumerate(tree.body):
        start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
        end = tree.body[i + 1].lineno - 1 if i + 1 < len(tree.body) else len(lines)
        if getattr(tree.body[i + 1] if i + 1 < len(tree.body) else None, "decorator_list", None):
            end = tree.body[i + 1].decorator_list[0].lineno - 1
        chunk = "\n".join(lines[start:end]).strip("\n")
        is_test = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            node.name.startswith("test") or node.name.startswith("Test"))
        is_doc = i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        is_pytest_import = isinstance(node, (ast.Import, ast.ImportFrom)) and "pytest" in ast.unparse(node)
        if is_doc:
            continue
        (tests if is_test or is_pytest_import else sol).append(chunk)
    return "\n\n".join(sol).strip() + "\n", "\n\n".join(tests).strip() + "\n"


def clean_pytest(out):
    lines = [l for l in out.rstrip().split("\n") if not re.match(r"^(platform |rootdir:|cachedir:|plugins:|collected |=+ test session starts)", l)]
    if any("FAILURES" in l for l in lines):
        lines = lines[next(i for i, l in enumerate(lines) if "FAILURES" in l):]
    lines = [l for l in lines if not l.startswith("!!!!")]
    return "\n".join(lines[-40:])


def tinker_prompt(module, broken, out, rng):
    err = clean_pytest(out)
    if rng.random() < 0.5:
        return (f"This Python module fails its tests.\n\n```python\n{broken.strip()}\n```\n\npytest output:\n\n```\n{err}\n```\n\nFix it."), module
    sol_b, tests_b = split_tests(broken)
    sol_f, _ = split_tests(module)
    if sol_b == sol_f or not tests_b.strip():
        return None, None
    return (f"My code:\n\n```python\n{sol_b.strip()}\n```\n\nThese tests fail:\n\n```python\n{tests_b.strip()}\n```\n\n"
            f"pytest says:\n\n```\n{err}\n```\n\n" + rng.choice(["Fix my code (not the tests).", "Fix the code so the tests pass."])), sol_f


# ---------------------------------------------------------------- jobs

def mine(pid):
    return int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) % NSHARDS == SHARD


def do_bank(rec):
    if time.time() - T0 > BUDGET:
        return []
    rng = random.Random(rec["pid"])
    tests = sorted(rec["tests"], key=lambda t: len(t["input"]) + len(t["output"]))
    fixed = rec["_formatted"]
    if judge(fixed, tests, rec["fn_name"])["status"] != "pass":  # the formatted copy must still pass
        return [("format_changed_result", None)]
    out = []
    for kind, m in mutants(fixed, lambda s: True, rec["pid"], MAX_TRIES):
        r = judge(m, tests, rec["fn_name"])
        if r["status"] == "pass":
            out.append(("equivalent", kind))
            continue
        fail = r["first_fail"] or {}
        if fail.get("kind") == "timeout":
            with LOCK:
                share = KIND_COUNTS["timeout"] / max(1, sum(KIND_COUNTS.values()))
            if share > TIMEOUT_SHARE:
                out.append(("timeout_skipped", kind))
                continue
        if fail.get("kind") not in ("runtime_error", "wrong_answer", "timeout"):
            out.append((f"other_{fail.get('kind')}", kind))
            continue
        prompt = bank_prompt(rec, m, fail, rng)
        if not prompt:
            out.append(("no_prompt", kind))
            continue
        with LOCK:
            TYPE_COUNTS[kind] += 1
            KIND_COUNTS[fail["kind"]] += 1
        h = hashlib.sha1(m.encode()).hexdigest()[:16]
        out.append(("kept", {"pid": rec["pid"], "source": rec["source"], "mutation": kind, "failure": fail["kind"],
                             "broken": m, "fixed": fixed, "error": fail, "id": h,
                             "row": {"instruction": prompt, "output": fixed, "tag": f"repair_{fail['kind']}", "seq_id": h}}))
        if sum(1 for o in out if o[0] == "kept") >= PER_PROGRAM:
            break
    return out


def do_tinker(rec):
    if time.time() - T0 > BUDGET:
        return []
    rng = random.Random(rec["id"])
    module = rec["_formatted"]
    ok, _ = judge_pytest(module)
    if not ok:
        return [("format_changed_result", None)]
    allowed = lambda s: not (isinstance(s, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and s.name.lower().startswith("test")) and not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) and not (isinstance(s, (ast.Import, ast.ImportFrom)) and "pytest" in ast.unparse(s))
    out = []
    for kind, m in mutants(module, allowed, rec["id"], MAX_TRIES):
        ok, res = judge_pytest(m)
        if ok:
            out.append(("equivalent", kind))
            continue
        if res == "timeout":
            out.append(("timeout_skipped", kind))
            continue
        try:
            prompt, answer = tinker_prompt(module, m, res, rng)
        except (SyntaxError, StopIteration, ValueError):
            prompt, answer = None, None
        if not prompt:
            out.append(("no_prompt", kind))
            continue
        failure = "runtime_error" if re.search(r"^E\s+(\w+Error|\w+Exception)\b", res, re.M) and "AssertionError" not in res else "wrong_answer"
        with LOCK:
            TYPE_COUNTS[kind] += 1
            KIND_COUNTS[failure] += 1
        h = hashlib.sha1(m.encode()).hexdigest()[:16]
        out.append(("kept", {"pid": "tinker/" + rec["id"], "source": "tinker_short", "mutation": kind, "failure": failure,
                             "broken": m, "fixed": module, "error": {"pytest": clean_pytest(res)}, "id": h,
                             "row": {"instruction": prompt, "output": answer, "tag": f"repair_pytest_{failure}", "seq_id": h}}))
        if sum(1 for o in out if o[0] == "kept") >= PER_EXERCISE:
            break
    return out


def main():
    ruff = ensure_ruff()
    log(f"ruff: {ruff}")
    bank_files = sorted(glob.glob(os.path.join(SRC, "**", "verified", "bank-*.jsonl"), recursive=True))
    bank = [r for f in bank_files for r in map(json.loads, open(f)) if mine(r["pid"]) and r.get("passing") and not r.get("inconsistent")]
    tink_files = sorted(p for p in glob.glob(os.path.join(SRC, "**", "verified", "batch-*.jsonl"), recursive=True)
                        if not p.endswith(".failed.jsonl"))
    tinker, seen = [], set()
    for f in tink_files:
        for r in map(json.loads, open(f)):
            if r["id"] not in seen and r.get("kind") == "exercise" and mine("tinker/" + r["id"]):
                seen.add(r["id"])
                tinker.append(r)
    if CONFIG.get("limit"):
        bank, tinker = bank[:CONFIG["limit"]], tinker[:CONFIG["limit"]]
    for r in bank:
        r["_src"] = r["solutions"][r["passing"][0]]
    fb = ruff_format([r["_src"] for r in bank], ruff)
    ft = ruff_format([r["code"] for r in tinker], ruff)
    for r, f in zip(bank, fb):
        r["_formatted"] = f
    for r, f in zip(tinker, ft):
        r["_formatted"] = f
    log(f"shard {SHARD}/{NSHARDS}: {len(bank)} bank problems from {len(bank_files)} files, {len(tinker)} exercises, {WORKERS} workers")
    os.makedirs(os.path.join(OUTW, "repair"), exist_ok=True)
    os.makedirs(os.path.join(OUTW, "sft_rows"), exist_ok=True)
    fr = open(os.path.join(OUTW, "repair", f"repair-{SHARD}.jsonl"), "w")
    fs = open(os.path.join(OUTW, "sft_rows", f"repair-{SHARD}.jsonl"), "w")
    st = collections.Counter()
    by_type = collections.Counter()
    seen_ids = set()
    jobs = [(do_tinker, r) for r in tinker] + [(do_bank, r) for r in bank]
    random.Random(SHARD).shuffle(jobs)
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for n, res in enumerate(ex.map(lambda j: j[0](j[1]), jobs), 1):
            for what, x in res:
                if what != "kept":
                    st[what] += 1
                    if x:
                        st[f"{what}/{x}"] += 1
                    continue
                if x["id"] in seen_ids:
                    st["duplicate"] += 1
                    continue
                seen_ids.add(x["id"])
                st["kept"] += 1
                st[f"kept/{x['source']}"] += 1
                st[f"failure/{x['failure']}"] += 1
                by_type[x["mutation"]] += 1
                row = x.pop("row")
                fr.write(json.dumps(x) + "\n")
                fs.write(json.dumps(row) + "\n")
            if n % 500 == 0:
                log(f"{n}/{len(jobs)} programs: {st['kept']} kept, by type {dict(by_type)}")
    fr.close()
    fs.close()
    stats = {"shard": SHARD, "programs": len(jobs), "counts": dict(sorted(st.items())), "by_mutation": dict(by_type),
             "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(stats, open(os.path.join(OUTW, "repair", f"stats-{SHARD}.json"), "w"), indent=1)
    log("STATS", json.dumps(stats))


if __name__ == "__main__" and not CONFIG.get("library"):
    main()

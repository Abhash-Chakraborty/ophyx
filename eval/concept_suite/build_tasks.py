"""Build tasks.jsonl from the authoring files in src/<area>*.txt and lint every task.

The source format keeps prompts, references and tests readable (JSONL with escaped code is not
something to edit by hand). One file per curriculum area; each task is:

    @@@ task <slug> leaf=<index into curriculum.json> diff=<1-5> entry=<Name,other_name> [min_python=3.11]
    @@@ prompt
    plain-English request
    @@@ reference
    reference module
    @@@ tests
    pytest module doing `from solution import ...`

task_id is concept/<area>/<slug>/<k>, k counting repeats of the slug. Lint checks: the leaf
exists, the prompt names every entry point, both modules parse, the tests import from solution
and have at least 5 test functions, no time.sleep longer than 0.05 s in the tests. Prints leaf
coverage and exits non-zero on any lint error.

    python eval/concept_suite/build_tasks.py
"""

import ast
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CURRICULUM = os.path.join(ROOT, "data", "curriculum.json")
HEAD_RE = re.compile(r"^@@@ task (\S+)((?: \w+=\S+)*)\s*$")
SECTIONS = ("prompt", "reference", "tests")


def parse_file(path):
    tasks, cur, sec = [], None, None
    for n, line in enumerate(open(path, encoding="utf-8").read().split("\n"), 1):
        if line.startswith("@@@ "):
            m = HEAD_RE.match(line)
            if m:
                opts = dict(kv.split("=", 1) for kv in m.group(2).split())
                cur = {"slug": m.group(1), "opts": opts, "line": n, "file": os.path.basename(path), **{s: [] for s in SECTIONS}}
                tasks.append(cur)
                sec = None
            elif line[4:].strip() in SECTIONS and cur is not None:
                sec = line[4:].strip()
            else:
                raise SystemExit(f"{path}:{n}: bad marker {line!r}")
        elif sec:
            cur[sec].append(line)
        elif line.strip() and cur is not None:
            raise SystemExit(f"{path}:{n}: text outside a section")
    return tasks


def count_tests(tree):
    return sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_") for n in tree.body)


def long_sleeps(tree):
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            owner = getattr(f.value, "id", "") if isinstance(f, ast.Attribute) else ""
            if name == "sleep" and owner in ("time", "asyncio") and isinstance(n.args[0].value, (int, float)) \
                    and n.args[0].value > 0.05:
                bad.append(n.lineno)
    return bad


def lint(t, leaves):
    errs = []
    if not (0 <= t["leaf_index"] < len(leaves)):
        errs.append(f"leaf index {t['leaf_index']} out of range")
    if not 1 <= t["difficulty"] <= 5:
        errs.append("difficulty must be 1-5")
    for e in t["entry_points"]:
        if e not in t["prompt"]:
            errs.append(f"prompt does not name entry point {e}")
    if len(t["prompt"]) < 40:
        errs.append("prompt too short")
    for key in ("reference", "tests"):
        try:
            ast.parse(t[key])
        except SyntaxError as ex:
            errs.append(f"{key} does not parse: {ex}")
            return errs
    tree = ast.parse(t["tests"])
    if count_tests(tree) < 5:
        errs.append(f"only {count_tests(tree)} test functions")
    if "from solution import" not in t["tests"] and "import solution" not in t["tests"]:
        errs.append("tests do not import from solution")
    if long_sleeps(tree):
        errs.append(f"sleep > 0.05 s at test lines {long_sleeps(tree)}")
    ref_names = {n.name for n in ast.parse(t["reference"]).body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    ref_names |= {tg.id for n in ast.parse(t["reference"]).body if isinstance(n, (ast.Assign, ast.AnnAssign))
                  for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(tg, ast.Name)}
    ref_names |= {n.name.id for n in ast.parse(t["reference"]).body if type(n).__name__ == "TypeAlias"}
    ref_names |= {a.asname or a.name for n in ast.parse(t["reference"]).body if isinstance(n, ast.ImportFrom)
                  for a in n.names}
    for e in t["entry_points"]:
        if e not in ref_names:
            errs.append(f"reference does not define {e} at top level")
    return errs


def main():
    cur = json.load(open(CURRICULUM))["areas"]
    out, errors, seen = [], [], {}
    for area in cur:
        paths = sorted(glob.glob(os.path.join(HERE, "src", f"{area}*.txt")))
        if not paths:
            print(f"missing src/{area}*.txt")
        for raw in (r for p in paths for r in parse_file(p)):
            o = raw["opts"]
            k = seen.get((area, raw["slug"]), 0)
            seen[(area, raw["slug"])] = k + 1
            leaf_index = int(o["leaf"])
            t = {
                "task_id": f"concept/{area}/{raw['slug']}/{k}",
                "area": area,
                "leaf": cur[area][leaf_index] if 0 <= leaf_index < len(cur[area]) else "?",
                "difficulty": int(o["diff"]),
                "prompt": "\n".join(raw["prompt"]).strip(),
                "entry_points": [e for e in o["entry"].split(",") if e],
                "tests": "\n".join(raw["tests"]).strip() + "\n",
                "reference": "\n".join(raw["reference"]).strip() + "\n",
                "leaf_index": leaf_index,
            }
            if "min_python" in o:
                t["min_python"] = o["min_python"]
            for e in lint(t, cur[area]):
                errors.append(f"{t['task_id']} (src/{raw['file']}:{raw['line']}): {e}")
            out.append(t)
    with open(os.path.join(HERE, "tasks.jsonl"), "w") as f:
        for t in out:
            t.pop("leaf_index")
            f.write(json.dumps(t) + "\n")
    by_area = {}
    for t in out:
        by_area.setdefault(t["area"], []).append(t)
    print(f"{'area':22} {'tasks':>5} {'leaves':>6} {'covered':>7}")
    for area, leaves in cur.items():
        ts = by_area.get(area, [])
        covered = {t["leaf"] for t in ts}
        print(f"{area:22} {len(ts):5} {len(leaves):6} {len(covered):7}")
        for leaf in leaves:
            if leaf not in covered and ts:
                print(f"    uncovered: {leaf}")
    print(f"total {len(out)} tasks, {len({t['leaf'] for t in out})} leaves covered")
    for e in errors:
        print("LINT", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

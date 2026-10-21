"""Our own real-use training problems: everyday Python tasks with a reference solution and pytest tests.

These are training data (SFT, bug injection, pass rates for search and RL). They are not an eval:
the private eval is eval/concept_suite, and sft_build.py decontaminates against it.

Authoring format, data/own/src/*.txt, one problem per block:

    === slug-in-kebab-case [area]
    --- task
    Plain-English request, naming the function or class to write.
    --- solution
    the reference module
    --- tests
    pytest functions (no imports of the solution: the tests run in the same module)

Builds data/own/problems.jsonl in the verified-exercise format the other jobs read
({"id", "kind": "exercise", "area", "code"}, code = task docstring + solution + tests).

    python data/own/build.py            build and lint
    python data/own/build.py --check    also run every module's tests (our own code: safe to run here)
"""

import ast
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = re.compile(r"^=== ([a-z0-9-]+) \[([a-z_]+)\]\n--- task\n(.*?)\n--- solution\n(.*?)\n--- tests\n(.*?)(?=^=== |\Z)", re.S | re.M)


def build():
    recs, errors = [], []
    for path in sorted(glob.glob(os.path.join(HERE, "src", "*.txt"))):
        text = open(path).read()
        n_blocks = text.count("\n=== ") + text.startswith("=== ")
        found = BLOCK.findall(text)
        if len(found) != n_blocks:
            errors.append(f"{path}: {n_blocks} blocks but {len(found)} parsed")
        for slug, area, task, sol, tests in found:
            task = " ".join(task.split())
            code = f'"""Task: {task}"""\n\n{sol.strip()}\n\n\n{tests.strip()}\n'
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                errors.append(f"{slug}: {e}")
                continue
            n_tests = sum(isinstance(x, ast.FunctionDef) and x.name.startswith("test_") for x in tree.body)
            if n_tests < 4:
                errors.append(f"{slug}: only {n_tests} tests")
            recs.append({"id": f"own-{slug}", "kind": "exercise", "area": area, "leaf": area, "code": code})
    ids = [r["id"] for r in recs]
    errors += [f"duplicate id {i}" for i in sorted(set(ids)) if ids.count(i) > 1]
    return recs, errors


def check(recs):
    bad = []
    for r in recs:
        d = tempfile.mkdtemp(prefix="own-")
        with open(os.path.join(d, "mod.py"), "w") as f:
            f.write(r["code"])
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "mod.py"], cwd=d,
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            bad.append((r["id"], p.stdout[-1200:]))
    return bad


if __name__ == "__main__":
    recs, errors = build()
    for e in errors:
        print("ERROR", e)
    with open(os.path.join(HERE, "problems.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    areas = {}
    for r in recs:
        areas[r["area"]] = areas.get(r["area"], 0) + 1
    print(f"{len(recs)} problems: {areas}")
    if "--check" in sys.argv:
        bad = check(recs)
        for i, out in bad:
            print("FAIL", i, out)
        print(f"{len(recs) - len(bad)}/{len(recs)} pass")
        errors += bad
    sys.exit(1 if errors else 0)

"""Shape checks for the task sets the benchmarks read. The reference solutions themselves are run by
`eval/concept_suite/run.py --self-test` (see test_references_pass)."""

import json
import os
import shutil
import subprocess
import sys

import pytest

SETS = ["eval/manual/manual.jsonl", "eval/concept_suite/tasks.jsonl", "eval/realworld/realworld.jsonl"]
KEYS = {"task_id", "area", "leaf", "difficulty", "prompt", "entry_points", "tests", "reference"}


def rows(root, rel):
    with open(os.path.join(root, rel)) as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("rel", SETS)
def test_task_rows(root, rel):
    data = rows(root, rel)
    ids = [r["task_id"] for r in data]
    assert len(ids) == len(set(ids))
    for r in data:
        assert KEYS <= set(r), r["task_id"]
        assert r["prompt"].strip() and r["tests"].strip() and r["reference"].strip()
        if not r.get("raw"):  # raw tasks (SQL, shell, JavaScript, prose) aren't Python answers
            compile(r["reference"], r["task_id"], "exec")
        compile(r["tests"], r["task_id"], "exec")


def test_sizes(root):
    assert len(rows(root, "eval/manual/manual.jsonl")) == 42
    assert len(rows(root, "eval/concept_suite/tasks.jsonl")) == 298
    assert len(rows(root, "eval/realworld/realworld.jsonl")) == 33


@pytest.mark.skipif(not shutil.which("prlimit"), reason="the sandbox needs prlimit (util-linux)")
@pytest.mark.parametrize("rel", ["eval/manual/manual.jsonl", "eval/realworld/realworld.jsonl"])
def test_references_pass(root, tmp_path, rel):
    """Every hands-on and real-world reference answer passes its own tests in the scoring sandbox."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("requests")
    if "realworld" in rel and not (shutil.which("node") and shutil.which("bash")):
        pytest.skip("the real-world set's shell and JavaScript tasks need bash and node")
    r = subprocess.run([sys.executable, os.path.join(root, "eval/concept_suite/run.py"), "--self-test",
                        "--tasks", os.path.join(root, rel), "--out", str(tmp_path),
                        "--tmp", str(tmp_path / "t"), "--python", sys.executable],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

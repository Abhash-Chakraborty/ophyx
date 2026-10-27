import json

import pytest
from conftest import load


@pytest.fixture(scope="module")
def bx(tmp_path_factory):
    import os
    os.environ["OPHYX_CONFIG"] = json.dumps({"out": str(tmp_path_factory.mktemp("bx"))})
    try:
        return load("eval/bench_exec.py", "bench_exec")
    finally:
        os.environ.pop("OPHYX_CONFIG")


@pytest.fixture(scope="module")
def run():
    return load("eval/concept_suite/run.py", "concept_run")


def test_trim_special_cuts_repo_markers(bx):
    s = [{"completion": "def f():\n    return 1\n<file_sep>other.py\nx = 2"}, {"completion": "print(1)\n"}]
    out, cut = bx.trim_special(s)
    assert cut == 1
    assert out[0]["completion"] == "def f():\n    return 1\n"
    assert out[1]["completion"] == "print(1)\n"


def test_extract_code_prefers_longest_compiling_block(bx):
    text = "Here:\n```python\ndef f(:\n```\nand\n```python\ndef g():\n    return 2\n```\n"
    assert bx.extract_code(text) == "def g():\n    return 2"
    assert bx.extract_code("x = 1\n") == "x = 1\n"


def test_crux_prediction(bx):
    assert bx.crux_prediction("o", "assert f([1]) == [1, 1]\n[/ANSWER]") == "[1, 1]"
    assert bx.crux_prediction("i", "[ANSWER]assert f('ab') == 'ba'[/ANSWER]") == "f('ab')"


def test_extract_module_stops_at_end_token(run):
    text = "def f():\n    return 1\n<|end|>\nignored"
    assert run.extract_module(text) == "def f():\n    return 1\n"


def test_chat_prompt_template(run):
    assert run.chat_prompt({"prompt": "Add two numbers."}) == "<|user|>\nAdd two numbers.\n<|python|>\n"

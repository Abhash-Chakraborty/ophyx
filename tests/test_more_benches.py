"""MultiPL-E, full BigCodeBench and DS-1000: prompts, answer cutting and the smoke answer sets. Nothing here runs
benchmark code."""

import json

import pytest
from conftest import load

JS = {"task_id": "HumanEval_0_f", "language": "js", "prompt": "//Add one.\nfunction add_one(x){\n",
      "tests": "const assert = require('node:assert');\n\nfunction test() {\n  let candidate = add_one;\n"
               "  assert.deepEqual(candidate(1),2);\n}\n\ntest();", "stop_tokens": ["\nfunction ", "\n//"],
      "smoke_ref": "\n  return x + 1;\n}\n"}
CPP = {"task_id": "HumanEval_0_f", "language": "cpp",
       "prompt": "#include<assert.h>\n#include<bits/stdc++.h>\n// Add one.\nlong add_one(long x) {\n",
       "tests": "}\nint main() {\n    auto candidate = add_one;\n    assert(candidate((1)) == (2));\n}\n",
       "stop_tokens": ["\n}"], "smoke_ref": "\n    return x + 1;"}
SH = {"task_id": "HumanEval_0_f", "language": "sh", "prompt": "#!/bin/bash\n# Add one.\nadd_one() {\n",
      "tests": "}\n\ncandidate() {\n    add_one \"$@\"\n}\n\nset -e\nrun_test() {\n    [[ $(candidate \"1\") = \"2\" ]]\n}\n"
               "\nrun_test", "stop_tokens": ["\n}"], "smoke_ref": "    echo $(( $1 + 1 ))"}
DS = {"task_id": "0", "prompt": "Problem\nA:\n<code>\nimport numpy as np\na = np.ones(3)\n</code>\n"
                                 "result = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n",
      "reference_code": "result = a.sum()\n", "code_context": "def test_execution(solution: str):\n    pass\n",
      "library": "Numpy", "perturbation": "Origin"}


@pytest.fixture(scope="module")
def bx(tmp_path_factory):
    import os
    os.environ["OPHYX_CONFIG"] = json.dumps({"out": str(tmp_path_factory.mktemp("bx"))})
    try:
        return load("eval/bench_exec.py", "bench_exec")
    finally:
        os.environ.pop("OPHYX_CONFIG")


@pytest.fixture(scope="module")
def bg():
    return load("eval/bench_gen.py", "bench_gen")


@pytest.fixture(scope="module")
def smoke():
    return load("eval/bench_smoke.py", "bench_smoke")


class FakeTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"


def test_mple_entry(bx):
    assert bx.mple_entry(JS["tests"]) == "add_one"
    assert bx.mple_entry(CPP["tests"]) == "add_one"
    assert bx.mple_entry(SH["tests"]) == "add_one"


def test_mple_extract_prefers_the_named_block_in_the_language(bx):
    text = ("```python\ndef add_one(x):\n    return x + 1\n```\n```js\nconst y = 1;\n```\n"
            "```javascript\nfunction add_one(x) {\n  return x + 1;\n}\n```")
    assert bx.mple_extract("js", text, "add_one") == "function add_one(x) {\n  return x + 1;\n}"
    assert bx.mple_extract("js", "function add_one(x) { return x + 1; }", "add_one").startswith("function")
    assert bx.mple_extract("cpp", "```python\nprint(1)\n```", "add_one") == "print(1)"  # a Python answer still runs, and fails


def test_drop_main_removes_nested_braces(bx):
    code = "long f(long x) {\n    return x;\n}\nint main() {\n    if (1) { f(1); }\n    return 0;\n}\nint g() { return 1; }\n"
    assert bx.drop_main(code) == "long f(long x) {\n    return x;\n}\n\nint g() { return 1; }\n"
    assert bx.drop_main("int f() { return 1; }") == "int f() { return 1; }"


def test_mple_program_base_is_prompt_completion_tests(bx):
    s = {"prefill": CPP["prompt"], "completion": CPP["smoke_ref"]}
    assert bx.mple_program("cpp", "base", s, CPP) == CPP["prompt"] + CPP["smoke_ref"] + "\n" + CPP["tests"]


def test_mple_program_chat_cpp_and_sh(bx):
    s = {"prefill": "", "completion": "long add_one(long x) {\n    return x + 1;\n}\nint main() {\n    return 0;\n}"}
    prog = bx.mple_program("cpp", "instruct", s, CPP)
    assert prog.startswith("#include<assert.h>\n#include<bits/stdc++.h>\nlong add_one")
    assert prog.count("int main") == 1 and "auto candidate" in prog
    assert prog.count("}") == prog.count("{")
    s = {"prefill": "", "completion": "add_one() {\n    echo $(( $1 + 1 ))\n}"}
    prog = bx.mple_program("sh", "instruct", s, SH)
    assert prog.count("}") == prog.count("{") and prog.startswith("add_one() {")


def test_ds1000_code_cuts_like_the_official_postprocess(bx):
    assert bx.ds1000_code("result = a.sum()\n") == "result = a.sum()\n"
    assert bx.ds1000_code("result = a.sum()\n</code>\nEND") == "result = a.sum()\n"
    assert bx.ds1000_code("<code>\nresult = 1\nEND SOLUTION\nmore") == "\nresult = 1"
    assert bx.ds1000_code("Use sum:\n```python\nresult = a.sum()\n```\nDone.") == "result = a.sum()"


def test_pass_rates_over_runnable(bx):
    recs = [{"task_id": "a", "status": "pass"}, {"task_id": "b", "status": "pass"}, {"task_id": "c", "status": "fail"},
            {"task_id": "d", "status": "timeout"}]
    assert bx.pass_rates(recs, {"a", "c"}) == {"n": 4, "pass@1": 50.0, "pass@1_runnable": 50.0, "n_runnable": 2}
    assert bx.pass_rates(recs, None) == {"n": 4, "pass@1": 50.0}


def test_canonical_families_name_their_files(bx):
    assert {k: v[0] for k, v in bx.CANONICAL.items()} == {"bcb": "bcb_runnable.json", "bcbf": "bcbf_runnable.json",
                                                         "ds1000": "ds1000_runnable.json"}


def test_prompts(bg):
    base, ours, chat = bg.Prompter("base", None), bg.Prompter("ours", None), bg.Prompter("instruct", FakeTok())
    assert base.mple("js", JS) == (JS["prompt"], JS["stop_tokens"], JS["prompt"])
    text, stops, pre = chat.mple("cpp", CPP)
    assert "C++ function" in text and text.endswith("<assistant>```cpp\n") and stops == ["\n```"] and pre == ""
    text, stops, pre = ours.mple("sh", SH)
    assert text.startswith("<|user|>\nComplete the following Bash function") and text.endswith("<|python|>\n")
    assert base.ds1000(DS) == (DS["prompt"], bg.DS1000_STOPS, "")
    text, stops, _ = chat.ds1000(DS)
    assert text.startswith("<system>Write a short code") and stops == ["</code>"]


def test_build_and_base_exclusions(bg, tmp_path):
    rows = {"mple_js.jsonl": [JS], "ds1000.jsonl": [DS],
            "bcb_full.jsonl": [{"task_id": "BigCodeBench/0", "complete_prompt": "def task_func():\n    \"\"\"x\"\"\"\n",
                                "instruct_prompt": "Do x.", "code_prompt": "def task_func():\n"}]}
    for name, rs in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    bg.CONFIG["kit_dir"] = str(tmp_path)
    try:
        pr = bg.Prompter("base", None)
        assert [x[0] for x in bg.build("mple_js", pr)] == ["HumanEval_0_f"]
        assert bg.build("ds1000", pr)[0][1] == DS["prompt"]
        assert bg.build("bcbf_complete", pr)[0][4] == {"entry_point": "task_func"}
    finally:
        bg.CONFIG.pop("kit_dir")
    assert set(bg.EXTRA_BENCHES) <= set(bg.MAX_NEW)
    assert "bcbf_instruct" not in bg.benches_for({"kind": "base", "benches": ["bcbf_complete", "bcbf_instruct"]})


def test_smoke_sets_rebuild_the_reference_program(bx, smoke):
    for lang, task in (("js", JS), ("cpp", CPP), ("sh", SH)):
        sets = smoke.smoke_sets(f"mple_{lang}", [task, {**task, "task_id": "no_ref", "smoke_ref": None}])
        assert {k: len(v) for k, v in sets.items()} == {"smoke-ref-base": 1, "smoke-ref-chat": 1, "smoke-wrong": 1}
        base = bx.mple_program(lang, "base", sets["smoke-ref-base"][0], task)
        assert base == task["prompt"] + task["smoke_ref"] + "\n" + task["tests"]
        chat = bx.mple_program(lang, smoke.KIND["smoke-ref-chat"], sets["smoke-ref-chat"][0], task)
        assert task["smoke_ref"] in chat and chat.count("{") == chat.count("}")
        assert chat.count("int main") == (1 if lang == "cpp" else 0)
    sets = smoke.smoke_sets("ds1000", [DS])
    assert bx.ds1000_code(sets["smoke-ref-chat"][0]["completion"]) == DS["reference_code"]
    assert bx.ds1000_code(sets["smoke-ref-base"][0]["completion"]) == DS["reference_code"]

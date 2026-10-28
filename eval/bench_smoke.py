"""Check bench_exec.py's scorers on Kaggle CPU without a generation run: write answer sets whose outcome is
known, and let bench_exec score them like any model's.

  smoke-ref-base   kind base: the reference solutions as completions of the prompt (BigCodeBench, DS-1000;
                   MultiPL-E has none, so its kit's smoke_ref, a completion that passed in MultiPL-E's own run)
  smoke-ref-chat   kind instruct: the same solutions as a chat answer, a fenced block with prose around it
                   (and a main() of its own in C++), so the code extraction is checked too
  smoke-wrong      kind base: answers that run and fail (raise, throw, exit 1)

The two reference sets should score the canonical rate and the wrong set 0. It goes in front of the scorer,
in the same kernel, and runs no model:

  python kaggle/launch.py eval/bench_exec.py ophyx-bench-smoke --include eval/bench_smoke.py --no-internet \\
      --config '{"benches": ["mple_js", "ds1000"]}' --datasets ophyx-evaldata --kernel-sources ophyx-bench-kit2
"""

import glob
import json
import os

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
MPLE_FENCE = {"js": "javascript", "cpp": "cpp", "sh": "bash"}
MPLE_WRONG = {"js": "\n  throw new Error('wrong answer');\n}\n", "cpp": "\n    exit(1);", "sh": "\n    exit 1"}
RAISE = "\n    raise NotImplementedError\n"


def sample(task_id, completion, prefill=""):
    return {"task_id": task_id, "prefill": prefill, "completion": completion, "finish": "stop", "prompt_tokens": 0}


def smoke_sets(bench, tasks):
    """{model name: [samples]} for one bench, from its kit rows."""
    if bench.startswith("mple_"):
        lang = bench[5:]
        tasks = [t for t in tasks if t.get("smoke_ref")]
        whole = {t["task_id"]: t["prompt"] + t["smoke_ref"] + ("" if lang == "js" else "\n}") for t in tasks}
        extra = "\n\nint main() {\n    return 0;\n}" if lang == "cpp" else ""
        return {"smoke-ref-base": [sample(t["task_id"], t["smoke_ref"], t["prompt"]) for t in tasks],
                "smoke-ref-chat": [sample(t["task_id"], f"Here it is:\n```{MPLE_FENCE[lang]}\n{whole[t['task_id']]}{extra}\n"
                                          "```\nIt checks every case.") for t in tasks],
                "smoke-wrong": [sample(t["task_id"], MPLE_WRONG[lang], t["prompt"]) for t in tasks]}
    if bench.startswith(("bcb_", "bcbf_")):
        if bench.endswith("_complete"):
            ref = {"smoke-ref-base": [sample(t["task_id"], "\n" + t["canonical_solution"], t["complete_prompt"].rstrip())
                                      for t in tasks]}
        else:
            ref = {"smoke-ref-chat": [sample(t["task_id"], f"Sure.\n```python\n{t['code_prompt']}\n{t['canonical_solution']}"
                                                           "\n```\n") for t in tasks]}
        return {**ref, "smoke-wrong": [sample(t["task_id"], RAISE, t["code_prompt"].rstrip()) for t in tasks]}
    if bench == "ds1000":
        return {"smoke-ref-base": [sample(t["task_id"], t["reference_code"]) for t in tasks],
                "smoke-ref-chat": [sample(t["task_id"], f"Sure:\n```python\n{t['reference_code']}\n```\n") for t in tasks],
                "smoke-wrong": [sample(t["task_id"], "raise NotImplementedError\n") for t in tasks]}
    raise ValueError(f"no smoke set for {bench}")


KIND = {"smoke-ref-base": "base", "smoke-ref-chat": "instruct", "smoke-wrong": "base"}


def write_smoke(root, kit, benches):
    for bench in benches:
        kit_file = "bcb_hard.jsonl" if bench.startswith("bcb_") else "bcb_full.jsonl" if bench.startswith("bcbf_") \
            else f"{bench}.jsonl"
        tasks = [json.loads(line) for line in open(os.path.join(kit, kit_file)) if line.strip()]
        for model, samples in smoke_sets(bench, tasks).items():
            d = os.path.join(root, "samples", model)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{bench}.jsonl"), "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            json.dump({"name": model, "kind": KIND[model], "model": "smoke"}, open(os.path.join(d, "meta.json"), "w"))
            print(f"smoke: {model} {bench}: {len(samples)} answers", flush=True)


if __name__ == "__main__":
    _kit = os.path.dirname(sorted(glob.glob(os.path.join(CONFIG.get("input_root", "/kaggle/input"), "**", "manifest.json"),
                                            recursive=True))[0])
    _root = os.path.join(CONFIG.get("out", "/kaggle/working"), "smoke")
    write_smoke(_root, _kit, CONFIG["benches"])
    CONFIG.setdefault("sample_roots", []).append(_root)

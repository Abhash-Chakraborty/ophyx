"""Render benchmark prompts for API models (Kaggle Benchmarks model proxy) in the instruct format.

    python eval/frontier_prompts.py --kit <dir with the bench_kit files, manual.jsonl, tasks.jsonl> --out prompts.jsonl
           [--benches humaneval mbpp concept manual ...]

Uses bench_gen.Prompter with kind "instruct", so API models get exactly the chat messages the open
instruct models got. Two differences, both because a chat API can't continue a half-written reply:
the response prefix bench_gen prefills for EvalPlus and BigCodeBench ("Below is a Python script ...
```python") is dropped, and eval/frontier_collect.py takes the first ```python block of the reply
instead. Each line: {bench, task_id, messages, max_tokens, stops, code_block, extra}.
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = "\x00"


class MessageTok:
    """Stands in for a tokenizer: apply_chat_template hands back the messages instead of rendering them."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return SEP + json.dumps(msgs) + SEP


def load_bench_gen(kit):
    os.environ["PYCODER_CONFIG"] = json.dumps({"kit_dir": kit, "out": os.path.join(kit, "_unused")})
    spec = importlib.util.spec_from_file_location("bench_gen", os.path.join(HERE, "bench_gen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benches", nargs="*")
    a = ap.parse_args()
    bg = load_bench_gen(a.kit)
    pr = bg.Prompter("instruct", MessageTok())
    benches = a.benches or bg.ALL_BENCHES
    n = 0
    with open(a.out, "w") as f:
        for bench in benches:
            for task_id, text, stops, _pre, extra in bg.build(bench, pr):
                _, msgs, prefix = text.split(SEP)
                f.write(json.dumps({"bench": bench, "task_id": task_id, "messages": json.loads(msgs),
                                    "max_tokens": bg.MAX_NEW[bench], "stops": stops,
                                    "code_block": prefix.rstrip().endswith("```python"), "extra": extra}) + "\n")
                n += 1
            print(bench, "done", file=sys.stderr)
    print(f"{n} prompts -> {a.out}")


if __name__ == "__main__":
    main()

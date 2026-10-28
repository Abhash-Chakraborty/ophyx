"""Merge bench_exec.py results into one table (Markdown + JSON).

    python eval/bench_report.py results-a.json results-b.json ... --md eval/R1_RESULTS.md --json bench.json

Params come from the HF model cards when known (PARAMS below); a model's row lists every benchmark it ran.
"""

import argparse
import json
import sys

PARAMS = {  # billions, from the safetensors metadata on Hugging Face (21 Oct 2026)
    "ophyx-350m-sft": 0.349, "ophyx-350m-base": 0.349, "ophyx-125m-sft": 0.126, "ophyx-125m-base": 0.126,
    "SmolLM2-135M": 0.135, "SmolLM2-360M": 0.362, "SmolLM2-360M-Instruct": 0.362, "SmolLM2-1.7B": 1.71,
    "SmolLM2-1.7B-Instruct": 1.71, "Qwen2.5-Coder-0.5B": 0.494, "Qwen2.5-Coder-0.5B-Instruct": 0.494,
    "Qwen2.5-Coder-1.5B": 1.54, "Qwen2.5-Coder-1.5B-Instruct": 1.54, "Qwen2.5-Coder-3B": 3.09,
    "Qwen2.5-Coder-3B-Instruct": 3.09, "Qwen2.5-Coder-7B": 7.62, "Qwen2.5-Coder-7B-Instruct": 7.62,
    "Qwen3-0.6B-Base": 0.596, "Qwen3-0.6B": 0.752, "Qwen3-1.7B-Base": 1.72, "Qwen3-1.7B": 2.03, "Qwen3-4B": 4.02,
    "Qwen3-8B": 8.19, "deepseek-coder-1.3b-base": 1.35, "deepseek-coder-1.3b-instruct": 1.35,
    "deepseek-coder-6.7b-base": 6.74, "deepseek-coder-6.7b-instruct": 6.74, "phi-1": 1.42, "phi-1_5": 1.42,
    "phi-2": 2.78, "codegen-350M-mono": 0.357, "gpt_bigcode-santacoder": 1.12, "starcoder2-3b": 3.03,
    "starcoder2-7b": 7.17, "OpenCoder-1.5B-Base": 1.91, "OpenCoder-1.5B-Instruct": 1.91, "Yi-Coder-1.5B": 1.48,
    "Yi-Coder-1.5B-Chat": 1.48, "stable-code-3b": 2.80, "granite-3b-code-base-2k": 3.48,
    "granite-3b-code-instruct-2k": 3.48, "CodeLlama-7b-Python-hf": 6.74,
}
COLS = [("humaneval", "base", "HumanEval"), ("humaneval", "plus", "HumanEval+"), ("mbpp", "base", "MBPP"),
        ("mbpp", "plus", "MBPP+"), ("concept", "pass@1", "Concept"), ("bcb_complete", "pass@1_runnable", "BCB-H complete"),
        ("bcb_instruct", "pass@1_runnable", "BCB-H instruct"), ("lcb", "pass@1", "LCB"), ("crux_i", "pass@1", "CRUX-I"),
        ("crux_o", "pass@1", "CRUX-O"), ("manual", "pass@1", "Hands-on")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    merged = {}
    for p in a.results:
        for name, r in json.load(open(p)).items():
            if name == "canonical":
                merged.setdefault("_canonical", r)
                continue
            # a bench that errored in a later file must not hide a good result from an earlier one
            merged.setdefault(name, {}).update({k: v for k, v in r.items()
                                                if not (isinstance(v, dict) and "error" in v and k in merged[name])})
            for bench, v in r.items():
                if isinstance(v, dict) and "n_runnable" in v:
                    seen = merged.setdefault("_n_runnable", {}).setdefault(bench, v["n_runnable"])
                    if seen != v["n_runnable"]:
                        sys.exit(f"{bench}: {name} was scored on {v['n_runnable']} runnable tasks, others on {seen}")
    rows = []
    for name, r in merged.items():
        if name.startswith("_"):
            continue
        size = name.split("-")[1] if name.startswith("ophyx-") else ""  # DPO/GRPO runs keep their base's size
        params = PARAMS.get(name) or {"350m": 0.349, "125m": 0.126}.get(size)
        row = {"model": name, "kind": r.get("_kind"), "params_b": params}
        for bench, key, label in COLS:
            v = r.get(bench, {})
            row[label] = v.get(key) if isinstance(v, dict) else None
        rows.append(row)
    rows.sort(key=lambda x: (x["params_b"] or 99, x["model"]))
    if a.json:
        json.dump({"rows": rows, "canonical": merged.get("_canonical")}, open(a.json, "w"), indent=1)
    if a.md:
        head = "| Model | Kind | Params | " + " | ".join(c[2] for c in COLS) + " |"
        lines = [head, "|" + "---|" * (3 + len(COLS))]
        for x in rows:
            cells = [x["model"], x["kind"] or "", f"{x['params_b']:.2f}B" if x["params_b"] else ""]
            cells += ["" if x[c[2]] is None else f"{x[c[2]]:.1f}" for c in COLS]
            lines.append("| " + " | ".join(cells) + " |")
        open(a.md, "w").write("\n".join(lines) + "\n")
    print(f"{len(rows)} models")


if __name__ == "__main__":
    main()

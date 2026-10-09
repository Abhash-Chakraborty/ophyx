"""HumanEval+ and MBPP+ for base (non-chat) models with Hugging Face transformers, greedy pass@1.

Runs in a Kaggle T4 session (or anywhere with a GPU). Used for the baselines now and for our own
checkpoints once they're exported to HF format. Reports base and plus pass@1 and the share of
outputs that parse as Python.

CONFIG keys: models (list of HF ids or local paths), datasets, batch_size, max_new_tokens,
smoke (score the canonical solutions instead of generating, to check the harness itself).
"""

import ast
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
MODELS = CONFIG.get("models", ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-Coder-0.5B"])
DATASETS = CONFIG.get("datasets", ["humaneval", "mbpp"])
BATCH = CONFIG.get("batch_size", 16)
MAX_NEW = CONFIG.get("max_new_tokens", 512)
SMOKE = CONFIG.get("smoke", False)
OUT = CONFIG.get("out", "/kaggle/working" if os.path.isdir("/kaggle") else os.getcwd())
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

# Stop strings: a base model keeps writing after the solution; cut at the first new top-level block.
STOPS = {
    "humaneval": ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert ", "\n```"],
    "mbpp": ["\nassert ", '\n"""', "\nif __name__", "\nprint(", "\n# Test", "\n```", "\n\n\n\n"],
}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def ensure_evalplus():
    try:
        import evalplus  # noqa: F401
    except ImportError:
        pip = [sys.executable, "-m", "pip", "install", "-q"]
        subprocess.run(pip + ["--no-deps", "evalplus==0.3.1", "stop-sequencer"], check=True)
        subprocess.run(pip + ["wget", "tempdir", "multipledispatch", "appdirs", "termcolor", "fire",
                              "tree-sitter", "tree-sitter-python", "rich", "psutil"], check=True)


def load_keys():
    paths = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    return json.load(open(paths[0])) if paths else {}


KEYS = load_keys()


def notify(title, msg, tags="bar_chart"):
    topic = KEYS.get("NTFY_TOPIC")
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode()[:3000],
                                                          headers={"Title": title, "Tags": tags}, method="POST"), timeout=15)
        except Exception as e:
            log("ntfy failed:", e)


def problems_for(ds):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if ds == "humaneval" else get_mbpp_plus()


def truncate(text, stops):
    cut = min((text.find(s) for s in stops if s in text), default=-1)
    return text if cut < 0 else text[:cut]


def generate(model_id, prompts, stops):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()

    class StopOnStrings(StoppingCriteria):
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kw):
            tails = tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
            return torch.tensor([any(s in t for s in stops) for t in tails], device=input_ids.device)

    outs = []
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths batch well
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([StopOnStrings(enc["input_ids"].shape[1])]))
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.extend(zip(idx, texts))
    del model
    torch.cuda.empty_cache()
    return [t for _, t in sorted(outs)]


def evaluate(ds, samples_path):
    subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", ds, "--samples", samples_path,
                    "--i_just_wanna_run"], check=True, stdout=subprocess.DEVNULL)
    res_path = samples_path.replace(".jsonl", "_eval_results.json")
    res = json.load(open(res_path))["eval"]
    n = len(res)
    base = sum(r[0]["base_status"] == "pass" for r in res.values()) / n
    plus = sum(r[0]["base_status"] == "pass" and r[0]["plus_status"] == "pass" for r in res.values()) / n
    return round(100 * base, 1), round(100 * plus, 1)


def main():
    ensure_evalplus()
    from evalplus.sanitize import sanitize

    results = {}
    for model_id in (["canonical"] if SMOKE else MODELS):
        results[model_id] = {}
        for ds in DATASETS:
            probs = problems_for(ds)
            ids = list(probs)
            t = time.time()
            if SMOKE:
                sols = [probs[i]["prompt"] + probs[i]["canonical_solution"] if ds == "humaneval"
                        else probs[i]["canonical_solution"] for i in ids]
            else:
                prompts = [probs[i]["prompt"] for i in ids]
                comps = generate(model_id, prompts, STOPS[ds])
                raw = [p + truncate(c, STOPS[ds]) for p, c in zip(prompts, comps)]
                sols = [sanitize(r, probs[i]["entry_point"]) for r, i in zip(raw, ids)]
            parse_ok = sum(1 for s in sols if _parses(s)) / len(sols)
            tag = model_id.replace("/", "__")
            sp = os.path.join(OUT, f"{tag}-{ds}.jsonl")
            with open(sp, "w") as f:
                for i, s in zip(ids, sols):
                    f.write(json.dumps({"task_id": i, "solution": s}) + "\n")
            base, plus = evaluate(ds, sp)
            results[model_id][ds] = {"base": base, "plus": plus, "parses": round(100 * parse_ok, 1),
                                     "seconds": round(time.time() - t)}
            log(f"{model_id} {ds}: base {base}%, plus {plus}%, parses {parse_ok:.1%}")
            json.dump(results, open(os.path.join(OUT, "evalplus_results.json"), "w"), indent=1)
    lines = [f"{m.split('/')[-1]}: " + ", ".join(f"{d} {r['plus']}%+" for d, r in v.items()) for m, v in results.items()]
    log("SUMMARY\n" + "\n".join(lines))
    notify("baselines done", "\n".join(lines))


def _parses(s):
    try:
        ast.parse(s)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


if __name__ == "__main__":
    main()

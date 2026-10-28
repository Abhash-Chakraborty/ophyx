"""Sample a model on the verified problem bank. Kaggle T4 x2, vLLM; it executes nothing it generates.

Appended after data/jobs/verify_bank.py by the launcher (CONFIG "library": true), which supplies the
prompt wording (sft_row) so the questions read exactly as in the bank SFT set. Attach the
verify-bank kernel outputs (verified/bank-*.jsonl) and ophyx-keys when the model repo is private.
Score the samples with data/jobs/bank_judge.py on a CPU session with internet off and no keys.

CONFIG: model (HF id), name, n (samples per problem, default 8), temperature (0.8), top_p (0.95),
max_new (1024), max_len (2048, the model's window), limit (problems, for smoke runs),
include_inconsistent (false: problems whose tests can't reject wrong answers are left out),
gpus (["0", "1"]: one vLLM process per GPU, each on half the problems), greedy (also add one
temperature-0 sample, default true, so pass@1 greedy comes out of the same run), bank_glob (default
"verified/bank-*.jsonl"; "verified/unsolved-*.jsonl" for the problems no reference solves), and chat
(false: the ophyx template; true: the model's own chat template, for a teacher such as
Qwen/Qwen2.5-Coder-14B-Instruct-AWQ, whose answer is the longest ```python block in its reply).

Output, under /kaggle/working/samples/<name>/:
  bank-<gpu>.jsonl   {pid, source, level, fn_name, prompt_tokens, samples: [code], finish: [reason],
                      greedy: code or null}
  meta.json          model, settings, problem count, tokens and seconds per GPU
"""

import glob as _glob
import hashlib as _hashlib
import json as _json
import multiprocessing as _mp
import os as _os
import subprocess as _subprocess
import sys as _sys
import time as _time

S_MODEL = CONFIG.get("model", "abhash-chakraborty/ophyx-350m-sft")
S_NAME = CONFIG.get("name", S_MODEL.split("/")[-1])
S_N = CONFIG.get("n", 8)
S_TEMP = CONFIG.get("temperature", 0.8)
S_TOP_P = CONFIG.get("top_p", 0.95)
S_MAX_NEW = CONFIG.get("max_new", 1024)
S_MAX_LEN = CONFIG.get("max_len", 2048)
S_GREEDY = CONFIG.get("greedy", True)
S_GPUS = CONFIG.get("gpus", ["0", "1"])
S_OUT = _os.path.join(OUTW, "samples", S_NAME)
STOPS = ["<|end|>", "<|user|>", "<|python|>"]
S_CHAT = CONFIG.get("chat", False)
TEACHER_SYSTEM = ("You are an expert competitive programmer. Answer with one complete Python 3 program in a single "
                  "```python code block, with no explanation.")


def bank_problems():
    files = sorted(_glob.glob(_os.path.join(SRC, "**", CONFIG.get("bank_glob", "verified/bank-*.jsonl")), recursive=True))
    recs = [r for f in files for r in map(_json.loads, open(f))]
    if not CONFIG.get("include_inconsistent"):
        recs = [r for r in recs if not r.get("inconsistent")]
    recs.sort(key=lambda r: _hashlib.sha1(r["pid"].encode()).hexdigest())  # stable, mixed order
    if CONFIG.get("limit"):
        recs = recs[:CONFIG["limit"]]
    return recs


def strip_fence(text):
    """The SFT target is a bare module, but a model may still wrap it in a ```python block. A chat teacher's
    reply may also have prose around the code: take its longest fenced block."""
    if S_CHAT:
        import re as _re
        blocks = _re.findall(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)(?:```|\Z)", text, _re.S)
        return (max(blocks, key=len) if blocks else text).strip("\n") + "\n"
    t = text.strip("\n")
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if "```" in t:
            t = t[:t.index("```")]
    return t.rstrip() + "\n"


def sample_child(gi, gpus, recs, logf):
    fd = _os.open(logf, _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC)
    _os.dup2(fd, 1)
    _os.dup2(fd, 2)
    _os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    _os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    code = 0
    try:
        from vllm import LLM, SamplingParams
        t0 = _time.time()
        llm = LLM(model=S_MODEL, dtype="float16", max_model_len=S_MAX_LEN, gpu_memory_utilization=0.88, seed=gi,
                  enable_prefix_caching=True)
        tok = llm.get_tokenizer()
        end_id = tok.eos_token_id if S_CHAT else tok.convert_tokens_to_ids("<|end|>")
        prompts, keep = [], []
        for r in recs:
            if S_CHAT:
                ids = tok.apply_chat_template([{"role": "system", "content": TEACHER_SYSTEM},
                                               {"role": "user", "content": sft_row(r)["instruction"]}],
                                              add_generation_prompt=True, tokenize=True)
            else:
                ids = tok(f"<|user|>\n{sft_row(r)['instruction']}\n<|python|>\n", add_special_tokens=False)["input_ids"]
            if len(ids) + 64 <= S_MAX_LEN:
                prompts.append(ids)
                keep.append(r)
        print(f"gpu {gi}: {len(keep)}/{len(recs)} problems fit, load {_time.time() - t0:.0f}s", flush=True)

        def params(n, temp, ids):
            return SamplingParams(n=n, temperature=temp, top_p=S_TOP_P if temp > 0 else 1.0, seed=None,
                                  max_tokens=min(S_MAX_NEW, S_MAX_LEN - len(ids) - 1), stop=None if S_CHAT else STOPS,
                                  stop_token_ids=[end_id], skip_special_tokens=False)

        t1 = _time.time()
        n_tok = 0
        with open(_os.path.join(S_OUT, f"bank-{gi}.jsonl"), "w") as f:
            for b in range(0, len(prompts), 512):
                chunk, rs = prompts[b:b + 512], keep[b:b + 512]
                outs = llm.generate([{"prompt_token_ids": x} for x in chunk], [params(S_N, S_TEMP, x) for x in chunk],
                                    use_tqdm=False)
                greedy = (llm.generate([{"prompt_token_ids": x} for x in chunk], [params(1, 0.0, x) for x in chunk],
                                       use_tqdm=False) if S_GREEDY else [None] * len(chunk))
                for r, x, o, g in zip(rs, chunk, outs, greedy):
                    n_tok += sum(len(c.token_ids) for c in o.outputs) + (len(g.outputs[0].token_ids) if g else 0)
                    f.write(_json.dumps({"pid": r["pid"], "source": r["source"], "level": r.get("level"),
                                         "fn_name": r["fn_name"], "prompt_tokens": len(x),
                                         "samples": [strip_fence(c.text) for c in o.outputs],
                                         "finish": [c.finish_reason for c in o.outputs],
                                         "greedy": strip_fence(g.outputs[0].text) if g else None}) + "\n")
                f.flush()
                print(f"gpu {gi}: {b + len(chunk)}/{len(prompts)}, {n_tok / (_time.time() - t1):.0f} tok/s", flush=True)
        _json.dump({"problems": len(keep), "skipped_long": len(recs) - len(keep), "gen_tokens": n_tok,
                    "seconds": round(_time.time() - t1)}, open(_os.path.join(S_OUT, f"stats-{gi}.json"), "w"))
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(code)


def sample_main():
    keys = _glob.glob("/kaggle/input/**/keys.json", recursive=True)
    if keys and _json.load(open(keys[0])).get("HF_TOKEN"):
        _os.environ["HF_TOKEN"] = _json.load(open(keys[0]))["HF_TOKEN"]
    if _os.path.isdir("/kaggle"):
        _subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
        _subprocess.run([_sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
    _os.makedirs(S_OUT, exist_ok=True)
    recs = bank_problems()
    log(f"{len(recs)} problems, n={S_N} at T={S_TEMP}" + (" plus greedy" if S_GREEDY else "") + f", model {S_MODEL}")
    ctx = _mp.get_context("fork")
    procs = []
    for gi, g in enumerate(S_GPUS):
        p = ctx.Process(target=sample_child, args=(gi, g, recs[gi::len(S_GPUS)], _os.path.join(S_OUT, f"log-{gi}.txt")))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    for gi in range(len(S_GPUS)):
        log(f"gpu {gi} exit {procs[gi].exitcode}; log tail:\n" + open(_os.path.join(S_OUT, f"log-{gi}.txt")).read()[-2500:])
    stats = [_json.load(open(f)) for f in sorted(_glob.glob(_os.path.join(S_OUT, "stats-*.json")))]
    _json.dump({"model": S_MODEL, "name": S_NAME, "n": S_N, "temperature": S_TEMP, "top_p": S_TOP_P,
                "max_new": S_MAX_NEW, "max_len": S_MAX_LEN, "greedy": S_GREEDY, "gpus": stats},
               open(_os.path.join(S_OUT, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    sample_main()

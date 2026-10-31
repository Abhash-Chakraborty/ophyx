"""Sample Ophyx models on chat prompts (round three's eval, and DPO's on-policy samples). Kaggle T4, vLLM.

Prompts are rows {"id", "turns": [[prompt, answer-or-null], ...], "check", ...} from
data/jobs/chat_build.py (chat_eval.jsonl or chat_prompts.jsonl). Earlier turns go in front of the
last one exactly as serve/modal_app.py sends history. No generated code runs here: score the output
with data/jobs/chat_check.py (it reads chat/samples-*.jsonl and their inline checks).

CONFIG: prompts_glob, models [{"name", "id" (HF id or /kaggle glob)}], n (samples per prompt besides
greedy, 0), temperature (0.8), max_new (1024), limit (first N prompts, all by default), budget_s.
Output: /kaggle/working/chat/samples-<name>.jsonl, each row the prompt row plus "model", "answers"
(greedy first) and "finish".
"""

import glob
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
SRC = CONFIG.get("src", "/kaggle/input")
OUT = os.path.join(CONFIG.get("out", "/kaggle/working"), "chat")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def render(turns):
    text = ""
    for prompt, answer in turns[:-1]:
        text += f"<|user|>\n{prompt}\n<|python|>\n{answer.strip(chr(10))}\n<|end|>\n"
    return text + f"<|user|>\n{turns[-1][0]}\n<|python|>\n"


def rows():
    files = sorted(glob.glob(CONFIG["prompts_glob"], recursive=True))
    out = [json.loads(x) for f in files[:1] for x in open(f)]
    return out[:CONFIG["limit"]] if CONFIG.get("limit") else out


def run_model(spec):
    from vllm import LLM, SamplingParams
    mid = spec["id"]
    if mid.startswith("/") and "*" in mid:
        mid = sorted(glob.glob(mid, recursive=True))[0]
    llm = LLM(model=mid, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    stops = [tok.convert_tokens_to_ids("<|end|>"), tok.convert_tokens_to_ids("<|user|>")]
    data = rows()
    prompts, keep = [], []
    for r in data:
        ids = tok(render(r["turns"]), add_special_tokens=False)["input_ids"]
        if len(ids) + 128 < 4096:
            prompts.append(ids)
            keep.append(r)

    def params(n, temp, ids):
        return SamplingParams(n=n, temperature=temp, top_p=0.95 if temp else 1.0, stop_token_ids=stops,
                              max_tokens=min(CONFIG.get("max_new", 1024), 4096 - len(ids) - 1))

    t = time.time()
    greedy = llm.generate([{"prompt_token_ids": x} for x in prompts], [params(1, 0.0, x) for x in prompts], use_tqdm=False)
    n = CONFIG.get("n", 0)
    samples = (llm.generate([{"prompt_token_ids": x} for x in prompts],
                            [params(n, CONFIG.get("temperature", 0.8), x) for x in prompts], use_tqdm=False)
               if n else [None] * len(prompts))
    with open(os.path.join(OUT, f"samples-{spec['name']}.jsonl"), "w") as f:
        for r, g, s in zip(keep, greedy, samples):
            outs = [g.outputs[0]] + (list(s.outputs) if s else [])
            f.write(json.dumps({**r, "model": spec["name"], "answers": [o.text.strip("\n") for o in outs],
                                "finish": [o.finish_reason for o in outs]}) + "\n")
    print(f"{spec['name']}: {len(keep)}/{len(data)} prompts in {time.time() - t:.0f}s", flush=True)


def child(spec, logf):
    os.setsid()
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.get("gpu", 0))
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    code = 0
    try:
        run_model(spec)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    keys = glob.glob(os.path.join(SRC, "**", "keys.json"), recursive=True)
    if keys and json.load(open(keys[0])).get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = json.load(open(keys[0]))["HF_TOKEN"]
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    if os.path.isdir("/kaggle"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
    ctx = mp.get_context("fork")
    models = CONFIG["models"]
    budget = CONFIG.get("budget_s", 10 * 3600)
    # two at a time, one per GPU: the 350M model fits many times over, so a second GPU doubles throughput
    for b in range(0, len(models), 2):
        if time.time() - T0 > budget:
            log("time budget reached")
            break
        procs = []
        for k, spec in enumerate(models[b:b + 2]):
            spec = {**spec, "gpu": k}
            p = ctx.Process(target=child, args=(spec, os.path.join(OUT, "logs", f"{spec['name']}.log")))
            p.start()
            procs.append((spec, p))
        for spec, p in procs:
            p.join(timeout=max(60, budget + 1800 - (time.time() - T0)))
            if p.is_alive():
                p.kill()
                p.join()
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            logf = os.path.join(OUT, "logs", f"{spec['name']}.log")
            log(f"{spec['name']}: exit {p.exitcode}\n" + (open(logf).read()[-1500:] if os.path.exists(logf) else ""))


if __name__ == "__main__":
    main()

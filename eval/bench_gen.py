"""Generate answers for every benchmark with vLLM on Kaggle's 2x T4, greedy. Runs no generated code.

The kit (eval/bench_kit.py output) and the private eval data (Kaggle dataset pycoder-evaldata: the
concept suite and the hands-on prompts) are attached as inputs. Answers go to
<out>/samples/<model>/<bench>.jsonl; eval/bench_exec.py scores them offline, with no keys.

CONFIG:
  models   list of {"id", "kind": "base" | "instruct" | "ours", "name"?, "tp": 1 | 2, "max_len"?,
           "benches"?: [...] (default: all that apply to the kind), "think": false (Qwen3),
           "trust_remote_code": false}
  benches  default benches for every model
  budget_s no new model starts after this many seconds (default 10 h); a running one is killed 30 min later
  out      default /kaggle/working

Prompt formats, chosen to match how each family is usually reported:
  base      plain completion (EvalPlus base prompts, BigCodeBench "complete", CRUXEval few-shot)
  instruct  the model's chat template; EvalPlus/BigCodeBench use EvalPlus's instruction and response
            prefix; LiveCodeBench uses its official system and format prompt
  ours      the ophyx SFT template <|user|>\\n...\\n<|python|>\\n, answer ends at <|end|>;
            the function signature (HumanEval, BigCodeBench) or starter code (LiveCodeBench) is prefilled

Tiny models run two at a time, one per T4, each in its own forked process (vLLM frees GPU memory
reliably only on exit). Models with "tp": 2 use both GPUs. A model vLLM can't load falls back to transformers.
"""

import glob
import json
import os
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
OUT = CONFIG.get("out", "/kaggle/working" if os.path.isdir("/kaggle") else os.path.abspath("bench"))
ALL_BENCHES = ["humaneval", "mbpp", "concept", "crux_o", "crux_i", "lcb", "bcb_complete", "bcb_instruct", "manual"]
DEFAULT_BENCHES = CONFIG.get("benches", ALL_BENCHES)
T0 = time.time()

INSTR = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESP = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
LCB_SYS = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
           "generate a correct Python program that matches the specification and passes all tests.")
LCB_FMT_CALL = ("### Format: You will use the following starter code to write the solution to the problem and "
                "enclose your code within delimiters.\n```python\n{starter}\n```\n\n### Answer: (use the provided "
                "format with backticks)\n\n")
LCB_FMT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not "
                 "directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that "
                 "when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                 "\n```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n")
CRUX_O = '''You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
'''
CRUX_I = '''You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
'''
# A base model keeps writing after the solution; cut at the first new top-level block.
HE_STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert ", "\n```"]
MBPP_STOPS = ["\nassert ", '\n"""', "\nif __name__", "\nprint(", "\n# Test", "\n```", "\n\n\n\n"]
MODULE_STOPS = ["\nif __name__", '\n"""', "\n'''", "\n```", "\n# Test", "\n# Example", "\nassert ", "\nprint("]
STDIN_STOPS = ['\n"""', "\n'''", "\n```", "\n# Test", "\n# Example", "\nassert "]
OURS_STOPS = ["<|end|>", "<|user|>", "<|python|>"]
MAX_NEW = {"humaneval": 512, "mbpp": 512, "concept": 1024, "crux_o": 160, "crux_i": 160, "lcb": 1536,
           "bcb_complete": 1024, "bcb_instruct": 1024, "manual": 1536}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) + sorted(glob.glob(os.path.join(CONFIG.get("kit_dir", "/nonexistent"), name)))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def rows(name):
    return [json.loads(line) for line in open(find(name)) if line.strip()]


def model_name(spec):
    return spec.get("name") or spec["id"].rstrip("/").split("/")[-1]


def benches_for(spec):
    bs = spec.get("benches") or DEFAULT_BENCHES
    if spec["kind"] == "base":
        bs = [b for b in bs if b not in ("bcb_instruct", "manual")]
    return bs


# ---------------------------------------------------------------- prompts


class Prompter:
    """Builds (text, stops, prefill) per task for one model kind. `text` is the full prompt string; for chat
    models it is the rendered chat template, so it is tokenised without adding special tokens again."""

    def __init__(self, kind, tok, think=False):
        self.kind, self.tok, self.think = kind, tok, think

    def chat(self, user, prefix="", system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": False} if not self.think else {}
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + prefix

    @staticmethod
    def ours(user, prefill=""):
        return f"<|user|>\n{user}\n<|python|>\n" + prefill

    def evalplus(self, ds, p):
        if self.kind == "base":
            text = p["prompt"].rstrip() if ds == "humaneval" else p["prompt"]
            return text, HE_STOPS if ds == "humaneval" else MBPP_STOPS, text
        if self.kind == "instruct":
            return self.chat(f"{INSTR}\n```\n{p['prompt'].strip()}\n```", f"{RESP}\n```python\n"), ["\n```"], ""
        if ds == "humaneval":
            task = f"Complete this Python function:\n```python\n{p['prompt'].strip()}\n```"
            pre = p["prompt"].rstrip()
        else:
            task, pre = p["prompt"].strip().strip('"').strip(), ""
        return self.ours(task, pre), ["\n```"] + OURS_STOPS, pre

    def concept(self, t):
        if self.kind == "base":
            return '"""' + t["prompt"].replace('"""', "'''") + '\n"""\n', MODULE_STOPS, ""
        if self.kind == "instruct":
            return self.chat(t["prompt"] + "\n\nWrite the complete Python module in a single ```python code block."), [], ""
        return self.ours(t["prompt"]), OURS_STOPS, ""

    def crux(self, mode, r):
        text = CRUX_O.format(code=r["code"], input=r["input"]) if mode == "o" else CRUX_I.format(code=r["code"], output=r["output"])
        if self.kind == "instruct":
            return self.chat(text), ["[/ANSWER]"], ""
        return text, ["[/ANSWER]"], ""  # base and ours: the official few-shot completion prompt

    def lcb(self, r):
        call = r["testtype"] == "functional"
        if self.kind == "instruct":
            fmt = LCB_FMT_CALL.format(starter=r["starter_code"]) if call else LCB_FMT_STDIN
            return self.chat(f"### Question:\n{r['question']}\n\n{fmt}", system=LCB_SYS), [], ""
        if self.kind == "ours":
            if call:
                return (self.ours(r["question"] + "\n\nUse this starter code:\n```python\n" + r["starter_code"].rstrip() + "\n```",
                                  r["starter_code"]), OURS_STOPS, r["starter_code"])
            return (self.ours(r["question"] + "\n\nRead the input from standard input and print the answer to standard output."),
                    OURS_STOPS, "")
        head = '"""\n' + r["question"].replace('"""', "'''").strip() + '\n"""\n'
        if call:
            return head + "from typing import *\n\n" + r["starter_code"], MODULE_STOPS, "from typing import *\n\n" + r["starter_code"]
        return head + "import sys\n", STDIN_STOPS, "import sys\n"

    def bcb(self, split, r):
        if split == "complete":
            if self.kind == "base":
                return r["complete_prompt"].rstrip(), HE_STOPS, r["complete_prompt"].rstrip()
            if self.kind == "instruct":
                return self.chat(f"{INSTR}\n```\n{r['complete_prompt'].strip()}\n```", f"{RESP}\n```python\n"), ["\n```"], ""
            pre = r["complete_prompt"].rstrip()
            return self.ours(f"Complete this Python function:\n```python\n{r['complete_prompt'].strip()}\n```", pre), ["\n```"] + OURS_STOPS, pre
        if self.kind == "instruct":
            return self.chat(f"{INSTR}\n```\n{r['instruct_prompt'].strip()}\n```", f"{RESP}\n```python\n"), ["\n```"], ""
        pre = r["code_prompt"].rstrip()
        return self.ours(r["instruct_prompt"].strip(), pre), ["\n```"] + OURS_STOPS, pre

    def manual(self, r):
        if self.kind == "instruct":
            return self.chat(r["prompt"]), [], ""
        return self.ours(r["prompt"]), OURS_STOPS, ""


def build(bench, pr):
    """[(task_id, text, stops, prefill, extra)] for one bench."""
    out = []
    if bench in ("humaneval", "mbpp"):
        for p in rows(f"{bench}.jsonl"):
            out.append((p["task_id"], *pr.evalplus(bench, p), {"entry_point": p["entry_point"]}))
    elif bench == "concept":
        for t in rows("tasks.jsonl"):
            out.append((t["task_id"], *pr.concept(t), {}))
    elif bench in ("crux_o", "crux_i"):
        for r in rows("cruxeval.jsonl"):
            out.append((r["id"], *pr.crux(bench[-1], r), {}))
    elif bench == "lcb":
        for r in rows("lcb.jsonl"):
            out.append((r["task_id"], *pr.lcb(r), {}))
    elif bench.startswith("bcb_"):
        for r in rows("bcb_hard.jsonl"):
            out.append((r["task_id"], *pr.bcb(bench[4:], r), {"entry_point": "task_func"}))
    elif bench == "manual":
        for r in rows("manual.jsonl"):
            out.append((r["task_id"], *pr.manual(r), {}))
    return out[:CONFIG["limit"]] if CONFIG.get("limit") else out  # limit: smoke runs only


def cut(text, stops):
    idx = [i for i in (text.find(s) for s in stops) if i >= 0]
    return text[:min(idx)] if idx else text


def looks_bytelevel(text):
    """Raw byte-level BPE text ("\u0120return" for " return"): the tokenizer lost its byte-level steps."""
    return ("\u0120" in text or "\u010a" in text) and " " not in text and "\n" not in text


def check_tokenizer(tok, name):
    """deepseek-coder's tokenizer on Kaggle (vLLM + that transformers build) encoded prompts without spaces and
    decoded raw byte-level text, so it scored 0 everywhere. Refuse to run a model whose round trip fails."""
    probe = "def f(x):\n    return x + 1  # ok"
    back = tok.decode(tok.encode(probe, add_special_tokens=False))
    if back != probe:
        raise SystemExit(f"{name}: tokenizer round trip fails ({back!r}); not running it")


def fixed_tokenizer(spec):
    """Some transformers builds convert deepseek-coder's byte-level BPE into a broken slow-style tokenizer.
    If the round trip fails, load tokenizer.json as-is and save it where vLLM and transformers can both read it."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    trc = spec.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=trc)
    probe = "def f(x):\n    return x + 1  # ok"
    if tok.decode(tok.encode(probe, add_special_tokens=False)) == probe:
        return None
    from huggingface_hub import hf_hub_download
    fixed = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(spec["id"], "tokenizer.json"),
                                    bos_token=tok.bos_token, eos_token=tok.eos_token, pad_token=tok.pad_token,
                                    unk_token=tok.unk_token)
    fixed.chat_template = tok.chat_template
    check_tokenizer(fixed, spec["id"] + " (rebuilt)")
    path = os.path.join(OUT, "tokenizers", model_name(spec))
    fixed.save_pretrained(path)
    log(f"{spec['id']}: tokenizer round trip failed; using tokenizer.json as-is from {path}")
    return path


# ---------------------------------------------------------------- engines


class VLLMEngine:
    def __init__(self, spec, max_len):
        from vllm import LLM
        self.llm = LLM(model=spec["id"], tokenizer=spec.get("tokenizer") or spec["id"], tensor_parallel_size=spec.get("tp", 1),
                       dtype=spec.get("dtype", "float16"), max_model_len=max_len, gpu_memory_utilization=0.88, seed=0,
                       trust_remote_code=spec.get("trust_remote_code", False), enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        check_tokenizer(self.tok, spec["id"])

    def generate(self, ids, max_tokens, stops, stop_ids):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=0.0, max_tokens=m, stop=s or None, stop_token_ids=stop_ids or None,
                                 skip_special_tokens=False, include_stop_str_in_output=False) for m, s in zip(max_tokens, stops)]
        outs = self.llm.generate([{"prompt_token_ids": x} for x in ids], params, use_tqdm=False)
        res = [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]
        bad = sum(looks_bytelevel(t) for t, _ in res)
        if bad > len(res) // 4:  # vLLM detokenizes on its own path, which the round-trip check doesn't cover
            raise SystemExit(f"{bad}/{len(res)} answers are raw byte-level text: the tokenizer is broken here")
        return res


class HFEngine:
    """Fallback for architectures vLLM doesn't support (e.g. CodeGen). Slower, same greedy decoding."""

    def __init__(self, spec, max_len):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_len = torch, max_len
        trc = spec.get("trust_remote_code", False)
        self.tok = AutoTokenizer.from_pretrained(spec.get("tokenizer") or spec["id"], trust_remote_code=trc)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        check_tokenizer(self.tok, spec["id"])
        self.model = AutoModelForCausalLM.from_pretrained(spec["id"], dtype=torch.float16,
                                                          trust_remote_code=trc).to("cuda").eval()

    def generate(self, ids, max_tokens, stops, stop_ids):
        torch = self.torch
        res = [None] * len(ids)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        eos = list(stop_ids or []) + ([self.tok.eos_token_id] if self.tok.eos_token_id is not None else [])
        for b in range(0, len(order), 8):
            idx = order[b:b + 8]
            L = max(len(ids[i]) for i in idx)
            pad = self.tok.pad_token_id
            x = torch.tensor([[pad] * (L - len(ids[i])) + ids[i] for i in idx], device="cuda")
            att = torch.tensor([[0] * (L - len(ids[i])) + [1] * len(ids[i]) for i in idx], device="cuda")
            new = min(max(max_tokens[i] for i in idx), self.max_len - L)  # padding counts against the window
            if new < 1:
                for i in idx:
                    res[i] = ("", "too_long")
                continue
            with torch.no_grad():
                g = self.model.generate(input_ids=x, attention_mask=att, max_new_tokens=new,
                                        do_sample=False, pad_token_id=pad, eos_token_id=eos or None,
                                        stop_strings=sorted({s for i in idx for s in stops[i]}) or None, tokenizer=self.tok)
            for j, i in enumerate(idx):
                t = self.tok.decode(g[j, L:], skip_special_tokens=False)
                for s in (self.tok.pad_token, self.tok.eos_token):
                    if s:
                        t = t.split(s)[0]
                res[i] = (cut(t, stops[i]), "stop")
        return res


def model_max_len(spec):
    from transformers import AutoConfig
    try:
        c = AutoConfig.from_pretrained(spec["id"], trust_remote_code=spec.get("trust_remote_code", False))
        native = getattr(c, "max_position_embeddings", None) or getattr(c, "n_positions", None) or 4096
    except Exception:
        native = 4096
    return int(min(spec.get("max_len", 4096), native))


def run_model(spec):
    name = model_name(spec)
    mdir = os.path.join(OUT, "samples", name)
    os.makedirs(mdir, exist_ok=True)
    max_len = model_max_len(spec)
    t = time.time()
    spec = {**spec, "tokenizer": fixed_tokenizer(spec)}
    try:
        eng = VLLMEngine(spec, max_len)
        engine = "vllm"
    except Exception as e:
        log(f"{name}: vLLM failed ({type(e).__name__}: {str(e)[:300]}); using transformers")
        eng = HFEngine(spec, max_len)
        engine = "transformers"
    tok = eng.tok
    load_s = time.time() - t
    pr = Prompter(spec["kind"], tok, spec.get("think", False))
    stop_ids = [tok.convert_tokens_to_ids("<|end|>")] if spec["kind"] == "ours" else []
    meta = {"model": spec["id"], "name": name, "kind": spec["kind"], "engine": engine, "max_len": max_len,
            "load_s": round(load_s), "benches": {}}
    for bench in benches_for(spec):
        t = time.time()
        try:
            items = build(bench, pr)
        except FileNotFoundError as e:
            log(f"{name} {bench}: no data ({e}); skipped")
            continue
        chat_text = spec["kind"] in ("instruct",)  # rendered chat templates already carry their special tokens
        ids = [tok(text, add_special_tokens=not chat_text)["input_ids"] for _, text, _, _, _ in items]
        maxes = [min(MAX_NEW[bench], max_len - len(x) - 1) for x in ids]
        live = [i for i, m in enumerate(maxes) if m >= 32]
        outs = dict(zip(live, eng.generate([ids[i] for i in live], [maxes[i] for i in live],
                                           [items[i][2] for i in live], stop_ids)))
        n_tok = 0
        with open(os.path.join(mdir, f"{bench}.jsonl"), "w") as f:
            for i, (tid, _, stops, pre, extra) in enumerate(items):
                text, fin = outs.get(i, ("", "too_long"))
                text = cut(text, stops)
                n_tok += len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
                f.write(json.dumps({"task_id": tid, "prefill": pre, "completion": text, "finish": fin,
                                    "prompt_tokens": len(ids[i]), **extra}) + "\n")
        sec = time.time() - t
        meta["benches"][bench] = {"n": len(items), "too_long": len(items) - len(live), "seconds": round(sec),
                                  "gen_tokens": n_tok, "tok_per_s": round(n_tok / max(sec, 1e-9))}
        log(f"{name} {bench}: {len(items)} prompts ({len(items) - len(live)} too long), {sec:.0f}s, {n_tok / max(sec, 1e-9):.0f} tok/s")
        json.dump(meta, open(os.path.join(mdir, "meta.json"), "w"), indent=1)
    return meta


# ---------------------------------------------------------------- orchestration


def setup():
    keys = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    if keys:  # private ophyx checkpoints on HF
        k = json.load(open(keys[0]))
        if k.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = k["HF_TOKEN"]
    if os.path.isdir("/kaggle"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
        # vllm brings its own torch build; the preinstalled torchaudio no longer matches and breaks transformers
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)


def child(spec, gpus, logf):
    """One model in a forked process: CUDA is first touched here, and all GPU memory goes back on exit."""
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        run_model(spec)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate():
    """Kaggle runs a script kernel as a notebook, so there is no file to re-run: fork one process per model.
    The parent never imports torch or vLLM, so forking is safe."""
    import multiprocessing as mp
    setup()
    ctx = mp.get_context("fork")
    models = list(CONFIG["models"])
    done = {}
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)

    def start_child(spec, gpus):
        p = ctx.Process(target=child, args=(spec, gpus, os.path.join(OUT, "logs", f"{model_name(spec)}.log")))
        p.start()
        return p

    def finish_child(p, spec, gpus, t):
        if p.is_alive():
            p.kill()
        p.join()
        name, rc = model_name(spec), p.exitcode
        logf = os.path.join(OUT, "logs", f"{name}.log")
        tail = open(logf).read()[-1500:] if os.path.exists(logf) else ""
        if not spec["id"].startswith("/"):  # 7B weights are 15 GB each: free the disk for the next model
            hub = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
            subprocess.run(["rm", "-rf", os.path.join(hub, "models--" + spec["id"].replace("/", "--"))], check=False)
        done[name] = {"rc": rc, "seconds": round(time.time() - t), "gpus": gpus}
        log(f"{name}: rc {rc} in {time.time() - t:.0f}s" + ("" if rc == 0 else f"\n{tail}"))
        json.dump(done, open(os.path.join(OUT, "gen_status.json"), "w"), indent=1)

    # No threads here: forking a multi-threaded parent can deadlock the child (one hung silently in the
    # smoke run). The main thread starts the children and polls them, one per free GPU.
    single = [m for m in models if m.get("tp", 1) == 1]
    double = [m for m in models if m.get("tp", 1) == 2]
    # Kaggle drops the whole output when a session hits its 12 h limit, so stop starting models after
    # budget_s and kill whatever still runs at budget_s + 30 min. Finished benches are already on disk.
    budget = CONFIG.get("budget_s", 10 * 3600)
    hard = budget + 1800
    timeout = CONFIG.get("model_timeout_s", 3 * 3600)

    def left():
        return hard - (time.time() - T0)

    def skip(rest):
        for spec in rest:
            done[model_name(spec)] = {"rc": None, "skipped": "session budget"}
            log(f"{model_name(spec)}: skipped, session budget used up")
        rest.clear()

    running = {}  # gpu -> (process, spec, start)
    while single or running:
        if time.time() - T0 > budget:
            skip(single)
        for gpu in ("0", "1"):
            if gpu not in running and single:
                spec = single.pop(0)
                running[gpu] = (start_child(spec, gpu), spec, time.time())
        for gpu, (p, spec, t) in list(running.items()):
            if not p.is_alive() or time.time() - t > timeout or left() < 0:
                finish_child(p, spec, gpu, t)
                del running[gpu]
        time.sleep(5)
    while double:
        if time.time() - T0 > budget:
            skip(double)
            break
        spec = double.pop(0)
        p, t = start_child(spec, "0,1"), time.time()
        p.join(max(1, min(timeout, left())))
        finish_child(p, spec, "0,1", t)
    json.dump(done, open(os.path.join(OUT, "gen_status.json"), "w"), indent=1)
    log("GEN DONE", json.dumps(done))


if __name__ == "__main__":
    orchestrate()

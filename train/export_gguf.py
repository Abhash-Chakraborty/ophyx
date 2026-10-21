"""Export an HF checkpoint to GGUF and check it against the HF tokenizer and model. Kaggle CPU, keys.

It runs the model but never executes what it writes. Steps: clone llama.cpp at a pinned tag, build
llama-tokenize and llama-cli, convert with convert_hf_to_gguf.py, quantise, then check:
  1. tokens: llama-tokenize on a set of Python snippets must give exactly the HF token ids
  2. text: greedy llama-cli output for a few prompts, next to greedy transformers output (fp32)
Our tokenizer is GPT-2 byte-level BPE with single digits split first. That is llama.cpp's
"smollm" pre-tokenizer (\\p{N}, then the GPT-2 regex), but the converter only knows tokenizers
by a hash of their output, so the hash lookup is patched to return "smollm" for ours.

CONFIG: model (HF id), out_repo (HF model repo for the .gguf files; none: keep them in
/kaggle/working), name (file prefix, default pycoder), types (["f32", "f16", "q8_0"]),
llama_tag (default b6800), prompts (for the text check).

Output under /kaggle/working/gguf/: <name>-<type>.gguf, check.json.
"""

import glob
import json
import os
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
MODEL = CONFIG.get("model", "abhash-chakraborty/pycoder-base350m-sft")
NAME = CONFIG.get("name", "pycoder")
TYPES = CONFIG.get("types", ["f32", "f16", "q8_0"])
TAG = CONFIG.get("llama_tag", "b6800")
WORK = "/kaggle/tmp/gguf" if os.path.isdir("/kaggle") else os.path.abspath("gguf-work")
OUT = os.path.join(CONFIG.get("out", "/kaggle/working"), "gguf")
T0 = time.time()
SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "x = 12345 + 0.5e-3\nprint(f\"{x:,.2f}\")\n",
    "import os\n\n\nclass Thing:\n    \"\"\"Docstring.\"\"\"\n\n    def __init__(self):\n        self.items = []\n",
    "for i in range(10):\n\tif i % 2 == 0:\n\t\tcontinue\n",
    "s = 'héllo wörld — ünïcode ✓'\n# comment with   spaces\n",
    "<|user|>\nWrite a function that reverses a string.\n<|python|>\n",
    "async def main():\n    await asyncio.sleep(0.1)\n\n\nasyncio.run(main())\n",
]
PROMPTS = CONFIG.get("prompts", [
    "Write a function `is_palindrome(s)` that ignores case and non-letters.",
    "Read a CSV file `data.csv` and print the average of the `price` column.",
    "Fix this code:\n```python\ndef mean(xs):\n    return sum(xs) / len(xs) + 1\n```",
])


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def sh(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, text=True, capture_output=True, **kw).stdout


def main():
    keys = glob.glob("/kaggle/input/**/keys.json", recursive=True)
    if keys:
        os.environ["HF_TOKEN"] = json.load(open(keys[0]))["HF_TOKEN"]
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    from huggingface_hub import snapshot_download
    mdir = snapshot_download(MODEL, local_dir=os.path.join(WORK, "hf"), allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tc = os.path.join(mdir, "tokenizer_config.json")
    cfg = json.load(open(tc))
    cfg.update(add_bos_token=False, add_eos_token=False)  # prompts start with <|user|>, not a BOS token
    json.dump(cfg, open(tc, "w"), indent=1)
    lc = os.path.join(WORK, "llama.cpp")
    if not os.path.isdir(lc):
        sh(["git", "clone", "--depth", "1", "--branch", TAG, "https://github.com/ggml-org/llama.cpp", lc])
    # not llama.cpp's requirements file: it pins torch 2.6 and would replace the image's torch
    sh([sys.executable, "-m", "pip", "install", "-q", "mistral-common>=1.8.3", "sentencepiece", "protobuf"])
    sh(f"cmake -S {lc} -B {lc}/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null && "
       f"cmake --build {lc}/build -j {os.cpu_count()} --target llama-tokenize llama-cli llama-quantize >/dev/null")

    conv = os.path.join(lc, "convert_hf_to_gguf.py")
    src = open(conv).read()
    hook = "    def get_vocab_base_pre(self, tokenizer) -> str:\n"
    assert hook in src, "convert_hf_to_gguf.py changed: get_vocab_base_pre not found"
    patched = src.replace(hook, hook + "        if getattr(tokenizer, 'convert_tokens_to_ids', None) and "
                          "tokenizer.convert_tokens_to_ids('<|python|>') == 2:\n            return 'smollm'  # pycoder\n", 1)
    open(os.path.join(lc, "convert_pycoder.py"), "w").write(patched)

    files = {}
    base = os.path.join(OUT, f"{NAME}-f32.gguf")
    sh([sys.executable, os.path.join(lc, "convert_pycoder.py"), mdir, "--outtype", "f32", "--outfile", base])
    files["f32"] = base
    for t in TYPES:
        if t == "f32":
            continue
        dst = os.path.join(OUT, f"{NAME}-{t}.gguf")
        if t in ("f16", "bf16"):
            sh([sys.executable, os.path.join(lc, "convert_pycoder.py"), mdir, "--outtype", t, "--outfile", dst])
        else:
            sh([os.path.join(lc, "build", "bin", "llama-quantize"), base, dst, t.upper()])
        files[t] = dst
    if "f32" not in TYPES:
        os.remove(base)
        files.pop("f32")
    check = {"model": MODEL, "llama_tag": TAG, "files": {t: os.path.getsize(f) for t, f in files.items()}}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir)
    gg = files.get("f32") or files.get("f16") or next(iter(files.values()))
    toks = []
    for s in SNIPPETS:
        want = tok(s, add_special_tokens=False)["input_ids"]
        out = sh([os.path.join(lc, "build", "bin", "llama-tokenize"), "-m", gg, "-p", s, "--ids", "--log-disable",
                  "--no-bos", "--parse-special"])
        got = json.loads(out.strip().splitlines()[-1])
        toks.append({"text": s, "match": got == want, "hf": len(want), "gguf": len(got)})
    check["tokens_all_match"] = all(t["match"] for t in toks)
    check["tokens"] = toks
    log("tokens match:", check["tokens_all_match"])

    import torch
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32).eval()
    end = tok.convert_tokens_to_ids("<|end|>")
    texts = []
    for p in PROMPTS:
        full = f"<|user|>\n{p}\n<|python|>\n"
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=160, do_sample=False, eos_token_id=end, pad_token_id=end)
        hf_text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        row = {"prompt": p, "hf": hf_text}
        for t, f in files.items():
            out = subprocess.run([os.path.join(lc, "build", "bin", "llama-cli"), "-m", f, "-p", full, "-n", "160",
                                  "--temp", "0", "--no-warmup", "-no-cnv", "--no-display-prompt", "--log-disable"],
                                 text=True, capture_output=True, timeout=600, stdin=subprocess.DEVNULL).stdout
            row[t] = out
        texts.append(row)
    check["texts"] = texts
    json.dump(check, open(os.path.join(OUT, "check.json"), "w"), indent=1)
    log("check", json.dumps(check)[:3000])
    if CONFIG.get("out_repo"):
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(CONFIG["out_repo"], repo_type="model", private=True, exist_ok=True)
        for f in list(files.values()) + [os.path.join(OUT, "check.json")]:
            api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=CONFIG["out_repo"])
        log("uploaded to", CONFIG["out_repo"])


main()

"""ophyx demo endpoint on Modal. Scales to zero when nobody is using it.

    modal deploy serve/modal_app.py                    # CPU: the GGUF build on llama.cpp (default)
    OPHYX_HW=t4 modal deploy serve/modal_app.py        # GPU: fp32 weights with transformers on a T4

Both need `modal token new` once and a Modal secret "ophyx-hf" (HF_TOKEN; the repos are public now,
so an empty token works). OPHYX_APP renames the app, to run both side by side for a comparison.

Endpoints (FastAPI), private: Modal's edge refuses any request without a workspace proxy token
(`modal workspace proxy-tokens create`, sent as the Modal-Key and Modal-Secret headers), before a
container starts. Only the website's server holds the token; browsers talk to the website, which
checks Turnstile and its own rate limits first.
  GET  /health                     {"ok": true, "model": ..., "hw": ...}; also wakes the model ahead of a chat
  GET  /system                     {"system": "..."}: the rules every answer follows (replies.SYSTEM_PROMPT)
  POST /generate                   {"prompt": "...", "max_tokens": 512, "temperature": 0.2, "concise": false,
                                    "history": [{"prompt": "...", "answer": "..."}, ...]}
                                   -> text/event-stream of {"text": "..."} chunks, then {"done": true, ...}
                                   Prompts the model shouldn't answer get a fixed answer from replies.route():
                                   questions about Ophyx or Abhash, small talk, attempts to change its
                                   role and harmful requests. Everything else reaches the model. A first
                                   {"route": "about", "reply": <kind>} event marks them. Answers that start
                                   repeating themselves end early with "looped": true.
                                   The done event carries "tokens" (new), "prompt_tokens" and "length"
                                   (true when the answer hit max_tokens), "cold" (it waited for a container
                                   start; "boot_s" says how long loading took) and "queued_s" (time
                                   waiting for the one generation slot).
                                   "concise": true strips comments and docstrings and stops at an example
                                   or test block (replies.ConciseFilter), unless the prompt asks for them;
                                   the done event then says "concise" and "trimmed" (a tail was cut).
                                   "history" holds earlier turns the model answered, oldest first, so a
                                   follow-up ("no, print it in red") sees the code it refers to. Turns go
                                   in as <|user|> prompt <|python|> answer <|end|>, joined by a newline; the
                                   oldest are dropped until the whole prompt fits MAX_CONTEXT_TOKENS, and
                                   the done event says how many were used ("history").
                                   "live": true streams the answer token by token instead of line by line.
                                   The loop guard and the concise filter decide per line, so a token
                                   they later remove is taken back: an event may carry "undo": n, the
                                   number of characters to delete from the end before adding its
                                   "text". Without it, lines arrive whole and the last two wait for the
                                   loop check (what the OpenAI-style API needs, since it can't take
                                   text back).

Cost controls: one container at most, it shuts down after SCALEDOWN_S idle seconds, each request is
capped in prompt length, new tokens and wall time, and each client (the X-Ophyx-Client hash the
website sends) gets a small token bucket. Set a monthly budget for the workspace in the Modal
dashboard as well; that is the hard cap.
"""

import os
import time

import modal

HW = os.environ.get("OPHYX_HW", "cpu")
MODEL_REPO = os.environ.get("OPHYX_MODEL", "abhash-chakraborty/ophyx-350m-v2")
GGUF_REPO = os.environ.get("OPHYX_GGUF", "abhash-chakraborty/ophyx-350m-v2-gguf")
GGUF_FILE = os.environ.get("OPHYX_GGUF_FILE", "ophyx-350m-v2-f16.gguf")
MODEL_DIR = "/model"
CPU_CORES = 4
# Modal bills the higher of the request and actual use, so a small request with a 4-core limit costs
# little while the container waits for the next question and up to 4 cores while it answers
CPU_REQUEST = float(os.environ.get("OPHYX_CPU_REQUEST", "0.125"))  # idle llama.cpp uses ~0 CPU
MEMORY_MB = 2048
MEMORY_REQUEST_MB = int(os.environ.get("OPHYX_MEMORY_REQUEST_MB", "1024"))  # f16 model: ~0.95 GiB
SCALEDOWN_S = 90  # one cold start per visit; questions in a visit come 20-90 s apart
MAX_PROMPT_TOKENS = 1536  # the new message
MAX_CONTEXT_TOKENS = 3072  # the new message plus the history kept in front of it
MAX_HISTORY = 8
MAX_NEW_TOKENS = 1024
CTX = 4096
REQUEST_TIMEOUT_S = 90
PER_CLIENT_PER_MIN = 6
END_ID, USER_ID = 0, 1  # <|end|> and <|user|>: an answer stops at either


def download():
    from huggingface_hub import hf_hub_download, snapshot_download
    token = os.environ.get("HF_TOKEN") or None
    if HW == "cpu":
        hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=MODEL_DIR, token=token)
    else:
        snapshot_download(MODEL_REPO, local_dir=MODEL_DIR, token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


env = {"OPHYX_HW": HW, "OPHYX_MODEL": MODEL_REPO, "OPHYX_GGUF": GGUF_REPO, "OPHYX_GGUF_FILE": GGUF_FILE,
       "OPHYX_CPU_REQUEST": str(CPU_REQUEST), "OPHYX_MEMORY_REQUEST_MB": str(MEMORY_REQUEST_MB)}
if HW == "cpu":
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "cmake")
        # portable build: the image is built on one machine and runs on others
        .env({"CMAKE_ARGS": "-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"})
        .pip_install("llama-cpp-python==0.3.36", "fastapi[standard]==0.115.12", "huggingface_hub==0.30.2")
    )
else:
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "fastapi[standard]==0.115.12", "huggingface_hub==0.30.2")
image = image.env(env).run_function(download, secrets=[modal.Secret.from_name("ophyx-hf")])  # the container re-imports this file
image = image.add_local_python_source("replies")  # serve/replies.py; added last, so editing it needs no rebuild
app = modal.App(os.environ.get("OPHYX_APP", "ophyx-demo"), image=image)
resources = (dict(cpu=(CPU_REQUEST, CPU_CORES), memory=(MEMORY_REQUEST_MB, MEMORY_MB)) if HW == "cpu"
             else dict(gpu="T4"))


@app.cls(**resources, scaledown_window=SCALEDOWN_S, max_containers=1, timeout=600)
@modal.concurrent(max_inputs=4)
class Ophyx:
    @modal.enter()
    def load(self):
        import threading
        started = time.time()
        self.lock = threading.Lock()  # one generation at a time; the others wait their turn
        self.buckets = {}
        if HW == "cpu":
            from llama_cpp import Llama
            self.llm = Llama(model_path=f"{MODEL_DIR}/{GGUF_FILE}", n_ctx=CTX, n_threads=CPU_CORES,
                             n_batch=512, verbose=False)
            self.encode = lambda text: self.llm.tokenize(text.encode(), add_bos=False, special=True)
        else:
            import torch
            from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
            self.torch = torch
            # the repo's tokenizer_config names a transformers 5 class; the tokenizer file itself loads anywhere
            self.tok = PreTrainedTokenizerFast(tokenizer_file=f"{MODEL_DIR}/tokenizer.json")
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).to("cuda").eval()
            self.encode = lambda text: self.tok(text, add_special_tokens=False)["input_ids"]
        self.ready_at, self.boot_s, self.answered = time.time(), round(time.time() - started, 2), 0

    def allowed(self, client):
        now = time.monotonic()
        if len(self.buckets) > 10_000:  # a bucket refills in a minute; older ones carry no state
            self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < 60}
        tokens, last = self.buckets.get(client, (PER_CLIENT_PER_MIN, now))
        tokens = min(PER_CLIENT_PER_MIN, tokens + (now - last) * PER_CLIENT_PER_MIN / 60)
        ok = tokens >= 1
        self.buckets[client] = (tokens - 1 if ok else tokens, now)
        return ok

    def generate(self, ids, max_new, temperature, stop, out, used):
        """Runs in a thread: puts text chunks on `out`, then None. Stops early once `stop` is set.
        `used` gets the number of new tokens, for the API's usage and finish_reason, and how long
        the answer waited for the one generation slot."""
        try:
            with self.lock:
                used["queued_s"] = round(time.time() - used["t0"], 2)
                if stop.is_set():
                    return
                if HW == "cpu":
                    from llama_cpp import StoppingCriteriaList

                    def halt(tokens, logits):
                        return stop.is_set() or tokens[-1] in (END_ID, USER_ID)
                    for part in self.llm.create_completion(
                            ids, max_tokens=max_new, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                            stream=True, stopping_criteria=StoppingCriteriaList([halt])):
                        choice = part["choices"][0]
                        if choice.get("finish_reason") == "length":
                            used["length"] = True
                        text = choice["text"]
                        if text or not choice.get("finish_reason"):  # one streamed part per token
                            used["tokens"] = used.get("tokens", 0) + 1
                        if text:
                            out.put(text)
                else:
                    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

                    class Halt(StoppingCriteria):
                        def __call__(self, input_ids, scores, **kw):
                            return stop.is_set()

                    streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
                    kw = dict(input_ids=self.torch.tensor([ids], device="cuda"), max_new_tokens=max_new,
                              streamer=streamer, stopping_criteria=StoppingCriteriaList([Halt()]),
                              eos_token_id=[END_ID, USER_ID], pad_token_id=END_ID, do_sample=temperature > 0,
                              temperature=temperature or None, top_p=0.95 if temperature > 0 else None)
                    import threading

                    def work():
                        with self.torch.no_grad():  # thread-local, so it goes inside the worker
                            self.model.generate(**kw)

                    worker = threading.Thread(target=work, daemon=True)
                    worker.start()
                    for text in streamer:
                        if text:
                            used["tokens"] = used.get("tokens", 0) + len(self.encode(text))
                            out.put(text)
                    worker.join()
        finally:
            out.put(None)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        import asyncio
        import json
        import queue
        import re
        from os.path import commonprefix
        import threading

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
        from replies import SYSTEM_PROMPT, ConciseFilter, RepeatGuard, route, wants_extras

        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_REPO, "hw": HW}

        @api.get("/system")
        def system():
            return {"system": SYSTEM_PROMPT}

        @api.post("/generate")
        async def generate(req: Request):
            # a second line of defence behind the website's own limits: one bucket per visitor hash
            client = req.headers.get("x-ophyx-client", "")[:64] or "website"
            if not self.allowed(client):
                raise HTTPException(429, "Too many requests. Try again in a minute.")
            try:
                body = await req.json()
                assert isinstance(body, dict)
                prompt = str(body.get("prompt", "")).strip()
                max_new = max(1, min(int(body.get("max_tokens", 512)), MAX_NEW_TOKENS))
                temperature = max(0.0, min(float(body.get("temperature", 0.2)), 1.5))
                concise = body.get("concise") is True
                live = body.get("live") is True
                history = body.get("history") or []
                assert isinstance(history, list)
                turns = [(str(t.get("prompt", "")).strip(), str(t.get("answer", "")).strip())
                         for t in history[-MAX_HISTORY:] if isinstance(t, dict)]
                turns = [(p, a) for p, a in turns if p and a]
            except (ValueError, TypeError, AssertionError):
                raise HTTPException(400, "send a JSON object: prompt, and optionally max_tokens, temperature, concise and history")
            if not prompt:
                raise HTTPException(400, "prompt is empty")
            fixed = route(prompt)
            if fixed is not None:
                # the rules in SYSTEM_PROMPT: a fixed answer, streamed line by line like the model's.
                # "about" tells the website to show it as text, not code
                kind, reply = fixed

                async def canned():
                    yield f"data: {json.dumps({'route': 'about', 'reply': kind})}\n\n"
                    for word in re.findall(r"\S+\s*|\s+", reply):
                        yield f"data: {json.dumps({'text': word})}\n\n"
                        await asyncio.sleep(0.015)
                    done = {"done": True, "seconds": 0, "chunks": len(reply.splitlines()),
                            "tokens": len(self.encode(reply)), "route": "about", "reply": kind, "hw": HW}
                    yield f"data: {json.dumps(done)}\n\n"

                return StreamingResponse(canned(), media_type="text/event-stream")
            ids = self.encode(f"<|user|>\n{prompt}\n<|python|>\n")
            if len(ids) > MAX_PROMPT_TOKENS:
                raise HTTPException(413, f"prompt is too long ({len(ids)} tokens, max {MAX_PROMPT_TOKENS})")
            kept = 0
            for p, a in reversed(turns):
                # strings, not ids: the tokenizer must see the special tokens it was trained on
                prev = self.encode(f"<|user|>\n{p}\n<|python|>\n{a}\n<|end|>\n")
                if len(prev) + len(ids) > MAX_CONTEXT_TOKENS:
                    break
                ids, kept = prev + ids, kept + 1
            stop = threading.Event()  # set when the client leaves or time runs out, so generation stops too
            out = queue.Queue()
            t0 = time.time()
            # the first answer of a fresh container that arrives as start-up finishes waited for it
            cold = self.answered == 0 and t0 - self.ready_at < 5
            self.answered += 1
            budget = min(max_new, CTX - len(ids))
            used = {"t0": t0}
            # plain code unless the prompt asks for comments, docstrings, examples or tests
            trim = ConciseFilter() if concise and not wants_extras(prompt) else None
            threading.Thread(target=self.generate, args=(ids, budget, temperature, stop, out, used),
                             daemon=True).start()

            async def events():
                n, first = 0, None
                guard = RepeatGuard()
                settled = ""  # text past both checks; in live mode `shown` is what the client has
                shown = ""

                def event(text: str) -> str:
                    nonlocal n
                    n += 1
                    return f"data: {json.dumps({'text': text})}\n\n"

                def sync(target: str) -> str | None:
                    """The event that turns what the client shows into `target`, if they differ."""
                    nonlocal n, shown
                    keep = len(commonprefix([shown, target]))
                    undo, add = len(shown) - keep, target[keep:]
                    if not undo and not add:
                        return None
                    shown = target
                    n += 1
                    return f"data: {json.dumps({'undo': undo, 'text': add} if undo else {'text': add})}\n\n"

                try:
                    while True:
                        try:
                            chunk = await asyncio.to_thread(out.get, True, 1.0)
                        except queue.Empty:
                            if time.time() - t0 > REQUEST_TIMEOUT_S:
                                break
                            yield ": waiting\n\n"  # keeps the connection open while queued behind another answer
                            continue
                        if chunk is None:
                            break
                        first = first or time.time() - t0
                        text = guard.feed(chunk)
                        if trim:
                            text = trim.feed(text)
                        settled += text
                        if live:
                            # settled text plus a best guess at what the held lines and the line being
                            # written will add; a wrong guess is taken back with "undo"
                            pending = "" if guard.looping else "".join(guard.held) + guard.partial
                            ahead = trim.preview(pending) if trim and not trim.done else ("" if trim else pending)
                            update = sync(settled + ahead)
                            if update:
                                yield update
                        elif text:
                            yield event(text)
                        if guard.looping or (trim and trim.done):
                            break  # the finally below stops generation
                        if time.time() - t0 > REQUEST_TIMEOUT_S:
                            break
                    rest = guard.flush()
                    if trim:
                        rest = trim.feed(rest) + trim.flush()
                    settled += rest
                    if live:
                        update = sync(settled)
                        if update:
                            yield update
                    elif rest:
                        yield event(rest)
                    done = {"done": True, "seconds": round(time.time() - t0, 2), "chunks": n,
                            "first_s": round(first or 0, 2), "looped": guard.looping, "hw": HW,
                            "tokens": used.get("tokens", 0), "prompt_tokens": len(ids),
                            "length": used.get("length", used.get("tokens", 0) >= budget),
                            "concise": trim is not None, "trimmed": bool(trim and trim.done),
                            "cold": cold, "boot_s": self.boot_s if cold else 0,
                            "queued_s": used.get("queued_s", 0), "history": kept}
                    yield f"data: {json.dumps(done)}\n\n"
                finally:
                    stop.set()

            return StreamingResponse(events(), media_type="text/event-stream")

        return api

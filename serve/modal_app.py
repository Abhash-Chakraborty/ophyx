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
  POST /generate                   {"prompt": "...", "max_tokens": 512, "temperature": 0.2}
                                   -> text/event-stream of {"text": "..."} chunks, then {"done": true, ...}
                                   Questions about the model itself ("who made you?") get a fixed answer
                                   from replies.py, marked by a first {"route": "about"} event; answers
                                   that start repeating themselves end early with "looped": true.

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
SCALEDOWN_S = 180
MAX_PROMPT_TOKENS = 1536
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


env = {"OPHYX_HW": HW, "OPHYX_MODEL": MODEL_REPO, "OPHYX_GGUF": GGUF_REPO, "OPHYX_GGUF_FILE": GGUF_FILE}
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
resources = dict(cpu=CPU_CORES, memory=2048) if HW == "cpu" else dict(gpu="T4")


@app.cls(**resources, scaledown_window=SCALEDOWN_S, max_containers=1, timeout=600)
@modal.concurrent(max_inputs=4)
class Ophyx:
    @modal.enter()
    def load(self):
        import threading
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

    def allowed(self, client):
        now = time.monotonic()
        if len(self.buckets) > 10_000:  # a bucket refills in a minute; older ones carry no state
            self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < 60}
        tokens, last = self.buckets.get(client, (PER_CLIENT_PER_MIN, now))
        tokens = min(PER_CLIENT_PER_MIN, tokens + (now - last) * PER_CLIENT_PER_MIN / 60)
        ok = tokens >= 1
        self.buckets[client] = (tokens - 1 if ok else tokens, now)
        return ok

    def generate(self, ids, max_new, temperature, stop, out):
        """Runs in a thread: puts text chunks on `out`, then None. Stops early once `stop` is set."""
        try:
            with self.lock:
                if stop.is_set():
                    return
                if HW == "cpu":
                    from llama_cpp import StoppingCriteriaList

                    def halt(tokens, logits):
                        return stop.is_set() or tokens[-1] in (END_ID, USER_ID)
                    for part in self.llm.create_completion(
                            ids, max_tokens=max_new, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                            stream=True, stopping_criteria=StoppingCriteriaList([halt])):
                        text = part["choices"][0]["text"]
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
                            out.put(text)
                    worker.join()
        finally:
            out.put(None)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        import asyncio
        import json
        import queue
        import threading

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
        from replies import RepeatGuard, canned_reply

        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_REPO, "hw": HW}

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
            except (ValueError, TypeError, AssertionError):
                raise HTTPException(400, "send a JSON object: prompt, and optionally max_tokens and temperature")
            if not prompt:
                raise HTTPException(400, "prompt is empty")
            reply = canned_reply(prompt)
            if reply is not None:
                # questions about the model itself: a fixed answer, streamed line by line like the model's
                async def canned():
                    yield f"data: {json.dumps({'route': 'about'})}\n\n"
                    for line in reply.splitlines(keepends=True):
                        yield f"data: {json.dumps({'text': line})}\n\n"
                        await asyncio.sleep(0.03)
                    done = {"done": True, "seconds": 0, "chunks": len(reply.splitlines()), "route": "about", "hw": HW}
                    yield f"data: {json.dumps(done)}\n\n"

                return StreamingResponse(canned(), media_type="text/event-stream")
            ids = self.encode(f"<|user|>\n{prompt}\n<|python|>\n")
            if len(ids) > MAX_PROMPT_TOKENS:
                raise HTTPException(413, f"prompt is too long ({len(ids)} tokens, max {MAX_PROMPT_TOKENS})")
            stop = threading.Event()  # set when the client leaves or time runs out, so generation stops too
            out = queue.Queue()
            t0 = time.time()
            threading.Thread(target=self.generate, args=(ids, min(max_new, CTX - len(ids)), temperature, stop, out),
                             daemon=True).start()

            async def events():
                n, first = 0, None
                guard = RepeatGuard()
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
                        if text:
                            n += 1
                            yield f"data: {json.dumps({'text': text})}\n\n"
                        if guard.looping:
                            break  # the finally below stops generation
                        if time.time() - t0 > REQUEST_TIMEOUT_S:
                            break
                    rest = guard.flush()
                    if rest:
                        n += 1
                        yield f"data: {json.dumps({'text': rest})}\n\n"
                    done = {"done": True, "seconds": round(time.time() - t0, 2), "chunks": n,
                            "first_s": round(first or 0, 2), "looped": guard.looping, "hw": HW}
                    yield f"data: {json.dumps(done)}\n\n"
                finally:
                    stop.set()

            return StreamingResponse(events(), media_type="text/event-stream")

        return api

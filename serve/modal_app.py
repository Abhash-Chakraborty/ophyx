"""ophyx demo endpoint on Modal: fp32 weights on a T4, scales to zero when nobody is using it.

    modal deploy serve/modal_app.py          # needs `modal token new` once, and a Modal secret "ophyx-hf"
                                             # with HF_TOKEN while the model repo is private

Endpoints (FastAPI):
  GET  /health                     {"ok": true, "model": ...}
  POST /generate                   {"prompt": "...", "max_tokens": 512, "temperature": 0.2}
                                   -> text/event-stream of {"text": "..."} chunks, then {"done": true, ...}

Cost controls: one container at most, it shuts down after SCALEDOWN_S idle seconds, each request is
capped in prompt length, new tokens and wall time, and each client IP gets a small token bucket.
Set a monthly budget for the workspace in the Modal dashboard as well; that is the hard cap.
"""

import os
import time

import modal

MODEL_REPO = os.environ.get("OPHYX_MODEL", "abhash-chakraborty/ophyx-350m-v2")
MODEL_DIR = "/model"
GPU = os.environ.get("OPHYX_GPU", "T4")
SCALEDOWN_S = 180
MAX_PROMPT_TOKENS = 1536
MAX_NEW_TOKENS = 1024
REQUEST_TIMEOUT_S = 90
PER_IP_PER_MIN = 6
ALLOWED_ORIGINS = [o for o in os.environ.get("OPHYX_ORIGINS", "*").split(",") if o]


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR, token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "fastapi[standard]==0.115.12", "huggingface_hub==0.30.2")
    .env({"OPHYX_MODEL": MODEL_REPO, "OPHYX_ORIGINS": ",".join(ALLOWED_ORIGINS)})  # the container re-imports this file
    .run_function(download, secrets=[modal.Secret.from_name("ophyx-hf")])
)
app = modal.App("ophyx-demo", image=image)


@app.cls(gpu=GPU, scaledown_window=SCALEDOWN_S, max_containers=1, timeout=600)
@modal.concurrent(max_inputs=4)
class Ophyx:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).to("cuda").eval()
        self.end_id = self.tok.convert_tokens_to_ids("<|end|>")
        self.ctx = self.model.config.max_position_embeddings
        self.stop_ids = [self.end_id, self.tok.convert_tokens_to_ids("<|user|>")]
        self.buckets = {}

    def allowed(self, ip):
        now = time.monotonic()
        if len(self.buckets) > 10_000:  # a bucket refills in a minute; older ones carry no state
            self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < 60}
        tokens, last = self.buckets.get(ip, (PER_IP_PER_MIN, now))
        tokens = min(PER_IP_PER_MIN, tokens + (now - last) * PER_IP_PER_MIN / 60)
        ok = tokens >= 1
        self.buckets[ip] = (tokens - 1 if ok else tokens, now)
        return ok

    @modal.asgi_app()
    def web(self):
        import json
        import threading

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class Stop(StoppingCriteria):
            def __init__(self, event):
                self.event = event

            def __call__(self, input_ids, scores, **kw):
                return self.event.is_set()

        api = FastAPI()
        api.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                           allow_headers=["content-type"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_REPO}

        @api.post("/generate")
        async def generate(req: Request):
            # the proxy appends the address it saw; earlier entries come from the client and can be anything
            ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "?").split(",")[-1].strip()
            if not self.allowed(ip):
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
            ids = self.tok(f"<|user|>\n{prompt}\n<|python|>\n", return_tensors="pt")["input_ids"]
            if ids.shape[1] > MAX_PROMPT_TOKENS:
                raise HTTPException(413, f"prompt is too long ({ids.shape[1]} tokens, max {MAX_PROMPT_TOKENS})")
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=REQUEST_TIMEOUT_S)
            stop = threading.Event()  # set when the client leaves or time runs out, so the GPU stops too
            kw = dict(input_ids=ids.to("cuda"), max_new_tokens=min(max_new, self.ctx - ids.shape[1]), streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([Stop(stop)]),
                      eos_token_id=self.stop_ids, pad_token_id=self.end_id,
                      do_sample=temperature > 0, temperature=temperature or None, top_p=0.95 if temperature > 0 else None)

            def run():
                with self.torch.no_grad():
                    self.model.generate(**kw)

            threading.Thread(target=run, daemon=True).start()
            t0 = time.time()

            def events():
                n = 0
                try:
                    for chunk in streamer:
                        n += 1
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                        if time.time() - t0 > REQUEST_TIMEOUT_S:
                            break
                    yield f"data: {json.dumps({'done': True, 'seconds': round(time.time() - t0, 2), 'chunks': n})}\n\n"
                finally:
                    stop.set()

            return StreamingResponse(events(), media_type="text/event-stream")

        return api

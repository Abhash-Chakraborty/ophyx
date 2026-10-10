# Python-only code model: plan

A small decoder-only transformer, trained from scratch on free Kaggle GPUs, that reads English
and writes only Python. Explanations go into comments and docstrings, never into prose.

Working name: TBD. The data plan lives in [DATA.md](DATA.md).

## What specialising buys, and what it doesn't

General models spread their capacity across dozens of languages, natural-language chat, maths and
images. A model that only has to model English-for-understanding plus Python spends every
parameter on that one distribution. That's the core bet, and history backs it: Codex-12B beat
GPT-3-175B on Python, and phi-1 (1.3B) beat many 10× larger general models on HumanEval.

What it buys:

- A smaller model reaches a given Python quality, so the model is cheap to run (a 350M model is
  ~700 MB in fp16 and ~380 MB at 8-bit; it runs on a laptop CPU).
- The domain is narrow enough to cover deliberately. We can list the concepts that matter and
  check that the data covers each one (see the coverage matrix in DATA.md).
- High-quality data can be repeated for several epochs without much loss, so a few billion
  excellent tokens go a long way.
- A Python-specific tokenizer packs more code into each token, which makes training and
  inference cheaper per file.

What it doesn't buy:

- Training cost per token is the same. Compute is roughly 6 × parameters × tokens, whatever the
  domain. The free GPUs, not the data, set the ceiling.
- Python still takes billions of tokens to learn well. Narrow means fewer kinds of data, not few
  tokens.
- Large-model reasoning (long multi-step plans, big system designs) comes partly from scale. A
  350M model will write a solid LRU cache or rate limiter; it won't design a distributed system
  end to end from a vague prompt.
- English understanding will be limited to programming requests. Prompts need to be direct.

Goal for the release:

> The best Python model per parameter trained from scratch on free GPUs, with open data, open
> code and honest benchmarks.

Bets to measure against, not promises: ~350M model at 30–40% HumanEval+ pass@1 (stretch 45%),
~125M model at 15–25%. phi-1-small (350M) reached about 45% on plain HumanEval through curated
and synthetic data, so the ceiling at this size is set mostly by data quality.

## Compute: free sources only

The whole project runs on free offers. No rented GPUs, no paid APIs. Nothing heavy runs on your
laptop or your VPS: both can stay as they are, and the laptop can be switched off while training
runs.

### Tested on your account (9 October 2026)

Short probe notebooks were pushed through the Kaggle CLI for every accelerator the docs list.

| Finding | What it means |
|---|---|
| Only 2× T4 is free. Requests for L4, L4×1, A100 and H100 all ran on 2× T4 | The GPU plan stays on T4: fp16, no FlashAttention 2 |
| At most 2 GPU background sessions at once | One training chain plus one generation or eval job in parallel |
| At most 1 TPU session at once, and it can run alongside the 2 GPU sessions | TPU pretraining and T4 work really do run in parallel |
| TPU v5e-8 works on this account: JAX 0.11 sees all 8 chips (`v5litepod-8`). The host has 224 CPU cores, 377 GB RAM, 21 GB `/kaggle/working`, internet on | Plan A is viable. Speed measured on 9 October, see the throughput table below |
| The TPU probe sat in the queue ~9 hours (20:33 → ~05:40 UTC) before running for about a minute | Queue waits cost calendar time, not quota. The chain must tolerate gaps of many hours, and TPU sessions should be long to make each wait worth it |
| TPU speed test (our model, plain JAX, 2048 context, bf16): 22M at 1.28M tok/s, 126M at 260k tok/s, 349M at 108k tok/s, 17–19% of v5e peak. The 349M model doesn't fit at batch 8 per chip unless activations are fully recomputed | 350M on 10B tokens ≈ 26 TPU hours. About 9× the 2× T4 estimate. Queue wait was ~9 h again |
| Weekly quota (Kaggle profile, 10 October): GPU 30 h, TPU 20 h. Plus an "AI" quota of $10/day, $100/month that applies only to Kaggle Benchmarks | 20 TPU h/week ≈ 7B tokens/week for the 350M model. The AI quota is not usable for data generation |
| TPU image differs from the GPU one: Python 3.12, JAX 0.11, Flax 0.12.8, Optax 0.2.8, torch_xla 2.8 (torch CPU build), Transformers 5.12 | Pin TPU dependencies separately from the T4 ones |
| At most 5 CPU sessions at once, 4 cores and 31 GB RAM each, internet on | 20 cores and ~155 GB RAM for data work, without touching GPU or TPU quota |
| `/kaggle/working` is 20 GB and is kept after the run; scratch disk took a 30 GB test write easily | Process data in shards inside the session; keep only results |
| Image: Python 3.13, PyTorch 2.11 (CUDA 12.8), JAX 0.11.1, Flax 0.12, Optax 0.2.8, Transformers 5.16, Datasets 4.8. No vLLM, TRL or MaxText preinstalled | Pin and cache those as wheels in a Kaggle dataset |

### Where everything runs

| Source | Role in this plan |
|---|---|
| Kaggle GPU (2× T4), ~30 h/week | Development, nano model, data generation, post-training, evals |
| Kaggle TPU (v5e-8), separate weekly quota | Base model pretraining (plan A) |
| Kaggle CPU sessions, up to 5 at once | All data collection and processing, running tests, the relay notebooks that keep chains going |
| Hugging Face Hub (free account) | Storage for processed data and checkpoint backups, private until release |
| GitHub, public repo + Actions | The code, and a scheduled watchdog that restarts a stalled chain. Free for public repos |
| Free API tiers for open-weight models (e.g. OpenRouter free models, Groq, Cerebras) | Quality annotation and the hardest synthetic samples from a stronger teacher |
| Monthly free credits (e.g. Modal, Lightning AI) | Extra generation and GRPO rollouts |
| Google TPU Research Cloud | Apply in week 1. A bonus if accepted; nothing depends on it |
| W&B (free) + ntfy.sh notifications | Watching runs from your phone |
| Your laptop | Writing code and `git push`. A few MB of repo, nothing else |
| Your VPS | Not used |

Rules for free tiers: only open-weight models whose licence allows training on their outputs, and
only providers whose terms allow it. Closed-model APIs (Gemini, GPT, Claude) generally forbid using
outputs to train other models, so they're out. One account per person on every platform.

### Keys in `.env` and how each is used

| Key | Used for | Limits we keep to |
|---|---|---|
| `KAGGLE_API_TOKEN` | Pushing and watching all runs | Lives only in `.env` and an encrypted GitHub secret |
| `HF_TOKEN` (write) | Private dataset and checkpoint repos | Given to training sessions only, never to sessions that run untrusted code |
| `GITHUB_TOKEN` | GitHub search and metadata for data collection | Must be a no-scope or public-read-only token before it goes into any Kaggle session |
| `GROQ_TOKEN`, `CEREBRAS_TOKEN` | Synthetic data and quality grading with open-weight models (gpt-oss-120b, Qwen3.8-27B; licences checked on each model card) | Free-tier rate limits only; a daily token cap in our client |
| `WANDB_API_KEY` | Training curves | — |
| `NTFY_TOPIC` | Phone notifications | Messages contain only run status, never data or secrets |
| `OPENAI_KEY` | Not used for training data: OpenAI's terms forbid using outputs to build competing models, and the data-sharing programme means prompts are shared with OpenAI | Only if a specific need comes up, with your OK first |
| `ELEVENLABS_KEY`, `NARAROUTER_KEY` | Not needed for this project | Unused |

### Plan A and plan B

- Plan A (chosen): pretraining in our own JAX trainer on Kaggle TPU, while the T4s do everything
  else in parallel. Decided 10 October: we build on the speed-test code (`kaggle/tpu_speedtest/`)
  instead of porting to MaxText. It already runs the exact model at 19% MFU, it's small enough to
  read in one sitting, and it avoids MaxText's assumption of Google Cloud Storage. Still to add:
  Orbax checkpoint/resume, a streaming data loader, splash attention, chunked loss, export to
  Hugging Face format.
- Plan B (fallback): everything on the T4s. Same model and recipe, about 5 more weeks of calendar.

The port is worth trying because the TPU quota is separate from the GPU quota. Even at modest
utilisation, an hour on Kaggle's TPU should be worth several T4-pair hours. Measure it before
counting on it.

### Kaggle GPU details

Kaggle "GPU T4 x2":

| Resource | Value | What it means for us |
|---|---|---|
| GPUs | 2× T4, 16 GB each | DDP across 2 GPUs |
| Compute | ~65 TFLOPS fp16 peak per T4 | Expect 25–30% utilisation, so ~35 TFLOPS combined |
| Precision | fp16 only, no bf16 | Needs loss scaling and stability tricks |
| Attention kernels | Turing (sm_75) | FlashAttention 2 doesn't run. Use PyTorch SDPA (memory-efficient backend) |
| Quota | ~30 GPU hours per week | The real bottleneck |
| Session | ~12 hours max | Checkpoint often and resume across sessions |
| Persistent output | ~20 GB in `/kaggle/working` | Keep only the latest 2 checkpoints |
| CPU / RAM | 4 cores, 31 GB | Keep data processing in separate CPU-only sessions |

## How many parameters

### Memory is not the limit

With DDP each GPU holds the full model and optimizer state:

| Setup | Bytes per parameter | Largest model on a 16 GB T4 (with activation checkpointing) |
|---|---|---|
| AdamW, fp32 master weights | ~16 | ~600M |
| Muon (one momentum buffer) + AdamW for embeddings | ~12 | ~800M |
| Optimizer state sharded across both GPUs (ZeRO-1) | ~10 | ~1B |

So up to ~1B fits. But a 1B model on this budget would see only 2–3B tokens in ten weeks, which
is badly undertrained. Time decides the size.

### Compute decides the size

Each pretraining GPU hour (2× T4) gives about 1.1×10¹⁷ useful FLOPs after overhead for evals and
checkpoints. Tokens per parameter tells you how trained a model is: ~20 is "compute-optimal"
(Chinchilla), and modern small models go far beyond that because it makes them better at a fixed
size.

Throughput at 2048 context. T4 numbers are FLOP-count estimates; TPU numbers were measured on
Kaggle TPU v5e-8 on 9 October 2026 (`kaggle/tpu_speedtest/`):

| Model | Tokens/s (2× T4, est.) | Tokens/s (TPU v5e-8, measured) | Hours for 10B tokens on TPU |
|---|---|---|---|
| ~22M (debug) | — | 1.28M (15% MFU) | 2.2 |
| ~125M | ~33k | 260k (17.5% MFU) | 10.7 |
| ~350M | ~12k | 108k (19% MFU) | 25.8 |
| ~500M | ~8k | ~75k (extrapolated) | ~37 |

The TPU run used plain XLA attention and full-vocab fp32 logits. Batch size and recompute
settings barely changed speed, so the remaining time is in memory-bound work (attention scores,
the 32k-vocab softmax), not the matmuls. Splash (flash) attention and a chunked loss are the
obvious next steps and could plausibly reach 25–30% MFU, which would take the 350M run on 10B tokens
to ~16–20 h.

Options by pretraining budget:

| Pretraining budget | Weeks of quota | 125M | 350M | 500M | Compute-optimal size |
|---|---|---|---|---|---|
| 100 GPU h | ~3.5 | 12B tokens (96/param) | 4.5B (13/param) | 3B (6/param) | ~300M |
| 200 GPU h | ~7 | 24B (190/param) | 9B (26/param) | 6B (12/param) | ~430M |
| 300 GPU h | ~10 | 36B (290/param) | 13.5B (39/param) | 9B (18/param) | ~520M |

### Recommendation

- Flagship: ~350M parameters on ~9–10B tokens, about 200 GPU hours of pretraining. That's close
  to the lowest loss this budget can buy, while staying small enough to run anywhere. 26 tokens
  per parameter is slightly past compute-optimal, which favours inference efficiency.
- Dress rehearsal: ~125M on ~5B tokens (~40 GPU hours). It tests the whole pipeline end to end,
  runs the ablations, and is worth releasing as a "nano" model.
- Not 1B+: memory allows it, compute doesn't.
- If the TPU port gives more compute than expected, spend it on more tokens for the 350M model
  (up to ~20B) before making it bigger. That improves quality without raising inference cost.

Measured result (9 October): on TPU, ~350M on 10B tokens costs ~26 TPU hours, about the same
as one to one and a half weeks of TPU quota. Compute is no longer the main limit for the base model;
data is. The recommendation stays at ~350M. If quota allows, train it on 15–20B tokens (~40–50 TPU
hours) rather than going bigger, provided DATA.md delivers enough tier A/B tokens to keep
repetition at or below ~4 epochs.

## How "Python only" works

Three layers, from strongest to weakest:

1. Data. Pretraining is mostly Python. English appears in docstrings, comments, docs and a small
   slice of programming-focused web text so the model understands requests.
2. Format. Instruction tuning uses a fixed template where the response is always a complete
   Python module. Any explanation is rewritten into comments or docstrings.

   ```
   <|user|>
   Write an LRU cache with O(1) get and put, and explain the design.
   <|python|>
   """LRU cache backed by a dict and a doubly linked list. ..."""
   ...
   <|end|>
   ```

3. Inference guard. The serving wrapper checks every output with `ast.parse` (and optionally
   `ruff`). If it fails, resample at a lower temperature. Report the "parses on first try" rate as
   a metric.

Grammar-constrained decoding for full Python is hard because indentation isn't context-free.
Parse-and-resample is simpler and good enough for v1.

## Architecture

Llama/Qwen3-style decoder, so the weights load in Hugging Face `transformers`, `llama.cpp` and
vLLM without custom code. That matters for open-sourcing.

| Choice | Setting | Why |
|---|---|---|
| Block | Pre-norm, RMSNorm, no biases | Standard, stable |
| MLP | SwiGLU, hidden ≈ 8/3·d rounded to 64 | Better quality per FLOP than GELU |
| Positions | RoPE, base 100k | Easier context extension later |
| Attention | GQA (4 KV heads), QK-norm | GQA cuts KV cache; QK-norm prevents fp16 attention-logit blowups |
| Embeddings | Tied input/output | Embeddings are 10–20% of params at this size |
| Context | 2048 to start, 4096 in the final phase | Most files fit in 2k; long context costs throughput |
| Shape | Deep and thin | MobileLLM showed depth beats width below ~1B |
| Loss extras | z-loss 1e-4 on logits | Keeps logits bounded under fp16 |

Model configs:

| Name | d_model | Layers | Heads / KV | FFN | Vocab | Params |
|---|---|---|---|---|---|---|
| debug | 384 | 6 | 6 / 2 | 1024 | 32,768 | ~20M |
| nano | 768 | 16 | 12 / 4 | 2048 | 32,768 | ~125M |
| base | 1024 | 28 | 16 / 4 | 2816 | 32,768 | ~350M |

Things to skip in v1: mixture of experts (memory-bound on T4, little gain at this size), multi-token
prediction (try later as an ablation), custom kernels.

## Training recipe

- Optimizer: Muon for 2D weight matrices, AdamW for embeddings, norms and the output head. Muon has
  shown ~1.3–2× better compute efficiency in nanoGPT speedruns and Moonlight. Run the
  Newton-Schulz step in fp32 since T4 has no bf16. Keep a pure-AdamW baseline on the debug model
  to confirm the gain on our setup.
- Schedule: warmup-stable-decay (WSD). Constant LR for most of training, then a short decay at the
  end. This suits Kaggle well: any stable-phase checkpoint can be extended, and the decay phase
  doubles as the annealing phase on the best data.
- Precision: `torch.autocast(fp16)` with `GradScaler`. RMSNorm, softmax and loss in fp32. Gradient
  clipping at 1.0.
- Batch: ~0.5M tokens per optimizer step through gradient accumulation (micro-batch 8–16 × 2048
  per GPU). Sequence packing with document-boundary masking so files don't attend to each other.
- Fill-in-the-middle: 50% of code documents transformed to prefix-suffix-middle format, like
  StarCoder. This gives editor-style completion for free and doesn't hurt left-to-right quality.
- `torch.compile` if it works on sm_75 in the Kaggle image. Fall back to eager if it doesn't.
- Data loading: pre-tokenised `uint16` shards, memory-mapped. No tokenisation during training.

Session limits: every long job runs as a chain of Kaggle sessions, started and resumed by a
driver script through the Kaggle CLI. Checkpoints every 20–30 minutes, a clean stop before the
time limit, and a Hugging Face backup. Details in TRAINING.md, "Working within Kaggle's session
limits".

## Tokenizer

- Byte-level BPE, 32,768 vocab (a multiple of 64 for tensor cores), trained with HF `tokenizers`.
- Train it on the actual pretraining mix, mostly Python with some English.
- Dedicated tokens for runs of spaces (4, 8, 12, 16) and common indentation-plus-keyword patterns,
  so indentation doesn't eat the context window.
- Special tokens: `<|user|>`, `<|python|>`, `<|end|>`, FIM tokens (`<|fim_prefix|>`,
  `<|fim_suffix|>`, `<|fim_middle|>`), `<|file_sep|>`, `<|repo|>`, padding.
- Check: average characters per token on held-out Python should beat GPT-2's and Qwen's
  tokenizers clearly. Every 10% gained here is 10% more code per GPU hour.
- Measured 10 October (v1, trained on 0.8 GB of the deduplicated mix): 3.52 chars/token on held-out
  code and 4.38 on English, against GPT-2 at 2.10 / 4.50 and Qwen2.5-Coder at 3.97 / 4.72. We beat
  GPT-2 on code by 67%. Qwen compresses code 13% better but needs a 151k vocabulary, which would be
  ~155M embedding parameters in a 350M model; ours costs 34M. Kept at 32k as the better trade at
  this size. Runs of 3/4/7/8/11/12/15/16 spaces were all learned as single tokens.

## Data (summary)

Full plan in [DATA.md](DATA.md). The short version:

- ~10B training tokens for the flagship, drawn from ~6.7B unique tokens in three quality tiers.
  The best tier is repeated up to ~4 epochs.
- Sources: edu-filtered Python, LLM-rewritten code (SwallowCode), curated GitHub repositories and
  top PyPI packages (repo-level, with their tests, five quality gates), verified synthetic
  textbooks and exercises, Python docs and PEPs, a small programming-focused English slice.
- Cleanup keeps behaviour unchanged: licence headers shortened, dead commented code removed,
  everything formatted with `ruff format`.
- Our own quality classifier, trained on LLM-graded samples with an engineering-quality rubric.
- A concept coverage matrix that tracks how many tokens each curriculum topic gets, with gaps
  filled by synthetic data.
- Everything decontaminated against the eval sets before tokenisation.

## Post-training

Full regime in [TRAINING.md](TRAINING.md). After pretraining and annealing:

- Two-stage SFT (broad, then curated), with self-repair and test-writing examples.
- Search with recursive backtracking: the model plans a module as stubs, implements and tests
  each function, repairs from tracebacks, splits hard functions further, and abandons dead-end
  plans. Verified results are trained back into the model over 2–3 expert-iteration rounds,
  with DPO on passing vs failing pairs.
- GRPO with an execution reward (pass hidden tests +1, fail 0, not Python -1), DAPO-style fixes
  and a 10–90% pass-rate curriculum.
- Checkpoint averaging, then export.

## Evaluation

Run at every checkpoint (small set) and at every phase end (full set):

| Eval | What it measures |
|---|---|
| Held-out Python loss | Raw modelling quality, cheap, every few hundred steps |
| HumanEval+ / MBPP+ (EvalPlus) | Function-level correctness with stronger tests |
| BigCodeBench | Realistic tasks using many libraries, closest to "build anything" |
| LiveCodeBench | Contamination-resistant, recent problems |
| CRUXEval | Understanding what code does (input/output prediction) |
| Own concept suite | ~300 tasks over the curriculum, each with pytest tests, kept private until release |
| Python-only rate | % of outputs that parse on first try, target > 98% |

Baselines: run the same harness on open models of similar size (SmolLM2-135M/360M,
Qwen2.5-Coder-0.5B or its successor) so the numbers have context.

Optional track B: QLoRA fine-tune an existing ~0.5–1.5B open code model on the same SFT data. It
will probably beat the from-scratch model. That's fine: it shows how much of the gap is
pretraining compute versus data, and it's useful to release alongside.

## Compute budget

| Item | Where | Hours |
|---|---|---|
| Setup, throughput tests, debug runs | T4 (+ TPU for the port) | ~10 (+ ~10 TPU) |
| Quality annotation and corpus scoring | T4 + free API tiers | ~10–20 |
| Synthetic data generation | T4 + free API tiers + monthly credits | ~40–60 |
| Ablations on debug model (mix, optimizer, loss, FIM) | T4 | ~20 |
| Nano (125M) dress rehearsal, ~5B tokens | T4 | ~40 |
| Base (350M) pretraining + annealing, ~10B tokens | Plan A: TPU / Plan B: T4 | Plan A: ~26 TPU h measured for 10B tokens (+ ~4 h annealing) / Plan B: ~200 T4 h |
| SFT (two stages) | T4 | ~15 |
| Search + expert iteration (2–3 rounds) | T4 | ~40 |
| GRPO | T4 + monthly credits | ~50 |
| Evaluation and export | T4 | ~10 |

| | T4 hours | TPU hours | Weeks of quota |
|---|---|---|---|
| Plan A | ~235–265 | ~40–50 | ~9 of T4, ~2–3 of TPU, running in parallel |
| Plan B | ~435–465 | 0 | ~15–16 of T4 |

If the T4 quota gets tight, cut in this order: GRPO down to ~25 h (stage 4 often captures most of
the gain), then the nano run down to ~2B tokens.

## Repo layout

```
.
├── PLAN.md
├── DATA.md
├── configs/            # model + training YAMLs (debug, nano, base, sft, dpo)
├── data/
│   ├── sources/        # one fetcher per source, with licence metadata
│   ├── pipeline/       # datatrove steps: normalise, filter, dedup, score, decontaminate
│   ├── quality/        # rubric, annotation scripts, classifier training
│   ├── synth/          # prompt templates, generation, execution verifier
│   ├── coverage/       # topic tagger and coverage matrix
│   └── tokenize.py     # text -> uint16 shards
├── tokenizer/          # training script + trained tokenizer.json
├── model/              # model.py (HF-compatible config), init, export
├── tpu/               # MaxText configs, HF <-> MaxText checkpoint conversion (plan A)
├── train/
│   ├── pretrain.py     # DDP loop, Muon/AdamW, WSD, FIM, checkpoint/resume
│   ├── sft.py
│   ├── search.py       # decompose-test-backtrack search, expert-iteration data
│   ├── dpo.py
│   └── grpo.py         # TRL GRPOTrainer + vLLM, execution reward
├── sandbox/            # shared test runner: limits, timeouts, hack detection
├── problems/           # problem bank builders and splits
├── eval/               # harness wrappers, concept suite
├── serve/              # fast and search modes, GGUF export
└── kaggle/             # thin notebooks, driver.py (chains sessions, tracks quota, notifies)
```

Keep notebooks thin. All logic lives in the repo, so a Kaggle session is `git clone`, attach data,
run one command.

## Timeline: fast track, about 8 weeks

The measured TPU speed and the ready-made datasets let us overlap far more than the first draft
did. The rule: never let a quota sit idle. Each week the TPU, the T4s and the CPU sessions all
have work. Pretraining starts on a first version of the data and the best data, which takes
longer to make, is saved for the annealing phase at the end (the WSD schedule allows this).

| Week | TPU (20 h/week) | T4 (30 h/week) | CPU sessions + free APIs | Done when |
|---|---|---|---|---|
| 1 | Heavy data processing on the 224-core TPU host (dedup, tokenising), since the TPU quota is otherwise idle this week | Eval harness; baselines for SmolLM2 and Qwen2.5-Coder-0.5B | Repo, trainer with resume, session chain, notifications. Download and filter the ready-made sets. Concept suite and decontamination lists | Resume works across sessions; data v1 filtered |
| 2 | Tokenizer on the real mix; debug-model ablations (mix, LR, Muon vs AdamW), ~13 min per 1B tokens | Quality-classifier annotation with a local teacher | GitHub collection gates 1–4; synthetic generation with gpt-oss-120b / Qwen3.8-27B on Groq and Cerebras free tiers | Data v1 tokenised (≥ 5B tokens); mix chosen from ablations |
| 3 | 125M rehearsal, ~2B tokens (~2 h). Then the 350M stable phase starts | Synthetic exercises with execution checks | GitHub repos, PyPI, StackExchange; classifier scoring | 350M training, loss curve healthy |
| 4 | 350M stable phase continues (~7B tokens/week) | Problem bank and sandbox | Data v2: tier A for annealing | ≥ 10B tokens seen |
| 5 | Annealing on tier A, then 4096 context | SFT data conversion and checks | SFT and preference data | Annealed base; HumanEval+ measured |
| 6 | Two-stage SFT on TPU (it's ordinary training, so it's fast here); sampling for expert iteration | Sandboxed test runs for the search | — | Python-only rate > 98% |
| 7 | Expert-iteration rounds | GRPO with TRL | — | Each stage gains on dev, or is cut |
| 8 | Final evals | Exports (HF, GGUF) | Model and data cards | Release |

What can still slow it down: TPU queue waits (each session can wait about 9 hours), data quality
work that turns out to need another pass, and GRPO needing more T4 hours than budgeted. If that
happens, cut GRPO first. The 16-week schedule is the fallback if the fast track slips badly.

## Open-sourcing

- Code: Apache-2.0, public GitHub repo.
- Weights: Hugging Face Hub, safetensors, HF-compatible config, plus GGUF quantisations.
- Data: publish the pipeline and the synthetic data where licences allow. Data card lists every
  source, licence and filter step with token counts.
- Model card: training compute (GPU hours), data mix, every eval with decontamination method,
  known weaknesses, and the comparison with same-size baselines.

## Risks

| Risk | Mitigation |
|---|---|
| fp16 instability (loss spikes, NaNs) | QK-norm, z-loss, fp32 norms and softmax, grad clipping, roll back to last checkpoint and skip the bad batch |
| Kaggle quota or limits change | Code stays hardware-agnostic; checkpoints resume on any free GPU (monthly credits, Colab) |
| TPU port takes too long or runs slowly | Hard deadline at week 5, then plan B on T4 |
| Free API tiers shrink or change terms | All generation also works on Kaggle T4 with a 7B teacher, just slower; lean more on ready-made datasets |
| Measured throughput lower than estimated | Drop to ~250M or accept fewer tokens; decide in week 1, not week 8 |
| Synthetic data is repetitive or wrong | Execution verification, diversity prompts, MinHash dedup, cap per-topic share |
| Benchmark contamination | Decontaminate early, report LiveCodeBench, keep the concept suite private until release |
| Licence problems in data | Permissive-only filter, document every source, honour opt-outs |
| Small model just isn't good enough | Track B fine-tune as fallback; frame the release around per-parameter results on free compute |

## First steps

1. Hardware probes are done, TPU included (see "Tested on your account"). Still to confirm: weekly
   GPU and TPU hours, how TPU queue times vary (first wait was ~9 h), HF Hub and Kaggle dataset storage limits, the free API tiers and
   the monthly credits. Apply to TPU Research Cloud.
2. Write `model/model.py` and `train/pretrain.py` for the debug config; run it on 2× T4 with DDP.
3. Measure tokens/s for debug, nano and base configs; update the throughput tables and confirm
   the size.
4. Write the curriculum and the 300-task concept suite. It drives both data collection and
   evaluation.
5. Request access to Stack v2 / Stack-Edu and start DATA.md phase 1 in Kaggle CPU sessions.
6. Set up EvalPlus and run the baselines, so there's a number to beat before training starts.

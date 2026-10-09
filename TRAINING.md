# Training regime

From random weights to the released model in seven stages. Model sizes and hardware are in
[PLAN.md](PLAN.md); data is in [DATA.md](DATA.md).

| Stage | What | Data | GPU hours | Output |
|---|---|---|---|---|
| 1 | Pretraining, stable phase (TPU on plan A) | Full mix, 2048 context | ~170 T4 or ~40–50 TPU | Base checkpoint |
| 2 | Annealing + context extension | Tier A heavy, then 4096 context | ~30 | Annealed base |
| 3 | Supervised fine-tuning, two stages | Broad SFT, then curated SFT | ~15 | SFT model |
| 4 | Search with recursive backtracking + expert iteration | Problem bank | ~40 | Model that has learned from its own verified search |
| 5 | Reinforcement learning (GRPO) | Problem bank, filtered by difficulty | ~50 | RL model |
| 6 | Checkpoint averaging and final eval | None | ~5 | Release candidate |
| 7 | Export | None | ~1 | safetensors + GGUF |

Stages 1–3 teach the model Python. Stages 4–5 teach it to be right. At 350M most of the quality
comes from stages 1–3; 4 and 5 sharpen it. Expect gains of a few to ~10 points of pass@1 from
4 and 5 together, not a different model.

## Stage 1: pretraining, stable phase

Recipe details are in PLAN.md (Muon + AdamW, WSD schedule, fp16 with loss scaling, FIM at 50%,
packing with document masks). Additions aimed at efficiency:

- Sequence-length warmup: start at 512 tokens for the first ~2% of steps, then 1024, then 2048.
  Short sequences early are cheaper and the model only learns local syntax at first anyway.
- Batch-size warmup: ~128k tokens per step at the start, rising to ~0.5M. Small models learn
  faster per token with small early batches.
- Repo-level documents packed whole where they fit, so cross-file context is seen as one unit.
- Evals every ~500 steps: held-out tier A loss, and a fast 40-task subset of HumanEval+ and the
  concept suite (greedy, so it's cheap).
- Ablation worth one debug run: selective token loss (Rho-1). A small reference model trained on
  tier A scores each token, and the main model's loss skips tokens that are already easy or
  pure noise. It reported large efficiency gains on maths; untested on code at this size, so it
  only goes in if the ablation shows a gain.

## Stage 2: annealing and context extension

- The WSD decay phase, last ~15% of tokens. LR decays to ~0 (linear or 1-sqrt).
- Mix shifts to tier A: synthetic, rewritten code, curated repos, docs (DATA.md has the numbers).
- Last ~1–2% of tokens at 4096 context with RoPE base raised, on repo-level documents, so
  multi-file projects fit.
- Keep the last stable-phase checkpoint. If a better data mix appears later, anneal again from it
  without redoing stage 1.

## Stage 3: supervised fine-tuning

Two stages, like OpenCoder: broad first, then a smaller, cleaner set.

| | Stage 3a: broad | Stage 3b: curated |
|---|---|---|
| Size | ~300k examples | ~50–80k examples |
| Content | General coding tasks, explain-code, fix/refactor, multi-file | Curriculum tasks with tests, design-as-code, self-repair, hardest verified tasks |
| Epochs | 2 | 3 |
| Peak LR | ~5e-5 | ~2e-5 |

- Template and conversion rules from DATA.md: the response is always one Python module.
- Loss on response tokens only. Pack several short examples per sequence with attention masks
  between them.
- Self-repair examples (new in 3b): the prompt holds code plus the traceback or failing test; the
  response is the fixed module. This is what makes stage 4 work.

  ```
  <|user|>
  This fails. Fix it.
  <|code|>
  ...module...
  <|traceback|>
  Traceback (most recent call last): ...
  <|python|>
  ...fixed module...
  <|end|>
  ```

- Test-writing examples: the prompt is a module, the response is its pytest file. Stage 4 needs
  the model to write tests for its own code.
- Exit check: Python-only rate > 98%, HumanEval+ and concept suite above the base model, no drop
  in held-out loss on tier A code beyond noise.

## Stage 4: search with recursive backtracking, then expert iteration

The idea: let the model solve hard problems slowly, with search, testing and backtracking. Keep
only verified solutions. Then train the model to produce them directly. This is expert iteration
(STaR, ReST-EM): search acts as the teacher and the model learns to need less search over time.

### The search: decompose, test, backtrack

Hard tasks are broken into functions, each function is tested on its own, and a dead end sends
the search back up a level. Every step is plain Python, so it fits the Python-only format.

```python
def solve(spec, depth=0):
    for plan in model.sample_plans(spec, k=K):          # module of stubs with docstrings + tests
        module = plan
        for fn in plan.functions_in_dependency_order():
            impl = implement(fn, tries=R)               # sample, run its tests, repair from traceback
            if impl is None and depth < MAX_DEPTH and fn.is_decomposable():
                impl = solve(fn.as_spec(), depth + 1)   # recurse: split the hard function further
            if impl is None:
                break                                   # backtrack: abandon this plan, try the next
            module = module.fill(fn, impl)
        else:
            if run_hidden_tests(module, spec).passed:
                return module
    return None

def implement(fn, tries):
    candidate = model.sample_body(fn)
    for _ in range(tries):
        result = sandbox.run(candidate, fn.self_tests)
        if result.passed:
            return candidate
        candidate = model.repair(candidate, result.traceback)   # self-repair from stage 3b
    return None
```

Settings to start with: K = 4 plans, R = 3 repair attempts, MAX_DEPTH = 2.

Guards:

- Self-written tests guide the search but never decide acceptance. Only the problem's hidden
  tests accept a solution, since the model's own tests can be wrong.
- When several candidates pass self-tests, rank them by agreement with the other candidates'
  tests (CodeT-style) before spending a hidden-test run.
- Simple problems skip decomposition: try best-of-8 first and only search if all fail. Most
  compute should go to problems near the model's edge.

### What gets trained on

Each round produces four kinds of data from the search logs:

| Data | Becomes | Teaches |
|---|---|---|
| Final verified solution for the original prompt | SFT example | Solving in one shot what used to need search |
| Plans that led to success | SFT example (spec → stubbed module) | Decomposition |
| Failed code + traceback → working fix | SFT example | Self-repair |
| Passing vs failing candidate for the same function | DPO pair | Telling right from almost right |

Focus on "frontier" problems: solved by search but not by the model's one-shot greedy answer.
Problems it already solves add little; problems search can't solve add nothing.

### Rounds

- 2–3 rounds. Each round: search over the problem bank, collect data, fine-tune from the previous
  round's model (small LR, 1–2 epochs, mixed with ~30% of stage 3b data so nothing is forgotten),
  then one DPO pass on the pairs.
- Stop when a round adds less than ~1 point on the dev split.

### Compute

The model is small, so generation is cheap: a 350M model in vLLM on 2× T4 should manage roughly
10k tokens/s. 10k problems × ~60 samples × ~300 tokens ≈ 180M tokens ≈ 5 hours of generation per
round.

The real bottleneck is CPU for running tests. Kaggle gives 4 cores. Batch many candidates per
worker process, use short timeouts (2 s per test file), and run heavy rounds' tests on a local
machine if needed.

## Stage 5: reinforcement learning with GRPO

Online RL with execution rewards, starting from the stage 4 model.

### Setup

- Algorithm: GRPO with the fixes from DAPO and Dr. GRPO: token-level loss, asymmetric clipping
  (ε_low 0.2, ε_high 0.28), no per-group std normalisation, and dynamic sampling that drops groups
  where every sample got the same reward (they carry no signal).
- KL to the reference model: start at 0. Add ~0.001 only if outputs drift (format breaks, length
  explodes).
- Group size 8, ~64 prompts per step, max 768 new tokens.
- LR ~1e-6 to 3e-6, constant after a short warmup. Sweep on the debug model first.
- Infra: TRL's `GRPOTrainer` with vLLM. The model is HF-compatible (PLAN.md), so no custom
  rollout code. One T4 generates, one trains, weights synced every step (700 MB, fast).

### Reward

| Outcome | Reward |
|---|---|
| Doesn't parse as Python | -1 |
| Parses, fails hidden tests | 0 |
| Passes all hidden tests | +1 |

All-or-nothing on tests on purpose: partial credit for passing some tests is easy to game. The
parse penalty keeps the Python-only format alive without a separate format reward.

### Stopping reward hacking

- Tests run in a separate process. The solution is imported as a module; the harness, not the
  solution, writes the result.
- Sandbox: no network, read-only filesystem except a temp dir, CPU and memory limits, timeout.
- Block the known tricks: `sys.exit`/`os._exit` before tests, patching `builtins`, editing
  `unittest`/`pytest` internals, reading test files. Flag any solution that touches these and
  give it 0.
- Audit 50 random high-reward samples every ~100 steps by reading them.

### Curriculum

- Estimate each problem's pass rate from the stage 4 samples. Keep problems the model solves 10–90%
  of the time; outside that range GRPO gets no gradient.
- Recompute pass rates every ~200 steps and refresh the pool.
- Weight engineering tasks (curriculum, design-as-code, repo-mined) above algorithm puzzles,
  since engineering is the goal.

### What to watch

- Pass@1 on dev should rise. Pass@8 should not fall: if it does, the model is collapsing to one
  style of answer, so lower the LR or stop.
- Mean response length: a slow rise is fine, a jump means it's padding.
- Python-only rate must stay above 98%.

## Stage 6: checkpoint averaging and final evaluation

- Average the last few checkpoints of stage 5 (and separately test averaging stage 4 and 5
  models). Weight averaging is free and usually gives a small, steady gain.
- Run the full eval suite from PLAN.md on the base, SFT, stage 4, stage 5 and averaged models, so
  the release shows what each stage added.
- Pick the release model by dev scores, then report test scores once.

## Stage 7: export

- safetensors with an HF config that loads without custom code.
- GGUF in fp16, Q8_0 and Q4_K_M; check that the quantised models keep their pass@1 within ~1
  point.
- The serving wrapper ships two modes:
  - Fast: one sample, parse check, resample on failure.
  - Search: the stage 4 decompose-test-backtrack loop with the model's own tests, for when
    accuracy matters more than speed.

## Problem bank for stages 4 and 5

Both stages need problems with hidden tests. Target 20k–50k, all Python.

| Source | Share | Notes |
|---|---|---|
| Our curriculum and design-as-code tasks (synthetic, verified) | ~40% | The engineering focus lives here |
| Repo-mined: functions and their real tests from the verified repos (DATA.md, gate 5) | ~30% | Real-world code. "Implement this function so the repo's tests pass" |
| Verified public sets: KodCode, TACO, CodeContests, APPS train split | ~30% | Algorithmic. Check each licence; decontaminate against APPS test and all evals |

Splits: train (search and RL), dev (~1k, for curriculum and early stopping), and the eval sets,
which are never touched. MBPP and HumanEval stay out of every split.

## The sandbox

One shared runner used by data verification, stage 4, stage 5 and eval:

- Subprocess per batch of candidates, fresh temp dir, no network, `resource` limits on CPU time
  and memory, wall-clock timeout.
- Runs in Kaggle CPU sessions. Docker is only needed if you ever want to debug it locally.
- Sessions that run untrusted code (model outputs, repo test suites, `pip install` of repos)
  never have secrets attached: no Kaggle, HF or GitHub tokens. They write results to their own
  output, and a separate trusted CPU notebook picks those up and uploads them. Training and
  relay notebooks only ever run our own code.
- Returns: passed or failed per test, traceback, runtime, and flags for any blocked behaviour.

## Working within Kaggle's session limits

A Kaggle session stops after 12 hours (TPU sessions may be shorter; check). No stage in this plan
fits in one session, and none needs to. Every long job is built as a chain of sessions that each
pick up exactly where the last one stopped. The weekly quota is the real limit; session length
only costs a few percent in restart overhead.

### What the CLI supports (checked)

The Kaggle token in `.env` works with the official CLI (2.2.4), and `kaggle kernels push` can
start runs without the web UI:

- `machine_shape` in `kernel-metadata.json` (or `--accelerator`) picks the hardware. The docs list
  `NvidiaTeslaT4` (2× T4), `NvidiaL4`, `NvidiaL4X1`, `TpuV5E8`, `TpuV6E8`, `NvidiaTeslaA100` and
  `NvidiaH100` as of September 2026. Which of these are free on your account, and with what
  weekly hours, gets tested in week 1 with a 5-minute probe notebook per accelerator.
- `--timeout` caps a run's length.
- `kernel_sources` attaches another notebook's output as input, `dataset_sources` attaches data,
  `model_sources` attaches a Kaggle Model version.
- `kaggle kernels status` and `kaggle kernels output` watch a run and download what it wrote.

TPU v5e-8 has 8 chips at ~197 bf16 TFLOPS each. Measured on 9 October: the 350M config runs at
~108k tokens/s (19% utilisation), about 9× the 2× T4 estimate, so 10B tokens take ~26 TPU hours:
about 3 sessions if Kaggle's TPU session limit is 9 hours, as reported (to confirm). Each session can wait ~9 h in the queue first.

### Rule 1: every job can stop at any moment and resume exactly

Each checkpoint holds:

- Model weights and optimizer state (Muon momentum, AdamW moments), plus the fp16 loss scaler on T4.
- Step number, LR schedule position, tokens seen.
- Data position: which shard, which offset, the shuffle seed and epoch. Resuming never repeats or
  skips data.
- RNG states, and the W&B run id so the loss curve continues as one line.

Saving rules:

- Every ~20–30 minutes, so a crash loses at most half an hour.
- Atomic: write to a temp folder, then rename. A session killed mid-save never corrupts the last
  good checkpoint.
- Keep the last 2, plus a permanent one every ~1B tokens (for rollbacks and the final averaging).
- On TPU, MaxText's Orbax checkpointing does this natively, including async saves so training
  doesn't pause. Confirm in the port that it works with a local path, since MaxText usually
  expects a Google Cloud bucket.

### Rule 2: stop cleanly before Kaggle stops you

The training script knows when its session started. At ~25 minutes before the limit it stops
taking new steps, writes a final checkpoint, uploads the backup and exits normally, so a hard kill
from Kaggle never happens. `--timeout` is set as a second safety net.

### Rule 3: the checkpoint lives outside the session

| Store | How | Role |
|---|---|---|
| Notebook output (`/kaggle/working`, ~20 GB) | Next session attaches it through `kernel_sources` | Primary. No secrets needed |
| Hugging Face Hub, private model repo | Upload from inside the session with an HF token stored as a Kaggle secret | Backup off Kaggle, and the way checkpoints move between TPU and T4 stages |

A notebook can't attach its own output, so the chain alternates between two notebooks:
`train-a` attaches `train-b`'s output, then `train-b` attaches `train-a`'s. Each run loads
whichever checkpoint has the higher step.

A 350M checkpoint with optimizer state is ~4–5 GB, so two of them fit in the 20 GB output.

### Rule 4: the chain runs itself, nothing runs on your machines

Your laptop can be off and the VPS isn't involved. Two pieces keep a chain going:

1. Relay notebook (Kaggle CPU, free, no quota). As its last step, a training session pushes a
   tiny CPU notebook. The relay polls until the training session has fully ended, reads its log
   (last step, loss, any errors), then pushes the next session with the right accelerator, data
   and previous output attached. A session can't start its own successor directly: the TPU allows
   only one session at a time, and two overlapping GPU sessions would waste quota.
2. Watchdog (GitHub Actions, scheduled every 30 minutes in the public repo, about 1 minute per
   run). If nothing is running, the chain isn't finished, and quota is left, it pushes the next
   session. This covers a relay that failed or a session that died without starting one. The
   Kaggle token lives in an encrypted GitHub secret.

Rules both follow:

- On a failure: retry once from the last good checkpoint. On a second failure, or a loss spike,
  stop the chain and send a notification instead of burning quota.
- Hours used per accelerator per week are logged in a small JSON file in the HF repo. When the
  quota is nearly spent, the chain pauses and the watchdog resumes it after the weekly reset.
- Every session end sends a notification to a free ntfy.sh topic, so you can follow from your
  phone. W&B shows the curves.

TPU queue waits are expected (the probe waited in the queue). The relay and watchdog simply wait;
a gap costs calendar time, not quota.

### Rule 5: make session start-up fast

Every minute of setup is paid again in each session.

- Data shards are Kaggle datasets, attached read-only. Nothing is downloaded at start.
- Python wheels are pinned and stored in a Kaggle dataset, installed offline in seconds.
- On TPU, the XLA compilation cache is saved with the checkpoint and reloaded, so the model
  compiles once rather than every session.
- Target: training running within ~10 minutes of session start, so ~2–3% overhead.

### How each stage splits into sessions

| Stage | Hardware | Sessions | How it splits |
|---|---|---|---|
| Base pretraining + annealing | TPU | ~3–6 | Checkpoint chain as above |
| Nano pretraining | T4 | ~4 | Same chain on T4 |
| Synthetic generation, annotation | T4 | Many short ones | Work split into numbered shards. Each session takes the next unfinished shards and writes results to its output, so nothing is redone |
| SFT | T4 | 1–2 | Checkpoint chain |
| Search + expert iteration | T4 | ~3–4 per round | Problem bank split into shards like generation; the fine-tune step is a short chain |
| GRPO | T4 | ~4–5 | Checkpoint chain; the rollout buffer is small and is regenerated after a resume |
| Evals | T4 | 1 per checkpoint | Each eval is its own short session |

### Running TPU and T4 side by side

The TPU and GPU quotas are separate, so in weeks 7–10 the base model trains on TPU while T4
sessions generate data and build the problem bank. Check in week 1 whether Kaggle allows a TPU and
a GPU session at the same time on one account. If it doesn't, the driver alternates them, which
still uses both quotas fully over a week.

### What's lost when something goes wrong

| Event | Loss |
|---|---|
| Session hits its time limit | Nothing; it stopped itself 25 minutes early |
| Session crashes or is preempted | At most ~30 minutes of training |
| Bad checkpoint or loss spike | Roll back to the previous or a permanent checkpoint, skip the bad data window |
| Weekly quota runs out mid-stage | Nothing; the driver waits for the reset |
| Kaggle output lost | Resume from the Hugging Face backup |

## Ablations before spending real compute

All on the 20M debug model, ~1–2 GPU hours each:

| Question | Compare |
|---|---|
| Optimizer | Muon + AdamW vs AdamW |
| Data | Chosen mix vs raw Stack Python; with vs without rewritten code; formatted vs unformatted |
| Loss | Standard vs selective token loss |
| FIM rate | 0% vs 50% |
| Post-training | SFT only vs SFT + one expert-iteration round (on the 125M nano model) |

Anything that doesn't show a clear gain at small scale stays out of the big run.

# Reproducing Ophyx

Every heavy step runs as a Kaggle kernel pushed from a small machine (the "orchestration box"),
which only writes a script, calls the Kaggle CLI and reads results. Nothing here needs a local GPU.

## Accounts and keys

1. **Kaggle**, phone-verified so GPU and TPU sessions are allowed. The weekly free quota was 30 h of
   T4 time and 20 h of TPU v5e-8 time while this ran. Create an API token and put it in `.env` as
   `KAGGLE_API_TOKEN`, plus `KAGGLE_USERNAME` if you aren't the author.
2. **Hugging Face**, with a write token as `HF_TOKEN`. Create a private dataset repo for the data and
   a private model repo for checkpoints, then replace `abhash-chakraborty/ophyx-data` and
   `abhash-chakraborty/ophyx-runs` with yours:

   ```bash
   grep -rlE "abhash-chakraborty/ophyx-(runs|data)" data train kaggle
   ```
3. **A private Kaggle dataset** holding `keys.json` = `{"HF_TOKEN": "..."}`. The author's is called
   `ophyx-keys`; the commands below use that name. Attach it only to sessions that run no model
   output.
4. Optional: `NTFY_TOPIC` for phone notifications from the watchdog, and the same three values as
   GitHub Actions secrets if the watchdog should run there.

Install the CLI the way the scripts call it: `uv run --no-project --with kaggle kaggle ...`, or
`pip install kaggle==2.2.4`.

## Launching a job

```bash
python kaggle/launch.py <script> <kernel-slug> --accel cpu|t4|tpu --timeout <seconds> \
    --config '<json>' [--datasets ...] [--kernel-sources ...] [--include ...] [--no-internet] [--dry-run]
```

`launch.py` writes the script with a `CONFIG = {...}` line on top (and any `--include` files before
it), lints the assembled file with ruff, and pushes it. `--dry-run` builds the kernel folder without
pushing. Outputs come back with `kaggle kernels output <user>/<slug> -p <dir>`.

One rule holds for every job: a session that executes model-written code gets no keys, and every
session that runs the trained model's code also gets `--no-internet`. Model weights and wheels reach
those sessions through `data/jobs/relay_in.py`, a separate session that downloads them and runs
nothing. The one session with internet on is `data/synth/t4_generate.py`, which runs the public
teacher model's code next to the teacher, because it installs vLLM from PyPI.

## Stages

Each stage's script documents its CONFIG keys in its docstring.

### 1. Data

| Step | Script | Session |
|---|---|---|
| Fetch, clean and filter each source into Parquet parts | `data/jobs/ingest.py` (one kernel per shard) | CPU, keys |
| Signatures, global dedup, decontamination | `data/jobs/stageb.py`, jobs `sig` then `dedup` | CPU, keys |
| Train the 32k tokenizer, tokenise | `data/jobs/stageb.py`, jobs `tokenizer` then `tokenize` | CPU, keys |
| Copy token files into a kernel output for fast TPU attach | `data/jobs/pack_tokens.py` | CPU, keys |

The sources, filters and mix are in [data.md](data.md).

### 2. Problem bank and repair data

| Step | Script | Session |
|---|---|---|
| Download TACO, CodeContests and APPS; decontaminate | `data/jobs/problem_bank.py` (`--include data/jobs/sft_build.py`) | CPU, keys |
| Run every reference solution against its tests | `data/jobs/verify_bank.py` (`--include data/jobs/judge.py`) | CPU, offline, no keys |
| Mutate passing programs, keep the real failures | `data/jobs/bug_inject.py` (`--include data/jobs/judge.py`) | CPU, offline, no keys |
| Build the tokenised SFT sets | `data/jobs/sft_build.py` | CPU, keys |

### 3. Pretraining and fine-tuning

Long runs are chains: two kernels per run take turns, each resuming from the other's output or the
HF backup until the run finishes. The configs are in `kaggle/chains/`, run in this order:
`nano125m` (a 125M rehearsal), `base350m`, `ctx350m`, `sft350m`, `sft350m2`. Set `"enabled": true`
in the one you want, then run the watchdog by hand or on a schedule:

```bash
python kaggle/watchdog.py
```

It pushes the next session when neither kernel is queued or running, and stops the chain when the
run finishes, hits a NaN or fails twice in a row. State lives in `<run>/status.json` and
`<run>/chain_state.json` in the checkpoint repo. `train/export_ckpt.py` turns a checkpoint into a
Hugging Face model.

### 4. Benchmarks

```bash
# once: download every public test set into a kernel output
python kaggle/launch.py eval/bench_kit.py ophyx-bench-kit --accel cpu

# generate answers on 2x T4 (configs in eval/runs/)
python kaggle/launch.py eval/bench_gen.py ophyx-bench-gen-x --accel t4 --timeout 43200 \
    --config "$(cat eval/runs/r3_sft2.json)" --datasets ophyx-evaldata --kernel-sources ophyx-bench-kit

# score them offline, then merge the results
python kaggle/launch.py eval/bench_exec.py ophyx-bench-exec-x --accel cpu --timeout 43200 --no-internet \
    --config '{"jobs": 4}' --datasets ophyx-evaldata --kernel-sources ophyx-bench-kit ophyx-bench-gen-x
python eval/bench_report.py results.json --md eval/R1_RESULTS.md
```

`ophyx-evaldata` is a Kaggle dataset with the concept suite (`eval/concept_suite`) and the
hands-on set (`eval/manual`). A model of your own is attached as a kernel source and named by a glob,
for example `"/kaggle/input/**/grpo/p4-lr2e6/final"`.

### 5. Search, DPO and RL

| Step | Script | Session |
|---|---|---|
| Sample a model on the bank, 8 tries plus greedy | `data/jobs/bank_sample.py` (`--include data/jobs/verify_bank.py`) | T4, no generated code runs |
| Judge the samples: pass rates, expert rows, pass/fail pairs | `data/jobs/bank_judge.py` (`--include` judge, verify_bank, bug_inject) | CPU, offline, no keys |
| DPO on pass/fail pairs | `train/dpo_t4.py` | T4, keys |
| GRPO with a pass-the-tests reward | `train/grpo_t4.py` | T4 x2, offline, no keys |

GRPO trains on problems the model solves 1 to 6 times in 8, so it needs the judge's pass rates.

### 6. Averaging and export

```bash
python train/average_ckpt.py dirA dirB --weights 0.7 0.3 --out merged    # or as a Kaggle CPU job
```

`train/export_gguf.py` builds llama.cpp at a pinned tag, converts and quantises, then checks the
GGUF files against the transformers model: the RoPE base, token ids on test snippets, and greedy
log-probs. Run it on a Kaggle CPU session with keys.

### 7. Demo

```bash
modal secret create ophyx-hf HF_TOKEN=...      # the image build downloads the model with it
OPHYX_ORIGINS=https://your.site modal deploy serve/modal_app.py
```

One container at most, scale to zero after 3 idle minutes, and 6 requests a minute per IP. Set a
monthly budget in the Modal dashboard as the hard cap.

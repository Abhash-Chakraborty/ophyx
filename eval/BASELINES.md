# Baselines

Measured 10 October 2026 with `eval/evalplus_run.py` on Kaggle T4: EvalPlus 0.3.1, greedy decoding,
pass@1, base (non-instruct) models, max 512 new tokens. HumanEval prompts have trailing whitespace
stripped (so tokenizers that merge a newline with the next indent aren't penalised). MBPP prompts are
used as given. Our own checkpoints are scored with exactly the same script and settings.

| Model | Params | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|
| SmolLM2-135M | 135M | 4.3 | 3.0 | 7.7 | 4.8 |
| SmolLM2-360M | 360M | 11.6 | 9.8 | 27.0 | 22.2 |
| Qwen2.5-Coder-0.5B | 494M | 26.2 | 22.0 | 55.6 | 48.4 |

Harness check: Qwen2.5-Coder-0.5B with the unstripped HumanEval prompt scores 28.0 / 23.8, which
matches the numbers in Qwen's report exactly. Reference solutions score 99.4% on HumanEval+ (one
timeout on a slow CPU).

Targets for our models (PLAN.md): nano (125M) above SmolLM2-360M, base (350M) at 30–40% HumanEval+.

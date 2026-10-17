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

## Our checkpoints

Same script and settings as the baselines above.

| Model | Params | Tokens | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|---|
| nano125m-v1 (step 7629, 15 Oct 2026) | 126M | 2B | 17.7 | 15.9 | 25.4 | 20.9 |
| nano125m-sft-v1, completion prompt | 126M | 2B + 0.6B SFT | 20.7 | 19.5 | 0.0 | 0.0 |
| nano125m-sft-v1, SFT template | 126M | 2B + 0.6B SFT | 17.1 | 15.2 | 43.1 | 36.8 |
| nano125m-sft-v1, SFT template + signature prefill | 126M | 2B + 0.6B SFT | 17.7 | 16.5 | 43.1 | 36.8 |

nano125m-v1 beats SmolLM2-360M on HumanEval+ (15.9 vs 9.8) and lands just under it on MBPP+ (20.9 vs
22.2), at a third of the size. Every output parsed as Python.

nano125m-sft-v1 is nano125m-v1 after 600M tokens of SFT (OpenCoder stage 1 then stage 2, filtered
to Python and decontaminated; 0.7 h on a v5e-8). MBPP+ goes from 20.9 to 36.8 with the SFT template.
HumanEval+ is best as plain completion (19.5). The completion MBPP score of 0 is a prompt mismatch:
the MBPP prompt is a bare docstring, and the SFT model ends its answer there instead of writing a
function. The MBPP numbers are the same with and without prefill because prefill only applies to
HumanEval.

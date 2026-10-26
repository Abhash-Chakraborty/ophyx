# R1 results

Every open model below ran the same way on Kaggle's free T4s and was scored offline on Kaggle CPUs,
in October 2026: 46 models from 0.13B to 4B parameters, ours included. Seven frontier API models
answered two or three of the benchmarks through Kaggle Benchmarks and were scored the same way.

## How it was run

- Generation: `eval/bench_gen.py`, vLLM 0.32.0 on 2× T4, greedy. Base models get plain completion
  prompts, instruct models their own chat template, and ours the Ophyx SFT template with the function
  signature prefilled. Configs are in `eval/runs/`.
- Scoring: `eval/bench_exec.py` in a Kaggle CPU session with no internet and no keys. Each answer runs
  in its own process with CPU, memory and file limits.
- Empty cells: base models skip BigCodeBench instruct and the hands-on tasks, which need instructions.

| Benchmark | Problems | What counts |
|---|---|---|
| HumanEval, HumanEval+ | 164 | EvalPlus 0.3.1, base and plus tests |
| MBPP, MBPP+ | 378 | EvalPlus 0.3.1, base and plus tests |
| Concept | 298 | our concept suite, `eval/concept_suite/run.py` |
| BCB-H complete, instruct | 148 | BigCodeBench-Hard unit tests, scored on the 131 tasks whose reference solution passes offline |
| LCB | 342 | LiveCodeBench problems dated 22 Sep 2024 to 6 Apr 2025, every public and private test |
| CRUX-I, CRUX-O | 800 each | CRUXEval input and output prediction |
| Hands-on | 42 | our own small programming tasks, `eval/manual` |

## Ophyx against open models

All numbers are pass@1 in percent.

| Model | Kind | Params | HumanEval | HumanEval+ | MBPP | MBPP+ | Concept | BCB-H complete | BCB-H instruct | LCB | CRUX-I | CRUX-O | Hands-on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ophyx-125m-base | base | 0.13B | 17.7 | 15.2 | 25.4 | 20.9 | 0.0 | 0.8 |  | 0.3 | 14.8 | 9.0 |  |
| ophyx-125m-sft | ours | 0.13B | 17.7 | 16.5 | 43.1 | 36.8 | 0.0 | 0.0 | 0.0 | 0.0 | 8.5 | 10.6 | 9.5 |
| SmolLM2-135M | base | 0.14B | 4.3 | 3.0 | 7.7 | 5.0 | 0.0 | 0.0 |  | 0.3 | 10.0 | 13.6 |  |
| ophyx-350m-base | base | 0.35B | 32.9 | 31.7 | 49.7 | 42.3 | 0.0 | 0.0 |  | 0.3 | 28.5 | 20.9 |  |
| ophyx-350m-ctx-base | base | 0.35B | 30.5 | 28.0 | 39.4 | 33.1 | 0.3 | 1.5 |  | 0.3 | 25.1 | 22.0 |  |
| ophyx-350m-ctxsoup-50 | base | 0.35B | 30.5 | 28.7 | 46.8 | 39.2 | 0.3 | 0.8 |  | 0.9 | 27.4 | 22.0 |  |
| ophyx-350m-ctxsoup-70 | base | 0.35B | 31.1 | 29.3 | 43.1 | 35.7 | 0.3 | 0.8 |  | 0.0 | 26.8 | 22.2 |  |
| ophyx-350m-dpo-p1 | ours | 0.35B | 39.0 | 37.2 | 57.9 | 48.7 | 1.0 | 3.1 | 3.1 | 1.5 | 24.8 | 19.1 | 21.4 |
| ophyx-350m-grpo-p3-lr1e6 | ours | 0.35B | 35.4 | 32.9 | 57.1 | 48.1 | 1.0 | 3.1 | 2.3 | 2.0 | 24.6 | 19.5 | 19.0 |
| ophyx-350m-grpo-p3-lr3e6 | ours | 0.35B | 39.0 | 36.0 | 57.9 | 48.1 | 0.7 | 2.3 | 3.1 | 1.2 | 24.2 | 19.4 | 19.0 |
| ophyx-350m-sft | ours | 0.35B | 37.2 | 35.4 | 57.4 | 48.1 | 1.7 | 2.3 | 3.8 | 1.5 | 24.9 | 19.5 | 23.8 |
| ophyx-350m-sft2 | ours | 0.35B | 39.0 | 37.8 | 55.8 | 46.0 | 0.3 | 3.1 | 1.5 | 2.3 | 19.2 | 19.1 | 19.0 |
| ophyx-350m-sft2soup-50 | ours | 0.35B | 40.2 | 39.6 | 55.8 | 45.8 | 1.0 | 2.3 | 5.3 | 2.3 | 23.6 | 22.4 | 19.0 |
| ophyx-350m-sft2soup-70 | ours | 0.35B | 42.1 | 41.5 | 56.1 | 46.8 | 1.3 | 3.1 | 3.8 | 1.8 | 21.1 | 21.5 | 21.4 |
| ophyx-350m-soup-all | ours | 0.35B | 36.0 | 34.1 | 57.7 | 48.1 | 1.7 | 3.1 | 2.3 | 1.8 | 24.6 | 19.5 | 16.7 |
| ophyx-350m-soup-sdg | ours | 0.35B | 36.6 | 34.8 | 57.7 | 48.1 | 1.7 | 3.1 | 2.3 | 1.2 | 24.6 | 19.4 | 19.0 |
| codegen-350M-mono | base | 0.36B | 14.0 | 11.6 | 25.1 | 20.9 | 0.0 | 0.0 |  | 0.0 | 20.4 | 15.8 |  |
| SmolLM2-360M | base | 0.36B | 12.2 | 10.4 | 27.0 | 22.0 | 0.0 | 0.8 |  | 0.3 | 31.6 | 19.6 |  |
| SmolLM2-360M-Instruct | instruct | 0.36B | 18.3 | 15.9 | 34.7 | 27.5 | 0.0 | 0.0 | 0.0 | 0.9 | 3.6 | 20.4 | 7.1 |
| Qwen2.5-Coder-0.5B | base | 0.49B | 26.2 | 22.0 | 55.6 | 48.4 | 0.0 | 3.8 |  | 2.0 | 26.4 | 24.9 |  |
| Qwen2.5-Coder-0.5B-Instruct | instruct | 0.49B | 58.5 | 54.9 | 50.5 | 42.1 | 0.0 | 0.0 | 2.3 | 0.6 | 15.4 | 25.1 | 9.5 |
| Qwen3-0.6B-Base | base | 0.60B | 29.3 | 23.8 | 51.6 | 43.1 | 0.0 | 3.1 |  | 4.4 | 26.8 | 26.9 |  |
| Qwen3-0.6B | instruct | 0.75B | 41.5 | 36.6 | 56.1 | 46.6 | 0.0 | 2.3 | 0.8 | 7.9 | 32.2 | 24.8 | 11.9 |
| gpt_bigcode-santacoder | base | 1.12B | 18.9 | 16.5 | 37.6 | 30.2 | 0.3 | 0.8 |  | 1.2 | 26.5 | 23.0 |  |
| deepseek-coder-1.3b-base | base | 1.35B | 29.3 | 24.4 | 57.9 | 47.9 | 2.0 | 8.4 |  | 3.8 | 27.0 | 31.6 |  |
| deepseek-coder-1.3b-instruct | instruct | 1.35B | 63.4 | 61.6 | 62.2 | 52.6 | 4.0 | 2.3 | 3.8 | 5.6 | 12.4 | 28.8 | 31.0 |
| phi-1 | base | 1.42B | 52.4 | 46.3 | 27.8 | 20.6 | 0.0 | 4.6 |  | 1.8 | 13.5 | 22.1 |  |
| phi-1_5 | base | 1.42B | 35.4 | 30.5 | 53.2 | 43.4 | 0.3 | 0.0 |  | 2.3 | 23.9 | 27.6 |  |
| Yi-Coder-1.5B | base | 1.48B | 34.8 | 28.7 | 58.5 | 50.0 | 3.0 | 3.8 |  | 0.6 | 29.1 | 31.9 |  |
| Yi-Coder-1.5B-Chat | instruct | 1.48B | 70.7 | 65.9 | 65.9 | 57.1 | 3.7 | 3.8 | 10.7 | 7.3 | 27.2 | 30.6 | 23.8 |
| Qwen2.5-Coder-1.5B | base | 1.54B | 42.7 | 36.6 | 71.4 | 59.5 | 4.4 | 11.5 |  | 7.0 | 28.1 | 36.0 |  |
| Qwen2.5-Coder-1.5B-Instruct | instruct | 1.54B | 70.1 | 65.2 | 69.6 | 59.5 | 6.4 | 6.1 | 3.8 | 5.6 | 29.6 | 33.6 | 26.2 |
| SmolLM2-1.7B | base | 1.71B | 23.2 | 17.7 | 46.0 | 36.2 | 0.0 | 0.8 |  | 2.3 | 27.0 | 27.1 |  |
| SmolLM2-1.7B-Instruct | instruct | 1.71B | 36.0 | 29.9 | 55.0 | 45.0 | 1.3 | 0.8 | 3.1 | 2.6 | 28.8 | 27.6 | 26.2 |
| Qwen3-1.7B-Base | base | 1.72B | 47.0 | 41.5 | 69.0 | 58.7 | 1.3 | 6.9 |  | 6.1 | 36.6 | 35.5 |  |
| OpenCoder-1.5B-Base | base | 1.91B | 53.0 | 49.4 | 68.8 | 56.3 | 0.7 | 4.6 |  | 5.0 | 29.9 | 29.6 |  |
| OpenCoder-1.5B-Instruct | instruct | 1.91B | 68.3 | 65.2 | 72.2 | 62.2 | 3.4 | 12.2 | 8.4 | 7.3 | 24.4 | 28.1 | 33.3 |
| Qwen3-1.7B | instruct | 2.03B | 71.3 | 65.2 | 56.9 | 49.2 | 3.7 | 9.2 | 9.2 | 14.3 | 40.9 | 35.6 | 26.2 |
| phi-2 | base | 2.78B | 50.0 | 45.1 | 64.6 | 52.9 | 1.0 | 6.1 |  | 7.0 | 32.4 | 34.0 |  |
| stable-code-3b | base | 2.80B | 30.5 | 26.8 | 55.0 | 45.5 | 1.7 | 12.2 |  | 7.6 | 34.1 | 31.8 |  |
| starcoder2-3b | base | 3.03B | 31.7 | 26.8 | 60.3 | 48.7 | 1.7 | 5.3 |  | 4.4 | 32.2 | 34.6 |  |
| Qwen2.5-Coder-3B | base | 3.09B | 45.1 | 36.6 | 74.6 | 62.7 | 7.4 | 12.2 |  | 10.5 | 35.4 | 39.4 |  |
| Qwen2.5-Coder-3B-Instruct | instruct | 3.09B | 82.9 | 79.9 | 74.1 | 62.2 | 10.7 | 13.0 | 14.5 | 15.2 | 44.2 | 44.1 | 38.1 |
| granite-3b-code-base-2k | base | 3.48B | 35.4 | 29.3 | 52.1 | 44.7 | 2.0 | 0.0 |  | 5.8 | 30.5 | 32.0 |  |
| granite-3b-code-instruct-2k | instruct | 3.48B | 53.0 | 45.1 | 60.6 | 48.9 | 4.4 | 5.3 | 6.9 | 7.6 | 24.6 | 30.5 | 19.0 |
| Qwen3-4B | instruct | 4.02B | 83.5 | 76.8 | 71.4 | 61.1 | 12.1 | 26.0 | 16.8 | 21.3 | 54.0 | 45.1 | 42.9 |

## What the numbers say

- At 0.35B, ophyx-350m-sft scores 35.4 on HumanEval+. Base models up to 0.6B score lower:
  Qwen2.5-Coder-0.5B 22.0, Qwen3-0.6B-Base 23.8. Instruct models of that size do better:
  Qwen3-0.6B 36.6, Qwen2.5-Coder-0.5B-Instruct 54.9.
- On MBPP+ it scores 48.1, level with Qwen2.5-Coder-0.5B (48.4) and ahead of Qwen3-0.6B-Base (43.1).
- On the hands-on tasks it solves 10 of 42 (23.8%). That is more than Qwen2.5-Coder-0.5B-Instruct
  (9.5) and Qwen3-0.6B (11.9), the same as Yi-Coder-1.5B-Chat, and less than
  deepseek-coder-1.3b-instruct (31.0) and Qwen3-4B (42.9).
- The weak spots are the concept suite (1.7), BigCodeBench-Hard (2.3 and 3.8) and LiveCodeBench (1.5).
  They need library knowledge, longer context and multi-step algorithms. The 4,096-token run and the
  RL stages target these.
- The DPO pilot (2,181 preference pairs) is within noise of the SFT model: +1.8 on HumanEval+, +0.6 on
  MBPP+, one hands-on task fewer.
- GRPO p3 (141 steps on 1,169 problems from our bank, learning rates 1e-6 and 3e-6) is also within
  noise of SFT on every benchmark: HumanEval+ 32.9 and 36.0, MBPP+ 48.1 for both, and 8 hands-on
  tasks instead of 10. On 200 held-out bank problems it went from 15 to 18 and 20 solved, so the
  gain stayed inside the bank's own problems.
- Averaging weights does not help either. SFT + DPO + GRPO (lr 3e-6) scores 34.8 on HumanEval+ and
  8 hands-on tasks; all four checkpoints averaged score 34.1 and 7. All four start from the same SFT
  weights, so the averages stay close to them.
- The 4,096-token base (2B more tokens at seq 4,096 from ophyx-350m-base) got worse on short
  tasks: HumanEval+ 31.7 to 28.0, MBPP+ 42.3 to 33.1. It runs past the end of the function more
  often: 51 of 378 MBPP answers hit the length limit, against 19 before. Averaging it 50/50 with the
  old base brings MBPP+ back to 39.2.
- SFT v2 (800M tokens at seq 4,096 on top of that base, with the problem bank and repair rows added)
  gains on HumanEval+ (37.8) and LiveCodeBench (2.3) and loses on MBPP+ (46.0), CRUX-I (19.2) and
  hands-on tasks (8). Averaged 70/30 with SFT v1, it scores 41.5 on HumanEval+, the best of our
  models, and stays within 1.3 points of SFT v1 on MBPP+, concept and BigCodeBench-Hard. CRUX-I
  (21.1) and the hands-on tasks (9) are still lower.

Scoring notes:

- starcoder2-3b and stable-code-3b write repo markers such as `<file_sep>` as plain text after a
  program. `bench_exec.py` cuts completions there; before that fix, 317 of starcoder2-3b's 342
  LiveCodeBench answers failed as syntax errors.
- granite-3b-code-base-2k's 0.0 on BigCodeBench-Hard complete is real: its programs run and fail the tests.
- Scoring the same answers twice moved two cells by up to 0.7 points, from tests near their time limit.

## Frontier API models

Seven API models answered HumanEval and the hands-on tasks through Kaggle Benchmarks' model proxy,
at temperature 0, and `bench_exec.py` scored the replies in the same offline sandbox as the open
models. The GPT models ran with reasoning effort low, and models that reason before answering got
4,096 more output tokens. A reply cut off at the output limit was asked again once with 8,192 more
tokens. After that, two HumanEval replies are still cut off (Gemini 3.8 Flash on HumanEval/116,
Gemini 3.1 Pro on HumanEval/32) and count as failures. The proxy offered no route to Qwen3-Coder-480B
or gpt-oss-20b, so they are missing. The concept suite (298 tasks) ran on the five cheaper models over
several days of the proxy's daily quota; GPT-5.5 and Gemini 3.1 Pro did not take it. Eleven of Gemini
3.8 Flash's concept replies are still cut off after the re-ask, and the proxy returned no reply to
12 of Gemma 4 31B's prompts in two tries; both count as failures.

| Model | HumanEval | HumanEval+ | Concept | Hands-on |
|---|---|---|---|---|
| GPT-5.5 | 98.8 | 93.9 |  | 95.2 (40 of 42) |
| Claude Sonnet 5.5 | 98.8 | 95.7 | 95.3 | 90.5 (38) |
| Gemini 3.1 Pro | 99.4 | 95.7 |  | 85.7 (36) |
| Gemini 3.8 Flash | 99.4 | 93.9 | 91.6 | 90.5 (38) |
| Gemma 4 31B | 99.4 | 95.1 | 77.5 | 92.9 (39) |
| Claude Haiku 5.5 | 98.8 | 94.5 | 93.0 | 90.5 (38) |
| GPT-5.4 mini | 93.9 | 89.6 | 83.2 | 92.9 (39) |

The lowest of them still passes 147 of 164 HumanEval+ problems, 231 of 298 concept tasks and 36
of 42 hands-on tasks; ophyx-350m-sft passes 58, 5 and 10.

## LiveCodeBench leaderboard on our 342 problems

The leaderboard's per-problem results, averaged over exactly the problems we run (`eval/lcb_window.py`),
so these rows compare directly with the LCB column above.

| Model | pass@1 |
|---|---|
| O4-Mini (High) | 78.9 |
| O3 (High) | 74.9 |
| O4-Mini (Medium) | 73.7 |
| Gemini-2.5-Pro-06-05 | 73.7 |
| DeepSeek-R1-0528 | 72.7 |
| Gemini-2.5-Pro-05-06 | 72.2 |
| OpenReasoning-Nemotron-32B | 69.7 |
| EXAONE-4.0-32B | 69.6 |
| O3-Mini-2025-01-31 (High) | 68.1 |
| Grok-3-Mini (High) | 67.3 |
| OpenCodeReasoning-Nemotron-1.1-32B | 66.5 |
| O4-Mini (Low) | 64.9 |
| Qwen3-235B-A22B | 64.0 |
| XBai-o4-medium | 63.5 |
| Gemini-2.5-Flash-05-20 | 61.4 |

## Published scores of large models

From the public leaderboards (`eval/frontier_table.py`, sources in `eval/published/`). Each column
uses one leaderboard's settings; the notes under it say how they differ from our runs. Our BCB-H
columns count the 131 runnable tasks; over all 148 they would be about 11% lower.

### EvalPlus

| Model | Size | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|
| O1 Preview (Sept 2024) | closed | 96.3 | 89.0 | 95.5 | 80.2 |
| O1 Mini (Sept 2024) | closed | 96.3 | 89.0 | 93.1 | 78.8 |
| Qwen2.5-Coder-32B-Instruct | 32B | 92.1 | 87.2 | 90.5 | 77.0 |
| GPT 4o (Aug 2024) | closed | 92.7 | 87.2 | 87.6 | 72.2 |
| DeepSeek-V3 (Nov 2024) | 671B | 91.5 | 86.6 | 87.6 | 73.0 |
| GPT-4-Turbo (April 2024) | closed | 90.2 | 86.6 |  |  |
| DeepSeek-V2.5 (Nov 2024) | 236B | 90.2 | 83.5 | 87.6 | 74.1 |
| GPT 4o Mini (July 2024) | closed | 88.4 | 83.5 | 85.4 | 72.2 |
| DeepSeek-Coder-V2-Instruct | 236B | 85.4 | 82.3 | 89.4 | 75.1 |
| Claude Sonnet 3.5 (June 2024) | closed | 87.2 | 81.7 | 89.4 | 74.3 |
| GPT-4-Turbo (Nov 2023) | closed | 85.4 | 81.7 | 85.7 | 73.3 |
| Grok Beta | closed | 88.4 | 80.5 | 86.0 | 65.6 |
| Gemini 1.5 Pro 002 | closed | 89.0 | 79.3 | 89.7 | 74.6 |
| GPT-4 (May 2023) | closed | 88.4 | 79.3 |  |  |
| claude-3-opus (Mar 2024) | closed | 82.9 | 77.4 | 89.4 | 73.3 |
| Gemini 1.5 Flash 002 | closed | 82.3 | 75.6 | 84.7 | 67.5 |
| DeepSeek-Coder-33B-instruct | 33B | 81.1 | 75.0 | 80.4 | 70.1 |
| Codestral-22B-v0.1 | 22B | 79.9 | 73.8 | 72.5 | 61.9 |
| OpenCodeInterpreter-DS-33B | 33B | 79.3 | 73.8 | 80.2 | 68.5 |
| WizardCoder-33B-V1.1 | 33B | 79.9 | 73.2 |  |  |

### BigCodeBench

| Model | Size | BCB-H complete | BCB-H instruct |
|---|---|---|---|
| Gemini-Exp-1206 | closed | 40.5 | 27.7 |
| DeepSeek-V3 | 671B | 40.5 | 28.4 |
| DeepSeek-R1 | 671B | 40.5 | 29.7 |
| o1-2024-12-17 (temperature=1, reasoning=low) | closed | 39.2 | 29.7 |
| Claude-3.7-Sonnet-20250219 (temperature=1, length=12800, reasoning=3200) | closed | 39.2 | 32.4 |
| Grok-3-Beta | closed | 39.2 | 27.0 |
| o1-2024-12-17 (temperature=1, reasoning=high) | closed | 38.5 | 32.4 |
| o3-mini-2025-01-31 (temperature=1, reasoning=medium) | closed | 37.8 | 33.1 |
| o3-mini-2025-01-31 (temperature=1, reasoning=low) | closed | 37.8 | 31.1 |
| o3-mini-2025-01-31 (temperature=1, reasoning=high) | closed | 37.8 | 32.4 |
| Quasar-Alpha | closed | 37.8 | 31.8 |
| Athene-V2-Chat | 72B | 37.2 | 27.0 |
| o1-2024-12-17 (temperature=1, reasoning=medium) | closed | 37.2 | 28.4 |
| GPT-4o-2024-08-06 | closed | 36.5 | 25.0 |
| Gemini-2.0-Pro-Exp-02-05 | closed | 36.5 | 27.0 |
| Gemini-2.5-Pro-Exp-03-25 | closed | 36.5 | 29.7 |
| Qwen2.5-Max | closed | 35.8 | 29.1 |
| DeepSeek-V3-0324 | 671B | 35.8 | 27.7 |
| GPT-4-Turbo-2024-04-09 | closed | 35.1 | 29.1 |
| Claude-3.5-Sonnet-20241022 | closed | 35.1 | 25.7 |

### LiveCodeBench

| Model | Size | LCB v6 Feb-May 25 |
|---|---|---|
| O4-Mini (High) | closed | 77.1 |
| Gemini-2.5-Pro-06-05 | closed | 72.5 |
| O4-Mini (Medium) | closed | 71.8 |
| O3 (High) | closed | 71.8 |
| DeepSeek-R1-0528 | 671B | 68.7 |
| Gemini-2.5-Pro-05-06 | closed | 67.2 |
| EXAONE-4.0-32B | 32B | 66.0 |
| OpenReasoning-Nemotron-32B | 32B | 65.6 |
| O3-Mini-2025-01-31 (High) | closed | 62.6 |
| O4-Mini (Low) | closed | 62.6 |
| Grok-3-Mini (High) | closed | 62.6 |
| OpenCodeReasoning-Nemotron-1.1-32B | 32B | 61.4 |
| Gemini-2.5-Flash-05-20 | closed | 59.5 |
| XBai-o4-medium | 32.8B | 59.5 |
| O3-Mini-2025-01-31 (Med) | closed | 56.5 |
| Gemini-2.5-Flash-04-17 | closed | 56.5 |
| Qwen3-235B-A22B | 235B | 55.7 |
| O3-Mini-2025-01-31 (Low) | closed | 51.1 |
| Claude-Sonnet-4 (Thinking) | closed | 51.1 |
| Claude-Opus-4 (Thinking) | closed | 48.9 |

### CRUXEval

| Model | Size | CRUX-I | CRUX-O |
|---|---|---|---|
| gpt-4-0613 | closed | 69.8 | 68.7 |
| gpt-4-turbo-2024-04-09 (n=3) | closed | 68.5 | 67.7 |
| gpt-4o (n=3) | closed | 65.1 | 70.0 |
| claude-3-opus (n=1) | closed | 64.2 | 65.8 |
| codetulu-2-34b | 34B | 49.3 | 45.8 |
| gpt-3.5-turbo-0613 | closed | 49.0 | 49.4 |
| starcoder2-15b | 15B | 48.1 | 47.1 |
| codellama-34b | 34B | 47.2 | 42.4 |
| phind | 34B | 47.2 | 39.7 |
| deepseek-base-33b | 33B | 46.5 | 48.6 |
| deepseek-instruct-33b | 33B | 46.5 | 49.9 |
| codellama-python-34b | 34B | 43.9 | 41.4 |
| wizard-34b | 34B | 42.7 | 43.4 |
| codellama-13b | 13B | 42.5 | 39.7 |
| codellama-python-13b | 13B | 39.7 | 39.8 |
| mixtral-8x7b | 46.7B | 39.3 | 40.5 |
| wizard-13b | 13B | 36.5 | 41.3 |
| starcoderbase-16b | 15.5B | 31.3 | 34.2 |

- EvalPlus: leaderboard results.json, last modified 2024-12-26, greedy, mostly instruct prompts.
- BigCodeBench-Hard: leaderboard, last modified 2025-04-16, all 148 tasks.
- LiveCodeBench: leaderboard per-problem data (2025-08-01), recomputed for the Feb-May 2025 window.
- CRUXEval: leaderboard data.csv (2024-12-06), T=0.2 with n=10; our runs are greedy, so they are close but not identical.
- Rows from model cards and papers are left out here: they use their own prompts and windows.

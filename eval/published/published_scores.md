# Published code-benchmark scores for comparison with pycoder (350M)

Retrieved 2026-10-21. Every number in `published_scores.json` (1,788 rows) has a source URL. Leaderboard rows come from the raw data files the leaderboard pages load. Rows from papers and model cards are marked `AUTHOR/VENDOR-REPORTED` in `note`. Nothing was estimated. If a number wasn't found, it isn't in the file.

## Sources

| Source | Raw data used | Version / freshness | What it contains |
|---|---|---|---|
| EvalPlus leaderboard, https://evalplus.github.io/leaderboard.html | https://evalplus.github.io/results.json | file last-modified 2024-12-26 (125 models). Not updated since | HumanEval, HumanEval+, MBPP, MBPP+ pass@1, greedy. MBPP is the 378-task EvalPlus subset (v0.2) |
| BigCodeBench leaderboard, https://bigcode-bench.github.io/ | https://bigcode-bench.github.io/results.json and results-hard.json | last-modified 2025-04-16 (202 models). HF dataset bigcode/bigcodebench-results was last modified 2025-04-17, the same snapshot | Full (1,140 tasks) and Hard (148 tasks), Complete and Instruct splits, pass@1 (greedy unless the model name states a temperature) |
| LiveCodeBench leaderboard, https://livecodebench.github.io/leaderboard.html | https://livecodebench.github.io/performances_generation.json (release_v6, 28 models) and v5.json (leaderboard_v5.html, 35 models) | both last-modified 2025-08-01 | Per-problem pass@1. I recomputed the leaderboard number exactly as the page's JS does (mean over problems in the date window). Default windows: v6 2024-08-01 to 2025-05-01 (454 problems), v5 2024-08-01 to 2025-02-01 (279 problems). I also added the v6 window 2025-02-01 to 2025-05-01, because vendor cards cite '25.02-25.05' |
| CRUXEval leaderboard, https://crux-eval.github.io/leaderboard.html | https://crux-eval.github.io/data.csv | last-modified 2024-12-06 (44 entries) | CRUXEval-I and -O pass@1 (T=0.2, n=10 unless marked). Rows with +cot are chain-of-thought runs |
| Papers (arXiv) | phi-1 2306.11644, phi-1.5 2309.05463, SmolLM2 2502.02737, Qwen2.5-Coder 2409.12186, Qwen3 2505.09388, DeepSeek-Coder 2401.14196, StarCoder2 2402.19173, OpenCoder 2411.04905, CodeGen 2203.13474, Granite Code 2405.04324, Code Llama 2308.12950, Gemma 3 2503.19786 | latest arXiv PDF on retrieval date | Author-reported tables, transcribed from the PDF text |
| Model cards / blogs | HF cards: SantaCoder, stable-code-3b, Yi-Coder-9B-Chat, Gemma-4-E2B/E4B-it, Qwen3.5-4B, Qwen3.5-397B-A17B, Qwen3.8-27B, Qwen3.8-Flash-Next, Qwen3-235B-A22B-Thinking/Instruct-2507, DeepSeek-V3.1, DeepSeek-R1-0528, DeepSeek-V4.1-Flash, Kimi-K2-Instruct. Also the Yi-Coder GitHub README, the Phi-2 MSR blog and the Codestral 25.01 blog | retrieved 2026-10-21 | Vendor-reported. Competitor numbers in these cards were run by the vendor, not by the competitor |

**The leaderboards are stale.** None of the four official leaderboards has data newer than August 2025. The models from 2025-26 (GPT-5.x, Claude Opus 4.5/4.6, Gemini 3, DeepSeek-V4, Qwen3.5-3.8, Kimi K2.5) appear only through vendor model cards, and only on LiveCodeBench v6. The cards mostly don't state the window, so those numbers are not comparable to the leaderboard's own.

## Frontier: top entries per benchmark (leaderboard data)

Here 'frontier' means closed API models plus open flagship models of 200B or more parameters.

### EvalPlus (sorted by HumanEval+)
| Model | Size (B) | Kind | HE | HE+ | MBPP | MBPP+ |
|---|---|---|---|---|---|---|
| O1 Preview (Sept 2024) |  | reasoning | 96.3 | 89 | 95.5 | 80.2 |
| O1 Mini (Sept 2024) |  | reasoning | 96.3 | 89 | 93.1 | 78.8 |
| GPT 4o (Aug 2024) |  | instruct | 92.7 | 87.2 | 87.6 | 72.2 |
| DeepSeek-V3 (Nov 2024) | 671 | instruct | 91.5 | 86.6 | 87.6 | 73 |
| GPT-4-Turbo (April 2024) |  | instruct | 90.2 | 86.6 |  |  |
| DeepSeek-V2.5 (Nov 2024) | 236 | instruct | 90.2 | 83.5 | 87.6 | 74.1 |
| GPT 4o Mini (July 2024) |  | instruct | 88.4 | 83.5 | 85.4 | 72.2 |
| DeepSeek-Coder-V2-Instruct | 236 | instruct | 85.4 | 82.3 | 89.4 | 75.1 |
| Claude Sonnet 3.5 (June 2024) |  | instruct | 87.2 | 81.7 | 89.4 | 74.3 |
| GPT-4-Turbo (Nov 2023) |  | instruct | 85.4 | 81.7 | 85.7 | 73.3 |
| Grok Beta |  | instruct | 88.4 | 80.5 | 86 | 65.6 |
| Gemini 1.5 Pro 002 |  | instruct | 89 | 79.3 | 89.7 | 74.6 |
| GPT-4 (May 2023) |  | instruct | 88.4 | 79.3 |  |  |
| claude-3-opus (Mar 2024) |  | instruct | 82.9 | 77.4 | 89.4 | 73.3 |
| Gemini 1.5 Flash 002 |  | instruct | 82.3 | 75.6 | 84.7 | 67.5 |

### BigCodeBench (sorted by Hard-Complete)
| Model | Size (B) | Kind | Hard-Complete | Hard-Instruct | Full-Complete | Full-Instruct |
|---|---|---|---|---|---|---|
| Gemini-Exp-1206 |  | instruct | 40.5 | 27.7 | 62.4 | 47 |
| DeepSeek-V3 | 671 | instruct | 40.5 | 28.4 | 62.2 | 50 |
| DeepSeek-R1 | 671 | reasoning | 40.5 | 29.7 |  |  |
| o1-2024-12-17 (temperature=1, reasoning=low) |  | reasoning | 39.2 | 29.7 |  |  |
| Claude-3.7-Sonnet-20250219 (temperature=1, length=12800, reasoning=3200) |  | reasoning | 39.2 | 32.4 |  |  |
| Grok-3-Beta |  | instruct | 39.2 | 27 |  |  |
| o1-2024-12-17 (temperature=1, reasoning=high) |  | reasoning | 38.5 | 32.4 |  |  |
| Quasar-Alpha |  | instruct | 37.8 | 31.8 | 60.6 | 49.6 |
| o3-mini-2025-01-31 (temperature=1, reasoning=medium) |  | reasoning | 37.8 | 33.1 |  |  |
| o3-mini-2025-01-31 (temperature=1, reasoning=low) |  | reasoning | 37.8 | 31.1 |  |  |
| o3-mini-2025-01-31 (temperature=1, reasoning=high) |  | reasoning | 37.8 | 32.4 |  |  |
| o1-2024-12-17 (temperature=1, reasoning=medium) |  | reasoning | 37.2 | 28.4 |  |  |
| GPT-4o-2024-08-06 |  | instruct | 36.5 | 25 |  |  |
| Gemini-2.0-Pro-Exp-02-05 |  | instruct | 36.5 | 27 |  |  |
| Gemini-2.5-Pro-Exp-03-25 |  | reasoning | 36.5 | 29.7 |  |  |

Most models from 2025 have only Hard-subset results. Full is null for them in the data file.

### LiveCodeBench (leaderboard data, all groups, sorted by v6 default window)

**Cross-check.** My recomputed 2025-02..2025-05 v6 window reproduces the numbers on the Qwen3-235B-A22B-Thinking-2507 card exactly: DeepSeek-R1-0528 68.7, o4-mini 71.8, Gemini-2.5-Pro-06-05 72.5, Claude-Opus-4 (Thinking) 48.9, Qwen3-235B-A22B 55.7. Qwen evidently took those numbers from this leaderboard data. The one mismatch is o3: Qwen reports 58.6, but the leaderboard has only O3 (High) at 71.8.

| Model | Size (B) | Kind | v6 2024-08..2025-05 | v6 2025-02..2025-05 | v5 2024-08..2025-02 |
|---|---|---|---|---|---|
| O4-Mini (High) |  | reasoning | 80.2 | 77.1 | 74.6 |
| O3 (High) |  | reasoning | 75.8 | 71.8 |  |
| O4-Mini (Medium) |  | reasoning | 74.2 | 71.8 |  |
| Gemini-2.5-Pro-06-05 |  | reasoning | 73.6 | 72.5 |  |
| DeepSeek-R1-0528 | 671 | reasoning | 73.1 | 68.7 |  |
| Gemini-2.5-Pro-05-06 |  | reasoning | 71.8 | 67.2 |  |
| EXAONE-4.0-32B | 32 | reasoning | 70 | 66 |  |
| OpenReasoning-Nemotron-32B | 32 | reasoning | 69.8 | 65.6 |  |
| O3-Mini-2025-01-31 (High) |  | reasoning | 67.4 | 62.6 | 69.5 |
| OpenCodeReasoning-Nemotron-1.1-32B | 32 | reasoning | 66.8 | 61.4 |  |
| Grok-3-Mini (High) |  | reasoning | 66.7 | 62.6 | 69.9 |
| O4-Mini (Low) |  | reasoning | 65.9 | 62.6 |  |
| Qwen3-235B-A22B | 235 | unknown | 65.9 | 55.7 |  |
| XBai-o4-medium | 32.8 | reasoning | 65 | 59.5 |  |
| O3-Mini-2025-01-31 (Med) |  | reasoning | 63 | 56.5 | 67.4 |
| Gemini-2.5-Flash-05-20 |  | reasoning | 61.9 | 59.5 |  |
| Gemini-2.5-Flash-04-17 |  | reasoning | 60.6 | 56.5 |  |
| O3-Mini-2025-01-31 (Low) |  | reasoning | 57 | 51.1 | 60.9 |
| Claude-Opus-4 (Thinking) |  | reasoning | 56.6 | 48.9 |  |
| Claude-Sonnet-4 (Thinking) |  | reasoning | 55.9 | 51.1 |  |

v5-only models (sorted by v5 default window):

| Model | Size (B) | Kind | v6 2024-08..2025-05 | v6 2025-02..2025-05 | v5 2024-08..2025-02 |
|---|---|---|---|---|---|
| Kimi-k1.6-IOI-high |  | reasoning |  |  | 73.8 |
| O1-2024-12-17 (High) |  | reasoning |  |  | 71 |
| Gemini-2.5-Pro |  | reasoning |  |  | 69.2 |
| Kimi-k1.6-IOI |  | reasoning |  |  | 65.9 |
| QwQ-Max-Preview |  | reasoning |  |  | 65.6 |
| Llama-3_1-Nemotron-Ultra-253B-v1 | 253 | reasoning |  |  | 64.5 |
| O1-2024-12-17 (Med) |  | reasoning |  |  | 63.4 |
| DeepSeek-R1-Preview | 671 | reasoning |  |  | 62.8 |
| DeepCoder-14B-Preview | 14 | reasoning |  |  | 60.6 |
| O1-2024-12-17 (Low) |  | reasoning |  |  | 59.5 |
| O1-Mini-2024-09-12 |  | reasoning |  |  | 52 |
| Llama-3_1-Nemotron-Nano-8B-v1 | 8 | reasoning |  |  | 51.3 |
| DeepSeek-R1-Lite-Preview |  | reasoning |  |  | 48.1 |
| O1-Preview-2024-09-12 |  | reasoning |  |  | 42.7 |
| QwQ-32B-Preview | 32 | reasoning |  |  | 42.7 |

### CRUXEval (leaderboard, closed models only, sorted by -O)
| Model | Size (B) | Kind | CRUXEval-I | CRUXEval-O |
|---|---|---|---|---|
| gpt-4-turbo-2024-04-09+cot (n=3) |  | instruct | 75.7 | 82 |
| claude-3-opus+cot (n=1) |  | instruct | 73.4 | 82 |
| gpt-4-0613+cot |  | instruct | 75.5 | 77.1 |
| gpt-4o+cot (n=3) |  | instruct | 75.6 | 76 |
| gpt-4o (n=3) |  | instruct | 65.1 | 70 |
| gpt-4-0613 |  | instruct | 69.8 | 68.7 |
| gpt-4-turbo-2024-04-09 (n=3) |  | instruct | 68.5 | 67.7 |
| claude-3-opus (n=1) |  | instruct | 64.2 | 65.8 |
| gpt-3.5-turbo-0613+cot |  | instruct | 50.3 | 59 |
| gpt-3.5-turbo-0613 |  | instruct | 49 | 49.4 |

### Recent frontier models (vendor model cards, LiveCodeBench v6)

| Model | Score | Reported by | Window |
|---|---|---|---|
| GPT-5.2 | 87.7 | Qwen/Qwen3.5-397B-A17B | window not stated |
| Claude 4.5 Opus | 84.8 | Qwen/Qwen3.5-397B-A17B | window not stated |
| Gemini-3 Pro | 90.7 | Qwen/Qwen3.5-397B-A17B | window not stated |
| Qwen3-Max-Thinking | 85.9 | Qwen/Qwen3.5-397B-A17B | window not stated |
| Kimi K2.5 (1T-A32B) | 85 | Qwen/Qwen3.5-397B-A17B | window not stated |
| Qwen3.5-397B-A17B | 83.6 | Qwen/Qwen3.5-397B-A17B | window not stated |
| Qwen3.7-Plus | 89.6 | Qwen/Qwen3.8-27B | window not stated |
| Claude Opus 4.6 (Max) | 88.8 | Qwen/Qwen3.8-27B | window not stated |
| Qwen3.8-Flash-Next | 91.9 | Qwen/Qwen3.8-Flash-Next | window not stated |
| DeepSeek-V4-Flash-0731 | 90.6 | Qwen/Qwen3.8-Flash-Next | window not stated |
| DeepSeek-R1-0528 | 68.7 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| OpenAI o4-mini | 71.8 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| OpenAI o3 | 58.6 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| Gemini-2.5 Pro | 72.5 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| Claude 4 Opus Thinking | 48.9 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| Qwen3-235B-A22B (thinking) | 55.7 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| Qwen3-235B-A22B-Thinking-2507 | 74.1 | Qwen/Qwen3-235B-A22B-Thinking-2507 | 25.02-25.05 |
| DeepSeek-V3-0324 | 45.2 | Qwen/Qwen3-235B-A22B-Instruct-2507 | 25.02-25.05 |
| GPT-4o-0327 | 35.8 | Qwen/Qwen3-235B-A22B-Instruct-2507 | 25.02-25.05 |
| Claude Opus 4 (non-thinking) | 44.6 | Qwen/Qwen3-235B-A22B-Instruct-2507 | 25.02-25.05 |
| Kimi K2 | 48.9 | Qwen/Qwen3-235B-A22B-Instruct-2507 | 25.02-25.05 |
| Qwen3-235B-A22B (non-thinking) | 32.9 | Qwen/Qwen3-235B-A22B-Instruct-2507 | 25.02-25.05 |
| Qwen3-235B-A22B-Instruct-2507 | 51.8 | Qwen/Qwen3-235B-A22B-Instruct-2507 | 25.02-25.05 |
| Kimi K2 Instruct | 53.7 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| DeepSeek-V3-0324 | 46.9 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| Qwen3-235B-A22B (non-thinking) | 37 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| Claude Sonnet 4 (no ext. thinking) | 48.5 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| Claude Opus 4 (no ext. thinking) | 47.4 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| GPT-4.1 | 44.7 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| Gemini 2.5 Flash Preview (05-20) | 44.7 | moonshotai/Kimi-K2-Instruct | Aug 24 - May 25) pass@1 |
| Kimi K2 Base | 26.3 | moonshotai/Kimi-K2-Instruct | , 1-shot, pass@1 (base model |
| DeepSeek-V3-Base | 22.9 | moonshotai/Kimi-K2-Instruct | , 1-shot, pass@1 (base model |
| Llama 4 Maverick (base) | 25.1 | moonshotai/Kimi-K2-Instruct | , 1-shot, pass@1 (base model |

## Small models (10B or fewer): every number found

### EvalPlus leaderboard (pass@1 greedy)
| Model | Size (B) | Kind | HE | HE+ | MBPP | MBPP+ |
|---|---|---|---|---|---|---|
| CodeQwen1.5-7B-Chat | 7 | instruct | 83.5 | 78.7 | 79.4 | 69 |
| OpenCoder-8B-Instruct | 8 | instruct | 81.7 | 77.4 | 82 | 71.4 |
| Artigenz-Coder-DS-6.7B | 6.7 | instruct | 75.6 | 72.6 | 80.7 | 69.6 |
| OpenCodeInterpreter-DS-6.7B | 6.7 | instruct | 77.4 | 72 | 76.5 | 66.4 |
| DeepSeek-Coder-6.7B-instruct | 6.7 | instruct | 74.4 | 71.3 | 74.9 | 65.6 |
| DeepSeek-Coder-7B-instruct-v1.5 | 7 | instruct | 75.6 | 71.3 | 75.2 | 62.2 |
| Magicoder-S-DS-6.7B | 6.7 | instruct | 76.8 | 71.3 | 79.4 | 69 |
| WaveCoder-Ultra-6.7B | 7 | instruct | 75 | 69.5 | 74.9 | 63.5 |
| Magicoder-S-CL-7B | 7 | instruct | 70.7 | 67.7 | 70.6 | 60.1 |
| OpenChat-3.5-7B-0106 | 7 | instruct | 72.6 | 67.7 | 63.8 | 54.5 |
| speechless-coder-ds-6.7B | 6.7 | instruct | 71.3 | 65.9 | 75.9 | 64.4 |
| Llama3.1-8B-instruct | 8 | instruct | 69.5 | 62.8 | 68.3 | 55.6 |
| DeepSeek-Coder-1.3B-instruct | 1.3 | instruct | 65.9 | 60.4 | 65.3 | 54.8 |
| Code-290k-6.7B-Instruct | 6.7 | instruct | 64.6 | 59.7 |  |  |
| Phi-3-mini-4k-instruct | 3.8 | instruct | 64.6 | 59.1 | 65.9 | 54.2 |
| Llama3-8B-instruct | 8 | instruct | 61.6 | 56.7 | 64.6 | 54.8 |
| codegemma-7b-it | 7 | instruct | 60.4 | 51.8 | 70.4 | 56.9 |
| speechless-starcoder2-7b | 7 | instruct | 56.1 | 51.8 | 66.7 | 56.3 |
| speechless-coding-7B-16k-tora | 7 | instruct | 54.9 | 50.6 | 64.2 | 50.6 |
| CodeQwen1.5-7B | 7 | base | 51.8 | 45.7 | 73.5 | 60.8 |
| WizardCoder-Python-7B-V1.0 | 7 | instruct | 50.6 | 45.1 | 58.5 | 49.5 |
| phi-2-2.7B | 2.7 | base | 49.4 | 45.1 | 64 | 54.2 |
| Mistral-codealpaca-7B | 7 | base | 48.2 | 42.1 |  |  |
| MistralHermes-CodePro-7B-v1 | 7 | instruct | 47.6 | 42.1 | 57.4 | 46.4 |
| codegemma-7b | 7 | base | 44.5 | 41.5 | 65.1 | 52.4 |
| speechless-code-mistral-7B-v1.0 | 7 | instruct | 48.2 | 41.5 | 57.4 | 48.7 |
| DeepSeek-Coder-6.7B-base | 6.7 | base | 47.6 | 39.6 | 72 | 58.7 |
| Mistral-7B-Instruct-v0.2 | 7 | instruct | 42.1 | 36 | 44.7 | 37 |
| CodeLlama-7B | 7 | base | 37.8 | 35.4 | 59.5 | 46.8 |
| gemma-1.1-7b-it | 7 | instruct | 42.7 | 35.4 | 57.1 | 45 |
| xDAN-L1-Chat-RL-v1-7B | 7 | instruct | 40.2 | 32.9 | 50.3 | 41.3 |
| StarCoder2-7B | 7 | base | 35.4 | 29.9 |  |  |
| Llama3-8B-base | 8 | base | 33.5 | 29.3 | 61.4 | 51.6 |
| gemma-7b | 7 | base | 35.4 | 28.7 | 52.6 | 43.4 |
| StarCoder2-3B | 3 | base | 31.7 | 27.4 |  |  |
| CodeGen-6B | 6 | base | 29.3 | 25.6 | 50.8 | 42.9 |
| DeepSeek-Coder-1.3B-base | 1.3 | base | 28.7 | 25.6 | 56.9 | 47.9 |
| stable-code-3B | 3 | base | 29.3 | 25.6 | 54.8 | 45.8 |
| gemma-7b-it | 7 | instruct | 28.7 | 25 | 47.1 | 36.8 |
| CodeT5+-6B | 6 | base | 29.3 | 24.4 | 52.9 | 41.5 |
| Mistral-7B | 7 | base | 28.7 | 23.8 | 51.9 | 42.1 |
| Zephyr β-7B | 7 | base | 30 | 23.2 | 42.1 | 34.7 |
| CodeGen-2B | 2 | base | 24.4 | 22.6 | 46.3 | 36 |
| CodeT5+-2B | 2 | base | 25 | 22 | 48.4 | 38.1 |
| StarCoderBase-7B | 7 | base | 24.4 | 21.3 |  |  |
| codegemma-2b | 2 | base | 26.8 | 20.7 | 55.6 | 46.6 |
| gemma-2b | 2 | base | 25 | 20.7 | 41.8 | 34.1 |
| CodeGen2-7B | 7 | base | 18.3 | 17.7 |  |  |
| gemma-1.1-2b-it | 2 | instruct | 22.6 | 17.7 | 29.8 | 23.3 |
| StarCoderBase-3B | 3 | base | 17.7 | 15.9 |  |  |
| gemma-2b-it | 2 | instruct | 17.7 | 15.2 |  |  |
| SantaCoder-1.1B | 1.1 | base | 14.6 | 14 |  |  |
| CodeGen2-3B | 3 | base | 15.9 | 12.8 |  |  |
| InCoder-6.7B | 6.7 | base | 15.9 | 12.2 |  |  |
| StarCoderBase-1B | 1 | base | 14.6 | 12.2 |  |  |
| Vicuna-7B | 7 | base | 11.6 | 11.6 |  |  |
| GPT-J-6B | 6 | base | 12.2 | 11 |  |  |
| InCoder-1.3B | 1.3 | base | 12.2 | 11 |  |  |
| CodeGen2-1B | 1 | base | 11 | 9.1 |  |  |
| GPT-Neo-2.7B | 2.7 | base | 7.9 | 6.7 |  |  |
| PolyCoder-2.7B | 2.7 | base | 6.1 | 6.1 |  |  |
| StableLM-7B | 7 | base | 2.4 | 2.4 |  |  |
| zyte-1B | 1 | instruct | 2.4 | 1.8 | 9.8 | 7.5 |

### BigCodeBench leaderboard (pass@1)
| Model | Size (B) | Kind | Hard-Complete | Hard-Instruct | Full-Complete | Full-Instruct |
|---|---|---|---|---|---|---|
| OpenCoder-8B-Instruct | 8 | instruct | 18.9 | 18.2 | 50.9 | 43.2 |
| CodeGeex4-All-9B | 9 | instruct | 15.5 | 18.9 | 49 | 40 |
| Yi-Coder-9B-Chat | 9 | instruct | 17.6 | 11.5 | 49 | 38.1 |
| Qwen2.5-Coder-7B-Instruct | 7 | instruct | 20.3 | 20.3 | 48.8 | 40.4 |
| ReflectionCoder-DS-6.7B | 6.7 | instruct | 18.9 | 11.5 | 47.8 | 36.8 |
| Magicoder-S-DS-6.7B | 6.7 | instruct | 12.8 | 13.5 | 47.6 | 36.2 |
| Artigenz-Coder-DS-6.7B | 6.7 | instruct | 16.2 | 12.2 | 47.1 | 38 |
| AutoCoder-S-6.7B | 6.7 | instruct | 12.2 | 8.8 | 46.5 | 36.1 |
| Phi-3.1-Mini-128K-Instruct | 3.8 | instruct | 26.4 | 17.6 | 46.2 | 36.8 |
| Qwen2.5-7B-Instruct | 7 | instruct | 16.2 | 12.2 | 46.1 | 37.6 |
| CodeQwen1.5-7B | 7 | base | 15.5 |  | 45.6 |  |
| OpenCodeInterpreter-DS-6.7B | 6.7 | instruct | 16.9 | 13.5 | 44.6 | 37.1 |
| Codestral-Mamba | 7 | instruct | 14.9 | 12.8 | 44.6 | 35.9 |
| Nxcode-CQ-7B-Orpo | 7 | instruct | 16.2 | 18.2 | 44.6 | 39.6 |
| DeepSeek-Coder-6.7B-Instruct | 6.7 | instruct | 15.5 | 10.1 | 43.8 | 35.5 |
| WaveCoder-Ultra-6.7B | 6.7 | instruct | 16.9 | 12.8 | 43.7 | 33.9 |
| CodeQwen1.5-7B-Chat | 7 | instruct | 15.5 | 18.9 | 43.6 | 39.6 |
| Yi-1.5-9B-Chat | 9 | instruct | 12.2 | 10.1 | 42.4 | 34.5 |
| Qwen2-7B-Instruct | 7 | instruct | 13.5 | 10.1 | 42.1 | 29.1 |
| DeepSeek-Coder-6.7B-Base | 6.7 | base | 13.5 |  | 41.8 |  |
| AutoCoder-QW-7B | 7 | instruct | 12.8 | 8.8 | 41.8 | 32.2 |
| Gemma-2-9B-Instruct | 9 | instruct | 10.1 | 10.1 | 40.6 | 34.7 |
| Phi-3-Mini-128K-Instruct | 3.8 | instruct | 14.2 | 7.4 | 40.6 | 29.6 |
| ReflectionCoder-CL-7B | 7 | instruct | 8.8 | 10.1 | 40.5 | 28.4 |
| Llama-3.1-8B-Instruct | 8 | instruct | 13.5 | 13.5 | 40.5 | 32.8 |
| Granite-Code-8B-Instruct | 8 | instruct | 8.1 |  | 39.7 |  |
| Phi-3-Small-128K-Instruct | 7 | instruct | 14.2 | 10.8 | 39.4 | 31.1 |
| CodeGemma-7B-Instruct | 7 | instruct | 13.5 | 7.4 | 39.3 | 32.3 |
| OpenCoder-1.5B-Instruct | 1.5 | instruct | 8.8 | 9.5 | 38.8 | 34.9 |
| Phi-3.5-Mini-Instruct | 3.8 | instruct | 12.8 | 16.9 | 38.5 | 32.8 |
| CodeGemma-7B | 7 | base | 10.1 |  | 38.3 |  |
| OpenChat-3.6-8B-20240522 | 8 | instruct | 14.2 | 8.8 | 37.1 | 28.1 |
| Llama-3-8B-Instruct | 8 | instruct | 8.1 | 6.8 | 36.9 | 31.9 |
| Granite-Code-8B-Base | 8 | base | 8.1 |  | 35.7 |  |
| Yi-1.5-9B | 9 | base | 8.8 |  | 35.5 |  |
| Granite-3.0-8B-Instruct | 8 | instruct | 9.5 | 7.4 | 35.4 | 29.3 |
| Yi-1.5-6B-Chat | 6 | instruct | 6.8 | 4.1 | 33.9 | 25.6 |
| InternLM2.5-7B-Chat | 7 | instruct | 7.4 | 3.4 | 33.1 | 25.8 |
| Qwen2.5-Coder-1.5B-Instruct | 1.5 | instruct | 6.1 | 5.4 | 32.7 | 27 |
| Granite-Code-3B-Instruct | 3 | instruct | 5.4 |  | 31.4 |  |
| OpenCodeInterpreter-DS-1.3B | 1.3 | instruct | 3.4 | 5.4 | 29.8 | 25.3 |
| DeepSeek-Coder-1.3B-Instruct | 1.3 | instruct | 2 | 3.4 | 29.6 | 22.8 |
| Llama-3-8B-Base | 8 | base | 6.1 |  | 28.8 |  |
| CodeLlama-7B-Base | 7 | base | 5.4 |  | 28.7 |  |
| Llama-3.2-3B-Instruct | 3 | instruct | 5.4 | 5.4 | 28.3 | 23.4 |
| StarCoder2-7B | 7 | base | 8.8 |  | 27.7 |  |
| Yi-1.5-6B | 6 | base | 4.1 |  | 27.5 |  |
| Mistral-7B-Instruct-v0.3 | 7 | instruct | 6.1 | 4.7 | 27.3 | 19.5 |
| Qwen2.5-1.5B-Instruct | 1.5 | instruct | 4.7 | 1.4 | 27.1 | 20.3 |
| CodeLlama-7B-Instruct | 7 | instruct | 4.1 | 3.4 | 25.7 | 21.9 |
| DeepSeek-R1-Distill-Qwen-7B | 7 | reasoning | 6.1 | 3.4 | 24.9 | 17.5 |
| CodeGemma-2B | 2 | base | 7.4 |  | 23.9 |  |
| Mistral-7B-v0.3 | 7 | base | 4.7 |  | 23.5 |  |
| DeepSeek-Coder-1.3B-Base | 1.3 | base | 3.4 |  | 22.2 |  |
| StarCoder2-3B | 3 | base | 4.7 |  | 21.4 |  |
| Granite-Code-3B-Base | 3 | base | 1.4 |  | 20 |  |
| DeepSeek-R1-Distill-Llama-8B | 8 | reasoning | 3.4 | 2 | 15.3 | 10.6 |
| Qwen2.5-0.5B-Instruct | 0.5 | instruct | 2 | 0.7 | 11.3 | 8.8 |
| Llama-3.2-1B-Instruct | 1 | instruct | 1.4 | 0.7 | 11.3 | 8.2 |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5 | reasoning |  | 0 | 7.9 | 7 |
| Granite-3.0-2B-Instruct | 2 | instruct |  | 2 |  | 20.5 |
| WarriorCoder-6.7B (Reproduced) | 6.7 | instruct |  | 14.2 |  | 39.9 |
| InverseCoder-CL-7B | 7 | instruct | 12.8 | 14.2 |  |  |
| InverseCoder-DS-6.7B | 6.7 | instruct | 14.9 | 10.8 |  |  |

### CRUXEval leaderboard (pass@1, T=0.2)
| Model | Size (B) | Kind | CRUXEval-I | CRUXEval-O |
|---|---|---|---|---|
| semcoder-6.7b+cot (under verification) | 6.7 | instruct | 62.5 | 64.4 |
| semcoder-s-6.7b+cot (under verification) | 6.7 | instruct | 63.1 | 64.1 |
| magicoder-ds-6.7b | 6.7 | instruct | 41.7 | 44.4 |
| deepseek-base-6.7b | 6.7 | base | 41.9 | 43.5 |
| deepseek-instruct-6.7b | 6.7 | instruct | 37.4 | 41.2 |
| starcoder2-7b | 7 | base | 34.6 | 36 |
| codellama-python-7b | 7 | base | 37.3 | 35.9 |
| mistral-7b | 7 | base | 35 | 34.3 |
| codellama-7b | 7 | base | 35.9 | 34.2 |
| starcoder2-3b | 3 | base | 32.7 | 34.2 |
| phi-2 | 2.7 | base | 31.6 | 33.5 |
| starcoderbase-7b | 7 | base | 29.7 | 32.2 |
| deepseek-base-1.3b | 1.3 | base | 27.8 | 31 |
| codellama-7b+cot | 7 | base | 40.4 | 29.9 |
| deepseek-instruct-1.3b | 1.3 | instruct | 27.2 | 28.7 |
| phi-1.5 | 1.3 | base | 23.2 | 27.5 |
| stablecode-3b | 2.7 | base | 33.5 | 26.7 |
| phi-1 | 1.3 | base | 13.1 | 21.7 |

### LiveCodeBench leaderboard
| Model | Size (B) | Kind | v6 2024-08..2025-05 | v6 2025-02..2025-05 | v5 2024-08..2025-02 |
|---|---|---|---|---|---|
| Llama-3_1-Nemotron-Nano-8B-v1 | 8 | reasoning |  |  | 51.3 |
| MetaStone-L1-7B | 7 | reasoning |  |  | 39.3 |

### Author-reported (papers, cards): small models

Each row is one model in one source. Full variant names and notes are in the JSON.

| Model | Size (B) | Kind | Source | Scores |
|---|---|---|---|---|
| phi-1 | 1.3 | base | https://arxiv.org/abs/2306.11644 | HumanEval pass@1 (original tests): **50.6**; MBPP pass@1: **55.5** |
| phi-1-small | 0.35 | base | https://arxiv.org/abs/2306.11644 | HumanEval pass@1 (original tests): **45** |
| phi-1-base | 1.3 | base | https://arxiv.org/abs/2306.11644 | HumanEval pass@1 (original tests): **29** |
| CodeGen-Mono-350M | 0.35 | base | https://arxiv.org/abs/2306.11644 | HumanEval pass@1 (as cited in phi-1 Table 2, rounded): **13** |
| phi-1.5 | 1.3 | base | https://arxiv.org/abs/2309.05463 | HumanEval pass@1: **34.1**; MBPP pass@1: **37.7** |
| phi-1.5-web | 1.3 | base | https://arxiv.org/abs/2309.05463 | HumanEval pass@1: **41.4**; MBPP pass@1: **43.5** |
| phi-2 | 2.7 | base | https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/ | HumanEval+MBPP Average of HumanEval and MBPP (3-shot): **53.7** |
| SmolLM2-1.7B | 1.7 | base | https://arxiv.org/abs/2502.02737 | HumanEval pass@1: **22.6** |
| Llama-3.2-1B | 1.23 | base | https://arxiv.org/abs/2502.02737 | HumanEval pass@1: **18.9** |
| Qwen2.5-1.5B | 1.5 | base | https://arxiv.org/abs/2502.02737 | HumanEval pass@1: **37.2** |
| SmolLM2-1.7B-Instruct | 1.7 | instruct | https://arxiv.org/abs/2502.02737 | HumanEval pass@1: **28.1** |
| Llama-3.2-1B-Instruct | 1.23 | instruct | https://arxiv.org/abs/2502.02737 | HumanEval pass@1: **33.5** |
| Qwen2.5-1.5B-Instruct | 1.5 | instruct | https://arxiv.org/abs/2502.02737 | HumanEval pass@1: **30.5** |
| SmolLM2-360M | 0.36 | unknown | https://pages.github.khoury.northeastern.edu/4130/2025S/hw/humaneval/ | HumanEval mean pass rate, T=0.2, 20 samples/prompt, 300 new tokens: **12** |
| Qwen2.5-Coder-0.5B | 0.5 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **28**; EvalPlus HumanEval+ pass@1: **23.8**; EvalPlus MBPP pass@1 (EvalPlus subset): **52.9**; EvalPlus MBPP+ pass@1: **47.1**; MBPP 3-shot pass@1 (original 500-task test split): **40.4**; BigCodeBench Full, Complete split, pass@1: **16.1**; BigCodeBench Hard, Complete split, pass@1: **4.7**; CRUXEval-I (Input-CoT): **35.2**; CRUXEval-O (Output-CoT): **23** |
| DeepSeek-Coder-1.3B-Base | 1.3 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **34.8**; EvalPlus HumanEval+ pass@1: **26.8**; EvalPlus MBPP pass@1 (EvalPlus subset): **55.6**; EvalPlus MBPP+ pass@1: **46.9**; MBPP 3-shot pass@1 (original 500-task test split): **46.2**; BigCodeBench Full, Complete split, pass@1: **26.1**; BigCodeBench Hard, Complete split, pass@1: **3.4**; CRUXEval-I (Input-CoT): **32.1**; CRUXEval-O (Output-CoT): **28.2** |
| Qwen2.5-Coder-1.5B | 1.5 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **43.9**; EvalPlus HumanEval+ pass@1: **36.6**; EvalPlus MBPP pass@1 (EvalPlus subset): **69.2**; EvalPlus MBPP+ pass@1: **58.6**; MBPP 3-shot pass@1 (original 500-task test split): **59.2**; BigCodeBench Full, Complete split, pass@1: **34.6**; BigCodeBench Hard, Complete split, pass@1: **9.5**; CRUXEval-I (Input-CoT): **43.8**; CRUXEval-O (Output-CoT): **34.6** |
| StarCoder2-3B | 3 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **31.7**; EvalPlus HumanEval+ pass@1: **27.4**; EvalPlus MBPP pass@1 (EvalPlus subset): **60.2**; EvalPlus MBPP+ pass@1: **49.1**; MBPP 3-shot pass@1 (original 500-task test split): **47.4**; BigCodeBench Full, Complete split, pass@1: **21.4**; BigCodeBench Hard, Complete split, pass@1: **4.7**; CRUXEval-I (Input-CoT): **42.1**; CRUXEval-O (Output-CoT): **29.2** |
| Qwen2.5-Coder-3B | 3 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **52.4**; EvalPlus HumanEval+ pass@1: **42.7**; EvalPlus MBPP pass@1 (EvalPlus subset): **72.2**; EvalPlus MBPP+ pass@1: **61.4**; MBPP 3-shot pass@1 (original 500-task test split): **65.2**; BigCodeBench Full, Complete split, pass@1: **41.1**; BigCodeBench Hard, Complete split, pass@1: **11.5**; CRUXEval-I (Input-CoT): **46.5**; CRUXEval-O (Output-CoT): **43.8** |
| StarCoder2-7B | 7 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **35.4**; EvalPlus HumanEval+ pass@1: **29.9**; EvalPlus MBPP pass@1 (EvalPlus subset): **54.4**; EvalPlus MBPP+ pass@1: **45.6**; MBPP 3-shot pass@1 (original 500-task test split): **51.8**; BigCodeBench Full, Complete split, pass@1: **27.7**; BigCodeBench Hard, Complete split, pass@1: **8.8**; CRUXEval-I (Input-CoT): **39.5**; CRUXEval-O (Output-CoT): **35.1** |
| DeepSeek-Coder-6.7B-Base | 6.7 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **47.6**; EvalPlus HumanEval+ pass@1: **39.6**; EvalPlus MBPP pass@1 (EvalPlus subset): **70.2**; EvalPlus MBPP+ pass@1: **56.6**; MBPP 3-shot pass@1 (original 500-task test split): **60.6**; BigCodeBench Full, Complete split, pass@1: **41.1**; BigCodeBench Hard, Complete split, pass@1: **11.5**; CRUXEval-I (Input-CoT): **39**; CRUXEval-O (Output-CoT): **41** |
| Qwen2.5-Coder-7B | 7 | base | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **61.6**; EvalPlus HumanEval+ pass@1: **53**; EvalPlus MBPP pass@1 (EvalPlus subset): **76.9**; EvalPlus MBPP+ pass@1: **62.9**; MBPP 3-shot pass@1 (original 500-task test split): **68.8**; BigCodeBench Full, Complete split, pass@1: **45.8**; BigCodeBench Hard, Complete split, pass@1: **16.2**; CRUXEval-I (Input-CoT): **56.5**; CRUXEval-O (Output-CoT): **56** |
| Qwen2.5-Coder-0.5B-Instruct | 0.5 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **61.6**; EvalPlus HumanEval+ pass@1: **57.3**; EvalPlus MBPP pass@1: **52.4**; EvalPlus MBPP+ pass@1: **43.7**; BigCodeBench Full, Instruct split, pass@1: **11.1**; BigCodeBench Hard, Instruct split, pass@1: **1.4**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **2**; CRUXEval-I (Input-CoT): **33.9**; CRUXEval-O (Output-CoT): **27.8** |
| DeepSeek-Coder-1.3B-Instruct | 1.3 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **65.9**; EvalPlus HumanEval+ pass@1: **60.4**; EvalPlus MBPP pass@1: **65.3**; EvalPlus MBPP+ pass@1: **54.8**; BigCodeBench Full, Instruct split, pass@1: **22.8**; BigCodeBench Hard, Instruct split, pass@1: **3.4**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **5.1**; CRUXEval-I (Input-CoT): **12.9**; CRUXEval-O (Output-CoT): **28.1** |
| Yi-Coder-1.5B-Chat | 1.5 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **69.5**; EvalPlus HumanEval+ pass@1: **64**; EvalPlus MBPP pass@1: **65.9**; EvalPlus MBPP+ pass@1: **57.7**; BigCodeBench Full, Instruct split, pass@1: **23.8**; BigCodeBench Hard, Instruct split, pass@1: **11.5**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **4.8**; CRUXEval-I (Input-CoT): **19.9**; CRUXEval-O (Output-CoT): **24.9** |
| Qwen2.5-Coder-1.5B-Instruct | 1.5 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **70.7**; EvalPlus HumanEval+ pass@1: **66.5**; EvalPlus MBPP pass@1: **69.2**; EvalPlus MBPP+ pass@1: **59.4**; BigCodeBench Full, Instruct split, pass@1: **32.5**; BigCodeBench Hard, Instruct split, pass@1: **6.8**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **6.1**; CRUXEval-I (Input-CoT): **45.4**; CRUXEval-O (Output-CoT): **37.5** |
| Qwen2.5-Coder-3B-Instruct | 3 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **84.1**; EvalPlus HumanEval+ pass@1: **80.5**; EvalPlus MBPP pass@1: **73.6**; EvalPlus MBPP+ pass@1: **62.4**; BigCodeBench Full, Instruct split, pass@1: **35.8**; BigCodeBench Hard, Instruct split, pass@1: **14.2**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **10.8**; CRUXEval-I (Input-CoT): **53.2**; CRUXEval-O (Output-CoT): **56** |
| CodeLlama-7B-Instruct | 7 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **40.9**; EvalPlus HumanEval+ pass@1: **33.5**; EvalPlus MBPP pass@1: **54**; EvalPlus MBPP+ pass@1: **44.4**; BigCodeBench Full, Instruct split, pass@1: **21.9**; BigCodeBench Hard, Instruct split, pass@1: **3.4**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **7.1**; CRUXEval-I (Input-CoT): **36.1**; CRUXEval-O (Output-CoT): **36.2** |
| DeepSeek-Coder-6.7B-Instruct | 6.7 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **74.4**; EvalPlus HumanEval+ pass@1: **71.3**; EvalPlus MBPP pass@1: **74.9**; EvalPlus MBPP+ pass@1: **65.6**; BigCodeBench Full, Instruct split, pass@1: **35.5**; BigCodeBench Hard, Instruct split, pass@1: **10.1**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **15.5**; CRUXEval-I (Input-CoT): **42.6**; CRUXEval-O (Output-CoT): **45.1** |
| Yi-Coder-9B-Chat | 9 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **82.3**; EvalPlus HumanEval+ pass@1: **74.4**; EvalPlus MBPP pass@1: **82**; EvalPlus MBPP+ pass@1: **69**; BigCodeBench Full, Instruct split, pass@1: **38.1**; BigCodeBench Hard, Instruct split, pass@1: **11.5**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **17.2**; CRUXEval-I (Input-CoT): **47.5**; CRUXEval-O (Output-CoT): **55.6** |
| Qwen2.5-Coder-7B-Instruct | 7 | instruct | https://arxiv.org/abs/2409.12186 | EvalPlus HumanEval pass@1: **88.4**; EvalPlus HumanEval+ pass@1: **84.1**; EvalPlus MBPP pass@1: **83.5**; EvalPlus MBPP+ pass@1: **71.7**; BigCodeBench Full, Instruct split, pass@1: **41**; BigCodeBench Hard, Instruct split, pass@1: **18.2**; LiveCodeBench pass@1, window 2024.07-2024.11 per report Fig./appendix (body text says 2407-2409): **18.2**; CRUXEval-I (Input-CoT): **65.8**; CRUXEval-O (Output-CoT): **65.9** |
| Qwen3-0.6B-Base | 0.6 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **36.23**; MBPP 3-shot pass@1: **36.6**; CRUXEval-O (1-shot): **27** |
| Qwen3-1.7B-Base | 1.7 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **52.7**; MBPP 3-shot pass@1: **55.4**; CRUXEval-O (1-shot): **36.4** |
| Qwen3-4B-Base | 4 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **63.53**; MBPP 3-shot pass@1: **67**; CRUXEval-O (1-shot): **55** |
| Qwen3-8B-Base | 8 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **67.65**; MBPP 3-shot pass@1: **69.8**; CRUXEval-O (1-shot): **62** |
| Qwen2.5-0.5B (base) | 0.5 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **31.85**; MBPP 3-shot pass@1: **29.8**; CRUXEval-O (1-shot): **12.1** |
| Gemma-3-1B (base) | 1 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **8.98**; MBPP 3-shot pass@1: **9.2**; CRUXEval-O (1-shot): **3.8** |
| Qwen2.5-1.5B (base) | 1.5 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **44.8**; MBPP 3-shot pass@1: **43.6**; CRUXEval-O (1-shot): **29.6** |
| Gemma-3-4B (base) | 4 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **43.23**; MBPP 3-shot pass@1: **46.4**; CRUXEval-O (1-shot): **34** |
| Qwen2.5-3B (base) | 3 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **46.28**; MBPP 3-shot pass@1: **54.6**; CRUXEval-O (1-shot): **36.5** |
| Llama-3-8B (base) | 8 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **44.13**; MBPP 3-shot pass@1: **48.4**; CRUXEval-O (1-shot): **36.8** |
| Qwen2.5-7B (base) | 7 | base | https://arxiv.org/abs/2505.09388 | EvalPlus average of HumanEval, MBPP, HumanEval+, MBPP+ (0-shot): **62.18**; MBPP 3-shot pass@1: **63.4**; CRUXEval-O (1-shot): **48.5** |
| Qwen3-0.6B (thinking) | 0.6 | reasoning | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **12.3** |
| Qwen3-0.6B (non-thinking) | 0.6 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **3.6** |
| Qwen3-1.7B (thinking) | 1.7 | reasoning | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **33.2** |
| Qwen3-1.7B (non-thinking) | 1.7 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **11.6** |
| Qwen3-4B (thinking) | 4 | reasoning | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **54.2** |
| Qwen3-4B (non-thinking) | 4 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **21.3** |
| Qwen3-8B (thinking) | 8 | reasoning | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **57.5** |
| Qwen3-8B (non-thinking) | 8 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **22.8** |
| Gemma-3-1B-IT | 1 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **1.8** |
| Phi-4-mini | 3.8 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **10.4** |
| Qwen2.5-1.5B-Instruct | 1.5 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **5.3** |
| Qwen2.5-3B-Instruct | 3.1 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **9.2** |
| Llama-3.1-8B-Instruct | 8 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **10.8** |
| Qwen2.5-7B-Instruct | 7 | instruct | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **14.4** |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5 | reasoning | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **13.2** |
| DeepSeek-R1-Distill-Llama-8B | 8 | reasoning | https://arxiv.org/abs/2505.09388 | LiveCodeBench LCB v5, 2024.10-2025.02, pass@1: **42.5** |
| DeepSeek-Coder-1.3B-Base | 1.3 | base | https://arxiv.org/abs/2401.14196 | HumanEval (Python) pass@1 greedy: **34.8**; MBPP (500 test problems, few-shot) pass@1 greedy: **46.2** |
| DeepSeek-Coder-6.7B-Base | 6.7 | base | https://arxiv.org/abs/2401.14196 | HumanEval (Python) pass@1 greedy: **49.4**; MBPP (500 test problems, few-shot) pass@1 greedy: **60.6** |
| DeepSeek-Coder-1.3B-Instruct | 1.3 | instruct | https://arxiv.org/abs/2401.14196 | HumanEval (Python) pass@1 greedy: **65.2**; MBPP (500 test problems, few-shot) pass@1 greedy: **49.4** |
| DeepSeek-Coder-6.7B-Instruct | 6.7 | instruct | https://arxiv.org/abs/2401.14196 | HumanEval (Python) pass@1 greedy: **78.6**; MBPP (500 test problems, few-shot) pass@1 greedy: **65.4** |
| StarCoder2-3B | 3 | base | https://arxiv.org/abs/2402.19173 | EvalPlus HumanEval pass@1 greedy: **31.7**; EvalPlus HumanEval+ pass@1 greedy: **27.4**; EvalPlus MBPP pass@1 greedy (EvalPlus subset): **57.4**; EvalPlus MBPP+ pass@1 greedy: **47.4** |
| StarCoder2-7B | 7 | base | https://arxiv.org/abs/2402.19173 | EvalPlus HumanEval pass@1 greedy: **35.4**; EvalPlus HumanEval+ pass@1 greedy: **29.9**; EvalPlus MBPP pass@1 greedy (EvalPlus subset): **54.4**; EvalPlus MBPP+ pass@1 greedy: **45.6** |
| StableCode-3B | 2.7 | base | https://arxiv.org/abs/2402.19173 | EvalPlus HumanEval pass@1 greedy: **28.7**; EvalPlus HumanEval+ pass@1 greedy: **24.4**; EvalPlus MBPP pass@1 greedy (EvalPlus subset): **53.1**; EvalPlus MBPP+ pass@1 greedy: **43.1** |
| CodeLlama-7B | 7 | base | https://arxiv.org/abs/2402.19173 | EvalPlus HumanEval pass@1 greedy: **33.5**; EvalPlus HumanEval+ pass@1 greedy: **25.6**; EvalPlus MBPP pass@1 greedy (EvalPlus subset): **52.1**; EvalPlus MBPP+ pass@1 greedy: **41.6** |
| DeepSeekCoder-1.3B (base) | 1.3 | base | https://arxiv.org/abs/2402.19173 | EvalPlus HumanEval pass@1 greedy: **28.7**; EvalPlus HumanEval+ pass@1 greedy: **23.8**; EvalPlus MBPP pass@1 greedy (EvalPlus subset): **55.4**; EvalPlus MBPP+ pass@1 greedy: **46.9** |
| DeepSeekCoder-6.7B (base) | 6.7 | base | https://arxiv.org/abs/2402.19173 | EvalPlus HumanEval pass@1 greedy: **47.6**; EvalPlus HumanEval+ pass@1 greedy: **39.6**; EvalPlus MBPP pass@1 greedy (EvalPlus subset): **70.2**; EvalPlus MBPP+ pass@1 greedy: **56.6** |
| OpenCoder-1.5B-Base | 1.5 | base | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **54.3**; EvalPlus HumanEval+ pass@1: **49.4**; EvalPlus MBPP pass@1 (EvalPlus subset): **70.6**; EvalPlus MBPP+ pass@1: **58.7**; MBPP 3-shot pass@1 (500 tasks): **51.8**; BigCodeBench Full, Complete split, pass@1: **24.5**; BigCodeBench Hard, Complete split, pass@1: **5.4** |
| OpenCoder-8B-Base | 8 | base | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **66.5**; EvalPlus HumanEval+ pass@1: **63.4**; EvalPlus MBPP pass@1 (EvalPlus subset): **79.9**; EvalPlus MBPP+ pass@1: **70.4**; MBPP 3-shot pass@1 (500 tasks): **60.6**; BigCodeBench Full, Complete split, pass@1: **40.5**; BigCodeBench Hard, Complete split, pass@1: **9.5** |
| Yi-Coder-1.5B (base) | 1.5 | base | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **41.5**; EvalPlus HumanEval+ pass@1: **32.9**; EvalPlus MBPP pass@1 (EvalPlus subset): **27**; EvalPlus MBPP+ pass@1: **22.2**; MBPP 3-shot pass@1 (500 tasks): **51.6**; BigCodeBench Full, Complete split, pass@1: **23.5**; BigCodeBench Hard, Complete split, pass@1: **3.4** |
| Yi-Coder-9B (base) | 9 | base | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **53.7**; EvalPlus HumanEval+ pass@1: **46.3**; EvalPlus MBPP pass@1 (EvalPlus subset): **48.4**; EvalPlus MBPP+ pass@1: **40.7**; MBPP 3-shot pass@1 (500 tasks): **69.4**; BigCodeBench Full, Complete split, pass@1: **42.9**; BigCodeBench Hard, Complete split, pass@1: **14.2** |
| CodeLlama-7B | 7 | base | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **33.5**; EvalPlus HumanEval+ pass@1: **26.2**; EvalPlus MBPP pass@1 (EvalPlus subset): **55.3**; EvalPlus MBPP+ pass@1: **46.8**; MBPP 3-shot pass@1 (500 tasks): **41.4**; BigCodeBench Full, Complete split, pass@1: **28.7**; BigCodeBench Hard, Complete split, pass@1: **5.4** |
| OpenCoder-1.5B-Instruct | 1.5 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **72.5**; EvalPlus HumanEval+ pass@1: **67.7**; EvalPlus MBPP pass@1: **72.7**; EvalPlus MBPP+ pass@1: **61.9**; BigCodeBench Full, Instruct split, pass@1: **33.3**; BigCodeBench Hard, Instruct split, pass@1: **11.5**; LiveCodeBench LCB pass@1, 2305-2409 split: **12.8** |
| OpenCoder-8B-Instruct | 8 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **83.5**; EvalPlus HumanEval+ pass@1: **78.7**; EvalPlus MBPP pass@1: **79.1**; EvalPlus MBPP+ pass@1: **69**; BigCodeBench Full, Instruct split, pass@1: **40.3**; BigCodeBench Hard, Instruct split, pass@1: **16.9**; LiveCodeBench LCB pass@1, 2305-2409 split: **23.2** |
| Yi-Coder-1.5B-Chat | 1.5 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **67.7**; EvalPlus HumanEval+ pass@1: **63.4**; EvalPlus MBPP pass@1: **68**; EvalPlus MBPP+ pass@1: **59**; BigCodeBench Full, Instruct split, pass@1: **24**; BigCodeBench Hard, Instruct split, pass@1: **6.8**; LiveCodeBench LCB pass@1, 2305-2409 split: **11.6** |
| Yi-Coder-9B-Chat | 9 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **82.3**; EvalPlus HumanEval+ pass@1: **72.6**; EvalPlus MBPP pass@1: **81.5**; EvalPlus MBPP+ pass@1: **69.3**; BigCodeBench Full, Instruct split, pass@1: **38.1**; BigCodeBench Hard, Instruct split, pass@1: **11.5**; LiveCodeBench LCB pass@1, 2305-2409 split: **23.4** |
| DeepSeek-Coder-1.3B-Instruct | 1.3 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **65.2**; EvalPlus HumanEval+ pass@1: **61.6**; EvalPlus MBPP pass@1: **61.6**; EvalPlus MBPP+ pass@1: **52.6**; BigCodeBench Full, Instruct split, pass@1: **22.8**; BigCodeBench Hard, Instruct split, pass@1: **3.4**; LiveCodeBench LCB pass@1, 2305-2409 split: **9.3** |
| Qwen2.5-Coder-1.5B-Instruct | 1.5 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **70.7**; EvalPlus HumanEval+ pass@1: **66.5**; EvalPlus MBPP pass@1: **69.2**; EvalPlus MBPP+ pass@1: **59.4**; BigCodeBench Full, Instruct split, pass@1: **32.5**; BigCodeBench Hard, Instruct split, pass@1: **6.8**; LiveCodeBench LCB pass@1, 2305-2409 split: **15.7** |
| Qwen2.5-Coder-7B-Instruct | 7 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **88.4**; EvalPlus HumanEval+ pass@1: **84.1**; EvalPlus MBPP pass@1: **83.5**; EvalPlus MBPP+ pass@1: **71.7**; BigCodeBench Full, Instruct split, pass@1: **41**; BigCodeBench Hard, Instruct split, pass@1: **18.2**; LiveCodeBench LCB pass@1, 2305-2409 split: **37.6** |
| DeepSeek-Coder-6.7B-Instruct | 6.7 | instruct | https://arxiv.org/abs/2411.04905 | EvalPlus HumanEval pass@1: **78.6**; EvalPlus HumanEval+ pass@1: **70.7**; EvalPlus MBPP pass@1: **75.1**; EvalPlus MBPP+ pass@1: **66.1**; BigCodeBench Full, Instruct split, pass@1: **35.5**; BigCodeBench Hard, Instruct split, pass@1: **10.1**; LiveCodeBench LCB pass@1, 2305-2409 split: **20.5** |
| Yi-Coder-1.5B (base) | 1.5 | base | https://github.com/01-ai/Yi-Coder | HumanEval (Python) pass@1, DeepSeek-Coder eval codebase: **41.5** |
| Yi-Coder-9B (base) | 9 | base | https://github.com/01-ai/Yi-Coder | HumanEval (Python) pass@1, DeepSeek-Coder eval codebase: **53.7** |
| Yi-Coder-1.5B-Chat | 1.5 | instruct | https://github.com/01-ai/Yi-Coder | HumanEval (Python) pass@1, DeepSeek-Coder eval codebase: **67.7** |
| Yi-Coder-9B-Chat | 9 | instruct | https://github.com/01-ai/Yi-Coder | HumanEval (Python) pass@1, DeepSeek-Coder eval codebase: **85.4** |
| Yi-Coder-9B-Chat | 9 | instruct | https://huggingface.co/01-ai/Yi-Coder-9B-Chat | LiveCodeBench LCB pass@1 (window shown only in blog figure): **23** |
| CodeGen-350M-mono | 0.35 | base | https://arxiv.org/abs/2203.13474 | HumanEval pass@1, sampled (best of T in {0.2,0.6,0.8}): **12.76**; MBPP (sanitized) pass@1, n=100, T=0.8: **14.59** |
| SantaCoder | 1.1 | base | https://huggingface.co/bigcode/santacoder | HumanEval MultiPL-E HumanEval (Python) pass@1: **18**; MBPP MultiPL-E MBPP (Python) pass@1: **35** |
| stable-code-3b | 2.7 | base | https://huggingface.co/stabilityai/stable-code-3b | HumanEval MultiPL-E HumanEval (Python) pass@1, bigcode-evaluation-harness: **32.4** |
| Granite-3B-Code-Base | 3 | base | https://arxiv.org/abs/2405.04324 | HumanEvalSynthesize (Python) pass@1 greedy: **36.6**; EvalPlus MBPP pass@1 greedy, zero-shot: **36**; EvalPlus MBPP+ pass@1 greedy, zero-shot: **45.1** |
| Granite-3B-Code-Instruct | 3 | instruct | https://arxiv.org/abs/2405.04324 | HumanEvalSynthesize (Python) pass@1 greedy: **51.2** |
| CodeLlama-7B | 7 | base | https://arxiv.org/abs/2308.12950 | HumanEval pass@1 greedy, 0-shot: **33.5**; MBPP pass@1 greedy, 3-shot: **41.4** |
| CodeLlama-7B-Instruct | 7 | instruct | https://arxiv.org/abs/2308.12950 | HumanEval pass@1 greedy, 0-shot: **34.8**; MBPP pass@1 greedy, 3-shot: **44.4** |
| Gemma-3-1B-IT | 1 | instruct | https://arxiv.org/abs/2503.19786 | HumanEval pass@1, sampling, 0-shot: **41.5**; MBPP pass@1, sampling, 3-shot: **35.2**; LiveCodeBench LCB, average over 8 samples (window not stated): **1.9** |
| Gemma-4-E2B-it | 2.3 | reasoning | https://huggingface.co/google/gemma-4-E2B-it | LiveCodeBench v6 (window not stated): **44** |
| Gemma-4-E4B-it | 4.5 | reasoning | https://huggingface.co/google/gemma-4-E4B-it | LiveCodeBench v6 (window not stated): **52** |
| Qwen3.5-4B | 4 | reasoning | https://huggingface.co/Qwen/Qwen3.5-4B | LiveCodeBench v6 (window not stated): **55.8** |
| Qwen3.5-9B | 9 | reasoning | https://huggingface.co/Qwen/Qwen3.5-4B | LiveCodeBench v6 (window not stated): **65.6** |
| DeepSeek-R1-0528-Qwen3-8B | 8 | reasoning | https://huggingface.co/deepseek-ai/DeepSeek-R1-0528 | LiveCodeBench LCB 2408-2505 pass@1: **60.5** |

## Gaps: requested models with no published number found

- **phi-1 / phi-1-small**: neither appears on any leaderboard. phi-1-small was never released. The only numbers are author-reported HumanEval (50.6 / 45) and MBPP (phi-1, 55.5). CRUXEval has phi-1 (13.1 / 21.7). There is no HumanEval+, MBPP+, BigCodeBench or LiveCodeBench for either.
- **phi-1.5**: author-reported HumanEval 34.1 / MBPP 37.7, plus CRUXEval on the leaderboard. Not on EvalPlus, BigCodeBench or LCB.
- **SmolLM2 135M / 360M**: no author-reported code numbers; the cards and paper report HumanEval only for 1.7B. The only figure is a third-party ~12% HumanEval for 360M from course material. Not on any leaderboard.
- **Qwen2.5-Coder base (0.5B-7B)**: not on the EvalPlus or BigCodeBench leaderboards. Only the instruct 1.5B/7B are on BigCodeBench, and EvalPlus has only the 32B-Instruct. The author report covers HE/HE+/MBPP/MBPP+/BCB-Complete/CRUXEval-CoT for base models, and HE/HE+/MBPP/MBPP+/BCB-Instruct/LCB/CRUXEval-CoT for instruct.
- **Qwen3 0.6B-8B**: on no leaderboard. The Qwen3 report gives only an EvalPlus average (no per-subset HE+/MBPP+), MBPP 3-shot and CRUX-O 1-shot for base models, plus LCB v5 for post-trained models.
- **CodeGen-350M-mono**: not on EvalPlus (only 2B/6B/16B CodeGen are). The only numbers are paper HumanEval 12.76 and MBPP 14.59, both from sampling.
- **Llama-3.2-1B/3B, Gemma-3-1B**: no Meta-reported code numbers. Llama-3.2-1B/3B-Instruct are on BigCodeBench, Gemma-3-1B only via its report and the Qwen3 report. None is on EvalPlus.
- **LiveCodeBench for small models**: the official leaderboard has no small non-reasoning models. All small-model LCB numbers are author-reported, on windows that differ from one another.
- **OpenCoder-1.5B base on the leaderboards**: only OpenCoder-8B-Instruct (EvalPlus, BCB) and 1.5B-Instruct (BCB) are there.
- **Frontier 2025-26 models**: none on EvalPlus, CRUXEval or BigCodeBench, and none newer than Aug 2025 on the LCB leaderboard. GPT-5.x, Claude 4.5/4.6 and Gemini 3 numbers exist only as competitor-reported LCB v6 in Qwen cards.

## Caveats (non-comparability)

- **Base vs instruct prompting.** EvalPlus and BigCodeBench run base models in completion mode and instruct models with a chat template ('prompted' flag). Compare pycoder base against base rows and pycoder SFT against instruct rows.
- **MBPP has at least three versions.** There is the original 500-task test split (often 3-shot: DeepSeek-Coder, Code Llama, Qwen 'MBPP 3-shot', Gemma, Qwen3 'MBPP'), the EvalPlus 378-task sanitized subset (EvalPlus MBPP/MBPP+, StarCoder2, Qwen2.5-Coder, OpenCoder), and the sanitized MBPP with sampling (CodeGen). Numbers from different versions are not comparable.
- **Greedy vs sampled pass@1.** EvalPlus, StarCoder2, Code Llama, DeepSeek-Coder and Granite use greedy decoding. CodeGen reports the best of T in {0.2, 0.6, 0.8} for HumanEval and T=0.8 for MBPP. SantaCoder uses T=0.2 with 200 samples, CRUXEval T=0.2 with n=10, and Gemma 3 samples (LCB is an average of 8). Sampled pass@1 at high T is lower than greedy.
- **The same model gets different scores from different harnesses.** DeepSeek-Coder-1.3B-base scores HE 28.7 on the EvalPlus leaderboard, 34.8 in the DeepSeek and Qwen reports, and 28.7 in StarCoder2. Qwen2.5-Coder-1.5B-Instruct scores LCB 6.1 in its own report and 15.7 in OpenCoder's (different window). Yi-Coder-9B base MBPP is 48.4 in OpenCoder's table against 69.4 for 3-shot. Prefer one source per comparison table.
- **EvalPlus versions.** The leaderboard uses EvalPlus HumanEval+ and MBPP+ (v0.2), so a self-run with a newer EvalPlus release could differ slightly. In Qwen3 and Kimi K2, 'EvalPlus' is a single average of the four numbers, not HE+.
- **LiveCodeBench windows.** The score depends heavily on the problem window, and newer releases deliberately add harder problems. Windows seen here: leaderboard v6 2024-08..2025-05, v5 2024-08..2025-02, Qwen3 v5 2024-10..2025-02, Qwen2.5-Coder 2024-07..2024-11 (its text says 2407-2409), OpenCoder 2305-2409, DeepSeek 2408-2505, Kimi K2 v6 Aug24-May25, Qwen-2507 v6 25.02-25.05, and newer cards 'v6' with no window. Only compare numbers that share a window.
- **LCB contamination flags.** The leaderboard flags a model as possibly contaminated if its release date is after the window start. In the v5 default window that applies to MetaStone-L1-7B and DeepSeek-V3. DeepSeek-V3 also has duplicate per-problem records in v5.json (558 against 279 problems), averaged as the page does.
- **Reasoning models.** o1/o3/o4-mini, R1, QwQ, Gemini 2.5 and 'thinking' modes spend long chains of thought. Their LCB and BigCodeBench numbers are not comparable with non-reasoning baselines of the same size. BigCodeBench runs some of them at temperature=1.
- **BigCodeBench subsets.** Hard (148 tasks) and Full (1,140 tasks) differ a lot, and Complete (docstring completion) is not Instruct (natural-language instruction). DeepSeek-V4's 'BigCodeBench 3-shot' comes from an internal protocol and is not the leaderboard metric.
- **CRUXEval prompting.** The leaderboard uses direct prompting at T=0.2. Qwen2.5-Coder reports CoT variants, which are higher, and Qwen3 reports CRUX-O 1-shot. These are not interchangeable.
- **phi-1-small 45% is rounded** (abstract and Table 2). phi-1 Table 2's HumanEval column also lists integer-rounded values (CodeGen-Mono-350M 13%, phi-1-base 29%).
- **Model sizes.** The leaderboards sometimes list active rather than total parameters for MoE models. I corrected the known cases (DeepSeek-V2.5/V3, Mixtral, DBRX, Llama-4-Maverick) and noted each in `note`. Closed models have `size_b: null`.
- **Grouping.** 'frontier' means closed models plus open models of 200B or more. 'open-large' means open models of 10B to 200B (I also put AzeroGPT-64b here, though its openness is unclear). 'small' means 10B or fewer.

# Data plan

> The data plan, updated with what was built: the sections on the problem bank and repair data
> describe the shipped datasets. The tokenised data lives in a private Hugging Face dataset.

Data is the biggest lever at this model size. phi-1 showed that a small model trained on
"textbook-quality" code beats models trained on 10× more raw GitHub code. This file covers what
we collect, how we judge it, and how we fill the gaps.

## Principles

1. Quality over volume. A file only gets in if we'd be happy for the model to write code like it.
2. Modern Python 3. Prefer 3.9+ idioms (f-strings, type hints, dataclasses, `pathlib`, `match`).
   Python 2 and deprecated APIs get filtered or down-weighted.
3. Whole projects, not only loose files. The model should learn how a good repository is laid
   out: package structure, configuration, tests next to code.
4. Coverage is planned. Every concept in the curriculum gets a minimum token budget, and we
   measure it.
5. Everything synthetic is verified by running it.
6. Permissive licences only, every document carries its source and licence.
7. Every step is measured: token counts in and out, plus a random sample read by a human.

## How much data we need

Target: ~10B training tokens for the 350M flagship (see docs/plan.md), drawn from ~6.7B unique
tokens. Repeating good data up to ~4 epochs costs very little (Muennighoff et al., 2023), so the
best tier is repeated and the weakest tier is left out.

| Bucket | Tier | Unique tokens | Epochs | Training tokens | Share |
|---|---|---|---|---|---|
| Verified synthetic (our own textbooks, exercises, design-as-code) | A | ~0.25B | 4 | ~1.0B | 10% |
| Rewritten code and ready-made verified sets (OpenCoder annealing; our own rewrites) | A | ~1.3B | 2 | ~2.6B | 26% |
| Curated repos and top PyPI packages, repo-level | A | ~0.8B | 2 | ~1.6B | 16% |
| Python docs, PEPs, stdlib, docs of top packages | A | ~0.15B | 4 | ~0.6B | 6% |
| Edu-filtered Python files | B | ~2.2B | 1 | ~2.2B | 22% |
| Notebooks converted to scripts | B | ~0.4B | 1 | ~0.4B | 4% |
| Python Q&A (StackExchange) | B | ~0.4B | 1 | ~0.4B | 4% |
| Commits, diffs, issue-fix pairs | B | ~0.3B | 1 | ~0.3B | 3% |
| Programming-focused English text | B | ~0.9B | 1 | ~0.9B | 9% |
| Total | | ~6.7B | | ~10B | 100% |

Tier C (everything that passed the hard filters but scored low) is kept on disk, not trained on,
and used only if tiers A and B come in short.

During the final ~15% of training (the LR decay phase), the mix shifts to tier A: synthetic and
curated repos go up, English and raw files go down.

If plan A (TPU) gives more compute and the base run goes past 10B tokens, add more unique tier B
tokens rather than more epochs of tier A.

These numbers are starting estimates. Real counts come out of phases 2–4 below, and the mix gets
tuned with small ablations (see "Checking the data").

## Sources

| Source | Tier | How to get it | Licence handling | Notes |
|---|---|---|---|---|
| Stack-Edu, Python split | B (top scores to A) | HF dataset; file contents fetched from Software Heritage | Licence per file in metadata | Already filtered by an educational-quality classifier. Check access terms and current token count on the card |
| The Stack v2 / v1-dedup, Python | B/C | HF (gated). v2 needs content download from Software Heritage S3 (AWS account); v1-dedup has contents inline | Permissive subset, opt-outs honoured | Volume source if Stack-Edu runs short. v1-dedup is the easier fallback. If downloading from Software Heritage's S3 would cost anything, skip it and use sources with contents inline (v1-dedup, SwallowCode) |
| Curated GitHub repos | A | GitHub API search, shallow clone | SPDX licence from API, confirmed with a licence detector | Selection rules below. Target 5k–15k repos |
| Top PyPI packages | A | Top-packages list + PyPI JSON API, download sdists | Licence from metadata, confirmed from LICENSE file | Real, packaged, widely used code. Target top ~10k |
| CPython stdlib and test suite | A | Clone CPython | PSF | Reference-quality code and tests |
| Python docs, HOWTOs, PEPs | A | CPython `Doc/`, PEPs repo | PSF / public domain | English concept explanations with code |
| Docs of top packages | A | Sphinx sources in the repos above | Same as the package | Explains APIs people actually use |
| Jupyter notebooks | B | Kaggle notebooks and GitHub notebooks in the Stack | Per-source | Convert: markdown cells become comments, drop outputs |
| StackExchange Python Q&A | B | Official data dump | CC BY-SA (see licence section) | Keep accepted or high-score answers with code |
| CommitPack (Python) | B | HF dataset | Permissive subset | Commit message + diff teaches editing |
| Programming English | B | FineWeb-Edu filtered to CS and programming topics; Wikipedia CS articles | ODC-By / CC BY-SA | Only for understanding requests |
| Synthetic | A | Generated (see below) | Teacher licence must allow training on outputs | The coverage-gap filler |
| SwallowCode | A | HF dataset | Inherits the rewriting model's licence (Llama 3.3 terms); check before use | Stack Python rewritten by an LLM into clean, self-contained code. The paper reported large HumanEval gains over Stack-Edu at the same token budget. Saves us most of the rewriting cost |
| OpenCoder corpora (annealing, SFT stage 1/2) | A | HF datasets | Check each card | Filter to Python. Built for exactly the annealing and SFT stages we plan |

Ready-made datasets are the efficient route: other teams already spent the GPU time on
filtering and rewriting. We still run our own filters, scorer and decontamination over them, since
their thresholds aren't ours.

## Pipeline

Built on HF `datatrove` where it fits. Every stage writes Parquet with the same columns:
`id, source, repo, path, licence, text, n_tokens, scores, topics, tier, flags`. A manifest per
stage records counts in and out.

### Phase 1: inventory and access (week 1)

- Request access to gated datasets (Stack v2, Stack-Edu).
- Write the licence allowlist (MIT, Apache-2.0, BSD-2/3, ISC, PSF, Unlicense, CC0, Zlib) and the
  policy for CC BY-SA text.
- Storage: nothing is kept on your machines. Sources are streamed into Kaggle CPU sessions
  shard by shard (up to 5 sessions in parallel). Each session processes its shard and uploads
  the result as Parquet to a private Hugging Face dataset repo or a Kaggle dataset. Raw data is
  never stored in full; only processed shards are kept. Check HF and Kaggle storage limits in
  week 1.
- Write the curriculum and the 300-task concept eval first. The eval must exist before collection
  so decontamination can use it.

### Phase 2: acquire (weeks 1–3)

- One fetcher per source in `data/sources/`, each resumable and rate-limit aware.
- Record provenance for every file: source, repo URL, commit, path, licence.

#### Collecting from GitHub

The aim is the best Python on GitHub, not all of it. Five gates, cheapest first, so expensive
checks only run on survivors.

1. Discovery (API only). GitHub search for `language:Python fork:false archived:false stars:>=50`,
   sliced by star ranges and push dates to get past the 1,000-results-per-query cap. Add repo URLs
   from the top ~10k PyPI packages and from curated lists such as awesome-python. Expect
   50k–100k candidates.
2. Metadata gate (API only). Licence on the allowlist (`license.spdx_id`), Python ≥ 60% of
   language bytes, repo under ~200 MB, pushed in the last ~3 years, not a template, tutorial
   dump or course repo (name, topic and description heuristics).
3. Shallow clone. `git clone --depth 1 --filter=blob:limit=1m`, keep only `.py`, `.pyi`,
   `README*`, `pyproject.toml`, `setup.cfg`, `setup.py`, `docs/**/*.md|rst` and CI config. Record
   the commit SHA.
4. Static quality gate (per repo). Has tests and CI; ruff issues per 100 lines; type-hint and
   docstring coverage; mean file quality score; share of files that parse. Keep roughly the top
   10–20k repos.
5. Execution gate (top ~2k repos, optional but valuable). Install each repo in a throwaway
   sandbox (`uv venv`, `uv pip install -e ".[test]"`), run `pytest` with a timeout. Repos whose
   tests pass become the "verified" set: their functions and tests feed repo-mined problems for
   post-training and execution traces for pretraining.

Needs a GitHub token (5,000 REST requests per hour; search is 30 requests per minute), stored as a
Kaggle secret. Gates 1–4 run in Kaggle CPU sessions, a few days with 5 in parallel. Gate 5 runs in
CPU sessions with no secrets attached, since it installs and runs other people's code: a fresh
venv per repo, subprocess limits, results written to the session output.

### Phase 3: normalise, filter, dedup (weeks 2–4)

Hard filters (failing any one drops the file):

1. Licence not on the allowlist.
2. Doesn't parse with Python 3 `ast.parse`.
3. Auto-generated: header markers ("generated by", "do not edit"), protobuf/gRPC stubs, Django
   migrations, Qt UI files, vendored copies of big libraries.
4. Data, not code: alphanumeric fraction below ~0.25, huge literal lists or base64 blobs, max line
   length over ~1000, average line length over ~100.
5. Too short (under ~10 lines of code) or too long (over ~100k chars).
6. Secrets and PII: API keys, tokens, private keys, emails, IPs. Redact where possible, drop if
   the file is mostly secrets.

Then dedup:

- Exact dedup by content hash.
- Near-dup with MinHash LSH (5-gram shingles, Jaccard ≥ 0.7), keep the copy from the
  highest-quality source.
- Repo-level near-dup so a thousand forks of one tutorial count once.

Cleanup transforms (all keep behaviour unchanged, and the file must still parse afterwards):

- Strip trailing whitespace, normalise line endings and encodings to UTF-8 / `\n`.
- Replace long licence header blocks with a one-line `# SPDX-License-Identifier:` comment. They
  repeat across millions of files and waste tokens.
- Remove `# -*- coding: utf-8 -*-` lines and editor modelines.
- Remove large commented-out code blocks (comment runs that parse as Python).
- Format everything with `ruff format`. The model then learns one consistent PEP 8 style and
  writes it back. Keep the unformatted version of a small sample to check formatting doesn't hurt
  the ablations.
- Drop notebook outputs and execution counts; turn markdown cells into comments.

### Phase 4: score, tag, tier, decontaminate (weeks 3–5)

Quality score, topic tags and tier, described in their own sections below. Then decontamination:

- Remove any document with a 10-gram overlap (after whitespace normalisation) with the prompt or
  solution of any eval task: HumanEval(+), MBPP(+), BigCodeBench, LiveCodeBench, CRUXEval,
  DS-1000, APPS and our concept suite.
- MinHash against eval solutions too, to catch renamed copies.
- Apply the same checks to synthetic data. Teacher models have memorised HumanEval and will
  reproduce it.
- Log every removal with the matching benchmark, and publish the counts.

### Phase 5: synthetic generation (weeks 2–6, continues later)

See "Synthetic data" below.

### Phase 6: tokenise and ship (weeks 5–6)

- Train the tokenizer on a sample of the final mix (docs/plan.md, Tokenizer).
- Tokenise into `uint16` shards of ~100M tokens, one directory per bucket, so the mix is set at
  load time by sampling weights instead of rebuilding data.
- Upload shards as private Kaggle datasets (check current size limits). Public at release.
- Write the data card: every source, licence, filter, removed counts, final token counts.

### Phase 7: instruction and preference data (weeks 8–13, during pretraining)

See "Instruction data" and "Preference data" below.

## Quality scoring

Stack-Edu and FineWeb-Edu scored "educational value". We want something closer to "would a
senior Python engineer approve this", so we train our own scorer the same way.

1. Rubric, 0–5, applied to a whole file:
   - Structure: clear separation into functions and classes, sensible names, no 500-line
     functions.
   - Idiom: modern, Pythonic code; uses the stdlib well; no reinvented built-ins.
   - Failure handling: specific exceptions, cleanup with `with`/`finally`, input validation where it
     matters.
   - Documentation: docstrings and comments that explain why, not what.
   - Typing and testability: type hints, pure functions where possible, injectable dependencies.
   - Teaching value: a learner would pick up a real concept from reading it.
   - Penalties: boilerplate, dead code, commented-out blocks, magic numbers, global state, Python 2.
2. Annotate ~50k–100k files sampled across all sources with an open-weight teacher model using
   the rubric. Ask for a short justification plus the score, keep the score.
3. Hand-grade ~300 of those yourself to check the teacher agrees with you. Fix the rubric until
   it does.
4. Train a fast scorer: a small code encoder (or frozen code embeddings plus a regression head)
   on the teacher scores. Check it on a held-out set (target: Spearman ≥ 0.75 against teacher
   scores).
5. Score every file. Combine with cheap static signals:

| Signal | Tool |
|---|---|
| Lint issues per 100 lines | `ruff` |
| Cyclomatic complexity, function length | `radon` or AST walk |
| Docstring and type-hint coverage | AST walk |
| Modern syntax use, deprecated APIs | AST walk + list of deprecated modules |
| Has nearby tests | Path heuristics (`tests/`, `test_*.py`) |

Tiers: A is roughly the top 10–15% of files plus curated sources, B is the next ~40%, the rest is
C. Thresholds get tuned with the ablations, not guessed.

Annotation cost, free routes only: ~50k–100k files × ~1.5k tokens, almost all of it input
(prompt processing is fast, outputs are a few tokens). On Kaggle T4s with a 7–14B teacher that's
roughly 10–20 GPU hours. Free API tiers can carry part of it at no quota cost, spread over a week
or two of daily limits. Scoring the whole corpus with the small encoder takes another ~5–10 GPU
hours.

## Concept coverage matrix

This makes "knows all the concepts" measurable.

1. Write the curriculum as a tree (docs/plan.md lists the top level). Leaves are specific: "asyncio
   task cancellation", "LRU cache with OrderedDict", "custom exception hierarchy",
   "`__slots__` for memory".
2. Tag every document with the leaves it touches:
   - AST detectors for concrete ones: imports `asyncio`, defines `__enter__`/`__exit__`, subclasses
     `ABC` or `Protocol`, uses `weakref`, raises a custom `Exception` subclass, uses `@dataclass`.
   - A small classifier (trained like the quality scorer) for abstract ones: design patterns,
     separation of concerns, retry logic.
3. Count tier A + B tokens per leaf. Set a floor, for example ≥ 5M tokens per leaf.
4. Leaves under the floor become synthetic-data assignments.

Rerun the matrix after every data change. Publish it with the data card.

Curriculum top level, to expand into leaves:

- Language core: data model, dunder methods, iterators, generators, comprehensions, closures,
  decorators, descriptors, metaclasses, context managers
- OOP: the four pillars, SOLID, composition vs inheritance, ABCs and protocols, dataclasses,
  design patterns
- Errors: exception hierarchies, custom exceptions, retries and backoff, cleanup, logging
- Memory and performance: references and refcounting, `gc`, `__slots__`, `weakref`, streaming
  with generators, profiling, caching, complexity
- Concurrency: threading and the GIL, multiprocessing, asyncio, queues, locks, executors
- Typing: annotations, generics, `Protocol`, `TypedDict`, mypy-clean code
- Testing: pytest, fixtures, parametrisation, mocking, property-based tests
- System design in Python: LRU cache, rate limiter, job queue, pub/sub, connection pool, KV store,
  URL shortener, event sourcing, plugin systems, CLI tools, small web services
- Project structure: packaging, `pyproject.toml`, configuration, logging setup, CLI entry points
- Ecosystem: stdlib first, then common libraries (requests/httpx, numpy, pandas, fastapi,
  sqlalchemy, pydantic, click)
- Algorithms and data structures

## Repository-level data

This is where the model learns what good projects look like.

Selection rules for curated repos:

- Permissive licence, not a fork, Python is most of the code.
- Has tests and a CI config, has `pyproject.toml` or `setup.cfg`/`setup.py`.
- Commits within the last ~3 years.
- Stars ≥ ~50 as a weak signal only; the repo's average file quality score matters more.
- Not homework or course dumps (detected by near-dup clusters and names).

Building samples (StarCoder2 / DeepSeek-Coder style):

```
<|repo|>owner/name
<|file_sep|>README.md
...
<|file_sep|>pyproject.toml
...
<|file_sep|>src/pkg/core.py
...
<|file_sep|>tests/test_core.py
...
```

- Order source files by the import graph so dependencies come before the files that use them.
  Tests go after the code they test.
- Repos longer than the context window are split on file boundaries into groups of related files
  (a module plus its imports plus its tests), not cut at random.
- Source/test pairs are also extracted on their own: they feed "write tests for this" SFT tasks.
- README and config files stay in as context. They're short, and they teach how a project fits
  together.

## Synthetic data

phi-1's main lesson. The goal is not volume, it's filling planned gaps with verified examples.

### Kinds

| Kind | What it looks like | Verification |
|---|---|---|
| Textbook modules | A concept taught as a Python module: long docstrings explaining why, worked examples, small demos in `__main__` | Parses, runs, ruff clean |
| Exercises | Signature + docstring spec, solution, pytest tests | Solution passes its own tests; tests fail on a mutated solution |
| Design-as-code | A system (rate limiter, job queue) with a module docstring covering requirements, trade-offs and alternatives, then the implementation and tests | Tests pass |
| Refactoring pairs | Messy code, then clean code with the same behaviour | Same tests pass before and after |
| Bug-fix pairs | Code with an injected bug, the traceback, the fix | Bug reproduces; fix passes tests |
| Execution traces | A function plus `assert f(x) == y` lines, with `y` from actually running it | Values come from execution, never the teacher |
| Error-handling scenarios | Code that handles a specific failure (timeouts, partial writes, bad input) with tests that trigger it | Tests pass |

Execution traces and bug-fix pairs are cheap: they come from running real code on CPU, so the
teacher is only needed for parts of them.

### Process

1. Each generation job takes a curriculum leaf, a kind, a difficulty (1–5), a domain (finance,
   games, devops, science, web...) and a style seed. The combination drives diversity.
2. Generate with an open-weight code model whose licence allows training on outputs. The
   Qwen-Coder family is one option (check the licence of the exact size). Don't use outputs from
   APIs whose terms forbid training other models.
3. Run every sample in the shared sandbox (docs/training.md): subprocess, CPU and memory limits,
   timeout, temporary directory, in a Kaggle CPU session with no secrets attached.
4. Keep only what passes. Expect to discard 30–50%.
5. MinHash dedup within synthetic data and against the eval sets. Cap any single leaf at ~3% of
   synthetic tokens.
6. Read 50 random samples per leaf before scaling that leaf up.

### Cost (free routes only)

Measured 10 October 2026 (these replace the estimates below where they differ):

| Route | Measured | Notes |
|---|---|---|
| Cerebras free tier (Qwen3.8-27B) | ~3.7M tokens, then HTTP 402 | The 216M tokens/day in its headers is a rate limit, not a free allowance. Not used further |
| Groq free tier (gpt-oss-120b, Qwen3.8-27B) | ~1,800 samples/day | 1,000 requests/day and 8k tokens/min per model. Runs continuously as a capped service on the orchestration box |
| Kaggle 2x T4, vLLM, Qwen2.5-Coder-14B-Instruct-AWQ | ~1M tokens per GPU-hour at batch 96; 44% pass execution | 7B-AWQ was faster (1.5M/GPU-h) but passed only 29%. Production run uses large batches plus one repair round |
| Kaggle 2x T4 production run (14B-AWQ, with one repair round) | 1,481 verified of 3,000 (49%: 37% first pass + 12% after repair) and 367 bug-fix pairs per ~8 h | ~2.7 h per 1,000 prompts, about 1.5M verified tokens per session. Too slow to matter for pretraining volume; the verified exercises are kept for the RL problem bank and the bug-fix pairs for SFT |
| API pilot (Qwen3.8-27B, prompt v1) | 24% pass execution | 37% of test-requiring samples had no tests; prompt v3 makes tests mandatory and checks for them |

Revised target: ~50–80M verified synthetic tokens over the project rather than 250M, used mainly in
the annealing phase where quality matters most. Every repaired failure also yields a bug-fix pair
(broken code, error output, fixed code) for the bugfix bucket.


Target ~250M verified tokens of our own, so ~400M generated. Free ready-made verified datasets
(KodCode, OpenCoder's SFT and annealing sets, execution-filtered instruction sets, each checked
for licence) cover part of the same ground, which is why our own target can be modest.

| Route | Throughput | Share of the ~400M |
|---|---|---|
| Kaggle 2× T4, vLLM, 7B teacher (4-bit) | ~2–3k tokens/s, ~10M tokens per GPU hour | ~300M, ~30–40 GPU hours |
| Free API tiers, stronger open-weight teacher | Daily rate limits, typically from a few hundred thousand to a few million tokens a day; check current limits | ~50–80M over the project, for textbooks and design-as-code |
| Monthly free credits (L4/A10-class GPUs) | ~2–4k tokens/s with a 7–14B teacher | ~20–50M a month |
| Execution-derived data (traces, bug-fix pairs from real repos) | CPU only, no teacher needed for most of it | As much as the CPUs allow |
| TPU via vLLM's TPU backend | Unmeasured | Only if plan A frees TPU hours after pretraining |

Split: the stronger teacher on free API tiers for textbooks and design-as-code, where quality
matters most. The 7B teacher on Kaggle for exercises, refactoring and bug-fix pairs, where
execution catches mistakes. Start with 20M tokens, read samples, then scale.

## Rewriting tier B into tier A

SwallowCode showed that rewriting ordinary GitHub Python with an LLM (fix style, add types and
docstrings, make it self-contained, improve the algorithm) beats filtering alone at the same token
budget. It's the most efficient upgrade available, because it turns existing tier B data into
tier A instead of generating from nothing.

- SwallowCode itself is not used (decision of 10 October, below). We do our own rewrites with
  gpt-oss-120b (Apache-2.0) on the Groq and Cerebras free tiers, plus the free Nemotron models on
  the Nara router.
- Rewrite files that score in the middle band (good ideas, weak style), Stack-Edu first, as much
  as free API quota allows. Top files stay as written; low files are dropped.
- Keep behaviour: when the file has tests (verified repos), the rewrite must pass them. Otherwise
  it must parse, pass ruff, and keep the same public names and signatures.
- Never rewrite in a way that loses the repo structure. Rewritten files go into the file-level
  bucket, originals stay in the repo-level bucket.

## Programming English

The model needs to read requests, not write essays. ~1B tokens is enough.

- FineWeb-Edu filtered with a topic classifier to programming, CS and software engineering.
- Python docs, HOWTOs and PEPs (already in tier A).
- Wikipedia articles in computing categories.
- StackExchange question text (answers are mostly code and go in the Q&A bucket).

## Instruction data (SFT)

Target 200k–500k examples. Every response is one Python module, with any explanation in
docstrings and comments.

| Share | Kind | Sources |
|---|---|---|
| ~35% | General coding tasks | Magicoder OSS-Instruct, StarCoder2 self-oss-instruct (execution-filtered), OpenCodeInstruct, filtered to Python |
| ~25% | Curriculum tasks with tests | Our own, generated per leaf |
| ~10% | Fix, refactor, add tests, add types | Bug-fix and refactoring pairs, source/test pairs |
| ~10% | Multi-file projects | CLI tool, small API service, package scaffold, using `<|file_sep|>` |
| ~10% | "Explain this code" | Response is the same code annotated with docstrings and comments |
| ~8% | Vague or underspecified requests | Response states its assumptions in the module docstring |
| ~2% | Non-code requests | Still answered in Python (a program, or a docstring), so the format never breaks |

Conversion rules for every response:

- Must parse, must pass a selected set of ruff rules, must pass tests when tests exist.
- No markdown, no fences, no prose outside comments and docstrings.
- Type hints on public functions; `if __name__ == "__main__":` demo where it helps.
- Check licences of each public dataset, and decontaminate them like everything else.

## Problem bank and repair data

Built 21–22 October 2026 by `data/jobs/problem_bank.py`, `verify_bank.py` and `bug_inject.py`, all on
Kaggle CPU. Anything that runs solutions or mutants runs offline with no keys attached.

The bank takes the train splits of TACO (Apache-2.0), CodeContests (CC BY 4.0) and APPS (MIT).
I keep Python 3 solutions only, drop problems with images or file I/O, and keep up to 5
solutions and 25 tests per problem. Of 43,119 source rows, 19,585 survive those filters. MinHash
dedup (similarity 0.8 on 5-word shingles) merges 6,941 copies, leaving 12,644 problems: 9,583
TACO, 2,807 CodeContests and 254 APPS. 2,602 are call-based (a function with JSON arguments) and
the rest read stdin.

Decontamination uses the same 10-gram index as the SFT builder, plus APPS test, CodeContests
test/valid and the concept suite. The first run flagged 1,172 of APPS's 5,000 train problems as
APPS test copies, though the two splits are disjoint: competitive statements share boilerplate
("The first line contains an integer n"). For those two benchmarks a match now has to cover a
quarter of the statement's 10-grams. It still removes 8,104 problems, mostly real APPS test
copies inside TACO and Codeforces problems that APPS test also has.

Verification runs every stored solution against every test in a sandbox (prlimit, 2 GB, 4 s CPU
per test, empty env). 8,729 problems (69%) keep at least one passing solution: APPS 96%,
CodeContests 82%, TACO 65%. 66% of the 57,941 solutions pass. Failures are mostly wrong answers
(16,230), then runtime errors (2,835) and timeouts (303). 600 problems have one solution passing
and another giving a different answer. Those almost always accept several answers (a special
judge), so they're flagged `inconsistent`, and anything that trusts the tests to reject wrong
code should skip them.

Repair data comes from mutating passing programs: the 8,129 consistent bank problems, the 2,586
verified Tinker exercises and our own problems. There are 14 AST mutations (comparison,
off-by-one, wrong variable in scope, dropped statement, missing import, swapped arguments and so
on), balanced so each type ends up with about 1,720 kept rows. A mutant is kept only when it
really fails, and the prompt shows what the user would see: the traceback trimmed to the
program's own frames, or the input with expected and actual output, or the failing pytest run.
20% of mutants turned out to be equivalent and were dropped, and timeouts are capped at 3%. The
result is 24,099 rows: 16,100 from the bank, 7,744 from Tinker and 255 from our own problems.
14,076 show a runtime error, 9,857 a wrong answer and 166 a timeout.

Our own problems (`data/own`) are 85 everyday tasks with a reference solution and at least four
pytest tests: parsing, CSV and JSON wrangling, files, small apps, dates, money. They're separate
from the concept suite and all pass `build.py --check`.

The SFT builder applies the same weaker rule for the two contest benchmarks. With the plain rule
it dropped 3,859 repair rows and 3,045 bank rows, almost all of them APPS test boilerplate. Now
it drops 305 and 75. After its other filters (mostly one- and two-line programs), `sft/repair`
has 22,246 examples and 25.1M training tokens, of which 8.2M carry loss. `sft/bank` has 7,518
examples and 5.4M tokens, 1.3M with loss.

On HF, under `v1/`: `problems/raw`, `problems/verified`, `problems/own`, `repair`, and the
tokenised `sft/bank` and `sft/repair`.

## Preference data

Built during post-training from the model's own outputs:

- For each task with tests, sample 8–16 solutions from the SFT model.
- Passing solutions feed rejection-sampling fine-tuning.
- (passing, failing) pairs feed DPO. Prefer pairs where both parse, so the model learns
  correctness, not just syntax.

## Licences and ethics

Decided 10 October 2026:

- SwallowCode is excluded. It's under the Llama 3.3 licence, which would require the released
  model's name to start with "Llama" and to carry "Built with Llama".
- KodCode is excluded. It's CC BY-NC 4.0 (non-commercial), which clashes with an Apache-2.0
  release. OpenCoder's MIT-licensed SFT sets and our own verified exercises cover the same ground.
- Stack-Edu: only files whose `license_type` is `permissive` and whose detected licences are all
  on the allowlist. That's 16.7% of its Python rows; "no_license" files are dropped.
- Teacher models for synthetic data and rewrites: open-weight models whose licences allow training
  on outputs (gpt-oss, Nemotron, Qwen). No closed-model outputs (OpenAI, Anthropic, Google), even
  through a reseller.


- Code: permissive allowlist only, confirmed per file. Track it per document so the data card can
  state exact numbers.
- CC BY-SA text (StackExchange, Wikipedia): widely used in open models, but share-alike terms are
  debated for model training. Decide once, write it down in the data card, and keep it in its own
  buckets so it can be dropped cleanly.
- Honour opt-outs (the Stack's opt-out list).
- Redact secrets and personal data before anything is tokenised.
- Synthetic data inherits the teacher's terms. Record the teacher for every sample.

## Checking the data

- Human reading: 100 random samples per source per tier, every time a filter changes. It's cheap
  and catches more than any metric.
- Small ablations (FineWeb-style): train the 20M debug model on ~1B tokens from one bucket or mix
  at a time, compare held-out tier-A loss and HumanEval+ / concept-suite scores. ~2 GPU hours each,
  6–8 runs total.
- Tokenizer check: characters per token on held-out code, compared with other tokenizers.
- Coverage matrix: every leaf over its floor before pretraining starts.

## Done when

- Tier A, B and C token counts are known per source.
- Every curriculum leaf is over its token floor.
- Decontamination report is written, with zero hits on the eval sets.
- Shards are uploaded to Kaggle and load correctly in the training script.
- Data card draft exists.
- The ablation runs show tier A beats tier B, and the chosen mix beats raw Stack Python.

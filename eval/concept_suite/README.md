# Concept suite

A private eval of about 300 Python tasks, one or more for each of the 223 leaves in
`data/curriculum.json`. Each task is a plain-English prompt that names its entry points, a
reference solution, and hidden pytest tests that check the concept itself, not only the output
format. Scoring is greedy pass@1: the model writes one module and the hidden tests either pass or
they don't.

The suite stays private until the model is released. Its prompts and references are also on the
training-data decontamination list (10-gram overlap and MinHash, see DATA.md, phase 4), so none of
it may be pasted into training data, synthetic-data prompts or public issues.

## Files

- `src/<area>.N.txt`: the authoring files. The format and the lint rules are in the docstring of
  `build_tasks.py`.
- `tasks.jsonl`: the built suite, one task per line (`task_id`, `area`, `leaf`, `difficulty`,
  `prompt`, `entry_points`, `reference`, `tests`, optional `min_python`). Generated; edit `src/`.
- `build_tasks.py`: builds `tasks.jsonl`, lints every task and prints leaf coverage.
- `check_refs.py`: runs every reference against its tests. Every task must pass.
- `run.py`: the scoring harness.

## Build and check

```
python eval/concept_suite/build_tasks.py
nice -n 19 ~/.cache/ophyx-smoke/venv/bin/python eval/concept_suite/check_refs.py --jobs 2
nice -n 19 ~/.cache/ophyx-smoke/venv/bin/python eval/concept_suite/check_refs.py --filter concurrency --repeat 3
```

The build exits non-zero on any lint error. Run the concurrency repeat after touching a threading
or asyncio task, to catch flaky tests. Tests must be deterministic, use no network, write only
under `tmp_path`, sleep at most 0.05 s and finish in about 3 s.

## Run

```
# check the harness and the tasks: scores the references, must be 100%
python eval/concept_suite/run.py --self-test --out /tmp/concept

# a model (HF id or local path), greedy on a GPU
OPHYX_CONFIG='{"model": "/kaggle/working/hf-ckpt"}' python eval/concept_suite/run.py

# precomputed outputs: JSONL rows of {"task_id", "completion"} or {"task_id", "solution"}
python eval/concept_suite/run.py --completions samples.jsonl --tag mymodel
```

CONFIG follows `eval/evalplus_run.py`: a `CONFIG` dict in the notebook globals or JSON in
`OPHYX_CONFIG`, and CLI flags override it. The keys are listed in the `run.py` docstring.

A tokenizer with `<|user|>`, `<|python|>` and `<|end|>` gets the project chat template
(`<|user|>\n{prompt}\n<|python|>\n`, stopping at `<|end|>`). A base model gets the task as a module
docstring instead. The harness extracts the module (it unwraps ``` fences) and writes it as
`solution.py` next to `test_task.py`. Pytest then runs in a subprocess with a timeout, prlimit
limits, nice 19, an empty environment and its own temp dir.

It reports pass@1 overall, per area and per difficulty, plus the share of outputs that compile.
Pytest exits 0 when tests skip (the zoneinfo tasks skip without tzdata), so passes with skipped
tests are counted separately; on a normal image that count should be 0.
Tasks whose `min_python` is newer than the sandbox Python are skipped and counted separately.
Results go to `<out>/concept-<tag>.jsonl` and `<out>/concept_results.json`.

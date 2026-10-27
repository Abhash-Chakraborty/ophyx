# Hands-on review: ophyx-350m-v2

I read all 42 answers that the final model gave on the hands-on set (`eval/manual`), next to the
prompt and the test output. Generation was greedy with the Ophyx template, and scoring ran in the
offline sandbox (`eval/bench_exec.py`). The tests pass 9 of 42 answers.

| Difficulty | Tasks | Passed |
|---|---|---|
| 1 | 10 | 4 |
| 2 | 20 | 4 |
| 3 | 10 | 1 |
| 4 | 2 | 0 |

## What passes

Tasks with a well-known shape pass: merging intervals, checking a sudoku, Dijkstra with `heapq`,
summing a CSV column, a timing context manager, a token bucket and the `KeyError` fix with a
traceback. All of them are short, and each has one standard answer the model has seen many times.

Two of the nine passes would not survive a code review. In `explain_code` the docstring says the
function returns the `seen` set, which is wrong. `explain_recursion` defines the same function twice.

## Why the rest fail

**It writes tests nobody asked for.** 19 of 42 answers end with the model's own `test_*` functions,
and 8 more call the function at module level. That makes 27 of 42 answers with extra code. The
invented tests are often wrong: `evaluate("2 - 3") == 1`, or an expected 24 business days in a
whole year. Three answers fail at import because their own module-level code raises
(`change_add_param`, `flatten_dict`, `lru_function_cache`). In `business_days` the model kept
repeating test functions until it hit the length limit, so the answer doesn't parse.

**Fix requests come back unchanged.** For `fix_no_traceback` and `fix_mutable_default` the model
returns the input code byte for byte. `refactor_to_class` rewrites the function with `Counter`,
which the prompt rules out, and never writes the class.

**It drops details in the spec.** Each of these is the one rule the test checks:

- `word_freq_cli`: ties in alphabetical order.
- `inventory_dataclass`: validation in the frozen dataclass.
- `deep_merge`: the result must not share nested dicts with the input.
- `slugify`: transliterate accented letters. The model drops them instead.
- `ttl_cache`: entries never expire, since the TTL is stored and never checked.
- `parallel_map`: wait for every call before raising the first error.
- `moving_average`: use a cumulative sum. The model wrote a loop that crashes on the first window.

**It misuses library APIs.** `sqlite_report` inserts orders into a table with different columns.
`async_limited_gather` calls coroutine objects as if they were functions. `roman` runs `int("I")`.
`generator_pipeline` takes `len()` of a generator. `enum_temperature` puts `convert` on the enum
as a static method, so `from solution import convert` fails.

**Chat prompts get code.** Asked "hi! what can you do?", it writes a program that calls `input()`,
which blocks under test. The fine-tuning data has almost no plain conversation, and it shows.

## What would help next

1. Strip module-level tests and demo calls from fine-tuning targets. The RL reward should also
   penalise module-level side effects, because right now extra code costs nothing unless it
   crashes on import.
2. Add edit rows where the bug is described in a sentence. Every repair row shows a traceback, a
   failing test run, or contest input with expected and actual output. The one fix prompt with a
   traceback passes, and both prompts that describe the bug in words fail.
3. Add test cases for the spec's details to RL problems. The bank's tests mostly check the main
   path, so the reward never teaches the model to read the last sentence of a prompt.

Samples: Kaggle output `ophyx-bench-gen-grpo-p4` (`samples/ophyx-350m-grpo-p4-lr2e6/manual.jsonl`).
Per-task results: `ophyx-bench-exec-grpo-p4` (`manual/concept-ophyx-350m-grpo-p4-lr2e6.jsonl`).

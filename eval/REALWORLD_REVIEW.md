# Real-world review: ophyx-350m-v2

The benchmarks ask for one function from a docstring. Real requests look different: parse a log,
total a CSV, retry a flaky call, write a SQL query, rename files in Bash. `eval/realworld` holds
33 such tasks, each with tests. 25 ask for Python, 2 for SQL, 2 for Bash, 2 for JavaScript and 2
for a plain explanation. I ran eight models on them and read every Ophyx answer next to its
prompt and test output.

Ophyx 350M v2 passes 2 of 33. Only one of those passes is a correct answer.

## Scores

Greedy, one answer per task, scored in the offline sandbox (`eval/bench_exec.py`). Ophyx gets its
own template; the other models get their chat template.

| Model | Params | Python (25) | SQL, Bash, JavaScript (6) | Explanations (2) | Total (33) |
|---|---|---|---|---|---|
| **Ophyx 350M v2** | 0.35B | 1 | 0 | 1 | 2 |
| Ophyx 350M v1 | 0.35B | 0 | 0 | 1 | 1 |
| SmolLM2-360M-Instruct | 0.36B | 0 | 0 | 2 | 2 |
| Qwen2.5-Coder-0.5B-Instruct | 0.49B | 0 | 1 | 1 | 2 |
| Qwen3-0.6B | 0.75B | 0 | 1 | 2 | 3 |
| deepseek-coder-1.3b-instruct | 1.35B | 3 | 3 | 2 | 8 |
| Qwen2.5-Coder-1.5B-Instruct | 1.54B | 1 | 4 | 2 | 7 |
| Qwen2.5-Coder-7B-Instruct | 7.62B | 10 | 5 | 2 | 17 |

The same 33 tasks through Kaggle Benchmarks' model proxy, scored in the same sandbox:

| Model | Python (25) | SQL, Bash, JavaScript (6) | Explanations (2) | Total (33) |
|---|---|---|---|---|
| GPT-5.5 | 25 | 6 | 2 | 33 |
| Claude Sonnet 5.5 | 25 | 6 | 2 | 33 |
| Gemini 3.1 Pro | 25 | 6 | 2 | 33 |
| GPT-5.4 mini | 25 | 6 | 2 | 33 |
| Claude Haiku 5.5 | 25 | 6 | 2 | 33 |
| Gemini 3.8 Flash | 24 | 6 | 2 | 32 |
| Gemma 4 31B | 23 | 6 | 2 | 31 |

Their three misses are real: Gemma's `retry_backoff` reads a variable before setting it, its
`fix_pagination_described` answer fails before the tests run, and Flash spent its whole token budget
reasoning about `extract_links` and never finished the code. Haiku's answers exposed two problems on
my side, both fixed before these numbers: the first `node` start in a session could take longer than a
JavaScript test allows, and the `order_total` test wanted the quantity check in `LineItem` although the
prompt puts it in `Order`. Neither fix changed a score in the table above. So the tasks are fair: a
current API model passes nearly all of them, and the gap below 7B is the models, not the tests.

The two prose tasks are checked for one fact each, which is a low bar. Ophyx passes
`list_vs_tuple` because its answer contains the word "immutable", but the code below that says
`my_list[0] = 4  # This will raise an error, as the list is immutable.` That's wrong, so by hand v2
gets 1 of 33. The one real pass is `retry_backoff`: a decorator with exponential backoff that keeps
the wrapped function's name and re-raises the last error.

## It only writes Python

Every one of the six SQL, Bash and JavaScript tasks came back as a Python module:

- `bash_top_ips` asked for an `awk | sort | uniq` pipeline. Ophyx wrote a Python script that sorts
  log lines by their second field.
- `js_group_by` came back as a Python `def groupBy(items, keyFn)`, and its docstring says "returns
  a dictionary".
- Both SQL tasks became `sqlite3` scripts that connect to a made-up database file, and their
  queries are wrong: no join to `customers`, no 2024 filter, grouped by a column that doesn't exist.

This is by design, not a bug. The prompt template ends with `<|python|>`, so every answer starts as
a Python module. The fine-tuning data was filtered to Python only: an example with a block in any
other language was dropped (`data/jobs/sft_build.py`, step 1). So the model has almost no training
signal for SQL, Bash or JavaScript, and its template doesn't let it answer in them.

For comparison, Qwen2.5-Coder-7B-Instruct answers all six in the right language, and 5 of them
pass.

## Why the Python answers fail

**The logic looks right and isn't.** These are the most common failures, and the hardest to spot:

- `log_status_counts` looks for three space-separated numbers at the start of the line, where the
  client IP is. It never finds the status code, so it returns an empty dict.
- `password_policy` checks `pw.isupper()`, which is true only when the whole password is upper
  case. Every real password gets "no uppercase" and "no lowercase".
- `order_total` takes 10.00 off for `SAVE10` (it should be 10%), multiplies by 0.825 / 100 for
  tax, and raises `ValueError` on an order with no discount code.
- `jsonl_stats` sums the key instead of the value and then divides a dict by an int.

**It ignores the constraints in the prompt.** `worker_pool` says "not concurrent.futures" and the
answer uses `ThreadPoolExecutor`. `rate_limiter` drops the `clock` argument the tests rely on.
`rename_cli` never reads `--ext` or `--prefix`. `async_fetch_all` ignores both the limit and the
timeout, and drops failed results instead of returning the exceptions.

**It adds code nobody asked for.** 12 of the 24 coding answers end with the model's own `test_*`
functions, and 4 more run example calls on import. The invented tests are wrong as often as the code:
`find_duplicates` expects two files with different contents to be duplicates. `async_fetch_all`
calls `asyncio.run(main())` on import, so the module fails before the tests start.

**Fixes are half done.** `fix_median_described` names two bugs. The answer fixes the even-length
median but still sorts the caller's list in place, and its docstring opens with "The function is
correct". `fix_pagination_described` keeps the off-by-one start and the floor division, and adds
a `ValueError` that fires on page 1.

**The explanation is wrong.** For `'NoneType' object is not subscriptable`, the answer says `user`
is a dict missing the key `"name"`, then prints a hard-coded dict. It never mentions that
`find_user` returned `None`.

## What this means

Ophyx is a Python model, and on everyday Python requests it's weak: 1 correct answer in 25. The
benchmark numbers (43.9 on HumanEval+) come from short, well-specified functions it has seen many
times. Longer prompts with three or four rules lose a rule, and the model fills the gap with
plausible code.

No model under 1B does better on the Python tasks: SmolLM2-360M, Qwen2.5-Coder-0.5B-Instruct and
Qwen3-0.6B pass none of the 25. Their 2 or 3 points come from the explanations and one SQL or
JavaScript task.
deepseek-coder-1.3b-instruct passes 3 Python tasks and Qwen2.5-Coder-7B-Instruct passes 10, so
these tasks are hard at every size up to 7B.

The contest-problem round (CodeContests, TACO, APPS, LeetCode and Codeforces, with checked
answers) trains algorithmic correctness on stdin programs. It should help the benchmarks more than
these tasks. Getting better here would take training data shaped like these requests: multi-rule
specs with tests, fix-this-code pairs, and no trailing test code in the targets.

Rerun it with `eval/runs/r7_realworld.json` (generation) and `bench_exec.py` with
`"benches": ["realworld"]`. The tasks and their reference answers are in `eval/realworld/build.py`.

"""Everyday requests, answers in the house style, checks and follow-ups, on Kaggle 2x T4 with vLLM.

Round three. The released model handles benchmark-style tasks ("write f(x) that ...") but fails the
way people actually ask: "write an python code to print my name Saksham in capital with color red in
terminal", then "no, it should show in red, don't print the word red". This job writes that kind of
data. No keys attached and no generated code runs here; data/jobs/chat_check.py runs the code later,
offline.

Phases, each in a forked process so vLLM gives all GPU memory back when it ends:
  1. requests   the teacher writes requests from a grid of topic x style x kind x persona
  2. answers    the teacher answers each one in the house style (TEACHER_STYLE) and, in a separate
                call that never sees the answer, writes a check: stdin plus strings the output must
                contain (programs), or assert lines (functions)
  3. followups  for some requests, a realistic next message (a change or an addition) with its own
                check, and the teacher's full updated program
  4. ophyx      the current model answers a share of the requests (greedy plus samples)
  5. complaints the teacher plays the user: reads Ophyx's answer and, when it misses, writes the
                user's complaint; then answers the complaint with a full fixed program

CONFIG: teacher (HF id, default Qwen/Qwen2.5-Coder-14B-Instruct-AWQ), ophyx (HF id or /kaggle glob),
n_calls (request calls, 4 requests each), followup_frac (0.3), ophyx_n (requests Ophyx answers),
ophyx_samples (2), phases (default all), budget_s, seed.

Output in /kaggle/working/chat/: requests.jsonl, answers.jsonl, followups.jsonl, ophyx.jsonl,
complaints.jsonl, logs/, meta.json.
"""

import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import signal
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("OPHYX_CONFIG", "{}"))
OUT = os.path.join(CONFIG.get("out", "/kaggle/working"), "chat")
TEACHER = CONFIG.get("teacher", "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
SEED = CONFIG.get("seed", 0)
T0 = time.time()
BUDGET = CONFIG.get("budget_s", 10 * 3600)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


TOPICS = [
    "printing coloured or styled text in the terminal", "tables and aligned columns in the terminal",
    "progress bars and countdowns", "reading and writing text files", "CSV files", "JSON files and data",
    "renaming, moving or listing files in folders", "file sizes, extensions and paths", "word and character counts",
    "string formatting and padding", "palindromes, anagrams and string puzzles", "cleaning messy text",
    "regular expressions for emails, phone numbers and dates", "dates, times, ages and birthdays",
    "time zones and durations", "calendars and weekdays", "simple calculators", "unit conversions",
    "temperature, currency and percentage maths", "prime numbers and factors", "fibonacci and sequences",
    "number guessing and dice games", "rock paper scissors and other small games", "tic tac toe in the terminal",
    "random passwords and random choices", "shopping lists and to-do lists", "student marks and grade reports",
    "bank account and wallet classes", "inventory and stock tracking", "contact books and phone directories",
    "sorting and searching lists", "dictionaries and counting things", "sets and removing duplicates",
    "matrices and 2D lists", "stacks and queues", "linked lists and trees", "recursion",
    "generators and iterators", "decorators", "dataclasses", "classes and inheritance", "exceptions and input validation",
    "command line tools with argparse", "logging", "simple sqlite databases", "simulations with random numbers",
    "statistics: mean, median, mode", "patterns with stars and numbers", "multiplication tables",
    "ASCII art and banners", "text based menus", "quizzes", "timers and stopwatches", "parsing log files",
    "configuration files (ini, toml, env)", "hashing and checksums", "encoding: base64, caesar cipher, morse code",
    "working with bytes and binary numbers", "roman numerals and number words", "async tasks with asyncio",
    "threads and running things in parallel", "itertools and combinations", "collections: Counter, deque, defaultdict",
    "sorting objects by several keys", "pathlib", "zip files and archives", "email and URL parsing",
    "HTTP requests with the standard library (no network in tests)", "web scraping HTML strings with html.parser",
    "pandas dataframes (only when the user asks for pandas)", "plotting with matplotlib (only when asked)",
    "small data analysis of a list of records", "school maths: area, volume, quadratic equations",
    "geometry: points, distances, shapes", "games of chance and probability", "text adventure rooms",
    "weather and temperature records", "employee and salary records", "library book management",
]
STYLES = [
    ("casual", "casual and lowercase, a couple of typos or missing words, like a student in a hurry "
               "(for example 'write an python code to ...', 'can u make a program that ...')"),
    ("plain", "clear and polite, plain English"),
    ("terse", "very short, under 15 words"),
    ("detailed", "detailed, with three or four specific requirements"),
    ("learner", "simple English from a non-native speaker, maybe with a small grammar mistake"),
    ("precise", "precise, like a working developer, naming the function or file exactly"),
]
KINDS = [
    ("program", 5, "a complete small program or script that does something when run"),
    ("function", 3, "one function, named in the request (for example 'write slugify(text) that ...')"),
    ("class", 1, "a class with a few methods"),
    ("howto", 1, "how to do something in Python ('how do i ...', 'show me how to ...')"),
]
PERSONAS = ["student", "beginner", "data analyst", "teacher", "hobbyist", "system administrator",
            "backend developer", "scientist", "shop owner", "game maker"]

TEACHER_STYLE = """You write Python answers that a small code model will learn from, so every answer must be correct, complete and \
in exactly this style. Reply with one ```python block and nothing else: no text before or after it.

1. Always start with a module docstring, even for a single function: one to three plain sentences on what the code does, then a "Usage:" line saying how to run \
it or call it.
2. Give every function and class a docstring (a one-line summary, then Args: and Returns: when it takes parameters or returns \
something) and type hints.
3. Add a short comment only where a step isn't obvious.
4. Do exactly what the user asked and use their exact names, text, numbers, file names and formats. If they ask to print \
"Saksham" in capitals in red, the program prints SAKSHAM in red, not the word "red".
5. A program or script must run as is: put the work in main() and end with if __name__ == "__main__": main(). A request \
for a function or class gets just that (and small helpers): no main block and no example calls.
6. Standard library only, unless the user names a package. Colour in the terminal uses ANSI escape codes.
7. No tests, asserts or printed examples unless asked. Read input() only when the task needs the user to type something.
8. Keep it as short as the task allows: no extra options or features nobody asked for.
9. In a follow-up, reply with the whole updated program, not just the changed part."""

CHECK_SYSTEM = """You write a quick automatic check for a Python coding request, before seeing any answer. Reply with one JSON \
object and nothing else.

For a request for a program or script: {"kind": "program", "stdin": "<what a user would type, lines separated by \\n, or \
empty>", "expect": ["<text the output must contain>", ...]}. Choose 1 to 4 expected strings that ANY correct program prints \
for that stdin (exact values the request fixes, such as a name in capitals or a computed number), never wording a program \
could reasonably phrase differently. Write numbers with at most one decimal place ("82.2", not "82.22222222222223"), \
so any sensible rounding matches. For terminal colours, expect the ANSI code written as \\u001b[31m (red), \\u001b[32m \
(green) and so on. If no output can be predicted, use "expect": [].

For a request for a function or class: {"kind": "asserts", "code": "<3 to 6 assert lines that call it, one per line>"}. Use \
only the names the request gives; when the request gives no name, use {"kind": "none"}.

If the task needs files, the network or a GUI, reply {"kind": "none"}."""

REQUEST_PROMPT = """Write {n} different requests that a {persona} might type to a Python coding assistant.
Topic: {topic}.
Each request asks for {kind}.
Writing style: {style}.
Give concrete details the code must follow, such as names, numbers, text, colours or formats, so a wrong answer is easy to \
spot. Vary the length. Don't number them and don't mention this list.
Reply with a JSON list of {n} strings and nothing else."""

FOLLOWUP_PROMPT = """Here is a request a user sent to a Python coding assistant, and the program they got back.

Request:
{request}

Program:
```python
{answer}
```

Write the user's next message. It should be {kind}. Write it in the same voice as the request: if the request was casual \
with typos, so is this message; if it was terse, keep it terse. Don't start with "Could you". Keep it under 40 words and \
don't repeat the original request. Reply with the message only."""
FOLLOWUP_KINDS = [
    "a correction: the user meant something slightly different from what the program does, and says what they wanted",
    "a small addition: one more thing the program should do",
    "a change to the output: different wording, format, colours or layout",
    "a request to handle bad or unexpected input",
    "a request to save or load the results from a file",
    "a request to make it shorter or simpler",
    "a request to turn it into a function or class they can reuse",
    "a request to let the user type the values instead of hard-coding them",
]

COMPLAINT_PROMPT = """You are the user. You sent this request to a Python coding assistant:

{request}

and got back this program:
```python
{answer}
```

Does the program do what you asked, with your exact details, and would it run? If yes, reply with the single word OK.
If not, write your next message to the assistant in the same voice as your request: short, saying what's wrong the way a \
real user would ("it doesn't show my name", "this prints the word red instead of making it red", "it crashes with ..."). \
Don't write code and don't explain how to fix it. Reply with the message only."""


def chat_prompt(tok, turns, system):
    msgs = [{"role": "system", "content": system}]
    for user, assistant in turns:
        msgs.append({"role": "user", "content": user})
        if assistant is not None:
            msgs.append({"role": "assistant", "content": f"```python\n{assistant}\n```"})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def code_block(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = max(m, key=len) if m else text
    return code.strip("\n")


def documented(code):
    """The house style: a module docstring, and a docstring on every function and class."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return ast.get_docstring(tree) is not None and all(ast.get_docstring(n) is not None for n in defs)


def parses(code):
    import ast
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def json_value(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except ValueError:
                return None
    return None


def rid(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def read(name):
    p = os.path.join(OUT, name)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def write(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class Teacher:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(model=TEACHER, tensor_parallel_size=2, dtype="float16", max_model_len=8192,
                       gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=SEED)
        self.tok = self.llm.get_tokenizer()

    def run(self, prompts, temperature=0.3, max_tokens=2048):
        from vllm import SamplingParams
        out = []
        for b in range(0, len(prompts), 1024):
            chunk = prompts[b:b + 1024]
            res = self.llm.generate(chunk, SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens,
                                                          seed=None), use_tqdm=False)
            out += [(o.outputs[0].text, o.outputs[0].finish_reason) for o in res]
            log(f"teacher {b + len(chunk)}/{len(prompts)}")
        return out

    def answer(self, turns_list, temperature=0.3):
        """Answers that miss the house style's docstrings get one more try at a higher temperature."""
        prompts = [chat_prompt(self.tok, turns, TEACHER_STYLE) for turns in turns_list]
        out = [code_block(t) if f == "stop" else None for t, f in self.run(prompts, temperature)]
        again = [i for i, a in enumerate(out) if not (a and documented(a))]
        log(f"retrying {len(again)}/{len(out)} answers without the docstrings")
        for i, (t, f) in zip(again, self.run([prompts[i] for i in again], 0.7)):
            a = code_block(t) if f == "stop" else None
            if a and documented(a):
                out[i] = a
        return out

    def checks(self, requests):
        prompts = [chat_prompt(self.tok, [(r, None)], CHECK_SYSTEM) for r in requests]
        out = []
        for t, f in self.run(prompts, 0.0, 600):
            c = json_value(t) if f == "stop" else None
            out.append(c if isinstance(c, dict) and c.get("kind") in ("program", "asserts", "none") else None)
        return out


def plain(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)


def phase_requests():
    t = Teacher()
    rng = random.Random(SEED)
    kinds = [k for k, w, _ in KINDS for _ in range(w)]
    kind_text = {k: d for k, _, d in KINDS}
    specs = []
    for i in range(CONFIG.get("n_calls", 1500)):
        style, style_text = rng.choice(STYLES)
        specs.append({"topic": rng.choice(TOPICS), "style": style, "style_text": style_text, "kind": rng.choice(kinds),
                      "persona": rng.choice(PERSONAS)})
    prompts = [plain(t.tok, REQUEST_PROMPT.format(n=4, persona=s["persona"], topic=s["topic"], kind=kind_text[s["kind"]],
                                                  style=s["style_text"])) for s in specs]
    rows, seen = [], set()
    for s, (text, fin) in zip(specs, t.run(prompts, 0.9, 1200)):
        reqs = json_value(text) if fin == "stop" else None
        if not isinstance(reqs, list):
            continue
        for r in reqs:
            if not isinstance(r, str) or not 12 <= len(r) <= 600:
                continue
            key = re.sub(r"\W+", " ", r.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": rid(r), "request": r.strip(), "topic": s["topic"], "style": s["style"], "kind": s["kind"],
                         "persona": s["persona"]})
    write("requests.jsonl", rows)
    log(f"requests: {len(rows)} from {len(specs)} calls")


def phase_answers():
    t = Teacher()
    reqs = read("requests.jsonl")
    answers = t.answer([[(r["request"], None)] for r in reqs])
    checks = t.checks([r["request"] for r in reqs])
    rows = []
    for r, a, c in zip(reqs, answers, checks):
        if a and parses(a):
            rows.append({**r, "answer": a, "check": c})
    write("answers.jsonl", rows)
    log(f"answers: {len(rows)}/{len(reqs)} parse")


def phase_followups():
    t = Teacher()
    rows = read("answers.jsonl")
    rng = random.Random(SEED + 1)
    pick = [r for r in rows if r["kind"] in ("program", "class", "howto") and rng.random() < CONFIG.get("followup_frac", 0.3)]
    msgs = t.run([plain(t.tok, FOLLOWUP_PROMPT.format(request=r["request"], answer=r["answer"], kind=rng.choice(FOLLOWUP_KINDS)))
                  for r in pick], 0.8, 200)
    convs = []
    for r, (m, fin) in zip(pick, msgs):
        m = m.strip().strip('"')
        if fin == "stop" and 4 <= len(m) <= 400:
            convs.append({"id": rid(r["id"] + m), "base": r["id"], "turns": [[r["request"], r["answer"]], [m, None]],
                          "source": "followup"})
    answers = t.answer([c["turns"] for c in convs])
    checks = t.checks([f"{c['turns'][0][0]}\n\nThen the user asked: {c['turns'][1][0]}" for c in convs])
    out = []
    for c, a, ch in zip(convs, answers, checks):
        if a and parses(a):
            c["turns"][1][1] = a
            out.append({**c, "check": ch})
    write("followups.jsonl", out)
    log(f"followups: {len(out)}/{len(pick)}")


def phase_ophyx():
    from vllm import LLM, SamplingParams
    model = CONFIG.get("ophyx", "abhash-chakraborty/ophyx-350m-v2")
    if model.startswith("/") and "*" in model:
        model = sorted(glob.glob(model, recursive=True))[0]
    rows = read("answers.jsonl")
    rng = random.Random(SEED + 2)
    rng.shuffle(rows)
    rows = rows[:CONFIG.get("ophyx_n", 2500)]
    llm = LLM(model=model, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.85, seed=SEED)
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids("<|end|>")
    prompts = [f"<|user|>\n{r['request']}\n<|python|>\n" for r in rows]

    def params(n, temp):
        return SamplingParams(n=n, temperature=temp, top_p=0.95 if temp else 1.0, max_tokens=1024,
                              stop_token_ids=[end_id, tok.convert_tokens_to_ids("<|user|>")])

    greedy = llm.generate(prompts, params(1, 0.0), use_tqdm=False)
    samples = llm.generate(prompts, params(CONFIG.get("ophyx_samples", 2), 0.7), use_tqdm=False)
    out = []
    for r, g, s in zip(rows, greedy, samples):
        outs = [g.outputs[0]] + list(s.outputs)
        out.append({"id": r["id"], "request": r["request"], "answers": [o.text.strip("\n") for o in outs],
                    "finish": [o.finish_reason for o in outs]})
    write("ophyx.jsonl", out)
    log(f"ophyx: {len(out)} requests x {len(out[0]['answers']) if out else 0}")


def phase_complaints():
    t = Teacher()
    oph = read("ophyx.jsonl")
    # one complaint per request, about the greedy answer: what a visitor to the site would see
    msgs = t.run([plain(t.tok, COMPLAINT_PROMPT.format(request=r["request"], answer=r["answers"][0])) for r in oph], 0.7, 200)
    convs, ok = [], 0
    for r, (m, fin) in zip(oph, msgs):
        m = m.strip().strip('"')
        if fin != "stop" or not m:
            continue
        if re.fullmatch(r"ok\.?", m, re.I):
            ok += 1
            continue
        if len(m) <= 400:
            convs.append({"id": rid(r["id"] + m), "base": r["id"], "turns": [[r["request"], r["answers"][0]], [m, None]],
                          "source": "complaint"})
    answers = t.answer([c["turns"] for c in convs])
    out = []
    for c, a in zip(convs, answers):
        if a and parses(a):
            c["turns"][1][1] = a
            out.append(c)
    write("complaints.jsonl", out)
    log(f"complaints: {len(out)} (teacher said OK to {ok} of {len(oph)})")


PHASES = {"requests": phase_requests, "answers": phase_answers, "followups": phase_followups, "ophyx": phase_ophyx,
          "complaints": phase_complaints}


def child(name, logf):
    os.setsid()  # vLLM's engine core is a child of this process; the parent kills the whole group
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    code = 0
    try:
        PHASES[name]()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    """Kaggle runs a script kernel as a notebook, so there's no file to re-run: fork one process per phase.
    The parent never imports torch or vLLM, so forking is safe."""
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    for f in glob.glob("/kaggle/input/**/chat/*.jsonl", recursive=True):  # resume from an earlier session's output
        if not os.path.exists(os.path.join(OUT, os.path.basename(f))):
            subprocess.run(["cp", f, OUT], check=False)
    if os.path.isdir("/kaggle"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
    ctx = mp.get_context("fork")
    status = {}
    files = {"requests": "requests.jsonl", "answers": "answers.jsonl", "followups": "followups.jsonl",
             "ophyx": "ophyx.jsonl", "complaints": "complaints.jsonl"}
    for name in CONFIG.get("phases", list(PHASES)):
        if os.path.exists(os.path.join(OUT, files[name])):
            log(f"{name}: already done")
            continue
        if time.time() - T0 > BUDGET:
            log(f"{name}: time budget reached, skipped")
            continue
        t = time.time()
        logf = os.path.join(OUT, "logs", f"{name}.log")
        p = ctx.Process(target=child, args=(name, logf))
        p.start()
        p.join(timeout=max(60, BUDGET + 1800 - (time.time() - T0)))
        if p.is_alive():
            p.kill()
            p.join()
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(3)
        status[name] = {"rc": p.exitcode, "seconds": round(time.time() - t)}
        tail = open(logf).read()[-3000:] if os.path.exists(logf) else ""
        log(f"{name}: rc {p.exitcode} in {time.time() - t:.0f}s\n{tail}")
        json.dump({"teacher": TEACHER, "config": CONFIG, "phases": status}, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
        if p.exitcode != 0:
            break


if __name__ == "__main__":
    main()

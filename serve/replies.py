"""Answers the demo gives without the model, and a guard that ends answers stuck in a loop.

Ophyx was trained on Python tasks only, so it has never seen "who are you?". Asked that, it writes a
docstring about some imaginary web app and repeats it. Questions about the model itself get a fixed,
true answer instead; everything else goes to the model. Kept free of Modal and llama.cpp so the
tests can import it.
"""

import re

ABOUT = """\
I'm Ophyx 350M v2, a small code model that writes Python.

Abhash Chakraborty built me from scratch. He wrote my tokenizer and training code and trained me on \
free compute: Kaggle's TPUs and GPUs. I read 12 billion tokens of open code and text, then learned \
from instructions and from programming problems with tests, and finally from reinforcement learning \
on whether my answers passed those tests.

I have 349 million parameters and read up to 4,096 tokens. I score 43.9 on HumanEval+.

What I'm good at: short Python functions from a clear description. What I'm not: chat, facts, other \
languages (I answer those in Python) and long programs. I don't remember earlier messages.

Weights (Apache-2.0): https://huggingface.co/abhash-chakraborty/ophyx-350m-v2
Code and results: https://github.com/Abhash-Chakraborty/ophyx
Abhash's other work: https://abhashchakraborty.tech"""

HELLO = """\
Hi, I'm Ophyx, a small Python code model that Abhash Chakraborty trained from scratch.

Describe a function and I'll write it, for example:
  Write is_palindrome(s) that ignores case and anything that isn't a letter.
  Write chunks(xs, n) that yields lists of length n; the last one may be shorter."""

_IDENTITY = re.compile(
    r"\b(who|what)\s+(are|r)\s+(you|u)\b"
    r"|\byour\s+(name|creator|maker|author|developer|owner|origin)\b"
    r"|\bwho\s+(made|built|created|developed|trained|wrote|owns|designed|programmed)\s+(you|u|this)\b"
    r"|\b(about|introduce)\s+yourself\b"
    r"|\b(what|which)\s+(model|llm|ai|language model)\b"
    r"|\bare\s+(you|u)\s+(an?\s+)?(ai|bot|robot|human|chatgpt|gpt|claude|gemini|llama|qwen|copilot)\b"
    r"|\bwhat\s+can\s+(you|u)\s+do\b"
    r"|\bwhat\s+is\s+ophyx\b|\btell\s+me\s+about\s+(you|ophyx)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|hola|namaste|hii+|sup|good\s+(morning|afternoon|evening))"
    r"(\s+(there|ophyx))?[\s!.,?]*$",
    re.IGNORECASE,
)
# A prompt that asks for code goes to the model even if it mentions "you"
_CODE = re.compile(r"\bdef\b|\bclass\b|\bimport\b|\blambda\b|\bfunction\b|\bimplement\b|```|\w\(.*\)")


def canned_reply(prompt: str) -> str | None:
    """The fixed answer for questions about the model itself, or None to let the model answer."""
    text = prompt.strip()
    if _GREETING.match(text):
        return HELLO
    if len(text) <= 200 and not _CODE.search(text) and _IDENTITY.search(text):
        return ABOUT
    return None


class RepeatGuard:
    """Holds back the newest lines of a streamed answer and ends it when they repeat earlier lines.

    Small models fall into loops, writing the same block again and again until the token limit.
    When the last WINDOW lines match an earlier run of lines word for word, the held lines are
    dropped and `looping` turns true, so the caller stops generating. Short or blank windows don't
    count: real code repeats `return None` and empty lines all the time.
    """

    WINDOW = 3
    MIN_CHARS = 40

    def __init__(self):
        self.lines = []  # every complete line so far, stripped, for matching
        self.held = []  # complete lines not yet sent, as written
        self.partial = ""
        self.looping = False

    def feed(self, chunk: str) -> str:
        """Takes the next chunk of the answer and returns the text that is safe to send now."""
        if self.looping:
            return ""
        self.partial += chunk
        *complete, self.partial = self.partial.split("\n")
        out = []
        for line in complete:
            self.lines.append(line.strip())
            self.held.append(line + "\n")
            if self._repeats():
                self.looping = True
                self.held, self.partial = [], ""
                return "".join(out)
            while len(self.held) > self.WINDOW - 1:
                out.append(self.held.pop(0))
        return "".join(out)

    def flush(self) -> str:
        """The held text, once the answer has ended normally."""
        if self.looping:
            return ""
        rest = "".join(self.held) + self.partial
        self.held, self.partial = [], ""
        return rest

    def _repeats(self) -> bool:
        window = self.lines[-self.WINDOW :]
        if len(window) < self.WINDOW or sum(1 for line in window if line) < 2:
            return False
        if sum(len(line) for line in window) < self.MIN_CHARS:
            return False
        earlier = self.lines[: -self.WINDOW]
        return any(earlier[i : i + self.WINDOW] == window for i in range(len(earlier) - self.WINDOW + 1))


# Prompts that ask for the parts the concise filter removes keep them
_WANTS_EXTRAS = re.compile(
    r"\b(comments?|commented|docstrings?|documentation|document(ed)?|docs|explain\w*|annotat\w*"
    r"|examples?|usage|tests?|testing|pytest|unittest|demo|doctests?)\b",
    re.IGNORECASE,
)
_TAIL = re.compile(
    r"(if\s+__name__\s*==|import\s+(pytest|unittest)\b|from\s+(pytest|unittest)\b|def\s+test_|class\s+Test"
    r"|print\(|assert\b)"
)
_TAIL_COMMENT = re.compile(r"#.*\b(example|usage|test|tests|testing|demo|output)\b", re.IGNORECASE)
_DEF = re.compile(r"(async\s+def|def|class)\s+(\w+)")


def wants_extras(prompt: str) -> bool:
    """True when the prompt asks for comments, docstrings, examples or tests: then nothing is stripped."""
    return bool(_WANTS_EXTRAS.search(prompt))


def _split_comment(line: str, in_string: str | None) -> tuple[str, str | None]:
    """Drops a trailing # comment that sits outside strings. Returns the line and the triple quote
    still open at its end, if any, so the next line knows it starts inside a string."""
    i, quote = 0, in_string
    while i < len(line):
        if quote:
            if line[i] == "\\":
                i += 2
                continue
            if line.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        ch = line[i]
        if ch == "#":
            return line[:i].rstrip(), None
        if ch in "\"'":
            quote = ch * 3 if line.startswith(ch * 3, i) else ch
            i += len(quote)
            continue
        i += 1
    if quote and len(quote) == 1:  # an unclosed one-quote string ends with the line (a syntax error anyway)
        quote = None
    return line, quote


class ConciseFilter:
    """Turns a streamed answer into plain code: no comments, no docstrings and no example or test
    block after the solution. It only removes text, line by line, and never changes a line of code:
    comments and docstrings don't change what the code does, and the tail it cuts (a __main__ demo,
    prints, asserts, pytest tests) only runs when the file is run as a script. A function left with
    no body after its docstring goes gets `pass`. When the tail starts, `done` turns true so the
    caller can stop generating. A __main__ block stays when the code defines main(): that's a CLI.
    """

    def __init__(self):
        self.partial = ""
        self.string = None  # the triple quote of a multi-line string we're inside, if any
        self.docstring = None  # the same, when that string is a docstring being dropped
        self.after_header = None  # indent of a def/class whose body hasn't started yet
        self.open_header = None  # indent of a def/class whose signature spans several lines
        self.pass_indent = None  # a body emptied by a dropped docstring, waiting to see the next line
        self.seen_def = False
        self.has_main = False
        self.blank = 0
        self.started = False
        self.done = False

    def feed(self, chunk: str) -> str:
        if self.done:
            return ""
        self.partial += chunk
        *complete, self.partial = self.partial.split("\n")
        out = []
        for line in complete:
            out.append(self._line(line))
            if self.done:
                break
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.done else self._line(self.partial, final=True) if self.partial else ""
        if self.pass_indent is not None:
            rest += f"{self.pass_indent}    pass\n"
            self.pass_indent = None
        self.partial = ""
        return rest

    def _line(self, line: str, final: bool = False) -> str:
        end = "" if final else "\n"
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if self.docstring:  # inside a docstring being dropped
            if self.docstring in line:
                self.docstring = None
            return ""
        if self.string:  # inside a multi-line string that is data: keep it as written
            _, self.string = _split_comment(line, self.string)
            return line + end

        if not stripped:  # held until more code follows, so nothing trails a cut answer
            if self.started:
                self.blank = min(self.blank + 1, 2)
            return ""

        # a docstring: the first statement after a def/class header, or of the file
        opener = re.match(r"[rRuU]?(\"\"\"|''')", stripped)
        if opener and (self.after_header is not None or not self.started):
            quote = opener.group(1)
            if stripped.count(quote) < 2 or stripped == quote:
                self.docstring = quote
            if self.after_header is not None:
                self.pass_indent = self.after_header
                self.after_header = None
            return ""

        if stripped.startswith("#"):
            if not indent and self.seen_def and _TAIL_COMMENT.match(stripped):
                self.done = True
            return ""

        if not indent and self.seen_def and _TAIL.match(stripped):
            if not (stripped.startswith("if") and self.has_main):
                self.done = True
                return ""

        out = ""
        if self.pass_indent is not None:
            if len(indent) <= len(self.pass_indent):
                out += f"{self.pass_indent}    pass\n"
            self.pass_indent = None
        self.after_header = None

        code, self.string = _split_comment(line, None)
        header = _DEF.match(stripped)
        if header:
            self.seen_def = self.seen_def or not indent
            self.has_main = self.has_main or (not indent and header.group(2) == "main")
            self.open_header = indent
        if self.open_header is not None and code.rstrip().endswith(":"):
            self.after_header, self.open_header = self.open_header, None
        elif self.open_header is not None and header and not code.rstrip().endswith((",", "(", "[", "\\")):
            self.open_header = None  # a one-line def such as `def f(x): return x`
        out += "\n" * self.blank
        self.started = True
        self.blank = 0
        return out + code + end

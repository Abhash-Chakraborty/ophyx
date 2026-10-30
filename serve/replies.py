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

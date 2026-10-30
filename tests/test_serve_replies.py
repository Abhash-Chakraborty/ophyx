import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from replies import ABOUT, HELLO, RepeatGuard, canned_reply  # noqa: E402


def test_questions_about_the_model_get_the_fixed_answer():
    for prompt in [
        "what are you and who developed you?? what are your works and name your creator",
        "Who are you?",
        "who made you",
        "What's your name?",
        "which model is this",
        "are you chatgpt?",
        "tell me about yourself",
        "what can you do",
    ]:
        assert canned_reply(prompt) == ABOUT, prompt
    for prompt in ["hi", "Hello!", "hey there", "good morning"]:
        assert canned_reply(prompt) == HELLO, prompt


def test_code_requests_go_to_the_model():
    for prompt in [
        "Write is_palindrome(s) that ignores case and anything that isn't a letter.",
        "Write a function that returns the n most common words in a text file.",
        "write a class that tells you who made the last commit",
        "hi, write a function that adds two numbers",
    ]:
        assert canned_reply(prompt) is None, prompt


def stream(text, guard, size=7):
    out = "".join(guard.feed(text[i : i + size]) for i in range(0, len(text), size))
    return out + guard.flush()


def test_guard_passes_normal_code_through_unchanged():
    code = 'def f(x):\n    if x:\n        return None\n    return None\n\n\n\ndef g():\n    """Doc."""\n    return 1\n'
    guard = RepeatGuard()
    assert stream(code, guard) == code
    assert not guard.looping


def test_guard_cuts_a_repeated_block():
    block = (
        '"""\nThis is a simple web application that allows users to log in and log out.\n'
        'It uses Python as the programming language.\n"""\n'
    )
    guard = RepeatGuard()
    out = stream("# 1. Purpose\n" + block + "\n# 2. Again\n" + block + block, guard)
    assert guard.looping
    assert out.count("simple web application") == 1
    assert out.startswith("# 1. Purpose\n" + block)

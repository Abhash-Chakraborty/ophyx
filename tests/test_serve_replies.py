import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from replies import ABOUT, HELLO, ConciseFilter, RepeatGuard, canned_reply, wants_extras  # noqa: E402


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


def concise(text, size=5):
    f = ConciseFilter()
    out = "".join(f.feed(text[i : i + size]) for i in range(0, len(text), size))
    return out + f.flush(), f.done


def test_concise_strips_comments_and_docstrings_but_not_code():
    src = (
        "import re  # regex\n"
        "\n"
        "\n"
        "def f(s):\n"
        '    """Count hashes.\n'
        "\n"
        "    More text.\n"
        '    """\n'
        "    # count them\n"
        "    return s.count(\"#\") + len('# not a comment')  # trailing\n"
    )
    out, cut = concise(src)
    assert out == "import re\n\n\ndef f(s):\n    return s.count(\"#\") + len('# not a comment')\n"
    assert not cut


def test_concise_keeps_data_strings_and_fills_empty_bodies():
    src = (
        "def g(\n    a,\n    b,\n) -> str:\n"
        '    """Only a docstring."""\n'
        "\n"
        "class C:\n"
        '    """Doc."""\n'
        "\n"
        "    def h(self):\n"
        '        return """\n'
        "# kept: this is data\n"
        '"""\n'
    )
    out, _ = concise(src)
    assert "    pass\n" in out.split("class C")[0]
    assert "Doc." not in out and "Only a docstring" not in out
    assert "# kept: this is data" in out
    compile(out, "<concise>", "exec")


def test_concise_cuts_the_test_block_and_keeps_a_cli():
    src = "def add(a, b):\n    return a + b\n\n\nimport pytest\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    out, cut = concise(src)
    assert cut and out == "def add(a, b):\n    return a + b\n"
    src = 'def main():\n    print(1)\n\n\nif __name__ == "__main__":\n    main()\n'
    out, cut = concise(src)
    assert not cut and out == src
    src = "def f():\n    return 1\n\n# Example usage\nprint(f())\n"
    out, cut = concise(src)
    assert cut and out == "def f():\n    return 1\n"


def test_prompts_that_ask_for_docs_or_tests_keep_them():
    assert wants_extras("Add type hints and a docstring to this function")
    assert wants_extras("Write pytest tests for is_palindrome")
    assert wants_extras("Explain this code with comments")
    assert not wants_extras("Write is_palindrome(s) that ignores case and anything that isn't a letter.")

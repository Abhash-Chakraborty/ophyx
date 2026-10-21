"""Hands-on prompts: requests a real user would type, each with a reference and hidden pytest tests.

These are for reading the answers by hand, not a leaderboard. They are scored with
eval/concept_suite/run.py (same task schema), and every answer is also read. Like the concept suite
they are eval only: never put them in training data or synthetic-data prompts.

    python eval/manual/build.py        # writes eval/manual/manual.jsonl
"""

import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
T = []


def task(slug, area, prompt, tests, reference, difficulty=2, entry_points=()):
    T.append({"task_id": f"manual/{slug}", "area": area, "leaf": slug, "difficulty": difficulty,
              "prompt": textwrap.dedent(prompt).strip(), "entry_points": list(entry_points),
              "tests": textwrap.dedent(tests).strip() + "\n", "reference": textwrap.dedent(reference).strip() + "\n"})


IMPORTS_ONLY = """
import importlib

def test_module_imports():
    importlib.import_module("solution")
"""

# ---------------------------------------------------------------- everyday scripts

task("word_freq_cli", "scripts", """
    Write a command-line script that prints the 10 most common words in a text file.
    Lower-case the words, treat anything that isn't a letter or an apostrophe as a separator, and
    print one "word count" pair per line, most common first (ties in alphabetical order). Put the
    logic in `top_words(text, n=10)` returning a list of (word, count) tuples, and a
    `main(argv=None)` that takes the file path and an optional `-n` option with argparse.
""", """
from solution import top_words, main

def test_top_words():
    assert top_words("the cat and the hat. The END", 2) == [("the", 3), ("and", 1)]

def test_apostrophes_and_ties():
    assert top_words("don't stop, don't b a", 3) == [("don't", 2), ("a", 1), ("b", 1)]

def test_main(tmp_path, capsys):
    p = tmp_path / "t.txt"
    p.write_text("b a b c b a")
    main([str(p), "-n", "2"])
    assert capsys.readouterr().out.split() == ["b", "3", "a", "2"]
""", """
import argparse
import re
from collections import Counter


def top_words(text, n=10):
    words = re.findall(r"[a-z']+", text.lower())
    counts = Counter(w for w in words if w.strip("'"))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description="most common words in a file")
    ap.add_argument("path")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args(argv)
    with open(a.path, encoding="utf-8") as f:
        for word, count in top_words(f.read(), a.n):
            print(word, count)


if __name__ == "__main__":
    main()
""", entry_points=["top_words", "main"])

task("csv_totals", "data", """
    I have a CSV file with columns date,category,amount (amount is a decimal number, there's a header
    row). Write `category_totals(path)` that returns a dict mapping each category to its total amount,
    rounded to 2 decimals, skipping rows where the amount is empty or not a number. Use the csv module.
""", """
from solution import category_totals

def test_totals(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("date,category,amount\\n2024-01-01,food,10.50\\n2024-01-02,rent,500\\n2024-01-03,food,2.25\\n"
                 "2024-01-04,food,\\n2024-01-05,fun,abc\\n")
    assert category_totals(str(p)) == {"food": 12.75, "rent": 500.0}
""", """
import csv


def category_totals(path):
    totals = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                amount = float(row["amount"])
            except (TypeError, ValueError):
                continue
            totals[row["category"]] = totals.get(row["category"], 0.0) + amount
    return {k: round(v, 2) for k, v in totals.items()}
""", entry_points=["category_totals"])

task("nginx_log_status", "data", """
    Parse web server access log lines in the common log format, like
    `127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326`.
    Write `status_counts(lines)` returning a dict of HTTP status code (int) to count, and
    `top_paths(lines, n)` returning the n most requested paths as a list of (path, count), most first,
    ties by path. Ignore lines that don't match.
""", """
from solution import status_counts, top_paths

LINES = [
    '1.2.3.4 - - [10/Oct/2023:13:55:36 +0000] "GET /a HTTP/1.1" 200 10',
    '1.2.3.4 - - [10/Oct/2023:13:55:37 +0000] "POST /b HTTP/1.1" 404 0',
    'garbage line',
    '5.6.7.8 - bob [10/Oct/2023:13:55:38 +0000] "GET /a HTTP/1.0" 200 99',
    '5.6.7.8 - - [10/Oct/2023:13:55:39 +0000] "GET /c?x=1 HTTP/1.1" 500 -',
]

def test_status():
    assert status_counts(LINES) == {200: 2, 404: 1, 500: 1}

def test_paths():
    assert top_paths(LINES, 2) == [("/a", 2), ("/b", 1)]
""", """
import re
from collections import Counter

LINE = re.compile(r'^(\\S+) \\S+ \\S+ \\[[^\\]]+\\] "(\\S+) (\\S+) [^"]*" (\\d{3}) (\\S+)')


def _parse(lines):
    for line in lines:
        m = LINE.match(line)
        if m:
            yield m.group(3), int(m.group(4))


def status_counts(lines):
    return dict(Counter(s for _, s in _parse(lines)))


def top_paths(lines, n):
    c = Counter(p for p, _ in _parse(lines))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
""", entry_points=["status_counts", "top_paths"])

task("deep_merge", "data", """
    Write `deep_merge(base, override)` for config dictionaries: return a new dict where keys from
    override win, nested dicts are merged recursively, and lists or other values are replaced, not
    merged. Neither input may be modified.
""", """
import copy
from solution import deep_merge

def test_merge():
    a = {"db": {"host": "x", "port": 1}, "debug": False, "tags": [1]}
    b = {"db": {"port": 2}, "debug": True, "tags": [2], "new": {"k": 1}}
    a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
    assert deep_merge(a, b) == {"db": {"host": "x", "port": 2}, "debug": True, "tags": [2], "new": {"k": 1}}
    assert a == a0 and b == b0

def test_no_aliasing():
    a = {"x": {"y": 1}}
    out = deep_merge(a, {})
    out["x"]["y"] = 5
    assert a["x"]["y"] == 1
""", """
import copy


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
""", entry_points=["deep_merge"])

task("retry_decorator", "patterns", """
    Write a decorator factory `retry(times=3, exceptions=(Exception,), delay=0.1, backoff=2.0, sleep=time.sleep)`
    that retries the wrapped function when it raises one of `exceptions`, waiting `delay` seconds
    before the first retry and multiplying the wait by `backoff` each time. After the last attempt
    it re-raises the error. Other exceptions propagate immediately. Keep the function's name and
    docstring.
""", """
import pytest
from solution import retry

def test_retries_then_succeeds():
    waits = []
    calls = {"n": 0}
    @retry(times=3, exceptions=(ValueError,), delay=1, backoff=2, sleep=waits.append)
    def f():
        "doc"
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("no")
        return "ok"
    assert f() == "ok" and calls["n"] == 3 and waits == [1, 2]
    assert f.__name__ == "f" and f.__doc__ == "doc"

def test_gives_up():
    waits = []
    @retry(times=2, exceptions=(ValueError,), delay=0.5, sleep=waits.append)
    def g():
        raise ValueError("always")
    with pytest.raises(ValueError):
        g()
    assert waits == [0.5]

def test_other_exceptions_propagate():
    @retry(times=5, exceptions=(ValueError,), sleep=lambda s: None)
    def h():
        raise KeyError("x")
    with pytest.raises(KeyError):
        h()
""", """
import functools
import time


def retry(times=3, exceptions=(Exception,), delay=0.1, backoff=2.0, sleep=time.sleep):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
                    sleep(wait)
                    wait *= backoff
        return wrapper
    return deco
""", entry_points=["retry"])

task("inventory_dataclass", "classes", """
    Model a small shop inventory. Write a frozen dataclass `Item(sku: str, name: str, price: float, qty: int)`
    that raises ValueError for a negative price or quantity, and a class `Inventory` with
    `add(item)` (raises KeyError if the sku exists), `remove(sku, qty)` (lowers the quantity, raises
    ValueError if not enough stock, deletes the item when it reaches 0), `total_value()` and
    `low_stock(threshold)` returning the skus with qty below the threshold, sorted.
""", """
import dataclasses
import pytest
from solution import Item, Inventory

def test_item_validation():
    with pytest.raises(ValueError):
        Item("a", "A", -1, 1)
    with pytest.raises(ValueError):
        Item("a", "A", 1, -1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Item("a", "A", 1, 1).qty = 3

def test_inventory():
    inv = Inventory()
    inv.add(Item("b", "B", 2.5, 4))
    inv.add(Item("a", "A", 10, 1))
    with pytest.raises(KeyError):
        inv.add(Item("a", "A2", 1, 1))
    assert inv.total_value() == pytest.approx(20)
    assert inv.low_stock(2) == ["a"]
    inv.remove("b", 3)
    assert inv.low_stock(2) == ["a", "b"]
    with pytest.raises(ValueError):
        inv.remove("b", 5)
    inv.remove("a", 1)
    assert inv.low_stock(5) == ["b"]
""", """
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Item:
    sku: str
    name: str
    price: float
    qty: int

    def __post_init__(self):
        if self.price < 0 or self.qty < 0:
            raise ValueError("price and qty must be non-negative")


class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, item):
        if item.sku in self._items:
            raise KeyError(item.sku)
        self._items[item.sku] = item

    def remove(self, sku, qty):
        item = self._items[sku]
        if qty > item.qty:
            raise ValueError("not enough stock")
        if item.qty == qty:
            del self._items[sku]
        else:
            self._items[sku] = replace(item, qty=item.qty - qty)

    def total_value(self):
        return sum(i.price * i.qty for i in self._items.values())

    def low_stock(self, threshold):
        return sorted(s for s, i in self._items.items() if i.qty < threshold)
""", difficulty=3, entry_points=["Item", "Inventory"])

task("duplicate_files", "files", """
    Write `find_duplicates(root)` that walks a directory tree with pathlib and returns groups of files
    with identical content: a list of lists of paths (as strings, relative to root, using "/"), each
    group sorted, groups sorted by their first path, only groups with 2 or more files. Compare by
    size first and hash (sha256, read in chunks) only files whose sizes collide.
""", """
from solution import find_duplicates

def test_dupes(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "a.txt").write_text("same")
    (tmp_path / "d" / "b.txt").write_text("same")
    (tmp_path / "c.txt").write_text("diff")
    (tmp_path / "d" / "e.txt").write_text("other")
    (tmp_path / "f.txt").write_text("other")
    (tmp_path / "g.txt").write_text("samf")
    assert find_duplicates(tmp_path) == [["a.txt", "d/b.txt"], ["d/e.txt", "f.txt"]]

def test_empty(tmp_path):
    assert find_duplicates(tmp_path) == []
""", """
import hashlib
from collections import defaultdict
from pathlib import Path


def _digest(path, chunk=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_duplicates(root):
    root = Path(root)
    by_size = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file():
            by_size[p.stat().st_size].append(p)
    groups = []
    for paths in by_size.values():
        if len(paths) < 2:
            continue
        by_hash = defaultdict(list)
        for p in paths:
            by_hash[_digest(p)].append(p.relative_to(root).as_posix())
        groups += [sorted(g) for g in by_hash.values() if len(g) > 1]
    return sorted(groups, key=lambda g: g[0])
""", difficulty=3, entry_points=["find_duplicates"])

task("pandas_monthly", "data", """
    Using pandas: given a DataFrame `df` with columns `date` (strings like "2024-03-15"), `product`
    and `revenue`, write `monthly_report(df)` that returns a DataFrame indexed by month as "YYYY-MM"
    strings (sorted) with two columns: `revenue` (the month's total) and `top_product` (the product
    with the most revenue that month; ties go to the alphabetically first product).
""", """
import pandas as pd
from solution import monthly_report

def test_report():
    df = pd.DataFrame({"date": ["2024-01-05", "2024-01-20", "2024-02-01", "2024-01-07", "2024-02-02"],
                       "product": ["b", "a", "c", "a", "d"], "revenue": [5.0, 3.0, 1.0, 2.0, 1.0]})
    out = monthly_report(df)
    assert list(out.index) == ["2024-01", "2024-02"]
    assert list(out["revenue"]) == [10.0, 2.0]
    assert list(out["top_product"]) == ["a", "c"]
""", """
import pandas as pd


def monthly_report(df):
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m")
    by = d.groupby(["month", "product"], as_index=False)["revenue"].sum()
    by = by.sort_values(["month", "revenue", "product"], ascending=[True, False, True])
    top = by.groupby("month")["product"].first()
    total = d.groupby("month")["revenue"].sum()
    return pd.DataFrame({"revenue": total, "top_product": top}).sort_index()
""", difficulty=3, entry_points=["monthly_report"])

task("numpy_moving_average", "data", """
    Write `moving_average(x, k)` with numpy: the mean of each window of k consecutive values of a 1-D
    array, as a float array of length len(x) - k + 1. Raise ValueError if k < 1 or k > len(x). Use a
    cumulative sum, not a Python loop.
""", """
import numpy as np
import pytest
from solution import moving_average

def test_values():
    np.testing.assert_allclose(moving_average(np.array([1, 2, 3, 4, 5]), 2), [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(moving_average([2.0, 4.0, 6.0], 3), [4.0])

def test_bad_k():
    for k in (0, 4):
        with pytest.raises(ValueError):
            moving_average([1, 2, 3], k)
""", """
import numpy as np


def moving_average(x, k):
    x = np.asarray(x, dtype=float)
    if k < 1 or k > len(x):
        raise ValueError("k must be between 1 and len(x)")
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k
""", entry_points=["moving_average"])

task("slugify", "text", """
    Write `slugify(title)` for blog URLs: transliterate accented letters to ASCII (use unicodedata),
    lower-case, replace every run of characters that aren't letters or digits with a single hyphen,
    and strip hyphens from both ends. "Héllo, Wörld!  2024" becomes "hello-world-2024".
""", """
from solution import slugify

def test_slug():
    assert slugify("Héllo, Wörld!  2024") == "hello-world-2024"
    assert slugify("  --Already-Slugged--  ") == "already-slugged"
    assert slugify("C'est déjà l'été") == "c-est-deja-l-ete"
    assert slugify("!!!") == ""
""", """
import re
import unicodedata


def slugify(title):
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")
""", difficulty=1, entry_points=["slugify"])

task("business_days", "dates", """
    Write `business_days(start, end, holidays=())` that counts weekdays (Monday to Friday) from date
    `start` to date `end`, both included, skipping any dates in `holidays`. If end is before start,
    return 0. Arguments are datetime.date objects.
""", """
from datetime import date
from solution import business_days

def test_week():
    assert business_days(date(2024, 1, 1), date(2024, 1, 7)) == 5
    assert business_days(date(2024, 1, 6), date(2024, 1, 7)) == 0
    assert business_days(date(2024, 1, 1), date(2024, 1, 31), holidays=[date(2024, 1, 1), date(2024, 1, 6)]) == 22
    assert business_days(date(2024, 1, 5), date(2024, 1, 1)) == 0
""", """
from datetime import timedelta


def business_days(start, end, holidays=()):
    skip = set(holidays)
    n, d = 0, start
    while d <= end:
        if d.weekday() < 5 and d not in skip:
            n += 1
        d += timedelta(days=1)
    return n
""", difficulty=1, entry_points=["business_days"])

# ---------------------------------------------------------------- fixing and changing code

task("fix_traceback", "fixing", """
    This code crashes. Fix it and return the whole corrected module.

    ```python
    def average_by_key(records, key):
        totals = {}
        counts = {}
        for r in records:
            totals[r[key]] += r["value"]
            counts[r[key]] += 1
        return {k: totals[k] / counts[k] for k in totals}
    ```

    ```
    Traceback (most recent call last):
      File "report.py", line 12, in <module>
        print(average_by_key(rows, "city"))
      File "report.py", line 5, in average_by_key
        totals[r[key]] += r["value"]
    KeyError: 'Paris'
    ```
""", """
from solution import average_by_key

def test_avg():
    rows = [{"city": "Paris", "value": 2}, {"city": "Rome", "value": 5}, {"city": "Paris", "value": 4}]
    assert average_by_key(rows, "city") == {"Paris": 3.0, "Rome": 5.0}
    assert average_by_key([], "city") == {}
""", """
def average_by_key(records, key):
    totals = {}
    counts = {}
    for r in records:
        totals[r[key]] = totals.get(r[key], 0) + r["value"]
        counts[r[key]] = counts.get(r[key], 0) + 1
    return {k: totals[k] / counts[k] for k in totals}
""", entry_points=["average_by_key"])

task("fix_no_traceback", "fixing", """
    `chunks([1, 2, 3, 4, 5], 2)` should give `[[1, 2], [3, 4], [5]]` but my function drops the last
    partial chunk and returns `[[1, 2], [3, 4]]`. Fix it.

    ```python
    def chunks(items, size):
        out = []
        for i in range(0, len(items) - size + 1, size):
            out.append(items[i:i + size])
        return out
    ```
""", """
import pytest
from solution import chunks

def test_chunks():
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    assert chunks([], 3) == []
    assert chunks([1], 5) == [[1]]
""", """
def chunks(items, size):
    out = []
    for i in range(0, len(items), size):
        out.append(items[i:i + size])
    return out
""", difficulty=1, entry_points=["chunks"])

task("fix_mutable_default", "fixing", """
    Calling `add_tag("a")` and then `add_tag("b")` returns `["a", "b"]` the second time, but each
    call without a `tags` argument should start from an empty list. Fix this function.

    ```python
    def add_tag(tag, tags=[]):
        tags.append(tag)
        return tags
    ```
""", """
from solution import add_tag

def test_fresh_default():
    assert add_tag("a") == ["a"]
    assert add_tag("b") == ["b"]

def test_given_list():
    t = ["x"]
    assert add_tag("y", t) == ["x", "y"] and t == ["x", "y"]
""", """
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
""", difficulty=1, entry_points=["add_tag"])

task("change_add_param", "changing", """
    Change this function so it takes an optional `currency` argument (default "USD") and formats
    EUR as "€1,234.50" and GBP as "£1,234.50" (USD stays "$1,234.50"). Raise ValueError for any other
    currency. Negative amounts get a leading minus: "-$5.00". Return the whole updated module.

    ```python
    def format_price(amount):
        return f"${amount:,.2f}"
    ```
""", """
import pytest
from solution import format_price

def test_currencies():
    assert format_price(1234.5) == "$1,234.50"
    assert format_price(1234.5, "EUR") == "€1,234.50"
    assert format_price(1234.5, currency="GBP") == "£1,234.50"
    assert format_price(-5) == "-$5.00"
    with pytest.raises(ValueError):
        format_price(1, "JPY")
""", """
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


def format_price(amount, currency="USD"):
    if currency not in SYMBOLS:
        raise ValueError(f"unsupported currency: {currency}")
    sign = "-" if amount < 0 else ""
    return f"{sign}{SYMBOLS[currency]}{abs(amount):,.2f}"
""", entry_points=["format_price"])

task("refactor_to_class", "changing", """
    Turn this into a `Counter`-free class `WordStats` with `add(text)` (can be called many times),
    `count(word)` and `most_common(n)` (list of (word, count), most first, ties alphabetical), keeping
    the same word splitting rule.

    ```python
    def stats(text):
        counts = {}
        for w in text.lower().split():
            w = w.strip(".,!?")
            if w:
                counts[w] = counts.get(w, 0) + 1
        return counts
    ```
""", """
from solution import WordStats

def test_stats():
    s = WordStats()
    s.add("Hi there. hi!")
    s.add("there, you")
    assert s.count("hi") == 2 and s.count("there") == 2 and s.count("nope") == 0
    assert s.most_common(2) == [("hi", 2), ("there", 2)]
    assert s.most_common(5)[-1] == ("you", 1)
""", """
class WordStats:
    def __init__(self):
        self._counts = {}

    def add(self, text):
        for w in text.lower().split():
            w = w.strip(".,!?")
            if w:
                self._counts[w] = self._counts.get(w, 0) + 1

    def count(self, word):
        return self._counts.get(word, 0)

    def most_common(self, n):
        return sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
""", entry_points=["WordStats"])

# ---------------------------------------------------------------- tests, explaining, off-topic

task("write_tests", "testing", """
    Write pytest tests for this function from the module `textutils`. Import it with
    `from textutils import is_palindrome`. Cover normal words, mixed case, spaces and punctuation
    (which are ignored), the empty string and a non-palindrome.

    ```python
    def is_palindrome(s):
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]
    ```
""", """
import subprocess
import sys

GOOD = '''
def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
'''
BUGGY = {
    "case": '''
def is_palindrome(s):
    cleaned = [c for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
''',
    "punct": '''
def is_palindrome(s):
    cleaned = [c.lower() for c in s]
    return cleaned == cleaned[::-1]
''',
    "always": '''
def is_palindrome(s):
    return True
''',
}

def run_with(impl, tmp_path):
    (tmp_path / "textutils.py").write_text(impl)
    (tmp_path / "test_model.py").write_text(open("solution.py").read())
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_model.py"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout[-500:]

def test_passes_on_correct_code(tmp_path):
    rc, out = run_with(GOOD, tmp_path)
    assert rc == 0, out

def test_catches_bugs(tmp_path):
    caught = 0
    for name, impl in BUGGY.items():
        d = tmp_path / name
        d.mkdir()
        rc, _ = run_with(impl, d)
        caught += rc != 0
    assert caught == len(BUGGY)
""", """
import pytest
from textutils import is_palindrome


@pytest.mark.parametrize("s", ["racecar", "level", "Racecar", "A man, a plan, a canal: Panama", ""])
def test_palindromes(s):
    assert is_palindrome(s)


@pytest.mark.parametrize("s", ["hello", "ab"])
def test_not_palindromes(s):
    assert not is_palindrome(s)
""", difficulty=3)

task("explain_code", "explaining", """
    Explain what this function does and when it would be useful. Put the explanation in a docstring
    and comments.

    ```python
    def f(xs):
        seen = set()
        return [x for x in xs if not (x in seen or seen.add(x))]
    ```
""", IMPORTS_ONLY, '''
def f(xs):
    """Remove duplicates from xs while keeping the first occurrence of each item, in order.

    Useful for cleaning a list of ids or tags where order matters. Items must be hashable.
    """
    seen = set()
    # seen.add(x) returns None, so "x in seen or seen.add(x)" is falsy only the first time x appears
    return [x for x in xs if not (x in seen or seen.add(x))]
''', difficulty=1)

task("off_topic_hi", "off_topic", """
    hi! what can you do?
""", IMPORTS_ONLY, '''
"""I write Python: functions, scripts, classes, tests and fixes for broken code.

Describe what you need, paste an error message, or share code you want changed.
"""
''', difficulty=1)

task("explain_recursion", "off_topic", """
    explain recursion to a beginner
""", IMPORTS_ONLY, '''
"""Recursion: a function that solves a problem by calling itself on a smaller version of it.

Every recursive function needs a base case that stops the calls.
"""


def factorial(n):
    if n <= 1:  # base case
        return 1
    return n * factorial(n - 1)  # the same problem, one step smaller
''', difficulty=1)

# ---------------------------------------------------------------- patterns and the standard library

task("async_limited_gather", "async", """
    Write `async def gather_limited(coros, limit)` that runs the given coroutines concurrently with at
    most `limit` running at the same time (use asyncio.Semaphore) and returns their results in the
    original order.
""", """
import asyncio
from solution import gather_limited

def test_order_and_limit():
    running = {"now": 0, "max": 0}
    async def job(i):
        running["now"] += 1
        running["max"] = max(running["max"], running["now"])
        await asyncio.sleep(0.01 * (5 - i % 5))
        running["now"] -= 1
        return i * i
    out = asyncio.run(gather_limited([job(i) for i in range(10)], 3))
    assert out == [i * i for i in range(10)]
    assert running["max"] == 3
""", """
import asyncio


async def gather_limited(coros, limit):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
""", difficulty=3, entry_points=["gather_limited"])

task("ttl_cache", "patterns", """
    Write a class `TTLCache(maxsize, ttl, clock=time.monotonic)`: `get(key, default=None)` and
    `set(key, value)`. Entries expire `ttl` seconds after they were set. When full, setting a new key
    evicts the least recently used entry (a get counts as a use). `len()` counts entries that haven't
    expired.
""", """
from solution import TTLCache

class Clock:
    t = 0.0
    def __call__(self):
        return self.t

def test_ttl_and_lru():
    c = Clock()
    cache = TTLCache(2, 10, clock=c)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    c.t = 11
    assert cache.get("a", "gone") == "gone"
    assert len(cache) == 0
    cache.set("d", 4)
    assert len(cache) == 1
""", """
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, maxsize, ttl, clock=time.monotonic):
        self.maxsize, self.ttl, self.clock = maxsize, ttl, clock
        self._data = OrderedDict()

    def _purge(self):
        now = self.clock()
        for k in [k for k, (_, exp) in self._data.items() if exp <= now]:
            del self._data[k]

    def get(self, key, default=None):
        self._purge()
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key][0]

    def set(self, key, value):
        self._purge()
        if key in self._data:
            self._data.move_to_end(key)
        elif len(self._data) >= self.maxsize:
            self._data.popitem(last=False)
        self._data[key] = (value, self.clock() + self.ttl)

    def __len__(self):
        self._purge()
        return len(self._data)
""", difficulty=3, entry_points=["TTLCache"])

task("sqlite_report", "data", """
    Use sqlite3 with an in-memory database. Write `top_customers(orders, n)` where orders is a list of
    (customer, amount) tuples: create a table, insert the rows with executemany, and use one SQL query
    to return the n customers with the highest total, as a list of (customer, total) tuples, highest
    first, ties by name.
""", """
from solution import top_customers

def test_top():
    orders = [("ann", 10), ("bob", 5), ("ann", 2), ("cid", 12.5), ("bob", 7.5)]
    assert top_customers(orders, 2) == [("bob", 12.5), ("cid", 12.5)]
    assert top_customers([], 3) == []
""", """
import sqlite3


def top_customers(orders, n):
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE orders (customer TEXT, amount REAL)")
        con.executemany("INSERT INTO orders VALUES (?, ?)", orders)
        q = "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC, customer LIMIT ?"
        return [tuple(r) for r in con.execute(q, (n,))]
    finally:
        con.close()
""", entry_points=["top_customers"])

task("requests_json", "web", """
    Write `fetch_user(user_id, session=None)` that GETs https://api.example.com/users/<user_id> with
    the requests library (use the given session or `requests`), a 5 second timeout, and returns the
    JSON body as a dict. Return None on a 404. Raise RuntimeError with the status code for any other
    error status, and also when the request times out.
""", """
import pytest
import requests
from solution import fetch_user

class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body
    def json(self):
        return self._body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

class Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []
    def get(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.resp

def test_ok():
    s = Session(Resp(200, {"id": 7}))
    assert fetch_user(7, s) == {"id": 7}
    url, kw = s.calls[0]
    assert url == "https://api.example.com/users/7" and kw.get("timeout") == 5

def test_404():
    assert fetch_user(1, Session(Resp(404))) is None

def test_500_and_timeout():
    with pytest.raises(RuntimeError, match="500"):
        fetch_user(1, Session(Resp(500)))
    with pytest.raises(RuntimeError):
        fetch_user(1, Session(exc=requests.Timeout("slow")))
""", """
import requests


def fetch_user(user_id, session=None):
    http = session or requests
    try:
        r = http.get(f"https://api.example.com/users/{user_id}", timeout=5)
    except requests.Timeout as e:
        raise RuntimeError("request timed out") from e
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()
""", difficulty=3, entry_points=["fetch_user"])

task("flatten_dict", "data", """
    Write `flatten(d, sep=".")` that turns nested dicts into one level with joined keys:
    {"a": {"b": 1, "c": {"d": 2}}, "e": 3} becomes {"a.b": 1, "a.c.d": 2, "e": 3}. Empty nested dicts
    are kept as values. Also write `unflatten(flat, sep=".")` that reverses it.
""", """
from solution import flatten, unflatten

def test_roundtrip():
    d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3, "f": {}}
    flat = flatten(d)
    assert flat == {"a.b": 1, "a.c.d": 2, "e": 3, "f": {}}
    assert unflatten(flat) == d
    assert flatten({"x": {"y": 1}}, sep="/") == {"x/y": 1}
""", """
def flatten(d, sep=".", prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, sep, key))
        else:
            out[key] = v
    return out


def unflatten(flat, sep="."):
    out = {}
    for key, v in flat.items():
        node = out
        *parents, last = key.split(sep)
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = v
    return out
""", entry_points=["flatten", "unflatten"])

task("timer_context", "patterns", """
    Write a context manager `timer(name, log=print)` (either a class or with contextlib) that measures
    how long the block took with time.perf_counter and calls `log(f"{name}: {seconds:.3f}s")` when the
    block ends, even if it raised. The exception must still propagate.
""", """
import pytest
from solution import timer

def test_logs():
    lines = []
    with timer("work", log=lines.append):
        sum(range(1000))
    assert len(lines) == 1 and lines[0].startswith("work: ") and lines[0].endswith("s")

def test_exception_propagates():
    lines = []
    with pytest.raises(ZeroDivisionError):
        with timer("bad", log=lines.append):
            1 / 0
    assert lines and lines[0].startswith("bad: ")
""", """
import time
from contextlib import contextmanager


@contextmanager
def timer(name, log=print):
    start = time.perf_counter()
    try:
        yield
    finally:
        log(f"{name}: {time.perf_counter() - start:.3f}s")
""", difficulty=1, entry_points=["timer"])

task("parallel_map", "concurrency", """
    Write `parallel_map(fn, items, workers=4)` using concurrent.futures.ThreadPoolExecutor that
    returns results in the input order. If any call raises, re-raise the first exception (by input
    position) after all calls have finished.
""", """
import pytest
import threading
from solution import parallel_map

def test_order():
    assert parallel_map(lambda x: x * 2, range(20), workers=5) == [x * 2 for x in range(20)]

def test_first_error_by_position():
    done = []
    lock = threading.Lock()
    def f(x):
        with lock:
            done.append(x)
        if x in (3, 7):
            raise ValueError(x)
        return x
    with pytest.raises(ValueError) as e:
        parallel_map(f, range(10), workers=3)
    assert e.value.args == (3,)
    assert sorted(done) == list(range(10))
""", """
from concurrent.futures import ThreadPoolExecutor, wait


def parallel_map(fn, items, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, x) for x in items]
        wait(futures)
    for f in futures:
        if f.exception() is not None:
            raise f.exception()
    return [f.result() for f in futures]
""", difficulty=2, entry_points=["parallel_map"])

task("token_bucket", "patterns", """
    Implement a token bucket rate limiter: `TokenBucket(rate, capacity, clock=time.monotonic)` refills
    `rate` tokens per second up to `capacity` (it starts full). `allow(cost=1)` returns True and takes
    the tokens if there are enough, otherwise returns False and takes nothing.
""", """
from solution import TokenBucket

class Clock:
    t = 100.0
    def __call__(self):
        return self.t

def test_bucket():
    c = Clock()
    b = TokenBucket(rate=2, capacity=3, clock=c)
    assert [b.allow() for _ in range(4)] == [True, True, True, False]
    c.t += 0.5
    assert b.allow() and not b.allow()
    c.t += 10
    assert b.allow(3) and not b.allow(1)
    assert not b.allow(5)
""", """
import time


class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens = float(capacity)
        self.last = clock()

    def allow(self, cost=1):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
""", difficulty=2, entry_points=["TokenBucket"])

task("markdown_table", "text", """
    Parse a Markdown table into a list of dicts keyed by the header cells. Strip whitespace from cells,
    skip the separator row (the one made of dashes, colons and pipes), allow optional leading and
    trailing pipes, and ignore blank lines. Function: `parse_table(md)`.
""", """
from solution import parse_table

def test_table():
    md = '''
| name | age |
|:-----|----:|
| Ann  | 31  |
| Bob  |  4  |
'''
    assert parse_table(md) == [{"name": "Ann", "age": "31"}, {"name": "Bob", "age": "4"}]
    assert parse_table("a | b\\n--- | ---\\n1 | 2") == [{"a": "1", "b": "2"}]
""", """
import re


def parse_table(md):
    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [c.strip() for c in line.split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append(cells)
    header, *body = rows
    return [dict(zip(header, r)) for r in body]
""", entry_points=["parse_table"])

task("duration_parse", "text", """
    Write `parse_duration(s)` that turns strings like "1h30m", "45s", "2d 4h" or "1h 2m 3s" into a
    number of seconds (int). Units are d, h, m and s, each used at most once, in that order; spaces
    between parts are allowed. Raise ValueError for anything else, including an empty string.
""", """
import pytest
from solution import parse_duration

def test_ok():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("45s") == 45
    assert parse_duration("2d 4h") == 187200
    assert parse_duration("1h 2m 3s") == 3723

@pytest.mark.parametrize("bad", ["", "5", "1x", "3m2h", "1h1h", "h", "1.5h"])
def test_bad(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)
""", """
import re

PATTERN = re.compile(r"^\\s*(?:(\\d+)d)?\\s*(?:(\\d+)h)?\\s*(?:(\\d+)m)?\\s*(?:(\\d+)s)?\\s*$")


def parse_duration(s):
    m = PATTERN.match(s)
    if not m or not any(m.groups()):
        raise ValueError(f"bad duration: {s!r}")
    d, h, mi, sec = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + sec
""", entry_points=["parse_duration"])

task("calculator", "algorithms", """
    Write `evaluate(expr)` that computes arithmetic expressions with + - * /, parentheses, unary minus
    and decimal numbers, with the usual precedence, without using eval. "2 * (3 + 4) - -1" is 15.
    Division returns a float. Raise ValueError on malformed input.
""", """
import pytest
from solution import evaluate

def test_values():
    assert evaluate("2 * (3 + 4) - -1") == 15
    assert evaluate("1 + 2 * 3") == 7
    assert evaluate("(1 + 2) * 3") == 9
    assert evaluate("7 / 2") == 3.5
    assert evaluate("-(2.5 * 2)") == -5
    assert evaluate("10 - 4 - 3") == 3

@pytest.mark.parametrize("bad", ["1 +", "(1", "2 3", "", "1 $ 2"])
def test_bad(bad):
    with pytest.raises(ValueError):
        evaluate(bad)
""", """
import re

TOKEN = re.compile(r"\\s*(?:(\\d+\\.?\\d*|\\.\\d+)|(.))")


def _tokens(expr):
    out = []
    for num, op in TOKEN.findall(expr):
        if num:
            out.append(float(num))
        elif op.strip():
            if op not in "+-*/()":
                raise ValueError(f"bad character {op!r}")
            out.append(op)
    return out


def evaluate(expr):
    toks = _tokens(expr)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected end")
        pos += 1
        return toks[pos - 1]

    def expression():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v

    def term():
        v = factor()
        while peek() in ("*", "/"):
            v = v * factor() if take() == "*" else v / factor()
        return v

    def factor():
        t = take()
        if t == "-":
            return -factor()
        if t == "+":
            return factor()
        if t == "(":
            v = expression()
            if take() != ")":
                raise ValueError("missing )")
            return v
        if isinstance(t, float):
            return t
        raise ValueError(f"unexpected {t!r}")

    v = expression()
    if pos != len(toks):
        raise ValueError("trailing input")
    return int(v) if v == int(v) and "/" not in expr else v
""", difficulty=4, entry_points=["evaluate"])

task("dijkstra", "algorithms", """
    Write `shortest_path(graph, start, goal)` where graph maps a node to a list of (neighbour, weight)
    pairs with non-negative weights. Return (distance, path) with path as a list of nodes from start
    to goal, using Dijkstra's algorithm with heapq. Return (float("inf"), []) if goal can't be reached.
""", """
from solution import shortest_path

G = {"a": [("b", 1), ("c", 4)], "b": [("c", 2), ("d", 5)], "c": [("d", 1)], "d": [], "e": [("a", 1)]}

def test_paths():
    assert shortest_path(G, "a", "d") == (4, ["a", "b", "c", "d"])
    assert shortest_path(G, "a", "a") == (0, ["a"])
    assert shortest_path(G, "a", "e") == (float("inf"), [])
""", """
import heapq


def shortest_path(graph, start, goal):
    dist = {start: 0}
    prev = {}
    heap = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return d, path[::-1]
        if d > dist.get(node, float("inf")):
            continue
        for nxt, w in graph.get(node, []):
            nd = d + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt], prev[nxt] = nd, node
                heapq.heappush(heap, (nd, nxt))
    return float("inf"), []
""", difficulty=3, entry_points=["shortest_path"])

task("merge_intervals", "algorithms", """
    Merge overlapping intervals: `merge(intervals)` takes a list of [start, end] pairs in any order and
    returns the merged list sorted by start. Intervals that touch ([1, 2] and [2, 3]) merge too.
""", """
from solution import merge

def test_merge():
    assert merge([[8, 10], [1, 3], [2, 6], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge([[1, 2], [2, 3]]) == [[1, 3]]
    assert merge([]) == []
    assert merge([[1, 10], [2, 3]]) == [[1, 10]]
""", """
def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
""", difficulty=1, entry_points=["merge"])

task("roman", "algorithms", """
    Write `to_roman(n)` for 1 to 3999 (ValueError outside that range) and `from_roman(s)` that accepts
    only canonical numerals (ValueError for things like "IIII", "IC" or "").
""", """
import pytest
from solution import to_roman, from_roman

def test_roundtrip():
    for n in (1, 4, 9, 14, 40, 90, 400, 1994, 2024, 3999):
        assert from_roman(to_roman(n)) == n
    assert to_roman(1994) == "MCMXCIV"

@pytest.mark.parametrize("bad", ["IIII", "IC", "", "MMMM", "VX", "abc"])
def test_bad(bad):
    with pytest.raises(ValueError):
        from_roman(bad)

def test_range():
    for n in (0, 4000):
        with pytest.raises(ValueError):
            to_roman(n)
""", """
VALUES = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(n):
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for v, s in VALUES:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def from_roman(s):
    total, i = 0, 0
    for v, sym in VALUES:
        while s.startswith(sym, i):
            total += v
            i += len(sym)
    if not s or i != len(s) or total > 3999 or to_roman(total) != s:
        raise ValueError(f"not a canonical numeral: {s!r}")
    return total
""", difficulty=2, entry_points=["to_roman", "from_roman"])

task("sudoku_valid", "algorithms", """
    Write `is_valid_sudoku(board)` for a 9x9 list of lists of strings, "1"-"9" or "." for empty. Return
    True if no row, column or 3x3 box repeats a digit (the board doesn't need to be solvable).
""", """
from solution import is_valid_sudoku

def board():
    b = [["."] * 9 for _ in range(9)]
    b[0][0], b[1][3], b[4][4], b[8][8] = "5", "5", "1", "9"
    return b

def test_valid():
    assert is_valid_sudoku(board())

def test_row_col_box():
    b = board(); b[0][5] = "5"; assert not is_valid_sudoku(b)
    b = board(); b[7][0] = "5"; assert not is_valid_sudoku(b)
    b = board(); b[2][2] = "5"; assert not is_valid_sudoku(b)
""", """
def is_valid_sudoku(board):
    seen = set()
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == ".":
                continue
            keys = {("r", r, v), ("c", c, v), ("b", r // 3, c // 3, v)}
            if keys & seen:
                return False
            seen |= keys
    return True
""", difficulty=2, entry_points=["is_valid_sudoku"])

task("enum_temperature", "classes", """
    Write an Enum `Unit` with members C, F and K, and `convert(value, src, dst)` that converts a
    temperature between them, rounded to 2 decimals. Raise ValueError for temperatures below absolute
    zero.
""", """
import pytest
from solution import Unit, convert

def test_convert():
    assert convert(100, Unit.C, Unit.F) == 212
    assert convert(32, Unit.F, Unit.C) == 0
    assert convert(0, Unit.C, Unit.K) == 273.15
    assert convert(300, Unit.K, Unit.F) == 80.33
    assert convert(5, Unit.C, Unit.C) == 5

def test_absolute_zero():
    with pytest.raises(ValueError):
        convert(-300, Unit.C, Unit.K)
    with pytest.raises(ValueError):
        convert(-1, Unit.K, Unit.C)
""", """
from enum import Enum


class Unit(Enum):
    C = "C"
    F = "F"
    K = "K"


def convert(value, src, dst):
    kelvin = {Unit.C: value + 273.15, Unit.F: (value - 32) * 5 / 9 + 273.15, Unit.K: value}[src]
    if kelvin < 0:
        raise ValueError("below absolute zero")
    out = {Unit.C: kelvin - 273.15, Unit.F: (kelvin - 273.15) * 9 / 5 + 32, Unit.K: kelvin}[dst]
    return round(out, 2)
""", difficulty=2, entry_points=["Unit", "convert"])

task("generator_pipeline", "patterns", """
    Write generator functions for processing a large log file lazily: `read_lines(path)` yields lines
    without the newline, `grep(lines, pattern)` yields lines matching a regex, and `batched(items, n)`
    yields lists of up to n items. Nothing may read the whole file into memory.
""", """
import types
from solution import read_lines, grep, batched

def test_pipeline(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("ok 1\\nERROR a\\nok 2\\nERROR b\\nERROR c\\n")
    lines = read_lines(str(p))
    assert isinstance(lines, types.GeneratorType)
    out = list(batched(grep(lines, r"^ERROR"), 2))
    assert out == [["ERROR a", "ERROR b"], ["ERROR c"]]

def test_batched_empty():
    assert list(batched(iter([]), 3)) == []
""", """
import re


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\\n")


def grep(lines, pattern):
    rx = re.compile(pattern)
    for line in lines:
        if rx.search(line):
            yield line


def batched(items, n):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
""", difficulty=2, entry_points=["read_lines", "grep", "batched"])

task("password_check", "text", """
    Write `check_password(pw)` that returns a list of the rules a password breaks, in this order, using
    these exact messages: "at least 12 characters", "a lowercase letter", "an uppercase letter",
    "a digit", "a symbol" (anything not a letter, digit or space), "no spaces". An empty list means
    the password is fine.
""", """
from solution import check_password

def test_rules():
    assert check_password("Abcdefghijk1!") == []
    assert check_password("short") == ["at least 12 characters", "an uppercase letter", "a digit", "a symbol"]
    assert check_password("ALLUPPERCASE1!") == ["a lowercase letter"]
    assert check_password("has space Aa1!xx") == ["no spaces"]
""", """
def check_password(pw):
    problems = []
    if len(pw) < 12:
        problems.append("at least 12 characters")
    if not any(c.islower() for c in pw):
        problems.append("a lowercase letter")
    if not any(c.isupper() for c in pw):
        problems.append("an uppercase letter")
    if not any(c.isdigit() for c in pw):
        problems.append("a digit")
    if not any(not c.isalnum() and not c.isspace() for c in pw):
        problems.append("a symbol")
    if any(c.isspace() for c in pw):
        problems.append("no spaces")
    return problems
""", difficulty=1, entry_points=["check_password"])

task("json_schema_lite", "data", """
    Write `validate(data, schema)` for a tiny schema format: a schema is a Python type (int, str,
    float, bool, list, dict), a list with one schema (every element must match it), or a dict mapping
    keys to schemas (all keys required, extra keys allowed; a key ending in "?" is optional). Return a
    list of error strings like "user.age: expected int, got str" (paths joined with ".", list indexes
    as [i]); empty if valid. bool must not count as int.
""", """
from solution import validate

SCHEMA = {"user": {"name": str, "age": int, "email?": str}, "tags": [str], "score": float}

def test_valid():
    assert validate({"user": {"name": "a", "age": 3}, "tags": ["x"], "score": 1.5}, SCHEMA) == []

def test_errors():
    errs = validate({"user": {"name": "a", "age": "3"}, "tags": ["x", 2], "score": True}, SCHEMA)
    assert sorted(errs) == sorted(["user.age: expected int, got str", "tags[1]: expected str, got int",
                                    "score: expected float, got bool"])

def test_missing():
    assert validate({"user": {"name": "a"}, "tags": [], "score": 1.0}, SCHEMA) == ["user.age: missing"]
""", """
def _name(t):
    return t.__name__


def validate(data, schema, path=""):
    errors = []
    if isinstance(schema, dict):
        if not isinstance(data, dict):
            return [f"{path or '<root>'}: expected dict, got {type(data).__name__}"]
        for key, sub in schema.items():
            optional = key.endswith("?")
            k = key[:-1] if optional else key
            p = f"{path}.{k}" if path else k
            if k not in data:
                if not optional:
                    errors.append(f"{p}: missing")
                continue
            errors += validate(data[k], sub, p)
        return errors
    if isinstance(schema, list):
        if not isinstance(data, list):
            return [f"{path}: expected list, got {type(data).__name__}"]
        for i, x in enumerate(data):
            errors += validate(x, schema[0], f"{path}[{i}]")
        return errors
    ok = isinstance(data, schema) and not (schema in (int, float) and isinstance(data, bool))
    return [] if ok else [f"{path}: expected {_name(schema)}, got {type(data).__name__}"]
""", difficulty=4, entry_points=["validate"])

task("lru_function_cache", "patterns", """
    Without functools.lru_cache, write a decorator `memoize(maxsize=128)` that caches results by the
    positional and keyword arguments, evicts the least recently used entry when full, and exposes
    `cache_info()` on the wrapped function returning a dict {"hits", "misses", "size"}.
""", """
from solution import memoize

def test_cache():
    calls = []
    @memoize(maxsize=2)
    def f(x, y=0):
        calls.append((x, y))
        return x + y
    assert f(1) == 1 and f(1) == 1 and f(2, y=3) == 5
    f(4)
    f(1)
    assert calls == [(1, 0), (2, 3), (4, 0), (1, 0)]
    assert f.cache_info() == {"hits": 1, "misses": 4, "size": 2}
""", """
import functools
from collections import OrderedDict


def memoize(maxsize=128):
    def deco(fn):
        cache = OrderedDict()
        stats = {"hits": 0, "misses": 0}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                stats["hits"] += 1
                cache.move_to_end(key)
                return cache[key]
            stats["misses"] += 1
            value = fn(*args, **kwargs)
            cache[key] = value
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return value

        wrapper.cache_info = lambda: {**stats, "size": len(cache)}
        return wrapper
    return deco
""", difficulty=3, entry_points=["memoize"])

task("text_wrap_table", "text", """
    Write `format_table(rows, headers)` that returns a plain-text table as a string: columns padded to
    the widest cell, numbers (int or float) right-aligned and everything else left-aligned, a header
    line, a line of dashes under each header (as wide as the column), cells separated by two spaces,
    no trailing spaces on any line.
""", """
from solution import format_table

def test_table():
    out = format_table([["apple", 3, 1.5], ["kiwi", 12, 0.25]], ["fruit", "qty", "price"])
    assert out.split("\\n") == [
        "fruit  qty  price",
        "-----  ---  -----",
        "apple    3    1.5",
        "kiwi    12   0.25",
    ]
""", """
def format_table(rows, headers):
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    numeric = [bool(rows) and all(isinstance(r[i], (int, float)) for r in rows) for i in range(len(headers))]

    def line(vals):
        parts = [v.rjust(w) if num else v.ljust(w) for v, w, num in zip(vals, widths, numeric)]
        return "  ".join(parts).rstrip()

    out = [line(list(headers)), line(["-" * w for w in widths])]
    out += [line(r) for r in cells]
    return "\\n".join(out)
""", difficulty=2, entry_points=["format_table"])

task("cli_todo", "scripts", """
    Write a small todo-list CLI that stores tasks in a JSON file. Use argparse subcommands: `add TEXT`,
    `list` (prints "1. [ ] text" or "1. [x] text", numbered from 1), `done N` and `remove N`. The file
    path comes from `--file` (default "todo.json"). Put everything in `main(argv=None)`; an invalid
    number prints an error to stderr and returns exit code 1, otherwise return 0.
""", """
from solution import main

def test_flow(tmp_path, capsys):
    f = str(tmp_path / "t.json")
    assert main(["--file", f, "add", "buy milk"]) == 0
    assert main(["--file", f, "add", "write tests"]) == 0
    assert main(["--file", f, "done", "2"]) == 0
    capsys.readouterr()
    assert main(["--file", f, "list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["1. [ ] buy milk", "2. [x] write tests"]
    assert main(["--file", f, "remove", "1"]) == 0
    assert main(["--file", f, "done", "5"]) == 1
    capsys.readouterr()
    main(["--file", f, "list"])
    assert capsys.readouterr().out.splitlines() == ["1. [x] write tests"]
""", """
import argparse
import json
import os
import sys


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="todo")
    ap.add_argument("--file", default="todo.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add").add_argument("text")
    sub.add_parser("list")
    sub.add_parser("done").add_argument("n", type=int)
    sub.add_parser("remove").add_argument("n", type=int)
    a = ap.parse_args(argv)
    items = _load(a.file)
    if a.cmd == "add":
        items.append({"text": a.text, "done": False})
    elif a.cmd == "list":
        for i, t in enumerate(items, 1):
            print(f"{i}. [{'x' if t['done'] else ' '}] {t['text']}")
        return 0
    else:
        if not 1 <= a.n <= len(items):
            print(f"no task {a.n}", file=sys.stderr)
            return 1
        if a.cmd == "done":
            items[a.n - 1]["done"] = True
        else:
            del items[a.n - 1]
    _save(a.file, items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
""", difficulty=3, entry_points=["main"])


if __name__ == "__main__":
    out = os.path.join(HERE, "manual.jsonl")
    with open(out, "w") as f:
        for t in T:
            f.write(json.dumps(t) + "\n")
    print(f"{len(T)} tasks -> {out}")

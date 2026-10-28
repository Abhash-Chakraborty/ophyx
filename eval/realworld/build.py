"""Builds realworld.jsonl: everyday tasks of the kind people ask a coding assistant, with hidden tests.

The hands-on set (eval/manual) is a first look at practical use. This set covers different ground:
logs, CSV and JSON files, dates, CLIs, HTTP clients with a fake session, SQLite, async and threads,
money with Decimal, bug reports written in words, writing tests, and some requests that aren't
Python at all (SQL, shell, JavaScript and plain questions). Those "raw" tasks get the reply as written
in solution.py, and their tests read it as text: SQL runs on an in-memory SQLite database, shell runs
under bash, JavaScript under node, and prose is checked for the one fact it must contain.

    python eval/realworld/build.py      # writes eval/realworld/realworld.jsonl
"""

import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
T = []


def task(slug, area, prompt, tests, reference, difficulty=2, raw=False):
    T.append({"task_id": f"realworld/{slug}", "area": area, "leaf": slug, "difficulty": difficulty,
              "prompt": textwrap.dedent(prompt).strip(), "entry_points": [],
              "tests": textwrap.dedent(tests).strip() + "\n", "reference": textwrap.dedent(reference).strip() + "\n",
              **({"raw": True} if raw else {})})


# Test helper for raw answers: the longest fenced block in the reply, or the whole reply.
ANSWER = r'''
import re
from pathlib import Path


def answer(langs=(), must=None):
    """The answer's code: the longest fenced block in one of langs, preferring blocks that contain
    `must`, so a longer usage example after the solution isn't taken for it."""
    text = Path("solution.py").read_text()
    blocks = re.findall(r"```([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    if langs:
        blocks = [b for b in blocks if b[0].lower() in langs or not b[0]] or blocks
    if must:
        blocks = [b for b in blocks if must in b[1]] or blocks
    return max((b[1] for b in blocks), key=len) if blocks else text
'''

# ---------------------------------------------------------------- files and text

task("log_status_counts", "logs", """
    Write `status_counts(lines)` that takes lines from a web server access log in the combined
    format and returns a dict mapping each HTTP status code (as an int) to the number of requests
    that got it. Skip lines that don't parse.
""", r'''
from solution import status_counts

LINES = [
    '127.0.0.1 - - [10/Oct/2026:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326 "-" "curl/8.0"',
    '10.0.0.2 - bob [10/Oct/2026:13:55:37 +0000] "POST /api/login HTTP/1.1" 401 12 "https://x.test/" "Mozilla/5.0"',
    '10.0.0.2 - - [10/Oct/2026:13:55:38 +0000] "GET /missing HTTP/1.1" 404 0 "-" "Mozilla/5.0 (X11; Linux)"',
    'garbage line',
    '',
    '10.0.0.3 - - [10/Oct/2026:13:56:01 +0000] "GET /a?q=1 HTTP/2.0" 200 - "-" "-"',
]

def test_counts():
    assert status_counts(LINES) == {200: 2, 401: 1, 404: 1}

def test_empty():
    assert status_counts([]) == {}
''', r'''
import re

LINE = re.compile(r'^\S+ \S+ \S+ \[[^\]]+\] "[^"]*" (\d{3}) \S+')


def status_counts(lines):
    counts = {}
    for line in lines:
        m = LINE.match(line)
        if m:
            code = int(m.group(1))
            counts[code] = counts.get(code, 0) + 1
    return counts
''', difficulty=1)

task("csv_group_totals", "data", """
    Write `group_totals(in_path, out_path)`. in_path is a CSV file with the header
    `region,product,amount`, where amount is a decimal number such as 12.50 (product names may be
    quoted and contain commas). Write out_path as a CSV with the header `region,total` and one row per
    region, the total formatted with exactly two decimals, sorted by total (highest first) and then
    by region name. Return the number of regions.
""", r'''
import csv
from solution import group_totals

def test_totals(tmp_path):
    src = tmp_path / "sales.csv"
    src.write_text('region,product,amount\nnorth,"Widget, large",10.25\nsouth,Gadget,5\n'
                   'north,Gizmo,20.25\neast,Gadget,31\nsouth,Widget,25.5\n')
    out = tmp_path / "totals.csv"
    assert group_totals(str(src), str(out)) == 3
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["region", "total"], ["east", "31.00"], ["north", "30.50"], ["south", "30.50"]]
''', r'''
import csv
from decimal import Decimal


def group_totals(in_path, out_path):
    totals = {}
    with open(in_path, newline="") as f:
        for row in csv.DictReader(f):
            totals[row["region"]] = totals.get(row["region"], Decimal(0)) + Decimal(row["amount"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "total"])
        for region, total in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([region, f"{total:.2f}"])
    return len(totals)
''')

task("env_file_parse", "config", """
    Write `parse_env(text)` that returns a dict from the contents of a .env file. Lines are
    KEY=value. Ignore blank lines and lines starting with #. Allow an optional `export ` prefix.
    Strip whitespace around keys and values. A value in single or double quotes keeps everything
    inside the quotes, including #. In an unquoted value, ` #` starts a comment. In double-quoted
    values, the two characters \\n become a newline.
""", r'''
from solution import parse_env

TEXT = """
# comment
export A=1
B = hello world  # trailing
C="quoted # not a comment"
D='single $X \\n'
E="line1\\nline2"
EMPTY=
"""

def test_parse():
    assert parse_env(TEXT) == {"A": "1", "B": "hello world", "C": "quoted # not a comment",
                               "D": "single $X \\n", "E": "line1\nline2", "EMPTY": ""}
''', r'''
def parse_env(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace("\\n", "\n")
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        out[key] = value
    return out
''')

task("extract_links", "text", """
    Write `extract_links(markdown)` that returns the http and https URLs in a Markdown document, in
    order of first appearance and without duplicates. Handle [text](url) links, <url> links and bare
    URLs. A URL never ends with a `.`, `,`, `)` or `>` that belongs to the sentence or the link syntax.
""", r'''
from solution import extract_links

def test_links():
    md = ("See [docs](https://example.com/docs) and https://example.com/a.\n"
          "Also <https://x.test/path?q=1>, and (https://y.test/z). Dup: https://example.com/a")
    assert extract_links(md) == ["https://example.com/docs", "https://example.com/a",
                                 "https://x.test/path?q=1", "https://y.test/z"]

def test_none():
    assert extract_links("no links, just ftp://old.example") == []
''', r'''
import re

URL = re.compile(r"https?://[^\s<>()\[\]\"']+")


def extract_links(markdown):
    seen, out = set(), []
    for m in URL.finditer(markdown):
        url = m.group(0).rstrip(".,;:!?)>")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out
''', difficulty=1)

task("minutes_per_day", "dates", """
    Write `minutes_per_day(sessions)`. sessions is a list of (start, end) ISO 8601 strings with UTC
    offsets, such as "2026-03-01T23:30:00+01:00". Return a dict mapping each UTC date (a
    datetime.date) to the total time on that date in whole minutes, rounded down after adding up the
    seconds. A session that crosses midnight UTC is split between the two dates. Raise ValueError if
    a session ends before it starts.
""", r'''
import datetime as dt
import pytest
from solution import minutes_per_day

def test_split_and_sum():
    got = minutes_per_day([("2026-03-01T23:30:00+01:00", "2026-03-02T01:15:00+01:00"),
                           ("2026-03-02T10:00:00+00:00", "2026-03-02T10:45:30+00:00")])
    assert got == {dt.date(2026, 3, 1): 90, dt.date(2026, 3, 2): 60}

def test_backwards():
    with pytest.raises(ValueError):
        minutes_per_day([("2026-03-02T10:00:00+00:00", "2026-03-02T09:00:00+00:00")])
''', r'''
import datetime as dt


def minutes_per_day(sessions):
    seconds = {}
    for start, end in sessions:
        a = dt.datetime.fromisoformat(start).astimezone(dt.timezone.utc)
        b = dt.datetime.fromisoformat(end).astimezone(dt.timezone.utc)
        if b < a:
            raise ValueError(f"session ends before it starts: {start} -> {end}")
        while a < b:
            midnight = dt.datetime.combine(a.date() + dt.timedelta(days=1), dt.time(), dt.timezone.utc)
            stop = min(b, midnight)
            seconds[a.date()] = seconds.get(a.date(), 0) + (stop - a).total_seconds()
            a = stop
    return {d: int(s // 60) for d, s in seconds.items()}
''')

task("rename_cli", "cli", """
    Write a command-line tool with `main(argv=None)` that renames files in a directory.
    `main([dir, "--ext", ".jpg", "--prefix", "holiday_"])` renames every file with that extension
    (matched case-insensitively) to <prefix><n><ext in lower case>, numbered 001, 002, ... in sorted
    order of the original names. With `--dry-run` it prints one `old -> new` line per file and
    renames nothing. Use argparse and return 0.
""", r'''
import os
from solution import main

def make(d, names):
    for n in names:
        (d / n).write_text(n)

def test_rename(tmp_path):
    make(tmp_path, ["b.jpg", "a.JPG", "c.png", "z.jpg"])
    assert main([str(tmp_path), "--ext", ".jpg", "--prefix", "holiday_"]) == 0
    assert sorted(os.listdir(tmp_path)) == ["c.png", "holiday_001.jpg", "holiday_002.jpg", "holiday_003.jpg"]
    assert (tmp_path / "holiday_001.jpg").read_text() == "a.JPG"
    assert (tmp_path / "holiday_003.jpg").read_text() == "z.jpg"

def test_dry_run(tmp_path, capsys):
    make(tmp_path, ["b.jpg", "a.JPG"])
    main([str(tmp_path), "--ext", ".jpg", "--prefix", "x", "--dry-run"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["a.JPG -> x001.jpg", "b.jpg -> x002.jpg"]
    assert sorted(os.listdir(tmp_path)) == ["a.JPG", "b.jpg"]
''', r'''
import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser(description="Number files with one extension.")
    ap.add_argument("dir")
    ap.add_argument("--ext", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ext = a.ext.lower()
    names = sorted(n for n in os.listdir(a.dir)
                   if n.lower().endswith(ext) and os.path.isfile(os.path.join(a.dir, n)))
    for i, old in enumerate(names, 1):
        new = f"{a.prefix}{i:03d}{ext}"
        if a.dry_run:
            print(f"{old} -> {new}")
        else:
            os.rename(os.path.join(a.dir, old), os.path.join(a.dir, new))
    return 0
''')

task("find_duplicates", "files", """
    Write `find_duplicates(root)` that walks a directory tree and returns groups of files with
    identical contents: a list of lists of paths relative to root (with / separators), each group
    sorted, the groups sorted by their first path, and only groups of two or more files. Compare
    sizes first and hash (sha256, read in chunks) only files whose size matches another file's.
""", r'''
from solution import find_duplicates

def test_groups(tmp_path):
    (tmp_path / "sub").mkdir()
    for name, data in [("a.txt", "x"), ("sub/b.txt", "x"), ("c.txt", "y"), ("d.txt", ""), ("sub/e.txt", ""),
                       ("f.txt", "z")]:
        (tmp_path / name).write_text(data)
    assert find_duplicates(str(tmp_path)) == [["a.txt", "sub/b.txt"], ["d.txt", "sub/e.txt"]]
''', r'''
import hashlib
import os


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(root):
    by_size = {}
    for d, _, files in os.walk(root):
        for name in files:
            p = os.path.join(d, name)
            by_size.setdefault(os.path.getsize(p), []).append(p)
    groups = {}
    for paths in by_size.values():
        if len(paths) > 1:
            for p in paths:
                groups.setdefault(_digest(p), []).append(os.path.relpath(p, root).replace(os.sep, "/"))
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)
''')

# ---------------------------------------------------------------- services and data stores

task("paginate_api", "http", """
    Write `fetch_all(session, url, max_pages=50)` for a JSON API whose pages look like
    {"items": [...], "next": <url or null>}. Use `session.get(url, timeout=10)`, follow `next` until it
    is null, and return all the items in order. Raise RuntimeError if a page comes back with a status
    other than 200, or if there would be more than max_pages pages.
""", r'''
import json
import pytest
from solution import fetch_all

class Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body
        self.ok, self.text = code < 400, json.dumps(body)
    def json(self):
        return self._body
    def raise_for_status(self):  # as in requests
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

class Session:
    def __init__(self, pages):
        self.pages, self.calls = pages, []
    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.pages[url]

def test_follows_next():
    s = Session({"u1": Resp(200, {"items": [1, 2], "next": "u2"}), "u2": Resp(200, {"items": [], "next": "u3"}),
                 "u3": Resp(200, {"items": [3], "next": None})})
    assert fetch_all(s, "u1") == [1, 2, 3]
    assert [c[0] for c in s.calls] == ["u1", "u2", "u3"]
    assert all(c[1].get("timeout") == 10 for c in s.calls)

def test_error_status():
    s = Session({"u1": Resp(200, {"items": [1], "next": "u2"}), "u2": Resp(503, {"error": "busy"})})
    with pytest.raises(RuntimeError):
        fetch_all(s, "u1")

def test_loop_stops():
    s = Session({"u1": Resp(200, {"items": [1], "next": "u1"})})
    with pytest.raises(RuntimeError):
        fetch_all(s, "u1", max_pages=5)
    assert len(s.calls) <= 6
''', r'''
def fetch_all(session, url, max_pages=50):
    items, pages = [], 0
    while url:
        pages += 1
        if pages > max_pages:
            raise RuntimeError(f"more than {max_pages} pages")
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"{url}: HTTP {r.status_code}")
        body = r.json()
        items.extend(body.get("items", []))
        url = body.get("next")
    return items
''')

task("retry_backoff", "patterns", """
    Write a decorator factory `retry(exceptions, tries=3, base_delay=0.5, sleep=time.sleep)`. The
    decorated function is called; if it raises one of `exceptions`, wait base_delay seconds, then
    twice that, then four times that, and so on, and call it again, up to `tries` calls in total.
    After the last call, re-raise its error. Other exceptions propagate at once. Keep the wrapped
    function's name and docstring.
""", r'''
import pytest
from solution import retry

def test_backoff_then_success():
    sleeps, calls = [], []
    @retry((ConnectionError,), tries=4, base_delay=0.5, sleep=sleeps.append)
    def flaky():
        "doc"
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "ok"
    assert flaky() == "ok"
    assert sleeps == [0.5, 1.0]
    assert flaky.__name__ == "flaky" and flaky.__doc__ == "doc"

def test_gives_up():
    sleeps = []
    @retry((TimeoutError,), tries=3, base_delay=1, sleep=sleeps.append)
    def never():
        raise TimeoutError("slow")
    with pytest.raises(TimeoutError):
        never()
    assert sleeps == [1, 2]

def test_other_errors_pass_through():
    sleeps = []
    @retry((ConnectionError,), sleep=sleeps.append)
    def bad():
        raise ValueError("no")
    with pytest.raises(ValueError):
        bad()
    assert sleeps == []
''', r'''
import functools
import time


def retry(exceptions, tries=3, base_delay=0.5, sleep=time.sleep):
    exceptions = tuple(exceptions) if not isinstance(exceptions, type) else (exceptions,)

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == tries:
                        raise
                    sleep(delay)
                    delay *= 2
        return wrapper
    return deco
''')

task("sqlite_upsert", "database", """
    Write `upsert_users(conn, users)`. conn is a sqlite3 connection with a table
    `users(email TEXT PRIMARY KEY, name TEXT, updated_at TEXT)`. users is a list of dicts with
    email, name and updated_at (ISO 8601 strings). Store emails in lower case and treat them
    case-insensitively. Insert unknown emails. For known ones, update name and updated_at only if the
    incoming updated_at is newer than the stored one. Do it in one transaction and return
    (inserted, updated).
""", r'''
import sqlite3
from solution import upsert_users

def db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE users(email TEXT PRIMARY KEY, name TEXT, updated_at TEXT)")
    c.execute("INSERT INTO users VALUES ('a@x.com', 'Ann', '2026-01-01T00:00:00')")
    c.commit()
    return c

def test_upsert():
    c = db()
    got = upsert_users(c, [{"email": "A@X.com", "name": "Ann B", "updated_at": "2026-02-01T00:00:00"},
                           {"email": "b@x.com", "name": "Bo", "updated_at": "2026-02-01T00:00:00"}])
    assert got == (1, 1)
    assert c.execute("SELECT * FROM users ORDER BY email").fetchall() == [
        ("a@x.com", "Ann B", "2026-02-01T00:00:00"), ("b@x.com", "Bo", "2026-02-01T00:00:00")]

def test_older_is_ignored():
    c = db()
    assert upsert_users(c, [{"email": "a@x.com", "name": "Old", "updated_at": "2025-06-01T00:00:00"}]) == (0, 0)
    assert c.execute("SELECT name FROM users").fetchall() == [("Ann",)]
''', r'''
def upsert_users(conn, users):
    inserted = updated = 0
    with conn:
        for u in users:
            email = u["email"].lower()
            row = conn.execute("SELECT updated_at FROM users WHERE email = ?", (email,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO users(email, name, updated_at) VALUES (?, ?, ?)",
                             (email, u["name"], u["updated_at"]))
                inserted += 1
            elif u["updated_at"] > row[0]:
                conn.execute("UPDATE users SET name = ?, updated_at = ? WHERE email = ?",
                             (u["name"], u["updated_at"], email))
                updated += 1
    return inserted, updated
''')

task("order_total", "money", """
    Write dataclasses `LineItem(sku, unit_price, qty)` and `Order(items, discount_code=None)` with a
    method `Order.total()` that returns a Decimal: the sum of unit_price * qty, minus the discount,
    plus 8.25% tax on the discounted amount, rounded once at the end to cents with ROUND_HALF_UP.
    unit_price is given as a string such as "19.99" and must be stored as a Decimal. Discount codes:
    "SAVE10" takes 10% off and "FLAT5" takes 5.00 off (never going below zero). Any other code raises
    ValueError when the Order is created, and so does a qty that isn't a positive int.
""", r'''
from decimal import Decimal
import pytest
from solution import LineItem, Order

ITEMS = [LineItem("A", "19.99", 2), LineItem("B", "5.00", 1)]

def test_totals():
    assert Order(ITEMS).total() == Decimal("48.69")
    assert Order(ITEMS, "SAVE10").total() == Decimal("43.82")
    assert Order(ITEMS, "FLAT5").total() == Decimal("43.28")
    assert Order([LineItem("C", "3.00", 1)], "FLAT5").total() == Decimal("0.00")

def test_half_up():
    assert Order([LineItem("C", "10.00", 1)]).total() == Decimal("10.83")

def test_price_is_decimal():
    assert isinstance(LineItem("A", "0.10", 3).unit_price, Decimal)

def test_bad_input():
    with pytest.raises(ValueError):
        Order(ITEMS, "FREE")
    with pytest.raises(ValueError):  # the prompt allows the check in LineItem or in Order
        Order([LineItem("A", "1.00", 0)])
''', r'''
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

TAX = Decimal("0.0825")
CODES = {None, "SAVE10", "FLAT5"}


@dataclass
class LineItem:
    sku: str
    unit_price: Decimal
    qty: int

    def __post_init__(self):
        self.unit_price = Decimal(str(self.unit_price))
        if not isinstance(self.qty, int) or isinstance(self.qty, bool) or self.qty < 1:
            raise ValueError(f"qty must be a positive int, got {self.qty!r}")


@dataclass
class Order:
    items: list = field(default_factory=list)
    discount_code: str = None

    def __post_init__(self):
        if self.discount_code not in CODES:
            raise ValueError(f"unknown discount code {self.discount_code!r}")

    def total(self):
        subtotal = sum((i.unit_price * i.qty for i in self.items), Decimal(0))
        if self.discount_code == "SAVE10":
            subtotal -= subtotal * Decimal("0.10")
        elif self.discount_code == "FLAT5":
            subtotal = max(Decimal(0), subtotal - Decimal("5.00"))
        return (subtotal * (1 + TAX)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
''', difficulty=3)

# ---------------------------------------------------------------- concurrency

task("async_fetch_all", "async", """
    Write `async def fetch_all(urls, fetch, limit=5, timeout=2.0)`. Call the coroutine function
    `fetch(url)` for every URL, with at most `limit` calls running at once and a timeout on each call.
    Return a list in the same order as urls holding each result, or the exception object for a call
    that raised or timed out.
""", r'''
import asyncio
from solution import fetch_all

def test_order_limit_and_errors():
    running, peak = 0, 0

    async def fetch(url):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        try:
            if url == "bad":
                raise ValueError(url)
            await asyncio.sleep(0.5 if url == "slow" else 0.02)
            return url.upper()
        finally:
            running -= 1

    urls = ["a", "bad", "b", "slow", "c", "d", "e", "f"]
    got = asyncio.run(fetch_all(urls, fetch, limit=3, timeout=0.2))
    assert got[0] == "A" and got[2] == "B" and got[4:] == ["C", "D", "E", "F"]
    assert isinstance(got[1], ValueError)
    assert isinstance(got[3], (asyncio.TimeoutError, TimeoutError))
    assert 1 < peak <= 3
''', r'''
import asyncio


async def fetch_all(urls, fetch, limit=5, timeout=2.0):
    sem = asyncio.Semaphore(limit)

    async def one(url):
        async with sem:
            try:
                return await asyncio.wait_for(fetch(url), timeout)
            except Exception as e:
                return e

    return await asyncio.gather(*(one(u) for u in urls))
''', difficulty=3)

task("worker_pool", "threads", """
    Write `process_all(items, func, workers=4)` that runs func(item) for every item on `workers`
    threads that you start yourself with threading and queue.Queue (not concurrent.futures), and
    returns the results in input order. If any call raises, let the other calls finish, then raise
    the exception of the earliest failing item.
""", r'''
import threading
import time
from pathlib import Path
import pytest
from solution import process_all

def test_order_and_threads():
    seen = set()
    def f(x):
        seen.add(threading.get_ident())
        time.sleep(0.02)
        return x * x
    assert process_all(list(range(20)), f, workers=4) == [x * x for x in range(20)]
    assert 1 < len(seen) <= 4

def test_first_error_after_all_ran():
    done = []
    def f(x):
        time.sleep(0.01)
        if x in (3, 7):
            raise ValueError(x)
        done.append(x)
        return x
    with pytest.raises(ValueError) as e:
        process_all(list(range(10)), f, workers=3)
    assert e.value.args == (3,)
    assert sorted(done) == [0, 1, 2, 4, 5, 6, 8, 9]

def test_no_executor():
    assert "concurrent.futures" not in Path("solution.py").read_text()
''', r'''
import queue
import threading


def process_all(items, func, workers=4):
    q = queue.Queue()
    for i, item in enumerate(items):
        q.put((i, item))
    results, errors = [None] * len(items), {}

    def work():
        while True:
            try:
                i, item = q.get_nowait()
            except queue.Empty:
                return
            try:
                results[i] = func(item)
            except Exception as e:
                errors[i] = e

    threads = [threading.Thread(target=work) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[min(errors)]
    return results
''')

task("rate_limiter", "patterns", """
    Write a class `RateLimiter(limit, window, clock=time.monotonic)` with a method `allow(key)` that
    returns True if fewer than `limit` calls for that key were allowed in the last `window` seconds
    (a sliding window, not fixed time buckets), and records the call when it is allowed. Keys are
    independent. Don't keep more than `limit` timestamps per key.
""", r'''
from solution import RateLimiter

def test_sliding_window():
    now = [0.0]
    rl = RateLimiter(2, 10, clock=lambda: now[0])
    seq = []
    for t in (0, 1, 5, 10.5, 11.5, 12):
        now[0] = t
        seq.append(rl.allow("u"))
    assert seq == [True, True, False, True, True, False]

def test_keys_are_independent():
    now = [0.0]
    rl = RateLimiter(1, 60, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("b") and not rl.allow("a")
''', r'''
import collections
import time


class RateLimiter:
    def __init__(self, limit, window, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.calls = collections.defaultdict(lambda: collections.deque(maxlen=limit))

    def allow(self, key):
        now = self.clock()
        q = self.calls[key]
        while q and q[0] <= now - self.window:
            q.popleft()
        if len(q) < self.limit:
            q.append(now)
            return True
        return False
''')

# ---------------------------------------------------------------- small libraries

task("markdown_table", "text", """
    Write `markdown_table(rows, columns=None)`. rows is a list of dicts; columns defaults to the keys
    of the first row. Return a GitHub Markdown table as a string with no trailing newline: a header
    row, a separator row, then one row per dict. Missing values are empty cells and `|` in a cell is
    written as `\\|`. Pad every column to its widest cell (at least 3). A column whose present values
    are all numbers (int or float, not bool) is right-aligned, header included, and its separator
    ends in `:`. For example rows=[{"name": "apple", "qty": 3}, {"name": "kiwi", "qty": 12}] gives

    | name  | qty |
    | ----- | --: |
    | apple |   3 |
    | kiwi  |  12 |
""", r'''
from solution import markdown_table

def test_example():
    assert markdown_table([{"name": "apple", "qty": 3}, {"name": "kiwi", "qty": 12}]) == (
        "| name  | qty |\n| ----- | --: |\n| apple |   3 |\n| kiwi  |  12 |")

def test_missing_and_pipes():
    got = markdown_table([{"a": "x|y"}, {"a": "z", "b": 1.5}], columns=["a", "b"])
    assert got == "| a    |   b |\n| ---- | --: |\n| x\\|y |     |\n| z    | 1.5 |"
''', r'''
def markdown_table(rows, columns=None):
    columns = list(columns if columns is not None else (rows[0].keys() if rows else []))

    def cell(v):
        return "" if v is None else str(v).replace("|", "\\|")

    numeric = {}
    for c in columns:
        vals = [r[c] for r in rows if c in r and r[c] is not None]
        numeric[c] = bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
    width = {c: max([3, len(c)] + [len(cell(r.get(c))) for r in rows]) for c in columns}

    def line(values):
        parts = [v.rjust(width[c]) if numeric[c] else v.ljust(width[c]) for c, v in zip(columns, values)]
        return "| " + " | ".join(parts) + " |"

    sep = "| " + " | ".join("-" * (width[c] - 1) + ":" if numeric[c] else "-" * width[c] for c in columns) + " |"
    return "\n".join([line(columns), sep] + [line([cell(r.get(c)) for c in columns]) for r in rows])
''', difficulty=1)

task("run_with_timeout", "subprocess", """
    Write `run(cmd, timeout=10, input=None)` that runs a command (a list of strings) without a shell
    and returns (returncode, stdout, stderr) as text. `input` is text to send on stdin. If the command
    runs longer than timeout seconds, kill it and raise TimeoutError. If the program doesn't exist,
    raise FileNotFoundError.
""", r'''
import sys
import pytest
from solution import run

def test_output():
    code, out, err = run([sys.executable, "-c", "import sys; print(sys.stdin.read().upper()); "
                          "print('e', file=sys.stderr); sys.exit(3)"], input="hi")
    assert (code, out.strip(), err.strip()) == (3, "HI", "e")

def test_timeout():
    with pytest.raises(TimeoutError):
        run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5)

def test_missing():
    with pytest.raises(FileNotFoundError):
        run(["definitely-not-a-program-xyz"])
''', r'''
import subprocess


def run(cmd, timeout=10, input=None):
    try:
        p = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"{cmd[0]} ran longer than {timeout}s") from e
    return p.returncode, p.stdout, p.stderr
''')

task("jsonl_stats", "data", """
    Write `average_by(path, key, value)`. Read a JSON Lines file and compute, for each distinct
    record[key], the mean of record[value] over the records that have both fields and a numeric value
    (int or float, not bool). Skip blank lines silently. Lines that aren't valid JSON objects are
    skipped and counted. Return (averages, skipped).
""", r'''
from solution import average_by

def test_stats(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"user": "a", "ms": 10}\n{"user": "a", "ms": 20}\n\n{"user": "b", "ms": 5.5}\n'
                 'not json\n[1, 2]\n{"user": "b", "ms": "fast"}\n{"user": "c"}\n{"user": "b", "ms": true}\n')
    avg, skipped = average_by(str(p), "user", "ms")
    assert avg == {"a": 15, "b": 5.5}
    assert skipped == 2
''', r'''
import json


def average_by(path, key, value):
    sums, counts, skipped = {}, {}, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(rec, dict):
                skipped += 1
                continue
            v = rec.get(value)
            if key in rec and isinstance(v, (int, float)) and not isinstance(v, bool):
                k = rec[key]
                sums[k] = sums.get(k, 0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}, skipped
''')

task("password_policy", "validation", """
    Write `check_password(pw, username="")` that returns a list of problems, empty if the password is
    fine, in this order and with these exact strings: "too short" (fewer than 12 characters), "no
    uppercase", "no lowercase", "no digit", "no symbol" (a symbol is any character that isn't a letter,
    a digit or a space), and "contains username" (case-insensitive, checked only when the username has
    at least 3 characters).
""", r'''
from solution import check_password

def test_good():
    assert check_password("Correct-Horse-9", "bob") == []

def test_problems():
    assert check_password("abc") == ["too short", "no uppercase", "no digit", "no symbol"]
    assert check_password("ALICE-in-chains-7", "alice") == ["contains username"]
    assert check_password("Aaaaaaaaaaa1 ", "al") == ["no symbol"]
''', r'''
def check_password(pw, username=""):
    problems = []
    if len(pw) < 12:
        problems.append("too short")
    if not any(c.isupper() for c in pw):
        problems.append("no uppercase")
    if not any(c.islower() for c in pw):
        problems.append("no lowercase")
    if not any(c.isdigit() for c in pw):
        problems.append("no digit")
    if not any(not c.isalnum() and not c.isspace() for c in pw):
        problems.append("no symbol")
    if len(username) >= 3 and username.lower() in pw.lower():
        problems.append("contains username")
    return problems
''', difficulty=1)

task("semver_sort", "sorting", """
    Write `sort_versions(versions)` that sorts Semantic Versioning 2.0 strings by precedence, lowest
    first, and returns the original strings. MAJOR.MINOR.PATCH compare as numbers. A pre-release
    (after `-`) sorts before its release. Pre-release identifiers compare one dot-separated part at a
    time: numeric parts as numbers and below alphanumeric parts, which compare as ASCII, and a
    shorter list sorts first when the rest is equal. Build metadata (after `+`) is ignored. Accept an
    optional leading "v".
""", r'''
from solution import sort_versions

def test_spec_order():
    want = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
            "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.0.1+build.5", "1.2.0", "1.10.0", "v2.0.0"]
    shuffled = [want[i] for i in (7, 3, 11, 0, 9, 5, 1, 10, 2, 8, 6, 4)]
    assert sort_versions(shuffled) == want
''', r'''
def _key(v):
    v = v[1:] if v.startswith("v") else v
    v = v.split("+", 1)[0]
    core, _, pre = v.partition("-")
    nums = tuple(int(x) for x in core.split("."))
    if not pre:
        return nums, (1,)
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return nums, (0, parts)


def sort_versions(versions):
    return sorted(versions, key=_key)
''', difficulty=3)

task("html_page_info", "parsing", """
    Write `page_info(html)` using only the standard library (html.parser). Return a dict with
    "title": the text of the first <title>, with whitespace collapsed to single spaces and entities
    decoded, or None if there is none; and "links": the href of every <a> tag in order, skipping
    missing or empty hrefs and those starting with "#" or "javascript:".
""", r'''
from solution import page_info

HTML = """<!doctype html><html><head><title>
  Tom &amp; Jerry   - Home</title></head>
<body><a href="/a">A</a><a>no href</a><a href="#top">top</a><A HREF="https://x.test/b?c=1&amp;d=2">B</A>
<a href="javascript:void(0)">js</a><a href="">empty</a></body></html>"""

def test_info():
    assert page_info(HTML) == {"title": "Tom & Jerry - Home", "links": ["/a", "https://x.test/b?c=1&d=2"]}

def test_no_title():
    assert page_info("<p>hi</p>") == {"title": None, "links": []}
''', r'''
from html.parser import HTMLParser


class _Info(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.links, self._in_title, self._buf = None, [], False, []

    def handle_starttag(self, tag, attrs):
        if tag == "title" and self.title is None:
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("#", "javascript:")):
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self._in_title = False
            self.title = " ".join("".join(self._buf).split())

    def handle_data(self, data):
        if self._in_title:
            self._buf.append(data)


def page_info(html):
    p = _Info()
    p.feed(html)
    p.close()
    return {"title": p.title, "links": p.links}
''')

task("diff_configs", "data", """
    Write `diff(old, new)` for two nested dicts. Return a sorted list of strings: "+ path = value" for
    added keys, "- path" for removed keys and "~ path: old -> new" for changed values, where path joins
    nested keys with "." and values are shown with repr(). Recurse into values that are dicts on both
    sides; anything else, lists included, compares as a whole.
""", r'''
from solution import diff

def test_diff():
    old = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1]}
    new = {"a": 1, "b": {"c": 5, "f": "x"}, "e": [1, 2], "g": None}
    assert diff(old, new) == ["+ b.f = 'x'", "+ g = None", "- b.d", "~ b.c: 2 -> 5", "~ e: [1] -> [1, 2]"]

def test_same():
    assert diff({"a": {"b": 1}}, {"a": {"b": 1}}) == []
''', r'''
def diff(old, new, prefix=""):
    out = []
    for k in old.keys() | new.keys():
        path = f"{prefix}{k}"
        if k not in new:
            out.append(f"- {path}")
        elif k not in old:
            out.append(f"+ {path} = {new[k]!r}")
        elif isinstance(old[k], dict) and isinstance(new[k], dict):
            out.extend(diff(old[k], new[k], path + "."))
        elif old[k] != new[k]:
            out.append(f"~ {path}: {old[k]!r} -> {new[k]!r}")
    return sorted(out)
''')

task("bank_account", "classes", """
    Write a class `Account(owner, overdraft=0)` with deposit(amount), withdraw(amount) and
    transfer(to, amount), a read-only `balance` property, and `history`: a list of
    (kind, amount, balance_after) tuples with kinds "deposit", "withdraw", "transfer_out" and
    "transfer_in". Amounts must be positive numbers (ValueError otherwise). A withdrawal or transfer
    that would take the balance below -overdraft raises InsufficientFunds (define it as a subclass of
    Exception) and changes nothing.
""", r'''
import pytest
from solution import Account, InsufficientFunds

def test_flow():
    a, b = Account("ann", overdraft=50), Account("bo")
    a.deposit(100)
    a.withdraw(120)
    a.transfer(b, 30)
    assert a.balance == -50 and b.balance == 30
    assert a.history == [("deposit", 100, 100), ("withdraw", 120, -20), ("transfer_out", 30, -50)]
    assert b.history == [("transfer_in", 30, 30)]

def test_limits():
    a, b = Account("ann"), Account("bo")
    a.deposit(10)
    with pytest.raises(InsufficientFunds):
        a.transfer(b, 11)
    assert a.balance == 10 and b.balance == 0 and len(a.history) == 1
    with pytest.raises(ValueError):
        a.deposit(0)
    with pytest.raises(AttributeError):
        a.balance = 5
''', r'''
class InsufficientFunds(Exception):
    pass


class Account:
    def __init__(self, owner, overdraft=0):
        self.owner, self.overdraft = owner, overdraft
        self._balance = 0
        self.history = []

    @property
    def balance(self):
        return self._balance

    @staticmethod
    def _check(amount):
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError(f"amount must be a positive number, got {amount!r}")

    def _move(self, kind, amount):
        self._balance += amount if kind in ("deposit", "transfer_in") else -amount
        self.history.append((kind, amount, self._balance))

    def deposit(self, amount):
        self._check(amount)
        self._move("deposit", amount)

    def _can_take(self, amount):
        self._check(amount)
        if self._balance - amount < -self.overdraft:
            raise InsufficientFunds(f"{self.owner}: balance {self._balance}, asked for {amount}")

    def withdraw(self, amount):
        self._can_take(amount)
        self._move("withdraw", amount)

    def transfer(self, to, amount):
        self._can_take(amount)
        self._move("transfer_out", amount)
        to._move("transfer_in", amount)
''')

# ---------------------------------------------------------------- fixing and testing code

task("fix_median_described", "fix", """
    This function should return the median of a list of numbers, but for lists of even length it
    returns the wrong value, and it also reorders the caller's list. Fix both bugs.

    ```python
    def median(xs):
        xs.sort()
        n = len(xs)
        if n == 0:
            raise ValueError("median of empty list")
        return xs[n // 2]
    ```
""", r'''
import pytest
from solution import median

def test_values():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5

def test_input_untouched():
    xs = [3, 1, 2]
    median(xs)
    assert xs == [3, 1, 2]

def test_empty():
    with pytest.raises(ValueError):
        median([])
''', r'''
def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        raise ValueError("median of empty list")
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2
''', difficulty=1)

task("fix_pagination_described", "fix", """
    Users say page 1 of our list starts at the wrong item, and the last page is sometimes missing
    from total_pages. Pages are numbered from 1; a page past the end has no items, and a page number
    below 1 should raise ValueError. Fix the function.

    ```python
    def paginate(items, page, per_page=10):
        total_pages = len(items) // per_page
        start = page * per_page
        return {"items": items[start:start + per_page], "page": page, "total_pages": total_pages}
    ```
""", r'''
import pytest
from solution import paginate

ITEMS = list(range(25))

def test_pages():
    assert paginate(ITEMS, 1) == {"items": list(range(10)), "page": 1, "total_pages": 3}
    assert paginate(ITEMS, 3)["items"] == [20, 21, 22, 23, 24]
    assert paginate(ITEMS, 4)["items"] == []
    assert paginate(list(range(20)), 1)["total_pages"] == 2

def test_bad_page():
    with pytest.raises(ValueError):
        paginate(ITEMS, 0)
''', r'''
def paginate(items, page, per_page=10):
    if page < 1:
        raise ValueError("pages start at 1")
    total_pages = -(-len(items) // per_page)
    start = (page - 1) * per_page
    return {"items": items[start:start + per_page], "page": page, "total_pages": total_pages}
''')

task("write_tests_slugify", "testing", """
    Write pytest tests (test functions only, no implementation) for this function, imported with
    `from slug import slugify`: slugify(text) lower-cases the text, replaces every run of characters
    that aren't ASCII letters or digits with a single "-", and strips "-" from both ends, so
    "  Hello, World!  " becomes "hello-world". The tests should pass for a correct implementation and
    catch the common bugs.
""", r'''
import os
import shutil
import subprocess
import sys
from pathlib import Path

GOOD = """
import re
def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
"""
BUGS = {
    "no_lower": 'import re\ndef slugify(text):\n    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-")\n',
    "no_collapse": 'import re\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]", "-", text.lower()).strip("-")\n',
    "no_strip": 'import re\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]+", "-", text.lower())\n',
}

def run_with(impl, tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "slug.py").write_text(impl)
    shutil.copy("solution.py", d / "test_model.py")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_model.py"],
                       cwd=d, capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return r.returncode

def test_has_tests():
    assert "def test" in Path("solution.py").read_text()

def test_passes_on_correct_code(tmp_path):
    assert run_with(GOOD, tmp_path, "good") == 0

def test_catches_bugs(tmp_path):
    caught = sum(run_with(code, tmp_path, name) != 0 for name, code in BUGS.items())
    assert caught >= 2
''', r'''
from slug import slugify


def test_basic():
    assert slugify("  Hello, World!  ") == "hello-world"


def test_lowercases():
    assert slugify("ABC def") == "abc-def"


def test_collapses_runs():
    assert slugify("a -- b__c") == "a-b-c"


def test_strips_ends():
    assert slugify("--x--") == "x"
    assert slugify("!!!") == ""


def test_digits_kept():
    assert slugify("Python 3.12") == "python-3-12"
''', difficulty=3)

# ---------------------------------------------------------------- not Python

SQL_FIXTURE = r'''
import sqlite3

def db():
    c = sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, created_at TEXT);
    INSERT INTO customers VALUES (1, 'Ann'), (2, 'Bo'), (3, 'Cy'), (4, 'Di'), (5, 'Ed');
    INSERT INTO orders(customer_id, amount, created_at) VALUES
      (1, 100, '2024-01-05'), (1, 50, '2024-03-10'), (2, 120, '2024-02-01'), (2, 30, '2023-12-31'),
      (3, 150, '2024-03-15'), (4, 500, '2023-06-01'), (5, 10, '2024-01-20'), (2, 30.556, '2024-01-07');
    """)
    return c

def run_sql(sql):
    return db().execute(sql.strip().rstrip(";")).fetchall()
'''

task("sql_top_customers", "sql", """
    Tables in SQLite:

    customers(id INTEGER PRIMARY KEY, name TEXT)
    orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, created_at TEXT)  -- 'YYYY-MM-DD'

    Write one SQLite SELECT query that returns the three customers with the highest total order
    amount in 2024, as rows of (name, total), highest first, ties broken by name. Leave out customers
    with no orders in 2024.
""", ANSWER + SQL_FIXTURE + r'''

def test_query():
    rows = run_sql(answer(("sql", "sqlite")))
    assert [(n, round(t, 3)) for n, t in rows] == [("Bo", 150.556), ("Ann", 150.0), ("Cy", 150.0)]
''', """
```sql
SELECT c.name, SUM(o.amount) AS total
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE o.created_at >= '2024-01-01' AND o.created_at < '2025-01-01'
GROUP BY c.id, c.name
ORDER BY total DESC, c.name
LIMIT 3;
```
""", raw=True)

task("sql_monthly_totals", "sql", """
    Tables in SQLite:

    customers(id INTEGER PRIMARY KEY, name TEXT)
    orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, created_at TEXT)  -- 'YYYY-MM-DD'

    Write one SQLite SELECT query that returns, for each month of 2024 that has orders, a row of
    (month as 'YYYY-MM', number of orders, total amount rounded to 2 decimals), in month order.
""", ANSWER + SQL_FIXTURE + r'''

def test_query():
    assert run_sql(answer(("sql", "sqlite"))) == [("2024-01", 3, 140.56), ("2024-02", 1, 120.0), ("2024-03", 2, 200.0)]
''', """
```sql
SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS orders, ROUND(SUM(amount), 2) AS total
FROM orders
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'
GROUP BY month
ORDER BY month;
```
""", raw=True)

task("bash_top_ips", "shell", """
    Write a bash command that reads access.log in the current directory (combined log format, client
    IP first on each line) and prints the 3 most frequent client IPs with their counts, one
    `count ip` pair per line, most frequent first. Use standard tools such as awk, sort, uniq and head.
""", ANSWER + r'''
import subprocess

LOG = "\n".join(
    [f'10.0.0.1 - - [10/Oct/2026:13:55:{i:02d} +0000] "GET / HTTP/1.1" 200 1 "-" "-"' for i in range(5)]
    + [f'10.0.0.2 - - [10/Oct/2026:13:56:{i:02d} +0000] "GET / HTTP/1.1" 200 1 "-" "-"' for i in range(3)]
    + [f'192.168.1.9 - - [10/Oct/2026:13:57:{i:02d} +0000] "GET / HTTP/1.1" 200 1 "-" "-"' for i in range(4)]
    + ['10.0.0.3 - - [10/Oct/2026:13:58:00 +0000] "GET / HTTP/1.1" 200 1 "-" "-"']) + "\n"

def test_top_ips(tmp_path):
    (tmp_path / "access.log").write_text(LOG)
    r = subprocess.run(["bash", "-c", answer(("bash", "sh", "shell", "console"))], cwd=tmp_path,
                       capture_output=True, text=True, timeout=20)
    lines = [l.split() for l in r.stdout.strip().splitlines() if l.strip()]
    lines = [l[1:] if l and l[0] == "$" else l for l in lines]
    assert lines == [["5", "10.0.0.1"], ["4", "192.168.1.9"], ["3", "10.0.0.2"]]
''', """
```bash
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -3
```
""", raw=True)

task("bash_rename_ext", "shell", """
    Write a bash script that renames every .txt file in the current directory to the same name with
    a .md extension. It must work for file names that contain spaces.
""", ANSWER + r'''
import os
import subprocess

def test_rename(tmp_path):
    for n in ["a.txt", "my notes.txt", "b.log"]:
        (tmp_path / n).write_text(n)
    r = subprocess.run(["bash", "-c", answer(("bash", "sh", "shell"))], cwd=tmp_path, capture_output=True,
                       text=True, timeout=20)
    assert sorted(os.listdir(tmp_path)) == ["a.md", "b.log", "my notes.md"], r.stderr
    assert (tmp_path / "my notes.md").read_text() == "my notes.txt"
''', """
```bash
#!/usr/bin/env bash
for f in *.txt; do
  [ -e "$f" ] || continue
  mv -- "$f" "${f%.txt}.md"
done
```
""", raw=True, difficulty=1)

NODE = r'''
import glob
import os
import shutil
import subprocess

def node():
    """The command that starts node, as a list."""
    found = shutil.which("node", path="/usr/bin:/bin:/usr/local/bin:/opt/conda/bin")
    found = found or next(iter(sorted(glob.glob("/home/*/.nvm/versions/node/*/bin/node"))), None)
    if found:
        return [found]
    # Kaggle has no node: the eval dataset carries the official linux-x64 binary. Datasets drop the
    # executable bit and the sandbox's file-size limit rules out a copy, so start it through the loader.
    shipped = sorted(glob.glob("/kaggle/input/**/node-v*-linux-x64", recursive=True))
    assert shipped, "node is not installed in this session"
    return [shipped[-1]] if os.access(shipped[-1], os.X_OK) else ["/lib64/ld-linux-x86-64.so.2", shipped[-1]]

def run_js(tmp_path, code, script):
    (tmp_path / "sol.js").write_text(code)
    r = subprocess.run([*node(), "-e", script], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip()
'''

task("js_group_by", "javascript", """
    Write a JavaScript function `groupBy(items, keyFn)` that returns an object mapping each key to the
    array of items with that key, keeping their order. Export it with `module.exports = { groupBy }`.
""", ANSWER + NODE + r'''
import json

def test_group_by(tmp_path):
    out = run_js(tmp_path, answer(("javascript", "js", "node"), must="module.exports"), """
const { groupBy } = require('./sol.js');
const r = groupBy([{t: 'a', v: 1}, {t: 'b', v: 2}, {t: 'a', v: 3}], x => x.t);
console.log(JSON.stringify(r));
console.log(JSON.stringify(groupBy([], x => x)));
""")
    first, second = out.splitlines()
    assert json.loads(first) == {"a": [{"t": "a", "v": 1}, {"t": "a", "v": 3}], "b": [{"t": "b", "v": 2}]}
    assert json.loads(second) == {}
''', """
```javascript
function groupBy(items, keyFn) {
  const out = {};
  for (const item of items) {
    const key = keyFn(item);
    (out[key] ||= []).push(item);
  }
  return out;
}

module.exports = { groupBy };
```
""", raw=True, difficulty=1)

task("js_parse_query", "javascript", """
    Write a JavaScript function `parseQuery(qs)` that turns a URL query string into an object. A
    leading "?" is optional. Decode %XX escapes and "+" as a space. A key without "=" gets the value
    "". A key that appears more than once maps to an array of its values in order. Export it with
    `module.exports = { parseQuery }`.
""", ANSWER + NODE + r'''
import json

def test_parse(tmp_path):
    out = run_js(tmp_path, answer(("javascript", "js", "node"), must="module.exports"), """
const { parseQuery } = require('./sol.js');
console.log(JSON.stringify(parseQuery('?q=hello+world&tag=a&tag=b&flag&name=J%C3%BCrgen')));
console.log(JSON.stringify(parseQuery('')));
""")
    first, second = out.splitlines()
    assert json.loads(first) == {"q": "hello world", "tag": ["a", "b"], "flag": "", "name": "Jürgen"}
    assert json.loads(second) == {}
''', """
```javascript
function parseQuery(qs) {
  const out = {};
  const s = qs.startsWith('?') ? qs.slice(1) : qs;
  for (const part of s.split('&')) {
    if (!part) continue;
    const i = part.indexOf('=');
    const decode = x => decodeURIComponent(x.replace(/\\+/g, ' '));
    const key = decode(i < 0 ? part : part.slice(0, i));
    const value = i < 0 ? '' : decode(part.slice(i + 1));
    if (!(key in out)) out[key] = value;
    else if (Array.isArray(out[key])) out[key].push(value);
    else out[key] = [out[key], value];
  }
  return out;
}

module.exports = { parseQuery };
```
""", raw=True)

task("explain_none_error", "explain", """
    I get this error. What does it mean, and how do I fix it?

    ```
    Traceback (most recent call last):
      File "app.py", line 8, in <module>
        print(user["name"])
    TypeError: 'NoneType' object is not subscriptable
    ```

    `user` comes from `user = find_user(email)`.
""", ANSWER + r'''
from pathlib import Path

def test_explains_none():
    text = Path("solution.py").read_text()
    low = text.lower()
    assert len(text.strip()) > 80
    assert "none" in low
    assert "find_user" in low or "not found" in low or "no user" in low
''', """
`find_user(email)` returned `None`, which usually means no user matched that email. Indexing
`None` with `["name"]` raises this TypeError. Check the result before using it:

```python
user = find_user(email)
if user is None:
    print(f"no user with email {email}")
else:
    print(user["name"])
```
""", raw=True, difficulty=1)

task("list_vs_tuple", "explain", """
    What's the difference between a list and a tuple in Python, and when should I use each?
""", ANSWER + r'''
import re
from pathlib import Path

def test_mentions_mutability():
    text = Path("solution.py").read_text()
    assert len(text.strip()) > 100
    assert re.search(r"immutable|can(no|')t be (changed|modified)|not be (changed|modified)", text, re.I)
''', """
A list is mutable: you can add, remove and replace items after creating it. A tuple is immutable:
once created, it can't be changed. Use a list for a collection that grows or changes, such as
results you append to. Use a tuple for a fixed group of values, such as a point (x, y) or a record
returned from a function. Because tuples are immutable and hashable (when their items are), they can
be dict keys or set members, and lists can't.
""", raw=True, difficulty=1)


if __name__ == "__main__":
    out = os.path.join(HERE, "realworld.jsonl")
    with open(out, "w") as f:
        for t in T:
            f.write(json.dumps(t) + "\n")
    print(f"{len(T)} tasks -> {out}")

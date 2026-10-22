"""Fill frontier_task.py's constants and write a script ready for `kaggle b t push`.

    python eval/frontier_fill.py out.py --models gpt-5.5-2026-04-23 ... --benches manual humaneval --budget 9.5 \
        [--only only.json] [--extra 8192]
"""

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--benches", nargs="+", required=True)
ap.add_argument("--budget", type=float, required=True)
ap.add_argument("--only", help="json {slug: [\"bench|task_id\", ...]}: ask only these")
ap.add_argument("--extra", type=int, default=0, help="output tokens added to every request")
a = ap.parse_args()
s = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontier_task.py")).read()
only = json.load(open(a.only)) if a.only else None
for k, v in {"__MODELS__": json.dumps(a.models), "__BENCHES__": json.dumps(a.benches), "__BUDGET__": repr(a.budget),
             "__ONLY__": json.dumps(only) if only else "None", "__EXTRA__": str(a.extra)}.items():
    assert s.count(k) == 1, k
    s = s.replace(k, v)
open(a.out, "w").write(s)
print("wrote", a.out)

"""Upload files from attached kernel outputs to the HF dataset. Kaggle CPU, keys attached; runs no data.

CONFIG: repo (default abhash-chakraborty/pycoder-data), items [{"glob": "/kaggle/input/**/x-*.jsonl",
"dest": "v1/problems/verified"}], message, repo_type ("dataset", or "model" for a checkpoint; a missing
model repo is created private). Each matched file goes to <dest>/<basename>; a basename
that matches twice (two kernels wrote the same name) is an error. Files go up in commits of 40.
"""

import glob
import json
import os
import time

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("PYCODER_CONFIG", "{}"))
REPO = CONFIG.get("repo", "abhash-chakraborty/pycoder-data")
REPO_TYPE = CONFIG.get("repo_type", "dataset")
KEYS = json.load(open(glob.glob("/kaggle/input/**/keys.json", recursive=True)[0]))


def main():
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=KEYS["HF_TOKEN"])
    if REPO_TYPE == "model":
        api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    ops, seen = [], {}
    for item in CONFIG["items"]:
        for f in sorted(glob.glob(item["glob"], recursive=True)):
            dest = f"{item['dest'].rstrip('/')}/{os.path.basename(f)}"
            if dest in seen:
                raise SystemExit(f"two files for {dest}: {seen[dest]} and {f}")
            seen[dest] = f
            ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=f))
    print(f"{len(ops)} files, {sum(os.path.getsize(f) for f in seen.values()) / 1e6:.1f} MB", flush=True)
    for i in range(0, len(ops), 40):
        for attempt in range(6):
            try:
                api.create_commit(REPO, repo_type=REPO_TYPE, operations=ops[i:i + 40],
                                  commit_message=CONFIG.get("message", "upload") + f" ({i // 40 + 1})")
                break
            except Exception as e:
                print(f"commit failed: {type(e).__name__}: {str(e)[:200]}; retrying", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            raise SystemExit("HF commit kept failing")
    json.dump(sorted(seen), open("/kaggle/working/uploaded.json", "w"), indent=1)
    print("DONE", len(ops), flush=True)


main()

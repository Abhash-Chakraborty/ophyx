"""Several training runs back to back in one TPU session (appended after train/pretrain.py).

TPU sessions wait hours in the queue, so the 125M rehearsal and the first stretch of the 350M run
share one session instead of queueing twice. CONFIG["runs"] is a list of full run configs. All
runs share the session's wall-clock budget (budget_s counts from session start), so a long run
listed last simply stops at the deadline, checkpoints, and is continued by its chain (which
restores from the HF backup).
"""

def finished_on_hf(run):
    """A run whose HF status says finished (e.g. an SFT run done in an earlier chain session) is skipped
    before anything is downloaded."""
    if not run.get("hf_ckpt_repo") or not KEYS.get("HF_TOKEN"):
        return False
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(run["hf_ckpt_repo"], f"{run.get('run_name', 'run')}/status.json", token=KEYS["HF_TOKEN"],
                            force_download=True)
        return bool(json.load(open(p)).get("finished"))
    except Exception:
        return False


start_heartbeat()
set_phase("sequence start")
RUNS = CONFIG["runs"]
SESSION = {k: v for k, v in CONFIG.items() if k != "runs"}
summary = []
for i, run in enumerate(RUNS):
    CONFIG.clear()
    name = {**SESSION, **run}.get("run_name", f"run{i}")  # an empty run entry is the session's own run
    out = os.path.join("/kaggle/working" if KAGGLE else os.path.abspath("run"), name)
    CONFIG.update({**SESSION, **run, "run_name": name, "library": True, "out": out})  # own dir per run
    log(f"=== run {i + 1}/{len(RUNS)}: {name} ===")
    if finished_on_hf(CONFIG):
        log(f"{name} already finished (HF status); skipping")
        summary.append({"run": name, "skipped": "finished"})
        continue
    try:
        _, status = train()
        summary.append({"run": name, **status})
    except Exception as e:
        import traceback
        traceback.print_exc()
        summary.append({"run": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        notify(f"sequence run {name} FAILED", str(e)[:500], "x")
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    jax.clear_caches()
    if time.time() > T0 + SESSION.get("budget_s", 11 * 3600) - SESSION.get("margin_s", 1500):
        break
json.dump(summary, open(os.path.join("/kaggle/working" if KAGGLE else ".", "sequence.json"), "w"), indent=1)
log("SEQUENCE", json.dumps(summary))

"""Several training runs back to back in one TPU session (appended after train/pretrain.py).

TPU sessions wait hours in the queue, so the 125M rehearsal and the first stretch of the 350M run
share one session instead of queueing twice. CONFIG["runs"] is a list of full run configs. All
runs share the session's wall-clock budget (budget_s counts from session start), so a long run
listed last simply stops at the deadline, checkpoints, and is continued by its chain (which
restores from the HF backup).
"""

start_heartbeat()
set_phase("sequence start")
RUNS = CONFIG["runs"]
SESSION = {k: v for k, v in CONFIG.items() if k != "runs"}
summary = []
for i, run in enumerate(RUNS):
    CONFIG.clear()
    out = os.path.join("/kaggle/working" if KAGGLE else os.path.abspath("run"), run.get("run_name", f"run{i}"))
    CONFIG.update({**SESSION, **run, "library": True, "out": out})  # own dir: never resume another run's checkpoint
    log(f"=== run {i + 1}/{len(RUNS)}: {run.get('run_name')} ===")
    try:
        _, status = train()
        summary.append({"run": run.get("run_name"), **status})
    except Exception as e:
        import traceback
        traceback.print_exc()
        summary.append({"run": run.get("run_name"), "error": f"{type(e).__name__}: {str(e)[:300]}"})
        notify(f"sequence run {run.get('run_name')} FAILED", str(e)[:500], "x")
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

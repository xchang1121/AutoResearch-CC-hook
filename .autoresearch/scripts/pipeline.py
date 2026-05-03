#!/usr/bin/env python3
"""
Post-edit pipeline — runs ALL mechanical steps after Claude Code edits code.

Claude Code does the LLM work (plan, edit, diagnose). Then calls this:
    python .autoresearch/scripts/pipeline.py <task_dir>

This script does:
    1. quick_check → fail? rollback, report
    2. eval → get metrics
    3. keep_or_discard → KEEP/DISCARD/FAIL
    4. settle → update plan.md, advance (ACTIVE)
    5. compute next phase → write .phase
    6. print status + next guidance

Output: human-readable status to stdout. Claude Code sees it and acts accordingly.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config
from failure_extractor import format_for_stdout
from phase_machine import (
    write_phase, compute_next_phase, get_active_item,
    get_guidance, auto_rollback, load_progress, edit_marker_path,
    parse_last_json_line,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <task_dir>")
        sys.exit(1)

    task_dir = os.path.abspath(sys.argv[1])
    config = load_task_config(task_dir)
    if config is None:
        print("[PIPELINE] ERROR: task.yaml not found")
        sys.exit(1)

    progress = load_progress(task_dir) or {}
    active = get_active_item(task_dir)
    # Persist the full description — dashboards/logs do their own display-time
    # truncation based on terminal width.
    desc = active["description"] if active else "optimization round"
    plan_item = active["id"] if active else None

    # Worker flag
    worker_flag = []
    if config.worker_urls:
        worker_flag = ["--worker-url", config.worker_urls[0]]

    # === Step 1: Quick check ===
    print("[PIPELINE] Running quick_check...", flush=True)
    qc = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "quick_check.py"), task_dir],
        capture_output=True, text=True, timeout=60,
    )
    if qc.returncode != 0 or "OK" not in qc.stdout:
        auto_rollback(task_dir)
        # Clear edit marker — rollback means we're back to clean state
        marker = edit_marker_path(task_dir)
        if os.path.exists(marker):
            os.remove(marker)
        print(f"[PIPELINE] QUICK CHECK FAIL: {qc.stdout[:200]}")
        print(f"[PIPELINE] Auto-rolled back. Fix and re-edit.")
        print(get_guidance(task_dir))
        sys.exit(0)

    print("[PIPELINE] Quick check PASS", flush=True)

    # === Step 2: Eval ===
    print("[PIPELINE] Running eval...", flush=True)
    eval_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "eval_wrapper.py"), task_dir] + worker_flag
    try:
        ev = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        auto_rollback(task_dir)
        print("[PIPELINE] EVAL TIMEOUT. Rolled back.")
        sys.exit(0)

    eval_json = parse_last_json_line(ev.stdout)
    if eval_json is None:
        auto_rollback(task_dir)
        print(f"[PIPELINE] EVAL ERROR: no JSON output. stderr: {ev.stderr[:200]}")
        sys.exit(1)

    correctness = eval_json.get("correctness", False)
    metrics = eval_json.get("metrics", {})
    print(f"[PIPELINE] Eval: correctness={correctness}, metrics={metrics}", flush=True)

    # Surface structured failure signals (UB overflow, aivec trap, OOM, ...)
    # extracted from the worker's raw log. Without this, Claude sees only a
    # generic "verify failed" string and has nothing to act on. Fall back
    # through increasingly coarse sources so *something* always reaches the
    # user on failure.
    if not correctness or eval_json.get("error"):
        if eval_json.get("error"):
            print(f"[PIPELINE] Error: {eval_json['error']}", flush=True)
        pretty = format_for_stdout(eval_json.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_json.get("raw_output_tail"):
            # No known pattern matched — dump the tail raw so Claude still
            # has something concrete to work with.
            print("[PIPELINE] Worker log tail (no structured signals matched):",
                  flush=True)
            print(eval_json["raw_output_tail"], flush=True)

    # === Step 3: Keep or discard ===
    kd_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "keep_or_discard.py"),
              task_dir, json.dumps(eval_json), "--description", desc]
    if plan_item:
        kd_cmd += ["--plan-item", plan_item]
    kd = subprocess.run(kd_cmd, capture_output=True, text=True, timeout=30)
    kd_json = parse_last_json_line(kd.stdout)
    if kd_json is None:
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {kd.stdout[:200]}")
        sys.exit(1)

    decision = kd_json.get("decision", "FAIL")

    # === Step 4: Settle (update plan.md) ===
    # progress.json + history.jsonl were already mutated by keep_or_discard;
    # plan.md is the only state piece settle.py owns. If settle fails (no
    # ACTIVE item, malformed plan.md, etc.) and we still advance phase, the
    # next round sees inconsistent state — same ACTIVE item, advanced
    # eval_rounds, and pipeline will keep dispatching the same plan_item
    # forever. Surface the failure loud and abort phase advance instead.
    settle = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "settle.py"),
         task_dir, json.dumps(kd_json)],
        capture_output=True, text=True, timeout=10,
    )
    if settle.returncode != 0:
        tail_out = (settle.stdout or "").strip()[-400:]
        tail_err = (settle.stderr or "").strip()[-400:]
        # plan.md is machine-maintained (CLAUDE.md invariant #1); never
        # hand-edit it. Read the settle stderr below for the real cause
        # (typically: no (ACTIVE) item, malformed plan.md, or a script
        # error). Once that's resolved, re-run pipeline.py — the next
        # invocation re-runs settle on the same kd_json. If settle keeps
        # failing on the same plan.md, REPLAN via create_plan.py
        # (preferred) or DIAGNOSE if consecutive_failures triggered it.
        print(f"[PIPELINE] SETTLE FAILED (rc={settle.returncode}). "
              f"plan.md was NOT updated even though progress.json + "
              f"history.jsonl already moved. Phase NOT advanced. "
              f"DO NOT hand-edit plan.md — read the settle stderr below "
              f"for the underlying cause, then re-run pipeline.py. If "
              f"settle keeps failing, REPLAN via create_plan.py.\n"
              f"stdout tail: {tail_out}\n"
              f"stderr tail: {tail_err}", file=sys.stderr)
        sys.exit(1)

    # === Step 5: Compute next phase + clear edit marker ===
    next_phase = compute_next_phase(task_dir)
    write_phase(task_dir, next_phase)

    # Clear edit-started marker (round is complete)
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        os.remove(marker)

    # === Step 6: Status report ===
    progress = load_progress(task_dir) or {}
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)

    improv = ""
    if (
        best is not None
        and baseline is not None
        and isinstance(best, (int, float))
        and isinstance(baseline, (int, float))
        and baseline != 0
        and best != 0
    ):
        pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    settled_id = active["id"] if active else "?"

    print(f"\n{'=' * 50}")
    print(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | Best: {best}{improv} | Failures: {failures}")
    print(f"Phase -> {next_phase}")
    print(f"{'=' * 50}")
    print(get_guidance(task_dir))


if __name__ == "__main__":
    main()

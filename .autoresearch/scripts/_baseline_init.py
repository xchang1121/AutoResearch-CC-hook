#!/usr/bin/env python3
"""Initialize .ar_state/progress.json from baseline eval output.

Called by baseline.py:
    python _baseline_init.py <task_dir> <eval_json>
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config
from phase_machine import (
    write_phase, append_history, save_progress, PLAN,
)


def _valid(v):
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def main():
    task_dir = sys.argv[1]
    eval_json = sys.argv[2]

    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        sys.exit(1)

    eval_data = json.loads(eval_json)
    correctness = eval_data.get("correctness", False)
    metrics = eval_data.get("metrics", {})
    seed_val = metrics.get(config.primary_metric) if _valid(metrics.get(config.primary_metric)) else None
    ref_val = metrics.get("ref_latency_us") if _valid(metrics.get("ref_latency_us")) else None

    # baseline_metric anchors speedup display. Prefer PyTorch reference latency;
    # fall back to seed only when the worker couldn't measure ref (e.g. local
    # eval with no reference script).
    if ref_val is not None:
        baseline_val = ref_val
        baseline_source = "ref"
        print(f"[baseline] baseline = ref_latency_us = {ref_val} (PyTorch reference)", file=sys.stderr)
    else:
        baseline_val = seed_val
        baseline_source = "seed_fallback"
        print(f"[baseline] WARNING: ref_latency_us missing — baseline falls back to seed metric",
              file=sys.stderr)

    # best_metric needs a real number for is_improvement() comparisons. Prefer
    # seed; if seed profile failed, use ref so any correct kernel counts as
    # progress. Never substitute ref into seed_metric itself — that was the old
    # bug (seed == baseline == best, all ref, fake "1.00x").
    initial_best = seed_val if seed_val is not None else ref_val

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=task_dir, capture_output=True, text=True,
        )
        baseline_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        baseline_commit = "unknown"

    save_progress(task_dir, {
        "task": config.name,
        "eval_rounds": 0,
        "max_rounds": config.max_rounds,
        "best_metric": initial_best,
        "best_commit": baseline_commit if seed_val is not None else "seed_profile_failed",
        "baseline_commit": baseline_commit,
        "baseline_metric": baseline_val,
        "baseline_source": baseline_source,
        "baseline_correctness": correctness,
        "seed_metric": seed_val,
        "consecutive_failures": 0,
        "plan_version": 0,
        "status": "no_plan",
    }, stamp=True)

    # Round 0 logs the SEED kernel's initial eval. `metrics.latency_us` is the
    # seed's timing; `metrics.ref_latency_us` (if present) is the PyTorch
    # baseline used as the speedup anchor. Don't call this "BASELINE" in the
    # history table — "Baseline" in the summary means PyTorch ref.
    append_history(task_dir, {
        "round": 0,
        "description": "seed kernel initial eval",
        "decision": "SEED",
        "metrics": metrics,
        "correctness": correctness,
        "commit": baseline_commit,
    })

    # Hard-fail when baseline correctness is False. Downstream (PLAN/EDIT)
    # assumes the seed is a correct kernel; if it isn't, every improvement
    # comparison is meaningless and the loop will spin generating "optimizations"
    # against a broken reference point. Leave phase pinned at BASELINE so
    # hook_post_bash's retry/demote logic fires and routes back to
    # GENERATE_KERNEL for a manual fix. eval_wrapper.py intentionally exits 0
    # even on correctness failure (metrics may still be informative), so this
    # is the single correct place to gate phase advancement.
    if not correctness:
        print(
            f"[baseline] ERROR: baseline eval failed correctness check.\n"
            f"[baseline] seed {config.primary_metric}={seed_val} but output "
            f"did not match reference. Phase stays at BASELINE; "
            f"hook_post_bash will demote to GENERATE_KERNEL for a fix.",
            file=sys.stderr,
        )
        sys.exit(3)

    # Hard-fail when seed didn't profile — proceeding would compare future
    # rounds against ref instead of seed, producing misleading "speedup" numbers.
    if seed_val is None:
        print(
            f"[baseline] ERROR: seed kernel produced no valid {config.primary_metric}.\n"
            f"[baseline] Worker ran profile_{config.name}_generation.py but no timing came back\n"
            f"[baseline] (result JSON missing, inf, or 0). Likely causes:\n"
            f"[baseline]   - Triton compile error surfaced only during profiling\n"
            f"[baseline]   - kernel runs once under verify but OOMs/hangs under repeated invocation\n"
            f"[baseline]   - generation_profile_result.json not written by profile script\n"
            f"[baseline] Check worker logs, fix kernel.py, rerun baseline. "
            f"progress.json kept for diagnostics.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Baseline succeeded — advance phase to PLAN atomically so activation
    # goes straight to planning (no implicit phase inference needed).
    write_phase(task_dir, PLAN)

    print(f"[baseline] Initialized: task={config.name}, "
          f"seed_{config.primary_metric}={seed_val}, "
          f"baseline({baseline_source})={baseline_val}, commit={baseline_commit}",
          file=sys.stderr)
    print(f"[baseline] Phase -> PLAN", file=sys.stderr)


if __name__ == "__main__":
    main()

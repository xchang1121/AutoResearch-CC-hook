#!/usr/bin/env python3
"""
Keep-or-discard decision engine for Claude Code autoresearch.

Zero external dependency — uses local task_config.py.

Usage:
    python .autoresearch/scripts/keep_or_discard.py <task_dir> <eval_json>
    python .autoresearch/scripts/keep_or_discard.py <task_dir> --eval-file <path>

Output (stdout, last line):
    {"decision": "KEEP", "best_metric": 145.3, "eval_rounds": 6, "consecutive_failures": 0}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config, EvalResult, is_improvement, check_constraints
from phase_machine import load_progress, save_progress, append_history, auto_rollback
from git_utils import commit_in_task


def main():
    parser = argparse.ArgumentParser(description="Keep or discard eval result")
    parser.add_argument("task_dir", help="Path to task directory")
    parser.add_argument("eval_json", nargs="?", help="Eval result as JSON string")
    parser.add_argument("--eval-file", help="Path to file containing eval JSON")
    parser.add_argument("--description", default="optimization round", help="Round description")
    parser.add_argument("--plan-item", default=None,
                        help="Plan item id (pN) this round settles — recorded in history")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)

    if args.eval_file:
        with open(args.eval_file, "r") as f:
            eval_data = json.load(f)
    elif args.eval_json:
        eval_data = json.loads(args.eval_json)
    else:
        print(json.dumps({"decision": "ERROR", "error": "No eval result provided"}))
        sys.exit(1)

    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({"decision": "ERROR", "error": "task.yaml not found"}))
        sys.exit(1)

    progress = load_progress(task_dir) or {}

    eval_result = EvalResult(
        correctness=eval_data.get("correctness", False),
        metrics=eval_data.get("metrics", {}),
        error=eval_data.get("error"),
    )

    round_num = progress.get("eval_rounds", 0) + 1
    decision = "DISCARD"
    commit_hash = None

    # `consecutive_failures` counts ONLY real failures (FAIL = kernel broken).
    # DISCARD (correct but not faster) is not a failure — it's a signal to
    # REPLAN with different ideas, which happens naturally when all plan
    # items settle. DIAGNOSE is reserved for broken kernels.
    if not eval_result.correctness:
        decision = "FAIL"
        progress["consecutive_failures"] = progress.get("consecutive_failures", 0) + 1
        print(f"[keep_or_discard] FAIL: correctness check failed", file=sys.stderr)
    elif config.constraints:
        violations = check_constraints(eval_result, config.constraints)
        if violations:
            decision = "FAIL"
            progress["consecutive_failures"] = progress.get("consecutive_failures", 0) + 1
            print(f"[keep_or_discard] FAIL: constraint violations: {violations}", file=sys.stderr)
    else:
        best_metric_val = progress.get("best_metric")
        if best_metric_val is None:
            decision = "KEEP"
        else:
            best_result = EvalResult(
                correctness=True,
                metrics={config.primary_metric: best_metric_val},
            )
            if is_improvement(
                eval_result, best_result,
                metric=config.primary_metric,
                lower_is_better=config.lower_is_better,
                threshold=config.improvement_threshold,
            ):
                decision = "KEEP"
            else:
                decision = "DISCARD"

    if decision == "KEEP":
        metric_val = eval_result.metrics.get(config.primary_metric)
        metric_str = f"{config.primary_metric}={metric_val}"
        ok, info = commit_in_task(
            task_dir,
            config.editable_files,
            f"autoresearch: {args.description} | {metric_str}",
        )
        commit_hash = info if ok and info != "noop" else None
        if not ok:
            print(f"[keep_or_discard] git commit failed: {info}", file=sys.stderr)
        progress["best_metric"] = metric_val
        progress["best_commit"] = commit_hash
        progress["consecutive_failures"] = 0
        print(f"[keep_or_discard] KEEP: {metric_str} (commit: {commit_hash})", file=sys.stderr)
    else:
        auto_rollback(task_dir)
        print(f"[keep_or_discard] {decision}: rolled back editable files", file=sys.stderr)

    progress["eval_rounds"] = round_num
    save_progress(task_dir, progress)

    append_history(task_dir, {
        "round": round_num,
        "plan_item": args.plan_item,
        "description": args.description,
        "decision": decision,
        "metrics": eval_result.metrics,
        "correctness": eval_result.correctness,
        "error": eval_result.error,
        "commit": commit_hash,
    })

    output = {
        "decision": decision,
        "best_metric": progress.get("best_metric"),
        "eval_rounds": round_num,
        "max_rounds": progress.get("max_rounds", config.max_rounds),
        "consecutive_failures": progress.get("consecutive_failures", 0),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()

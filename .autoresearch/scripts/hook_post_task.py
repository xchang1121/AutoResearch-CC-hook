#!/usr/bin/env python3
"""PostToolUse hook for Task — DIAGNOSE artifact validator.

Runs after every Task call. Only acts when phase==DIAGNOSE; otherwise no-op.

Behavior:
  - Validate `<task_dir>/.ar_state/diagnose_v<plan_version>.md` against the
    contract (see validators.validate_diagnose).
  - Pass → emit_status("[AR] DIAGNOSE artifact validated. Proceed to
    create_plan.py."). Reset diagnose_attempts counter.
  - Fail → increment diagnose_attempts (per plan_version). Inject
    additionalContext telling the main agent EXACTLY what's wrong and to
    re-issue Task.
  - On reaching DIAGNOSE_ATTEMPTS_CAP attempts: switch to the
    manual-planning fallback. The artifact gate on create_plan.py is
    relaxed downstream (hook_guard_bash + hook_post_bash check
    diagnose_attempts vs cap); the main agent is told to write
    plan_items.xml directly using history.jsonl + plan.md and run
    create_plan.py. The DIAGNOSE phase still must end via create_plan.py
    advancing to EDIT — the task does NOT terminate here. The subagent
    failed; the goal (a new plan) hasn't.

This hook is independent of hook_post_bash, which handles create_plan.py /
pipeline.py / baseline.py post-actions in DIAGNOSE and other phases.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status
from phase_machine import (
    DIAGNOSE, DIAGNOSE_ATTEMPTS_CAP, diagnose_artifact_path,
    diagnose_state, get_task_dir, read_phase, update_progress,
)


def _emit_retry_context(task_dir: str, plan_version: int, reason: str,
                        attempts: int) -> None:
    """Tell the main agent what went wrong and to retry Task — never to
    backstop the diagnose work itself. We surface attempts/cap so the model
    knows it has finite tries."""
    artifact = diagnose_artifact_path(task_dir, plan_version)
    msg = (
        f"[AR Phase: DIAGNOSE retry {attempts}/{DIAGNOSE_ATTEMPTS_CAP}] "
        f"Subagent did not produce a valid artifact: {reason}\n"
        f"\n"
        f"Required action: re-issue Task with subagent_type='ar-diagnosis'. "
        f"In your prompt, restate that the subagent's FINAL action must be "
        f"a Write call to:\n"
        f"  {artifact}\n"
        f"and that the file body must contain headings 'Root cause', "
        f"'Fix directions', 'What to avoid', and end with the marker line "
        f"the host gave in the previous DIAGNOSE guidance.\n"
        f"\n"
        f"Do NOT call create_plan.py, do NOT Edit kernel.py, do NOT Stop. "
        f"Only Task is legal in DIAGNOSE until the artifact validates."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }))


def _emit_manual_planning_context(task_dir: str, plan_version: int,
                                  reason: str) -> None:
    """At cap, tell the agent to write the plan itself and run create_plan.

    This is the fallback path: the subagent route is exhausted but the
    DIAGNOSE phase's end goal — a new plan — is unchanged. The artifact
    gate on create_plan.py is relaxed in hook_guard_bash / hook_post_bash
    once attempts reach the cap, so the agent is free to skip Task and
    proceed via the normal plan_items.xml + create_plan.py path.
    """
    msg = (
        f"[AR Phase: DIAGNOSE — manual planning fallback] "
        f"Subagent invocation failed {DIAGNOSE_ATTEMPTS_CAP} times for "
        f"plan_version={plan_version}; last reason: {reason}\n"
        f"\n"
        f"Stop re-issuing Task — further Task calls are blocked here. "
        f"The DIAGNOSE phase still requires a NEW plan; produce one "
        f"yourself:\n"
        f"  1. Read .ar_state/history.jsonl (focus on the last ~10 rounds, "
        f"especially FAIL rows — their description fields contain the "
        f"raw failure signal the subagent would have analyzed).\n"
        f"  2. Read .ar_state/plan.md to see what's already been tried.\n"
        f"  3. Write your <items>...</items> XML to "
        f"$AR_TASK_DIR/.ar_state/plan_items.xml. The plan must contain "
        f"≥3 items, ≥2 STRUCTURALLY different from the last plan "
        f"(algorithmic / fusion / memory layout / data movement, NOT "
        f"parameter tuning).\n"
        f"  4. Run python .autoresearch/scripts/create_plan.py "
        f"\"$AR_TASK_DIR\". The artifact gate is relaxed in this state; "
        f"only create_plan.py's own structural validation applies.\n"
        f"\n"
        f"Do NOT Edit kernel.py, do NOT Stop — phase advances to EDIT "
        f"once create_plan.py validates the new plan."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }))


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Task":
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    if read_phase(task_dir) != DIAGNOSE:
        sys.exit(0)

    state = diagnose_state(task_dir)
    pv = state.plan_version

    if state.artifact_ok:
        # Reset attempts so a future DIAGNOSE round (different
        # plan_version) starts fresh.
        update_progress(task_dir, diagnose_attempts=0,
                        diagnose_attempts_for_version=pv)
        emit_status(
            f"[AR] DIAGNOSE artifact validated for plan_version={pv}. "
            f"Proceed to create_plan.py with diagnose_v{pv}.md as input."
        )
        sys.exit(0)

    # Failure path: increment per-plan-version counter and route to retry
    # or fallback based on cap.
    new_attempts = state.attempts + 1
    update_progress(task_dir, diagnose_attempts=new_attempts,
                    diagnose_attempts_for_version=pv,
                    last_diagnose_failure_reason=state.artifact_reason)

    if new_attempts >= DIAGNOSE_ATTEMPTS_CAP:
        emit_status(
            f"[AR] DIAGNOSE subagent exhausted {DIAGNOSE_ATTEMPTS_CAP} "
            f"attempts for plan_version={pv}; switching to manual "
            f"planning fallback. Last reason: {state.artifact_reason}"
        )
        _emit_manual_planning_context(task_dir, pv, state.artifact_reason)
        sys.exit(0)

    _emit_retry_context(task_dir, pv, state.artifact_reason, new_attempts)
    sys.exit(0)


if __name__ == "__main__":
    main()

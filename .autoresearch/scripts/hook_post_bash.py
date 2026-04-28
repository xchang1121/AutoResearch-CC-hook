#!/usr/bin/env python3
"""
PostToolUse hook for Bash — phase auto-advancement after user-issued commands.

The only commands that advance phase from this hook are those Claude runs
directly via the Bash tool:
  - `export AR_TASK_DIR=...`  → activate task, compute starting phase
                                (fresh task: validate ref/kernel and pin the
                                appropriate GENERATE_* / BASELINE phase)
  - `baseline.py`             → PLAN on success;
                                GENERATE_KERNEL on seed-metric failure
                                (so kernel.py becomes editable again)
  - `pipeline.py`             → whatever phase pipeline.py itself wrote
  - `create_plan.py`          → EDIT on plan validation pass
                                (called from PLAN / DIAGNOSE / REPLAN)

The inner pipeline steps (quick_check / eval_wrapper / keep_or_discard /
settle) are subprocess children of pipeline.py and never re-enter this hook,
so they don't need their own phase constants or branches here.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, emit_todowrite_context
from phase_machine import (
    read_phase, write_phase, get_guidance, compute_resume_phase,
    get_task_dir, set_task_dir, get_active_item, touch_heartbeat,
    load_progress, update_progress,
    validate_reference, validate_kernel, is_placeholder_file,
    parse_invoked_ar_script,
    progress_path, history_path, plan_path, edit_marker_path, state_path,
    PHASE_FILE,
    BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN, GENERATE_REF, GENERATE_KERNEL,
)


def _activation_target(command: str) -> str | None:
    if "AR_TASK_DIR=" not in command:
        return None
    m = re.search(r'AR_TASK_DIR=["\']?([^"\';\s&]+)', command)
    return m.group(1) if m else None


# Script-invocation parsing lives in phase_machine.parse_invoked_ar_script —
# single regex shared with hook_guard_bash to avoid silent disagreement on
# edge cases like `python3 baseline.py` or `bash some.py`.


def _clean_stale_edit_marker(task_dir: str):
    """Remove .edit_started if git is clean (nothing to resume)."""
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    try:
        import subprocess as _sp
        diff = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=task_dir, capture_output=True, text=True, timeout=5,
        )
        if not diff.stdout.strip():
            os.remove(marker)
            emit_status("[AR] Cleaned stale edit marker (git is clean).")
    except Exception:
        pass


def _handle_activation(new_task_dir: str):
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        emit_status(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    set_task_dir(new_task_dir)
    _clean_stale_edit_marker(new_task_dir)

    has_phase = os.path.exists(state_path(new_task_dir, PHASE_FILE))
    has_progress = os.path.exists(progress_path(new_task_dir))

    if has_phase:
        phase = read_phase(new_task_dir)
        emit_status(f"[AR] Resuming. Phase: {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    elif has_progress:
        phase = compute_resume_phase(new_task_dir)
        write_phase(new_task_dir, phase)
        emit_status(f"[AR] Resuming from progress. Phase -> {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    else:
        _fresh_start(new_task_dir)


def _fresh_start(task_dir: str):
    """Pick initial phase for a fresh task based on which files are present
    AND validate them. `is_placeholder_file` (canonical) lets us short-
    circuit the subprocess-import step on a known stub; otherwise the same
    validate_reference / validate_kernel that gates phase advances also
    pins the right phase from the moment of activation."""
    ref_path = os.path.join(task_dir, "reference.py")
    kernel_path = os.path.join(task_dir, "kernel.py")

    if is_placeholder_file(ref_path):
        write_phase(task_dir, GENERATE_REF)
        emit_status(f"[AR] Fresh start (no reference). Phase -> GENERATE_REF. {get_guidance(task_dir)}")
        return

    ok, err = validate_reference(task_dir)
    if not ok:
        write_phase(task_dir, GENERATE_REF)
        emit_status(
            f"[AR] reference.py present but invalid — Phase -> GENERATE_REF.\n"
            f"     {err}"
        )
        return

    if is_placeholder_file(kernel_path):
        write_phase(task_dir, GENERATE_KERNEL)
        emit_status(f"[AR] Fresh start (no kernel). Phase -> GENERATE_KERNEL. {get_guidance(task_dir)}")
        return

    ok, err = validate_kernel(task_dir)
    if not ok:
        write_phase(task_dir, GENERATE_KERNEL)
        emit_status(
            f"[AR] kernel.py present but invalid — Phase -> GENERATE_KERNEL.\n"
            f"     {err}"
        )
        return

    write_phase(task_dir, BASELINE)
    emit_status(f"[AR] Fresh start. Phase -> BASELINE. {get_guidance(task_dir)}")


def _progress_update_for_plan(task_dir: str, phase: str):
    """Set status=active after a valid new plan. `plan_version` is owned and
    bumped by create_plan.py — this hook must NOT re-bump it (caused double
    increments that jumped plan_version by 2 each REPLAN)."""
    fields = {"status": "active"}
    if phase == DIAGNOSE:
        fields["consecutive_failures"] = 0
    update_progress(task_dir, **fields)


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    stdout = str(hook_input.get("tool_output", ""))

    # --- Activation (export AR_TASK_DIR=...) ---
    target = _activation_target(command)
    if target:
        _handle_activation(target)
        # DO NOT early-exit. Compound commands like
        # `export AR_TASK_DIR=... && python .autoresearch/scripts/create_plan.py ...`
        # carry both an activation and a real script invocation in a single
        # Bash call (the model joins them because Claude Code's Bash tool
        # spawns a fresh shell per call so a previous `export AR_TASK_DIR=`
        # doesn't persist; on Windows it also prepends KMP_DUPLICATE_LIB_OK=
        # TRUE to the same call). If we returned here the script invocation
        # would be silently skipped — the plan would never get validated,
        # .phase would stay at PLAN, and the loop would stall.

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if invoked == "baseline.py" and phase == BASELINE:
        progress = load_progress(task_dir)
        if not progress:
            emit_status("[AR] Baseline failed (no progress.json). Retry.")
        elif (progress.get("seed_metric") is None
              or progress.get("baseline_correctness") is False):
            # Demote to GENERATE_KERNEL so Edit on kernel.py is permitted
            # again — BASELINE's _EDIT_RULES forbid it. The previous
            # cap-at-3 branch left phase pinned at BASELINE, which
            # deadlocked the task (kernel.py uneditable, baseline.py the
            # only allowed command, and re-running it just kept failing).
            # Always demoting back to GENERATE_KERNEL avoids that trap.
            reason = ("seed kernel produced no timing"
                      if progress.get("seed_metric") is None
                      else "seed kernel failed correctness check")
            write_phase(task_dir, GENERATE_KERNEL)
            emit_status(
                f"[AR] Baseline failed: {reason}. "
                f"Phase -> GENERATE_KERNEL so kernel.py becomes editable. "
                f"Fix the kernel, then re-run baseline.py."
            )
        else:
            write_phase(task_dir, PLAN)
            emit_status(f"[AR] Baseline complete. Phase -> PLAN. {get_guidance(task_dir)}")

    elif invoked == "pipeline.py":
        # pipeline.py writes .phase itself; just project state + notify.
        new_phase = read_phase(task_dir)
        emit_status(f"[AR] Pipeline complete. Phase -> {new_phase}. {get_guidance(task_dir)}")
        emit_todowrite_context(task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif invoked == "create_plan.py" and phase in (PLAN, DIAGNOSE, REPLAN):
        from phase_machine import validate_plan
        ok, err = validate_plan(task_dir)
        if ok:
            _progress_update_for_plan(task_dir, phase)
            write_phase(task_dir, EDIT)
            emit_status(f"[AR] Plan validated. Phase -> EDIT. {get_guidance(task_dir)}")
            emit_todowrite_context(task_dir, "[AR] Plan validated. Phase -> EDIT.")
        else:
            emit_status(f"[AR] Plan not valid yet: {err}")

    sys.exit(0)


def _print_resume_context(task_dir: str):
    progress = load_progress(task_dir)
    if not progress:
        return
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)

    emit_status(
        f"[AR] Resume context: Round {rounds}/{max_rounds} | "
        f"Best: {best} | Baseline: {baseline} | "
        f"Failures: {failures} | Plan v{plan_ver}"
    )

    hpath = history_path(task_dir)
    if os.path.exists(hpath):
        with open(hpath, "r") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            emit_status(f"[AR] Last {min(3, len(lines))} rounds:")
            for rec in lines[-3:]:
                rnd = rec.get("round")
                rnd = "?" if rnd is None else str(rnd)
                dec = rec.get("decision", "?")
                desc = rec.get("description", "")[:40]
                emit_status(f"[AR]   R{rnd}: {dec} — {desc}")

    if os.path.exists(plan_path(task_dir)):
        active = get_active_item(task_dir)
        if active:
            emit_status(f"[AR] Active item: {active['id']}: {active['description'][:50]}")
        emit_status("[AR] Read .ar_state/plan.md and .ar_state/history.jsonl for full context.")


if __name__ == "__main__":
    main()

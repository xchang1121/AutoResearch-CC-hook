#!/usr/bin/env python3
"""
PostToolUse hook for Bash — phase auto-advancement after user-issued commands.

Handlers:
  export AR_TASK_DIR=...    activate task (resume or fresh-start)
  baseline.py               PLAN on success; GENERATE_KERNEL on retry
  pipeline.py               whatever .phase pipeline.py wrote
  create_plan.py            EDIT on plan validation pass

Each handler is a one-shot function that ends with a single
`emit_transition` (or `emit_to_claude` for pure error / nudge paths).
That single emit produces ONE additionalContext JSON, which is what
makes the message survive PostToolUse parsers that keep only the final
JSON line.

The inner pipeline steps (quick_check / eval_wrapper / keep_or_discard /
settle) are subprocess children of pipeline.py and never re-enter this
hook, so they don't need their own branches here.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, emit_to_claude
from phase_machine import (
    BASELINE, DIAGNOSE, EDIT, GENERATE_KERNEL, GENERATE_REF,
    PHASE_FILE, PLAN, REPLAN,
    compute_resume_phase, edit_marker_path, emit_transition,
    get_task_dir, is_placeholder_file, load_progress,
    parse_invoked_ar_script, parse_last_json_line, progress_path,
    read_phase, set_task_dir, state_path, touch_heartbeat,
    update_progress, validate_kernel, validate_plan, validate_reference,
)


def _activation_target(command: str):
    if "AR_TASK_DIR=" not in command:
        return None
    m = re.search(r'AR_TASK_DIR=["\']?([^"\';\s&]+)', command)
    return m.group(1) if m else None


def _clean_stale_edit_marker(task_dir: str) -> None:
    """Remove .edit_started if git is clean — it's leftover from an earlier
    interrupted round and would otherwise make the next EDIT see a 'second
    edit' state immediately."""
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    try:
        diff = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=task_dir, capture_output=True, text=True, timeout=5,
        )
        if not diff.stdout.strip():
            os.remove(marker)
            emit_status("[AR] Cleaned stale edit marker (git is clean).")
    except Exception:
        pass


def _handle_activation(new_task_dir: str) -> None:
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        emit_to_claude(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    set_task_dir(new_task_dir)
    _clean_stale_edit_marker(new_task_dir)

    has_phase = os.path.exists(state_path(new_task_dir, PHASE_FILE))
    has_progress = os.path.exists(progress_path(new_task_dir))

    if has_phase:
        emit_transition(
            new_task_dir,
            f"[AR] Resuming. Phase: {read_phase(new_task_dir)}.",
            include_resume_context=True,
        )
    elif has_progress:
        phase = compute_resume_phase(new_task_dir)
        emit_transition(
            new_task_dir,
            f"[AR] Resuming from progress. Phase -> {phase}.",
            write_to_phase=phase,
            include_resume_context=True,
        )
    else:
        _fresh_start(new_task_dir)


def _fresh_start(task_dir: str) -> None:
    """Pick initial phase based on which task files are present and runnable.
    Stays in lockstep with `is_placeholder_file` and the validators — same
    rules as the EDIT post-hook uses for kernel.py advancement."""
    ref_path = os.path.join(task_dir, "reference.py")
    kernel_path = os.path.join(task_dir, "kernel.py")

    if is_placeholder_file(ref_path):
        emit_transition(task_dir, "[AR] Fresh start (no reference).",
                        write_to_phase=GENERATE_REF)
        return

    ok, err = validate_reference(task_dir)
    if not ok:
        emit_transition(task_dir,
                        "[AR] reference.py present but invalid.",
                        write_to_phase=GENERATE_REF, extra=err)
        return

    if is_placeholder_file(kernel_path):
        emit_transition(task_dir, "[AR] Fresh start (no kernel).",
                        write_to_phase=GENERATE_KERNEL)
        return

    ok, err = validate_kernel(task_dir)
    if not ok:
        emit_transition(task_dir,
                        "[AR] kernel.py present but invalid.",
                        write_to_phase=GENERATE_KERNEL, extra=err)
        return

    emit_transition(task_dir, "[AR] Fresh start.", write_to_phase=BASELINE)


def _progress_update_for_plan(task_dir: str, phase: str) -> None:
    """Mark progress as active on a fresh plan. plan_version belongs to
    create_plan.py (it bumps on write); the hook MUST NOT re-bump or
    plan_version jumps by 2 each REPLAN."""
    fields = {"status": "active"}
    if phase == DIAGNOSE:
        fields["consecutive_failures"] = 0
    update_progress(task_dir, **fields)


def _handle_baseline(task_dir: str) -> None:
    """baseline.py post-action.

    Success: clear retry counter, advance to PLAN.
    Failure: bump retry counter; below 3 retries, demote to GENERATE_KERNEL
    so kernel.py becomes editable again (BASELINE forbids edits to it);
    at 3, stop auto-retry and ask for a manual fix."""
    progress = load_progress(task_dir)
    if not progress:
        emit_to_claude("[AR] Baseline failed (no progress.json). Retry.")
        return

    seed_ok = (progress.get("seed_metric") is not None
               and progress.get("baseline_correctness") is not False)
    if seed_ok:
        update_progress(task_dir, baseline_retries=0)
        emit_transition(task_dir, "[AR] Baseline complete.",
                        write_to_phase=PLAN)
        return

    retries = int(progress.get("baseline_retries", 0)) + 1
    update_progress(task_dir, baseline_retries=retries)
    reason = ("seed kernel produced no timing"
              if progress.get("seed_metric") is None
              else "seed kernel failed correctness check")
    if retries >= 3:
        emit_to_claude(
            f"[AR] Baseline failed {retries}x ({reason}). Stopping auto-retry. "
            f"Inspect the worker log, fix kernel.py manually, then re-run: "
            f"python .autoresearch/scripts/baseline.py \"{task_dir}\""
        )
    else:
        emit_transition(
            task_dir,
            f"[AR] Baseline failed (attempt {retries}/3): {reason}.",
            write_to_phase=GENERATE_KERNEL,
        )


def _handle_create_plan(task_dir: str, phase: str, stdout: str) -> None:
    """create_plan.py post-action.

    Success policy:
      * `{"ok": false, ...}` on stdout → block advance, surface the error.
      * `{"ok": true, ...}` OR no parseable JSON → fall through to
        validate_plan(plan.md). Falling through on unrecognized payload
        avoids punishing a successful run when the hook payload format
        drifts under us; validate_plan is the final gate either way.

    The substring-match bug (`echo create_plan.py` wrongly advancing) is
    blocked upstream by parse_invoked_ar_script's regex — it only matches
    a real python invocation."""
    run_blob = parse_last_json_line(stdout)
    if run_blob is not None and run_blob.get("ok") is False:
        err = run_blob.get("error", "(no error message in stdout)")
        emit_to_claude(
            f"[AR] Plan not advanced — create_plan.py reported failure: {err}"
        )
        return

    ok, err = validate_plan(task_dir)
    if not ok:
        emit_to_claude(f"[AR] Plan not valid yet: {err}")
        return

    _progress_update_for_plan(task_dir, phase)
    emit_transition(task_dir, "[AR] Plan validated.", write_to_phase=EDIT)


def _read_stdout(hook_input: dict) -> str:
    """Pull the bash command's stdout from any of the payload shapes Claude
    Code has used. Older runs flat-strung `tool_output`; current runs put
    stdout/stderr under `tool_response`. Walk most-specific to least so a
    rename doesn't silently drop our success-check."""
    tool_resp = hook_input.get("tool_response", {})
    if isinstance(tool_resp, dict):
        s = str(tool_resp.get("stdout", "") or tool_resp.get("output", ""))
    else:
        s = str(tool_resp)
    return s or str(hook_input.get("tool_output", ""))


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    stdout = _read_stdout(hook_input)

    target = _activation_target(command)
    if target:
        _handle_activation(target)
        # DO NOT early-exit. Compound commands like
        # `export AR_TASK_DIR=... && python .autoresearch/scripts/create_plan.py ...`
        # carry both an activation and a real script invocation in a single
        # Bash call (the model joins them because the shell doesn't persist
        # env between Bash tool calls). If we returned here the script
        # invocation would be silently skipped — the plan would never get
        # validated, .phase would stay at PLAN, and the loop would stall.
        # Fall through so the invoked-script dispatch below also runs.

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if invoked == "baseline.py" and phase == BASELINE:
        _handle_baseline(task_dir)
    elif invoked == "pipeline.py":
        # pipeline.py writes .phase itself; just project state + notify.
        emit_transition(
            task_dir,
            f"[AR] Pipeline complete. Phase -> {read_phase(task_dir)}.",
        )
    elif invoked == "create_plan.py" and phase in (PLAN, DIAGNOSE, REPLAN):
        _handle_create_plan(task_dir, phase, stdout)

    sys.exit(0)


if __name__ == "__main__":
    main()

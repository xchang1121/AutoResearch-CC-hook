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
                                (capped at 3 retries via baseline_retries)
  - `pipeline.py`             → whatever phase pipeline.py itself wrote
  - `create_plan.py`          → EDIT on plan validation pass
                                (called from PLAN / DIAGNOSE / REPLAN)

The inner pipeline steps (quick_check / eval_wrapper / keep_or_discard /
settle) are subprocess children of pipeline.py and never re-enter this hook,
so they don't need their own phase constants or branches here.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, emit_todowrite_context
from phase_machine import (
    read_phase, write_phase, get_guidance, compute_resume_phase,
    get_task_dir, set_task_dir, get_active_item, touch_heartbeat,
    load_history, load_progress, update_progress,
    validate_reference, validate_kernel, is_placeholder_file,
    parse_invoked_ar_script, parse_last_json_line,
    progress_path, plan_path, edit_marker_path, state_path,
    PHASE_FILE,
    BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN, GENERATE_REF, GENERATE_KERNEL,
)


def _activation_target(command: str) -> str | None:
    if "AR_TASK_DIR=" not in command:
        return None
    m = re.search(r'AR_TASK_DIR=["\']?([^"\';\s&]+)', command)
    return m.group(1) if m else None


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
    # Claude Code's PostToolUse payload shape varies by version: older runs
    # used a flat `tool_output` string; current runs put both stdout/stderr
    # under `tool_response`. We accept any, falling back from most-specific
    # to least. If none populate, `stdout` ends up "" and downstream
    # last-JSON-line parsing returns None — see _command_succeeded() below.
    tool_resp = hook_input.get("tool_response", {})
    if isinstance(tool_resp, dict):
        stdout = str(tool_resp.get("stdout", "") or tool_resp.get("output", ""))
    else:
        stdout = str(tool_resp)
    if not stdout:
        stdout = str(hook_input.get("tool_output", ""))

    # --- Activation (export AR_TASK_DIR=...) ---
    target = _activation_target(command)
    if target:
        _handle_activation(target)
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)

    # Identify which AR script the user actually invoked. Substring matching
    # ('baseline.py' in command) used to fire on `echo baseline.py` and
    # advance phase based on a coincidence; the parser only matches a real
    # python invocation of `.autoresearch/scripts/<name>.py`.
    invoked_script = parse_invoked_ar_script(command)

    if invoked_script == "baseline.py" and phase == BASELINE:
        progress = load_progress(task_dir)
        if not progress:
            emit_status("[AR] Baseline failed (no progress.json). Retry.")
        elif (progress.get("seed_metric") is None
              or progress.get("baseline_correctness") is False):
            # Demote to GENERATE_KERNEL so Edit on kernel.py is permitted
            # again — BASELINE's _EDIT_RULES forbid it. Cap at 3 attempts
            # so a fundamentally-broken kernel doesn't loop forever; after
            # that we leave the phase pinned and ask for a manual fix.
            retries = int(progress.get("baseline_retries", 0)) + 1
            update_progress(task_dir, baseline_retries=retries)
            reason = ("seed kernel produced no timing"
                      if progress.get("seed_metric") is None
                      else "seed kernel failed correctness check")
            if retries >= 3:
                emit_status(
                    f"[AR] Baseline failed {retries}x ({reason}). "
                    f"Stopping auto-retry. Inspect the worker log, fix "
                    f"kernel.py manually, then re-run: "
                    f"python .autoresearch/scripts/baseline.py \"{task_dir}\""
                )
            else:
                write_phase(task_dir, GENERATE_KERNEL)
                emit_status(
                    f"[AR] Baseline failed (attempt {retries}/3): {reason}. "
                    f"Phase -> GENERATE_KERNEL so kernel.py becomes editable. "
                    f"Fix the kernel, then re-run baseline.py."
                )
        else:
            update_progress(task_dir, baseline_retries=0)
            write_phase(task_dir, PLAN)
            emit_status(f"[AR] Baseline complete. Phase -> PLAN. {get_guidance(task_dir)}")

    elif invoked_script == "pipeline.py":
        # pipeline.py writes .phase itself; just project state + notify.
        new_phase = read_phase(task_dir)
        emit_status(f"[AR] Pipeline complete. Phase -> {new_phase}. {get_guidance(task_dir)}")
        emit_todowrite_context(task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif invoked_script == "create_plan.py" and phase in (PLAN, DIAGNOSE, REPLAN):
        # Success-check policy:
        #   - `{"ok": false, ...}` in stdout → block advance, surface error.
        #     (Reliable failure signal: create_plan.py prints this when it
        #     rejects the input.)
        #   - `{"ok": true, ...}` OR no parseable JSON (e.g. tool_output
        #     payload shape we don't recognize) → fall through to
        #     validate_plan(plan.md). Validate is the final gate; if plan.md
        #     is well-formed it advances, if not it stays. Falling through
        #     when stdout is missing avoids punishing a successful run just
        #     because the hook payload shape changed under us.
        #
        # The substring-match bug (echo create_plan.py wrongly advancing) is
        # already prevented by `invoked_script == "create_plan.py"` above —
        # parse_invoked_ar_script requires a real python invocation.
        run_blob = parse_last_json_line(stdout)
        if run_blob is not None and run_blob.get("ok") is False:
            err = run_blob.get("error", "(no error message in stdout)")
            emit_status(f"[AR] Plan not advanced — create_plan.py reported failure: {err}")
        else:
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

    history = load_history(task_dir, on_corrupt="skip")
    if history:
        emit_status(f"[AR] Last {min(3, len(history))} rounds:")
        for rec in history[-3:]:
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

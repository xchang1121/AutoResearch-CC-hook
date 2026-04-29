#!/usr/bin/env python3
"""
PostToolUse hook for Edit/Write — advances phase after code edits.

- reference.py in GENERATE_REF → GENERATE_KERNEL or BASELINE (depending on
  whether kernel.py is still a placeholder)
- editable file in GENERATE_KERNEL → BASELINE
- editable file in EDIT → no phase change; Claude runs pipeline.py when done

plan.md is never a legal target for Edit/Write — hook_guard_edit blocks it
at every phase and directs Claude to create_plan.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, norm_abs_fwd_slash, extract_target_path
from phase_machine import (
    read_phase, write_phase, get_guidance, _load_config_safe,
    get_task_dir, touch_heartbeat,
    validate_reference, validate_kernel, is_placeholder_file,
    EDIT, BASELINE, GENERATE_REF, GENERATE_KERNEL,
)
from git_utils import commit_in_task


def _same_path(a: str, b: str) -> bool:
    return norm_abs_fwd_slash(a) == norm_abs_fwd_slash(b)


_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in _WRITE_TOOLS:
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = extract_target_path(hook_input)
    if not file_path:
        sys.exit(0)

    phase = read_phase(task_dir)
    is_ref = _same_path(file_path, os.path.join(task_dir, "reference.py"))

    config = _load_config_safe(task_dir)
    is_editable = False
    if config:
        try:
            rel = os.path.relpath(file_path, task_dir).replace("\\", "/")
            is_editable = rel in set(config.editable_files)
        except ValueError:
            is_editable = False

    if is_ref and phase == GENERATE_REF:
        ok, err = validate_reference(task_dir)
        if not ok:
            emit_status(
                f"[AR] reference.py invalid — phase stays at GENERATE_REF.\n"
                f"     {err}\n"
                f"     Re-Edit reference.py to fix; baseline will not run "
                f"until it passes."
            )
            sys.exit(0)
        # Freeze the generated reference.py as a git baseline before advancing.
        # If the commit fails we HOLD phase here — the alternative is silent
        # advance with an uncommitted file, which surfaces as a misleading
        # "previous round" block several phases later.
        commit_ok, info = commit_in_task(
            task_dir, ["reference.py"],
            "autoresearch: seed reference.py (GENERATE_REF)",
        )
        if not commit_ok:
            emit_status(
                f"[AR] reference.py validated but seed commit FAILED — "
                f"phase stays at GENERATE_REF.\n"
                f"     git error: {info}\n"
                f"     Resolve the git issue (e.g. clear .git/index.lock, "
                f"check disk space, fix .git/config), then re-Edit reference.py "
                f"to retry the commit."
            )
            sys.exit(0)
        # Route to GENERATE_KERNEL if kernel.py is still the scaffold
        # placeholder, else straight to BASELINE.
        next_phase = GENERATE_KERNEL if is_placeholder_file(
            os.path.join(task_dir, "kernel.py")
        ) else BASELINE
        write_phase(task_dir, next_phase)
        emit_status(f"[AR] reference.py validated. Phase -> {next_phase}. {get_guidance(task_dir)}")

    elif is_editable and phase == GENERATE_KERNEL:
        ok, err = validate_kernel(task_dir)
        if not ok:
            emit_status(
                f"[AR] kernel.py invalid — phase stays at GENERATE_KERNEL.\n"
                f"     {err}\n"
                f"     Re-Edit kernel.py to fix; baseline will not run "
                f"until it passes."
            )
            sys.exit(0)
        # Freeze the seed kernel as git baseline. If commit fails we HOLD
        # phase at GENERATE_KERNEL — same reasoning as the GENERATE_REF
        # branch above. The agent's next Edit re-fires the hook and retries.
        editable_files = list(config.editable_files) if config else ["kernel.py"]
        commit_ok, info = commit_in_task(
            task_dir, editable_files,
            "autoresearch: seed kernel (GENERATE_KERNEL)",
        )
        if not commit_ok:
            emit_status(
                f"[AR] kernel.py validated but seed commit FAILED — "
                f"phase stays at GENERATE_KERNEL.\n"
                f"     git error: {info}\n"
                f"     Resolve the git issue, then re-Edit kernel.py to "
                f"retry the commit."
            )
            sys.exit(0)
        write_phase(task_dir, BASELINE)
        emit_status(f"[AR] kernel.py validated. Phase -> BASELINE. {get_guidance(task_dir)}")

    elif is_editable and phase == EDIT:
        emit_status(
            f"[AR] Code edited. Continue editing OR run: "
            f"python .autoresearch/scripts/pipeline.py \"{task_dir}\""
        )

    sys.exit(0)


if __name__ == "__main__":
    main()

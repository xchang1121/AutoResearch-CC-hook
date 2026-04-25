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
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_to_claude
from phase_machine import (
    read_phase, _load_config_safe,
    get_task_dir, touch_heartbeat, emit_transition,
    validate_reference, validate_kernel, is_placeholder_file,
    EDIT, BASELINE, GENERATE_REF, GENERATE_KERNEL,
)


def _same_path(a: str, b: str) -> bool:
    norm = lambda p: os.path.normpath(os.path.abspath(p)).replace("\\", "/")
    return norm(a) == norm(b)


def _commit_seed(task_dir: str, paths, message: str) -> str | None:
    """Stage `paths` (task-dir-relative) and create a commit with `message`.

    Makes the newly generated reference.py / seed kernel.py the git baseline
    for downstream steps: _baseline_init records HEAD as baseline_commit,
    rollback via `git checkout HEAD -- kernel.py` restores the seed (not the
    scaffold placeholder), and hook_guard_edit's "uncommitted previous-round"
    gate sees a clean tree on first entry to EDIT.

    Silently no-ops when there's nothing to commit (e.g. file was already
    committed by a prior path). Failures are logged to stderr but never abort
    the phase transition — the hook can't refuse to advance phase just because
    `git` is unhappy, that would strand the task.
    """
    try:
        for p in paths:
            full = os.path.join(task_dir, p)
            if not os.path.exists(full):
                continue
            subprocess.run(["git", "add", "--", p],
                           cwd=task_dir, capture_output=True, text=True, timeout=10)
        r = subprocess.run(["git", "commit", "-m", message],
                           cwd=task_dir, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            blob = (r.stdout or "") + (r.stderr or "")
            if "nothing to commit" in blob or "no changes added" in blob:
                return None
            print(f"[AR] WARNING: seed commit failed: {blob.strip()[-400:]}",
                  file=sys.stderr)
            return None
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=task_dir, capture_output=True, text=True, timeout=5)
        return h.stdout.strip() if h.returncode == 0 else None
    except Exception as e:
        print(f"[AR] WARNING: seed commit failed: {e}", file=sys.stderr)
        return None


def _resolved_file_path(tool_name: str, tool_input: dict) -> str:
    """Pick the single file path the post-edit hook should react to.

    Edit/Write/NotebookEdit have one path per call. MultiEdit's batch
    targets a single `file_path` in current Claude Code (the `edits` list
    only carries old/new strings), so the same field works there.
    Phase advancement is per-tool-call, not per-edit, so collapsing the
    batch to its file_path is correct.
    """
    if tool_name in ("Edit", "Write", "MultiEdit"):
        return tool_input.get("file_path", "") or ""
    if tool_name == "NotebookEdit":
        return tool_input.get("notebook_path", "") or tool_input.get("file_path", "") or ""
    return ""


def main():
    hook_input = read_hook_input()
    tool_name = hook_input.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = _resolved_file_path(tool_name, hook_input.get("tool_input", {}))
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
            emit_to_claude(
                "[AR] reference.py invalid — phase stays at GENERATE_REF.",
                f"     {err}",
                "     Re-Edit reference.py to fix; baseline will not run "
                "until it passes.",
            )
            sys.exit(0)
        # Reference is good. Route to GENERATE_KERNEL if kernel.py is still
        # the scaffold placeholder, else straight to BASELINE.
        next_phase = GENERATE_KERNEL if is_placeholder_file(
            os.path.join(task_dir, "kernel.py")
        ) else BASELINE
        # Freeze the generated reference.py as a git baseline before advancing
        # so later steps (baseline HEAD read, rollback on FAIL) see the seed,
        # not the scaffold placeholder.
        _commit_seed(task_dir, ["reference.py"],
                     "autoresearch: seed reference.py (GENERATE_REF)")
        emit_transition(task_dir, "[AR] reference.py validated.",
                        write_to_phase=next_phase)

    elif is_editable and phase == GENERATE_KERNEL:
        ok, err = validate_kernel(task_dir)
        if not ok:
            emit_to_claude(
                "[AR] kernel.py invalid — phase stays at GENERATE_KERNEL.",
                f"     {err}",
                "     Re-Edit kernel.py to fix; baseline will not run "
                "until it passes.",
            )
            sys.exit(0)
        # Freeze the seed kernel as git baseline. Without this, the first
        # EDIT round's guard would misread the seed as "uncommitted previous-
        # round leftover", and rollbacks would revert kernel.py to the
        # scaffold TODO placeholder instead of the seed.
        editable_files = list(config.editable_files) if config else ["kernel.py"]
        _commit_seed(task_dir, editable_files,
                     "autoresearch: seed kernel (GENERATE_KERNEL)")
        emit_transition(task_dir, "[AR] kernel.py validated.",
                        write_to_phase=BASELINE)

    elif is_editable and phase == EDIT:
        emit_to_claude(
            "[AR] Code edited. Continue editing OR run: "
            f"python .autoresearch/scripts/pipeline.py \"{task_dir}\""
        )

    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PreToolUse hook for Edit/Write — thin dispatcher.

Per-phase allow/block for file targets lives in phase_machine.check_edit.
This hook handles two concerns check_edit can't express as a pure function:

  1. Files outside the task dir are always allowed (out of scope).
  2. EDIT phase's "uncommitted-diff from previous round" gate — needs git
     state, not just phase; stays in the hook.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input
from phase_machine import (
    read_phase, get_guidance, get_task_dir, touch_heartbeat,
    edit_marker_path, check_edit, EDIT,
)


def _block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


def _rel_to_task(file_path: str, task_dir: str):
    """Return task-relative forward-slash path, or None if outside task_dir."""
    fp = os.path.normpath(os.path.abspath(file_path)).replace("\\", "/")
    td = os.path.normpath(os.path.abspath(task_dir)).replace("\\", "/")
    if not fp.startswith(td):
        return None
    return os.path.relpath(file_path, task_dir).replace("\\", "/")


def _edit_phase_git_gate(task_dir: str, editable_files):
    """In EDIT phase, if any editable file has uncommitted changes AND the
    edit-started marker is absent, that's leftover from a previous round —
    force Claude to run pipeline.py before editing more.

    Sets the marker on success so subsequent edits within the same round
    pass through.
    """
    try:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return  # git unavailable — skip gate rather than block

    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        return  # already in an active round

    for ef in editable_files:
        rel_in_repo = os.path.relpath(os.path.join(task_dir, ef), repo_root)
        try:
            diff = subprocess.run(
                ["git", "diff", "--name-only", "--", rel_in_repo],
                cwd=repo_root, capture_output=True, text=True, timeout=5,
            )
        except Exception:
            continue
        if diff.stdout.strip():
            _block(
                f"[AR] Uncommitted changes from previous round detected. "
                f"Run pipeline.py to finalize before editing: "
                f"python .autoresearch/scripts/pipeline.py \"{task_dir}\""
            )

    # Start-of-round marker so re-editing the same file doesn't re-fire this gate
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write("1")


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in ("Edit", "Write"):
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = hook_input.get("tool_input", {}).get("file_path", "")
    if not file_path:
        sys.exit(0)

    rel = _rel_to_task(file_path, task_dir)
    if rel is None:
        sys.exit(0)  # file outside task_dir — not our concern

    from task_config import load_task_config
    config = load_task_config(task_dir)
    editable_files = list(config.editable_files) if config else []

    phase = read_phase(task_dir)
    ok, reason = check_edit(phase, rel, editable_files)
    if not ok:
        _block(f"[AR] {reason}. {get_guidance(task_dir)}")

    # Phase says OK. For EDIT writes to editable files, also check the git
    # state gate (previous-round leftovers).
    if phase == EDIT and rel in set(editable_files):
        _edit_phase_git_gate(task_dir, editable_files)

    sys.exit(0)


if __name__ == "__main__":
    main()

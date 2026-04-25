#!/usr/bin/env python3
"""
PreToolUse hook for Bash — thin dispatcher.

All per-phase allow/block logic lives in phase_machine.check_bash.
This hook only handles two hook-specific concerns:
  1. Script-name sanity (blessed names / hallucinated-name suggestions)
  2. Turning check_bash's (False, reason) into the `{decision: block}` wire
     format Claude Code expects.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input
from phase_machine import (
    read_phase, get_guidance, get_task_dir, touch_heartbeat, check_bash,
)
from settings import hallucinated_scripts

# Real scripts under .autoresearch/scripts/ that have a __main__ entry and
# are intended to be invoked as CLIs. Sourced as the canonical list; mirrors
# the table in .claude/commands/autoresearch.md ("Scripts under ...").
_BLESSED_SCRIPTS = {
    "quick_check.py", "eval_wrapper.py", "keep_or_discard.py",
    "scaffold.py", "baseline.py", "_baseline_init.py", "dashboard.py",
    "create_plan.py", "settle.py", "pipeline.py", "resume.py",
    "code_checker.py", "final_report.py", "ar_cli.py",
}

# Library modules that look invokable but have no `__main__` and exist only
# to be imported. The block message points the model at the right place
# instead of just saying "unknown script". Keep this in sync with the
# "Library files — do NOT invoke directly" list in autoresearch.md.
_LIBRARY_SCRIPTS = {
    "phase_machine.py": "imported by hooks and pipeline; "
                        "to inspect phase, run `cat \"$AR_TASK_DIR/.ar_state/.phase\"` "
                        "or `python .autoresearch/scripts/dashboard.py`",
    "task_config.py":   "yaml loader and eval orchestrator; "
                        "use baseline.py / pipeline.py to drive eval",
    "local_worker.py":  "in-process verify/profile executor; "
                        "use baseline.py / pipeline.py — they pick local vs remote automatically",
    "hook_utils.py":    "hook helpers — invoked only by Claude Code's hook system, never directly",
    "hw_detect.py":     "hardware probe — used internally by local_worker / scaffold",
    "settings.py":      "config-yaml accessor — imported, not run",
    "hook_guard_bash.py":  "PreToolUse hook — invoked by Claude Code, never directly",
    "hook_guard_edit.py":  "PreToolUse hook — invoked by Claude Code, never directly",
    "hook_post_bash.py":   "PostToolUse hook — invoked by Claude Code, never directly",
    "hook_post_edit.py":   "PostToolUse hook — invoked by Claude Code, never directly",
    "hook_stop_save.py":   "Stop hook — invoked by Claude Code, never directly",
    "failure_extractor.py": "log parser — imported by eval_wrapper / pipeline",
}

# Alias → real script mapping lives in .autoresearch/config.yaml under
# `hallucinated_scripts`; loaded lazily so the config can be hot-edited.


def _block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


def _script_name_check(command: str):
    """Flag unknown / hallucinated .autoresearch/scripts/*.py names before
    they reach the phase rule — gives a clearer message than 'not allowed'."""
    m = re.search(r'python\s+["\']?([^\s"\']+\.py)', command)
    if not m:
        return
    script_path = m.group(1).replace("\\", "/")
    script_name = os.path.basename(script_path)

    aliases = hallucinated_scripts()
    if script_name in aliases:
        real = aliases[script_name]
        _block(f"[AR] '{script_name}' does not exist. "
               f"Use: python .autoresearch/scripts/{real}")

    if ".autoresearch/scripts/" not in script_path:
        return
    if script_name in _BLESSED_SCRIPTS:
        return

    # Distinguish "library, no CLI" from "totally unknown" — the former is
    # the most common Claude mistake and benefits from a directed hint.
    if script_name in _LIBRARY_SCRIPTS:
        hint = _LIBRARY_SCRIPTS[script_name]
        _block(f"[AR] '{script_name}' is a library module (no __main__) — "
               f"do not invoke it directly. {hint}.")
    _block(f"[AR] Unknown script '{script_name}'. "
           f"Valid CLIs: {sorted(_BLESSED_SCRIPTS)}. "
           f"Library files (phase_machine.py, task_config.py, hook_*.py, …) "
           f"are imported, not run — see .claude/commands/autoresearch.md.")


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    command = hook_input.get("tool_input", {}).get("command", "")
    _script_name_check(command)

    phase = read_phase(task_dir)
    ok, reason = check_bash(phase, command)
    if not ok:
        _block(f"[AR] {reason}. {get_guidance(task_dir)}")
    sys.exit(0)


if __name__ == "__main__":
    main()

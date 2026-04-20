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

# Real scripts that live under .autoresearch/scripts/.
_BLESSED_SCRIPTS = {
    "quick_check.py", "eval_wrapper.py", "keep_or_discard.py",
    "scaffold.py", "baseline.py", "_baseline_init.py", "dashboard.py",
    "create_plan.py", "settle.py", "pipeline.py", "resume.py",
    "reference_capture.py",
}

# Names Claude sometimes hallucinates → real script it should use instead.
_HALLUCINATED_SCRIPTS = {
    "eval.py":     "eval_wrapper.py",
    "run_eval.py": "eval_wrapper.py",
    "verify.py":   "eval_wrapper.py",
    "check.py":    "quick_check.py",
    "run.py":      "eval_wrapper.py",
    "test.py":     "quick_check.py",
    "profile.py":  "eval_wrapper.py",
}


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

    if script_name in _HALLUCINATED_SCRIPTS:
        real = _HALLUCINATED_SCRIPTS[script_name]
        _block(f"[AR] '{script_name}' does not exist. "
               f"Use: python .autoresearch/scripts/{real}")
    if ".autoresearch/scripts/" in script_path and script_name not in _BLESSED_SCRIPTS:
        _block(f"[AR] Unknown script '{script_name}'. "
               f"Valid scripts: {sorted(_BLESSED_SCRIPTS)}")


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

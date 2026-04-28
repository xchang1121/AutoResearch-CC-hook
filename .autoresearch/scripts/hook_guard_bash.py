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
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input
from phase_machine import (
    read_phase, get_guidance, get_task_dir, touch_heartbeat, check_bash,
    parse_script_name,
)
from settings import hallucinated_scripts

# Real scripts that live under .autoresearch/scripts/.
_BLESSED_SCRIPTS = {
    "quick_check.py", "eval_wrapper.py", "keep_or_discard.py",
    "scaffold.py", "baseline.py", "_baseline_init.py", "dashboard.py",
    "create_plan.py", "settle.py", "pipeline.py", "resume.py",
    "code_checker.py", "parse_args.py",
}

# Library modules that live under .autoresearch/scripts/ but have no CLI.
# The LLM repeatedly tries to invoke `phase_machine.py get_guidance ...`
# because CLAUDE.md / README.md used to namedrop `phase_machine.get_guidance()`
# in a way that read like a callable. The docs are fixed, but a pointed
# error here gives the LLM an unambiguous nudge if it reaches for them
# again instead of just listing _BLESSED_SCRIPTS and walking off.
_LIBRARY_NOT_CLI = {
    "phase_machine.py": (
        "phase_machine.py is a library used by hooks, not a CLI. "
        "Guidance ([AR Phase: ...]) is auto-emitted on stderr after every "
        "legal Bash/Edit. If you haven't seen a fresh guidance message, "
        "wait for the next hook output — do not try to fetch it manually."
    ),
    "task_config.py": "task_config.py is a library, not a CLI.",
    "settings.py": "settings.py is a library, not a CLI.",
    "hw_detect.py": "hw_detect.py is a library, not a CLI.",
    "hook_utils.py": "hook_utils.py is a library, not a CLI.",
    "failure_extractor.py": "failure_extractor.py is a library, not a CLI.",
}

# Alias → real script mapping lives in .autoresearch/config.yaml under
# `hallucinated_scripts`; loaded lazily so the config can be hot-edited.


def _block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


def _script_name_check(command: str):
    """Flag unknown / hallucinated .autoresearch/scripts/*.py names before
    they reach the phase rule — gives a clearer message than 'not allowed'.

    Uses phase_machine.parse_script_name (shared with hook_post_bash) so
    both hooks agree on what counts as a python invocation, including
    `python3 ...`, `py -3 ...`, `bash some.py`. The earlier hand-rolled
    regex here matched only `python <script>` and silently disagreed with
    hook_post_bash on edge cases.
    """
    parsed = parse_script_name(command)
    if parsed is None:
        return
    script_path, script_name = parsed

    aliases = hallucinated_scripts()
    if script_name in aliases:
        real = aliases[script_name]
        _block(f"[AR] '{script_name}' does not exist. "
               f"Use: python .autoresearch/scripts/{real}")
    if script_name in _LIBRARY_NOT_CLI:
        _block(f"[AR] {_LIBRARY_NOT_CLI[script_name]}")
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

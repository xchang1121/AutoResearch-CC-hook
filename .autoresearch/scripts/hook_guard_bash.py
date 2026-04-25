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
    read_phase, get_task_dir, check_bash, INIT,
)
from settings import hallucinated_scripts

# User-facing scripts under .autoresearch/scripts/. These are legal command
# targets when the phase machine says so. Pipeline internals are listed
# separately below so they can be rejected even before AR_TASK_DIR exists.
_USER_CLI_SCRIPTS = {
    "scaffold.py", "resume.py", "baseline.py", "create_plan.py",
    "pipeline.py", "final_report.py", "dashboard.py", "worker_ctl.py",
}

_INTERNAL_CLI_SCRIPTS = {
    "quick_check.py": "pipeline-internal static check; run pipeline.py instead",
    "eval_wrapper.py": "pipeline/baseline-internal eval runner; run baseline.py or pipeline.py instead",
    "keep_or_discard.py": "pipeline-internal decision step; run pipeline.py instead",
    "settle.py": "pipeline-internal plan settlement; run pipeline.py instead",
    "_baseline_init.py": "baseline.py internal state initializer; run baseline.py instead",
    "code_checker.py": "static-check engine imported by quick_check.py; run pipeline.py (or baseline.py during setup) instead",
}

# Library modules that Claude most commonly mis-invokes as CLIs. These get a
# directed hint pointing at the right alternative. Anything else under
# .autoresearch/scripts/ that isn't a CLI falls through to the generic
# fallback in `_script_name_check`. Keep this list SHORT — every entry here
# is a filename Claude will see in the block message and may try later.
_LIBRARY_HINTS = {
    "phase_machine.py": "to inspect phase, run "
                        "`cat \"$AR_TASK_DIR/.ar_state/.phase\"`",
    "task_config.py":   "yaml loader and eval orchestrator; "
                        "use baseline.py / pipeline.py to drive eval",
    "local_worker.py":  "in-process verify/profile executor; "
                        "use baseline.py / pipeline.py (they pick local "
                        "vs remote automatically)",
}

# Alias → real script mapping lives in .autoresearch/config.yaml under
# `hallucinated_scripts`; loaded lazily so the config can be hot-edited.


def _block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


_SCRIPT_INVOKE_RE = re.compile(
    r'\b(?:python(?:\d+(?:\.\d+)?)?|py|bash|sh)\b'
    r'(?:\s+-[A-Za-z][^\s"\']*)*'
    r'\s+["\']?([^\s"\']+\.py)'
)


def _script_name_check(command: str):
    """Flag unknown / hallucinated .autoresearch/scripts/*.py names before
    they reach the phase rule — gives a clearer message than 'not allowed'."""
    m = _SCRIPT_INVOKE_RE.search(command)
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
    if script_name in _USER_CLI_SCRIPTS:
        return
    if script_name in _INTERNAL_CLI_SCRIPTS:
        _block(f"[AR] '{script_name}' is an internal script — "
               f"{_INTERNAL_CLI_SCRIPTS[script_name]}.")

    # Directed hints for the handful of library modules Claude most often
    # mis-invokes. Everything else falls through to the generic message
    # below — deliberately no script-name list, to avoid teaching landmarks.
    if script_name in _LIBRARY_HINTS:
        _block(f"[AR] '{script_name}' is a library module (no __main__) — "
               f"{_LIBRARY_HINTS[script_name]}.")
    _block(f"[AR] '{script_name}' is not a user-facing CLI in this project. "
           f"Run only the script the latest [AR Phase: ...] guidance names; "
           f"see CLAUDE.md for the user-facing CLI list.")


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    _script_name_check(command)

    # Heartbeat is touched by activation (set_task_dir) and PostToolUse;
    # PreToolUse must not, or resume.py self-blocks on its own fresh stamp.
    task_dir = get_task_dir()
    # No active task → pre-activation. Model that as phase=INIT so the same
    # check_bash path enforces the small allowlist (scaffold / resume / always-
    # allowed). Previously the hook just exit-0'd here, which let baseline.py
    # / create_plan.py / pipeline.py / final_report.py all run pre-activation.
    phase = read_phase(task_dir) if task_dir else INIT
    ok, reason = check_bash(phase, command)
    if not ok:
        # Don't paste full guidance into every block — the model already
        # received it from the most recent [AR Phase: ...] message. Just
        # name the phase and direct back to that.
        _block(f"[AR] {reason}. (Phase: {phase}; see the latest "
               f"[AR Phase: {phase}] guidance for the next legal step.)")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""
Shared I/O primitives for Claude Code hook scripts.

Three channels:
  emit_status(msg)      stderr only. Incidental status (transcript).
  emit_to_claude(*parts) stderr + ONE PostToolUse `additionalContext` JSON,
                         combining all parts. Use for any message the model
                         must act on.
  block(reason)         PreToolUse refusal. Exits 2.

Phase-transition emission is built on `emit_to_claude` and lives in
`phase_machine.emit_transition` because it bakes in policy (which
guidance / resume context / TodoWrite addendum to attach). hook_utils.py
stays as pure I/O.
"""
import json
import re
import sys


def read_hook_input() -> dict:
    """Read and parse hook input from stdin.

    Tolerates the Windows-path JSON quirk: paths like `C:\\Users` come in
    with a single backslash that's not a valid JSON escape, so a fallback
    pass doubles every unescaped `\\` before retrying.

    If both attempts fail, prints a clear warning to stderr (with the head
    of the bad payload) and returns {}. The empty dict makes callers fall
    through to "no active task / no tool name" — but the warning means a
    real upstream JSON bug is no longer silently masked as "no-op hook"."""
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            preview = raw[:240].replace("\n", "\\n")
            print(f"[AR] WARNING: hook stdin JSON parse failed ({e}); "
                  f"hook will no-op. Payload head: {preview!r}",
                  file=sys.stderr)
            return {}


def emit_status(msg: str) -> None:
    """stderr-only print. For incidental status that doesn't need to reach
    the model (e.g. "cleaned stale edit marker"). Anything Claude must act
    on goes through `emit_to_claude` instead."""
    print(msg, file=sys.stderr)


def emit_to_claude(*parts: str) -> None:
    """Emit a load-bearing message to Claude in ONE PostToolUse
    additionalContext JSON, plus stderr for transcript visibility.

    Why a single JSON: PostToolUse stdout is parsed for `hookSpecificOutput`
    JSON; some surfaces — third-party API proxies in particular — keep
    only the final JSON line of a hook run. Splitting a transition into
    two emits previously dropped half of it (we hit this twice: activation
    guidance over stderr only, then phase guidance clobbered by a later
    TodoWrite blob). This helper takes any number of parts, joins them
    with newlines, and writes one JSON. Empty / falsy parts are skipped
    so callers can pass conditionally-built strings without guarding.

    Caller convention: at most ONE call per hook run. Multiple calls
    silently produce multiple JSON lines and re-introduce the bug. The
    sanctioned way to add structured pieces (guidance + resume context +
    TodoWrite addendum + ...) is `phase_machine.emit_transition`."""
    msg = "\n".join(p for p in parts if p)
    if not msg:
        return
    print(msg, file=sys.stderr)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }))


def block(reason: str) -> None:
    """PreToolUse refusal. Writes the reason as `{decision: block}` JSON,
    exits with code 2 so Claude Code surfaces it to the model. Never
    returns."""
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)

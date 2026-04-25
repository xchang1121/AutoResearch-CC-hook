"""
Shared utilities for Claude Code hook scripts.
"""
import json
import re
import sys


def read_hook_input() -> dict:
    """Read and parse hook input from stdin.

    Handles Windows paths with unescaped backslashes in JSON
    (e.g., C:\\Users becomes C:\\\\Users before JSON parsing).
    """
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fix unescaped backslashes in Windows paths
        fixed = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}


def emit_status(msg: str):
    """Print a human-readable hook status line to stderr."""
    print(msg, file=sys.stderr)


def emit_additional_context(text: str):
    """Print a PostToolUse `hookSpecificOutput.additionalContext` payload so
    `text` is injected into Claude's next prompt.

    `emit_status` (stderr) is the convenient transcript channel, but stderr
    surfacing back to the model depends on the Claude Code surface and any
    upstream proxy. Custom API gateways (e.g. third-party model routers)
    can drop hook stderr silently — the agent then sees `1 PostToolUse hook
    ran` with no payload and has nothing to act on. additionalContext rides
    inside the tool-result envelope Claude Code constructs, so it survives
    proxies. Use this for any hook output the model MUST see — first and
    foremost the `[AR Phase: ...]` guidance line.
    """
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": text,
        }
    }))


def emit_phase_guidance(msg: str):
    """Surface a phase-guidance message on both channels: stderr (for the
    transcript) and additionalContext (load-bearing — what Claude actually
    reads). Use for every `[AR Phase: ...]` line and any message that
    instructs Claude what to run next."""
    emit_status(msg)
    emit_additional_context(msg)


def emit_todowrite_context(task_dir: str, header: str):
    """Print PostToolUse `hookSpecificOutput.additionalContext` JSON that
    instructs Claude to mirror plan.md into TodoWrite on the next turn.

    Only pending + in_progress items from the CURRENT plan are projected.
    Settled items (KEEP / DISCARD / FAIL) live in plan.md's Settled History
    table and history.jsonl — they are the durable audit trail, not part of
    the live TodoWrite queue. This caps the TodoWrite list at
    `items_per_plan` entries (typically 3-5) regardless of how many REPLAN
    cycles have happened.

    plan.md is the source of truth; TodoWrite is a UI mirror of current work.
    """
    from phase_machine import get_plan_items
    live = [it for it in get_plan_items(task_dir) if not it["done"]]
    if not live:
        return  # All items settled, replan pending — let next emit handle it.

    todos = []
    for it in live:
        status = "in_progress" if it["active"] else "pending"
        todos.append({
            "content": f"[{it['id']}] {it['description'][:80]}",
            "activeForm": f"Working on {it['id']}: {it['description'][:60]}",
            "status": status,
        })
    context = (
        f"{header}\n"
        f"Required action: call TodoWrite NOW with the exact list below. "
        f"This REPLACES any existing todos — do NOT merge, append, or "
        f"preserve older entries. plan.md is the source of truth; TodoWrite "
        f"is a UI mirror of current live work only (completed items live in "
        f"plan.md's Settled History). Pass this payload verbatim.\n"
        f"TodoWrite payload:\n{json.dumps({'todos': todos}, ensure_ascii=False)}"
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        }
    }))

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

    Emits even when no live items remain — an empty `{"todos": []}` payload
    explicitly clears the UI. Without that, the last item of each plan was
    stuck as in_progress in the model's TodoWrite UI: when it settled, this
    function previously short-circuited (live=[]), the model received no
    instruction, the next emit at create_plan time only listed the new
    plan's items, and the stale in_progress survived a non-strict REPLACE
    on the model side. Clearing here makes the transition unambiguous.
    """
    from phase_machine import get_plan_items
    live = [it for it in get_plan_items(task_dir) if not it["done"]]

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

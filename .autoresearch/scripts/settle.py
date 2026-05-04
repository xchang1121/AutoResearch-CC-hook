#!/usr/bin/env python3
"""
Mechanical plan.md settlement — no LLM needed.

After keep_or_discard.py runs, this script:
1. Reads the decision (KEEP/DISCARD/FAIL) from keep_or_discard output
2. Updates plan.md: mark active item [x] with result, advance (ACTIVE)
3. Returns the next phase

Usage:
    python settle.py <task_dir> <decision_json>

Output (stdout, last line):
    {"next_phase": "EDIT", "settled_item": "p1", "decision": "KEEP", "metric": 1294.8}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
# Single owner of plan.md regex parsing lives in validators.py;
# settle.py mutates the file but uses the canonical matcher for finding
# the lines to mutate. _PLAN_ITEM_RE captures (status, item_id, rest)
# where status is ' '/'x'.
from phase_machine import (compute_next_phase, plan_path, _PLAN_ITEM_RE,
                           is_settled_table_header)


def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "invalid arguments",
            "usage": "python settle.py <task_dir> <decision_json>",
            "received_args": sys.argv[1:],
        }))
        sys.exit(1)

    task_dir = sys.argv[1]
    decision_json = sys.argv[2]

    try:
        decision_data = json.loads(decision_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({
            "error": "invalid decision_json",
            "details": str(exc),
        }))
        sys.exit(1)
    decision = decision_data.get("decision", "FAIL")
    best_metric = decision_data.get("best_metric")
    # For KEEP, best_metric is this round's value. For DISCARD we have no metric.
    metric_val = best_metric if decision == "KEEP" else None

    ppath = plan_path(task_dir)
    if not os.path.exists(ppath):
        print(json.dumps({"error": "plan.md not found"}))
        sys.exit(1)

    with open(ppath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    settled_item_id = None
    settled_item_desc = ""
    active_line_idx = None

    if decision == "KEEP" and metric_val is not None:
        tag = f"[KEEP, metric={metric_val:.1f}]"
    elif decision == "DISCARD":
        tag = "[DISCARD]"
    else:
        tag = "[FAIL]"

    # Find the (ACTIVE) item using the canonical _PLAN_ITEM_RE so this
    # parser can never drift from validators.get_plan_items.
    for i, line in enumerate(lines):
        m = _PLAN_ITEM_RE.match(line)
        if m is None or m.group(1) != ' ' or "(ACTIVE)" not in line:
            continue
        active_line_idx = i
        settled_item_id = m.group(2)
        rest = m.group(3).replace("(ACTIVE)", "").strip().lstrip(": ").strip()
        # Keep the full description. Display-time truncation (dashboard,
        # hook status lines) happens at render time against actual
        # terminal width.
        settled_item_desc = rest
        # Rewrite from `[ ]` onwards; preserve the leading "  - " prefix.
        b = line.index('[ ]')
        lines[i] = line[:b] + f"[x] **{settled_item_id}** {tag}: {rest}"
        break

    if active_line_idx is None:
        print(json.dumps({"error": "no (ACTIVE) item found in plan.md"}))
        sys.exit(1)

    # Find next pending item and mark it (ACTIVE).
    for i, line in enumerate(lines):
        if i == active_line_idx:
            continue
        m = _PLAN_ITEM_RE.match(line)
        if m is None or m.group(1) != ' ' or "(ACTIVE)" in line:
            continue
        item_id = m.group(2)
        rest = m.group(3).lstrip(": ").strip()
        b = line.index('[ ]')
        lines[i] = line[:b] + f"[ ] **{item_id}** (ACTIVE): {rest}"
        break

    # Add to settled history table
    history_line = f"| {settled_item_id} | {decision} | {metric_val if metric_val is not None else 'N/A'} | {settled_item_desc} |"
    # Find the table and append
    table_end = None
    for i, line in enumerate(lines):
        if is_settled_table_header(line):
            # Found header, skip header + separator
            table_end = i + 2
            # Find last row
            for j in range(i + 2, len(lines)):
                if lines[j].strip().startswith("|"):
                    table_end = j + 1
                else:
                    break

    if table_end is not None:
        lines.insert(table_end, history_line)

    # Write back
    with open(ppath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Compute next phase
    next_phase = compute_next_phase(task_dir)

    output = {
        "settled_item": settled_item_id,
        "decision": decision,
        "metric": metric_val,
        "next_phase": next_phase,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()

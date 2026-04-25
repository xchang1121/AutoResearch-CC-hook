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
from phase_machine import (
    compute_next_phase,
    parse_plan_line,
    plan_path,
    render_plan_line,
)


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

    # Find the (ACTIVE) pending item via the canonical parser, drop any
    # render-time tag (e.g. [REACTIVATED]) and rewrite as settled.
    for i, line in enumerate(lines):
        parsed = parse_plan_line(line)
        if parsed is None or parsed["done"] or not parsed["active"]:
            continue
        active_line_idx = i
        settled_item_id = parsed["id"]
        # Keep the full description. Display-time truncation (dashboard,
        # hook status lines) happens at render time against actual
        # terminal width.
        settled_item_desc = parsed["description"]

        if decision == "KEEP" and metric_val is not None:
            tag = f"KEEP, metric={metric_val:.1f}"
        elif decision == "DISCARD":
            tag = "DISCARD"
        else:
            tag = "FAIL"

        lines[i] = render_plan_line(
            settled_item_id,
            description=settled_item_desc,
            done=True,
            active=False,
            tag=tag,
            indent=parsed["indent"],
        )
        break

    if active_line_idx is None:
        print(json.dumps({"error": "no (ACTIVE) item found in plan.md"}))
        sys.exit(1)

    # Promote next pending item to ACTIVE.
    for i, line in enumerate(lines):
        if i == active_line_idx:
            continue
        parsed = parse_plan_line(line)
        if parsed is None or parsed["done"] or parsed["active"]:
            continue
        lines[i] = render_plan_line(
            parsed["id"],
            description=parsed["description"],
            done=False,
            active=True,
            tag=parsed["tag"],
            indent=parsed["indent"],
        )
        break

    # Add to settled history table
    history_line = f"| {settled_item_id} | {decision} | {metric_val if metric_val is not None else 'N/A'} | {settled_item_desc} |"
    # Find the table and append
    table_end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "Item" in line and "Outcome" in line:
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

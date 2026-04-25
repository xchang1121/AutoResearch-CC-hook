#!/usr/bin/env python3
"""
Create or replace plan.md from structured XML input.

Claude provides content, this script handles format. XML is preferred over JSON
because LLMs hallucinate fewer structural/escape errors in tag-delimited text.

Usage:
    python .autoresearch/scripts/create_plan.py <task_dir> '<items_xml>'

items_xml format:
    <items>
      <item>
        <desc>short sentence describing the change</desc>
        <rationale>30-400 char explanation of why it should help</rationale>
        <keywords>comma-separated tags</keywords>
      </item>
      ...
    </items>

Optional per-item element: <reactivate_pid>pN</reactivate_pid>
    Reuse a previously-settled pid (must be DISCARD or FAIL in history). The
    pid keeps its original id — the monotonic counter is NOT consumed for it.
    Used when a previously-explored idea may combine differently with current
    state (e.g. an autotune sweep that was DISCARD before a fusion landed).

If <items_xml> begins with '@', the remainder is treated as a path and the
XML is read from that file. If <items_xml> is exactly '-', XML is read from
stdin. Prefer these over inline argv — on Windows, multi-line XML passed
through bash argv can be silently truncated by the shell / CreateProcess,
producing misleading "missing <desc>" style errors that look like schema
bugs but are actually IPC truncation.

Output: writes plan.md, prints JSON status.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from phase_machine import (
    load_progress, save_progress, get_plan_items, append_history,
    plan_path, progress_path, load_history, render_plan_line,
)


# Words that indicate parameter tuning (matched at word level)
_PARAM_WORDS = {
    "block", "tile", "tiling", "autotune", "config", "configs",
    "warps", "stages", "size", "tune", "adjust", "sweep",
    "parameter", "param", "group", "num",
}
_PARAM_PHRASES = {
    "block_size", "block_m", "block_n", "block_k", "block_size_m",
    "block_size_n", "block_size_k", "num_warps", "num_stages",
    "group_size", "group_size_m",
}
_STOPWORDS = {"the", "a", "to", "of", "in", "for", "and", "with", "from", "by",
              "on", "is", "it", "as", "at", "or", "an", "be", "was", "that"}

_PID_RE = re.compile(r"^p\d+$")

# Tracks where the XML payload came from so error messages can steer the
# caller toward a robust input channel when argv looks suspicious. Set in
# main() before any _fail() call that depends on parsed content.
_SOURCE_MODE = "argv"  # one of: "argv", "file", "stdin"


def _fail(msg: str):
    hint = ""
    if _SOURCE_MODE == "argv":
        # If this trips, the model almost certainly got here by inline-quoting
        # multi-line XML — which is exactly the Windows-argv failure mode. Say
        # so loudly so retries don't loop on "fix the schema".
        hint = (" [hint: payload was passed inline via argv. On Windows this "
                "is often truncated by the shell, producing errors that look "
                "like schema bugs. Write the XML to a file and pass "
                "'@<path>', or pipe it via stdin with '-' as the 2nd arg.]")
    print(json.dumps({"ok": False, "error": msg + hint}))
    sys.exit(1)


_ALLOWED_ITEM_TAGS = {"desc", "rationale", "keywords", "reactivate_pid"}


def _parse_items_xml(xml_str: str) -> list:
    """Parse <items><item>...</item>...</items> into a list of dicts.

    Recognized child elements under <item>: desc, rationale, keywords,
    reactivate_pid. Unknown tags are rejected so typos surface loudly
    rather than silently dropping fields.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        _fail(f"Invalid XML: {e}")
    if root.tag != "items":
        _fail(f"Root element must be <items>, got <{root.tag}>")
    items = []
    for i, child in enumerate(list(root)):
        if child.tag != "item":
            _fail(f"Unexpected <{child.tag}> under <items> (only <item> allowed)")
        d = {}
        for sub in list(child):
            if sub.tag not in _ALLOWED_ITEM_TAGS:
                _fail(f"Item {i}: unknown element <{sub.tag}> "
                      f"(allowed: {sorted(_ALLOWED_ITEM_TAGS)})")
            if sub.tag in d:
                _fail(f"Item {i}: duplicate <{sub.tag}>")
            d[sub.tag] = (sub.text or "").strip()
        items.append(d)
    return items


def _validate_items(items):
    if not isinstance(items, list) or len(items) < 3:
        _fail(f"Need >= 3 items, got {len(items) if isinstance(items, list) else 'non-list'}")
    for i, item in enumerate(items):
        for field in ("desc", "rationale", "keywords"):
            if field not in item:
                _fail(f"Item {i}: missing <{field}>")
        kw = item["keywords"].strip()
        if not kw:
            _fail(f"Item {i}: <keywords> is empty")
        item["keywords"] = kw

        for field in ("desc", "rationale"):
            if not isinstance(item[field], str) or not item[field].strip():
                _fail(f"Item {i}: '{field}' must be a non-empty string")

        # desc must be a short prose sentence, not a snake_case identifier —
        # the history table and plan table in the dashboard surface this
        # field directly, and "fuse_swiglu_epilogue" is unreadable next to
        # "Fuse the SwiGLU epilogue into the matmul kernel".
        desc = item["desc"].strip()
        item["desc"] = desc
        if len(desc) < 12:
            _fail(f"Item {i}: desc too short ({len(desc)} chars, need >= 12 — "
                  f"write a short sentence, not a label)")
        if " " not in desc:
            _fail(f"Item {i}: desc looks like an identifier ({desc!r}) — "
                  f"write a short sentence describing the change instead "
                  f"(e.g. 'Fuse SwiGLU into the matmul epilogue')")

        rat = item["rationale"].strip()
        if len(rat) < 30:
            _fail(f"Item {i}: rationale too short ({len(rat)} chars, need >= 30)")
        if len(rat) > 400:
            item["rationale"] = rat[:397] + "..."
        rp = item.get("reactivate_pid")
        if rp is not None:
            if not isinstance(rp, str) or not _PID_RE.match(rp):
                _fail(f"Item {i}: reactivate_pid must be of form 'pN' (got {rp!r})")


def _check_diversity(items):
    """Reject plans where all but one item are pure parameter tuning."""
    keyword_sets, keyword_raw = [], []
    for item in items:
        flat, raw = set(), set()
        for k in item["keywords"].split(","):
            phrase = k.strip().lower().replace("-", "_").replace(" ", "_")
            raw.add(phrase)
            for w in phrase.split("_"):
                if w:
                    flat.add(w)
        keyword_sets.append(flat)
        keyword_raw.append(raw)

    param_only = 0
    for words, phrases in zip(keyword_sets, keyword_raw):
        has_param_phrase = bool(phrases & _PARAM_PHRASES)
        non_param = words - _PARAM_WORDS - {""}
        if (has_param_phrase or not non_param) and words:
            param_only += 1

    if param_only >= len(items) - 1:
        detected = _PARAM_WORDS & set().union(*keyword_sets)
        _fail(
            f"Diversity rejected: {param_only}/{len(items)} items are parameter tuning. "
            f"Bundle parameter sweeps into ONE item. Other items must be structurally "
            f"different (algorithmic changes, fusion, memory access patterns, data layout). "
            f"Param-only keywords detected: {detected}"
        )


def _warn_repeated_failures(task_dir: str):
    """Warn on stderr if recent history shows repeated failure keywords."""
    failed = [
        (rec.get("description") or "").lower()
        for rec in load_history(task_dir, on_corrupt="skip")
        if rec.get("decision") in ("DISCARD", "FAIL")
    ]
    if len(failed) < 3:
        return
    tokens = Counter()
    for desc in failed:
        for w in desc.replace("-", " ").replace("_", " ").split():
            w = w.strip(".,;:()[]{}\"'")
            if len(w) > 2 and w not in _STOPWORDS:
                tokens[w] += 1
    repeated = sorted(tok for tok, cnt in tokens.items() if cnt >= 2)
    if repeated:
        print(f"[WARN] Previously failed keywords (avoid repeating): {repeated[:10]}",
              file=sys.stderr)


def _load_history(task_dir: str) -> list:
    """Thin wrapper around ``phase_machine.load_history`` for the planner —
    silent drop policy preserved (planner only needs a best-effort summary).
    """
    return load_history(task_dir, on_corrupt="skip")


def _validate_reactivations(items: list, task_dir: str, old_pending_ids: set):
    """Check each item with reactivate_pid is sound.

    Returns list of (item_index, pid, last_decision, last_round) for reactivations.

    Rules:
      - pid must appear in history.jsonl with last decision in {DISCARD, FAIL}
        (KEEP items are already applied; reactivating them is nonsense)
      - pid must NOT currently be pending in the old plan (use the one that's
        already there, don't duplicate)
      - pid must not appear twice in this batch of reactivations
    """
    history = _load_history(task_dir)
    # Index: pid → list of records chronological
    by_pid = {}
    for rec in history:
        pid = rec.get("plan_item")
        if pid:
            by_pid.setdefault(pid, []).append(rec)

    seen = set()
    results = []
    for i, item in enumerate(items):
        rp = item.get("reactivate_pid")
        if rp is None:
            continue
        if rp in seen:
            _fail(f"Item {i}: reactivate_pid={rp} duplicated in this plan")
        seen.add(rp)
        if rp in old_pending_ids:
            _fail(f"Item {i}: reactivate_pid={rp} is still pending in current "
                  f"plan — reactivation only applies to settled pids")
        if rp not in by_pid:
            _fail(f"Item {i}: reactivate_pid={rp} has no history record — "
                  f"cannot reactivate a pid that was never settled")
        last = by_pid[rp][-1]
        last_decision = last.get("decision")
        if last_decision not in ("DISCARD", "FAIL"):
            _fail(f"Item {i}: reactivate_pid={rp} last decision was "
                  f"{last_decision!r} — only DISCARD / FAIL may be reactivated")
        results.append((i, rp, last_decision, last.get("round")))
    return results


def _parse_old_plan(task_dir: str):
    """Return (settled_rows_str, pending_items) from the existing plan.md.

    Pending items are produced by the canonical parser; settled-history rows
    are carried forward verbatim so history layout stays stable.
    """
    pending = [
        {"id": it["id"], "description": it["description"]}
        for it in get_plan_items(task_dir)
        if not it["done"]
    ]

    settled_rows = ""
    ppath = plan_path(task_dir)
    if os.path.exists(ppath):
        with open(ppath, "r", encoding="utf-8") as f:
            content = f.read()
        in_table = False
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("|") and "Item" in stripped and "Outcome" in stripped:
                in_table = True
                continue
            if in_table and stripped.startswith("|---"):
                continue
            if in_table and stripped.startswith("|"):
                settled_rows += line + "\n"
            elif in_table:
                in_table = False
    return settled_rows, pending


def _supersede_pending(task_dir: str, pending: list, new_version: int,
                       reactivated_pids: set) -> str:
    """Force-settle abandoned pending items as DISCARD (reason=superseded).

    Pids being explicitly reactivated are skipped — they'll reappear as new
    pending items in the new plan, not as abandoned superseded entries.

    Appends to history.jsonl and returns extra Settled History rows.
    """
    victims = [it for it in pending if it["id"] not in reactivated_pids]
    if not victims:
        return ""
    reason = f"superseded by replan v{new_version}"
    ts = datetime.now(timezone.utc).isoformat()
    extra_rows = ""
    for it in victims:
        append_history(task_dir, {
            "round": "-",
            "description": it["description"],
            "plan_item": it["id"],
            "decision": "DISCARD",
            "metrics": {},
            "correctness": None,
            "error": None,
            "commit": None,
            "reason": reason,
            "timestamp": ts,
        })
        extra_rows += f"| {it['id']} | DISCARD | N/A | {it['description']} ({reason}) |\n"
    return extra_rows


def _record_reactivations(task_dir: str, reactivations: list, new_version: int,
                          items: list) -> str:
    """Write REACTIVATE markers to history.jsonl + Settled History rows.

    Each reactivation becomes one history record with decision='REACTIVATE'
    so the audit trail shows exactly when a pid was revived and why. The
    subsequent round (when the reactivated pid is settled again) writes a
    normal KEEP/DISCARD/FAIL row — so one pid can have multiple outcomes
    across its lifetime.
    """
    if not reactivations:
        return ""
    ts = datetime.now(timezone.utc).isoformat()
    extra_rows = ""
    for idx, pid, last_decision, last_round in reactivations:
        item = items[idx]
        reason = (f"reactivated in plan v{new_version} "
                  f"(last outcome: {last_decision} in round {last_round})")
        append_history(task_dir, {
            "round": "-",
            "description": item["desc"],
            "plan_item": pid,
            "decision": "REACTIVATE",
            "metrics": {},
            "correctness": None,
            "error": None,
            "commit": None,
            "reason": reason,
            "timestamp": ts,
        })
        extra_rows += f"| {pid} | REACTIVATE | — | {item['desc']} (v{new_version}) |\n"
    return extra_rows


def _compute_next_pid(progress: dict, ppath: str) -> int:
    """Monotonic pid allocator. Falls back to scanning plan.md when the counter
    is missing (old tasks that predate the field)."""
    n = progress.get("next_pid")
    if n is not None:
        return n
    n = 1
    if os.path.exists(ppath):
        with open(ppath, "r", encoding="utf-8") as f:
            for m in re.finditer(r'\*\*p(\d+)\*\*', f.read()):
                n = max(n, int(m.group(1)) + 1)
    return n


def _allocate_ids(items: list, next_pid: int) -> tuple:
    """Assign pids in order. Reactivated items keep their existing pid;
    others consume from the monotonic counter.

    Returns (item_ids, new_next_pid).
    """
    out = []
    cursor = next_pid
    for item in items:
        rp = item.get("reactivate_pid")
        if rp:
            out.append(rp)
        else:
            out.append(f"p{cursor}")
            cursor += 1
    return out, cursor


def _render_plan(version: int, item_ids: list, items: list, settled_rows: str) -> str:
    lines = [f"# Plan v{version}", "", "## Active Items"]
    for i, (item, pid) in enumerate(zip(items, item_ids)):
        # The first item is ACTIVE; reactivated items keep [REACTIVATED]
        # tag so downstream readers can distinguish a fresh attempt at an
        # old idea from a brand-new one.
        lines.append(render_plan_line(
            pid,
            description=item["desc"].strip(),
            done=False,
            active=(i == 0),
            tag=("REACTIVATED" if item.get("reactivate_pid") else ""),
        ))
        lines.append(f"  - rationale: {item['rationale'].strip()}")
        lines.append(f"  - keywords: {item['keywords'].strip()}")
    lines.append("")
    lines.append("## Settled History")
    lines.append("| Item | Outcome | Metric | Reason |")
    lines.append("|------|---------|--------|--------|")
    if settled_rows:
        lines.append(settled_rows.rstrip())
    return "\n".join(lines) + "\n"


def main():
    global _SOURCE_MODE
    task_dir = sys.argv[1]
    arg = sys.argv[2]

    if arg == "-":
        _SOURCE_MODE = "stdin"
        xml_str = sys.stdin.read()
    elif arg.startswith("@"):
        _SOURCE_MODE = "file"
        path = arg[1:]
        try:
            with open(path, "r", encoding="utf-8") as f:
                xml_str = f.read()
        except OSError as e:
            _fail(f"Cannot read XML from {path!r}: {e}")
    else:
        _SOURCE_MODE = "argv"
        xml_str = arg

    items = _parse_items_xml(xml_str)
    _validate_items(items)
    _check_diversity(items)
    _warn_repeated_failures(task_dir)

    progress = load_progress(task_dir) or {}
    version = progress.get("plan_version", 0) + 1
    ppath = plan_path(task_dir)
    next_pid = _compute_next_pid(progress, ppath)

    settled_rows, old_pending = _parse_old_plan(task_dir)
    old_pending_ids = {it["id"] for it in old_pending}

    # Validate reactivations against history + current plan state
    reactivations = _validate_reactivations(items, task_dir, old_pending_ids)
    reactivated_pids = {r[1] for r in reactivations}

    # Abandon unclaimed old pendings; reactivated pids skip this path.
    extra_rows = _supersede_pending(task_dir, old_pending, version, reactivated_pids)
    if extra_rows:
        settled_rows = (settled_rows.rstrip() + "\n" + extra_rows) if settled_rows else extra_rows

    # Record REACTIVATE markers so the audit trail captures the revival
    reactivate_rows = _record_reactivations(task_dir, reactivations, version, items)
    if reactivate_rows:
        settled_rows = (settled_rows.rstrip() + "\n" + reactivate_rows) if settled_rows else reactivate_rows

    item_ids, new_next_pid = _allocate_ids(items, next_pid)

    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    with open(ppath, "w", encoding="utf-8") as f:
        f.write(_render_plan(version, item_ids, items, settled_rows))

    progress["next_pid"] = new_next_pid
    progress["plan_version"] = version
    if os.path.exists(progress_path(task_dir)):
        save_progress(task_dir, progress, stamp=False)

    print(json.dumps({
        "ok": True,
        "version": version,
        "items": item_ids,
        "active": item_ids[0],
        "superseded": [it["id"] for it in old_pending if it["id"] not in reactivated_pids],
        "reactivated": sorted(reactivated_pids),
        "path": ppath,
    }))


if __name__ == "__main__":
    main()

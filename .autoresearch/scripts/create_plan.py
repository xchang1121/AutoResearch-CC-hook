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
      </item>
      ...
    </items>

Behavior:
  Every successful run REPLACES plan.md's `## Active Items` with the new
  XML items. Any pending pid from the previous plan that hadn't run yet is
  silently dropped (no fake DISCARD record, no Settled History row). pids
  remain monotonic — `next_pid` keeps advancing, dropped pids are not
  reused — so the audit chain via plan_version + history.jsonl is still
  unambiguous: a pid that exists only in plan_version N's plan.md and has
  no history.jsonl entry was abandoned at the N → N+1 transition.

  In practice this only affects DIAGNOSE (which fires mid-plan after 3
  consecutive failures); REPLAN by construction only fires when every
  item has already settled, so old_pending is empty there.

  If a past DISCARD/FAIL idea looks promising again, just re-propose it
  as a new item with a fresh pid. The desc text carries the audit story;
  pid reuse adds no information.

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

sys.path.insert(0, os.path.dirname(__file__))
from phase_machine import (
    load_progress, save_progress, get_plan_items,
    plan_path, progress_path,
)


# Words that indicate parameter tuning, used by `_check_diversity` to flag
# plans where every item is a parameter sweep.
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


_ALLOWED_ITEM_TAGS = {"desc", "rationale"}


def _parse_items_xml(xml_str: str) -> list:
    """Parse <items><item>...</item>...</items> into a list of dicts.

    Recognized child elements under <item>: desc, rationale. Unknown tags
    are rejected so typos surface loudly rather than silently dropping
    fields. (An earlier schema had a `<keywords>` field that the model
    routinely omitted; it was removed because every keyword token already
    appears in <desc>, and `_check_diversity` now reads desc directly.)
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        _fail(f"Invalid XML: {e}")
    if root.tag != "items":
        _fail(f"Root element must be <items>, got <{root.tag}>")
    if root.attrib:
        _fail(f"<items> must have no attributes, got {sorted(root.attrib)}")
    items = []
    for i, child in enumerate(list(root)):
        if child.tag != "item":
            _fail(f"Unexpected <{child.tag}> under <items> (only <item> allowed)")
        # <item> takes no attributes — pids are auto-assigned, and the
        # XML example's inline comments say so explicitly. Reject anything
        # the model invents (id="p1", pid="p1", priority="high", ...) so
        # the lesson lands instead of slipping through silently.
        if child.attrib:
            _fail(f"Item {i}: <item> must have no attributes, got "
                  f"{sorted(child.attrib)} — pids are auto-assigned, do "
                  f"not supply them")
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
        for field in ("desc", "rationale"):
            if field not in item:
                _fail(f"Item {i}: missing <{field}>")
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


def _check_diversity(items):
    """Reject plans where all but one item are pure parameter tuning.

    Tokenizes <desc> directly. The earlier schema carried a separate
    <keywords> field for this signal; it was removed because every
    keyword token already appears in desc verbatim (you cannot describe
    a parameter sweep without using "block"/"tile"/"size"/etc.) and
    forcing the model to write keywords twice was pure friction.

    Detection rule per item: classify as parameter-only if its desc
    contains a known parameter phrase (block_size, num_warps, ...) OR
    if the only content tokens it has come from `_PARAM_WORDS`.
    """
    word_sets = []   # per-item: content tokens after stopword filter
    raw_descs = []   # per-item: lower+normalized for phrase matching
    for item in items:
        raw = item["desc"].lower().replace("-", "_")
        raw_descs.append(raw)
        words = set()
        for tok in raw.replace("_", " ").split():
            tok = tok.strip(".,;:()[]{}\"'")
            if tok and tok not in _STOPWORDS:
                words.add(tok)
        word_sets.append(words)

    param_only = 0
    for words, raw in zip(word_sets, raw_descs):
        has_param_phrase = any(p in raw for p in _PARAM_PHRASES)
        non_param = words - _PARAM_WORDS - {""}
        if (has_param_phrase or not non_param) and words:
            param_only += 1

    if param_only >= len(items) - 1:
        detected = _PARAM_WORDS & set().union(*word_sets) if word_sets else set()
        _fail(
            f"Diversity rejected: {param_only}/{len(items)} items are parameter tuning. "
            f"Bundle parameter sweeps into ONE item. Other items must be structurally "
            f"different (algorithmic changes, fusion, memory access patterns, data layout). "
            f"Param-only words detected in desc: {sorted(detected)}"
        )


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


def _allocate_ids(n_items: int, next_pid: int) -> tuple:
    """Assign `n_items` fresh pids from the monotonic counter.

    Returns (item_ids, new_next_pid). Pids are never reused — dropped pids
    from the previous plan stay dropped; they don't free their slot.
    """
    ids = [f"p{next_pid + i}" for i in range(n_items)]
    return ids, next_pid + n_items


def _render_plan(version: int, item_ids: list, items: list, settled_rows: str) -> str:
    lines = [f"# Plan v{version}", "", "## Active Items"]
    for i, (item, pid) in enumerate(zip(items, item_ids)):
        marker = " (ACTIVE)" if i == 0 else ""
        lines.append(f"- [ ] **{pid}**{marker}: {item['desc'].strip()}")
        lines.append(f"  - rationale: {item['rationale'].strip()}")
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

    progress = load_progress(task_dir) or {}
    version = progress.get("plan_version", 0) + 1
    ppath = plan_path(task_dir)
    next_pid = _compute_next_pid(progress, ppath)

    # Carry the existing Settled History table forward verbatim. Old
    # pending items (still-unrun pids from the previous plan) are silently
    # dropped: no fake DISCARD row, no history.jsonl entry. Their pids
    # remain consumed (next_pid does not regress) so the audit chain is
    # plan_version + history.jsonl absence — see module docstring.
    settled_rows, old_pending = _parse_old_plan(task_dir)
    dropped_pids = [it["id"] for it in old_pending]

    item_ids, new_next_pid = _allocate_ids(len(items), next_pid)

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
        "dropped": dropped_pids,
        "path": ppath,
    }))


if __name__ == "__main__":
    main()

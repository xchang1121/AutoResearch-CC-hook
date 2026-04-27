#!/usr/bin/env python3
"""
Append plan items to plan.md from structured XML input.

Behavior:
- PLAN: writes the initial plan (no existing items).
- REPLAN: appends new items to the end of the existing plan. Existing items
  keep their state (KEEP / DISCARD / FAIL / pending). pid is a monotonic
  counter — every new item gets a fresh pN; settled items stay as
  historical record. ACTIVE is the first pending item by pid order.
- DIAGNOSE: same as REPLAN, except *all currently pending items are first
  marked ABANDONED* (done=True, tag="ABANDONED, reason=diagnose_trigger").
  The 3-consecutive-FAIL trigger means the prior plan's assumption chain is
  broken; running its leftover items would just burn rounds on doomed
  branches. They survive in plan.md as audit history, but the work queue
  starts fresh from the new items.

XML schema:
  Single source of truth lives in `phase_machine._PLAN_FIELD_RULES` and the
  example in `phase_machine._PLAN_XML_EXAMPLE`. The [AR Phase: PLAN /
  DIAGNOSE / REPLAN] hook guidance you receive at runtime contains both,
  plus phase-specific framing — read that, do not duplicate the schema in
  this docstring.

Usage:
    python .autoresearch/scripts/create_plan.py <task_dir> <xml_source>

`<xml_source>` is `@<path>` (read XML from file — recommended), `-` (read
from stdin), or inline XML (avoid on Windows: bash / CreateProcess can
silently truncate multi-line argv, producing misleading "missing <desc>"
errors that look like schema bugs but are IPC truncation).

Output: writes plan.md, prints JSON status.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from phase_machine import (
    DIAGNOSE,
    load_progress, save_progress, get_plan_items,
    plan_path, progress_path, load_history, read_phase, render_plan_line,
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


_ALLOWED_ITEM_TAGS = {"desc", "rationale", "keywords"}


def _parse_items_xml(xml_str: str) -> list:
    """Parse <items><item>...</item>...</items> into a list of dicts.

    Recognized child elements under <item>: desc, rationale, keywords.
    Unknown tags are rejected so typos surface loudly rather than silently
    dropping fields.
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


def _read_settled_history_rows(task_dir: str) -> str:
    """Read the existing `## Settled History` table rows verbatim from plan.md.

    settle.py appends rows to this table when items KEEP/DISCARD/FAIL; we
    carry them through unchanged so a re-render preserves layout. Returns
    "" if no plan.md or no table yet.
    """
    ppath = plan_path(task_dir)
    if not os.path.exists(ppath):
        return ""
    with open(ppath, "r", encoding="utf-8") as f:
        content = f.read()
    rows = ""
    in_table = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and "Item" in stripped and "Outcome" in stripped:
            in_table = True
            continue
        if in_table and stripped.startswith("|---"):
            continue
        if in_table and stripped.startswith("|"):
            rows += line + "\n"
        elif in_table:
            in_table = False
    return rows


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


def _allocate_ids(n_new: int, next_pid: int) -> tuple:
    """Allocate `n_new` fresh pids starting at `next_pid`. Returns
    (item_ids, new_next_pid). Pids are monotonic and never reused."""
    ids = [f"p{next_pid + i}" for i in range(n_new)]
    return ids, next_pid + n_new


def _pid_num(pid: str) -> int:
    """Numeric suffix for sorting (`p9` < `p10`)."""
    try:
        return int(pid[1:])
    except ValueError:
        return 0


def _render_plan(old_items: list, new_pids: list, new_items: list,
                 settled_rows: str) -> str:
    """Render the merged plan with INSERT semantics for new items.

    File layout (matters because settle.py promotes the next pending in
    file order, not pid order):
      [old settled items, in original order]
      [new items, in pid order — INSERTED ahead of old pending]
      [old pending items, in pid order — pushed behind new]

    ACTIVE goes on the first new item, so the diagnosis insight runs
    IMMEDIATELY. After it settles, settle.py's file-order promotion picks
    the next new item, then the next, then finally rotates back to the old
    pending items that were queued behind. pid stays monotonic (new items
    get higher pids); only the queue position is reordered.

    Edge cases: PLAN (no old items at all) and REPLAN (all old items
    settled) both degenerate to "settled then new" with no displaced
    pending — same code path, no special branch.
    """
    settled_old = [it for it in old_items if it["done"]]
    pending_old = [it for it in old_items if not it["done"]]
    pending_old.sort(key=lambda it: _pid_num(it["id"]))

    if new_pids:
        active_pid = new_pids[0]
    elif pending_old:
        active_pid = pending_old[0]["id"]
    else:
        active_pid = None

    def _emit_old(it):
        lines.append(render_plan_line(
            it["id"],
            description=it["description"],
            done=it["done"],
            active=(it["id"] == active_pid),
            tag=it["tag"],
        ))
        if it.get("rationale"):
            lines.append(f"  - rationale: {it['rationale']}")
        if it.get("keywords"):
            lines.append(f"  - keywords: {it['keywords']}")

    lines = ["# Plan", "", "## Items"]
    for it in settled_old:
        _emit_old(it)
    for pid, item in zip(new_pids, new_items):
        lines.append(render_plan_line(
            pid,
            description=item["desc"].strip(),
            done=False,
            active=(pid == active_pid),
            tag="",
        ))
        lines.append(f"  - rationale: {item['rationale'].strip()}")
        lines.append(f"  - keywords: {item['keywords'].strip()}")
    for it in pending_old:
        _emit_old(it)
    lines.append("")
    lines.append("## Settled History")
    lines.append("| Item | Outcome | Metric | Reason |")
    lines.append("|------|---------|--------|--------|")
    if settled_rows:
        lines.append(settled_rows.rstrip())
    return "\n".join(lines) + "\n"


_USAGE = """\
usage: create_plan.py <task_dir> @<path>

The single supported flow:
  1. Write the <items> XML to "$AR_TASK_DIR/.ar_state/plan_items.xml"
     with the Write tool.
  2. Run: python .autoresearch/scripts/create_plan.py "$AR_TASK_DIR" \\
            @"$AR_TASK_DIR/.ar_state/plan_items.xml"

The XML schema is in the [AR Phase: PLAN/DIAGNOSE/REPLAN] hook guidance
message — read that for the exact <item>/<desc>/<rationale>/<keywords>
shape; this script does not duplicate it.

Legacy input modes still parsed but discouraged:
  '<items>…</items>'   inline argv — silently truncated by Windows shells
  -                     read XML from stdin
"""


def main():
    global _SOURCE_MODE
    if len(sys.argv) < 3:
        sys.stderr.write(_USAGE)
        sys.exit(2)
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
    ppath = plan_path(task_dir)
    next_pid = _compute_next_pid(progress, ppath)

    # Append-only: old items (settled + still-pending) survive untouched in
    # PLAN / REPLAN. In DIAGNOSE, the still-pending tail is converted to
    # ABANDONED first — see module docstring for the rationale. New items
    # always get fresh pids appended at the end. ACTIVE is recomputed in
    # render: first pending pid across the merged set (natural FIFO).
    old_items = get_plan_items(task_dir, include_meta=True)
    settled_rows = _read_settled_history_rows(task_dir)

    if read_phase(task_dir) == DIAGNOSE:
        for it in old_items:
            if not it["done"]:
                it["done"] = True
                it["tag"] = "ABANDONED, reason=diagnose_trigger"
                # Mirror the abandonment in the audit table so plan.md and
                # the settled-history view stay consistent.
                settled_rows += (
                    f"| {it['id']} | ABANDONED | - | diagnose_trigger |\n"
                )

    new_pids, new_next_pid = _allocate_ids(len(items), next_pid)

    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    with open(ppath, "w", encoding="utf-8") as f:
        f.write(_render_plan(old_items, new_pids, items, settled_rows))

    progress["next_pid"] = new_next_pid
    progress["plan_version"] = progress.get("plan_version", 0) + 1
    if os.path.exists(progress_path(task_dir)):
        save_progress(task_dir, progress, stamp=False)

    # Active pid mirrors _render_plan's selection: prefer first new pid so
    # diagnosis-informed items run immediately, not behind stale pending.
    if new_pids:
        active = new_pids[0]
    else:
        old_pending = sorted(
            (it["id"] for it in old_items if not it["done"]),
            key=_pid_num,
        )
        active = old_pending[0] if old_pending else None

    print(json.dumps({
        "ok": True,
        "items": new_pids,
        "active": active,
        "path": ppath,
    }))


if __name__ == "__main__":
    main()

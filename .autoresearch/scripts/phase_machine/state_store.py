"""State storage layer.

Single owner of `<task_dir>/.ar_state/` and `.autoresearch/.active_task`.
No other module reads/writes these files directly — go through the helpers
here.

What lives in this module:
  - Phase enum constants (used as keys / values throughout).
  - Canonical file basenames inside `.ar_state/` (PHASE_FILE, etc.).
  - Path builders (`state_path`, `plan_path`, `progress_path`, …).
  - Phase I/O (`read_phase`, `write_phase`).
  - Progress I/O (`load_progress`, `save_progress`, `update_progress`).
  - History append (`append_history`).
  - Active-task pointer (`get_task_dir`, `set_task_dir`).
  - Heartbeat touch.
  - JSON-tail parser used by every subprocess output.

Why phase constants live here and not in phase_policy: `read_phase` needs
`ALL_PHASES` to validate; phase_policy in turn needs `compute_next_phase`
to read progress, which lives here. Putting the constants at the bottom
of the dependency stack avoids the cycle.
"""
import json
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

INIT = "INIT"
GENERATE_REF = "GENERATE_REF"
GENERATE_KERNEL = "GENERATE_KERNEL"
BASELINE = "BASELINE"
PLAN = "PLAN"
EDIT = "EDIT"
DIAGNOSE = "DIAGNOSE"
REPLAN = "REPLAN"
FINISH = "FINISH"

ALL_PHASES = {INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
              DIAGNOSE, REPLAN, FINISH}


# ---------------------------------------------------------------------------
# Canonical filenames inside <task_dir>/.ar_state/
# ---------------------------------------------------------------------------

PHASE_FILE = ".phase"
PROGRESS_FILE = "progress.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
PLAN_ITEMS_FILE = "plan_items.xml"  # canonical XML payload path under .ar_state/
EDIT_MARKER_FILE = ".edit_started"
HEARTBEAT_FILE = ".heartbeat"
ACTIVE_TASK_FILE = ".active_task"  # under .autoresearch/, not .ar_state/


# ---------------------------------------------------------------------------
# Project root resolution + active-task pointer
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """Walk up from this file to find the dir that has `.autoresearch/`."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, ".autoresearch")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


_PROJECT_ROOT = _find_project_root()
_ACTIVE_TASK_FILE = os.path.join(_PROJECT_ROOT, ".autoresearch", ACTIVE_TASK_FILE)


def get_task_dir() -> str:
    """Get active task_dir. Reads from .autoresearch/.active_task file.

    Falls back to AR_TASK_DIR env var for backward compat.
    Returns "" if no active task.
    """
    if os.path.exists(_ACTIVE_TASK_FILE):
        with open(_ACTIVE_TASK_FILE, "r") as f:
            td = f.read().strip()
        if td and os.path.isdir(td):
            return td
    return os.environ.get("AR_TASK_DIR", "")


def set_task_dir(task_dir: str):
    """Write active task_dir to .autoresearch/.active_task."""
    os.makedirs(os.path.dirname(_ACTIVE_TASK_FILE), exist_ok=True)
    with open(_ACTIVE_TASK_FILE, "w") as f:
        f.write(os.path.abspath(task_dir))
    touch_heartbeat(task_dir)


def touch_heartbeat(task_dir: str):
    """Update .ar_state/.heartbeat file to signal this task is active.

    Called from every hook invocation. resume.py checks mtime to detect
    conflicting concurrent Claude Code sessions. A failed touch is reported
    to stderr — silently swallowing it would make the session look dead in
    a way that's nearly impossible to debug.
    """
    try:
        heartbeat = state_path(task_dir, HEARTBEAT_FILE)
        os.makedirs(os.path.dirname(heartbeat), exist_ok=True)
        import time
        with open(heartbeat, "w") as f:
            f.write(f"{int(time.time())}\n")
    except Exception as e:
        print(f"[AR] WARNING: heartbeat write failed ({e}); resume.py may "
              f"misreport this task as inactive.", file=sys.stderr)


# ---------------------------------------------------------------------------
# State file path builders
# ---------------------------------------------------------------------------

def state_path(task_dir: str, name: str) -> str:
    """Path to a file under <task_dir>/.ar_state/. Centralized so no module
    hand-builds state paths."""
    return os.path.join(task_dir, ".ar_state", name)


def plan_path(task_dir: str) -> str:
    return state_path(task_dir, PLAN_FILE)


def progress_path(task_dir: str) -> str:
    return state_path(task_dir, PROGRESS_FILE)


def history_path(task_dir: str) -> str:
    return state_path(task_dir, HISTORY_FILE)


def edit_marker_path(task_dir: str) -> str:
    return state_path(task_dir, EDIT_MARKER_FILE)


# ---------------------------------------------------------------------------
# Phase file I/O
# ---------------------------------------------------------------------------

def read_phase(task_dir: str) -> str:
    """Read current phase. Returns INIT if no phase file."""
    path = state_path(task_dir, PHASE_FILE)
    if not os.path.exists(path):
        return INIT
    with open(path, "r") as f:
        phase = f.read().strip()
    return phase if phase in ALL_PHASES else INIT


def write_phase(task_dir: str, phase: str):
    """Write phase to .ar_state/.phase."""
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    path = state_path(task_dir, PHASE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(phase)


# ---------------------------------------------------------------------------
# Progress + history I/O
# ---------------------------------------------------------------------------

def load_progress(task_dir: str) -> Optional[dict]:
    """Read .ar_state/progress.json, or None if absent/corrupt.

    Single canonical reader — no other module should re-implement this.
    """
    path = progress_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_progress(task_dir: str, progress: dict, *, stamp: bool = True):
    """Write progress dict to .ar_state/progress.json atomically.
    Optionally stamps last_updated. Single canonical writer.

    Atomicity matters: a non-atomic truncate-then-write would let a
    concurrent reader (e.g. ``compute_next_phase`` reading via
    ``load_progress``) catch an empty / partial file, which
    ``load_progress`` swallows as ``None`` (parse error) and
    ``compute_next_phase`` then treats as ``FINISH`` (budget exhausted).
    The probability per round is tiny but compounds with round count —
    long runs were occasionally short-circuited to FINISH well before
    ``max_rounds``. Writing to a sibling ``.tmp`` and ``os.replace``-ing
    into place gives readers an all-or-nothing view.
    """
    from datetime import datetime, timezone
    path = progress_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if stamp:
        progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, path)


def append_history(task_dir: str, record: dict):
    """Append one JSON record to history.jsonl. Single canonical writer
    used by keep_or_discard and _baseline_init."""
    path = history_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_progress(task_dir: str, **fields) -> Optional[dict]:
    """Load progress, apply **fields, save. Returns the new dict.

    Replaces the scattered `_update_progress_status`, `_reset_consecutive_failures`,
    `_increment_plan_version`, `_update_progress_for_plan` helpers. Silently
    no-ops if progress.json does not exist.
    """
    progress = load_progress(task_dir)
    if progress is None:
        return None
    progress.update(fields)
    try:
        save_progress(task_dir, progress, stamp=False)
    except Exception:
        return None
    return progress


# ---------------------------------------------------------------------------
# Subprocess output parser (every script tail-emits a JSON line)
# ---------------------------------------------------------------------------

def parse_last_json_line(text: str) -> Optional[dict]:
    """Scan `text` from the bottom up and return the last standalone JSON
    object. Our pipeline/baseline/local-eval scripts all follow the protocol
    "stdout last line is JSON"; this is the single place that reads it.
    """
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None

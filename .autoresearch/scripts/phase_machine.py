"""
Phase State Machine for Claude Code AutoResearch.

This is the single source of truth for the optimization loop's deterministic flow.
All hooks import from here. Claude Code never drives the flow — hooks do.

Phases (all transitions mediated by hooks):
    INIT → [GENERATE_REF → GENERATE_KERNEL →] BASELINE → PLAN → EDIT
         (EDIT → pipeline.py → EDIT ...)
         → DIAGNOSE (≥3 consecutive FAIL) → PLAN
         → REPLAN (all items settled) → PLAN
         → FINISH (budget exhausted)

pipeline.py is a subprocess chain (quick_check → eval → keep_or_discard →
settle). Those inner steps run without firing the Bash hook — so they don't
need their own user-visible phases.
"""

import json
import os
import re
import subprocess
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

# File-based task_dir tracking (env vars don't persist across Bash calls)
# Use a FIXED absolute path derived from the project root, not __file__
def _find_project_root() -> str:
    """Walk up from this script to find the project root (where .autoresearch/ lives)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, ".autoresearch")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

_PROJECT_ROOT = _find_project_root()
_ACTIVE_TASK_FILE = os.path.join(_PROJECT_ROOT, ".autoresearch", ".active_task")


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

    # Fallback: env var
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
    conflicting concurrent Claude Code sessions.
    """
    try:
        heartbeat = state_path(task_dir, ".heartbeat")
        os.makedirs(os.path.dirname(heartbeat), exist_ok=True)
        with open(heartbeat, "w") as f:
            import time
            f.write(f"{int(time.time())}\n")
    except Exception:
        pass


def clear_task_dir():
    """Remove .active_task file (session ended)."""
    if os.path.exists(_ACTIVE_TASK_FILE):
        os.remove(_ACTIVE_TASK_FILE)

# ---------------------------------------------------------------------------
# Phase rules — the declarative authority on what's allowed per phase.
#
# Hooks are thin dispatchers: they call check_bash() / check_edit() below and
# pass the verdict to Claude Code's decision channel. The phase machine owns
# every per-phase allow/block decision so adding or changing a phase never
# requires editing hook files.
# ---------------------------------------------------------------------------

# Scripts that are never callable via the user-facing Bash tool — they're
# subprocess children of pipeline.py and should never be user-invoked.
_GLOBAL_BASH_BANS = {
    "eval_wrapper.py":   "subprocess-only (invoked by pipeline.py)",
    "keep_or_discard.py": "subprocess-only (invoked by pipeline.py)",
    "quick_check.py":    "subprocess-only (invoked by pipeline.py)",
    "settle.py":         "subprocess-only (invoked by pipeline.py)",
}

# Read-only command prefixes allowed in every phase.
_READONLY_PATTERNS = [
    r"^(ls|cat|head|tail|wc|find|grep|git\s+(log|diff|status|show|branch))",
    r"dashboard\.py",
    r"^echo\s",
    r"^pwd$",
]


class _BashPolicy:
    """Per-phase Bash rule. Two modes:

    strict:     command must match one of `required` substrings (or be
                readonly / activation). Everything else → block.
    permissive: command is allowed UNLESS it matches one of `banned`
                substrings (or the global ban list).

    "strict" fits narrow phases (INIT/BASELINE/GENERATE_*) where Claude
    should only be running the one script that advances the phase.
    "permissive" fits work phases (PLAN/EDIT/DIAGNOSE/REPLAN) where Claude
    legitimately needs ad-hoc shell access (git log, Python one-liners,
    reading files) and we only block a few known-wrong actions.
    """
    __slots__ = ("mode", "required", "banned")

    def __init__(self, mode, required=None, banned=None):
        assert mode in ("strict", "permissive")
        self.mode = mode
        self.required = set(required or ())
        self.banned = set(banned or ())


_BASH_RULES = {
    INIT:            _BashPolicy("strict", required={"export AR_TASK_DIR="}),
    BASELINE:        _BashPolicy("strict", required={"baseline.py"}),
    GENERATE_REF:    _BashPolicy("strict", required=set()),
    GENERATE_KERNEL: _BashPolicy("strict", required=set()),
    PLAN:            _BashPolicy("permissive", banned=set()),
    DIAGNOSE:        _BashPolicy("permissive", banned=set()),
    REPLAN:          _BashPolicy("permissive", banned=set()),
    # EDIT is permissive for ad-hoc shell, but blocks create_plan.py —
    # Claude must finish the current plan item via pipeline before replanning.
    EDIT:            _BashPolicy("permissive", banned={"create_plan.py"}),
    FINISH:          _BashPolicy("permissive", banned=set()),
}

# Edit/Write rules: which file classes may be written per phase.
#   "ref"      — reference.py
#   "editable" — anything in task.yaml:editable_files
# plan.md is never in any set — it's machine-generated.
_EDIT_RULES = {
    GENERATE_REF:    {"ref"},
    GENERATE_KERNEL: {"editable"},
    EDIT:            {"editable"},
    # All other phases: no writable user files.
}

# Phases where Claude can edit editable_files (typically kernel.py).
# Kept as a set for hook_post_edit's `phase in CODE_EDIT_PHASES` check; it
# mirrors _EDIT_RULES.
CODE_EDIT_PHASES = {p for p, classes in _EDIT_RULES.items() if "editable" in classes}
REF_WRITE_PHASES = {p for p, classes in _EDIT_RULES.items() if "ref" in classes}


def _is_readonly_bash(command: str) -> bool:
    for pat in _READONLY_PATTERNS:
        if re.search(pat, command.strip()):
            return True
    return False


def check_bash(phase: str, command: str) -> tuple:
    """Return (allowed: bool, reason: str) for a Bash command at `phase`.

    Decision order:
      1. Global bans (subprocess-only scripts, `git commit`) — always block.
      2. Read-only commands — always allow.
      3. Activation (`export AR_TASK_DIR=…`) — always allow; hook_post_bash
         uses it to switch tasks regardless of current phase.
      4. Phase policy (strict whitelist or permissive blocklist).
    """
    for ban, why in _GLOBAL_BASH_BANS.items():
        if ban in command:
            return False, f"'{ban}' — {why}"
    if "git commit" in command:
        return False, ("manual 'git commit' forbidden — commits are produced "
                       "by pipeline.py via keep_or_discard")

    if _is_readonly_bash(command):
        return True, ""
    if "export AR_TASK_DIR=" in command:
        return True, ""

    policy = _BASH_RULES.get(phase)
    if policy is None:
        return False, f"unknown phase {phase!r}"

    if policy.mode == "strict":
        for req in policy.required:
            if req in command:
                return True, ""
        required_txt = sorted(policy.required) or "(no user bash legal here; only file edits)"
        return False, f"phase {phase}: allowed commands = {required_txt}"

    for b in policy.banned:
        if b in command:
            return False, f"phase {phase}: '{b}' is blocked here"
    return True, ""


def check_edit(phase: str, rel_path: str, editable_files) -> tuple:
    """Return (allowed: bool, reason: str) for an Edit/Write on `rel_path`
    (task-dir-relative, forward-slash form) at `phase`.

    plan.md is machine-generated and always blocked. Internal state under
    .ar_state/ is always allowed (hooks and scripts use it).
    """
    if rel_path == ".ar_state/plan.md":
        return False, (
            "plan.md is machine-generated — never hand-edit it. Use "
            "`python .autoresearch/scripts/create_plan.py \"<task_dir>\" '<items_json>'` "
            "to propose a new plan."
        )
    if rel_path.startswith(".ar_state/"):
        return True, ""

    allowed_classes = _EDIT_RULES.get(phase, set())
    if "ref" in allowed_classes and rel_path == "reference.py":
        return True, ""
    if "editable" in allowed_classes and rel_path in set(editable_files or ()):
        return True, ""

    return False, f"phase {phase} does not allow writing {rel_path!r}"


# ---------------------------------------------------------------------------
# State file paths (single source of truth — every module uses these)
# ---------------------------------------------------------------------------

def state_path(task_dir: str, name: str) -> str:
    """Path to a file under <task_dir>/.ar_state/. Centralized so no module
    hand-builds state paths."""
    return os.path.join(task_dir, ".ar_state", name)


def plan_path(task_dir: str) -> str:
    return state_path(task_dir, "plan.md")


def progress_path(task_dir: str) -> str:
    return state_path(task_dir, "progress.json")


def history_path(task_dir: str) -> str:
    return state_path(task_dir, "history.jsonl")


def edit_marker_path(task_dir: str) -> str:
    return state_path(task_dir, ".edit_started")


# ---------------------------------------------------------------------------
# Phase file I/O
# ---------------------------------------------------------------------------

def read_phase(task_dir: str) -> str:
    """Read current phase. Returns INIT if no phase file."""
    path = state_path(task_dir, ".phase")
    if not os.path.exists(path):
        return INIT
    with open(path, "r") as f:
        phase = f.read().strip()
    return phase if phase in ALL_PHASES else INIT


def write_phase(task_dir: str, phase: str):
    """Write phase to .ar_state/.phase."""
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    path = state_path(task_dir, ".phase")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(phase)


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(task_dir: str) -> tuple[bool, str]:
    """Validate plan.md structure. Returns (ok, error_message).

    Delegates item parsing to `get_plan_items` (canonical parser) and only
    enforces invariants here: ≥3 items, rationale length, keywords present,
    exactly one ACTIVE pending item.
    """
    if not os.path.exists(plan_path(task_dir)):
        return False, "plan.md does not exist"

    items = get_plan_items(task_dir, include_meta=True)
    if len(items) < 3:
        return False, f"Plan must have ≥ 3 items, found {len(items)}"

    pending = [it for it in items if not it["done"]]
    for it in pending:
        rat = it.get("rationale", "")
        if len(rat) < 30:
            return False, f"Item {it['id']}: rationale too short ({len(rat)} chars, need ≥ 30)"
        if len(rat) > 400:
            return False, f"Item {it['id']}: rationale too long ({len(rat)} chars, max 400)"
        if not it.get("keywords"):
            return False, f"Item {it['id']}: missing keywords"

    active_items = [it for it in pending if it["active"]]
    if len(active_items) != 1:
        return False, f"Must have exactly 1 (ACTIVE) pending item, found {len(active_items)}"

    return True, ""


# ---------------------------------------------------------------------------
# Next phase computation (mechanical, after each pipeline round)
# ---------------------------------------------------------------------------

def compute_next_phase(task_dir: str) -> str:
    """After a pipeline round finishes, mechanically determine the next phase.

    Reads progress.json for counters and plan.md for remaining items.
    """
    progress = load_progress(task_dir)
    if not progress:
        return FINISH

    consecutive_failures = progress.get("consecutive_failures", 0)
    eval_rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 999)

    # Budget exhausted
    if eval_rounds >= max_rounds:
        return FINISH

    # Diagnosis trigger
    if consecutive_failures >= 3:
        return DIAGNOSE

    # Check if plan has remaining pending items
    if has_pending_items(task_dir):
        return EDIT  # More items to work on

    # All items settled
    return REPLAN


def compute_resume_phase(task_dir: str) -> str:
    """Determine phase for resuming after interruption."""
    progress = load_progress(task_dir)
    if not progress:
        return BASELINE

    status = progress.get("status", "no_plan")
    eval_rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 999)

    if eval_rounds >= max_rounds:
        return FINISH

    # Baseline hasn't produced a valid seed metric yet → stay in BASELINE so
    # the user re-runs it after fixing the kernel. Otherwise we'd silently
    # jump to PLAN and start optimizing against no baseline.
    if progress.get("seed_metric") is None:
        return BASELINE

    if not os.path.exists(plan_path(task_dir)) or status == "no_plan":
        return PLAN

    items = get_plan_items(task_dir)
    has_active = any(it["active"] and not it["done"] for it in items)
    has_pending = any(not it["done"] for it in items)

    if has_active or has_pending:
        return EDIT  # has_pending-without-active means: mark next as active
    return REPLAN


# ---------------------------------------------------------------------------
# Active item extraction
# ---------------------------------------------------------------------------

def get_active_item(task_dir: str) -> Optional[dict]:
    """Return the (ACTIVE) pending item, or None. Thin wrapper over get_plan_items."""
    for it in get_plan_items(task_dir):
        if it["active"] and not it["done"]:
            return {"id": it["id"], "description": it["description"]}
    return None


# ---------------------------------------------------------------------------
# Plan item enumeration (for TodoWrite projection)
# ---------------------------------------------------------------------------

_PLAN_ITEM_RE = re.compile(r'\s*-\s*\[([ x])\]\s*\*\*(\w+)\*\*\s*(.*)')
_PLAN_TAG_RE = re.compile(r'^\[([^\]]*)\]:?\s*(.*)')


def get_plan_items(task_dir: str, include_meta: bool = False) -> list:
    """Canonical plan.md parser. Returns [{id, description, done, active, tag}, ...].

    Every plan reader in the codebase must go through this function — no ad-hoc
    regex scans. With include_meta=True, also captures the `- rationale:` and
    `- keywords:` sub-lines (used by validate_plan).
    """
    if not os.path.exists(plan_path(task_dir)):
        return []
    with open(plan_path(task_dir), "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    out = []
    i = 0
    while i < len(lines):
        m = _PLAN_ITEM_RE.match(lines[i])
        if not m:
            i += 1
            continue
        done = m.group(1) == 'x'
        pid = m.group(2)
        rest = m.group(3).strip()
        is_active = "(ACTIVE)" in rest
        tag = ""
        tm = _PLAN_TAG_RE.match(rest)
        if tm:
            tag = tm.group(1).strip()
            rest = tm.group(2)
        desc = rest.replace("(ACTIVE)", "").strip().lstrip(": ").strip()
        item = {"id": pid, "description": desc, "done": done,
                "active": is_active, "tag": tag}

        if include_meta:
            rationale, keywords = "", ""
            j = i + 1
            while j < len(lines):
                sub = lines[j].strip()
                if sub.startswith("- rationale:"):
                    rationale = sub.split(":", 1)[1].strip()
                elif sub.startswith("- keywords:"):
                    keywords = sub.split(":", 1)[1].strip()
                elif sub.startswith("- ") and not sub.startswith("- ["):
                    pass  # other sub-fields like backing_skill
                else:
                    break
                j += 1
            item["rationale"] = rationale
            item["keywords"] = keywords

        out.append(item)
        i += 1
    return out


def has_pending_items(task_dir: str) -> bool:
    """True iff plan.md has at least one unchecked item. Replaces ad-hoc
    `re.findall(r'-\\s*\\[ \\]\\s*\\*\\*\\w+\\*\\*', content)` scans."""
    return any(not it["done"] for it in get_plan_items(task_dir))


# ---------------------------------------------------------------------------
# Guidance messages (what Claude should do in each phase)
# ---------------------------------------------------------------------------

def _load_config_safe(task_dir: str):
    """Load TaskConfig, return None on any failure."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


def get_guidance(task_dir: str) -> str:
    """Return a context-aware instruction for Claude based on current phase.

    Reads task.yaml to inject dynamic info (DSL, editable files, worker URL,
    skills path) so the .md slash command doesn't need to hardcode anything.
    """
    phase = read_phase(task_dir)
    active = get_active_item(task_dir)
    progress = load_progress(task_dir)
    config = _load_config_safe(task_dir)

    # Extract config fields
    dsl = config.dsl if config else None
    editable = config.editable_files if config else []
    worker_urls = config.worker_urls if config else []
    worker_flag = f" --worker-url {worker_urls[0]}" if worker_urls else ""
    primary_metric = config.primary_metric if config else "score"

    if phase == INIT:
        return f"[AR Phase: INIT] Run: export AR_TASK_DIR=\"{task_dir}\""

    if phase == GENERATE_REF:
        description = config.description if config else "(no description)"
        return (f"[AR Phase: GENERATE_REF] Write reference.py for: {description}\n"
                f"Write to: {task_dir}/reference.py\n"
                f"Must contain: class Model(nn.Module) with forward(), get_inputs(), get_init_inputs().\n"
                f"This is the BASELINE implementation — no optimization, just correct.")

    if phase == GENERATE_KERNEL:
        return (f"[AR Phase: GENERATE_KERNEL] Generate initial kernel from reference.\n"
                f"Read {task_dir}/reference.py, then write an optimized version to {task_dir}/kernel.py.\n"
                f"Must contain: class ModelNew (can inherit from Model).\n"
                f"Start with a simple optimization — the autoresearch loop will iterate from here.")

    if phase == BASELINE:
        return (f"[AR Phase: BASELINE] Run: "
                f"python .autoresearch/scripts/baseline.py \"{task_dir}\"{worker_flag}")

    if phase == PLAN:
        skills_hint = ""
        if dsl:
            skills_hint = f'\nSearch skills: Glob("skills/{dsl}/**/*.md") and Read relevant ones.'
        metric_hint = ""
        if progress:
            baseline = progress.get("baseline_metric")
            if baseline is not None:
                metric_hint = f" Baseline {primary_metric}: {baseline}."

        return (f"[AR Phase: PLAN] "
                f"Read task.yaml, editable files ({editable}), and reference.py.{skills_hint}{metric_hint}\n"
                f"Then create the plan by running:\n"
                f'python .autoresearch/scripts/create_plan.py "{task_dir}" \'[{{"desc": "...", "rationale": "... (30-400 chars)", "keywords": "k1, k2"}}, ...]\'\n'
                f"Provide >= 3 items as JSON array. Each item needs: desc, rationale (30-400 chars), keywords.\n"
                f"The script writes plan.md in the correct format. Hook validates and advances to EDIT.\n"
                f"After plan creation, sync items to TodoWrite.")

    if phase == EDIT:
        desc = active["description"] if active else "(no active item)"
        item_id = active["id"] if active else "?"
        files_hint = f" (files: {', '.join(editable)})" if editable else ""
        return (f"[AR Phase: EDIT] ACTIVE item: **{item_id}** — {desc}\n"
                f"{files_hint}\n"
                f"CRITICAL: Implement ONLY {item_id}'s idea. Do NOT implement other plan items.\n"
                f"The pipeline will settle {item_id} with this round's metric.\n"
                f"Make your edit(s), then: python .autoresearch/scripts/pipeline.py \"{task_dir}\"\n"
                f"TodoWrite: mark {item_id} in_progress, other pending items stay pending.")

    if phase == DIAGNOSE:
        # Build failure summary for the diagnosis prompt
        hpath = history_path(task_dir)
        fail_summary = ""
        if os.path.exists(hpath):
            with open(hpath, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            recent = []
            for l in lines[-5:]:
                try:
                    recent.append(json.loads(l))
                except Exception:
                    pass
            for rec in recent:
                _r = rec.get("round")
                _r = "?" if _r is None else _r
                fail_summary += f"  R{_r}: {rec.get('decision','?')} — {rec.get('description','')[:60]}\n"

        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Spawn a SUBAGENT (Agent tool) for fresh-context diagnosis:\n"
                f"  - Have it Read {', '.join(editable)} and .ar_state/history.jsonl\n"
                f"  - Ask it to produce: Root cause / Fix direction / What to avoid\n"
                f"  - It must propose STRUCTURALLY different approaches (algorithmic, fusion, memory layout)\n"
                f"  - NOT more parameter tuning\n"
                f"Recent failures:\n{fail_summary}\n"
                f"After diagnosis, create NEW plan with >= 3 items:\n"
                f'python .autoresearch/scripts/create_plan.py "{task_dir}" \'[{{"desc":"...","rationale":"...","keywords":"..."}},...]\'\n'
                f"Items must be diverse: max 1 parameter-tuning item, rest must be structural changes.\n"
                f"If a past DISCARD/FAIL pid now looks salvageable (e.g. root "
                f"cause was unrelated, structural state has changed), add "
                f"`\"reactivate_pid\": \"pN\"` to an item to reuse that id "
                f"instead of consuming a fresh counter slot.\n"
                f"Then sync TodoWrite.")

    if phase == REPLAN:
        remaining = "?"
        plan_ver = 0
        if progress:
            remaining = str(progress.get("max_rounds", 0) - progress.get("eval_rounds", 0))
            plan_ver = progress.get("plan_version", 0)
        reactivation_hint = ""
        if plan_ver >= 2:
            reactivation_hint = (
                "\nREACTIVATION: plan_version is already {v}. Before inventing "
                "entirely new ideas, scan history.jsonl for DISCARD items whose "
                "metric was close to best (within ~20%) — those ideas may "
                "compose differently now that the kernel's structural baseline "
                "has shifted. To reactivate one, add "
                "`\"reactivate_pid\": \"pN\"` to an item; the old pid is reused "
                "(not a new pN allocated), and history.jsonl gets a REACTIVATE "
                "marker. Only DISCARD/FAIL pids may be reactivated."
                .format(v=plan_ver)
            )
        return (f"[AR Phase: REPLAN] All items settled. Budget: {remaining} rounds left. "
                f"Read .ar_state/history.jsonl. Analyze what worked/failed.\n"
                f"To continue, create new plan:\n"
                f'python .autoresearch/scripts/create_plan.py "{task_dir}" \'[{{"desc": "...", "rationale": "...", "keywords": "..."}},...]\'\n'
                f"Or if no promising directions, do nothing (hooks will advance to FINISH)."
                f"{reactivation_hint}")

    if phase == FINISH:
        best = progress.get("best_metric") if progress else "?"
        baseline = progress.get("baseline_metric") if progress else "?"
        return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} (baseline: {baseline}). "
                f"Write .ar_state/ranking.md summary. Report to user.")

    return f"[AR Phase: {phase}] Unknown phase."


# ---------------------------------------------------------------------------
# Helpers
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
    """Write progress dict to .ar_state/progress.json, optionally stamping
    last_updated. Single canonical writer."""
    from datetime import datetime, timezone
    path = progress_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if stamp:
        progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def append_history(task_dir: str, record: dict):
    """Append one JSON record to history.jsonl. Single canonical writer
    used by keep_or_discard, _baseline_init, and create_plan (supersede /
    reactivate markers)."""
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




def auto_rollback(task_dir: str):
    """Rollback editable files to HEAD."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from task_config import load_task_config
        config = load_task_config(task_dir)
        if config is None:
            return
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True,
        ).stdout.strip()
        for f in config.editable_files:
            fpath = os.path.relpath(os.path.join(task_dir, f), repo_root)
            subprocess.run(["git", "checkout", "HEAD", "--", fpath],
                           cwd=repo_root, capture_output=True)
    except Exception as e:
        print(f"[AR] Rollback failed: {e}", file=sys.stderr)

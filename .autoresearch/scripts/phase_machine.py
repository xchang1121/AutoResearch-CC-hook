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

# ---------------------------------------------------------------------------
# Canonical filenames and templates
# ---------------------------------------------------------------------------

# Files inside <task_dir>/.ar_state/. All path helpers below use these.
PHASE_FILE = ".phase"
PROGRESS_FILE = "progress.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
EDIT_MARKER_FILE = ".edit_started"
HEARTBEAT_FILE = ".heartbeat"
ACTIVE_TASK_FILE = ".active_task"  # under .autoresearch/, not .ar_state/

# Scaffold writes this when --kernel is omitted, so the placeholder is
# distinguishable from a real seed kernel. The matching predicate lives in
# `is_placeholder_file()` below — keep them in lockstep.
KERNEL_PLACEHOLDER = (
    "# TODO: GENERATE_KERNEL phase will fill this in.\n"
    "# Read reference.py and write an initial seed kernel.\n"
    "# Must define class ModelNew (may inherit from Model).\n"
)

# In --desc mode, scaffold.py writes reference.py as a parametric stub:
#   "# TODO: Claude Code will generate reference from description:\n# <desc>\n"
# We can't exact-match it (the description is per-task), so the predicate
# uses this prefix instead.
REFERENCE_PLACEHOLDER_PREFIX = (
    "# TODO: Claude Code will generate reference from description:"
)

def is_placeholder_file(path: str) -> bool:
    """True iff `path` is missing OR matches one of the scaffold placeholders.

    Single source of truth used by hook_post_edit, hook_post_bash._fresh_start,
    and validate_kernel. Update this rule and the placeholder templates
    (`KERNEL_PLACEHOLDER`, `REFERENCE_PLACEHOLDER_PREFIX`) together.

    Earlier versions used a "contains 'TODO' AND length < 200" heuristic,
    which false-positived a legitimate short seed kernel that happened to
    carry a TODO comment (e.g. `# TODO: tune block size later`) and trapped
    GENERATE_KERNEL forever. We now match against the canonical templates:
      - kernel.py: byte-for-byte match against KERNEL_PLACEHOLDER (fixed text)
      - reference.py: prefix match against REFERENCE_PLACEHOLDER_PREFIX
        (parametric — description text is appended per task)
    Anything Claude has actually written deviates and is no longer a stub.
    """
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return True
    stripped = content.strip()
    if stripped == KERNEL_PLACEHOLDER.strip():
        return True
    if stripped.startswith(REFERENCE_PLACEHOLDER_PREFIX):
        return True
    return False

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


# Shared plan-item scaffolding shown in PLAN / DIAGNOSE / REPLAN guidance.
# The example is deliberately a short SENTENCE (not a snake_case identifier) —
# dashboards surface `desc` directly in the history and plan tables, so
# "Fuse SwiGLU into the matmul epilogue to avoid a second launch" reads far
# better than "fuse_swiglu_epilogue". create_plan.py enforces the prose form.
#
# XML is the required format — tag-delimited text is structurally harder for
# LLMs to hallucinate than JSON (no stray commas / quote escaping / brace
# balance to track).
# Inline XML comments inside the example double as schema reminders. The
# model is far more likely to obey rules embedded in the structure it's
# mimicking than rules sitting in a separate paragraph it has to remember
# to apply. Anti-drift hints are placed where each drift tends to land:
# attributes on <item>, extra child elements, missing fields.
_PLAN_XML_EXAMPLE = (
    '<items>'
    '<!-- Provide >= 3 <item> elements. No attributes or extra tags on <items>. -->'
    '<item>'
    '<!-- An <item> has NO attributes and EXACTLY two child elements: '
    '<desc> and <rationale>. Do NOT add <id>, <pid>, <keywords>, '
    '<priority>, <reactivate_pid>, or id="..." / pid="..." attributes. '
    'Pids are auto-assigned by create_plan.py from a monotonic counter; '
    'the model never supplies them — supplying one is rejected. -->'
    '<desc>Fuse SwiGLU into the matmul epilogue to avoid a second launch</desc>'
    '<!-- <desc>: short SENTENCE (>=12 chars, has spaces). Not a '
    'snake_case label; the dashboard shows desc verbatim. -->'
    '<rationale>Separate SwiGLU kernel re-reads the matmul output from DRAM; '
    'fusing it into the epilogue cuts one round-trip and a launch.</rationale>'
    '<!-- <rationale>: 30-400 chars, explains WHY this should help. -->'
    '</item>'
    '<!-- Repeat <item> blocks for >= 3 total items. Same two-child rule '
    'each time; nothing per-item is optional and nothing extra is allowed. -->'
    '</items>'
)
_PLAN_FIELD_RULES = (
    "Schema reminders are embedded as <!-- comments --> inside the XML "
    "example above; read them — each comment marks the spot where a "
    "field rule applies. Beyond schema: escape '&', '<', '>' in text as "
    "'&amp;', '&lt;', '&gt;' (or wrap the offending field in "
    "<![CDATA[...]]>). If shell-quoting is awkward, write the XML to a "
    "file and pass '@path.xml' as the second argument instead."
)


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

    Writes under .ar_state/ are restricted to a precise allowlist. Phase,
    progress, history, plan.md, heartbeat, and markers are all machine-
    maintained — letting Claude Edit them would let the model skip phases,
    rewrite counters, or forge history. Only two paths are writable by the
    agent:
      - .ar_state/plan_items.xml: the XML input file /autoresearch hands to
        create_plan.py (see .claude/commands/autoresearch.md).
      - .ar_state/ranking.md: the FINISH-phase summary (phase-gated).
    """
    if rel_path.startswith(".ar_state/"):
        if rel_path == ".ar_state/plan_items.xml":
            return True, ""
        if rel_path == ".ar_state/ranking.md":
            if phase == FINISH:
                return True, ""
            return False, (
                "ranking.md is only writable in the FINISH phase — "
                "finish the optimization loop first."
            )
        if rel_path == f".ar_state/{PLAN_FILE}":
            return False, (
                "plan.md is machine-generated — never hand-edit it. Use "
                "`python .autoresearch/scripts/create_plan.py "
                "\"<task_dir>\" @<path>` to propose a new plan."
            )
        return False, (
            f"{rel_path!r} is machine-maintained state. Only "
            ".ar_state/plan_items.xml (plan input) and .ar_state/ranking.md "
            "(FINISH summary) are writable under .ar_state/; everything else "
            "is owned by hooks and scripts."
        )

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
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(task_dir: str) -> tuple[bool, str]:
    """Validate plan.md structure. Returns (ok, error_message).

    Delegates item parsing to `get_plan_items` (canonical parser) and only
    enforces invariants here: ≥3 items, rationale length within bounds,
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

    active_items = [it for it in pending if it["active"]]
    if len(active_items) != 1:
        return False, f"Must have exactly 1 (ACTIVE) pending item, found {len(active_items)}"

    return True, ""


# ---------------------------------------------------------------------------
# Reference / kernel runnability validators
#
# These are the single authority used by both hook_post_edit.py and
# hook_post_bash.py to decide whether a fresh / just-edited reference.py or
# kernel.py is real enough to advance the phase. Failure does NOT raise —
# returns (False, human-readable reason) so callers can keep the phase pinned
# and emit guidance for Claude to re-Edit.
# ---------------------------------------------------------------------------

# Subprocess template for running reference.py end-to-end on CPU. We only
# care that import + Model(*get_init_inputs())(*get_inputs()) survives;
# outputs are discarded (the worker captures them on first verify).
_REF_RUNCHECK_SCRIPT = r'''
import json, sys, traceback
sys.path.insert(0, {task_dir!r})
try:
    import torch
    from {ref_mod} import Model, get_inputs, get_init_inputs
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"ok": False, "stage": "import", "error": str(e)}}))
    sys.exit(1)
try:
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).cpu().eval()
    inputs = get_inputs()
    inputs = [x.cpu() if hasattr(x, "cpu") else x for x in inputs]
    with torch.no_grad():
        outs = model(*inputs)
    if outs is None:
        print(json.dumps({{"ok": False, "stage": "forward",
                           "error": "Model.forward() returned None"}}))
        sys.exit(1)
    print(json.dumps({{"ok": True}}))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"ok": False, "stage": "run", "error": str(e)}}))
    sys.exit(1)
'''


def validate_reference(task_dir: str) -> tuple[bool, str]:
    """Two-stage runnability check on <task_dir>/reference.py.

    Stage 1: AST symbol presence — delegates to scaffold.validate_ref so the
             rule lives in exactly one place.
    Stage 2: Subprocess that imports the module and runs Model.forward() on
             CPU. CUDA / Ascend devices are masked off; KMP_DUPLICATE_LIB_OK
             is set for Windows libiomp5 double-load.

    Never raises. Returns (True, "") on success, (False, reason) otherwise.
    """
    ref_path = os.path.join(task_dir, "reference.py")
    if not os.path.exists(ref_path):
        return False, "reference.py does not exist"

    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_code = f.read()
    except OSError as e:
        return False, f"cannot read reference.py: {e}"

    # Stage 1: AST symbols.
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from scaffold import validate_ref as _validate_ref_ast
        _validate_ref_ast(ref_code, ref_path)
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"AST check failed: {e}"

    # Stage 2: subprocess import + forward.
    code = _REF_RUNCHECK_SCRIPT.format(task_dir=task_dir, ref_mod="reference")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "ASCEND_RT_VISIBLE_DEVICES": "",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, cwd=task_dir, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "reference.py runnability check timed out (>60s)"
    except Exception as e:
        return False, f"subprocess launch failed: {e}"

    if r.returncode == 0:
        return True, ""

    # Parse the child's last JSON line for a clean error message; fall back to
    # the raw stderr tail if the child crashed before printing JSON.
    info = parse_last_json_line(r.stdout)
    if info and not info.get("ok", False):
        stage = info.get("stage", "?")
        err = info.get("error", "(no detail)")
        return False, f"reference.py failed at {stage}: {err}"
    tail = (r.stderr or "")[-400:].strip()
    return False, f"reference.py runnability check failed: {tail or '(no stderr)'}"


def validate_kernel(task_dir: str) -> tuple[bool, str]:
    """Static check on every editable file (typically kernel.py).

    Rejects the TODO placeholder up front, then delegates to
    quick_check._check_editable_files (which runs the CodeChecker pipeline:
    syntax → compile → imports → stray-text → DSL → autotune).

    Never raises. Returns (True, "") on success, (False, reason) otherwise.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from task_config import load_task_config
    except Exception as e:
        return False, f"cannot import task_config: {e}"

    config = load_task_config(task_dir)
    if config is None:
        return False, "task.yaml not found or invalid"

    # Placeholder fast-path: if any editable file is still the scaffold TODO,
    # the kernel hasn't been generated yet. Subprocess CodeChecker would
    # technically pass on a comment-only file, but the intent is to hold the
    # phase at GENERATE_KERNEL until real code lands.
    for fname in config.editable_files:
        fpath = os.path.join(task_dir, fname)
        if not os.path.exists(fpath):
            return False, f"editable file missing: {fname}"
        if is_placeholder_file(fpath):
            return False, (f"{fname} is still the scaffold TODO placeholder — "
                           f"write the seed kernel (must define class ModelNew)")

    try:
        from quick_check import _check_editable_files
    except Exception as e:
        return False, f"cannot import quick_check: {e}"

    try:
        issues = _check_editable_files(task_dir, config)
    except Exception as e:
        return False, f"CodeChecker pipeline crashed: {e}"

    if not issues:
        return True, ""

    parts = []
    for it in issues:
        parts.append(f"- {it.get('file', '?')}: {it.get('report', '(no report)')}")
    return False, "CodeChecker found issues:\n" + "\n".join(parts)


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

    # Baseline didn't settle cleanly → demote to GENERATE_KERNEL so Edit on
    # kernel.py is permitted again (BASELINE phase blocks editable_files
    # writes). Mirrors hook_post_bash's live-session demotion: both
    # seed_metric=None (no timing) and baseline_correctness=False (wrong
    # output) count as failure. A seed that profiled but didn't verify is
    # still a broken seed — letting resume enter PLAN would build a plan
    # against a reference that the kernel doesn't actually match.
    if (progress.get("seed_metric") is None
            or progress.get("baseline_correctness") is False):
        return GENERATE_KERNEL

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
    regex scans. With include_meta=True, also captures the `- rationale:`
    sub-line (used by validate_plan).
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
            rationale = ""
            j = i + 1
            while j < len(lines):
                sub = lines[j].strip()
                if sub.startswith("- rationale:"):
                    rationale = sub.split(":", 1)[1].strip()
                elif sub.startswith("- ") and not sub.startswith("- ["):
                    # other sub-fields (legacy `- keywords:` from older
                    # plan.md files, or future hand-written notes) are
                    # skipped silently
                    pass
                else:
                    break
                j += 1
            item["rationale"] = rationale

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
                f'python .autoresearch/scripts/create_plan.py "{task_dir}" \'{_PLAN_XML_EXAMPLE}\'\n'
                f"{_PLAN_FIELD_RULES}\n"
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

        editable_list = ", ".join(editable)
        # Pre-baked subagent prompt. The parent model passes this verbatim
        # to the Agent tool so the subagent doesn't improvise its own
        # research strategy (a previous open-ended brief sent it greppting
        # git log for 100+ tool calls before timing out).
        subagent_prompt = (
            f"Diagnose why the current optimization rounds are failing.\n\n"
            f"Read these files AND ONLY these files (no other Read / Glob / Grep):\n"
            f"  - {task_dir}/reference.py\n"
            f"  - {task_dir}/{editable_list}\n"
            f"  - {task_dir}/.ar_state/plan.md\n"
            f"  - {task_dir}/.ar_state/history.jsonl (focus on the last "
            f"~10 rounds; older entries are usually stale)\n\n"
            f"Hard constraints:\n"
            f"  - Do NOT run `git log`, `git show`, `git grep`, or any git "
            f"history search — the task git history only contains generic "
            f"per-round commits and grepping it for keywords ('vector', "
            f"'Welford', etc.) returns nothing useful and burns tool calls.\n"
            f"  - Do NOT Glob / Grep the wider codebase. Everything you need "
            f"is in the files above.\n"
            f"  - Stop after at most 8 tool uses total; if you can't fully "
            f"conclude, output what you have.\n\n"
            f"Produce a tight report (<300 words total) with three sections:\n"
            f"  1. Root cause: one paragraph on what's making rounds fail\n"
            f"  2. Fix directions: at most 3 STRUCTURALLY different "
            f"approaches (algorithmic change / fusion / memory layout / "
            f"data movement). One sentence each. NOT more parameter tuning.\n"
            f"  3. What to avoid: at most 3 patterns to NOT repeat. One "
            f"sentence each."
        )
        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Spawn a SUBAGENT (Agent tool) with this EXACT prompt — "
                f"do not paraphrase, do not add or remove constraints:\n"
                f"---BEGIN SUBAGENT PROMPT---\n"
                f"{subagent_prompt}\n"
                f"---END SUBAGENT PROMPT---\n"
                f"Recent failures (already in your context, no need to "
                f"re-fetch):\n{fail_summary}\n"
                f"After the subagent returns, create NEW plan with >= 3 "
                f"items:\n"
                f'python .autoresearch/scripts/create_plan.py "{task_dir}" '
                f"'{_PLAN_XML_EXAMPLE}'\n"
                f"{_PLAN_FIELD_RULES}\n"
                f"Items must be diverse: max 1 parameter-tuning item, rest "
                f"must be structural changes.\n"
                f"create_plan.py will REPLACE plan.md's Active Items — any "
                f"pid left pending in the previous plan is silently dropped "
                f"(its slot in the monotonic pid counter stays consumed). "
                f"If a past DISCARD/FAIL idea now looks salvageable, just "
                f"re-propose it as a new item; the desc text carries the "
                f"audit, fresh pid is the right signal that it's a new "
                f"attempt.\n"
                f"Then sync TodoWrite.")

    if phase == REPLAN:
        remaining = "?"
        plan_ver = 0
        if progress:
            remaining = str(progress.get("max_rounds", 0) - progress.get("eval_rounds", 0))
            plan_ver = progress.get("plan_version", 0)
        retry_hint = ""
        if plan_ver >= 2:
            retry_hint = (
                f"\nNote: plan_version is already {plan_ver}. Before "
                "inventing entirely new ideas, scan history.jsonl for "
                "DISCARD items whose metric was close to best (within "
                "~20%) — those ideas may compose differently now that "
                "the kernel's structural baseline has shifted. To revisit "
                "one, just include it as a new item with a fresh pid "
                "(reference the prior pid in <desc> for audit context)."
            )
        return (f"[AR Phase: REPLAN] All items settled. Budget: {remaining} rounds left. "
                f"Read .ar_state/history.jsonl. Analyze what worked/failed.\n"
                f"To continue, create new plan:\n"
                f'python .autoresearch/scripts/create_plan.py "{task_dir}" \'{_PLAN_XML_EXAMPLE}\'\n'
                f"{_PLAN_FIELD_RULES}\n"
                f"Or if no promising directions, do nothing (hooks will advance to FINISH)."
                f"{retry_hint}")

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

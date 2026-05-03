"""Reference / kernel / plan validators + plan.md parser.

Three families live here, all about "is this artifact OK enough to advance
the phase":

  - is_placeholder_file: does kernel.py / reference.py still contain the
    scaffold-time placeholder text? (placeholder constants live here too.)
  - validate_reference: AST symbols + CPU import-and-run check on
    reference.py.
  - validate_kernel: placeholder fast-path + CodeChecker pipeline (syntax,
    py_compile, imports, stray Chinese, DSL, autotune restore_value).
  - validate_plan: structural check on plan.md (≥3 items, rationale length,
    exactly one ACTIVE).

Plan.md parsing (`get_plan_items`, `has_pending_items`, `get_active_item`)
is bundled here because `validate_plan` is the primary consumer and the
parsing rules are validation-shaped (not just I/O). Other modules that
read plan items go through `get_plan_items`.
"""
import json
import os
import re
import subprocess
import sys
from typing import NamedTuple, Optional

# Sibling-module imports inside the package: state_store gives us paths,
# phase constants, and the JSON-tail parser used to interpret the
# subprocess output of validate_reference.
from .state_store import (
    plan_path, parse_last_json_line,
    diagnose_artifact_path, diagnose_marker, history_path,
    load_progress, DIAGNOSE_ATTEMPTS_CAP,
)


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

# Scaffold writes this when --kernel is omitted, so the placeholder is
# distinguishable from a real seed kernel. The matching predicate
# (`is_placeholder_file()` below) keeps the rule in lockstep.
KERNEL_PLACEHOLDER = (
    "# TODO: GENERATE_KERNEL phase will fill this in.\n"
    "# Read reference.py and write an initial seed kernel.\n"
    "# Must define class ModelNew (may inherit from Model).\n"
)

# In --desc mode, scaffold writes reference.py as a parametric stub:
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


# ---------------------------------------------------------------------------
# Reference runnability check
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


def validate_reference(task_dir: str) -> tuple:
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
        # scaffold lives one level up (in scripts/ root, alongside the
        # phase_machine/ package). Insert the package's parent into
        # sys.path so the import resolves regardless of who's calling us.
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
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


# ---------------------------------------------------------------------------
# Kernel static check (placeholder + CodeChecker pipeline)
# ---------------------------------------------------------------------------

def validate_kernel(task_dir: str) -> tuple:
    """Static check on every editable file (typically kernel.py).

    Rejects the TODO placeholder up front, then delegates to
    quick_check._check_editable_files (which runs the CodeChecker pipeline:
    syntax → compile → imports → stray-text → DSL → autotune).

    Never raises. Returns (True, "") on success, (False, reason) otherwise.
    """
    # quick_check + task_config live in scripts/ (one level up).
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

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
# plan.md parser + structural validation
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
    """True iff plan.md has at least one unchecked item."""
    return any(not it["done"] for it in get_plan_items(task_dir))


def get_active_item(task_dir: str) -> Optional[dict]:
    """Return the (ACTIVE) pending item, or None. Thin wrapper over get_plan_items."""
    for it in get_plan_items(task_dir):
        if it["active"] and not it["done"]:
            return {"id": it["id"], "description": it["description"]}
    return None


_DIAGNOSE_REQUIRED_SECTIONS = ("Root cause", "Fix directions", "What to avoid")


def _last_n_fail_rounds(task_dir: str, n: int = 3) -> list:
    """Read history.jsonl, return the last N records whose decision is FAIL.

    Used by validate_diagnose to require that the artifact references each
    failing round it was supposed to analyze. Tolerant of missing/corrupt
    history lines (they're skipped); returns at most N entries, fewer if
    history doesn't have enough FAILs yet (validator is lenient in that case).
    """
    hpath = history_path(task_dir)
    if not os.path.exists(hpath):
        return []
    fails = []
    try:
        with open(hpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("decision") == "FAIL":
                    fails.append(rec)
    except OSError:
        return []
    return fails[-n:]


def validate_diagnose(task_dir: str, plan_version: int) -> tuple:
    """Validate the DIAGNOSE artifact for `plan_version`.

    Contract (in lockstep with `.claude/agents/ar-diagnosis.md` and the
    DIAGNOSE guidance in `phase_machine/guidance.py`):
      1. File `<task_dir>/.ar_state/diagnose_v<plan_version>.md` exists and
         is non-empty.
      2. Contains the magic marker `[AR DIAGNOSE COMPLETE marker_v<N>]`.
      3. Contains the three required sections: "Root cause",
         "Fix directions", "What to avoid". Match is substring (so either
         "## Root cause" or "Root cause:" passes — generous on heading style,
         strict on content presence).
      4. References each of the last 3 FAIL rounds by `R<round>` token.
         Substring match is sufficient (`R29` matches `R29:` / `(R29)` / etc.)
         If history doesn't have 3 FAILs yet (early DIAGNOSE during
         GENERATE_KERNEL retries), this check is skipped.

    Returns (ok, reason). On failure, `reason` is a short user-facing
    string suitable for an `[AR Phase: DIAGNOSE retry]` message.
    """
    if plan_version is None or plan_version < 0:
        return False, f"invalid plan_version {plan_version!r}"

    path = diagnose_artifact_path(task_dir, plan_version)
    if not os.path.exists(path):
        return False, (
            f"missing artifact {os.path.basename(path)} — the ar-diagnosis "
            f"subagent must Write its report to that exact path")
    try:
        with open(path, "r", encoding="utf-8") as f:
            body = f.read()
    except OSError as e:
        return False, f"cannot read {os.path.basename(path)}: {e}"

    if not body.strip():
        return False, f"{os.path.basename(path)} is empty"

    marker = diagnose_marker(plan_version)
    if marker not in body:
        return False, (f"missing required marker line {marker!r} — the "
                       f"subagent must end the file with this exact string")

    missing_sections = [s for s in _DIAGNOSE_REQUIRED_SECTIONS if s not in body]
    if missing_sections:
        return False, (f"missing required section(s): "
                       f"{', '.join(missing_sections)}. Required headings: "
                       f"{', '.join(_DIAGNOSE_REQUIRED_SECTIONS)}.")

    last_fails = _last_n_fail_rounds(task_dir, n=3)
    if len(last_fails) >= 3:
        missing_refs = []
        for rec in last_fails:
            r = rec.get("round")
            if r is None:
                continue
            tok = f"R{r}"
            if tok not in body:
                missing_refs.append(tok)
        if missing_refs:
            return False, (f"artifact does not reference all of the last 3 "
                           f"FAIL rounds (missing: {', '.join(missing_refs)}). "
                           f"Cite each by its R<n> identifier.")

    return True, ""


class DiagnoseState(NamedTuple):
    """Snapshot of DIAGNOSE-phase state for the hook callers.

    `attempts` is the per-plan_version Task-failure count. The accessor
    folds the "did this counter belong to a different plan_version?"
    cross-check so callers don't reimplement it. `exhausted` is the
    cap-comparison; `artifact_ok` / `artifact_reason` come from
    `validate_diagnose`.
    """
    plan_version: int
    attempts: int
    exhausted: bool
    artifact_ok: bool
    artifact_reason: str


def diagnose_state(task_dir: str,
                   progress: Optional[dict] = None) -> DiagnoseState:
    """Single read of all DIAGNOSE-relevant state needed by hooks.

    Replaces a 6-line copy-paste that used to live in five hook files.
    Pass `progress` if you've already loaded it; otherwise this loads
    it. The artifact validation is always run because hook decisions
    branch on (exhausted) AND (artifact_ok) and the cost is one file
    read + a few regex/substring checks.
    """
    if progress is None:
        progress = load_progress(task_dir) or {}
    pv = progress.get("plan_version", 0) or 0
    if progress.get("diagnose_attempts_for_version") == pv:
        attempts = progress.get("diagnose_attempts", 0) or 0
    else:
        attempts = 0
    artifact_ok, artifact_reason = validate_diagnose(task_dir, pv)
    return DiagnoseState(
        plan_version=pv,
        attempts=attempts,
        exhausted=attempts >= DIAGNOSE_ATTEMPTS_CAP,
        artifact_ok=artifact_ok,
        artifact_reason=artifact_reason,
    )


def validate_plan(task_dir: str) -> tuple:
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

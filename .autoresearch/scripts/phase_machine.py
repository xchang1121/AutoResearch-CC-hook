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

# Reference / kernel filenames at the task_dir root.
DEFAULT_REF_FILE = "reference.py"

# Files inside <task_dir>/.ar_state/. All path helpers below use these.
PHASE_FILE = ".phase"
PROGRESS_FILE = "progress.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
REPORT_FILE = "report.md"
REPORT_JSON_FILE = "report.json"
REPORT_PLOT_FILE = "report.png"
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

# Maximum body length for a file to still count as the scaffold placeholder.
# 200 chars is well above the placeholder template (~150 chars) and well
# below any real implementation, even a one-liner ModelNew that imports
# torch.nn.
_PLACEHOLDER_MAX_LEN = 200


def is_placeholder_file(path: str) -> bool:
    """True iff `path` is missing OR is the scaffold TODO placeholder.

    Single source of truth used by hook_post_edit, hook_post_bash._fresh_start,
    and validate_kernel. Update this rule and the placeholder template
    (`KERNEL_PLACEHOLDER`) together.
    """
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return True
    return "TODO" in content and len(content) < _PLACEHOLDER_MAX_LEN

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

# Bash commands are a convenience surface for reading context and invoking the
# phase scripts. They must not become a second write API that bypasses
# hook_guard_edit's precise file allowlist.
# Shell write meta-syntax. Matched against the command AFTER quoted spans are
# blanked by `_strip_shell_quotes` so XML/text payloads do not false-positive.
#   `(^|[^<])>(?!&)` — bare `>file` (also catches `2>file`); excludes `>&fd`
#                      file-descriptor dup and the second `<` of `<<-` heredoc
#   `>>`             — `>>file` append
#   `<\(`            — process substitution `<(…)` plumbs another command's
#                      output as a fake file
#   `\|\s*(...)\b`   — pipe into a writer/interpreter
#
# Heredoc / herestring (`<<` / `<<<`) is INTENTIONALLY not in this set:
# `python create_plan.py "$DIR" - << 'EOF' ... EOF` is the documented
# stdin-driven plan submission path (see `.claude/commands/autoresearch.md`).
# Stdin-as-program — the actual bypass — is caught by
# `_INTERPRETER_STDIN_PROG_RE` below, which only fires when the heredoc /
# herestring lands on an interpreter with no intervening script argument.
_BASH_WRITE_META_RE = re.compile(
    r"(^|[^<])>(?!&)|>>|<\(|\|\s*(tee|cat|python|perl|ruby|node|powershell|pwsh)\b"
)
# Interpreter-stdin-as-program: `python << EOF`, `bash <<<"…"`, `node <<` —
# these feed the heredoc / herestring body to the interpreter as the program
# to execute (no `-c`, no script file). The middle `(?:\s+-\S+)*` allows
# leading flags like `python -u <<` but disallows a script-file token, so
# `python create_plan.py - << 'EOF'` is permitted. `bash -c "…"` is caught
# separately in `_BASH_MUTATING_PATTERNS`.
_INTERPRETER_STDIN_PROG_RE = re.compile(
    r"\b(python|bash|sh|zsh|dash|fish|ash|ksh|perl|ruby|node|powershell|pwsh)"
    r"(?:\d+(?:\.\d+)?)?(?:\s+-\S+)*\s*<<<?"
)
_BASH_MUTATING_PATTERNS = [
    # `python` invoked as a code-execution surface rather than a script runner:
    # `-c CMD`, `-` (read stdin), `-m pathlib|shutil|os|subprocess` (these
    # modules expose file mutation when run as scripts). Bare `python <<…`
    # is caught by `_INTERPRETER_STDIN_PROG_RE`.
    r"\bpython(?:\d+(?:\.\d+)?)?\s+(-c|-(?=\s|$)|-m\s+(?:pathlib|shutil|os|subprocess))\b",
    # Nested shell-as-interpreter: `bash -c "..."`, `sh -c …`, etc. The inner
    # command is quoted, so `_strip_shell_quotes` blanks it before the meta-RE
    # runs and `>file` inside the quotes would otherwise slip through.
    r"\b(bash|sh|zsh|dash|fish|ash|ksh)\s+-c\b",
    r"\b(powershell|pwsh)\b",
    r"\b(Set-Content|Add-Content|Out-File|New-Item|Remove-Item|Move-Item|Copy-Item)\b",
    # Bare-form write/move utilities. `tee` and `dd` were the two common
    # ways to write a file without a redirection token (`tee FILE`,
    # `dd of=FILE`). Word boundaries keep `committee` / `add` / etc. safe.
    r"\b(rm|mv|cp|touch|truncate|chmod|chown|mkdir|rmdir|tee|dd)\b",
    r"\bgit\s+(add|checkout|restore|reset|clean|merge|rebase|switch|stash|tag|push|pull|fetch|commit)\b",
]


def _strip_shell_quotes(command: str) -> str:
    """Blank quoted spans so XML/text payloads do not look like redirection."""
    out = []
    quote = None
    escaped = False
    for ch in command:
        if quote:
            if escaped:
                escaped = False
            elif quote == '"' and ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            out.append(" ")
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


_HEREDOC_OPEN_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_HERESTRING_RE = re.compile(r"<<<\s*(\S+)")


def _strip_heredoc_bodies(command: str) -> str:
    """Blank heredoc bodies and herestring payloads.

    Same intent as `_strip_shell_quotes`: textual data fed to a command
    should not look like shell metasyntax to the downstream regexes. The
    documented `python create_plan.py "$DIR" - << 'EOF' <items>... EOF`
    flow shipped XML bodies that contain `>` / `>>` characters; without
    this pass `_BASH_WRITE_META_RE` treats `</desc>` as a redirection.

    Handles `<<DELIM`, `<<-DELIM`, `<<'DELIM'`, `<<"DELIM"` (terminator
    must be a line equal to DELIM, leading tabs allowed for `<<-`) and
    `<<<token` herestrings.

    Idempotent. Apply BEFORE `_strip_shell_quotes` so the opener's quoted
    delimiter (`<< 'EOF'`) still has its quotes when the regex runs.
    """
    # Herestrings first: blank the single token that follows `<<<`.
    def _blank_herestring(m):
        return "<<<" + " " * (m.end() - m.start() - 3)
    command = _HERESTRING_RE.sub(_blank_herestring, command)

    # Heredocs: scan line-by-line, blank lines from after the opener until
    # the matching delimiter line.
    lines = command.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _HEREDOC_OPEN_RE.search(line)
        if not m:
            i += 1
            continue
        delim = m.group(1)
        # Optional `<<-` strips leading tabs from both body and delimiter.
        dash = line[m.start():m.start() + 3] == "<<-"
        i += 1
        while i < len(lines):
            body = lines[i]
            terminator = body.lstrip("\t") if dash else body
            if terminator == delim:
                out.append(lines[i])
                i += 1
                break
            out.append(" " * len(body))
            i += 1
    return "\n".join(out)


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


# Shared plan-item scaffolding shown in PLAN / DIAGNOSE / REPLAN guidance.
# The example is deliberately a short SENTENCE (not a snake_case identifier) —
# dashboards surface `desc` directly in the history and plan tables, so
# "Fuse SwiGLU into the matmul epilogue to avoid a second launch" reads far
# better than "fuse_swiglu_epilogue". create_plan.py enforces the prose form.
#
# XML is the required format — tag-delimited text is structurally harder for
# LLMs to hallucinate than JSON (no stray commas / quote escaping / brace
# balance to track).
_PLAN_XML_EXAMPLE = (
    '<items>'
    '<item>'
    '<desc>Fuse SwiGLU into the matmul epilogue to avoid a second launch</desc>'
    '<rationale>Separate SwiGLU kernel re-reads the matmul output from DRAM; '
    'fusing it into the epilogue cuts one round-trip and a launch.</rationale>'
    '<keywords>fusion, epilogue</keywords>'
    '</item>'
    '<!-- repeat <item> for >= 3 total -->'
    '</items>'
)
_PLAN_FIELD_RULES = (
    "Provide >= 3 items as an <items> XML document. Each <item> needs:\n"
    "  - <desc>:      short SENTENCE describing the change (>=12 chars, must "
    "have spaces — not a snake_case label; the dashboard shows this verbatim)\n"
    "  - <rationale>: 30-400 char explanation of WHY it should help\n"
    "  - <keywords>:  comma-separated tags, e.g. fusion, epilogue\n"
    "Optional: <reactivate_pid>pN</reactivate_pid> to reuse a previously "
    "DISCARD/FAIL pid. Escape '&', '<', '>' in text as '&amp;', '&lt;', '&gt;' "
    "(or wrap the field in <![CDATA[...]]>). "
    "Write this XML to .ar_state/plan_items.xml with the Write tool, then "
    "pass it to create_plan.py as @<path>. Do not inline multi-line XML on "
    "the command line; Windows shell quoting can truncate it."
)


def plan_items_xml_path(task_dir: str) -> str:
    return state_path(task_dir, "plan_items.xml")


def _plan_creation_guidance(task_dir: str, *, intro: str) -> str:
    xml_path = plan_items_xml_path(task_dir)
    return (
        f"{intro}\n"
        f"1. Write the XML <items> document to: {xml_path}\n"
        f"Example shape:\n{_PLAN_XML_EXAMPLE}\n"
        f"{_PLAN_FIELD_RULES}\n"
        f"2. Then run:\n"
        f'python .autoresearch/scripts/create_plan.py "{task_dir}" @"{xml_path}"'
    )


def _is_readonly_bash(command: str) -> bool:
    if _looks_mutating_bash(command):
        return False
    for pat in _READONLY_PATTERNS:
        if re.search(pat, command.strip()):
            return True
    return False


def _looks_mutating_bash(command: str) -> bool:
    """Conservative guard for Bash-side writes.

    Edit/Write hooks own file mutation policy. If a Bash command has shell
    write syntax, an interpreter reading its program from stdin (heredoc /
    herestring), or calls a commonly mutating command, it must go through
    the phase policy instead of the cross-phase readonly shortcut.
    """
    cmd = command.strip()
    # Heredoc bodies first — the opener (`<< 'EOF'`) needs its quoted
    # delimiter intact to find the closing line. After bodies are blanked,
    # strip remaining quoted spans (including the now-redundant 'EOF'
    # delimiter quotes themselves).
    unquoted = _strip_heredoc_bodies(cmd)
    unquoted = _strip_shell_quotes(unquoted)
    if _BASH_WRITE_META_RE.search(unquoted):
        return True
    if _INTERPRETER_STDIN_PROG_RE.search(unquoted):
        return True
    return any(re.search(pat, unquoted) for pat in _BASH_MUTATING_PATTERNS)


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
    if "final_report.py" in command and phase != FINISH:
        return False, "final_report.py only runs in the FINISH phase"

    if _looks_mutating_bash(command):
        return False, ("Bash-side file mutation is blocked — use Edit/Write "
                       "so phase file permissions are enforced")

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
    rewrite counters, or forge history. Only one path is writable by the
    agent:
      - .ar_state/plan_items.xml: the XML input file /autoresearch hands to
        create_plan.py (see .claude/commands/autoresearch.md).

    FINISH reports are generated by final_report.py, not hand-written.
    """
    if rel_path.startswith(".ar_state/"):
        if rel_path == ".ar_state/plan_items.xml":
            return True, ""
        if rel_path in (
            f".ar_state/{REPORT_FILE}",
            f".ar_state/{REPORT_JSON_FILE}",
            f".ar_state/{REPORT_PLOT_FILE}",
        ):
            return False, (
                f"{rel_path!r} is generated by final_report.py; do not "
                "hand-edit report artifacts."
            )
        if rel_path == f".ar_state/{PLAN_FILE}":
            return False, (
                "plan.md is machine-generated — never hand-edit it. Use "
                "`python .autoresearch/scripts/create_plan.py "
                "\"<task_dir>\" @<path>` to propose a new plan."
            )
        return False, (
            f"{rel_path!r} is machine-maintained state. Only "
            ".ar_state/plan_items.xml (plan input) is writable under "
            ".ar_state/; report.md/report.json/report.png are generated by "
            "final_report.py, and everything else is owned by hooks and scripts."
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
# History (audit log) loader — single source of truth
# ---------------------------------------------------------------------------

def load_history(task_dir: str, *, on_corrupt: str = "skip") -> list:
    """Read ``.ar_state/history.jsonl`` into a list of dicts (oldest first).

    ``on_corrupt`` selects how malformed JSON lines are handled:
      - ``"skip"``: drop the line silently. Used by call sites that summarize
        history (planner heuristics, dashboard rendering) where one bad line
        should not kill the run.
      - ``"warn"``: drop the line but write a one-line ``[history]`` warning
        to stderr. Use when the absence of a record could mislead downstream
        analysis.
      - ``"record"``: append a synthetic
        ``{"decision": "CORRUPT_HISTORY_LINE", ...}`` record so the corruption
        surfaces in the consumer's output. Used by ``final_report.py`` so the
        FINISH report makes audit gaps visible instead of hiding them.

    Empty / missing files yield ``[]`` regardless of ``on_corrupt``.
    """
    if on_corrupt not in ("skip", "warn", "record"):
        raise ValueError(f"on_corrupt must be skip/warn/record, got {on_corrupt!r}")
    path = history_path(task_dir)
    if not os.path.exists(path):
        return []
    out: list = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                if on_corrupt == "skip":
                    continue
                if on_corrupt == "warn":
                    print(f"[history] {path}:{lineno}: corrupt JSON ({exc})",
                          file=sys.stderr)
                    continue
                out.append({
                    "round": "?",
                    "decision": "CORRUPT_HISTORY_LINE",
                    "description": f"line {lineno}: {exc}",
                    "metrics": {},
                    "correctness": None,
                    "error": line[:200],
                })
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Task discovery — single source of truth
# ---------------------------------------------------------------------------

def _list_task_candidates(project_root: str) -> list:
    """Return [(task_dir_abs, mtime), ...] for every ar_tasks/ entry that
    looks like a real task (has task.yaml). The mtime prefers
    ``.ar_state/progress.json`` then ``.ar_state/.phase`` then dir mtime,
    so an actively running task floats to the top.
    """
    tasks_dir = os.path.join(project_root, "ar_tasks")
    if not os.path.isdir(tasks_dir):
        return []
    out = []
    for d in os.listdir(tasks_dir):
        full = os.path.join(tasks_dir, d)
        if not os.path.isdir(full) or not os.path.exists(os.path.join(full, "task.yaml")):
            continue
        latest = 0.0
        for c in (progress_path(full), state_path(full, PHASE_FILE), full):
            if os.path.exists(c):
                latest = max(latest, os.path.getmtime(c))
        out.append((full, latest))
    return out


def _read_active_pointer(project_root: str) -> Optional[str]:
    """Return the task_dir from ``.autoresearch/.active_task`` if it exists
    AND points at a real directory. Stale pointers are removed and yield None.
    """
    active_file = os.path.join(project_root, ".autoresearch", ACTIVE_TASK_FILE)
    if not os.path.exists(active_file):
        return None
    try:
        with open(active_file, "r", encoding="utf-8") as f:
            td = f.read().strip()
    except OSError:
        return None
    if td and os.path.isdir(td):
        return td
    try:
        os.remove(active_file)
    except OSError:
        pass
    return None


def discover_task_dir(*, prefer_active: bool) -> str:
    """Locate the current task_dir. Returns ``""`` when nothing is found.

    Two ordering policies, named for the caller's intent:

      - ``prefer_active=True`` (used by ``resume.py``): the user wants to
        continue the *foreground* task they were last working on. Honor
        ``.active_task`` first; only fall back to "newest by mtime" when
        the pointer is absent or stale.

      - ``prefer_active=False`` (used by ``dashboard.py``): the user wants
        to *watch* whatever is running right now. Newest-mtime wins so a
        long-stale ``.active_task`` does not pin the dashboard to an
        abandoned task; fall back to ``.active_task`` only when no
        ``ar_tasks/`` candidates exist.

    Both branches walk the same project root (the repo containing
    ``.autoresearch/``) so callers in different cwd land on the same task.
    """
    project_root = _find_project_root()

    if prefer_active:
        td = _read_active_pointer(project_root)
        if td:
            return td
        cands = _list_task_candidates(project_root)
        if not cands:
            return ""
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[0][0]

    cands = _list_task_candidates(project_root)
    if cands:
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[0][0]
    td = _read_active_pointer(project_root)
    return td or ""


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

_PLAN_ITEM_RE = re.compile(r'(?P<indent>\s*-\s*)\[(?P<status>[ x])\]\s*\*\*(?P<pid>\w+)\*\*\s*(?P<rest>.*)')
_PLAN_TAG_RE = re.compile(r'\[([^\]]*)\]')


def parse_plan_line(line: str) -> Optional[dict]:
    """Canonical parser for a single plan.md item line. Returns
    ``{indent, id, done, active, tag, description}`` or None if the line is
    not a plan item.

    The order of the ``(ACTIVE)`` marker and the ``[TAG]`` annotation in the
    rendered line is not significant — both ``(ACTIVE) [REACTIVATED]: desc``
    and ``[KEEP, metric=...]: desc`` parse correctly. Anything in
    ``[brackets]`` before the first ``:`` is captured as ``tag`` and stripped
    from ``description``; only the first such bracket is consumed so a
    description like ``[note] really [important]`` keeps the trailing
    bracketed text intact.
    """
    m = _PLAN_ITEM_RE.match(line)
    if not m:
        return None
    rest = m.group("rest").strip()
    is_active = "(ACTIVE)" in rest
    rest_no_active = rest.replace("(ACTIVE)", " ")

    tag = ""
    head, sep, tail = rest_no_active.partition(":")
    tm = _PLAN_TAG_RE.search(head)
    if tm:
        tag = tm.group(1).strip()
        head = (head[:tm.start()] + head[tm.end():])
    description = (head + sep + tail).strip().lstrip(":").strip()

    return {
        "indent": m.group("indent"),
        "id": m.group("pid"),
        "done": m.group("status") == "x",
        "active": is_active,
        "tag": tag,
        "description": description,
    }


def render_plan_line(pid: str, *, description: str, done: bool = False,
                     active: bool = False, tag: str = "",
                     indent: str = "- ") -> str:
    """Canonical renderer for a single plan.md item line. Always emits
    ``<indent>[ |x] **pid** [TAG] (ACTIVE): description`` with the tag
    *before* the ACTIVE marker, so re-parsing the rendered line is
    insensitive to render-time ordering choices.
    """
    box = "[x]" if done else "[ ]"
    parts = [f"{indent}{box} **{pid}**"]
    if tag:
        parts.append(f"[{tag.strip()}]")
    if active:
        parts.append("(ACTIVE)")
    head = " ".join(parts)
    desc = description.strip()
    if desc:
        return f"{head}: {desc}"
    return head


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
        parsed = parse_plan_line(lines[i])
        if parsed is None:
            i += 1
            continue
        item = {
            "id": parsed["id"],
            "description": parsed["description"],
            "done": parsed["done"],
            "active": parsed["active"],
            "tag": parsed["tag"],
        }

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
                    pass  # ignore unknown sub-fields — schema is rationale + keywords only
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
                f"{_plan_creation_guidance(task_dir, intro='Then create the plan:')}\n"
                f"The script writes plan.md in the correct format. Hook validates and advances to EDIT.\n"
                f"After plan creation, sync items to TodoWrite.")

    if phase == EDIT:
        desc = active["description"] if active else "(no active item)"
        item_id = active["id"] if active else "?"
        files_hint = f" (files: {', '.join(editable)})" if editable else ""

        # Surface the failure budget so a single FAIL doesn't read as "the
        # plan is broken, replan time". DIAGNOSE only fires at 3 consecutive
        # FAILs; until then the plan stays in force.
        status_line = ""
        last_line = ""
        if progress:
            failures = progress.get("consecutive_failures", 0)
            rounds = progress.get("eval_rounds", 0)
            max_r = progress.get("max_rounds", "?")
            best = progress.get("best_metric")
            best_str = f"{best}" if best is not None else "—"
            status_line = (
                f"Round {rounds}/{max_r} | Failures: {failures}/3 toward DIAGNOSE | "
                f"Best {primary_metric}: {best_str}\n"
            )
            history = load_history(task_dir, on_corrupt="skip")
            if history:
                rec = history[-1]
                rid = rec.get("plan_item") or "?"
                rdec = rec.get("decision", "?")
                rdesc = (rec.get("description") or "")[:50]
                if rdec in ("FAIL", "DISCARD", "KEEP", "SEED"):
                    last_line = f"Last round: {rid} → {rdec} ({rdesc})\n"

        return (f"[AR Phase: EDIT] ACTIVE item: **{item_id}** — {desc}\n"
                f"{files_hint}\n"
                f"{status_line}{last_line}"
                f"CRITICAL: Implement ONLY {item_id}'s idea. Do NOT implement other plan items.\n"
                f"NEVER call create_plan.py from EDIT. A single FAIL is not a "
                f"signal to replan — pipeline.py settles the active item, the "
                f"next pending item is promoted automatically, and only the "
                f"hook (at 3 consecutive FAILs or all-items-settled) decides "
                f"when to revise the plan.\n"
                f"Make your edit(s), then: python .autoresearch/scripts/pipeline.py \"{task_dir}\"\n"
                f"TodoWrite: mark {item_id} in_progress, other pending items stay pending.")

    if phase == DIAGNOSE:
        # Build failure summary for the diagnosis prompt
        fail_summary = ""
        for rec in load_history(task_dir, on_corrupt="skip")[-5:]:
            _r = rec.get("round")
            _r = "?" if _r is None else _r
            fail_summary += f"  R{_r}: {rec.get('decision','?')} — {rec.get('description','')[:60]}\n"

        # Untried pending items represent unspent budget. Surface them so
        # Claude knows what would be lost if it just writes a fresh plan
        # without any <reactivate_pid> tags.
        untried = [it for it in get_plan_items(task_dir) if not it["done"]]
        untried_summary = ""
        if untried:
            lines = "\n".join(
                f"  - {it['id']}: {it['description'][:70]}"
                for it in untried
            )
            untried_summary = (
                f"\nPENDING ITEMS that will be DISCARDed unless you reactivate:\n"
                f"{lines}\n"
            )

        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Spawn a SUBAGENT (Agent tool) for fresh-context diagnosis:\n"
                f"  - Have it Read {', '.join(editable)} and .ar_state/history.jsonl\n"
                f"  - Ask it to produce: Root cause / Fix direction / What to avoid\n"
                f"  - It must propose STRUCTURALLY different approaches (algorithmic, fusion, memory layout)\n"
                f"  - NOT more parameter tuning\n"
                f"Recent failures:\n{fail_summary}"
                f"{untried_summary}"
                f"{_plan_creation_guidance(task_dir, intro='After diagnosis, create the next plan revision with >= 3 items:')}\n"
                f"PRESERVE PENDING WORK. Items not in your new <items> document "
                f"are auto-DISCARDed as superseded — that loses unspent budget. "
                f"For every still-untried item that the diagnosis did not "
                f"invalidate, include it in the new plan with "
                f"`<reactivate_pid>pN</reactivate_pid>` (you may also refine its "
                f"desc/rationale). Use a fresh pid only for genuinely new ideas.\n"
                f"Items must be diverse: max 1 parameter-tuning item, rest must be structural changes.\n"
                f"For settled DISCARD/FAIL pids that now look salvageable "
                f"(root cause unrelated, structural state has shifted), the "
                f"same `<reactivate_pid>` mechanism reuses the old pid.\n"
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
                "`<reactivate_pid>pN</reactivate_pid>` to an item; the old pid is reused "
                "(not a new pN allocated), and history.jsonl gets a REACTIVATE "
                "marker. Only DISCARD/FAIL pids may be reactivated."
                .format(v=plan_ver)
            )
        return (f"[AR Phase: REPLAN] All items settled. Budget: {remaining} rounds left. "
                f"Read .ar_state/history.jsonl. Analyze what worked/failed.\n"
                f"{_plan_creation_guidance(task_dir, intro='To continue, create new plan:')}\n"
                f"Or if no promising directions, do nothing (hooks will advance to FINISH)."
                f"{reactivation_hint}")

    if phase == FINISH:
        best = progress.get("best_metric") if progress else "?"
        baseline = progress.get("baseline_metric") if progress else "?"
        return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} (baseline: {baseline}). "
                f"Run: python .autoresearch/scripts/final_report.py \"{task_dir}\". "
                f"Then read .ar_state/report.md and report the result to user.")

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

"""Phase rules + bash/edit gates + transition logic.

This is the policy layer: given a phase + a candidate Bash command (or
Edit/Write target), should it be allowed? After a pipeline round, what's
the next phase?

What's NOT here:
  - Phase string constants — those live in `state_store` (so the I/O
    layer can validate phase strings without circular imports).
  - Plan validation / placeholder detection — those live in `validators`,
    consumed by this module via `has_pending_items`.

What IS here:
  - Global bash bans (`_GLOBAL_BASH_BANS`).
  - Phase-agnostic command patterns (`_PHASE_AGNOSTIC_PATTERNS`).
  - Per-phase Bash policy table (`_BASH_RULES`).
  - Per-phase Edit allowance (`_EDIT_RULES`).
  - `check_bash` / `check_edit` — the two predicates hooks call.
  - `compute_next_phase` / `compute_resume_phase` — phase transition logic.
  - Script-invocation parser (`parse_script_names` etc.) — used by
    both hook_guard_bash (PreToolUse) and hook_post_bash (PostToolUse).
"""
import os
import re
import shlex
from typing import Optional

from .state_store import (
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH,
    PLAN_FILE, PLAN_ITEMS_FILE,
    load_progress, plan_path,
)
from .validators import (
    get_plan_items, has_pending_items,
)


# ---------------------------------------------------------------------------
# Script invocation parser (single source for hook_guard_bash + hook_post_bash)
# ---------------------------------------------------------------------------

# Detects an "interpreter" token (python, py, bash, sh, with optional
# version suffix). Anchored — must match the WHOLE token, not just a
# prefix; otherwise `pythoneer` would falsely register as `python`.
_INTERPRETER_RE = re.compile(
    r'^(?:python(?:\d+(?:\.\d+)?)?|py|bash|sh)$'
)

# Pre-interpreter command-line prefixes the parser is willing to skip
# past while still treating the segment as a real interpreter
# invocation. Currently only POSIX-style env-var assignments
# (`KMP_DUPLICATE_LIB_OK=TRUE python script.py`). Anything else as a
# prefix means this isn't a real launch — `printf python script.py`,
# `time python script.py`, `nohup python script.py` etc. all fail
# (printf prints; time/nohup are wrappers we don't bless). If a real
# wrapper becomes common in our flow, add it here explicitly rather
# than letting the parser scan anywhere in the segment for `python`
# (the old behavior conflated `printf python ...` with a real call).
_ENV_ASSIGNMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')

# Python flags that take their value as the NEXT separate argv token.
# Real python CLI: -X opt, -W setting, --check-hash-based-pycs mode.
# When we see one of these, advance past BOTH the flag and its value.
# (Combined forms like `-Xdev` (no space) appear as a single token —
# handled by the generic "starts with -" skip below.)
_FLAGS_WITH_SEPARATE_VALUE = frozenset({
    "-X", "-W", "--check-hash-based-pycs",
})

# Flags that REPLACE a script invocation. `python -c CODE` runs inline
# code; `python -m MODULE` runs a module; in both cases anything that
# follows is argv to the inline program, NOT a separate script. Same
# for `bash -c CODE`. When we hit one of these, the segment has no
# real script invocation regardless of what tokens come later — this
# is the fix for the smuggle pattern
# `python -c pass .autoresearch/scripts/create_plan.py` which the
# old regex parser misclassified as a real create_plan.py call.
_FLAGS_REPLACE_SCRIPT = frozenset({"-c", "-m"})


def _find_script_position(tokens: list, start: int) -> Optional[int]:
    """Walk `tokens` starting at `start` (just past an interpreter token).
    Return the index of the first non-flag token if it ends in `.py`,
    else None (no real script invocation in this run)."""
    j = start
    while j < len(tokens):
        t = tokens[j]
        if t in _FLAGS_REPLACE_SCRIPT:
            # `-c CODE` / `-m MODULE` — the rest is inline-program argv.
            return None
        if t in _FLAGS_WITH_SEPARATE_VALUE:
            j += 2  # skip flag AND its value token
            continue
        if t.startswith("-"):
            j += 1  # standalone flag like -O / -u / -3 / -3.10 / -Xdev
            continue
        # First non-flag positional token must be the script.
        return j if t.endswith(".py") else None
    return None


def _scan_segment_for_invocation(tokens: list) -> Optional[int]:
    """Return the index of the script token if `tokens` (one bash
    chain segment) is a real interpreter-runs-script invocation, else
    None.

    Interpreter detection is COMMAND-HEAD-ANCHORED. The parser:
      1. Skips zero or more env-var assignment tokens
         (`KEY=VALUE`-shaped) at the head.
      2. Requires the next token to match `_INTERPRETER_RE`. Anything
         else (printf, time, nohup, sudo, the result of a subshell,
         an arbitrary command) means this segment is NOT a real
         Python/bash launch — return None.
      3. Walks flag tokens after the interpreter via the rules in
         _find_script_position (`-c`/`-m` short-circuit; `-X`/`-W`
         consume value tokens; other flags skipped) and returns the
         script index iff we land on a non-flag positional `.py`
         token.

    This is the fix for the false positive
        printf python .autoresearch/scripts/create_plan.py
    which the old "scan anywhere for `python`" loop treated as a
    real create_plan.py call — the actual command runs printf, not
    Python, but strict-bash and recovery gates were both fooled into
    thinking create_plan.py had been launched.
    """
    if not tokens:
        return None
    j = 0
    while j < len(tokens) and _ENV_ASSIGNMENT_RE.match(tokens[j]):
        j += 1
    if j >= len(tokens) or not _INTERPRETER_RE.match(tokens[j]):
        return None
    return _find_script_position(tokens, j + 1)


def parse_script_names(command: str) -> list:
    """Return [(script_path_forward_slashes, basename), ...] for EVERY
    real python/bash interpreter-runs-script invocation in `command`,
    in order.

    Splits the command into bash chain segments (`&&` / `||` / `;` /
    `|`), tokenizes each via shlex (POSIX mode), then asks
    _scan_segment_for_invocation whether the SEGMENT HEAD looks like
    a real launch. Interpreter detection is anchored at the
    command-head position (after optional env-var assignments) — the
    parser does NOT scan deeper into the segment for stray `python`
    tokens. Subshell launches like `$(python script.py)` are
    deliberately not detected here; the global ban list (substring
    match over the whole command) already catches subprocess-only
    scripts smuggled into subshells.

    Windows backslash separators are normalized to forward slashes
    PER SEGMENT (after _split_bash_chain has finished its quote/escape
    walk). shlex(posix=True) treats `\\` as an escape character, so
    `python .autoresearch\\scripts\\create_plan.py` would tokenize to
    `.autoresearchscriptscreate_plan.py` and miss the `.py` suffix
    entirely (false-blocking real Windows invocations in strict-bash
    phases). Forward slashes are valid path separators on Windows
    from Python's perspective and have no shell-escape semantics, so
    the rewrite is safe. Per-segment (rather than whole-command)
    normalization means escape sequences like `\\&` outside quotes
    don't accidentally become a `/&` that the splitter has already
    walked past — the splitter sees the raw `\\&` and applies its
    backslash-escape rule, which is what we want.
    """
    out = []
    for raw_segment in _split_bash_chain(command):
        segment = raw_segment.replace("\\", "/")
        try:
            tokens = shlex.split(segment, comments=False, posix=True)
        except ValueError:
            # Unbalanced quotes etc. — skip the segment rather than
            # raise. The downstream gate's default of "no recognized
            # invocation" is the safe outcome here.
            continue
        pos = _scan_segment_for_invocation(tokens)
        if pos is None:
            continue
        path = tokens[pos]
        out.append((path, os.path.basename(path)))
    return out


def parse_script_name(command: str) -> Optional[tuple]:
    """Single-invocation form — first .py invocation in `command`.

    Kept for callers that only care about the first script. New code
    that needs full-chain awareness should use parse_script_names.
    """
    matches = parse_script_names(command)
    return matches[0] if matches else None


def parse_invoked_ar_script(command: str) -> Optional[str]:
    """Basename of an .autoresearch/scripts/*.py invocation, or None.

    Used by hook_post_bash to dispatch phase advances on
    `baseline.py` / `pipeline.py` / `create_plan.py` after the user-issued
    Bash returns. The dispatch logic deliberately inspects only the FIRST
    matching .autoresearch/scripts/*.py — chained invocations would have
    been blocked by hook_guard_bash before they reached the post hook
    (the chain-segment rule rejects any chain that includes a non-allowed
    script in the current phase), so post-hook only sees the legitimate
    single invocation that was just permitted.
    """
    for script_path, basename in parse_script_names(command):
        if ".autoresearch/scripts/" in script_path:
            return basename
    return None


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

# Commands the bash gate always allows, regardless of phase.
# Two reasons something lands here:
#   (a) Phase-agnostic inspection — ls/cat/head/tail/grep/wc/find,
#       read-only git, dashboard.py, parse_args.py, echo, pwd. No side
#       effects on task state.
#   (b) Task lifecycle ops — scaffold.py (creates a new task) and resume.py
#       (switches the active task pointer). These have side effects, but
#       the side effects are about WHICH task is active, not about the
#       inner state of an already-active task. The phase machine is meant
#       to keep the agent on the rails of a SINGLE task's optimization
#       loop; lifecycle ops sit above that. If they were subject to phase
#       rules, /autoresearch could not start a new task whenever a prior
#       `.active_task` happened to point at one mid-BASELINE — exactly
#       the deadlock that motivated this list.
_PHASE_AGNOSTIC_PATTERNS = [
    r"^(ls|cat|head|tail|wc|find|grep|git\s+(log|diff|status|show|branch))",
    r"dashboard\.py",
    r"parse_args\.py",
    r"scaffold\.py",
    r"resume\.py",
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
    # DIAGNOSE: only create_plan.py is a legal AR-script invocation. The
    # phase's contract is "produce a new plan via Task -> artifact ->
    # create_plan" (or manual-planning fallback after the cap). Read-only
    # ops and AR_TASK_DIR export still pass via the phase-agnostic /
    # activation special cases. hook_guard_bash adds the artifact gate on
    # create_plan.py itself.
    DIAGNOSE:        _BashPolicy("strict", required={"create_plan.py"}),
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


def _segment_is_phase_agnostic(segment: str) -> bool:
    s = segment.strip()
    if not s:
        return True  # empty segments (trailing `&&`) don't add restrictions
    for pat in _PHASE_AGNOSTIC_PATTERNS:
        if re.search(pat, s):
            return True
    return False


# Bash separators recognized as chain operators: && || ; | & (newline
# left out — the model rarely uses literal newlines in slash-issued
# commands). Bare `&` (backgrounding) was earlier omitted on the
# argument that the model rarely uses it; in practice it's a free
# smuggle channel — `python create_plan.py & python pipeline.py` ran
# both invocations while strict-bash and EDIT-recovery gates only saw
# the first one. We now split on it. Redirection forms like `2>&1`
# do split (`'... 2>'` + `'1'`), but the second segment carries no
# interpreter and yields no script invocation, so the gate result
# is unchanged for legitimate redirects. We split on these and
# require EVERY segment to be phase-agnostic OR pass the per-phase
# rule.
#
# Splitting must respect shell quoting and backslash escapes — a naive
# `re.split(r'&&|\|\||;|\|')` over `grep 'a|b' foo` cuts the quoted `|`
# and produces two nonsense segments that fail the rule, false-blocking
# a perfectly legal command. _split_bash_chain walks the string with a
# tiny quote/escape tracker; that's all this routine claims to do.
#
# What this gate is NOT:
# Subshells (`$(...)`, backticks) and process substitution (`<()`) are
# NOT parsed — substitution bodies appear as opaque text inside a
# single segment. The phase rules are substring heuristics, so:
#   - In strict phases, a substitution body can ride along with a legal
#     outer command. `python baseline.py $(python pipeline.py x)` is
#     accepted in BASELINE because the segment contains the required
#     substring "baseline.py". The pipeline.py invocation inside the
#     `$(...)` is invisible to this gate.
#   - In permissive phases, substring bans incidentally catch banned
#     names anywhere in the segment, including inside `$()` — but that
#     is coincidence, not enforcement.
#
# Threat model: this is an LLM-operations guardrail, not a security
# sandbox. It prevents the routine drift modes we've actually observed
# (chain-and-smuggle, wrong script name, off-phase command). It does
# not defend against an adversarial command author trying to evade.
# If `$()` / backtick smuggling shows up as an observed pattern in a
# real run, add a dedicated guard then — don't pre-build half a bash
# parser on the assumption.

def _split_bash_chain(command: str) -> list:
    """Split `command` on `&&` / `||` / `;` / `|` / `&` outside quotes."""
    segments = []
    cur = []
    i = 0
    n = len(command)
    in_single = False
    in_double = False
    while i < n:
        c = command[i]

        # Backslash escape: take the next char literally (only honored
        # outside single quotes, like real bash). At end of string just
        # emit the backslash.
        if c == "\\" and not in_single and i + 1 < n:
            cur.append(c)
            cur.append(command[i + 1])
            i += 2
            continue

        # Quote toggles: single quotes don't toggle inside double, and
        # vice versa (matches bash semantics).
        if c == "'" and not in_double:
            in_single = not in_single
            cur.append(c)
            i += 1
            continue
        if c == '"' and not in_single:
            in_double = not in_double
            cur.append(c)
            i += 1
            continue

        # Inside any quotes: literal byte, never a separator.
        if in_single or in_double:
            cur.append(c)
            i += 1
            continue

        # Two-char separators first (`&&`, `||`).
        if c == "&" and i + 1 < n and command[i + 1] == "&":
            segments.append("".join(cur))
            cur = []
            i += 2
            continue
        if c == "|" and i + 1 < n and command[i + 1] == "|":
            segments.append("".join(cur))
            cur = []
            i += 2
            continue

        # One-char separators (`;`, `|`, `&`). The `&&` / `||` two-char
        # forms are handled above and consume both chars first, so the
        # single-char branch only fires for true backgrounding `&` /
        # sequence `;` / pipe `|`.
        if c == ";" or c == "|" or c == "&":
            segments.append("".join(cur))
            cur = []
            i += 1
            continue

        cur.append(c)
        i += 1

    segments.append("".join(cur))
    return segments


def _segment_passes_phase_rule(segment: str, policy: "_BashPolicy") -> tuple:
    """Decide whether a single chain segment is allowed under `policy`.

    Returns (ok, reason). The phase-agnostic patterns (read-only +
    lifecycle ops) and the activation special case are checked first;
    only when both fail does the per-phase strict/permissive rule apply.
    """
    s = segment.strip()
    if not s:
        return True, ""  # empty (trailing `&&`) — no restriction

    if _segment_is_phase_agnostic(s):
        return True, ""
    if "export AR_TASK_DIR=" in s:
        return True, ""

    if policy.mode == "strict":
        # AR-script required entries (`.py`) are matched against the
        # actual python invocation via parse_invoked_ar_script — NOT via
        # raw substring. A naive `req in s` would let smuggling like
        # `python -c "print('create_plan.py')"` pass: the string
        # 'create_plan.py' appears in the command yet no AR script runs.
        # Non-script entries (e.g. "export AR_TASK_DIR=" for INIT) are
        # not python files; keep substring match for those.
        invoked = parse_invoked_ar_script(s)
        for req in policy.required:
            if req.endswith(".py"):
                if invoked == req:
                    return True, ""
            elif req in s:
                return True, ""
        required_txt = sorted(policy.required) or "(no user bash legal here; only file edits)"
        return False, f"segment {s!r}: allowed = {required_txt}"

    # permissive
    for b in policy.banned:
        if b in s:
            return False, f"segment {s!r}: '{b}' is blocked here"
    return True, ""


def check_bash(phase: str, command: str) -> tuple:
    """Return (allowed: bool, reason: str) for a Bash command at `phase`.

    Decision order:
      1. Global bans (subprocess-only scripts, `git commit`) — checked
         against the full command first; a chain that smuggles a banned
         substring anywhere is rejected.
      2. Per-chain-segment phase rules: split on bash chain separators,
         every
         segment must independently pass either the agnostic-shortcut
         (read-only + lifecycle), the activation special case, or the
         per-phase strict/permissive rule. The previous implementation
         applied phase rules to the whole command, so a strict phase
         like BASELINE accepted `python baseline.py && python pipeline.py`
         (the "baseline.py" required substring matched the WHOLE chain
         and let pipeline.py ride along). Per-segment evaluation closes
         that hole.
    """
    for ban, why in _GLOBAL_BASH_BANS.items():
        if ban in command:
            return False, f"'{ban}' — {why}"
    if "git commit" in command:
        return False, ("manual 'git commit' forbidden — commits are produced "
                       "by pipeline.py via keep_or_discard")

    policy = _BASH_RULES.get(phase)
    if policy is None:
        return False, f"unknown phase {phase!r}"

    segments = _split_bash_chain(command)
    for seg in segments:
        ok, reason = _segment_passes_phase_rule(seg, policy)
        if not ok:
            return False, f"phase {phase}: {reason}"
    return True, ""


_DIAGNOSE_ARTIFACT_RE = re.compile(r"^\.ar_state/diagnose_v\d+\.md$")


def check_edit(phase: str, rel_path: str, editable_files) -> tuple:
    """Return (allowed: bool, reason: str) for an Edit/Write on `rel_path`
    (task-dir-relative, forward-slash form) at `phase`.

    Writes under .ar_state/ are restricted to a precise allowlist. Phase,
    progress, history, plan.md, heartbeat, and markers are all machine-
    maintained — letting Claude Edit them would let the model skip phases,
    rewrite counters, or forge history. Three paths are agent-writable:
      - .ar_state/plan_items.xml: the XML input file /autoresearch hands to
        create_plan.py (see .claude/commands/autoresearch.md).
      - .ar_state/ranking.md: the FINISH-phase summary (phase-gated).
      - .ar_state/diagnose_v<N>.md: the DIAGNOSE-phase artifact. The
        ar-diagnosis subagent is the intended writer (per the prompt
        contract), but hook payloads do NOT distinguish main agent from
        subagent — provenance is not enforced. Only the artifact's
        CONTENT (sections, marker, R<n> citations) is validated, and only
        writable while phase=DIAGNOSE.
    """
    if rel_path.startswith(".ar_state/"):
        if rel_path == f".ar_state/{PLAN_ITEMS_FILE}":
            return True, ""
        if rel_path == ".ar_state/ranking.md":
            if phase == FINISH:
                return True, ""
            return False, (
                "ranking.md is only writable in the FINISH phase — "
                "finish the optimization loop first."
            )
        if _DIAGNOSE_ARTIFACT_RE.match(rel_path):
            if phase == DIAGNOSE:
                return True, ""
            return False, (
                f"{rel_path!r} is the DIAGNOSE artifact and is only "
                f"writable while phase=DIAGNOSE."
            )
        if rel_path == f".ar_state/{PLAN_FILE}":
            return False, (
                f"plan.md is machine-generated — never hand-edit it. Write "
                f"your <items>...</items> XML to .ar_state/{PLAN_ITEMS_FILE} "
                f"with the Write tool, then run "
                f"`python .autoresearch/scripts/create_plan.py \"<task_dir>\"`."
            )
        return False, (
            f"{rel_path!r} is machine-maintained state. Only "
            f".ar_state/{PLAN_ITEMS_FILE} (plan input), .ar_state/ranking.md "
            f"(FINISH summary), and .ar_state/diagnose_v<N>.md (DIAGNOSE "
            f"artifact) are writable under .ar_state/; everything else is "
            f"owned by hooks and scripts."
        )

    allowed_classes = _EDIT_RULES.get(phase, set())
    if "ref" in allowed_classes and rel_path == "reference.py":
        return True, ""
    if "editable" in allowed_classes and rel_path in set(editable_files or ()):
        return True, ""

    return False, f"phase {phase} does not allow writing {rel_path!r}"


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

def compute_next_phase(task_dir: str) -> str:
    """After a pipeline round finishes, mechanically determine the next phase.

    `eval_rounds >= max_rounds` is the only legitimate FINISH trigger; the
    `not progress` branch is an error fallback for unrecoverable state.
    """
    progress = load_progress(task_dir)
    if not progress:
        return FINISH  # error fallback: corrupt/missing progress.json

    consecutive_failures = progress.get("consecutive_failures", 0)
    eval_rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 999)

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

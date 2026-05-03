"""phase_machine package — facade over four single-concern submodules.

Layout (single-responsibility cuts the previous 1170-line phase_machine.py
made impossible to navigate):

    state_store   — phase enum constants, .ar_state file I/O, task_dir
                    pointer, JSON-tail parser. No internal deps.
    validators    — placeholder detection, validate_reference,
                    validate_kernel, plan.md parser, validate_plan.
                    Depends on state_store.
    phase_policy  — _BASH_RULES, _EDIT_RULES, check_bash, check_edit,
                    compute_next_phase, compute_resume_phase, script
                    invocation parser. Depends on state_store + validators.
    guidance      — get_guidance and the XML schema example shared by
                    PLAN/DIAGNOSE/REPLAN. Depends on state_store +
                    validators (no dep on phase_policy).

Why a package over a flat file split: with the package layout the
file tree itself shows the dependency direction — `state_store.py`
sits next to its callers, but in a clearly separate module. Adding a
new phase or a new validator no longer enlarges a god-module; it
either adds a small file or extends one of the four named layers.

This `__init__.py` re-exports everything so the 17 existing importers
of `phase_machine` continue to work without modification. New code may
prefer importing from the canonical sub-module
(`from phase_machine.state_store import load_progress`).

`auto_rollback` moved to git_utils because it is a pure git operation;
re-exported here for backward compatibility (one previous caller imported
it from phase_machine).
"""
# fmt: off
from .state_store import (
    # Phase constants
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH, ALL_PHASES,
    # File constants
    PHASE_FILE, PROGRESS_FILE, HISTORY_FILE, PLAN_FILE, PLAN_ITEMS_FILE,
    EDIT_MARKER_FILE, PENDING_SETTLE_FILE, HEARTBEAT_FILE, ACTIVE_TASK_FILE,
    DIAGNOSE_ARTIFACT_TEMPLATE, DIAGNOSE_MARKER_TEMPLATE, DIAGNOSE_ATTEMPTS_CAP,
    # Path builders
    state_path, plan_path, progress_path, history_path, edit_marker_path,
    pending_settle_path,
    diagnose_artifact_path, diagnose_marker,
    # Phase I/O
    read_phase, write_phase,
    # Progress + history I/O
    load_progress, save_progress, append_history, update_progress,
    # Active-task pointer
    get_task_dir, set_task_dir, touch_heartbeat,
    # Helpers
    parse_last_json_line,
)
from .validators import (
    KERNEL_PLACEHOLDER, REFERENCE_PLACEHOLDER_PREFIX,
    is_placeholder_file,
    validate_reference, validate_kernel, validate_plan, validate_diagnose,
    DiagnoseState, diagnose_state,
    get_plan_items, has_pending_items, get_active_item,
    # Internal — re-exported so debug / extension scripts that previously
    # reached into phase_machine can still find them at the old name.
    _PLAN_ITEM_RE, _PLAN_TAG_RE, _REF_RUNCHECK_SCRIPT,
)
from .phase_policy import (
    parse_script_name, parse_script_names, parse_invoked_ar_script,
    check_bash, check_edit,
    compute_next_phase, compute_resume_phase,
    # Policy tables — public-ish because tests / dashboards reference
    # them. Underscore-prefixed for a "do not mutate at runtime" hint
    # rather than for true privacy.
    _BASH_RULES, _EDIT_RULES, _GLOBAL_BASH_BANS, _PHASE_AGNOSTIC_PATTERNS,
    _BashPolicy,
)
from .guidance import (
    get_guidance,
    # _load_config_safe is consumed by hook_post_edit (only external user
    # of the helper today). Re-exported even though it's underscore-prefixed
    # because the call site predates the package split.
    _load_config_safe,
    # XML schema and field rules — referenced by name in create_plan.py's
    # docstring and used by tests / dashboards that want to render the
    # canonical example. Re-exported to keep the old `phase_machine.
    # _PLAN_XML_EXAMPLE` reference resolvable.
    _PLAN_XML_EXAMPLE, _PLAN_FIELD_RULES,
)

# auto_rollback used to live in phase_machine; the implementation moved to
# git_utils alongside commit_in_task / ensure_git_identity. Re-export so
# stale `from phase_machine import auto_rollback` still resolves.
import os
import sys
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from git_utils import auto_rollback  # noqa: E402
# fmt: on

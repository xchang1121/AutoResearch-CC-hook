"""Phase-specific guidance — what the LLM should do next.

`get_guidance(task_dir)` is the only public API; it reads phase + progress
+ task config + plan, then returns the `[AR Phase: …]` message that hooks
inject into Claude's context after every state-changing event.

The XML schema example for plan creation (`_PLAN_XML_EXAMPLE`) and the
field-rules tail (`_PLAN_FIELD_RULES`) live here — they're prompt content
shared between PLAN, DIAGNOSE, and REPLAN guidance.
"""
import json
import os
import sys
from typing import Optional

from .state_store import (
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH,
    PLAN_ITEMS_FILE, DIAGNOSE_ATTEMPTS_CAP,
    diagnose_artifact_path, diagnose_marker,
    history_path, load_progress, read_phase, state_path,
    _PROJECT_ROOT,
)
from .validators import get_active_item, diagnose_state


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
    "<![CDATA[...]]>)."
)


def _create_plan_instruction(task_dir: str) -> str:
    """Common 'how to invoke create_plan.py' block used by PLAN, DIAGNOSE,
    and REPLAN guidance. Emits the canonical two-step flow:

      1. Write XML to the FIXED path .ar_state/plan_items.xml.
      2. Run create_plan.py with just <task_dir> — it reads from that path.

    The fixed path eliminates the LLM-drift class where the model wrote
    to one path and then passed a different `@<path>` to create_plan
    (most often a hallucinated /tmp/... or a typoed task subdir).
    """
    xml_path = state_path(task_dir, PLAN_ITEMS_FILE)
    return (
        f"To create the plan, do EXACTLY these two steps:\n"
        f"  1. Use the Write tool to write your <items>...</items> XML to:\n"
        f"       {xml_path}\n"
        f"     (Path is fixed — do NOT invent a different path, do NOT use "
        f"/tmp/, do NOT pass it as a CLI arg later. The Write tool is the "
        f"only thing that touches this path.)\n"
        f"  2. Run:\n"
        f"       python .autoresearch/scripts/create_plan.py \"{task_dir}\"\n"
        f"     (No second argument. The script reads .ar_state/{PLAN_ITEMS_FILE} "
        f"automatically. Adding `@/some/path` reintroduces the drift this "
        f"two-step form exists to prevent.)\n"
        f"\n"
        f"XML schema (write this exact shape to the file in step 1):\n"
        f"{_PLAN_XML_EXAMPLE}\n"
        f"{_PLAN_FIELD_RULES}\n"
    )


def _load_config_safe(task_dir: str):
    """Load TaskConfig, return None on any failure.

    task_config lives in scripts/ root (one level up from this package);
    insert the parent dir into sys.path so the import resolves no matter
    who's importing us.
    """
    try:
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


def _skill_dir_for_dsl(dsl) -> Optional[str]:
    """Resolve the on-disk skills/<...> directory for `dsl`, or None if
    no such directory exists.

    The skills/ tree uses dash-separated names (`skills/triton-ascend/`,
    `skills/cuda-c/`) while the canonical DSL strings use underscores
    (`triton_ascend`, `cuda_c`). Without this translation, the prompts
    that previously read `Glob skills/{dsl}/**/*.md` produced **zero**
    matches at runtime — agents dutifully ran the Glob, got nothing back,
    and silently skipped the skill-reading step. This was the original
    "agent looks like it's reading skills but actually isn't" trap.

    Returns the directory NAME (relative to skills/), not a full path —
    callers want it for the prompt's Glob pattern. None when the DSL has
    no skills tree at all (e.g. ascendc / swft / torch / tilelang_npuir),
    in which case `_skills_hint` returns "" instead of pointing the agent
    at a dead path.
    """
    if not dsl:
        return None
    candidate = dsl.lower().replace("_", "-")
    if os.path.isdir(os.path.join(_PROJECT_ROOT, "skills", candidate)):
        return candidate
    # A few DSLs may legitimately not have a skills tree yet. Don't fall
    # back to the underscore form — that's the broken historical path.
    return None


def _skills_hint(dsl) -> str:
    """Recommend reading DSL skills when authoring plan items.

    Used by PLAN and REPLAN (parent-voice — the parent agent reads skills
    directly and writes the plan). DIAGNOSE has its own inline skills
    section because the subagent's framing differs: it's diagnosing
    failures, not opening a plan, and the prompt wording reflects that.
    Returns "" when the DSL has no skills directory so callers can
    interpolate unconditionally.
    """
    skill_dir = _skill_dir_for_dsl(dsl)
    if not skill_dir:
        return ""
    return (
        f"\nDSL skills: Glob skills/{skill_dir}/**/*.md, then Read 1-3 "
        f"SKILL.md files whose frontmatter description / keywords match "
        f"a candidate plan-item direction. Cite SKILL ids in the item "
        f"rationale."
    )


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
        # Retry detection: progress.json only exists once _baseline_init.py
        # has run, and hook_post_bash only demotes back to GENERATE_KERNEL
        # when seed_metric is None (compile/profile failed) or
        # baseline_correctness is False (numerical mismatch). On the first
        # entry progress is None, so this is a clean signal.
        is_retry = bool(progress) and (
            progress.get("seed_metric") is None
            or progress.get("baseline_correctness") is False
        )
        if is_retry:
            retry_reason = (
                "seed kernel produced no timing (compile/profile failed)"
                if progress.get("seed_metric") is None
                else "seed kernel ran but failed correctness vs reference"
            )
            header = f"[AR Phase: GENERATE_KERNEL — retry, prior seed failed: {retry_reason}]"
            verb = "Generate a corrected"
        else:
            header = "[AR Phase: GENERATE_KERNEL]"
            verb = "Generate an initial"

        description = config.description if config else "(no description)"
        target_file = editable[0] if editable else "kernel.py"
        editable_line = (
            f"Editable files: {', '.join(editable)}\n" if editable else ""
        )
        constraints_part = ""
        if config and getattr(config, "constraints", None):
            # constraints is {metric: (op_str, threshold)} — render compactly
            constraint_strs = [
                f"{m}{op}{thr}" for m, (op, thr) in config.constraints.items()
            ]
            constraints_part = f" | constraints: {', '.join(constraint_strs)}"

        retry_block = ""
        if is_retry:
            retry_block = (
                "\nThis is a retry. baseline.py just printed structured failure "
                "signals above (UB overflow / aivec trap / OOM / correctness "
                f"mismatch / ...). Read that output, then read the current "
                f"{task_dir}/{target_file} to see what failed. Use the skills "
                "Glob above to find a SKILL.md whose description matches the "
                "failure kind before rewriting. Do NOT rewrite from scratch "
                "unless the failure is structural — incremental fixes converge "
                "faster.\n"
            )

        return (
            f"{header} {verb} kernel from reference.\n"
            f"Task: {description}\n"
            f"DSL: {dsl} | primary metric: {primary_metric}{constraints_part}\n"
            f"{editable_line}"
            f"\n"
            f"Read {task_dir}/reference.py, then write to {task_dir}/{target_file}.\n"
            f"Must contain: class ModelNew (can inherit from Model)."
            f"{_skills_hint(dsl)}\n"
            f"{retry_block}"
            f"\n"
            f"Start simple — the autoresearch loop will iterate from here."
        )

    if phase == BASELINE:
        return (f"[AR Phase: BASELINE] Run: "
                f"python .autoresearch/scripts/baseline.py \"{task_dir}\"{worker_flag}")

    if phase == PLAN:
        metric_hint = ""
        if progress:
            baseline = progress.get("baseline_metric")
            if baseline is not None:
                metric_hint = f" Baseline {primary_metric}: {baseline}."

        return (f"[AR Phase: PLAN] "
                f"Read task.yaml, editable files ({editable}), and reference.py.{_skills_hint(dsl)}{metric_hint}\n"
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"\n"
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
        # Pre-bake the recent-rounds summary INTO the subagent prompt so the
        # subagent has it without spending a tool call re-reading
        # history.jsonl. The full file stays in the read list for deeper
        # digs (full traces / older rounds).
        hpath = history_path(task_dir)
        fail_summary = ""
        last_3_fail_rounds = []
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
            # Pull last 3 FAILs across the WHOLE history (not just last 5
            # records) — these are the rounds the subagent must reference
            # by R<n> token.
            all_recs = []
            for l in lines:
                try:
                    all_recs.append(json.loads(l))
                except Exception:
                    pass
            last_3_fail_rounds = [
                r.get("round") for r in all_recs
                if r.get("decision") == "FAIL" and r.get("round") is not None
            ][-3:]

        # Compact metric snapshot — saves the subagent from reading
        # history.jsonl just to answer "how big a delta do we need?".
        metric_line = ""
        if progress:
            seed = progress.get("seed_metric")
            base = progress.get("baseline_metric")
            best = progress.get("best_metric")
            if any(v is not None for v in (seed, base, best)):
                metric_line = (
                    f"\nMetrics ({primary_metric}): "
                    f"seed={seed} | ref_baseline={base} | current_best={best}"
                )
        # Single source of plan_version + per-pv attempt counter (also
        # validates the artifact, but the result is unused here — accepting
        # the small extra read so all callers go through the same helper).
        ds = diagnose_state(task_dir, progress=progress) if progress else None
        plan_version = ds.plan_version if ds else 0
        attempts = ds.attempts if ds else 0

        arch = (config.arch if config and config.arch else "<unknown>")
        backend = (config.backend if config and config.backend else "<unknown>")
        editable_list = ", ".join(editable)
        # Resolve the on-disk skills/<...> dir name (dash form). May be
        # None if this DSL has no curated skills tree, in which case the
        # whole skills section is dropped from the subagent prompt.
        skill_dir = _skill_dir_for_dsl(dsl)

        # Skills section is conditional — `skill_dir` is None when this DSL
        # has no curated skills tree (ascendc / swft / torch / tilelang_npuir
        # at time of writing). Without the conditional we'd hand the agent a
        # Glob pattern that returns zero matches, and they'd silently skip
        # the skill-reading step — defeating the whole point of the section.
        if skill_dir:
            skills_block = (
                f"Read DSL skills (curated {dsl} knowledge — use it to "
                f"ground fix directions in known-good patterns for this "
                f"hardware):\n"
                f"  - Glob skills/{skill_dir}/**/*.md, then Read 1-3 "
                f"SKILL.md files whose frontmatter description / keywords "
                f"match a candidate fix direction.\n"
                f"  - Cite SKILL ids in the rationale of items you "
                f"propose.\n\n"
            )
            scope_constraint = (
                f"  - Glob / Grep ONLY under skills/{skill_dir}/. The 4 "
                f"task files plus that skills subtree are the entire scope.\n"
            )
            cite_clause = " Cite SKILL ids where relevant."
        else:
            skills_block = ""
            scope_constraint = (
                "  - Do NOT Glob / Grep the wider codebase. The 4 task "
                "files are the entire scope (no curated skills tree exists "
                f"for dsl={dsl}).\n"
            )
            cite_clause = ""

        # Artifact contract — the host validates these literals after the
        # Task call returns. See validators.validate_diagnose.
        artifact_path = diagnose_artifact_path(task_dir, plan_version)
        marker = diagnose_marker(plan_version)
        last_3_str = (
            ", ".join(f"R{r}" for r in last_3_fail_rounds)
            if last_3_fail_rounds else "(no FAIL rounds yet — cite the FAILs you find in history.jsonl)"
        )

        # Pre-baked subagent prompt. Parent passes this verbatim to the Agent
        # tool so the subagent doesn't improvise (an earlier open-ended brief
        # sent it grepping git log for 100+ tool calls before timing out).
        subagent_prompt = (
            f"Diagnose why the current optimization rounds are failing, then "
            f"Write a structured report to a fixed path.\n\n"
            f"Target: dsl={dsl} backend={backend} arch={arch}{metric_line}\n"
            f"plan_version={plan_version}\n\n"
            f"Recent rounds (pre-baked from history.jsonl — read the file "
            f"itself only if you need full traces / older rounds):\n"
            f"{fail_summary or '  (none settled yet)'}\n"
            f"Read these task files for context:\n"
            f"  - {task_dir}/reference.py\n"
            f"  - {task_dir}/{editable_list}\n"
            f"  - {task_dir}/.ar_state/plan.md\n"
            f"  - {task_dir}/.ar_state/history.jsonl (focus on the last "
            f"~10 rounds; older entries are usually stale)\n\n"
            f"{skills_block}"
            f"Hard constraints:\n"
            f"  - Do NOT search git history (`git log` / `git show` / "
            f"`git grep`) — per-round commits carry no keyword signal and "
            f"burn tool calls.\n"
            f"{scope_constraint}"
            f"  - Stop after at most 12 tool uses.\n"
            f"  - Write tool may ONLY target the artifact path below. Do "
            f"NOT Write kernel.py, plan.md, or anywhere else.\n\n"
            f"REQUIRED OUTPUT — your final action MUST be a Write call to "
            f"this exact path:\n"
            f"  {artifact_path}\n\n"
            f"The file body must contain ALL of:\n"
            f"  - heading section 'Root cause' (one paragraph; cite each "
            f"of {last_3_str} by its R<n> token)\n"
            f"  - heading section 'Fix directions' (≤3 STRUCTURALLY "
            f"different approaches: algorithmic / fusion / memory layout "
            f"/ data movement; NOT parameter tuning.{cite_clause})\n"
            f"  - heading section 'What to avoid' (≤3 patterns to NOT "
            f"repeat)\n"
            f"  - the magic marker line on its own line at the end:\n"
            f"      {marker}\n"
            f"Total ≤ 300 words across the three sections. The host "
            f"validates path + marker + sections + R<n> citations after "
            f"this Task call returns; missing any element will force a "
            f"retry."
        )
        retry_note = ""
        if attempts > 0:
            retry_note = (
                f"\nThis is DIAGNOSE attempt {attempts + 1}/"
                f"{DIAGNOSE_ATTEMPTS_CAP}. The previous artifact was "
                f"missing or malformed — re-issue Task and ensure the "
                f"subagent ends its work with a Write of the marker line."
            )
        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Step 1 (mandatory, only legal action right now): call the "
                f"Task tool with subagent_type='ar-diagnosis' and this "
                f"EXACT prompt. Do not paraphrase. Do not add or remove "
                f"constraints. Do not Edit, Write, or Bash before this "
                f"Task call.\n"
                f"---BEGIN SUBAGENT PROMPT---\n"
                f"{subagent_prompt}\n"
                f"---END SUBAGENT PROMPT---\n"
                f"\n"
                f"Step 2 (after Task returns AND the host confirms the "
                f"artifact validated): create a NEW plan with >= 3 items "
                f"using {artifact_path} as input.\n"
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"\n"
                f"Items must be diverse: max 1 parameter-tuning item, rest "
                f"must be structural changes. Then sync TodoWrite.\n"
                f"\n"
                f"Artifact contract: the host gates Step 2 on a valid "
                f"{os.path.basename(artifact_path)} (path + marker + 3 "
                f"sections + R<n> citations). Up to "
                f"{DIAGNOSE_ATTEMPTS_CAP} Task attempts are allowed; after "
                f"that the gate is relaxed and you must write "
                f"plan_items.xml directly (manual-planning fallback) "
                f"before running create_plan.py — the DIAGNOSE phase "
                f"still requires a new plan, just without subagent help.{retry_note}")

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
                f"Read .ar_state/history.jsonl. Analyze what worked/failed.{_skills_hint(dsl)}\n"
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"{retry_hint}")

    if phase == FINISH:
        best = progress.get("best_metric") if progress else "?"
        baseline = progress.get("baseline_metric") if progress else "?"
        return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} (baseline: {baseline}). "
                f"Write .ar_state/ranking.md summary. Report to user.")

    return f"[AR Phase: {phase}] Unknown phase."

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
    PLAN_ITEMS_FILE,
    history_path, load_progress, read_phase, state_path,
)
from .validators import get_active_item


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

        arch = (config.arch if config and config.arch else "<unknown>")
        backend = (config.backend if config and config.backend else "<unknown>")
        editable_list = ", ".join(editable)

        # Pre-baked subagent prompt. Parent passes this verbatim to the Agent
        # tool so the subagent doesn't improvise (an earlier open-ended brief
        # sent it grepping git log for 100+ tool calls before timing out).
        # Skills under skills/<dsl>/ are now in scope — they're DSL-specific
        # curated knowledge of what works on this hardware, exactly the
        # input DIAGNOSE needs to propose structurally new directions.
        subagent_prompt = (
            f"Diagnose why the current optimization rounds are failing.\n\n"
            f"Target: dsl={dsl} backend={backend} arch={arch}{metric_line}\n\n"
            f"Recent rounds (pre-baked from history.jsonl — read the file "
            f"itself only if you need full traces / older rounds):\n"
            f"{fail_summary or '  (none settled yet)'}\n"
            f"Read these task files for context:\n"
            f"  - {task_dir}/reference.py\n"
            f"  - {task_dir}/{editable_list}\n"
            f"  - {task_dir}/.ar_state/plan.md\n"
            f"  - {task_dir}/.ar_state/history.jsonl (focus on the last "
            f"~10 rounds; older entries are usually stale)\n\n"
            f"Read DSL skills (curated {dsl} knowledge — use it to ground "
            f"fix directions in known-good patterns for this hardware):\n"
            f"  - Glob skills/{dsl}/**/*.md, then Read 1-3 SKILL.md files "
            f"whose frontmatter description / keywords match a candidate "
            f"fix direction.\n"
            f"  - Cite SKILL ids in the rationale of items you propose.\n\n"
            f"Hard constraints:\n"
            f"  - Do NOT search git history (`git log` / `git show` / "
            f"`git grep`) — per-round commits carry no keyword signal and "
            f"burn tool calls.\n"
            f"  - Glob / Grep ONLY under skills/{dsl}/. The 4 task files "
            f"plus that skills subtree are the entire scope.\n"
            f"  - Stop after at most 12 tool uses; output what you have if "
            f"you can't fully conclude.\n\n"
            f"Produce a tight report (<300 words total) with three sections:\n"
            f"  1. Root cause: one paragraph on what's making rounds fail.\n"
            f"  2. Fix directions: at most 3 STRUCTURALLY different "
            f"approaches (algorithmic change / fusion / memory layout / "
            f"data movement). One sentence each. NOT more parameter tuning. "
            f"Cite SKILL ids where relevant.\n"
            f"  3. What to avoid: at most 3 patterns to NOT repeat. One "
            f"sentence each."
        )
        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Spawn a SUBAGENT (Agent tool) with this EXACT prompt — "
                f"do not paraphrase, do not add or remove constraints:\n"
                f"---BEGIN SUBAGENT PROMPT---\n"
                f"{subagent_prompt}\n"
                f"---END SUBAGENT PROMPT---\n"
                f"After the subagent returns, create a NEW plan with >= 3 items.\n"
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"\n"
                f"Items must be diverse: max 1 parameter-tuning item, rest "
                f"must be structural changes. Then sync TodoWrite.")

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
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"{retry_hint}")

    if phase == FINISH:
        best = progress.get("best_metric") if progress else "?"
        baseline = progress.get("baseline_metric") if progress else "?"
        return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} (baseline: {baseline}). "
                f"Write .ar_state/ranking.md summary. Report to user.")

    return f"[AR Phase: {phase}] Unknown phase."

# Claude AutoResearch

An iterative optimization framework powered by Claude Code.

## What this is

Claude Code acts as the LLM agent, executing a plan → edit → eval → keep/discard loop
to optimize code against a measurable metric. Fully standalone with zero external
dependencies beyond Python + PyYAML.

## Quick Start

```bash
# 1. Open this project in Claude Code
cd claude-autoresearch
claude

# 2. Start a task (init + run is one command).
#    Drop the source ref/kernel files into workspace/ first, named
#    workspace/<op_name>_ref.py and workspace/<op_name>_kernel.py.
/autoresearch --ref workspace/<op_name>_ref.py --op-name <op_name> --backend cuda

# 3. Resume later
/autoresearch --resume

# 4. Monitor (in a separate terminal)
python .autoresearch/scripts/dashboard.py <task_dir> --watch
```

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/autoresearch` | The only slash command — scaffold, resume, or run the loop. Single entry point. |

Failure diagnosis runs as the `DIAGNOSE` phase (triggered automatically by the hook machine after 3 consecutive FAIL rounds). Progress reporting lives in the `dashboard.py` script, run in a separate terminal.

For long unattended runs, wrap in `/loop` self-paced mode: `/loop /autoresearch --resume`. Fixed-interval loops aren't useful here because phase duration varies wildly (PLAN seconds vs EDIT+eval minutes).

## Remote Worker

For eval on remote hardware (e.g. Ascend NPU), add to task.yaml:

```yaml
worker:
  urls:
    - 127.0.0.1:9111
```

Or pass `--worker-url 127.0.0.1:9111` directly to `/autoresearch` on init.

## Skills Library

`skills/` contains 88 optimization knowledge documents organized by DSL/backend:

```
skills/
  triton-ascend/   — Triton on Ascend NPU (guides + cases)
  triton-cuda/     — Triton on CUDA GPU (guides + cases)
  cuda-c/          — CUDA C kernels
  cpp/             — CPU C++ optimization
  tilelang-cuda/   — TileLang DSL
  pypto/           — PyTorch operator patterns (cases)
```

During the PLAN phase, use Glob to find relevant skills by DSL/backend:
```
Glob("skills/triton-ascend/**/*.md")
```
Then Read the SKILL.md files that match your optimization direction. Each SKILL.md has YAML frontmatter with category, description, and keywords.

## Invariants (hook-driven flow)

Step-by-step actions are injected by hooks as `[AR Phase: ...]` messages.
The text is generated INTERNALLY by hooks and emitted on stderr; you only
consume those messages. Do not try to fetch guidance yourself — there is
no CLI for it, `phase_machine.py` is a library and invoking it as a script
is rejected by `hook_guard_bash`. If you have not seen a fresh
`[AR Phase: ...]` message, wait for one (run the next legal command and
the hook will emit guidance), don't try to "refresh" it manually.

Do not try to memorize the flow — follow the latest hook message. The
following invariants are non-negotiable:

1. **`.ar_state/plan.md` is the source of truth.** Only `create_plan.py` and
   `settle.py` / `pipeline.py` may write to it. Never hand-edit `plan.md`.
   TodoWrite is a UI mirror projected from `plan.md` by hooks — not a
   substitute.
2. **Plan item IDs are globally monotonic.** `p1, p2, p3, ...` allocated from
   a single counter in `progress.json.next_pid`. Never reuse IDs, never skip.
3. **Every `pN` must end in KEEP / DISCARD / FAIL.** When DIAGNOSE or REPLAN
   supersedes a plan with pending items, `create_plan.py` auto-settles them
   as `DISCARD (superseded by replan vN)` and records to `history.jsonl`. No
   item may vanish without an outcome row.
4. **Phase transitions are hook-controlled.** Never write `.ar_state/.phase`
   manually and never "guess the next step" — wait for the hook's guidance.
5. **Editable files are scoped by `task.yaml.editable_files`.** Editing
   anything else is rejected by `hook_guard_edit.py`.
6. **After a session break, resume with `/autoresearch --resume`.** Do not
   patch state files to recover.
7. **`create_plan.py` rejects mean the plan has a real problem** (diversity,
   repeated failure keywords, short rationale). Read the stderr reason and
   rewrite — do not retry the same XML payload. The script consumes an
   `<items>` XML document (chosen over JSON because LLMs hallucinate fewer
   structural errors in tag-delimited text).
8. **TodoWrite sync is mandatory.** When a hook emits `additionalContext`
   with a `TodoWrite payload`, call TodoWrite with that payload verbatim on
   the next turn.

## Dependencies

- Python >= 3.10
- PyYAML (`pip install pyyaml`)
- Claude Code CLI

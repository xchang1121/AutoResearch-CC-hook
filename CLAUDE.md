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

**Always follow the latest `[AR Phase: ...]` message** injected by the hook
(see `phase_machine.get_guidance()`). It tells you exactly which script to run
for the current phase. Do not memorize a phase-to-script mapping; let the hook
drive.

Do not treat `.autoresearch/scripts/` as a command menu. Startup goes through
`/autoresearch`, which may call `scaffold.py` or `resume.py` and then
`export AR_TASK_DIR=...`. After activation, run only the exact script named
by the latest `[AR Phase: ...]` guidance. `dashboard.py` is read-only
inspection; `worker_ctl.py` only manages the worker service. Pipeline
substeps such as `quick_check.py`, `eval_wrapper.py`, `keep_or_discard.py`,
`settle.py`, `_baseline_init.py`, and `code_checker.py` are internal and the
Bash hook blocks direct invocation.

To inspect task state without rerunning anything, read the state files
directly (the hook's `[AR Phase: ...]` message already includes most of
this — only fish around if it doesn't):

- `cat "$AR_TASK_DIR/.ar_state/.phase"` — current phase
- `cat "$AR_TASK_DIR/.ar_state/progress.json"` — full progress (rounds,
  failure counter, baseline / seed / best metrics, plan version)
- `cat "$AR_TASK_DIR/.ar_state/plan.md"` — full plan + settled-history table
- `cat "$AR_TASK_DIR/.ar_state/history.jsonl"` — round-by-round outcomes
- `python .autoresearch/scripts/dashboard.py` — interactive TUI summarizing
  all of the above (separate terminal recommended)

`worker_ctl.py` only manages the worker service; do not look there for state
queries.

The following invariants are non-negotiable:

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
   In particular: **errors during a phase do not authorize replanning.** A
   single FAIL settles the active item via `pipeline.py`; the next pending
   item is promoted automatically. Only the phase machine triggers
   replanning — DIAGNOSE at 3 consecutive FAILs, REPLAN when all items are
   settled. If you call `create_plan.py` from EDIT to "fix" a failed
   approach, the hook blocks you; that is by design.
   When the phase machine does invoke DIAGNOSE / REPLAN, **preserve untried
   pending items** by including them in the new `<items>` document with
   `<reactivate_pid>pN</reactivate_pid>`. Items not listed are auto-DISCARDed
   as superseded — that is unspent budget thrown away. Refining the desc /
   rationale of a carried-forward pid is supported and does not consume a
   fresh pid.
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

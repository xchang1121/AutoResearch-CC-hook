# AutoResearch — Init / Resume / Run Optimization Loop

Single entry point for the whole loop: initialize a new task, resume an existing
one, or kick off the optimization. The hook machine takes it from there.

## Arguments

`$ARGUMENTS` — one of:

- **`--resume`** or **`--resume <task_dir>`** — continue the most recent task
  (or the specified one).
- **Task dir** — resume that specific task: `ar_tasks/my_task_123456_abc`.
- **Init flags** — new task from an existing reference file:
  `--ref <file> --op-name <name>
   [--dsl triton_ascend|triton_cuda|ascendc|cuda_c|cpp|tilelang_cuda|tilelang_npuir|pypto|swft|torch]
   [--framework torch|mindspore|numpy]
   (--devices <N[,M,...]> | --worker-url <host:port>)
   [--kernel <file>] [--max-rounds <N>]`

  **Hardware spec is exactly one of `--devices` (local eval) or
  `--worker-url` (remote).**

  - `--devices 5` → scaffold queries `npu-smi info -i 5` (or `nvidia-smi`,
    `uname -m`) to derive arch. task.yaml gets `devices: [5]`, arch derived.
  - `--worker-url 127.0.0.1:9070` → scaffold GETs `/api/v1/status` on that
    worker, uses its reported backend + arch + devices.
  - `--dsl` is the primary classifier. backend is a pure function of DSL
    (triton_ascend → ascend; cuda_c → cuda; ...). **`--backend` and `--arch`
    are not user flags** — they're auto-derived, never typed.

  Convention: source `--ref` / `--kernel` files live in `workspace/`, named
  `workspace/<op_name>_ref.py` and `workspace/<op_name>_kernel.py`. Put new
  candidates there before invoking `/autoresearch`.
- **Desc mode** — new task from a natural-language description:
  `--desc "fused ReLU + LayerNorm, (32,1024), fp16" --dsl triton_cuda --worker-url ...`

Required init flags: `--ref` (or `--desc`) and `--op-name`. `--output-dir`
defaults to `ar_tasks`.

### Four launch modes

| mode | flags | initial phase |
|------|-------|---------------|
| 1 | `--ref X.py --kernel Y.py` (both source files ready) | `PLAN` (baseline runs first) |
| 2 | `--ref X.py` (reference only, let agent author kernel) | `GENERATE_KERNEL` |
| 3 | `--desc "..."` (prose only) | `GENERATE_REF` → `GENERATE_KERNEL` |
| 4 | `--desc "..." --kernel Y.py` (prose + seed kernel) | `GENERATE_REF` |

`--dsl` applies to all four modes; it controls what adapter drives the
generated verify/profile scripts and which DSL-specific code_checker rules
quick_check enforces.

## Step 1: Decide path

1. `$ARGUMENTS` contains `--resume` → resume most recent (or given) task:
   ```bash
   python .autoresearch/scripts/resume.py [optional_task_dir]
   ```
   The last line of stdout is the task_dir. Non-zero exit ⇒ stop and report
   (likely an incompatible on-disk version).

2. `$ARGUMENTS` is an existing directory → resume it:
   ```bash
   python .autoresearch/scripts/resume.py "$ARGUMENTS"
   ```

3. `$ARGUMENTS` starts with `--` (and is not `--resume`) → scaffold a new task:
   ```bash
   python .autoresearch/scripts/scaffold.py $ARGUMENTS --output-dir ar_tasks --run-baseline
   ```
   `--run-baseline` runs the baseline eval immediately AND writes
   `.ar_state/.phase = PLAN` on success, so when **both `--ref` and `--kernel`
   are provided** there are no user-visible init/baseline steps: the next
   activation drops you straight into PLAN. (`--desc` mode and `--ref` without
   `--kernel` will instead start in GENERATE_REF / GENERATE_KERNEL.) Read the
   `task_dir` from the JSON output.

4. No arguments → ask the user: reference path, op name, **DSL**, worker URL,
   max rounds. Then use path 3. (Ask for DSL by name — e.g. `triton_ascend`,
   `ascendc`, `cuda_c` — not backend.)

## Step 2: Activate

```bash
export AR_TASK_DIR="<task_dir from step 1>"
```

The activation hook prints `[AR Phase: ...]` guidance. Follow it.

## Step 3: Loop

Follow the phase guidance. Never stop between phases.

- **GENERATE_REF / GENERATE_KERNEL** — Write `reference.py` / `kernel.py` with
  the Edit tool (only needed for `--desc` mode or when you skipped `--kernel`).
- **BASELINE** — `python .autoresearch/scripts/baseline.py "$AR_TASK_DIR"`
  (append `--worker-url` if configured). If scaffold already ran baseline,
  this phase is skipped automatically.
- **PLAN / DIAGNOSE / REPLAN** — run
  `python .autoresearch/scripts/create_plan.py "$AR_TASK_DIR" @<path>` after
  writing the XML `<items>` document to `<path>` with the Write tool (see the
  hook guidance for the exact schema — XML is used instead of JSON to reduce
  structural hallucinations). The canonical path is
  `"$AR_TASK_DIR/.ar_state/plan_items.xml"`.

  **Do not** quote multi-line XML inline on the command line — on Windows,
  bash / CreateProcess silently truncates it and the script then emits what
  looks like a schema error (`"missing <desc>"` etc.) even though your XML is
  correct. If you see that error after an inline invocation, stop retrying
  the schema and switch to the `@<path>` form.

  Alternative when writing a file is not convenient: pass `-` as the second
  argument and pipe the XML in on stdin via a single-quoted heredoc.

  When the hook's `additionalContext` gives you a TodoWrite payload, call
  TodoWrite with it verbatim.
- **EDIT** — Edit `kernel.py` (multiple Edit calls OK). When done:
  `python .autoresearch/scripts/pipeline.py "$AR_TASK_DIR"`.
- **FINISH** — Run
  `python .autoresearch/scripts/final_report.py "$AR_TASK_DIR"` to generate
  `.ar_state/report.md` and `.ar_state/report.json` plus `.ar_state/report.png`
  when matplotlib is installed. If matplotlib is unavailable, the script still
  succeeds with the text/JSON report. Read the Markdown report, summarize, stop.

## Rules

- Keep going between phases.
- Hooks block wrong actions and tell you what to do next — read their messages.
- Never hand-edit `plan.md` or `.ar_state/.phase`; always go through the scripts.
- Do not use Bash redirection, `python -c`, PowerShell, or mutating git/filesystem
  commands to write task files; use Edit/Write or the phase scripts so hooks
  can enforce the workflow.

## Scripts under `.autoresearch/scripts/`

Only the following are runnable CLIs. Everything else in that directory is a
**library imported by these scripts and by the hooks** — it has no
`__main__` block and `python .autoresearch/scripts/<name>.py` will be
blocked by the Bash hook with `[AR] Unknown script ...`.

| Use case | Script |
|---|---|
| New task | `scaffold.py` |
| Resume | `resume.py` |
| Baseline eval | `baseline.py` |
| Plan / replan / diagnose | `create_plan.py` |
| One optimization round | `pipeline.py` |
| Final report | `final_report.py` |
| Live status (read-only) | `dashboard.py` |
| Worker service control | `ar_cli.py` |

**Library files — do NOT invoke directly.** If you want to inspect
phase/progress, read the state files or run `dashboard.py`; do not
`python phase_machine.py` etc.

- `phase_machine.py` — phase constants, validators, guidance strings
- `task_config.py`   — yaml loader, eval orchestrator
- `local_worker.py`  — local subprocess executor for verify/profile
- `code_checker.py`  — static checker (called by `quick_check.py`)
- `hook_utils.py`, `hw_detect.py`, `settings.py` — shared helpers
- `hook_guard_*.py`, `hook_post_*.py`, `hook_stop_save.py` — hooks invoked
  by Claude Code itself, never by you

To inspect state instead of running a library:

| Want | Do |
|---|---|
| Current phase | `cat "$AR_TASK_DIR/.ar_state/.phase"` |
| Full progress | `cat "$AR_TASK_DIR/.ar_state/progress.json"` |
| Live dashboard | `python .autoresearch/scripts/dashboard.py` |
| History tail | `cat "$AR_TASK_DIR/.ar_state/history.jsonl"` |

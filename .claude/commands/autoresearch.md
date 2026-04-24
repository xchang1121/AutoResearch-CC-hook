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
   [--backend ascend|cuda|cpu] [--arch <arch>] [--framework torch|mindspore|numpy]
   [--kernel <file>] [--worker-url <host:port>] [--max-rounds <N>]`

  **`--dsl` is the primary pivot**, not `--backend`. It selects which
  ar_vendored adapter drives verify/profile script generation. When `--dsl`
  is omitted, the config.yaml `default_dsl` is used. `--backend` / `--arch` /
  `--framework` are optional overrides of the DSL's preset — if given, they
  must agree with the DSL (e.g. `--dsl triton_ascend --backend cuda` is a
  hard error, no silent correction). See config.yaml `dsls:` for the full
  preset table.

  Convention: source `--ref` / `--kernel` files live in `workspace/`, named
  `workspace/<op_name>_ref.py` and `workspace/<op_name>_kernel.py`. Put new
  candidates there before invoking `/autoresearch`.
- **Desc mode** — new task from a natural-language description:
  `--desc "fused ReLU + LayerNorm, (32,1024), fp16" --dsl triton_cuda`

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
- **FINISH** — Write `.ar_state/ranking.md`, summarize, stop.

## Rules

- Keep going between phases.
- Hooks block wrong actions and tell you what to do next — read their messages.
- Never hand-edit `plan.md` or `.ar_state/.phase`; always go through the scripts.

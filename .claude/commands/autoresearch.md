# AutoResearch — Init / Resume / Run Optimization Loop

Single entry point for the whole loop: initialize a new task, resume an existing
one, or kick off the optimization. The hook machine takes it from there.

## How this command works

This loop is owned by a phase machine, not by you. Each turn is exactly one
Bash call:

1. Read the most recent `[AR Phase: <name>]` message.
2. Run the single command it names — verbatim.
3. Wait for the next `[AR Phase: ...]` message and repeat.

The phase advances itself; the next command is always named for you.

### The full set of user-facing scripts

Every command `[AR Phase: ...]` will ever ask you to run is one of these:

| script            | when it appears                          |
|-------------------|------------------------------------------|
| `scaffold.py`     | Step 1 — new task                        |
| `resume.py`       | Step 1 — resume                          |
| `baseline.py`     | BASELINE phase                           |
| `create_plan.py`  | PLAN / DIAGNOSE / REPLAN                 |
| `pipeline.py`     | EDIT phase (one round of the loop)       |
| `final_report.py` | FINISH phase                             |
| `dashboard.py`    | any phase, read-only                     |
| `worker_ctl.py`   | only when guidance explicitly asks       |

When uncertain about what to type, the answer is already in the most recent
`[AR Phase: ...]` line — re-read it and copy the command it names.

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
   `--kernel` will instead start in GENERATE_REF / GENERATE_KERNEL.)

   On success scaffold prints a JSON line like
   `{"task_dir": "ar_tasks/<op>_<ts>_<id>", ...}`. The `task_dir` value from
   that JSON is the only thing Step 2 needs — copy it into the export.

4. No arguments → ask the user: reference path, op name, **DSL**, worker URL,
   max rounds. Then use path 3. (Ask for DSL by name — e.g. `triton_ascend`,
   `ascendc`, `cuda_c` — not backend.)

## Step 2: Activate

Issue **one** Bash call — a plain export of `AR_TASK_DIR`:

```bash
export AR_TASK_DIR="<task_dir from step 1>"
```

That single line is the entire activation. Claude Code's `PostToolUse` hook
reads `AR_TASK_DIR` from this command, validates the task, computes the
starting phase, and emits the first `[AR Phase: ...]` guidance back to you
on stderr.

**What success looks like:**
- The Bash tool returns with empty stdout.
- A hook message follows containing a line like
  `[AR Phase: PLAN] Read task.yaml, ... Then create the plan: ...`.
- That line is your Step 3 input — copy the command it names verbatim.

If no `[AR Phase: ...]` line appears, re-issue the same one-line export with
the exact quoted path from Step 1's JSON.

## Step 3: Follow the phase machine

Each `[AR Phase: ...]` message names the next command on a single line.
Each turn:

1. Read the `[AR Phase: ...]` message.
2. Run the command it names — verbatim, as one Bash call.
3. Wait for the next `[AR Phase: ...]` message and repeat.

Continue until you receive `[AR Phase: FINISH]`. At FINISH the loop is
complete and the hook will name the wrap-up command (typically
`final_report.py`).

**PLAN / DIAGNOSE / REPLAN convention:** when guidance asks for
`create_plan.py`, write the XML `<items>` document to
`"$AR_TASK_DIR/.ar_state/plan_items.xml"` first, then pass `@<path>` as the
second argument. Passing the XML through a file keeps newlines intact across
bash and Windows `CreateProcess`, where inline multi-line strings get
truncated and surface as a misleading `"missing <desc>"` schema error.

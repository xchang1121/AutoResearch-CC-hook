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

## Step 3: Run the loop

After activation the hook prints `[AR Phase: ...]` with the exact next
command. Follow it. Keep following each new `[AR Phase: ...]` message until
the hook itself emits FINISH guidance. Do not stop between phases and do not
choose your own command sequence — the phase machine owns it.

One gotcha the per-phase guidance can't fix: when PLAN / DIAGNOSE / REPLAN
asks you to run `create_plan.py`, write the XML `<items>` document to a file
first (canonical path: `"$AR_TASK_DIR/.ar_state/plan_items.xml"`) and pass
`@<path>` as the second argument. Quoting multi-line XML inline gets
truncated by bash/CreateProcess on Windows and surfaces as a misleading
`"missing <desc>"` schema error.

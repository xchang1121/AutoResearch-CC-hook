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
   --dsl ascendc|cpp|cuda_c|pypto|swft|tilelang_cuda|tilelang_npuir|torch|triton_ascend|triton_cuda
   [--framework torch|mindspore|numpy]
   (--devices <N[,M,...]> | --worker-url <host:port>)
   [--kernel <file>] [--max-rounds <N>]`

  <!-- DSL list above must stay in lockstep with hw_detect._DSL_BACKEND.
       parse_args.py and scaffold.py derive their copies from
       hw_detect.list_supported_dsls(); this prose copy is the only manual one. -->


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

`--output-dir` defaults to `ar_tasks`.

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

## Step 1: Parse `$ARGUMENTS` deterministically — DO NOT skip this

```bash
python .autoresearch/scripts/parse_args.py $ARGUMENTS
```

The script prints a single-line JSON dispatch record. Read it carefully:

```json
{
  "mode": "scaffold|resume|ask",
  "command": "python ... (verbatim, ready to exec)" or null,
  "values": {ref, desc, op_name, dsl, devices, worker_url, max_rounds, ...},
  "missing": [...]
}
```

**The values in this JSON are the SINGLE SOURCE OF TRUTH for every flag.**
You MUST NOT:

- Modify any flag value before re-emitting it (e.g. don't turn `"devices": "6"`
  into `--devices 0` because a docstring or earlier example used 0).
- Pull "default" values from `.autoresearch/scripts/scaffold.py`'s docstring,
  CLAUDE.md, or memory of past tasks — if a value isn't in `values`, it isn't
  set, and you must use `mode: "ask"` to get it from the user.
- Substitute one device id, dsl, or path for another to "match" a prior task.

This step exists because earlier versions of /autoresearch let the LLM
construct the scaffold bash directly from `$ARGUMENTS`, and the LLM
silently rewrote flag values on retries (e.g. `--devices 6` → `--devices 0`
on a hook-blocked retry, sourced from scaffold's docstring example). The
parser closes that drift.

## Step 2: Dispatch by mode

### `mode == "ask"`

Show the user the `missing` list and ask them to provide the fields. Then
re-invoke `/autoresearch` with the complete flag set. Do not guess values.

### `mode == "resume"`

Run `command` verbatim:

```bash
<paste command field exactly as printed>
```

The last line of stdout is the resolved task_dir. Non-zero exit ⇒ stop and
report (likely an incompatible on-disk version).

### `mode == "scaffold"`

Run `command` verbatim:

```bash
<paste command field exactly as printed>
```

Read the `task_dir` from the JSON output. **Before re-emitting any flag, sanity-check it against `values` in the parser's output**: if your bash command's `--devices` differs from `values.devices`, you have introduced drift — re-issue using the parser's `command` field directly.

`--run-baseline` runs the baseline eval immediately AND writes
`.ar_state/.phase = PLAN` on success, so when **both `--ref` and `--kernel`
are provided** there are no user-visible init/baseline steps: the next
activation drops you straight into PLAN. (`--desc` mode and `--ref` without
`--kernel` will instead start in GENERATE_REF / GENERATE_KERNEL.)

## Step 3: Activate

```bash
export AR_TASK_DIR="<task_dir from step 2>"
```

The activation hook prints `[AR Phase: ...]` guidance on stderr. Follow it.

**Do not** chain anything else onto this command to "also fetch guidance" —
in particular, do not write `export AR_TASK_DIR=... && python .../phase_machine.py get_guidance ...`.
`phase_machine.py` is a library, not a CLI; `hook_guard_bash` will reject
the invocation. The hook emits guidance automatically — read its output
on stderr and proceed.

## Step 4: Loop

Follow the phase guidance. Never stop between phases.

- **GENERATE_REF / GENERATE_KERNEL** — Write `reference.py` / `kernel.py` with
  the Edit tool (only needed for `--desc` mode or when you skipped `--kernel`).
- **BASELINE** — `python .autoresearch/scripts/baseline.py "$AR_TASK_DIR"`
  (append `--worker-url` if configured). If scaffold already ran baseline,
  this phase is skipped automatically.
- **PLAN / DIAGNOSE / REPLAN** — two-step flow, no path transcription:

  1. Use the **Write tool** to write your `<items>...</items>` XML to:
     ```
     $AR_TASK_DIR/.ar_state/plan_items.xml
     ```
     This path is **fixed**. Do not invent a different one (no `/tmp/...`,
     no shell expansion games). The hook's PLAN/DIAGNOSE/REPLAN guidance
     also prints this exact path — copy it from there.

  2. Run:
     ```bash
     python .autoresearch/scripts/create_plan.py "$AR_TASK_DIR"
     ```
     **No second argument.** `create_plan.py` reads from
     `.ar_state/plan_items.xml` automatically. Adding `@/some/path` or an
     inline XML string reintroduces the path-drift class of bugs this
     two-step form exists to prevent (LLM Writes to one path, then passes
     a different path to the script).

  See the hook guidance for the exact XML schema — XML is used instead
  of JSON to reduce structural hallucinations.

  Alternatives (only when the canonical-path form is genuinely unsuitable,
  e.g. an out-of-tree audit script):
  - `create_plan.py "$AR_TASK_DIR" @<path>` reads from `<path>`.
  - `create_plan.py "$AR_TASK_DIR" -` reads XML from stdin.

  When the hook's `additionalContext` gives you a TodoWrite payload, call
  TodoWrite with it verbatim.
- **EDIT** — Edit `kernel.py` (multiple Edit calls OK). When done:
  `python .autoresearch/scripts/pipeline.py "$AR_TASK_DIR"`.
- **FINISH** — Write `.ar_state/ranking.md`, summarize, stop.

## Rules

- Keep going between phases.
- Hooks block wrong actions and tell you what to do next — read their messages.
- Never hand-edit `plan.md` or `.ar_state/.phase`; always go through the scripts.
- **Never** invent flag values not produced by `parse_args.py` in step 1.
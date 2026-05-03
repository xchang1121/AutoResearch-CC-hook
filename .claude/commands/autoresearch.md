# AutoResearch — Init / Resume / Run Optimization Loop

Single entry point: initialize a new task, resume an existing one, or kick
off the optimization. Hooks drive every transition after activation —
follow the latest `[AR Phase: ...]` message and never stop between phases.

## Arguments

`$ARGUMENTS` — one of:

- **`--resume`** or **`--resume <task_dir>`** — continue the most recent task
  (or the specified one).
- **Task dir** — resume that specific task: `ar_tasks/my_task_123456_abc`.
- **Init flags** — new task from an existing reference file:
  ```
  --ref <file> --op-name <name>
  --dsl ascendc|cpp|cuda_c|pypto|swft|tilelang_cuda|tilelang_npuir|torch|triton_ascend|triton_cuda
  [--framework torch|mindspore|numpy]
  (--devices <N[,M,...]> | --worker-url <host:port>)
  [--kernel <file>] [--max-rounds <N>]
  ```
  Hardware spec is exactly one of `--devices` (local) or `--worker-url`
  (remote). `--backend` and `--arch` are auto-derived from `--dsl` +
  hardware probe; never typed by the user.

  Convention: source files live in `workspace/<op_name>_ref.py` and
  `workspace/<op_name>_kernel.py`.

- **Desc mode** — new task from a natural-language description:
  ```
  --desc "fused ReLU + LayerNorm, (32,1024), fp16" --dsl triton_cuda --worker-url ...
  ```

`--output-dir` defaults to `ar_tasks`.

### Four launch modes

| mode | flags | initial phase |
|------|-------|---------------|
| 1 | `--ref X.py --kernel Y.py` (both source files ready) | `PLAN` (baseline runs first) |
| 2 | `--ref X.py` (reference only, agent authors kernel) | `GENERATE_KERNEL` |
| 3 | `--desc "..."` (prose only) | `GENERATE_REF` → `GENERATE_KERNEL` |
| 4 | `--desc "..." --kernel Y.py` (prose + seed) | `GENERATE_REF` |

## Step 1 — Parse `$ARGUMENTS`

```bash
python .autoresearch/scripts/parse_args.py $ARGUMENTS
```

Returns a single-line JSON dispatch record:

```json
{
  "mode": "scaffold|resume|ask",
  "command": "python ... (verbatim, ready to exec)" or null,
  "values": {ref, desc, op_name, dsl, devices, worker_url, max_rounds, ...},
  "missing": [...]
}
```

**`values` is the single source of truth for every flag.** Do not modify,
do not pull defaults from elsewhere, do not substitute. If a value is
missing, dispatch via `mode: "ask"`.

## Step 2 — Dispatch by mode

- **`ask`** — show `missing` to the user; re-invoke `/autoresearch` with
  the complete flag set. Don't guess.
- **`resume`** / **`scaffold`** — run `command` verbatim. Last line of
  stdout is the resolved task_dir. Non-zero exit → stop and report.

For mode 1 (both `--ref` and `--kernel`), scaffold's `--run-baseline`
runs the seed and writes `.phase = PLAN` on success — the next activation
drops straight into PLAN. Modes 2-4 start in GENERATE_REF / GENERATE_KERNEL.

## Step 3 — Activate

```bash
export AR_TASK_DIR="<task_dir from step 2>"
```

The activation hook prints `[AR Phase: ...]` guidance on stderr. Follow it.

## Step 4 — Loop

Follow the phase guidance. Never stop between phases.

- **GENERATE_REF / GENERATE_KERNEL** — Write `reference.py` / `kernel.py`
  with the Edit tool.
- **BASELINE** — `python .autoresearch/scripts/baseline.py "$AR_TASK_DIR"`
  (append `--worker-url` if configured). Skipped automatically if scaffold
  already ran it.
- **PLAN / REPLAN** — two-step plan creation:
  1. Write `<items>...</items>` XML to `$AR_TASK_DIR/.ar_state/plan_items.xml`.
  2. Run `python .autoresearch/scripts/create_plan.py "$AR_TASK_DIR"` (no
     second argument — the script reads the canonical path).
  See the hook guidance for the XML schema. When the hook emits a
  TodoWrite payload, call TodoWrite with it verbatim.
- **DIAGNOSE** — three steps with a hard artifact contract (see below).
- **EDIT** — Edit `kernel.py` (multiple Edit calls OK). When done:
  `python .autoresearch/scripts/pipeline.py "$AR_TASK_DIR"`.
- **FINISH** — Write `.ar_state/ranking.md`, summarize, stop.

### DIAGNOSE flow

The phase exists to produce a new plan. Two paths to that end — preferred
(subagent) and fallback (manual). Stop is blocked the entire time.

**Preferred path (subagent):**

1. Call `Task(subagent_type='ar-diagnosis', ...)` with the prompt the hook
   printed inside `---BEGIN SUBAGENT PROMPT---` / `---END SUBAGENT PROMPT---`
   — verbatim, no paraphrasing. While the artifact is missing/invalid,
   Bash is locked to read-only / lifecycle ops, `create_plan.py` is
   gated on the artifact, Edit is restricted by `check_edit`, and Stop
   is blocked. PreToolUse on Task itself only enforces
   `subagent_type='ar-diagnosis'`.
2. PostToolUse checks `$AR_TASK_DIR/.ar_state/diagnose_v<plan_version>.md`
   for: marker `[AR DIAGNOSE COMPLETE marker_v<plan_version>]`, sections
   `Root cause` / `Fix directions` / `What to avoid`, citations of the
   last 3 FAIL rounds (`R<n>`).
   - Pass → `[AR] DIAGNOSE artifact validated …` → go to step 3.
   - Fail → `additionalContext` says what's missing; re-issue the same
     Task call.
3. Create the new plan: write `plan_items.xml`, run `create_plan.py`
   (same two-step flow as PLAN / REPLAN). Phase advances to EDIT.

**Fallback path (manual planning, after 5 failed Task attempts):**

The subagent path is exhausted but the phase still requires a new plan.
Further `Task` calls are blocked; the artifact gate on `create_plan.py`
is relaxed.

1. Read `.ar_state/history.jsonl` (focus on recent FAIL rows — their
   `description` fields are the raw failure signal) and `.ar_state/plan.md`
   (what's already been tried).
2. Write `<items>...</items>` to `$AR_TASK_DIR/.ar_state/plan_items.xml`.
   Plan must contain ≥3 items, ≥2 structurally different from the prior
   plan (algorithmic / fusion / memory layout / data movement, NOT
   parameter tuning).
3. Run `python .autoresearch/scripts/create_plan.py "$AR_TASK_DIR"`. Only
   create_plan.py's structural validation applies. Phase advances to EDIT.

In DIAGNOSE: do not Edit kernel.py, do not Stop. Bash is locked to
read-only / lifecycle ops + `create_plan.py` (gated on artifact). The
only path forward is `Task → artifact → create_plan` (preferred) or
`manual plan_items.xml → create_plan` (fallback after cap).

Provenance note: the host can't tell main-agent Write from subagent
Write, so the artifact contract is enforced by content (sections +
marker + R<n> citations), not by who wrote it. The subagent path is
preferred because its prompt + read-only tool isolation produce a more
reliable diagnosis, not because the host can prove subagent provenance.

## Rules

- Keep going between phases.
- Hooks block wrong actions and tell you what to do next — read their
  messages.
- Never hand-edit `plan.md` or `.ar_state/.phase`; always go through
  the scripts.
- Never invent flag values not produced by `parse_args.py`.

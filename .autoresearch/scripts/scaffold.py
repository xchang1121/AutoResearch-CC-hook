#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch.

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; required to import + run end-to-end
    on CPU — scaffold gates on `phase_machine.validate_reference`)
  - kernel.py (editable; --kernel writes the user file directly, otherwise
    the canonical KERNEL_PLACEHOLDER from phase_machine — the placeholder
    routes the task to GENERATE_KERNEL on first activation)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # NOTE: --devices values below are placeholders; pass the actual free
    # device id at invocation time. Earlier versions of these examples all
    # used `--devices 0`, which biased the LLM driving /autoresearch into
    # silently rewriting the user's --devices to 0 on hook-blocked retries.
    # parse_args.py is now the single source of truth for flag values.

    # Local eval (arch auto-derived via npu-smi):
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --dsl triton_ascend --devices <DEV>

    # With initial kernel:
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --dsl triton_cuda --devices <DEV>

    # Remote worker (arch fetched from /api/v1/status):
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --dsl triton_ascend --worker-url 127.0.0.1:9111

    # Custom output directory:
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --dsl triton_cuda --devices <DEV> --output-dir /tmp/tasks

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import yaml


# ---------------------------------------------------------------------------
# Reference validation — delegated to the standalone library module so
# phase_machine.validators can call the same rule without importing this
# CLI script. The local re-export keeps callers that imported
# `scaffold.validate_ref` working.
# ---------------------------------------------------------------------------
from ref_ast import validate_ref  # noqa: E402, F401  (re-export)


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    ref_code: str,
    kernel_code: str | None = None,
    op_name: str,
    desc: str = "",
    dsl: str = "",
    framework: str = "torch",
    backend: str = "",
    arch: str = "",
    devices: list | None = None,
    worker_urls: list | None = None,
    max_rounds: int = 20,
    eval_timeout: int = 120,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    code_checker_enabled: bool = True,
    correctness_atol: float = 1e-2,
    correctness_rtol: float = 1e-2,
) -> str:
    """Create task directory with all files. Returns absolute path.

    Mirrors task_scaffolder.scaffold_task_dir
    but with zero external dependency.
    """
    # Determine base directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference.py
    _write(task_dir, "reference.py", ref_code)

    # Write editable file (kernel.py). With no initial kernel, write the
    # canonical TODO placeholder from phase_machine — phase_machine.is_
    # placeholder_file uses the matching predicate, so the routing logic
    # in hooks/scaffold/validators stays in lockstep with this template.
    if kernel_code is not None:
        _write(task_dir, editable_filename, kernel_code)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase_machine import KERNEL_PLACEHOLDER
        _write(task_dir, editable_filename, KERNEL_PLACEHOLDER)

    # Generate task.yaml
    task_yaml = {
        "name": op_name,
        "description": desc or f"Optimize {op_name}",
        "dsl": dsl or None,
        "framework": framework or None,
        "backend": backend or None,
        "arch": arch or None,
        "editable_files": [editable_filename],
        "eval": {
            "timeout": eval_timeout,
        },
        "metric": {
            "primary": "latency_us",
            "lower_is_better": True,
            # Correctness tolerances are written explicitly so loader.py's
            # `metric_block.get("correctness_atol", 1e-2)` actually consumes
            # what the user (or default) chose, instead of always falling
            # through to the loader-side default.
            "correctness_atol": float(correctness_atol),
            "correctness_rtol": float(correctness_rtol),
        },
        "agent": {
            "ref_file": "reference.py",
            "max_rounds": max_rounds,
        },
    }
    if devices:
        task_yaml["devices"] = list(devices)

    # Only emit the code_checker block when disabled — default-true tasks
    # stay clean. quick_check.py and phase_machine.validate_kernel honor
    # this field; placeholder rejection still fires either way.
    if not code_checker_enabled:
        task_yaml["code_checker"] = {"enabled": False}

    # Add worker config if provided
    if worker_urls:
        task_yaml["worker"] = {"urls": worker_urls}

    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write(task_dir, "task.yaml", yaml_content)

    # Create .ar_state directory
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    # Git init + baseline commit
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _write(task_dir: str, rel_path: str, content: str):
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit.

    The actual commit goes through git_utils.commit_in_task — same code
    path hook_post_edit uses for seed commits, so reliability differences
    between Mode-1 (scaffold-time) and Mode-2 (GENERATE_KERNEL-time)
    commits are eliminated.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from git_utils import commit_in_task

    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True, check=True)
    ok, info = commit_in_task(task_dir, ["."], "scaffold: baseline")
    if not ok:
        raise RuntimeError(f"scaffold baseline commit failed: {info}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    """Construct scaffold's argparse, with no side effects.

    Extracted out of main() so parse_args.py can reuse the exact same flag
    spec without duplicating it. Single source of truth for which flags
    /autoresearch accepts and how they're typed/defaulted.
    """
    parser = argparse.ArgumentParser(
        description="Scaffold a task directory for Claude Code autoresearch",
    )
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref", default=None,
                           help="Path to reference.py (Model/get_inputs format)")
    ref_group.add_argument("--desc", default=None,
                           help="Natural language description → LLM generates reference")
    parser.add_argument("--kernel", default=None,
                        help="Path to initial kernel file (optional, skips generation)")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (auto-derived from --desc if omitted)")
    # DSL = primary pivot. backend is a pure function of DSL; arch is
    # derived from hardware (local: npu-smi on --devices; remote: worker
    # /api/v1/status). Neither needs to be user-facing.
    # Pull the canonical DSL list from hw_detect at construction time so
    # the help string can't drift from _DSL_BACKEND (and so we don't
    # silently prime the LLM with a stale or abbreviated list — this used
    # to read `--devices 0` from a stale example, etc.).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hw_detect import list_supported_dsls
    parser.add_argument("--dsl", default=None,
                        help=f"DSL name (one of: {', '.join(list_supported_dsls())}). "
                             f"Defaults to config.yaml:default_dsl.")
    parser.add_argument("--framework", default="torch",
                        choices=["torch", "mindspore", "numpy"],
                        help="Framework for the reference/kernel code "
                             "(default: torch).")
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Mutually exclusive "
                             "with --worker-url.")
    parser.add_argument("--worker-url", default=None,
                        help="Remote worker URL(s), comma-separated. "
                             "Mutually exclusive with --devices.")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    parser.add_argument("--no-code-checker", action="store_true",
                        help=("Disable the static CodeChecker pipeline "
                              "(syntax / imports / DSL / autotune compliance) "
                              "for this task. quick_check + validate_kernel "
                              "still reject the scaffold TODO placeholder; "
                              "everything else passes through. Useful when "
                              "the DSL rules are too strict for the chosen "
                              "kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    parser.add_argument("--correctness-atol", type=float, default=1e-2,
                        help=("torch.allclose absolute tolerance for the "
                              "ref/kernel correctness check. Written into "
                              "task.yaml's `metric.correctness_atol` and "
                              "consumed by the eval-package verify script. "
                              "Default 1e-2 matches autoresearch's loader "
                              "fallback."))
    parser.add_argument("--correctness-rtol", type=float, default=1e-2,
                        help=("torch.allclose relative tolerance for the "
                              "ref/kernel correctness check. Written into "
                              "task.yaml's `metric.correctness_rtol`. "
                              "Default 1e-2."))
    return parser


def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    # Resolve DSL (and via it, backend).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from settings import default_dsl
    from hw_detect import (
        backend_for_dsl, derive_arch, fetch_worker_hardware,
    )
    from ar_vendored.op.verifier.adapters.factory import (
        get_dsl_adapter, get_framework_adapter,
    )

    args.dsl = (args.dsl or default_dsl()).lower()
    try:
        get_dsl_adapter(args.dsl)
    except Exception as e:
        print(json.dumps({"status": "error",
                          "error": f"unsupported --dsl {args.dsl!r}: {e}"}))
        sys.exit(1)

    try:
        args.backend = backend_for_dsl(args.dsl)
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    try:
        get_framework_adapter(args.framework)
    except Exception as e:
        print(json.dumps({"status": "error",
                          "error": f"unsupported --framework "
                                   f"{args.framework!r}: {e}"}))
        sys.exit(1)

    # Hardware resolution: --devices XOR --worker-url.
    devices_list: list = []
    worker_urls: list = []
    args.arch = None

    if args.devices and args.worker_url:
        print(json.dumps({"status": "error",
                          "error": "--devices and --worker-url are mutually "
                                   "exclusive. Pick one (--devices for local "
                                   "eval, --worker-url for remote worker)."}))
        sys.exit(1)

    if args.devices:
        devices_list = [int(d.strip()) for d in args.devices.split(",")
                        if d.strip()]
        if not devices_list:
            print(json.dumps({"status": "error",
                              "error": "--devices parsed to an empty list"}))
            sys.exit(1)
        args.arch = derive_arch(args.backend, devices_list[0])
        if not args.arch:
            print(json.dumps({"status": "error",
                              "error": (f"could not derive arch from "
                                        f"{args.backend} device "
                                        f"{devices_list[0]} "
                                        f"(is the SMI tool on PATH?)")}))
            sys.exit(1)

    elif args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",")
                       if u.strip()]
        status = fetch_worker_hardware(worker_urls[0])
        if not status:
            print(json.dumps({"status": "error",
                              "error": (f"worker {worker_urls[0]} unreachable "
                                        f"or /api/v1/status failed")}))
            sys.exit(1)
        worker_backend = str(status.get("backend", "")).lower()
        worker_arch = str(status.get("arch", "")).lower()
        if worker_backend and worker_backend != args.backend:
            print(json.dumps({"status": "error",
                              "error": (f"worker backend {worker_backend!r} "
                                        f"incompatible with --dsl {args.dsl!r} "
                                        f"(requires {args.backend!r})")}))
            sys.exit(1)
        args.arch = worker_arch or None
        if not args.arch:
            print(json.dumps({"status": "error",
                              "error": (f"worker /api/v1/status returned no "
                                        f"arch: {status}")}))
            sys.exit(1)

    else:
        print(json.dumps({"status": "error",
                          "error": "must pass exactly one of --devices "
                                   "(local eval) or --worker-url (remote)."}))
        sys.exit(1)

    # Derive op-name if not provided
    if not args.op_name:
        if args.desc:
            import re as _re
            words = _re.findall(r"[a-zA-Z]+", args.desc)[:4]
            args.op_name = "_".join(w.lower() for w in words) or "custom_op"
        else:
            args.op_name = "custom_op"

    if args.ref:
        if not os.path.isfile(args.ref):
            print(json.dumps({"status": "error", "error": f"Reference file not found: {args.ref}"}))
            sys.exit(1)
        with open(args.ref, "r", encoding="utf-8") as f:
            ref_code = f.read()
        try:
            validate_ref(ref_code, args.ref)
        except ValueError as e:
            print(json.dumps({"status": "error", "error": str(e)}))
            sys.exit(1)
    else:
        # --desc mode: scaffold without reference. Claude Code fills it later.
        # Source the placeholder from phase_machine so is_placeholder_file's
        # prefix predicate stays in lockstep with what we actually write.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase_machine import REFERENCE_PLACEHOLDER_PREFIX
        ref_code = f"{REFERENCE_PLACEHOLDER_PREFIX}\n# {args.desc}\n"

    # Read initial kernel (optional)
    kernel_code = None
    if args.kernel:
        if not os.path.isfile(args.kernel):
            print(json.dumps({"status": "error", "error": f"Kernel file not found: {args.kernel}"}))
            sys.exit(1)
        with open(args.kernel, "r", encoding="utf-8") as f:
            kernel_code = f.read()

    # worker_urls / devices_list were resolved above.
    print(f"[scaffold] Creating task directory for {args.op_name}...", file=sys.stderr)

    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        desc=args.desc or "",
        dsl=args.dsl,
        framework=args.framework,
        backend=args.backend,
        devices=devices_list,
        arch=args.arch,
        worker_urls=worker_urls,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=not args.no_code_checker,
        correctness_atol=args.correctness_atol,
        correctness_rtol=args.correctness_rtol,
    )

    print(f"[scaffold] Task directory created: {task_dir}", file=sys.stderr)
    print(f"[scaffold] Files:", file=sys.stderr)
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}", file=sys.stderr)

    # Runnability gate: any mode that supplied a real --ref must produce a
    # reference.py that imports AND survives one Model.forward() pass on CPU.
    # The reference is the correctness baseline for every subsequent verify;
    # if it doesn't run, nothing downstream is meaningful. AST symbol presence
    # is checked earlier (see validate_ref); this catches torch import errors,
    # bad get_inputs shapes, missing ops, etc. Skipped in --desc mode where
    # reference.py is still a TODO stub.
    if args.ref:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase_machine import validate_reference
        ok, err = validate_reference(task_dir)
        if not ok:
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": f"reference.py failed runnability check: {err}",
                "hint": ("Fix the file under workspace/ and re-run /autoresearch. "
                         "scaffold left the partial task_dir in place for "
                         "inspection."),
            }))
            sys.exit(2)

    # Reference outputs are no longer captured locally. Worker side caches
    # them on the first verify round (keyed on reference.py sha) and reuses
    # across rounds. This saves a multi-GiB upload per large-tensor op.

    if args.run_baseline and args.ref and args.kernel:
        print(f"[scaffold] Running baseline eval...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_cmd = [sys.executable, os.path.join(script_dir, "baseline.py"), task_dir]
        if args.worker_url:
            baseline_cmd.extend(["--worker-url", args.worker_url])
        rc = subprocess.run(baseline_cmd).returncode
        if rc != 0:
            # /autoresearch reads the JSON from scaffold stdout and proceeds
            # straight to `export AR_TASK_DIR=...`; if baseline failed but we
            # still printed status=ok, the slash command would resume as if
            # the task were in PLAN. Surface the failure so the caller stops
            # and surfaces it to the user instead.
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": (f"baseline eval failed (exit {rc}); "
                          f"see [baseline]/[eval] stderr above"),
                "hint": ("Inspect kernel.py / reference.py / worker logs, "
                         "fix, then re-run: "
                         f"python .autoresearch/scripts/baseline.py "
                         f"\"{task_dir}\""),
            }))
            sys.exit(3)
    elif args.run_baseline:
        print(f"[scaffold] --run-baseline skipped: kernel.py not provided. "
              f"GENERATE_KERNEL phase will produce it; baseline runs after that.\n"
              f"[scaffold] Tip: baseline.py uses a local execution backend "
              f"automatically when torch / torch_npu for the selected backend "
              f"is installed — no --worker-url needed in that case.",
              file=sys.stderr)

    # Output
    print(json.dumps({"task_dir": task_dir, "status": "ok"}))


if __name__ == "__main__":
    main()

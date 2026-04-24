#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch.

Zero AKG dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; required to import + run end-to-end
    on CPU — scaffold gates on `phase_machine.validate_reference`)
  - kernel.py (editable; --kernel writes the user file directly, otherwise
    the canonical KERNEL_PLACEHOLDER from phase_machine — the placeholder
    routes the task to GENERATE_KERNEL on first activation)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # From a reference file (most common):
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --backend ascend --arch ascend910b3

    # With initial kernel (skip KernelGen):
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --backend cuda

    # With remote worker:
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --backend ascend --arch ascend910b3 --worker-url 127.0.0.1:9111

    # Custom output directory:
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --backend cuda --output-dir /tmp/tasks

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import uuid

import yaml


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------

def validate_ref(code: str, source: str = "reference"):
    """AST-level validation: must have class Model, get_inputs, get_init_inputs."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Reference from {source} has syntax error: {e}")

    names = {
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    required = {"Model": "class Model", "get_inputs": "get_inputs()",
                "get_init_inputs": "get_init_inputs()"}
    missing = [label for name, label in required.items() if name not in names]
    if missing:
        raise ValueError(f"Reference from {source} missing: {', '.join(missing)}")


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
    worker_urls: list | None = None,
    max_rounds: int = 20,
    eval_timeout: int = 120,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    code_checker_enabled: bool = True,
) -> str:
    """Create task directory with all files. Returns absolute path.

    Mirrors akg_agents.op.autoresearch.adapters.task_scaffolder.scaffold_task_dir
    but with zero AKG dependency.
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
        },
        "agent": {
            "ref_file": "reference.py",
            "max_rounds": max_rounds,
        },
    }

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
    """Initialize git repo and create baseline commit."""
    def _run(cmd):
        subprocess.run(cmd, cwd=task_dir, capture_output=True, check=True)

    _run(["git", "init"])
    _run(["git", "config", "user.name", "autoresearch"])
    _run(["git", "config", "user.email", "auto@research"])
    _run(["git", "add", "."])
    _run(["git", "commit", "-m", "scaffold: baseline"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
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
    # DSL is the primary pivot. backend / arch / framework default from the
    # DSL preset in config.yaml; user-supplied overrides must match the DSL
    # (no implicit inference — incompatible combos error out).
    parser.add_argument("--dsl", default=None,
                        help="DSL name — one of the keys in config.yaml:dsls "
                             "(triton_ascend, triton_cuda, ascendc, cuda_c, "
                             "cpp, tilelang_cuda, tilelang_npuir, pypto, "
                             "swft, torch). Defaults to config.yaml:default_dsl.")
    parser.add_argument("--backend", default=None,
                        help="Override the DSL's default backend "
                             "(ascend / cuda / cpu). Must match the DSL.")
    parser.add_argument("--arch", default=None,
                        help="Override the DSL's default arch "
                             "(e.g. ascend910b3, a100, x86_64).")
    parser.add_argument("--framework", default=None,
                        help="Override the DSL's default framework "
                             "(torch / mindspore / numpy).")
    parser.add_argument("--worker-url", default=None,
                        help="Remote worker URL(s), comma-separated")
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

    args = parser.parse_args()

    # Resolve DSL → backend / arch / framework. DSL is the single source of
    # truth: user picks a DSL, preset supplies defaults, explicit overrides
    # must be compatible with the DSL's preset (no string-match inference).
    from settings import dsl_preset, default_dsl
    args.dsl = (args.dsl or default_dsl()).lower()

    # Validate DSL against the vendored factory (authoritative list of
    # supported adapters). Unknown DSL → hard error with the full set.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from ar_vendored.op.verifier.adapters.factory import (
            get_dsl_adapter, get_backend_adapter, get_framework_adapter,
        )
        get_dsl_adapter(args.dsl)
    except Exception as e:
        print(json.dumps({"status": "error",
                          "error": f"unsupported --dsl {args.dsl!r}: {e}"}))
        sys.exit(1)

    preset = dsl_preset(args.dsl)
    if not preset:
        print(json.dumps({"status": "error",
                          "error": (f"DSL {args.dsl!r} is valid at the factory "
                                    f"but has no entry in config.yaml:dsls. "
                                    f"Add it to unblock scaffold.")}))
        sys.exit(1)

    args.backend = (args.backend or preset["backend"]).lower()
    args.arch = args.arch or preset["arch"]
    args.framework = (args.framework or preset["framework"]).lower()

    # Cross-validate: each explicit override must also be a known adapter,
    # and the (dsl, backend) pair must be internally consistent with the
    # DSL's preset. Mismatch = hard error, no silent correction.
    for label, value, getter in (
        ("backend", args.backend, get_backend_adapter),
        ("framework", args.framework, get_framework_adapter),
    ):
        try:
            getter(value)
        except Exception as e:
            print(json.dumps({"status": "error",
                              "error": f"unsupported --{label} {value!r}: {e}"}))
            sys.exit(1)

    if args.backend != preset["backend"]:
        print(json.dumps({"status": "error",
                          "error": (f"--backend {args.backend!r} is incompatible "
                                    f"with --dsl {args.dsl!r} (DSL preset "
                                    f"requires backend={preset['backend']!r}). "
                                    f"Pick one — they must agree.")}))
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
        ref_code = f"# TODO: Claude Code will generate reference from description:\n# {args.desc}\n"

    # Read initial kernel (optional)
    kernel_code = None
    if args.kernel:
        if not os.path.isfile(args.kernel):
            print(json.dumps({"status": "error", "error": f"Kernel file not found: {args.kernel}"}))
            sys.exit(1)
        with open(args.kernel, "r", encoding="utf-8") as f:
            kernel_code = f.read()

    # Parse worker URLs
    worker_urls = None
    if args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",") if u.strip()]

    # Scaffold
    print(f"[scaffold] Creating task directory for {args.op_name}...", file=sys.stderr)

    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        desc=args.desc or "",
        dsl=args.dsl,
        framework=args.framework,
        backend=args.backend,
        arch=args.arch,
        worker_urls=worker_urls,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=not args.no_code_checker,
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
        subprocess.run(baseline_cmd)
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

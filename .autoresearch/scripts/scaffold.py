#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch.

Zero AKG dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (baseline implementation)
  - kernel.py (editable, initially copied from reference or --kernel)
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

    # Write editable file (kernel.py)
    # If no initial kernel provided, copy from reference as starting point
    _write(task_dir, editable_filename, kernel_code or ref_code)

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
    parser.add_argument("--dsl", default=None,
                        choices=["triton_ascend", "triton_cuda", "torch",
                                 "cuda_c", "cpp", "ascendc", "tilelang_cuda"])
    parser.add_argument("--backend", default=None,
                        choices=["ascend", "cuda", "cpu"])
    parser.add_argument("--arch", default=None)
    parser.add_argument("--framework", default="torch")
    parser.add_argument("--worker-url", default=None,
                        help="Remote worker URL(s), comma-separated")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")

    args = parser.parse_args()

    # Apply backend/DSL/arch defaults from .autoresearch/config.yaml.
    # Inference rules when --backend is omitted:
    #   - `--dsl triton_cuda` / anything with "cuda" → cuda preset
    #   - `--dsl cpp`                                → cpu preset
    #   - otherwise                                  → config default_backend
    from settings import backend_preset, default_backend
    preset_key = args.backend or (
        "cuda" if args.dsl and "cuda" in args.dsl else
        "cpu" if args.dsl == "cpp" else
        default_backend()
    )
    preset = backend_preset(preset_key) or backend_preset(default_backend())
    if args.dsl is None:
        args.dsl = preset.get("dsl")
    if args.backend is None:
        args.backend = preset_key
    if args.arch is None:
        args.arch = preset.get("arch")

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
    )

    print(f"[scaffold] Task directory created: {task_dir}", file=sys.stderr)
    print(f"[scaffold] Files:", file=sys.stderr)
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}", file=sys.stderr)

    # Optionally run baseline eval. Skip in --desc mode: reference.py is a TODO
    # placeholder until Claude fills it, so baseline would fail.
    if args.ref:
        # Capture the PyTorch reference ONCE on CPU. All subsequent verify
        # rounds compare the kernel against this stored .pt — we never re-run
        # the reference Model during normal operation.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"[scaffold] Capturing PyTorch reference outputs (CPU)...",
              file=sys.stderr)
        cap = subprocess.run(
            [sys.executable, os.path.join(script_dir, "reference_capture.py"), task_dir],
            capture_output=True, text=True,
        )
        if cap.stderr:
            print(cap.stderr, end="", file=sys.stderr)
        if cap.returncode != 0:
            print(f"[scaffold] WARNING: reference capture failed — verify "
                  f"will fall back to running Model inline each round.",
                  file=sys.stderr)
        else:
            print(f"[scaffold] Reference saved.", file=sys.stderr)

    if args.run_baseline and args.ref:
        print(f"[scaffold] Running baseline eval...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_cmd = [sys.executable, os.path.join(script_dir, "baseline.py"), task_dir]
        if args.worker_url:
            baseline_cmd.extend(["--worker-url", args.worker_url])
        subprocess.run(baseline_cmd)
    elif args.run_baseline:
        print(f"[scaffold] --run-baseline skipped (--desc mode: reference not ready)",
              file=sys.stderr)

    # Output
    print(json.dumps({"task_dir": task_dir, "status": "ok"}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick static check for editable files before eval.

Runs the CodeChecker pipeline (syntax → compile → imports → stray Chinese →
DSL compliance → autotune compliance) on every editable file, then optionally
runs a user-configured smoke test. Catches the common LLM failure modes
(non-triton code under a triton DSL, missing kernel launch, forbidden torch
APIs in forward, stray Chinese comments leaking into code tokens) before we
pay the cost of a full worker eval.

Usage:
    python .autoresearch/scripts/quick_check.py <task_dir>

Output:
    stdout: 'OK' on pass, JSON error blob on fail
    exit 0 = pass, 1 = fail
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config
from code_checker import CodeChecker


def check_editable_files(task_dir: str, config) -> list:
    """Static-check every editable .py via the ported CodeChecker.

    Honors `config.code_checker_enabled` — when off, only the file-existence
    check fires; the AST/import/DSL/autotune pipeline is skipped. This is
    the single gate consulted by both the runtime quick check and
    `phase_machine.validate_kernel`. Public lib API (no leading
    underscore): both the CLI `main()` below and `validators.validate_kernel`
    call this directly; do not duplicate the body.
    """
    issues = []
    use_checker = config.code_checker_enabled
    checker = CodeChecker(backend=config.backend or "", dsl=config.dsl or "") if use_checker else None
    for fname in config.editable_files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(task_dir, fname)
        if not os.path.exists(fpath):
            issues.append({"file": fname, "report": "file not found", "errors": []})
            continue
        if not use_checker:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        passed, report, errors = checker.check(code)
        if not passed:
            issues.append({"file": fname, "report": report, "errors": errors})
    return issues


def _run_smoke_test(task_dir: str, config) -> list:
    if not config.smoke_test_script:
        return []
    smoke_path = os.path.join(task_dir, config.smoke_test_script)
    if not os.path.exists(smoke_path):
        return []
    try:
        r = subprocess.run(
            [sys.executable, smoke_path],
            capture_output=True, text=True,
            timeout=config.smoke_test_timeout, cwd=task_dir,
        )
    except subprocess.TimeoutExpired:
        return [f"smoke test timed out after {config.smoke_test_timeout}s"]
    except Exception as e:
        return [f"smoke test launch error: {e}"]
    if r.returncode != 0:
        tail = (r.stderr or "")[-500:]
        return [f"smoke test failed (exit {r.returncode}): {tail}"]
    return []


def main():
    parser = argparse.ArgumentParser(description="Quick static check")
    parser.add_argument("task_dir", help="Path to task directory")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({"ok": False, "error": "task.yaml not found"}))
        sys.exit(1)

    if not config.code_checker_enabled:
        print("[quick_check] CodeChecker disabled in task.yaml — only "
              "file-existence and smoke test will run.", file=sys.stderr)

    file_issues = check_editable_files(task_dir, config)
    smoke_errors = _run_smoke_test(task_dir, config)

    if not file_issues and not smoke_errors:
        print("OK")
        sys.exit(0)

    blob = {"ok": False}
    if file_issues:
        blob["file_issues"] = file_issues
    if smoke_errors:
        blob["smoke_errors"] = smoke_errors
    print(json.dumps(blob, ensure_ascii=False))
    sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Eval wrapper for Claude Code autoresearch.

Zero external dependency — uses local task_config.py for YAML parsing and eval execution.

Usage:
    python .autoresearch/scripts/eval_wrapper.py <task_dir> [--device-id N] [--worker-url URL[,URL,...]]
                                                                              ^ comma-separated; first reachable wins

Output (last line of stdout):
    {"correctness": true, "metrics": {"latency_us": 145.3}, "error": null}
"""

import argparse
import json
import os
import sys

# Import from sibling module
sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config, run_eval, format_result_summary
from failure_extractor import extract_failure_signals


def main():
    parser = argparse.ArgumentParser(description="Run eval and output JSON")
    parser.add_argument("task_dir", help="Path to the task directory")
    parser.add_argument("--device-id", type=int, default=None, help="Device ID (local eval)")
    parser.add_argument("--worker-url", default=None,
                        help="Remote worker URL(s), comma-separated. Overrides task.yaml worker.urls.")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)

    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({
            "correctness": False,
            "metrics": {},
            "error": f"task.yaml not found in {task_dir}",
        }))
        sys.exit(0)

    # CLI --worker-url overrides task.yaml
    worker_urls = None
    if args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",") if u.strip()]

    if worker_urls or config.worker_urls:
        urls = worker_urls or config.worker_urls
        print(f"[eval] Running remote eval for {config.name} via {urls[0]}"
              + (f" (+{len(urls)-1} fallback)" if len(urls) > 1 else ""),
              file=sys.stderr)
    else:
        # Probe the local backend up front so the log line tells the user
        # exactly why this run will / won't proceed locally.
        from local_worker import detect_local_backend
        backend_key = (config.backend or "cpu").lower()
        ok, why = detect_local_backend(backend_key)
        status = "available" if ok else "UNAVAILABLE"
        print(f"[eval] Running local ({backend_key}) eval for {config.name} "
              f"— backend {status}: {why}", file=sys.stderr)

    result = run_eval(task_dir, config, device_id=args.device_id, worker_urls=worker_urls)

    print(f"[eval] {format_result_summary(result)}", file=sys.stderr)

    output = {
        "correctness": result.correctness,
        "metrics": result.metrics,
        "error": result.error,
    }
    # Attach structured failure signals only when something went wrong — on
    # success the raw log is noisy and adds no value to downstream consumers.
    # `raw_output` is already capped upstream (4 KB tail in task_config), so
    # we forward it whole — the pipeline falls back to printing it verbatim
    # when no pattern matched.
    if not result.correctness or result.error:
        output["failure_signals"] = extract_failure_signals(result.raw_output)
        output["raw_output_tail"] = result.raw_output or ""
    print(json.dumps(output))


if __name__ == "__main__":
    main()

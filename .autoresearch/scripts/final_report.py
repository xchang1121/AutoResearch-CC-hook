#!/usr/bin/env python3
"""Generate deterministic FINISH reports from AutoResearch state.

The report is derived from task.yaml, progress.json, history.jsonl, and plan.md
so facts stay auditable and reproducible.

Usage:
    python .autoresearch/scripts/final_report.py <task_dir>

Outputs:
    <task_dir>/.ar_state/report.json
    <task_dir>/.ar_state/report.md
    <task_dir>/.ar_state/report.png (optional, when matplotlib is available)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from phase_machine import (  # noqa: E402
    FINISH,
    REPORT_FILE,
    REPORT_JSON_FILE,
    REPORT_PLOT_FILE,
    get_plan_items,
    history_path,
    load_history,
    load_progress,
    plan_path,
    read_phase,
    state_path,
)
from task_config import load_task_config  # noqa: E402


def _load_history(task_dir: str) -> list[dict[str, Any]]:
    """Audit-grade history read for the FINISH report. Corrupt JSONL lines
    surface as ``CORRUPT_HISTORY_LINE`` records so the report makes audit
    gaps visible (the planner / dashboard intentionally swallow them).
    """
    return load_history(task_dir, on_corrupt="record")


def _metric_value(rec: dict[str, Any], metric: str) -> Optional[float]:
    metrics = rec.get("metrics") or {}
    val = metrics.get(metric)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _is_better(candidate: float, incumbent: Optional[float],
               lower_is_better: bool) -> bool:
    if incumbent is None:
        return True
    if lower_is_better:
        return candidate < incumbent
    return candidate > incumbent


def _best_history_record(history: list[dict[str, Any]], metric: str,
                         lower_is_better: bool,
                         best_commit: Optional[str]) -> Optional[dict[str, Any]]:
    if best_commit:
        for rec in history:
            if rec.get("commit") == best_commit:
                return rec

    best_rec: Optional[dict[str, Any]] = None
    best_val: Optional[float] = None
    for rec in history:
        if rec.get("decision") not in ("SEED", "KEEP"):
            continue
        if rec.get("correctness") is False:
            continue
        val = _metric_value(rec, metric)
        if val is None:
            continue
        if _is_better(val, best_val, lower_is_better):
            best_rec = rec
            best_val = val
    return best_rec


def _format_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _improvement(best: Optional[float], baseline: Optional[float],
                 lower_is_better: bool) -> dict[str, Optional[float]]:
    if best is None or baseline is None or baseline == 0:
        return {"ratio_vs_baseline": None, "percent_vs_baseline": None}
    if lower_is_better:
        ratio = baseline / best if best != 0 else None
        pct = (baseline - best) / abs(baseline) * 100.0
    else:
        ratio = best / baseline
        pct = (best - baseline) / abs(baseline) * 100.0
    return {"ratio_vs_baseline": ratio, "percent_vs_baseline": pct}


def _safe_config_summary(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    return {
        "name": config.name,
        "description": config.description,
        "dsl": config.dsl,
        "framework": config.framework,
        "backend": config.backend,
        "arch": config.arch,
        "editable_files": config.editable_files,
        "primary_metric": config.primary_metric,
        "lower_is_better": config.lower_is_better,
        "max_rounds": config.max_rounds,
        "worker_urls": config.worker_urls,
        "devices": config.devices,
    }


def _collect_report(task_dir: str) -> dict[str, Any]:
    config = load_task_config(task_dir)
    progress = load_progress(task_dir) or {}
    history = _load_history(task_dir)
    phase = read_phase(task_dir)

    metric = config.primary_metric if config else progress.get("primary_metric", "score")
    lower = bool(config.lower_is_better) if config else True
    best_metric = progress.get("best_metric")
    baseline_metric = progress.get("baseline_metric")
    seed_metric = progress.get("seed_metric")

    best = float(best_metric) if isinstance(best_metric, (int, float)) else None
    baseline = float(baseline_metric) if isinstance(baseline_metric, (int, float)) else None
    seed = float(seed_metric) if isinstance(seed_metric, (int, float)) else None

    decisions = Counter(str(rec.get("decision", "UNKNOWN")) for rec in history)
    best_rec = _best_history_record(
        history,
        str(metric),
        lower,
        progress.get("best_commit"),
    )
    improvement = _improvement(best, baseline, lower)

    kept = [
        {
            "round": rec.get("round"),
            "plan_item": rec.get("plan_item"),
            "description": rec.get("description"),
            "metric": _metric_value(rec, str(metric)),
            "commit": rec.get("commit"),
        }
        for rec in history
        if rec.get("decision") == "KEEP"
    ]
    failures = [
        {
            "round": rec.get("round"),
            "plan_item": rec.get("plan_item"),
            "description": rec.get("description"),
            "decision": rec.get("decision"),
            "error": rec.get("error"),
        }
        for rec in history
        if rec.get("decision") in ("FAIL", "CORRUPT_HISTORY_LINE")
    ]
    discards = [
        {
            "round": rec.get("round"),
            "plan_item": rec.get("plan_item"),
            "description": rec.get("description"),
            "metric": _metric_value(rec, str(metric)),
            "reason": rec.get("reason"),
        }
        for rec in history
        if rec.get("decision") == "DISCARD"
    ]
    pending_items = [it for it in get_plan_items(task_dir) if not it.get("done")]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_dir": task_dir,
        "phase": phase,
        "is_finish_phase": phase == FINISH,
        "config": _safe_config_summary(config),
        "summary": {
            "task": progress.get("task") or (config.name if config else None),
            "primary_metric": metric,
            "lower_is_better": lower,
            "eval_rounds": progress.get("eval_rounds"),
            "max_rounds": progress.get("max_rounds") or (config.max_rounds if config else None),
            "plan_version": progress.get("plan_version"),
            "baseline_metric": baseline,
            "baseline_source": progress.get("baseline_source"),
            "seed_metric": seed,
            "best_metric": best,
            "ratio_vs_baseline": improvement["ratio_vs_baseline"],
            "percent_vs_baseline": improvement["percent_vs_baseline"],
            "baseline_commit": progress.get("baseline_commit"),
            "best_commit": progress.get("best_commit"),
            "consecutive_failures": progress.get("consecutive_failures"),
            "status": progress.get("status"),
        },
        "decisions": dict(sorted(decisions.items())),
        "best_record": best_rec,
        "kept_rounds": kept,
        "discarded_rounds": discards,
        "failed_rounds": failures,
        "pending_items": pending_items,
        "history": history,
        "files": {
            "progress": state_path(task_dir, "progress.json"),
            "history": history_path(task_dir),
            "plan": plan_path(task_dir),
            "report_json": state_path(task_dir, REPORT_JSON_FILE),
            "report_md": state_path(task_dir, REPORT_FILE),
            "report_plot": state_path(task_dir, REPORT_PLOT_FILE),
        },
    }


def _md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _numeric_round(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _plot_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    metric = str(report.get("summary", {}).get("primary_metric") or "score")
    points = []
    for rec in report.get("history", []) or []:
        rnd = _numeric_round(rec.get("round"))
        val = _metric_value(rec, metric)
        if rnd is None or val is None:
            continue
        points.append({
            "round": rnd,
            "metric": val,
            "decision": rec.get("decision") or "UNKNOWN",
            "correctness": rec.get("correctness"),
        })
    return points


def _best_so_far(points: list[dict[str, Any]],
                 lower_is_better: bool) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    best: Optional[float] = None
    for point in sorted(points, key=lambda item: item["round"]):
        if point["decision"] not in ("SEED", "KEEP"):
            continue
        if point.get("correctness") is False:
            continue
        val = point["metric"]
        if _is_better(val, best, lower_is_better):
            best = val
        if best is not None:
            xs.append(point["round"])
            ys.append(best)
    return xs, ys


def _generate_plot(report: dict[str, Any], output_path: str) -> dict[str, Any]:
    """Generate report.png with matplotlib; degrade to text on failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"available": False, "path": None,
                "error": f"matplotlib unavailable: {exc}"}

    points = _plot_points(report)
    if not points:
        return {"available": False, "path": None,
                "error": "no numeric history points to plot"}

    try:
        summary = report.get("summary", {})
        metric = str(summary.get("primary_metric") or "score")
        lower = bool(summary.get("lower_is_better", True))
        direction = "lower is better" if lower else "higher is better"
        baseline = summary.get("baseline_metric")

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=130)
        colors = {
            "SEED": "#64748b",
            "KEEP": "#15803d",
            "DISCARD": "#f97316",
            "FAIL": "#dc2626",
        }
        markers = {
            "SEED": "o",
            "KEEP": "o",
            "DISCARD": "x",
            "FAIL": "X",
        }
        for decision in ("SEED", "KEEP", "DISCARD", "FAIL"):
            subset = [p for p in points if p["decision"] == decision]
            if not subset:
                continue
            ax.scatter(
                [p["round"] for p in subset],
                [p["metric"] for p in subset],
                c=colors[decision],
                marker=markers[decision],
                s=58 if decision in ("SEED", "KEEP") else 46,
                label=f"{decision} ({len(subset)})",
                zorder=4,
                alpha=0.9,
            )

        other = [p for p in points if p["decision"] not in {"SEED", "KEEP", "DISCARD", "FAIL"}]
        if other:
            ax.scatter(
                [p["round"] for p in other],
                [p["metric"] for p in other],
                c="#475569",
                marker=".",
                s=35,
                label=f"OTHER ({len(other)})",
                zorder=3,
            )

        best_x, best_y = _best_so_far(points, lower)
        if best_x:
            ax.step(best_x, best_y, where="post", color="#1d4ed8",
                    linewidth=2.0, label="best so far", zorder=2)

        if isinstance(baseline, (int, float)):
            ax.axhline(float(baseline), color="#a16207", linestyle="--",
                       linewidth=1.5, alpha=0.75, label="baseline")

        task_name = summary.get("task") or report.get("config", {}).get("name") or "AutoResearch"
        ax.set_title(f"{task_name} - {metric} ({direction})")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return {"available": True, "path": output_path, "error": None}
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        return {"available": False, "path": None,
                "error": f"plot generation failed: {exc}"}


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    config = report.get("config") or {}
    metric = summary.get("primary_metric") or "metric"
    direction = "lower is better" if summary.get("lower_is_better") else "higher is better"

    ratio = summary.get("ratio_vs_baseline")
    pct = summary.get("percent_vs_baseline")
    if ratio is None:
        improvement = "N/A"
    else:
        improvement = f"{ratio:.4g}x vs baseline ({pct:+.2f}%)"

    lines = [
        "# AutoResearch Final Report",
        "",
        "## Task",
        "",
        f"- Name: `{_md_escape(summary.get('task') or config.get('name') or 'unknown')}`",
        f"- Task dir: `{_md_escape(report.get('task_dir'))}`",
        f"- Generated at: `{_md_escape(report.get('generated_at'))}`",
        f"- Phase when generated: `{_md_escape(report.get('phase'))}`",
        f"- DSL/framework/backend/arch: `{_md_escape(config.get('dsl'))}` / "
        f"`{_md_escape(config.get('framework'))}` / `{_md_escape(config.get('backend'))}` / "
        f"`{_md_escape(config.get('arch'))}`",
        f"- Editable files: `{_md_escape(', '.join(config.get('editable_files') or []))}`",
        "",
        "## Result Summary",
        "",
        f"- Primary metric: `{_md_escape(metric)}` ({direction})",
        f"- Rounds: `{_md_escape(summary.get('eval_rounds'))}` / `{_md_escape(summary.get('max_rounds'))}`",
        f"- Baseline metric: `{_format_num(summary.get('baseline_metric'))}` "
        f"source=`{_md_escape(summary.get('baseline_source'))}`",
        f"- Seed metric: `{_format_num(summary.get('seed_metric'))}`",
        f"- Best metric: `{_format_num(summary.get('best_metric'))}`",
        f"- Improvement: `{improvement}`",
        f"- Best commit: `{_md_escape(summary.get('best_commit'))}`",
        f"- Baseline commit: `{_md_escape(summary.get('baseline_commit'))}`",
        "",
        "## Optimization Plot",
        "",
    ]
    plot = report.get("plot") or {}
    if plot.get("available"):
        lines.append("![Optimization history](report.png)")
    else:
        lines.append(f"- Plot unavailable: {_md_escape(plot.get('error') or 'not generated')}")

    lines.extend([
        "",
        "## Decisions",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ])
    for decision, count in (report.get("decisions") or {}).items():
        lines.append(f"| {_md_escape(decision)} | {count} |")

    best_rec = report.get("best_record")
    lines.extend(["", "## Best Candidate", ""])
    if best_rec:
        lines.extend([
            f"- Round: `{_md_escape(best_rec.get('round'))}`",
            f"- Plan item: `{_md_escape(best_rec.get('plan_item'))}`",
            f"- Description: {_md_escape(best_rec.get('description'))}",
            f"- Commit: `{_md_escape(best_rec.get('commit'))}`",
            f"- Metrics: `{_md_escape(json.dumps(best_rec.get('metrics') or {}, ensure_ascii=False, sort_keys=True))}`",
        ])
    else:
        lines.append("- No KEEP/SEED record with a usable primary metric was found.")

    lines.extend(["", "## Kept Improvements", ""])
    kept = report.get("kept_rounds") or []
    if kept:
        lines.extend([
            "| Round | Item | Metric | Commit | Description |",
            "|---:|---|---:|---|---|",
        ])
        for rec in kept:
            lines.append(
                f"| {_md_escape(rec.get('round'))} | {_md_escape(rec.get('plan_item'))} | "
                f"{_format_num(rec.get('metric'))} | `{_md_escape(rec.get('commit'))}` | "
                f"{_md_escape(rec.get('description'))} |"
            )
    else:
        lines.append("- No kept improvement beyond the seed was recorded.")

    lines.extend(["", "## Failures And Diagnostics", ""])
    failures = report.get("failed_rounds") or []
    if failures:
        lines.extend([
            "| Round | Item | Decision | Description | Error |",
            "|---:|---|---|---|---|",
        ])
        for rec in failures[-10:]:
            lines.append(
                f"| {_md_escape(rec.get('round'))} | {_md_escape(rec.get('plan_item'))} | "
                f"{_md_escape(rec.get('decision'))} | {_md_escape(rec.get('description'))} | "
                f"{_md_escape(rec.get('error'))} |"
            )
    else:
        lines.append("- No failed rounds were recorded.")

    pending = report.get("pending_items") or []
    lines.extend(["", "## Residual State", ""])
    if pending:
        lines.append("- Pending items remain in plan.md:")
        for item in pending:
            marker = "ACTIVE" if item.get("active") else "pending"
            lines.append(
                f"- `{_md_escape(item.get('id'))}` [{marker}] "
                f"{_md_escape(item.get('description'))}"
            )
    else:
        lines.append("- No pending plan items remain.")

    if not report.get("is_finish_phase"):
        lines.extend([
            "",
            "> Warning: this report was generated outside FINISH. Treat it as an interim snapshot.",
        ])

    lines.append("")
    return "\n".join(lines)


def generate_report(task_dir: str, *, require_finish: bool = True) -> dict[str, Any]:
    task_dir = os.path.abspath(task_dir)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"task_dir not found: {task_dir}")
    report = _collect_report(task_dir)
    if require_finish and not report.get("is_finish_phase"):
        raise RuntimeError(
            f"final_report.py only runs in FINISH; current phase is "
            f"{report.get('phase')!r}"
        )
    state_dir = os.path.join(task_dir, ".ar_state")
    os.makedirs(state_dir, exist_ok=True)

    json_path = state_path(task_dir, REPORT_JSON_FILE)
    md_path = state_path(task_dir, REPORT_FILE)
    plot_path = state_path(task_dir, REPORT_PLOT_FILE)
    report["plot"] = _generate_plot(report, plot_path)
    if not report["plot"].get("available"):
        try:
            os.remove(plot_path)
        except FileNotFoundError:
            pass
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AutoResearch final report")
    parser.add_argument("task_dir")
    parser.add_argument(
        "--allow-interim",
        action="store_true",
        help="write a snapshot even when .phase is not FINISH (debug use only)",
    )
    args = parser.parse_args()

    try:
        report = generate_report(args.task_dir, require_finish=not args.allow_interim)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    print(f"[final_report] wrote {report['files']['report_md']}")
    print(f"[final_report] wrote {report['files']['report_json']}")
    if report.get("plot", {}).get("available"):
        print(f"[final_report] wrote {report['files']['report_plot']}")
    else:
        print(f"[final_report] plot skipped: {report.get('plot', {}).get('error')}")
    print(json.dumps({
        "ok": True,
        "report_md": report["files"]["report_md"],
        "report_json": report["files"]["report_json"],
        "report_plot": report["files"]["report_plot"] if report.get("plot", {}).get("available") else None,
        "best_metric": report["summary"].get("best_metric"),
        "baseline_metric": report["summary"].get("baseline_metric"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

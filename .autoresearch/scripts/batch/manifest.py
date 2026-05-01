"""Manifest loading + progress JSON I/O for the batch runner.

Workspace convention:
    <workspace_dir>/
        manifest.yaml | manifest.json    # user-authored
        batch_progress.json              # written here
        batch.log                        # written here
        <ref_dir>/<op_name>_ref.py
        <kernel_dir>/<op_name>_kernel.py

YAML support is optional (requires pyyaml). JSON works with stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "batch_progress.json"
LOG_FILENAME = "batch.log"
VALID_MODES = ("ref-kernel", "ref")
VALID_STATUSES = ("pending", "running", "done", "error", "skip")


class ManifestError(Exception):
    pass


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ManifestError(
            f"{path.name} is YAML but pyyaml is not installed. "
            f"Either `pip install pyyaml` or rename to manifest.json (JSON format)."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_manifest(workspace_dir: Path) -> Path:
    """Return path to manifest.yaml or manifest.json, preferring YAML."""
    yaml_path = workspace_dir / "manifest.yaml"
    if yaml_path.exists():
        return yaml_path
    json_path = workspace_dir / "manifest.json"
    if json_path.exists():
        return json_path
    raise ManifestError(
        f"no manifest.yaml or manifest.json in {workspace_dir}"
    )


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.suffix == ".yaml" or manifest_path.suffix == ".yml":
        data = _load_yaml(manifest_path)
    elif manifest_path.suffix == ".json":
        data = _load_json(manifest_path)
    else:
        raise ManifestError(f"unknown manifest extension: {manifest_path}")
    if not isinstance(data, dict):
        raise ManifestError(f"manifest root must be a mapping, got {type(data).__name__}")
    return data


def resolve_cases(workspace_dir: Path, manifest: dict, mode: str) -> list[dict]:
    """Apply the <op_name>_{ref,kernel}.py naming convention and return resolved
    case dicts. Pre-flight check that every referenced file exists.

    Returns a list of dicts with keys: op_name, ref (abs path), kernel (abs path
    or None).
    """
    if mode not in VALID_MODES:
        raise ManifestError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    ops = manifest.get("ops")
    if not ops or not isinstance(ops, list):
        raise ManifestError("manifest.ops must be a non-empty list")

    ref_dir_raw = manifest.get("ref_dir")
    if not ref_dir_raw:
        raise ManifestError("manifest.ref_dir is required")
    ref_dir = (workspace_dir / ref_dir_raw).resolve()
    if not ref_dir.is_dir():
        raise ManifestError(f"ref_dir not found: {ref_dir}")

    kernel_dir = None
    if mode == "ref-kernel":
        kernel_dir_raw = manifest.get("kernel_dir")
        if not kernel_dir_raw:
            raise ManifestError("kernel_dir required when mode=ref-kernel")
        kernel_dir = (workspace_dir / kernel_dir_raw).resolve()
        if not kernel_dir.is_dir():
            raise ManifestError(f"kernel_dir not found: {kernel_dir}")

    cases: list[dict] = []
    seen: set[str] = set()
    for entry in ops:
        if not isinstance(entry, str):
            raise ManifestError(
                f"manifest.ops entries must be strings (op names); got {entry!r}"
            )
        op_name = entry.strip()
        if not op_name:
            raise ManifestError("empty op_name in manifest.ops")
        if op_name in seen:
            raise ManifestError(f"duplicate op_name: {op_name}")
        seen.add(op_name)

        ref_path = ref_dir / f"{op_name}_ref.py"
        if not ref_path.is_file():
            raise ManifestError(f"{ref_path.relative_to(workspace_dir)} not found")

        kernel_path: Path | None = None
        if kernel_dir is not None:
            kernel_path = kernel_dir / f"{op_name}_kernel.py"
            if not kernel_path.is_file():
                raise ManifestError(
                    f"{kernel_path.relative_to(workspace_dir)} not found"
                )

        cases.append({
            "op_name": op_name,
            "ref": str(ref_path),
            "kernel": str(kernel_path) if kernel_path else None,
        })

    return cases


def load_progress(workspace_dir: Path) -> dict:
    path = workspace_dir / PROGRESS_FILENAME
    if not path.exists():
        return {"workspace_dir": str(workspace_dir.resolve()), "cases": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"corrupt progress file at {path}: {e}")


def save_progress(workspace_dir: Path, progress: dict) -> None:
    path = workspace_dir / PROGRESS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def merge_cases(progress: dict, resolved_cases: list[dict],
                mode: str, dsl: str) -> tuple[dict, list[str]]:
    """Merge freshly-resolved cases into the progress dict.

    The manifest is the source of truth: ops that exist in the old progress
    file but no longer in `resolved_cases` are dropped (so a user filtering
    or deleting ops via discover.py / manual manifest edits actually shrinks
    the queue, matching the docs' "ops list fully replaced" promise).

    New cases are inserted as pending; surviving cases keep their status but
    their ref/kernel paths refresh.

    Returns (progress, dropped_op_names).
    """
    progress["mode"] = mode
    progress["dsl"] = dsl
    old_cases = progress.get("cases", {})
    resolved_ops = {c["op_name"] for c in resolved_cases}
    dropped = sorted(op for op in old_cases if op not in resolved_ops)

    new_cases: dict = {}
    for c in resolved_cases:
        op = c["op_name"]
        existing = old_cases.get(op)
        if existing is None:
            new_cases[op] = {
                "op_name": op,
                "ref": c["ref"],
                "kernel": c["kernel"],
                "status": "pending",
                "task_dir": None,
                "started_at": None,
                "finished_at": None,
                "final_phase": None,
                "rc": None,
                "result": {
                    "baseline_metric": None,
                    "best_metric": None,
                    "rounds": None,
                    "consecutive_failures": None,
                },
                "note": "",
            }
        else:
            existing["ref"] = c["ref"]
            existing["kernel"] = c["kernel"]
            new_cases[op] = existing
    progress["cases"] = new_cases
    return progress, dropped


def update_case(workspace_dir: Path, op_name: str, **fields: Any) -> None:
    """Atomic update of one case's fields. Reloads progress file on each call
    so concurrent edits (e.g. by hand) aren't clobbered."""
    progress = load_progress(workspace_dir)
    case = progress.get("cases", {}).get(op_name)
    if case is None:
        raise ManifestError(f"unknown op_name: {op_name}")
    if "status" in fields and fields["status"] not in VALID_STATUSES:
        raise ManifestError(f"invalid status: {fields['status']}")
    case.update(fields)
    save_progress(workspace_dir, progress)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_task_state(task_dir: Path) -> dict:
    """Pull the result block from <task_dir>/.ar_state/progress.json. Returns
    a dict with whichever fields could be read."""
    out: dict = {
        "baseline_metric": None,
        "best_metric": None,
        "rounds": None,
        "consecutive_failures": None,
    }
    for name in ("progress.json", ".progress.json"):
        pf = task_dir / ".ar_state" / name
        if not pf.exists():
            continue
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out["baseline_metric"] = data.get("baseline_metric")
        out["best_metric"] = data.get("best_metric")
        out["rounds"] = data.get("eval_rounds")
        out["consecutive_failures"] = data.get("consecutive_failures")
        break
    return out


def read_phase(task_dir: Path) -> str:
    pf = task_dir / ".ar_state" / ".phase"
    if pf.exists():
        try:
            return pf.read_text(encoding="utf-8").strip() or "UNKNOWN"
        except OSError:
            pass
    return "UNKNOWN"


def repo_root() -> Path:
    """The claude-autoresearch repo root, derived from this file's location.

    Layout: <repo>/.autoresearch/scripts/batch/manifest.py
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def find_recent_task_dir(op_name: str, since_ts: float) -> Path | None:
    """Fallback when AUTORESEARCH_RESULT marker is missing: scan ar_tasks/ for
    the most recent task dir matching op_name with mtime >= since_ts."""
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for d in tasks_root.glob(f"{op_name}_*"):
        try:
            mt = d.stat().st_mtime
        except OSError:
            continue
        if mt >= since_ts:
            candidates.append((mt, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_active_task_dir() -> Path | None:
    """Most-recently-touched task dir (heartbeat preferred over dir mtime)."""
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return None
    best: Path | None = None
    best_mt = -1.0
    for d in tasks_root.iterdir():
        if not d.is_dir():
            continue
        hb = d / ".ar_state" / ".heartbeat"
        try:
            mt = hb.stat().st_mtime if hb.exists() else d.stat().st_mtime
        except OSError:
            continue
        if mt > best_mt:
            best_mt = mt
            best = d
    return best

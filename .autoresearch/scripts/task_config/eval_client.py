"""Eval dispatcher + transports.

Public entry point: `run_eval(task_dir, config, device_id=None,
worker_urls=None) -> EvalResult`. It picks one of three paths:

  1. Explicit worker URLs (from CLI or task.yaml.worker.urls) → remote.
  2. `local_worker.detect_local_backend(config.backend)` reports the
     runtime is available → local subprocess.
  3. Otherwise → EvalResult with a clear "no execution backend" error.

Both transports unpack the same tar.gz from package_builder and converge
on `_assemble_eval_result`, so downstream sees identical EvalResult shapes.

What lives here:
  - Worker URL discovery (`_normalize_worker_url`, `_worker_status`,
    `_select_worker`).
  - HTTP client (`_multipart_post`, `_worker_acquire_device`,
    `_worker_release_device`, `_worker_verify`, `_worker_profile`).
  - Result assembly (`_assemble_eval_result`).
  - The three eval entry points (`run_remote_eval`, `run_local_eval`,
    `run_eval`).

What's NOT here:
  - Tarball assembly / DSL adapters / verify-script templates — those
    live in package_builder.
  - EvalResult / improvement / constraints — those live in metric_policy.
  - YAML parsing — those live in loader.
"""
import json
import os
import sys
import uuid
from typing import Optional
from urllib.request import Request, urlopen

from .loader import TaskConfig
from .metric_policy import EvalResult
from .package_builder import _build_package


# ---------------------------------------------------------------------------
# Worker URL discovery
# ---------------------------------------------------------------------------

def _normalize_worker_url(url: str) -> str:
    """Ensure URL has scheme. '127.0.0.1:9111' → 'http://127.0.0.1:9111'."""
    url = url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    return url.rstrip("/")


def _worker_status(worker_url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET /api/v1/status. Returns parsed JSON or None on failure."""
    url = f"{_normalize_worker_url(worker_url)}/api/v1/status"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _select_worker(worker_urls: list) -> Optional[str]:
    """Pick the first reachable worker. Simple round-robin fallback."""
    for url in worker_urls:
        url = _normalize_worker_url(url)
        status = _worker_status(url)
        if status is not None:
            return url
    return None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def _multipart_post(url: str, fields: dict, files: dict, timeout: float) -> dict:
    """POST multipart/form-data using only stdlib.

    Args:
        url: Target URL
        fields: {name: value} for text fields
        files: {name: (filename, data_bytes, content_type)} for file fields
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response dict.
    """
    boundary = f"----AutoResearch{uuid.uuid4().hex}"
    body_parts = []

    for name, value in fields.items():
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        )

    for name, (filename, data, content_type) in files.items():
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        )
        body_parts.append(header)
        body_parts.append(data)
        body_parts.append(b"\r\n" if isinstance(data, bytes) else "\r\n")

    body_parts.append(f"--{boundary}--\r\n")

    # Assemble body as bytes
    body = b""
    for part in body_parts:
        if isinstance(part, str):
            body += part.encode("utf-8")
        else:
            body += part

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _worker_acquire_device(worker_url: str, task_id: str, timeout: float = 30.0) -> Optional[int]:
    """POST /api/v1/acquire_device → device_id or None."""
    url = f"{worker_url}/api/v1/acquire_device"
    try:
        resp = _multipart_post(url, {"task_id": task_id}, {}, timeout)
        return resp.get("device_id")
    except Exception as e:
        print(f"[worker] acquire_device failed: {e}", file=sys.stderr)
        return None


def _worker_release_device(worker_url: str, task_id: str, device_id: int, timeout: float = 10.0):
    """POST /api/v1/release_device."""
    url = f"{worker_url}/api/v1/release_device"
    try:
        _multipart_post(url, {"task_id": task_id, "device_id": str(device_id)}, {}, timeout)
    except Exception as e:
        print(f"[worker] release_device failed: {e}", file=sys.stderr)


def _worker_verify(worker_url: str, package: bytes, task_id: str,
                   op_name: str, timeout: float) -> dict:
    """POST /api/v1/verify with tar.gz package. Returns parsed JSON."""
    url = f"{worker_url}/api/v1/verify"
    fields = {
        "task_id": task_id,
        "op_name": op_name,
        "timeout": str(int(timeout)),
    }
    files = {
        "package": ("package.tar.gz", package, "application/gzip"),
    }
    return _multipart_post(url, fields, files, timeout=timeout + 30)


def _worker_profile(worker_url: str, package: bytes, task_id: str,
                    op_name: str, timeout: float,
                    profile_settings: Optional[dict] = None) -> dict:
    """POST /api/v1/profile with tar.gz package. Returns parsed JSON."""
    url = f"{worker_url}/api/v1/profile"
    fields = {
        "task_id": task_id,
        "op_name": op_name,
    }
    if profile_settings:
        fields["profile_settings"] = json.dumps(profile_settings)
    files = {
        "package": ("package.tar.gz", package, "application/gzip"),
    }
    return _multipart_post(url, fields, files, timeout=timeout + 30)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _assemble_eval_result(verify_resp: dict, profile_resp: dict) -> EvalResult:
    """Combine verify + profile responses into an EvalResult.

    Shared by `run_remote_eval` (HTTP transport) and `run_local_eval`
    (subprocess transport). Both transports return the same dict shape:

        verify_resp:  {"success": bool, "log": str, "artifacts": {...}}
        profile_resp: {"gen_time": float|None, "base_time": float|None,
                       "log": str, "artifacts": {...}}

    so this function is the single place that decides correctness, picks
    metrics, and computes speedup. Keeping it transport-agnostic means
    fixing a parsing bug in one place fixes it for both.
    """
    correctness = verify_resp.get("success", False)
    verify_log = verify_resp.get("log", "")

    metrics: dict = {}
    gen_time = profile_resp.get("gen_time")
    base_time = profile_resp.get("base_time")
    artifacts = profile_resp.get("artifacts", {}) or {}

    # Fallback: parse from artifact JSON files (when the transport returns
    # timing only inside result files, not in top-level fields).
    if gen_time is None and "generation_profile_result.json" in artifacts:
        try:
            gen_time = json.loads(artifacts["generation_profile_result.json"]).get("avg_time_us")
        except (json.JSONDecodeError, TypeError):
            pass
    if base_time is None and "base_profile_result.json" in artifacts:
        try:
            base_time = json.loads(artifacts["base_profile_result.json"]).get("avg_time_us")
        except (json.JSONDecodeError, TypeError):
            pass

    def _valid(v):
        return isinstance(v, (int, float)) and 0 < v < float("inf")

    gen_ok, base_ok = _valid(gen_time), _valid(base_time)
    if gen_ok:
        metrics["latency_us"] = gen_time
    else:
        print(f"[eval] WARNING: no valid gen_time (got {gen_time!r}) — "
              f"kernel profile likely failed", file=sys.stderr)
    if base_ok:
        metrics["ref_latency_us"] = base_time
    else:
        print(f"[eval] WARNING: no valid base_time (got {base_time!r}) — "
              f"speedup vs reference unavailable", file=sys.stderr)
    if gen_ok and base_ok:
        metrics["speedup_vs_ref"] = base_time / gen_time
    elif profile_resp.get("speedup"):
        metrics["speedup_vs_ref"] = profile_resp["speedup"]

    for k, v in profile_resp.items():
        if k not in ("success", "log", "gen_time", "base_time", "speedup",
                     "artifacts", "task_id", "returncode") \
                and isinstance(v, (int, float)):
            metrics[k] = v

    profile_log = profile_resp.get("log", "")
    return EvalResult(
        correctness=correctness,
        metrics=metrics,
        error=None if correctness else
              "verify failed (kernel broken); ref profile may still be present",
        raw_output=(verify_log + "\n" + profile_log)[-4096:],
    )


# ---------------------------------------------------------------------------
# Remote eval (HTTP transport)
# ---------------------------------------------------------------------------

def run_remote_eval(task_dir: str, config: TaskConfig,
                    worker_urls: Optional[list] = None) -> EvalResult:
    """Run eval via remote Worker Service.

    Flow:
      1. Select a reachable worker
      2. Acquire a device slot (the device id is baked into the package's
         generated scripts; the vendored worker doesn't override DEVICE_ID
         env at exec time, so the slot must be resolved BEFORE building).
      3. Build tar.gz package with the acquired device id baked in.
      4. POST /api/v1/verify → correctness check
      5. POST /api/v1/profile → latency metrics (always run, even on
         verify failure — the ref baseline is still useful)
      6. Release device, return EvalResult

    Compatible with the Worker Service API from ar_vendored.worker.server.
    """
    urls = worker_urls or config.worker_urls
    if not urls:
        return EvalResult(correctness=False, error="no worker_urls configured")

    urls = [_normalize_worker_url(u) for u in urls]

    # Select reachable worker
    worker_url = _select_worker(urls)
    if worker_url is None:
        return EvalResult(
            correctness=False,
            error=f"no reachable worker from: {urls}",
        )

    task_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
    print(f"[remote_eval] Using worker: {worker_url}", file=sys.stderr)

    # Acquire device BEFORE building the package. The vendored worker
    # (`ar_vendored/core/worker/local_worker.py`) by contract trusts the
    # device_id baked into the generated verify/profile scripts and does
    # NOT override DEVICE_ID env at execution time. If we built the
    # package first and acquired the device second, the package would
    # bake in `_build_package`'s default (device_id=0) and the entire
    # multi-device device_pool would be ineffective — every task would
    # land on NPU 0 regardless of which slot was acquired.
    acquired_id = _worker_acquire_device(worker_url, task_id)
    if acquired_id is None:
        print("[remote_eval] WARNING: acquire_device returned None; falling "
              "back to device 0 baked into the package. The release call "
              "in the finally block will be skipped — no device to release.",
              file=sys.stderr)
        device_id = 0
    else:
        device_id = acquired_id

    # Build package — device_id is now the acquired slot (or 0 fallback),
    # baked into the generated verify/profile scripts so the worker runs
    # them on the right card.
    try:
        package = _build_package(task_dir, config, device_id=device_id)
    except Exception as e:
        if acquired_id is not None:
            _worker_release_device(worker_url, task_id, acquired_id)
        return EvalResult(correctness=False, error=f"failed to build package: {e}")

    try:
        # Step 1: Verify (correctness check)
        print(f"[remote_eval] Running verify...", file=sys.stderr)
        try:
            verify_resp = _worker_verify(
                worker_url, package, task_id, config.name, config.eval_timeout,
            )
        except Exception as e:
            return EvalResult(correctness=False, error=f"verify request failed: {e}")

        # Step 2: Profile — ALWAYS run it, even if verify failed. The profile
        # endpoint runs both profile_base.py (PyTorch reference, uses
        # reference.py only) and profile_generation.py (the seed/kernel,
        # needs kernel.py correct). A broken kernel still lets us measure the
        # ref baseline, which is the user-facing anchor for speedup.
        print(f"[remote_eval] Running profile...", file=sys.stderr)
        try:
            profile_resp = _worker_profile(
                worker_url, package, task_id, config.name, config.eval_timeout,
            )
        except Exception as e:
            return EvalResult(
                correctness=verify_resp.get("success", False),
                metrics={},
                error=f"verify={verify_resp.get('success', False)}; "
                      f"profile request failed: {e}",
                raw_output=verify_resp.get("log", "")[-2048:],
            )

        return _assemble_eval_result(verify_resp, profile_resp)

    finally:
        # Release only what we actually acquired. The fallback path sets
        # device_id=0 without calling acquire — releasing that would
        # decrement the pool's count for a slot we never reserved.
        if acquired_id is not None:
            _worker_release_device(worker_url, task_id, acquired_id)


# ---------------------------------------------------------------------------
# Local eval (subprocess transport, same generated scripts as remote)
# ---------------------------------------------------------------------------

def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Run eval entirely in local subprocesses.

    Builds the same tar.gz package the remote worker would receive, then runs
    the auto-generated `verify_<op>.py` and `profile_<op>_*.py` scripts via
    `local_worker.local_verify` / `local_worker.local_profile`. Both
    transports converge on `_assemble_eval_result` so downstream code can't
    tell them apart.

    The pre-refactor implementation ran `config.eval_script` as a
    user-supplied entry point. That field was never set by scaffold and is
    no longer consulted; it stays in TaskConfig only for yaml back-compat.
    """
    # local_worker is a top-level script (one level up from this package).
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from local_worker import local_verify, local_profile

    if device_id is not None:
        dev = int(device_id)
    elif config.devices:
        dev = int(config.devices[0])
    else:
        # No explicit device on the call AND task.yaml has no `devices`
        # field — fall back to NPU 0. We emit a loud warning instead of
        # raising because legitimate callers (notebooks, ad-hoc reruns)
        # do hit this path, but a SILENT fallback to 0 is what once let
        # `--devices 6` get rewritten to 0 and OOM on a busy NPU. The
        # warning surfaces the implicit choice so the user can spot it
        # before sinking minutes into a wrong-card eval.
        dev = 0
        print(
            "[local_eval] WARNING: no device specified (no device_id arg, "
            "no `devices` field in task.yaml). Defaulting to NPU 0. If "
            "another card is intended, pass --device-id N or set "
            "`devices: [N]` in task.yaml.",
            file=sys.stderr,
        )
    try:
        package = _build_package(task_dir, config, device_id=dev)
    except Exception as e:
        return EvalResult(correctness=False, error=f"failed to build package: {e}")

    print(f"[local_eval] Running verify...", file=sys.stderr)
    verify_resp = local_verify(package, config.name, config.eval_timeout, dev)
    print(f"[local_eval] Running profile (dsl={config.dsl}, backend={config.backend})...",
          file=sys.stderr)
    profile_resp = local_profile(
        package, config.name, config.eval_timeout, dev,
        dsl=config.dsl, backend=config.backend,
    )
    return _assemble_eval_result(verify_resp, profile_resp)


# ---------------------------------------------------------------------------
# Unified eval entry point
# ---------------------------------------------------------------------------

def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Three-way routing:

      1. Explicit worker URLs (CLI or task.yaml) → remote.
      2. Else, if `local_worker.detect_local_backend(config.backend)`
         reports the runtime is available → local subprocess.
      3. Else → EvalResult with a clear "no execution backend" error so the
         user knows to either pass --worker-url or install the matching
         runtime (torch / torch_npu / CUDA driver).

    The local and remote branches share the same package and the same
    result-assembly function (`_assemble_eval_result`), so downstream code
    sees identical EvalResult shapes regardless of transport.
    """
    urls = worker_urls or config.worker_urls
    if urls:
        return run_remote_eval(task_dir, config, worker_urls=urls)

    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from local_worker import detect_local_backend
    backend_key = (config.backend or "cpu").lower()
    ok, why = detect_local_backend(backend_key)
    if ok:
        print(f"[eval] local backend ok ({backend_key}): {why}", file=sys.stderr)
        return run_local_eval(task_dir, config, device_id=device_id)

    return EvalResult(
        correctness=False,
        error=(
            f"no execution backend available for backend={backend_key!r}: "
            f"{why}. Either pass --worker-url to use a remote worker, or "
            f"install the matching runtime locally (torch + torch_npu for "
            f"ascend, torch + CUDA for cuda, torch alone for cpu)."
        ),
    )

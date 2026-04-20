"""
Standalone task.yaml parser + eval execution.

Parses the task config, generates the tarball shipped to the remote worker
(reference.py, kernel.py, auto-generated verify/profile scripts, cached
reference.pt), and dispatches to remote or local eval.

Only requires: stdlib + pyyaml.
"""

import io
import json
import operator as _op
import os
import subprocess
import sys
import tarfile
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import Request, urlopen

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Minimal task configuration parsed from task.yaml."""
    name: str
    description: str = ""

    # Adapter declaration
    dsl: Optional[str] = None
    framework: Optional[str] = None
    backend: Optional[str] = None
    arch: Optional[str] = None

    # Files
    eval_script: Optional[str] = None
    editable_files: list = field(default_factory=list)
    ref_file: Optional[str] = None

    # Eval params
    eval_timeout: int = 600

    # Metric
    primary_metric: str = "score"
    lower_is_better: bool = True
    improvement_threshold: float = 0.0

    # Correctness tolerance (torch.allclose against cached reference)
    correctness_atol: float = 1e-2
    correctness_rtol: float = 1e-2

    # Constraints: {metric_name: (operator_str, threshold)}
    constraints: dict = field(default_factory=dict)

    # Smoke test (optional — quick_check.py runs it before eval when configured)
    smoke_test_script: Optional[str] = None
    smoke_test_timeout: int = 10

    # Agent budget
    max_rounds: int = 30

    # Remote worker
    worker_urls: list = field(default_factory=list)
    """Worker Service URLs, e.g. ["http://127.0.0.1:9111"].
    When non-empty, eval is routed to remote workers instead of local subprocess."""


@dataclass
class EvalResult:
    """Evaluation result."""
    correctness: bool
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    raw_output: str = ""


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_task_config(task_dir: str) -> Optional[TaskConfig]:
    """Load TaskConfig from task_dir/task.yaml. Returns None if not found."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: expected YAML dict, got {type(raw).__name__}")

    name = raw.get("name")
    if not name:
        raise ValueError(f"{yaml_path}: 'name' is required")

    eval_block = raw.get("eval", {})
    metric_block = raw.get("metric", {})
    smoke_block = raw.get("smoke_test", {})
    agent_block = raw.get("agent", {})

    # Parse constraints
    constraints = {}
    for metric_name, spec in raw.get("constraints", {}).items():
        if isinstance(spec, dict):
            constraints[metric_name] = (spec["op"], spec["value"])
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            constraints[metric_name] = tuple(spec)

    # Parse worker URLs from task.yaml
    worker_block = raw.get("worker", {})
    worker_urls = worker_block.get("urls", [])
    if isinstance(worker_urls, str):
        worker_urls = [u.strip() for u in worker_urls.split(",") if u.strip()]

    return TaskConfig(
        name=name,
        description=raw.get("description", ""),
        dsl=raw.get("dsl"),
        framework=raw.get("framework"),
        backend=raw.get("backend"),
        arch=raw.get("arch"),
        eval_script=raw.get("eval_script"),
        editable_files=raw.get("editable_files", []),
        ref_file=agent_block.get("ref_file"),
        eval_timeout=eval_block.get("timeout", 600),
        primary_metric=metric_block.get("primary", "score"),
        lower_is_better=metric_block.get("lower_is_better", True),
        improvement_threshold=metric_block.get("improvement_threshold", 0.0),
        correctness_atol=metric_block.get("correctness_atol", 1e-2),
        correctness_rtol=metric_block.get("correctness_rtol", 1e-2),
        constraints=constraints,
        smoke_test_script=smoke_block.get("script"),
        smoke_test_timeout=smoke_block.get("timeout", 10),
        max_rounds=agent_block.get("max_rounds", 30),
        worker_urls=worker_urls,
    )


# ---------------------------------------------------------------------------
# Remote Worker Client (stdlib only — uses urllib + tarfile)
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


def _detect_device_type(config: TaskConfig) -> str:
    """torch.device prefix ('npu' / 'cuda' / 'cpu') derived from backend.
    Mapping lives in .autoresearch/config.yaml `backends.*.device_type`."""
    from settings import device_type_for
    return device_type_for(config.backend, fallback="cpu")


def _gen_verify_script(config: TaskConfig, device_id: int = 0) -> str:
    """Generate verify_{op_name}.py for the Worker Service.

    Two-phase precision check (AKG-style):
      - If `reference.pt` is present in the work dir, load cached PyTorch
        outputs and compare kernel against them. No PyTorch forward pass
        needed — much faster and decouples ref capture from kernel verify.
      - If `reference.pt` is missing (backward compat), fall back to
        importing Model and running it inline, like the old behavior.
    """
    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = (config.ref_file or "reference.py").replace(".py", "")
    atol = config.correctness_atol
    rtol = config.correctness_rtol
    return f'''\
#!/usr/bin/env python3
"""Auto-generated verify script for Worker Service."""
import os, sys, json, traceback

device_type = "{device}"
device_id = int(os.environ.get("DEVICE_ID", {device_id}))

if device_type == "npu":
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    import torch
    import torch_npu
    device = torch.device("npu:0")
elif device_type == "cuda":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_id))
    import torch
    device = torch.device("cuda:0")
else:
    import torch
    device = torch.device("cpu")

ATOL = {atol!r}
RTOL = {rtol!r}
REF_PT = "reference.pt"

# ModelNew is the ONLY import we require to succeed — ref is usually loaded
# from the cached .pt. Import failure here means the kernel file itself is
# broken (syntax, name missing, etc.).
try:
    from {kernel_file} import ModelNew
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"correctness": False,
                      "error": f"import failed: cannot import name 'ModelNew' from '{kernel_file}' ({{e}})"}}))
    sys.exit(1)

try:
    # --- Obtain reference outputs and inputs ---
    if os.path.exists(REF_PT):
        ref_data = torch.load(REF_PT, map_location="cpu", weights_only=False)
        ref_inputs = ref_data["inputs"]
        out_ref = ref_data["outputs"]
        # Need init_inputs for ModelNew — pull from reference.py.
        from {ref_file} import get_init_inputs
        init_inputs = get_init_inputs()
        ref_source = "cached"
    else:
        from {ref_file} import Model, get_inputs, get_init_inputs
        init_inputs = get_init_inputs()
        model_ref = Model(*init_inputs).cpu().eval()
        ref_inputs = get_inputs()
        with torch.no_grad():
            out_ref = model_ref(*ref_inputs)
        if isinstance(out_ref, torch.Tensor):
            out_ref = [out_ref]
        elif not isinstance(out_ref, (list, tuple)):
            out_ref = [out_ref]
        ref_source = "inline"

    # --- Run kernel on device with SAME inputs as ref was captured with ---
    model_new = ModelNew(*init_inputs).to(device).eval()
    inputs = [x.to(device) if hasattr(x, "to") else x for x in ref_inputs]

    with torch.no_grad():
        out_new = model_new(*inputs)

    if isinstance(out_new, torch.Tensor):
        out_new = [out_new]
    elif not isinstance(out_new, (list, tuple)):
        out_new = [out_new]

    # --- Compare ---
    all_close = True
    diagnostics = []
    for i, (r, n) in enumerate(zip(out_ref, out_new)):
        if not (isinstance(r, torch.Tensor) and isinstance(n, torch.Tensor)):
            continue
        rf = r.detach().cpu().float()
        nf = n.detach().cpu().float()
        if rf.shape != nf.shape:
            all_close = False
            diagnostics.append(f"out{{i}} shape {{tuple(rf.shape)}} != kernel {{tuple(nf.shape)}}")
            continue
        abs_diff = (rf - nf).abs()
        max_abs = abs_diff.max().item()
        # Element-wise allclose: |r - n| <= atol + rtol * |r|
        if not torch.allclose(rf, nf, atol=ATOL, rtol=RTOL):
            all_close = False
            rel_denom = rf.abs().clamp_min(1e-12)
            max_rel = (abs_diff / rel_denom).max().item()
            n_bad = ((abs_diff > (ATOL + RTOL * rf.abs())).sum().item())
            n_tot = rf.numel()
            diagnostics.append(
                f"out{{i}}: max_abs={{max_abs:.3e}} max_rel={{max_rel:.3e}} "
                f"bad_elems={{n_bad}}/{{n_tot}} ({{100.0*n_bad/n_tot:.2f}}%)"
            )
        else:
            diagnostics.append(f"out{{i}}: OK (max_abs={{max_abs:.3e}})")

    for d in diagnostics:
        print(d, file=sys.stderr)

    print(json.dumps({{
        "correctness": all_close,
        "ref_source": ref_source,
        "atol": ATOL, "rtol": RTOL,
        "diagnostics": diagnostics,
    }}))
    sys.exit(0 if all_close else 1)

except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"correctness": False, "error": str(e)}}))
    sys.exit(1)
'''


def _gen_profile_script(config: TaskConfig, device_id: int = 0,
                        mode: str = "generation",
                        warmup: int = 10, repeats: int = 100) -> str:
    """Generate profile_{op_name}_{mode}.py for the Worker Service.

    mode='base' profiles Model (reference), mode='generation' profiles ModelNew (kernel).
    """
    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = (config.ref_file or "reference.py").replace(".py", "")

    if mode == "base":
        import_line = f"from {ref_file} import Model as TargetModel, get_inputs, get_init_inputs"
    else:
        import_line = f"from {kernel_file} import ModelNew as TargetModel\nfrom {ref_file} import get_inputs, get_init_inputs"

    return f'''\
#!/usr/bin/env python3
"""Auto-generated {mode} profile script for Worker Service.

Two modes:
  - If run under msprof/nsys (Worker's profiler): just execute forward passes,
    profiler captures device-side timing. No CPU timing needed.
  - If run standalone (no profiler): use CPU timing as fallback.
"""
import os, sys, json, time

device_type = "{device}"
device_id = int(os.environ.get("DEVICE_ID", {device_id}))

if device_type == "npu":
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    import torch
    import torch_npu
    device = torch.device(f"npu:0")
elif device_type == "cuda":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_id))
    import torch
    device = torch.device(f"cuda:0")
else:
    import torch
    device = torch.device("cpu")

{import_line}

init_inputs = get_init_inputs()
model = TargetModel(*init_inputs).to(device)

inputs = get_inputs()
inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]

warmup_times = {warmup}
run_times = {repeats}

def benchmark_fn():
    with torch.no_grad():
        return model(*inputs)

# Try profiler_npu for accurate device-side timing (matches AKG's profiling)
profiler_available = False
if device_type == "npu":
    try:
        from akg_agents.op.verifier.profiler import profiler_npu
        profiler_available = True
    except ImportError:
        pass

if profiler_available:
    execution_time_us = profiler_npu(
        benchmark_fn,
        warmup=warmup_times,
        active=run_times,
        prof_dir_name="prof_{mode}_output",
        keep_res=False,
        suppress_warnings=True,
        clear_l2_cache=True,
        dsl="triton_ascend"
    )
    # profiler_npu returns inf when trace data is missing (e.g., VEC-only kernels)
    if execution_time_us < float('inf'):
        execution_time_ms = execution_time_us / 1000
        method = "profiler_npu"
        avg_us = execution_time_us
        profiler_available = True  # keep as True
    else:
        profiler_available = False  # fallback below

if not profiler_available:
    # Fallback: triton.testing.do_bench or manual timing
    try:
        import triton.testing
        execution_time_ms = triton.testing.do_bench(
            benchmark_fn, warmup=warmup_times, rep=run_times, return_mode="min"
        )
        avg_us = execution_time_ms * 1000
        method = "triton_do_bench"
    except Exception:
        # Last resort: manual CPU timing
        for _ in range(warmup_times):
            benchmark_fn()
        if device_type == "npu":
            torch.npu.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(run_times):
            if device_type == "npu":
                torch.npu.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            benchmark_fn()
            if device_type == "npu":
                torch.npu.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)
        avg_us = sum(times) / len(times)
        execution_time_ms = avg_us / 1000
        method = "cpu_timer"

result_data = {{
    "avg_time_us": avg_us,
    "execution_time_us": avg_us,
    "execution_time_ms": execution_time_ms,
    "warmup_times": warmup_times,
    "run_times": run_times,
    "method": method,
}}
result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "{mode}_profile_result.json")
with open(result_file, "w") as f:
    json.dump(result_data, f, indent=2)
print(f"PROFILE_RESULT: {{avg_us}}")
'''


def _build_package(task_dir: str, config: TaskConfig, device_id: int = 0) -> bytes:
    """Build a tar.gz package with worker-compatible scripts.

    Generates and includes:
      - verify_{op_name}.py     (correctness check)
      - profile_{op_name}_base.py (reference timing)
      - profile_{op_name}_generation.py (kernel timing)
      - kernel.py, reference.py, and any support .py files
    """
    op_name = config.name
    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add editable files
        for fname in config.editable_files:
            fpath = os.path.join(task_dir, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)

        # Add reference file
        if config.ref_file:
            ref_path = os.path.join(task_dir, config.ref_file)
            if os.path.exists(ref_path):
                tar.add(ref_path, arcname=config.ref_file)

        # Add any other .py files in task_dir root (support files)
        for f in os.listdir(task_dir):
            if (f.endswith(".py")
                    and f not in config.editable_files
                    and f != config.ref_file
                    and not f.startswith(".")):
                fpath = os.path.join(task_dir, f)
                if os.path.isfile(fpath):
                    tar.add(fpath, arcname=f)

        # Cached PyTorch reference outputs (AKG-style): verify script loads
        # this if present and skips running Model inline. Placed at the root
        # of the tarball as `reference.pt` so the worker's extract dir has
        # it alongside the scripts.
        ref_pt = os.path.join(task_dir, ".ar_state", "reference.pt")
        if os.path.isfile(ref_pt):
            tar.add(ref_pt, arcname="reference.pt")

        # Generate and add worker scripts
        def _add_script(name: str, content: str):
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        _add_script(f"verify_{op_name}.py",
                     _gen_verify_script(config, device_id))
        _add_script(f"profile_{op_name}_base.py",
                     _gen_profile_script(config, device_id, mode="base"))
        _add_script(f"profile_{op_name}_generation.py",
                     _gen_profile_script(config, device_id, mode="generation"))

    return buf.getvalue()


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


def run_remote_eval(task_dir: str, config: TaskConfig,
                    worker_urls: Optional[list] = None) -> EvalResult:
    """Run eval via remote Worker Service.

    Flow:
      1. Select a reachable worker
      2. Build tar.gz package (editable files + reference)
      3. POST /api/v1/verify → correctness check
      4. If correct: POST /api/v1/profile → latency metrics
      5. Return EvalResult

    Compatible with the Worker Service API from akg_agents.worker.server.
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

    # Build package
    try:
        package = _build_package(task_dir, config)
    except Exception as e:
        return EvalResult(correctness=False, error=f"failed to build package: {e}")

    # Acquire device
    device_id = _worker_acquire_device(worker_url, task_id)
    if device_id is None:
        print("[remote_eval] WARNING: acquire_device returned None, proceeding anyway",
              file=sys.stderr)

    try:
        # Step 1: Verify (correctness check)
        print(f"[remote_eval] Running verify...", file=sys.stderr)
        try:
            verify_resp = _worker_verify(
                worker_url, package, task_id, config.name, config.eval_timeout,
            )
        except Exception as e:
            return EvalResult(correctness=False, error=f"verify request failed: {e}")

        correctness = verify_resp.get("success", False)
        log = verify_resp.get("log", "")

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
                correctness=correctness,
                metrics={},
                error=f"verify={correctness}; profile request failed: {e}",
                raw_output=log[-2048:],
            )

        # Parse profile metrics — try top-level fields first, then artifacts
        metrics = {}
        gen_time = profile_resp.get("gen_time")
        base_time = profile_resp.get("base_time")

        # Fallback: parse from artifacts JSON files (when profiler returns
        # timing in result files rather than top-level response fields)
        artifacts = profile_resp.get("artifacts", {})
        if gen_time is None and "generation_profile_result.json" in artifacts:
            try:
                gen_data = json.loads(artifacts["generation_profile_result.json"])
                gen_time = gen_data.get("avg_time_us")
            except (json.JSONDecodeError, TypeError):
                pass
        if base_time is None and "base_profile_result.json" in artifacts:
            try:
                base_data = json.loads(artifacts["base_profile_result.json"])
                base_time = base_data.get("avg_time_us")
            except (json.JSONDecodeError, TypeError):
                pass

        # Worker returns inf when a profile script didn't produce a result file
        # (cross-backend, script crash, missing JSON). Treat inf/None/<=0 as
        # "no data" so downstream doesn't compute bogus speedups.
        def _valid(v):
            return isinstance(v, (int, float)) and 0 < v < float("inf")

        gen_ok, base_ok = _valid(gen_time), _valid(base_time)
        if gen_ok:
            metrics["latency_us"] = gen_time
        else:
            print(f"[remote_eval] WARNING: no valid gen_time (got {gen_time!r}) — kernel profile likely failed",
                  file=sys.stderr)
        if base_ok:
            metrics["ref_latency_us"] = base_time
        else:
            print(f"[remote_eval] WARNING: no valid base_time (got {base_time!r}) — speedup vs reference unavailable",
                  file=sys.stderr)
        if gen_ok and base_ok:
            metrics["speedup_vs_ref"] = base_time / gen_time
        elif profile_resp.get("speedup"):
            metrics["speedup_vs_ref"] = profile_resp["speedup"]

        # Also capture any extra numeric metrics from profile response
        for k, v in profile_resp.items():
            if k not in ("success", "log", "gen_time", "base_time", "speedup",
                         "artifacts", "task_id") and isinstance(v, (int, float)):
                metrics[k] = v

        profile_log = profile_resp.get("log", "")

        # Propagate verify failure via correctness=False, but keep any ref
        # timing we managed to capture. Downstream (_baseline_init.py,
        # keep_or_discard) treats correctness=False as FAIL, which is exactly
        # what a broken kernel should produce.
        return EvalResult(
            correctness=correctness,
            metrics=metrics,
            error=None if correctness else "verify failed (kernel broken); ref profile may still be present",
            raw_output=(log + "\n" + profile_log)[-4096:],
        )

    finally:
        # Always release device
        if device_id is not None:
            _worker_release_device(worker_url, task_id, device_id)


# ---------------------------------------------------------------------------
# Local eval execution (subprocess-based)
# ---------------------------------------------------------------------------

def _resolve_eval_command(task_dir: str, config: TaskConfig) -> Optional[list]:
    """Determine the eval command to run."""
    if config.eval_script:
        eval_script = os.path.join(task_dir, config.eval_script)
        if not os.path.exists(eval_script):
            return None
        return [sys.executable, eval_script]
    return None


def _resolve_env(config: TaskConfig, device_id: Optional[int] = None) -> dict:
    """Build environment variables for eval subprocess."""
    env = os.environ.copy()
    if device_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    return env


def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Run eval via local subprocess.

    Eval script protocol:
      - stdout last line must be JSON: {"correctness": true/false, "latency_us": ...}
      - exit code 0 = correctness pass, 1 = fail or crash
    """
    cmd = _resolve_eval_command(task_dir, config)
    if cmd is None:
        return EvalResult(
            correctness=False,
            error=f"eval script not found: {config.eval_script or '(none specified)'}",
        )

    env = _resolve_env(config, device_id)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=config.eval_timeout, cwd=task_dir, env=env,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(correctness=False, error=f"eval timed out after {config.eval_timeout}s")
    except Exception as e:
        return EvalResult(correctness=False, error=f"eval failed to launch: {e}")

    raw_output = (result.stdout or "") + (result.stderr or "")

    # Contract: eval script prints a JSON object as its last stdout line.
    from phase_machine import parse_last_json_line  # lazy: avoid import cycle
    data = parse_last_json_line(result.stdout)
    if data is not None:
        correctness = data.get("correctness", False)
        metrics = {k: v for k, v in data.items() if k != "correctness"}
        error = None
    else:
        correctness = False
        metrics = {}
        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            stdout_tail = (result.stdout or "")[-500:]
            error = f"exit code {result.returncode}\n{stderr_tail}\n{stdout_tail}".strip()
        else:
            error = "eval produced no JSON output"

    return EvalResult(correctness=correctness, metrics=metrics, error=error, raw_output=raw_output)


# ---------------------------------------------------------------------------
# Unified eval entry point
# ---------------------------------------------------------------------------

def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Run eval — automatically routes to remote worker or local subprocess.

    Priority:
      1. If worker_urls is passed explicitly → remote eval
      2. If config.worker_urls is non-empty → remote eval
      3. Otherwise → local subprocess eval
    """
    urls = worker_urls or config.worker_urls
    if urls:
        return run_remote_eval(task_dir, config, worker_urls=urls)
    return run_local_eval(task_dir, config, device_id=device_id)


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------

_CONSTRAINT_OPS = {"<=": _op.le, ">=": _op.ge, "<": _op.lt, ">": _op.gt, "==": _op.eq}


def check_constraints(result: EvalResult, constraints: dict) -> list:
    """Check hard constraints. Returns list of violation strings (empty = ok)."""
    violations = []
    for metric_name, (op_str, threshold) in constraints.items():
        func = _CONSTRAINT_OPS.get(op_str)
        if func is None:
            violations.append(f"{metric_name}: unknown operator '{op_str}'")
            continue
        value = result.metrics.get(metric_name)
        if value is None:
            violations.append(f"{metric_name}: metric missing (required {op_str} {threshold})")
            continue
        if not isinstance(value, (int, float)):
            violations.append(f"{metric_name}: non-numeric value {value!r}")
            continue
        if not func(value, threshold):
            violations.append(f"{metric_name}: {value} violates {op_str} {threshold}")
    return violations


def is_improvement(
    current: EvalResult,
    best: EvalResult,
    metric: str = "latency_ms",
    lower_is_better: bool = True,
    threshold: float = 0.0,
) -> bool:
    """Check if current result improves on best.

    threshold is a relative percentage (e.g. 2.0 = needs >2% improvement).
    """
    if not current.correctness:
        return False
    cur_val = current.metrics.get(metric)
    best_val = best.metrics.get(metric)
    if cur_val is None:
        return False
    if best_val is None:
        return True
    if best_val == 0:
        return cur_val < 0 if lower_is_better else cur_val > 0
    if lower_is_better:
        relative_pct = (best_val - cur_val) / abs(best_val) * 100
    else:
        relative_pct = (cur_val - best_val) / abs(best_val) * 100
    return relative_pct > threshold


def format_result_summary(result: EvalResult) -> str:
    """Human-readable one-line summary."""
    if not result.correctness:
        if result.error:
            return f"FAILED: {result.error}"
        return f"CORRECTNESS FAILED (metrics: {result.metrics})"
    parts = ["correctness: PASS"]
    for key, val in result.metrics.items():
        if isinstance(val, float):
            parts.append(f"{key}: {val:.4f}")
        else:
            parts.append(f"{key}: {val}")
    return "  |  ".join(parts)

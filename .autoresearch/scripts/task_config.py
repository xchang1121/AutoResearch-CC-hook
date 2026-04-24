"""
Standalone task.yaml parser + eval execution.

Parses the task config, builds the tar.gz package (reference.py + editable
files + auto-generated verify/profile scripts), and dispatches eval to
either the remote worker (HTTP) or the in-process local backend
(`local_worker`). Both transports consume the same package and converge on
`_assemble_eval_result`. Reference outputs are never shipped: worker
self-caches them under /tmp/ar_cache/<op>_<sha(reference.py)>/reference.pt
on first verify; local backend recomputes on each run.

Only requires: stdlib + pyyaml.
"""

import hashlib
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
    editable_files: list = field(default_factory=list)
    ref_file: str = "reference.py"

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

    # CodeChecker (static analysis on editable files).
    # Default on; disable per-task via `code_checker.enabled: false` in
    # task.yaml or scaffold's --no-code-checker flag. When off, quick_check
    # and validate_kernel skip the AST/import/DSL pipeline but still reject
    # the scaffold TODO placeholder.
    code_checker_enabled: bool = True

    # Agent budget
    max_rounds: int = 30

    # Remote worker
    worker_urls: list = field(default_factory=list)
    """Worker Service URLs, e.g. ["http://127.0.0.1:9111"].
    When non-empty, eval is routed to remote workers instead of local subprocess."""

    # Local devices
    devices: list = field(default_factory=list)
    """Device IDs for local eval (written by scaffold from --devices). When
    non-empty, run_local_eval uses devices[0] as default device_id."""


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
    code_checker_block = raw.get("code_checker", {})

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

    # Parse devices list. Accepts [5] / "5" / "0,1,2".
    devices_raw = raw.get("devices", [])
    if isinstance(devices_raw, int):
        devices = [devices_raw]
    elif isinstance(devices_raw, str):
        devices = [int(d.strip()) for d in devices_raw.split(",") if d.strip()]
    elif isinstance(devices_raw, list):
        devices = [int(d) for d in devices_raw]
    else:
        devices = []

    return TaskConfig(
        name=name,
        description=raw.get("description", ""),
        dsl=raw.get("dsl"),
        framework=raw.get("framework"),
        backend=raw.get("backend"),
        arch=raw.get("arch"),
        editable_files=raw.get("editable_files", []),
        ref_file=agent_block.get("ref_file") or "reference.py",
        eval_timeout=eval_block.get("timeout", 600),
        primary_metric=metric_block.get("primary", "score"),
        lower_is_better=metric_block.get("lower_is_better", True),
        improvement_threshold=metric_block.get("improvement_threshold", 0.0),
        correctness_atol=metric_block.get("correctness_atol", 1e-2),
        correctness_rtol=metric_block.get("correctness_rtol", 1e-2),
        constraints=constraints,
        smoke_test_script=smoke_block.get("script"),
        smoke_test_timeout=smoke_block.get("timeout", 10),
        code_checker_enabled=bool(code_checker_block.get("enabled", True)),
        max_rounds=agent_block.get("max_rounds", 30),
        worker_urls=worker_urls,
        devices=devices,
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
    """torch.device prefix ('npu' / 'cuda' / 'cpu'). Derived from DSL via
    hw_detect (DSL → backend → device_type)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from hw_detect import device_type_for_dsl
    try:
        return device_type_for_dsl(config.dsl or "")
    except Exception:
        return "cpu"


def _get_dsl_adapter(dsl: Optional[str]):
    """Return the vendored DSL adapter for `dsl`. Raises if unknown.

    Cached-once import — the factory touches all DSL adapter modules on first
    call, which pulls in pandas/numpy/etc. Keep the import local to callers
    that need it, not at module scope.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from ar_vendored.op.verifier.adapters.factory import get_dsl_adapter
    return get_dsl_adapter(dsl or "triton_ascend")


def _gen_verify_script(config: TaskConfig, device_id: int = 0,
                       worker_ref_path: str = "") -> str:
    """Generate verify_{op_name}.py for the Worker Service.

    Reference outputs live in a worker-local cache keyed by op_name + sha of
    reference.py (path passed as `worker_ref_path`). On first verify the
    script computes Model on the target device, writes the cache, and uses
    the result. On subsequent verifies it loads the cache. The cache is
    invalidated automatically when reference.py content changes (different
    sha → different path).

    DSL adapter (ar_vendored.op.verifier.adapters) supplies DSL-specific
    imports (triton autotune patches, tilelang compile patches, etc.) via
    `get_import_statements`. The verify body itself is uniform across DSLs:
    instantiate ModelNew, one forward, allclose vs cached reference.
    """
    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = config.ref_file.replace(".py", "")
    atol = config.correctness_atol
    rtol = config.correctness_rtol

    adapter = _get_dsl_adapter(config.dsl)
    dsl_imports = adapter.get_import_statements(config.framework or "torch")
    dsl_setup = adapter.get_special_setup_code() if hasattr(adapter, "get_special_setup_code") else ""

    return f'''\
#!/usr/bin/env python3
"""Auto-generated verify script (dsl={config.dsl}, backend={config.backend}).

Worker-side reference cache: if WORKER_REF_PT exists, load it; otherwise
run Model on device once, save outputs there, then use them. No local
PyTorch forward — local client never runs Model.
"""
import os, sys, json, traceback

# ar_vendored is bundled at the tarball root (same dir as this script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# DSL-specific imports (triton / tilelang patches, etc.)
{dsl_imports}
{dsl_setup}

ATOL = {atol!r}
RTOL = {rtol!r}
WORKER_REF_PT = {worker_ref_path!r}

try:
    from {kernel_file} import ModelNew
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"correctness": False,
                      "error": f"import failed: cannot import name 'ModelNew' from '{kernel_file}' ({{e}})"}}))
    sys.exit(1)

try:
    # Inputs are regenerated here via get_inputs() — deterministic seed, so
    # cache keyed only on reference.py content is sufficient.
    from {ref_file} import get_inputs, get_init_inputs
    init_inputs = get_init_inputs()
    ref_inputs_cpu = get_inputs()

    # --- Obtain reference outputs (cache-or-compute, worker-side only) ---
    if WORKER_REF_PT and os.path.isfile(WORKER_REF_PT):
        ref_data = torch.load(WORKER_REF_PT, map_location="cpu", weights_only=False)
        out_ref = ref_data["outputs"]
        ref_source = "cached-worker"
    else:
        from {ref_file} import Model
        model_ref = Model(*init_inputs).to(device).eval()
        ref_inputs_dev = [x.to(device) if hasattr(x, "to") else x for x in ref_inputs_cpu]
        with torch.no_grad():
            out_ref_raw = model_ref(*ref_inputs_dev)
        if isinstance(out_ref_raw, torch.Tensor):
            out_ref = [out_ref_raw.detach().cpu()]
        elif isinstance(out_ref_raw, (list, tuple)):
            out_ref = [o.detach().cpu() for o in out_ref_raw]
        else:
            out_ref = [out_ref_raw]
        if WORKER_REF_PT:
            try:
                os.makedirs(os.path.dirname(WORKER_REF_PT), exist_ok=True)
                torch.save({{"outputs": out_ref}}, WORKER_REF_PT)
            except Exception as _e:
                print(f"[verify] WARNING: ref cache write failed ({{WORKER_REF_PT}}): {{_e}}",
                      file=sys.stderr)
        ref_source = "computed-worker"
        # Free the ref model before ModelNew allocates — BatchNorm-scale
        # tensors on HBM don't fit both at once.
        del model_ref, out_ref_raw, ref_inputs_dev
        if device_type == "npu":
            torch.npu.empty_cache()
        elif device_type == "cuda":
            torch.cuda.empty_cache()

    # --- Run kernel on device with the same deterministic inputs ---
    model_new = ModelNew(*init_inputs).to(device).eval()
    inputs = [x.to(device) if hasattr(x, "to") else x for x in ref_inputs_cpu]

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
    """Generate profile_{op_name}_{mode}.py, adapter-driven.

    Structure:
      1. Outer skeleton (device setup, model instantiation) — uniform.
      2. Adapter-supplied `get_import_statements` + `get_special_setup_code`
         — DSL-specific imports and one-time patches (triton autotune,
         tilelang compile).
      3. Adapter-supplied `benchmark_impl` — the timing block. For
         triton_ascend this wraps `profiler_npu` (torch_npu.profiler); for
         triton_cuda / tilelang_cuda it's `triton.testing.do_bench`; for
         ascendc / cuda_c it's empty (those DSLs rely on msprof/nsys, routed
         at local_worker.py, not here).
      4. Fallback timing block — used when adapter's benchmark_impl is
         empty or crashes at runtime.

    mode='base' profiles Model (reference); mode='generation' profiles
    ModelNew (kernel).
    """
    import textwrap

    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = config.ref_file.replace(".py", "")

    if mode == "base":
        target_import = (f"from {ref_file} import Model as TargetModel, "
                         f"get_inputs, get_init_inputs")
    else:
        target_import = (f"from {kernel_file} import ModelNew as TargetModel\n"
                         f"from {ref_file} import get_inputs, get_init_inputs")

    adapter = _get_dsl_adapter(config.dsl)
    dsl_imports = adapter.get_import_statements(config.framework or "torch")
    dsl_setup = adapter.get_special_setup_code() if hasattr(adapter, "get_special_setup_code") else ""

    # Adapter's benchmark_impl returns a code string indented 8-space for
    # upstream's kernel_verifier (which calls it inside a `for case` loop).
    # Dedent to column 0, then re-indent at 4-space for our function body.
    raw = adapter.benchmark_impl(
        impl_func_name="TargetModel", inputs="inputs",
        warmup=warmup, runs=repeats,
        backend=config.backend or "", op_name=config.name,
        case_idx=0, device_id=device_id,
    )
    if raw and raw.strip():
        benchmark_body = textwrap.indent(textwrap.dedent(raw), "    ")
        benchmark_source = f"adapter ({type(adapter).__name__})"
    else:
        # Adapter had no benchmark (ascendc / cuda_c): do_bench fallback so
        # the local subprocess still produces a timing. Real msprof/nsys
        # goes through local_worker.py when backend matches.
        benchmark_body = textwrap.indent(textwrap.dedent(f"""\
            import triton.testing
            def _bench():
                with torch.no_grad():
                    return impl_model(*inputs)
            execution_time_ms = triton.testing.do_bench(
                _bench, warmup={warmup}, rep={repeats}, return_mode="min")
            execution_time_us = execution_time_ms * 1000
            method = "triton_do_bench (adapter has no benchmark_impl)"
        """), "    ")
        benchmark_source = "fallback-do_bench"

    return f'''\
#!/usr/bin/env python3
"""Auto-generated {mode} profile script (dsl={config.dsl}, benchmark={benchmark_source})."""
import os, sys, json, time

# ar_vendored is bundled at tarball root (same dir as this script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# --- DSL-specific imports + patches (from adapter.get_import_statements) ---
{dsl_imports}
{dsl_setup}

{target_import}

init_inputs = get_init_inputs()
impl_model = TargetModel(*init_inputs)
if hasattr(impl_model, "to"):
    impl_model = impl_model.to(device)
if hasattr(impl_model, "eval"):
    impl_model.eval()

inputs = get_inputs()
inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]

def _run_adapter_benchmark():
    # Variables the adapter code assigns:
    #   execution_time_us / execution_time_ms / method
    execution_time_us = None
    execution_time_ms = None
    method = None
{benchmark_body}
    if execution_time_us is None and execution_time_ms is not None:
        execution_time_us = execution_time_ms * 1000
    if execution_time_ms is None and execution_time_us is not None:
        execution_time_ms = execution_time_us / 1000
    return execution_time_us, execution_time_ms, method

try:
    avg_us, execution_time_ms, method = _run_adapter_benchmark()
    if avg_us is None or avg_us <= 0 or avg_us == float("inf"):
        raise RuntimeError(f"adapter benchmark returned invalid avg_us={{avg_us!r}}")
except Exception as e:
    import traceback
    print(f"[profile {mode}] adapter benchmark failed: {{e}}; falling back to cpu timer",
          file=sys.stderr)
    traceback.print_exc()
    for _ in range({warmup}):
        with torch.no_grad():
            impl_model(*inputs)
    if device_type == "npu":
        torch.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range({repeats}):
        if device_type == "npu":
            torch.npu.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            impl_model(*inputs)
        if device_type == "npu":
            torch.npu.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    avg_us = sum(times) / len(times)
    execution_time_ms = avg_us / 1000
    method = "cpu_timer_fallback"

result_data = {{
    "avg_time_us": avg_us,
    "execution_time_us": avg_us,
    "execution_time_ms": execution_time_ms,
    "warmup_times": {warmup},
    "run_times": {repeats},
    "method": method,
}}
result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "{mode}_profile_result.json")
with open(result_file, "w") as f:
    json.dump(result_data, f, indent=2)
print(f"PROFILE_RESULT: {{avg_us}}")
'''


_WORKER_CACHE_ROOT = "/tmp/ar_cache"


def _compute_worker_ref_path(task_dir: str, config: TaskConfig) -> str:
    """Stable worker-side cache path, keyed by op_name + sha(reference.py).

    Different reference.py content → different path → automatic invalidation.
    Same reference.py across many kernel iterations → cache hits after round 1.
    """
    ref_file = config.ref_file
    ref_full = os.path.join(task_dir, ref_file)
    if os.path.isfile(ref_full):
        with open(ref_full, "rb") as f:
            ref_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    else:
        ref_hash = "unknown"
    return f"{_WORKER_CACHE_ROOT}/{config.name}_{ref_hash}/reference.pt"


def _build_package(task_dir: str, config: TaskConfig, device_id: int = 0) -> bytes:
    """Build a tar.gz package with worker-compatible scripts.

    Generates and includes:
      - verify_{op_name}.py     (correctness check, self-caches ref on worker)
      - profile_{op_name}_base.py (reference timing)
      - profile_{op_name}_generation.py (kernel timing)
      - kernel.py, reference.py, and any support .py files

    Reference outputs are NEVER shipped in the tarball — worker computes them
    on first verify and caches under /tmp/ar_cache/<op>_<ref_sha>/reference.pt.
    """
    op_name = config.name
    buf = io.BytesIO()
    worker_ref_path = _compute_worker_ref_path(task_dir, config)

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add editable files
        for fname in config.editable_files:
            fpath = os.path.join(task_dir, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)

        # Add reference file (always set; default is "reference.py")
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

        # Generate and add worker scripts
        def _add_script(name: str, content: str):
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        _add_script(f"verify_{op_name}.py",
                     _gen_verify_script(config, device_id,
                                        worker_ref_path=worker_ref_path))
        _add_script(f"profile_{op_name}_base.py",
                     _gen_profile_script(config, device_id, mode="base"))
        _add_script(f"profile_{op_name}_generation.py",
                     _gen_profile_script(config, device_id, mode="generation"))

        # Bundle the vendored adapter/profiler tree at tarball root. Generated
        # verify/profile scripts prepend sys.path with their own dir, so
        # `import ar_vendored` resolves without any PYTHONPATH setup on the
        # worker side. ~150 KB compressed — acceptable overhead per eval.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vendored_root = os.path.join(script_dir, "ar_vendored")
        if os.path.isdir(vendored_root):
            tar.add(vendored_root, arcname="ar_vendored",
                    filter=_exclude_pycache)

    return buf.getvalue()


def _exclude_pycache(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    """tarfile.add filter: skip __pycache__ / *.pyc / editor temp files."""
    base = os.path.basename(tarinfo.name)
    if base == "__pycache__" or base.endswith(".pyc") or base.startswith("."):
        return None
    return tarinfo


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


def run_remote_eval(task_dir: str, config: TaskConfig,
                    worker_urls: Optional[list] = None) -> EvalResult:
    """Run eval via remote Worker Service.

    Flow:
      1. Select a reachable worker
      2. Build tar.gz package (editable files + reference)
      3. POST /api/v1/verify → correctness check
      4. If correct: POST /api/v1/profile → latency metrics
      5. Return EvalResult

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
        # Always release device
        if device_id is not None:
            _worker_release_device(worker_url, task_id, device_id)


# ---------------------------------------------------------------------------
# Local eval execution (subprocess-based, same generated scripts as remote)
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
    from local_worker import local_verify, local_profile

    if device_id is not None:
        dev = int(device_id)
    elif config.devices:
        dev = int(config.devices[0])
    else:
        dev = 0
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

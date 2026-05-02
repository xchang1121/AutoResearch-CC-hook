"""Verify/profile script generation + tar.gz package assembly.

The generated scripts are the contract between this client and the
remote worker (or local subprocess runner). Both transports unpack the
same tarball and run the same auto-generated `verify_<op>.py` /
`profile_<op>_<mode>.py`. This file owns:

  - DSL-adapter resolution (`_get_dsl_adapter`, `_detect_device_type`).
  - The verify-script template (`_gen_verify_script`).
  - The profile-script template (`_gen_profile_script`).
  - Ref-cache path computation (`_compute_worker_ref_path`).
  - Tarball assembly (`_build_package`) + the pycache/dotfile filter.

What's NOT here:
  - HTTP transport / device pool / `run_*_eval` — those live in
    eval_client; this module only produces bytes for them to ship.
  - Metric comparison / EvalResult — those live in metric_policy; the
    generated scripts emit JSON that eval_client parses, not us.
"""
import hashlib
import io
import os
import sys
import tarfile
from typing import Optional

from .loader import TaskConfig


# ---------------------------------------------------------------------------
# DSL / device-type resolution (kept here because only the script generators
# need them — eval_client never touches DSL adapters directly).
# ---------------------------------------------------------------------------

def _detect_device_type(config: TaskConfig) -> str:
    """torch.device prefix ('npu' / 'cuda' / 'cpu'). Derived from DSL via
    hw_detect (DSL → backend → device_type)."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from ar_vendored.op.verifier.adapters.factory import get_dsl_adapter
    return get_dsl_adapter(dsl or "triton_ascend")


# ---------------------------------------------------------------------------
# Verify script template
# ---------------------------------------------------------------------------

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

    # --- Compare (delegated to shared correctness module) ---
    # `correctness.py` is bundled into the tarball at root by
    # _build_package; both this generated script and the batch verifier
    # call into the same `compare_outputs` so semantics can't drift.
    from correctness import compare_outputs
    cmp_result = compare_outputs(list(out_ref), list(out_new), ATOL, RTOL)

    for d in cmp_result["diagnostics"]:
        print(d, file=sys.stderr)

    print(json.dumps({{
        "correctness": cmp_result["correctness"],
        "ref_source": ref_source,
        "atol": cmp_result["atol"], "rtol": cmp_result["rtol"],
        "diagnostics": cmp_result["diagnostics"],
    }}))
    sys.exit(0 if cmp_result["correctness"] else 1)

except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"correctness": False, "error": str(e)}}))
    sys.exit(1)
'''


# ---------------------------------------------------------------------------
# Profile script template
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tarball assembly
# ---------------------------------------------------------------------------

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


def _exclude_pycache(tarinfo: tarfile.TarInfo):
    """tarfile.add filter: skip __pycache__ / *.pyc / editor temp files."""
    base = os.path.basename(tarinfo.name)
    if base == "__pycache__" or base.endswith(".pyc") or base.startswith("."):
        return None
    return tarinfo


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

        # Shared correctness module — imported by the generated verify
        # script via `from correctness import compare_outputs`. Bundled at
        # tarball root so the worker subprocess can resolve it from the
        # same sys.path entry verify_{op}.py inserts.
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        correctness_src = os.path.join(script_dir, "correctness.py")
        if os.path.isfile(correctness_src):
            tar.add(correctness_src, arcname="correctness.py")

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
        # task_config package lives one level inside scripts/, so go up two.
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vendored_root = os.path.join(script_dir, "ar_vendored")
        if os.path.isdir(vendored_root):
            tar.add(vendored_root, arcname="ar_vendored",
                    filter=_exclude_pycache)

    return buf.getvalue()

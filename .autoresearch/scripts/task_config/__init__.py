"""task_config package — facade over four single-concern submodules.

The previous flat task_config.py was 1099 lines mixing YAML parsing,
verify/profile script code generation, tar.gz building, HTTP transport,
local subprocess execution, and metric arithmetic. Each grew its own
section over time but they didn't actually share data structures or
helpers — splitting them surfaces the dependency direction and lets
each layer be reasoned about (and changed) independently.

Layout:

    loader            — TaskConfig dataclass + load_task_config (YAML
                        parsing). No internal deps; everyone else
                        consumes TaskConfig from here.
    metric_policy     — EvalResult, is_improvement, check_constraints,
                        format_result_summary. Pure data + arithmetic;
                        no I/O. Imported by keep_or_discard, dashboard.
    package_builder   — DSL-adapter resolution, verify/profile script
                        generation, tar.gz assembly. Depends on loader.
    eval_client       — Worker URL discovery, HTTP transport, local-subprocess
                        transport, run_eval dispatcher, result assembly.
                        Depends on loader + metric_policy + package_builder.

This `__init__.py` re-exports every public name the previous flat module
exposed, so existing importers (`from task_config import TaskConfig`,
etc.) continue to work without modification. New code may prefer
sub-module imports for readability:

    from task_config.metric_policy import EvalResult, is_improvement
    from task_config.eval_client    import run_eval
"""
# fmt: off
from .loader import (
    TaskConfig, load_task_config,
)
from .metric_policy import (
    EvalResult, check_constraints, is_improvement, format_result_summary,
)
from .package_builder import (
    _build_package, _gen_verify_script, _gen_profile_script,
    _compute_worker_ref_path, _exclude_pycache,
    _detect_device_type, _get_dsl_adapter,
)
from .eval_client import (
    run_eval, run_remote_eval, run_local_eval,
    _normalize_worker_url, _worker_status, _select_worker,
    _multipart_post,
    _worker_acquire_device, _worker_release_device,
    _worker_verify, _worker_profile,
    _assemble_eval_result,
)
# fmt: on

"""Vendored evaluation stack for claude-autoresearch.

Zero-runtime-dep subset autoresearch needs: DSL adapters, profiler +
msprof/nsys + roofline runners, the HTTP worker server with its
DevicePool, and the triton / tilelang patch modules.

Layout (relative imports intact):

    ar_vendored/
        op/verifier/
            adapters/{dsl,backend,framework}/*.py, factory.py
            profiler.py, profiler_utils.py, l2_cache_clear.py, roofline_utils.py
        op/utils/
            triton_autotune_patch.py, tilelang_compile_patch.py
        core/worker/{interface,local_worker}.py
        core/async_pool/device_pool.py
        worker/server.py
        utils/process_utils.py
"""

import os


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

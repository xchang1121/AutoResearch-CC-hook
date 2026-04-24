"""Vendored subset of akg_agents for claude-autoresearch.

Zero-runtime-dep mirror of akg_agents/op/verifier + utils needed by task_config
to generate verify/profile scripts and by local_worker to run msprof/nsys.

Source of truth: akg-hitl/akg_agents/python/akg_agents/{op/verifier,op/utils,
op/tools,utils/process_utils}. Keep this tree byte-identical to upstream except
for the akg_agents → ar_vendored package rename — that keeps future rebases
trivial (`cp -r` + a single sed).

Layout mirrors upstream to preserve relative imports:

    ar_vendored/
        op/verifier/
            adapters/{dsl,backend,framework}/*.py
            factory.py
            profiler.py, profiler_utils.py, l2_cache_clear.py, roofline_utils.py
        op/utils/
            triton_autotune_patch.py, tilelang_compile_patch.py
        op/tools/
            calc_trace_span.py
        utils/
            process_utils.py
"""

import os


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

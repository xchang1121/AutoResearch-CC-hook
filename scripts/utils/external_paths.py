# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single source for paths that live OUTSIDE this scripts/ package.

The workspace uses the repository-owned skill tree:

    scripts/         <- this package
    skills/          <- per-DSL skill markdown

`OP_AUTORESEARCH_AR_SKILLS_ROOT` (set in .claude/settings.json) overrides the
relative fallback so the resolution works both when the slash command
runs from this dir and when callers cd elsewhere.

CA's `eval_dir()` (its vendored eval package) has no analogue here: the
verifier lives at `op_autoresearch.op.verifier` and is reached via
``utils.eval_bridge``, not via a filesystem path.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_SKILLS = os.path.join(_WS_ROOT, "skills")


def skills_dir() -> str:
    """Per-DSL skill tree. A relative OP_AUTORESEARCH_AR_SKILLS_ROOT is resolved
    against _WS_ROOT (the dir it's written relative to), not the process
    cwd — hooks/pipeline/quick_check run from assorted cwds, and a
    cwd-relative `..` would Glob a dead tree."""
    env = os.environ.get("OP_AUTORESEARCH_AR_SKILLS_ROOT")
    if not env:
        return _DEFAULT_SKILLS
    if os.path.isabs(env):
        return env
    return os.path.abspath(os.path.join(_WS_ROOT, env))

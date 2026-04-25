# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Shared safe-extraction helper for tar archives delivered to workers.

Both ``ar_vendored.core.worker.local_worker`` (HTTP / async path) and the
top-level ``local_worker`` script (sync subprocess path) accept tarballs
built by ``task_config._build_package`` and must reject any member that
escapes the destination directory or pulls in symlinks / hardlinks.
Centralizing the rule here means a future security fix is applied once,
not duplicated.
"""
from __future__ import annotations

import os
import tarfile


def safe_tar_extract(tar: tarfile.TarFile, dst_dir: str) -> None:
    """Extract ``tar`` into ``dst_dir``, raising ``ValueError`` on any member
    whose resolved path escapes ``dst_dir`` or that is not a regular file or
    directory.
    The check uses absolute path prefix matching (``startswith(root + sep)``)
    rather than ``os.path.commonpath`` so it stays correct on Windows where
    case folding can confuse common-path comparisons.
    """
    root = os.path.abspath(dst_dir)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(root, member.name))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"unsafe tar member path: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsafe tar member type: {member.name!r}")
    tar.extractall(root)

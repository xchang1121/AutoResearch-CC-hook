"""Shared config loader for framework reference tables.

Every backend/DSL/arch mapping, hallucinated-script alias, and worker-only
module list lives in `.autoresearch/config.yaml`. This module reads that
file once per process and exposes small typed accessors — callers never
hand-build these tables inside Python modules.
"""
from functools import lru_cache
import os
from typing import Dict, Optional

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "config.yaml"))


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and several modules depend on it."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{_CONFIG_PATH}: expected top-level mapping")
    return data


def default_backend() -> str:
    return str(_raw().get("default_backend", "ascend"))


def backends() -> Dict[str, dict]:
    return dict(_raw().get("backends", {}))


def backend_preset(name: Optional[str]) -> dict:
    """Return the preset dict for `name`. Unknown name → empty dict (callers
    decide whether to fall back)."""
    if not name:
        return {}
    return dict(backends().get(name.lower(), {}))


def device_type_for(backend: Optional[str], fallback: str = "cpu") -> str:
    return backend_preset(backend).get("device_type", fallback)


def worker_only_modules() -> frozenset:
    return frozenset(_raw().get("worker_only_modules", []))


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))

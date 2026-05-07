"""Hardware detection + DSL→backend mapping.

The goal: user picks `--dsl` + (`--devices N` or `--worker-url URL`), and we
derive the rest. `--backend` and `--arch` are never user-facing — backend is
a pure function of DSL; arch is a property of the hardware.

Three resolution paths:

    local:   --devices N  →  npu-smi / nvidia-smi / uname -m  →  arch
    remote:  --worker-url →  GET /api/v1/status              →  {backend, arch, devices}
    none:    ERROR

This module has no autoresearch/task dependencies — it only needs stdlib +
subprocess + urllib. Safe to import from scaffold, baseline, or anywhere.

Also provides `auto_*` helpers used by `ar_cli.py worker --start --backend
auto --arch auto --devices auto`. Selection rules — see each function's
docstring for the exact contract:

  backend:  npu-smi → ascend; nvidia-smi → cuda; both / neither → error
  devices:  list all cards; drop those whose HBM/memory used > 1 GiB OR
            utilization > 5%; pick the LOWEST surviving id (deterministic
            across runs — important for a long-lived worker daemon)
  arch:     run derive_arch on the picked device
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Optional
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# DSL → backend (hardcoded; DSL name literally encodes target backend)
# ---------------------------------------------------------------------------

_DSL_BACKEND = {
    "triton_ascend":   "ascend",
    "triton_cuda":     "cuda",
    "ascendc":         "ascend",
    "cuda_c":          "cuda",
    "tilelang_cuda":   "cuda",
    "tilelang_npuir":  "ascend",
    "pypto":           "ascend",
    "swft":            "ascend",
    "cpp":             "cpu",
    "torch":           "cpu",
}

_BACKEND_DEVICE_TYPE = {
    "ascend": "npu",    # torch.device("npu:N") via torch_npu
    "cuda":   "cuda",
    "cpu":    "cpu",
}


def backend_for_dsl(dsl: str) -> str:
    key = dsl.lower()
    if key not in _DSL_BACKEND:
        raise ValueError(f"Unknown DSL {dsl!r}; known: {sorted(_DSL_BACKEND)}")
    return _DSL_BACKEND[key]


def list_supported_dsls() -> tuple:
    """Sorted tuple of all DSL names autoresearch knows about.

    Single source of truth for the DSL menu surfaced to LLM-facing text
    (scaffold --help, parse_args missing-fields payload, slash-command
    docs). Earlier each surface hardcoded its own copy and they drifted
    silently — parse_args had `<...>` ellipsis that signaled "list is
    non-exhaustive, invent if needed", which is exactly the misread we
    want to prevent.
    """
    return tuple(sorted(_DSL_BACKEND))


def device_type_for_backend(backend: str) -> str:
    key = backend.lower()
    if key not in _BACKEND_DEVICE_TYPE:
        raise ValueError(f"Unknown backend {backend!r}")
    return _BACKEND_DEVICE_TYPE[key]


def device_type_for_dsl(dsl: str) -> str:
    return device_type_for_backend(backend_for_dsl(dsl))


# ---------------------------------------------------------------------------
# Arch derivation from local hardware
# ---------------------------------------------------------------------------

def derive_arch(backend: str, device_id: int) -> Optional[str]:
    """Return arch string (e.g. 'ascend910b3', 'a100', 'x86_64') or None
    if detection fails. Caller decides whether None is fatal."""
    backend = backend.lower()
    if backend == "ascend":
        return _npu_arch(device_id)
    if backend == "cuda":
        return _cuda_arch(device_id)
    if backend == "cpu":
        return _cpu_arch()
    return None


def _npu_arch(device_id: int) -> Optional[str]:
    """Parse `npu-smi info` main table for the NPU's Name column.

    The table row looks like:
        | 5     910B3               | Alarm         | ...
    We match the leading `<device_id>` and capture the next token →
    'ascend910b3'. `npu-smi info -t board -i N` exposes Product/Model but
    not the architecture string, so we go through the main table instead.
    """
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    pat = re.compile(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", re.MULTILINE)
    m = pat.search(r.stdout)
    if not m:
        return None
    name = m.group(1).strip().lower()
    # Names come back as '910b3', '910b4', '910b2', '310p3', etc. — prefix
    # with 'ascend' to match the ROOFLINE_ARCH_CONFIGS keys.
    return f"ascend{name}"


def _cuda_arch(device_id: int) -> Optional[str]:
    """nvidia-smi → common arch shorthand (a100 / h100 / etc.).
    Fallback to the full name if we don't recognize it."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             "-i", str(device_id)],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    name = r.stdout.strip().lower()
    for token in ("a100", "h100", "a800", "h800", "v100", "t4",
                  "rtx4090", "rtx3090", "l40", "l4"):
        if token in name.replace(" ", "").replace("-", ""):
            return token
    return name or None


def _cpu_arch() -> Optional[str]:
    try:
        r = subprocess.run(["uname", "-m"], capture_output=True, text=True,
                           timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


# ---------------------------------------------------------------------------
# Auto-selection (worker --start --backend auto / --arch auto / --devices auto)
# ---------------------------------------------------------------------------

# Threshold for "is this device idle". HBM/memory used in MiB; util in %.
# 1 GiB headroom because torch_npu / cuda runtime alone reserves a few hundred
# MiB on init even when no real work is queued.
_BUSY_MEM_MIB = 1024
_BUSY_UTIL_PCT = 5


class HwDetectError(RuntimeError):
    """Auto-selection failed for a reason worth reporting to the user
    verbatim (not enough info to fall back silently)."""


def auto_select_backend() -> str:
    """Detect ascend vs cuda by tool presence.

    Rule:
      - npu-smi in PATH and nvidia-smi NOT in PATH  →  'ascend'
      - nvidia-smi in PATH and npu-smi NOT in PATH  →  'cuda'
      - both in PATH                                →  HwDetectError
        (mixed-host machines do exist; refuse to guess)
      - neither                                     →  HwDetectError

    CPU is never auto-selected — opting in to backend=cpu requires an
    explicit --backend cpu, since CPU is a fallback target, not a default.
    """
    has_npu = shutil.which("npu-smi") is not None
    has_cuda = shutil.which("nvidia-smi") is not None
    if has_npu and has_cuda:
        raise HwDetectError(
            "both npu-smi and nvidia-smi are in PATH; cannot auto-pick a "
            "backend. Pass --backend ascend or --backend cuda explicitly.")
    if has_npu:
        return "ascend"
    if has_cuda:
        return "cuda"
    raise HwDetectError(
        "neither npu-smi nor nvidia-smi is in PATH; cannot auto-detect "
        "backend. Pass --backend explicitly (ascend / cuda / cpu).")


def list_devices(backend: str) -> list[dict]:
    """Return [{'id': int, 'busy': bool, 'detail': str}, ...] sorted by id.

    `busy` reflects current occupancy (HBM/memory > 1 GiB OR utilization
    > 5%). `detail` is a short human-readable string suitable for the
    'all cards busy' error message.

    Returns [] if the smi tool is missing or unparseable — callers should
    treat that as "can't auto-pick" and surface the error to the user.
    """
    backend = backend.lower()
    if backend == "ascend":
        return _list_ascend_devices()
    if backend == "cuda":
        return _list_cuda_devices()
    if backend == "cpu":
        # CPU has a single implicit "device 0". Always idle.
        return [{"id": 0, "busy": False, "detail": "cpu"}]
    raise HwDetectError(f"unknown backend {backend!r}")


def _list_ascend_devices() -> list[dict]:
    """Parse `npu-smi info` main table.

    The relevant columns are 'NPU' (id), 'HBM-Usage(MB)', 'AICore (%)'.
    npu-smi's table format is column-aligned ASCII with `|` separators;
    rows we care about look like:

      | 0     910B3      | OK    | ... |  3253       /  ... |
      | 0     0          | 0000:.. | 12345 / 65536  | 0   /  0  |

    The "stat" line (alarm/health) and the "data" line (memory/util) come
    in pairs per device. We pair them by NPU id.
    """
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True,
                           text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []

    # Two-row layout: the first row of each pair starts with the device id
    # and the device name; the second starts with the device id and the
    # chip-id (always 0 for single-die parts). Memory + AICore live on the
    # second row.
    devices: dict[int, dict] = {}
    name_re = re.compile(r"^\|\s*(\d+)\s+(\S+)\s*\|", re.MULTILINE)
    util_re = re.compile(
        r"^\|\s*(\d+)\s+\d+\s*\|.*?\|\s*(\d+)\s*/\s*\d+\s*\|"
        r"\s*(\d+)\s*/\s*(\d+)\s*\|", re.MULTILINE)

    for m in name_re.finditer(r.stdout):
        idx = int(m.group(1))
        name = m.group(2).strip()
        # Skip header rows ('NPU', 'Chip' etc.) — name token is non-numeric
        # and looks like an arch code (910B3, 310P3, ...).
        if not re.match(r"^\d{3}[A-Za-z]\d?$", name):
            continue
        devices.setdefault(idx, {"id": idx, "name": name,
                                 "mem_mib": None, "util_pct": None})

    for m in util_re.finditer(r.stdout):
        idx = int(m.group(1))
        if idx not in devices:
            continue
        try:
            devices[idx]["util_pct"] = int(m.group(2))
            devices[idx]["mem_mib"] = int(m.group(3))
        except ValueError:
            pass

    out: list[dict] = []
    for idx in sorted(devices):
        d = devices[idx]
        mem = d.get("mem_mib")
        util = d.get("util_pct")
        busy = ((mem is not None and mem > _BUSY_MEM_MIB)
                or (util is not None and util > _BUSY_UTIL_PCT))
        bits = [f"{d['name']}"]
        if mem is not None:
            bits.append(f"HBM={mem}MiB")
        if util is not None:
            bits.append(f"AICore={util}%")
        out.append({"id": idx, "busy": busy, "detail": " ".join(bits)})
    return out


def _list_cuda_devices() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []

    out: list[dict] = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            mem_used = int(parts[2])      # MiB
            util_pct = int(parts[4])      # %
        except ValueError:
            continue
        name = parts[1]
        busy = mem_used > _BUSY_MEM_MIB or util_pct > _BUSY_UTIL_PCT
        out.append({
            "id": idx, "busy": busy,
            "detail": f"{name} mem={mem_used}MiB util={util_pct}%",
        })
    out.sort(key=lambda d: d["id"])
    return out


def auto_select_device(backend: str) -> int:
    """Return the lowest-id device that's NOT busy.

    Rationale for "lowest id" (over "least loaded"):
      - The worker is a long-lived daemon. We want re-runs of `--devices
        auto` on the same machine to land on the same card across days,
        so logs / msprof traces / bug reports stay comparable.
      - "Least loaded" tracks instantaneous metrics that flicker, so two
        consecutive `--start` calls would frequently pick different cards.
      - Within the busy/idle bucket the id is a stable identifier; outside
        users override with `--devices N` if they care about a specific id.

    Raises HwDetectError if smi parsing failed (cannot enumerate) or all
    cards are busy (cannot decide for the user — refusing to fall back).
    """
    devs = list_devices(backend)
    if not devs:
        raise HwDetectError(
            f"could not enumerate {backend} devices "
            f"(smi tool missing or output unparseable). "
            f"Pass --devices N explicitly.")
    free = [d for d in devs if not d["busy"]]
    if not free:
        rows = "\n".join(f"  card {d['id']}: {d['detail']}  [BUSY]"
                         for d in devs)
        raise HwDetectError(
            f"all {backend} devices look busy (HBM > {_BUSY_MEM_MIB}MiB or "
            f"util > {_BUSY_UTIL_PCT}%):\n{rows}\n"
            f"Pass --devices N explicitly to override, or wait for one to "
            f"free up.")
    return free[0]["id"]


def auto_select_arch(backend: str, device_id: int) -> str:
    """Wrapper around derive_arch that turns None into a clear error
    instead of letting downstream code propagate it as a silent default."""
    arch = derive_arch(backend, device_id)
    if arch is None:
        raise HwDetectError(
            f"could not detect arch for {backend} device {device_id} "
            f"({'npu-smi' if backend == 'ascend' else 'nvidia-smi'} "
            f"unavailable or output not recognized). "
            f"Pass --arch <name> explicitly.")
    return arch


# ---------------------------------------------------------------------------
# Worker status fetch (remote path)
# ---------------------------------------------------------------------------

def fetch_worker_hardware(worker_url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET /api/v1/status on the worker. Returns a dict like
    {"status": "ready", "backend": "ascend", "arch": "ascend910b3",
     "devices": [5]} or None on failure.
    """
    url = worker_url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/") + "/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data

#!/usr/bin/env python3
"""Stop an AutoResearch Worker Service daemon by TCP port.

Finds the listening process with `ss` (or `lsof` fallback), sanity-checks
that its command line contains `ar_vendored.worker.server`, and sends
SIGTERM (then SIGKILL after 3s if it's still alive). Refuses to touch a
process that doesn't look like our worker — so accidentally reusing the
port for something else can't blow away that something else.

Example:
    python .autoresearch/scripts/server_related/stop_worker_service.py --port 9056
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional


def _find_pid_on_port(port: int) -> Optional[int]:
    """Return PID listening on `port`, or None. Tries ss, then lsof."""
    # ss (iproute2) — present on almost every modern Linux distro.
    try:
        out = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        out = ""
    for line in out.splitlines():
        # Local Address:Port column ends with ":<port>"
        fields = line.split()
        if len(fields) < 5:
            continue
        local = fields[3]
        if not local.endswith(f":{port}"):
            continue
        # Last column: users:(("python",pid=12345,fd=6),...)
        last = fields[-1]
        marker = "pid="
        idx = last.find(marker)
        if idx == -1:
            continue
        tail = last[idx + len(marker):]
        num = ""
        for ch in tail:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return int(num)

    # lsof fallback (macOS, or minimal Linux images without ss).
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if out:
            return int(out.splitlines()[0])
    except FileNotFoundError:
        pass

    return None


def _cmdline_of(pid: int) -> str:
    """Best-effort command line string for `pid`."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        if raw:
            return raw.strip()
    except OSError:
        pass
    try:
        return subprocess.run(
            ["ps", "-p", str(pid), "-o", "cmd="],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except FileNotFoundError:
        return ""


def _still_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stop an AutoResearch Worker Service daemon by port.",
    )
    p.add_argument("--port", type=int, required=True,
                   help="TCP port the daemon is bound to.")
    p.add_argument("--force", action="store_true",
                   help="Skip the cmdline safety check and SIGTERM anyway. "
                        "Use only if the cmdline lookup fails but you're "
                        "sure the PID is your worker.")
    args = p.parse_args()

    pid = _find_pid_on_port(args.port)
    if pid is None:
        print(f"No process listening on port {args.port}.")
        return 0

    cmd = _cmdline_of(pid)
    if "ar_vendored.worker.server" not in cmd and not args.force:
        print(f"ERROR: PID {pid} on port {args.port} does not look like an "
              f"ar_vendored worker:\n  {cmd or '(cmdline unavailable)'}\n"
              f"Refusing to kill. Use --force to override, or `kill {pid}` "
              f"manually if you're sure.", file=sys.stderr)
        return 2

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} already gone.")
        return 0

    for _ in range(12):   # up to 3s total
        if not _still_alive(pid):
            print(f"Stopped PID {pid} (port {args.port}).")
            return 0
        time.sleep(0.25)

    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Force-killed PID {pid} (port {args.port}).")
        return 0
    except ProcessLookupError:
        print(f"Stopped PID {pid} (port {args.port}).")
        return 0
    except PermissionError as e:
        print(f"WARNING: cannot SIGKILL PID {pid}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

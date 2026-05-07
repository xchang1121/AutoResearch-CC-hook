#!/usr/bin/env python3
"""AutoResearch CLI — single entry for worker / future subcommands.

Canonical invocation (from project root):

    python .autoresearch/scripts/ar_cli.py worker --start \
        --backend ascend --arch ascend910b3 --devices 2,5 \
        --host 127.0.0.1 --port 9111 --bg

    python .autoresearch/scripts/ar_cli.py worker --status --port 9111
    python .autoresearch/scripts/ar_cli.py worker --stop   --port 9111

The CLI is cross-platform (daemon mode uses start_new_session on POSIX /
DETACHED_PROCESS on Windows). Prerequisites are the user's: activate a
Python env where `fastapi + uvicorn + pyyaml + torch` (plus torch_npu /
triton / pandas / msprof / nsys per DSL) are importable — ar_cli itself
does not activate anything.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


SCRIPTS_DIR = Path(__file__).resolve().parent   # .autoresearch/scripts/

sys.path.insert(0, str(SCRIPTS_DIR))
import hw_detect  # noqa: E402


# ---------------------------------------------------------------------------
# worker subcommand
# ---------------------------------------------------------------------------

def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _worker_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["WORKER_BACKEND"] = args.backend
    env["WORKER_ARCH"] = args.arch
    env["WORKER_DEVICES"] = args.devices
    env["WORKER_PORT"] = str(args.port)
    env["WORKER_HOST"] = args.host
    return env


def _banner(args: argparse.Namespace, extra: Optional[dict] = None) -> str:
    rows = [
        ("Host", args.host), ("Backend", args.backend), ("Arch", args.arch),
        ("Devices", args.devices), ("Port", str(args.port)),
    ]
    if extra:
        rows.extend((k, str(v)) for k, v in extra.items())
    w = max(len(k) for k, _ in rows) + 1
    return "\n".join(f"  {k:<{w}}: {v}" for k, v in rows)


def _tail(path: Path, n: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError as e:
        return f"(cannot read {path}: {e})"


def _resolve_auto(args: argparse.Namespace) -> None:
    """Resolve `auto` placeholders for --backend / --devices / --arch in
    place. Order matters: backend first (devices/arch depend on it), then
    devices (arch depends on the picked device), then arch.

    Selection rules (also documented in hw_detect.auto_*):
      backend auto → npu-smi only → 'ascend';  nvidia-smi only → 'cuda';
                     both / neither → error (refuse to guess).
      devices auto → list every card via npu-smi/nvidia-smi, drop those
                     with HBM/mem > 1 GiB OR util > 5%, pick the
                     lowest-id survivor (deterministic across re-runs).
                     If all cards are busy → error (refuse to evict).
      arch auto    → derive from `npu-smi info` row (Name column) or
                     `nvidia-smi --query-gpu=name` for the picked device.

    Each step prints a one-line note to stderr so the user sees what was
    chosen. On any failure we exit with the HwDetectError message.
    """
    try:
        if args.backend == "auto":
            args.backend = hw_detect.auto_select_backend()
            print(f"[auto] backend = {args.backend}", file=sys.stderr)

        if args.devices == "auto":
            picked = hw_detect.auto_select_device(args.backend)
            args.devices = str(picked)
            print(f"[auto] devices = {args.devices}", file=sys.stderr)

        if args.arch == "auto":
            # If --devices is a comma list we derive arch from the first id
            # — all cards in one host are the same arch in practice.
            first_dev = int(args.devices.split(",")[0])
            args.arch = hw_detect.auto_select_arch(args.backend, first_dev)
            print(f"[auto] arch    = {args.arch}", file=sys.stderr)
    except hw_detect.HwDetectError as e:
        sys.exit(f"auto-detect failed: {e}")


def _worker_start(args: argparse.Namespace) -> int:
    _resolve_auto(args)
    if args.bg:
        return _worker_start_daemon(args)

    print("=" * 48)
    print("AutoResearch Worker Service (foreground)")
    print("-" * 48)
    print(_banner(args))
    print("=" * 48, flush=True)
    os.environ.update({k: v for k, v in _worker_env(args).items()
                       if k.startswith("WORKER_")})
    sys.path.insert(0, str(SCRIPTS_DIR))
    from ar_vendored.worker.server import start_server
    start_server(host=args.host, port=args.port)
    return 0


def _worker_start_daemon(args: argparse.Namespace) -> int:
    if _port_in_use(args.port):
        print(f"ERROR: port {args.port} is already in use. Stop the existing "
              f"daemon (python .autoresearch/scripts/ar_cli.py worker --stop "
              f"--port {args.port}) or pick another port.", file=sys.stderr)
        return 1

    log_path = Path(f"/tmp/ar_worker_{args.port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab", buffering=0)

    cmd = [sys.executable, "-m", "ar_vendored.worker.server"]
    popen_kwargs: dict = {
        "cwd": str(SCRIPTS_DIR),
        "env": _worker_env(args),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": log_file,
        "close_fds": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_PGROUP

    proc = subprocess.Popen(cmd, **popen_kwargs)
    log_file.close()

    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"ERROR: worker exited with code {proc.returncode}. "
                  f"Log tail:\n{_tail(log_path, 40)}", file=sys.stderr)
            return proc.returncode or 1
        if _port_in_use(args.port):
            break
        time.sleep(0.25)
    else:
        print(f"ERROR: worker PID {proc.pid} did not start listening on "
              f"{args.host}:{args.port} within 30s. Log tail:\n"
              f"{_tail(log_path, 40)}", file=sys.stderr)
        return 1

    print("=" * 48)
    print("AutoResearch Worker Service (daemon)")
    print("-" * 48)
    print(_banner(args, {"PID": proc.pid, "Log": log_path}))
    print("-" * 48)
    print(f"  Stop: python .autoresearch/scripts/ar_cli.py worker "
          f"--stop --port {args.port}")
    print("=" * 48)
    return 0


def _find_pid_on_port(port: int) -> Optional[int]:
    try:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True,
                             check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        out = ""
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[3].endswith(f":{port}"):
            continue
        idx = fields[-1].find("pid=")
        if idx == -1:
            continue
        num = ""
        for ch in fields[-1][idx + 4:]:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return int(num)

    try:
        out = subprocess.run(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True,
                             check=False).stdout.strip()
        if out:
            return int(out.splitlines()[0])
    except FileNotFoundError:
        pass
    return None


def _cmdline_of(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        if raw:
            return raw.strip()
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-p", str(pid), "-o", "cmd="],
                              capture_output=True, text=True,
                              check=False).stdout.strip()
    except FileNotFoundError:
        return ""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _worker_stop(args: argparse.Namespace) -> int:
    pid = _find_pid_on_port(args.port)
    if pid is None:
        print(f"No process listening on port {args.port}.")
        return 0

    cmd = _cmdline_of(pid)
    if "ar_vendored.worker.server" not in cmd and not args.force:
        print(f"ERROR: PID {pid} on port {args.port} does not look like an "
              f"ar_vendored worker:\n  {cmd or '(cmdline unavailable)'}\n"
              f"Refusing to kill. Use --force, or `kill {pid}` manually.",
              file=sys.stderr)
        return 2

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} already gone.")
        return 0

    for _ in range(12):
        if not _alive(pid):
            print(f"Stopped PID {pid} (port {args.port}).")
            return 0
        time.sleep(0.25)

    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Force-killed PID {pid} (port {args.port}).")
    except ProcessLookupError:
        print(f"Stopped PID {pid} (port {args.port}).")
    except PermissionError as e:
        print(f"WARNING: cannot SIGKILL PID {pid}: {e}", file=sys.stderr)
        return 1
    return 0


def _worker_status(args: argparse.Namespace) -> int:
    url = f"http://{args.host}:{args.port}/api/v1/status"
    try:
        with urlopen(Request(url, method="GET"), timeout=5) as resp:
            body = resp.read().decode()
    except Exception as e:
        print(f"Worker on {args.host}:{args.port} unreachable: {e}",
              file=sys.stderr)
        return 1

    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(body)
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    # --start / --stop / --status are mutually exclusive; argparse enforces it.
    # `--host` default is action-dependent: bind on 0.0.0.0 for --start,
    # connect to 127.0.0.1 for --stop/--status (0.0.0.0 is bind-only and
    # cannot be the target of an outbound TCP connect).
    if args.host is None:
        args.host = "0.0.0.0" if args.start else "127.0.0.1"
    if args.start:
        return _worker_start(args)
    if args.stop:
        return _worker_stop(args)
    if args.status:
        return _worker_status(args)
    # No action flag given → show help for `worker`.
    print("worker: specify --start, --stop, or --status.\n"
          "Run `python .autoresearch/scripts/ar_cli.py worker --help` for "
          "details.", file=sys.stderr)
    return 2


def _add_worker_subcommand(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "worker",
        help="Manage the vendored Worker Service (HTTP eval server).",
        description="Start / stop / check the vendored AutoResearch Worker "
                    "Service on this machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mx = p.add_mutually_exclusive_group(required=False)
    mx.add_argument("--start", action="store_true",
                    help="Start the worker on this machine.")
    mx.add_argument("--stop", action="store_true",
                    help="Stop the daemon listening on --port.")
    mx.add_argument("--status", action="store_true",
                    help="Curl /api/v1/status on --host:--port.")

    p.add_argument("--backend", default="auto",
                   choices=["ascend", "cuda", "cpu", "auto"],
                   help="Hardware backend (default: auto). 'auto' picks "
                        "ascend if npu-smi is in PATH, cuda if nvidia-smi "
                        "is in PATH; if both/neither are present, fails "
                        "and asks for an explicit value.")
    p.add_argument("--arch", default="auto",
                   help="Arch string, e.g. ascend910b3 / a100 / x86_64 "
                        "(default: auto). 'auto' derives the arch from "
                        "`npu-smi info` (Name column) or `nvidia-smi "
                        "--query-gpu=name` for the picked device.")
    p.add_argument("--devices", default="auto",
                   help="Comma-separated device IDs, e.g. '2,5' (default: "
                        "auto). 'auto' enumerates all cards, drops those "
                        "with HBM/mem > 1 GiB or util > 5%%, and picks the "
                        "LOWEST-id idle card (deterministic across re-runs "
                        "— important for a long-running daemon). If every "
                        "card is busy, fails rather than evicting one.")
    p.add_argument("--host", default=None,
                   help="Bind / probe address. Defaults to 0.0.0.0 for "
                        "--start (all interfaces) and 127.0.0.1 for "
                        "--status / --stop (loopback connect).")
    p.add_argument("--port", type=int, default=9001,
                   help="TCP port (default: 9001).")
    p.add_argument("--bg", action="store_true",
                   help="Daemon mode for --start. Detaches, logs to "
                        "/tmp/ar_worker_<port>.log, prints PID + log path.")
    p.add_argument("--force", action="store_true",
                   help="For --stop: skip the ar_vendored.worker.server "
                        "cmdline safety check.")
    p.set_defaults(func=_cmd_worker)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        prog="ar_cli",
        description="AutoResearch CLI. Subcommands: worker.",
    )
    sub = p.add_subparsers(dest="command", metavar="{worker}")
    _add_worker_subcommand(sub)

    args = p.parse_args()
    if not args.command:
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

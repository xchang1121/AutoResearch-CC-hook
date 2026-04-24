#!/usr/bin/env python3
"""Start the vendored AutoResearch Worker Service.

The worker itself is `ar_vendored.worker.server` (FastAPI + uvicorn). This
wrapper handles CLI parsing, env var population, optional daemonization,
and a port-conflict pre-check. Replaces the old bash script.

Examples:
    # Foreground (Ctrl-C to stop):
    python .autoresearch/scripts/server_related/start_worker_service.py \\
        --backend ascend --arch ascend910b3 --devices 5 --port 9056

    # Daemon (writes /tmp/ar_worker_<port>.log, prints PID, returns):
    python .autoresearch/scripts/server_related/start_worker_service.py \\
        --backend ascend --arch ascend910b3 --devices 5 --port 9056 --bg

    # Multi-card CUDA:
    python .autoresearch/scripts/server_related/start_worker_service.py \\
        --backend cuda --arch a100 --devices 0,1,2,3 --port 9001 --bg

Zero dependency on akg_agents. Prerequisites (user's responsibility; the
script does not activate anything): the Python that runs this file must be
able to `import fastapi, uvicorn, pyyaml, torch`; add torch_npu / triton /
pandas per DSL. For ascendc / cuda_c the `msprof` / `nsys` CLIs must be on
PATH.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


# .autoresearch/scripts/ — must be on sys.path before uvicorn imports
# `ar_vendored.worker.server`.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if `host:port` is already accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _build_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["WORKER_BACKEND"] = args.backend
    env["WORKER_ARCH"] = args.arch
    env["WORKER_DEVICES"] = args.devices
    env["WORKER_PORT"] = str(args.port)
    env["WORKER_HOST"] = args.host
    return env


def _banner(args: argparse.Namespace, extra: dict | None = None) -> str:
    rows = [
        ("Host",    args.host),
        ("Backend", args.backend),
        ("Arch",    args.arch),
        ("Devices", args.devices),
        ("Port",    str(args.port)),
    ]
    if extra:
        rows.extend((k, str(v)) for k, v in extra.items())
    width = max(len(k) for k, _ in rows) + 1
    lines = [f"  {k:<{width}}: {v}" for k, v in rows]
    return "\n".join(lines)


def _run_foreground(args: argparse.Namespace) -> int:
    """Run the worker in-process (exec-equivalent). Uvicorn takes over."""
    print("=" * 48)
    print("AutoResearch Worker Service (foreground)")
    print("-" * 48)
    print(_banner(args))
    print("=" * 48, flush=True)
    # Put WORKER_* into our own env, then import + run. No subprocess hop.
    os.environ.update({k: v for k, v in _build_env(args).items()
                       if k.startswith("WORKER_")})
    # Ensure the vendored package is importable from here.
    sys.path.insert(0, str(SCRIPTS_DIR))
    from ar_vendored.worker.server import start_server
    start_server(host=args.host, port=args.port)
    return 0


def _run_daemon(args: argparse.Namespace) -> int:
    """Spawn uvicorn in a detached subprocess with stdio → log file.

    Cross-platform (uses start_new_session / DETACHED_PROCESS instead of
    fork). After launching we poll the port for up to 30s so the caller
    gets a clear success/failure rather than a "maybe started" PID.
    """
    if _port_in_use(args.port):
        print(f"ERROR: port {args.port} is already in use. "
              f"Stop the existing daemon or pick another port.",
              file=sys.stderr)
        return 1

    log_path = Path(f"/tmp/ar_worker_{args.port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab", buffering=0)

    cmd = [sys.executable, "-m", "ar_vendored.worker.server"]

    popen_kwargs: dict = {
        "cwd": str(SCRIPTS_DIR),
        "env": _build_env(args),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": log_file,
        "close_fds": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True   # new session, survives parent exit
    else:
        # Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        popen_kwargs["creationflags"] = 0x00000008 | 0x00000200

    proc = subprocess.Popen(cmd, **popen_kwargs)
    log_file.close()   # child owns the fd now

    # Poll for readiness. If the child exits before the port opens,
    # surface the log so the user knows why.
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

    stop_script = Path(__file__).resolve().parent / "stop_worker_service.py"
    print("=" * 48)
    print("AutoResearch Worker Service (daemon)")
    print("-" * 48)
    print(_banner(args, {"PID": proc.pid, "Log": log_path}))
    print("-" * 48)
    print(f"  Stop: python {stop_script} --port {args.port}")
    print("=" * 48)
    return 0


def _tail(path: Path, n: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError as e:
        return f"(cannot read {path}: {e})"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Start the vendored AutoResearch Worker Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", default="ascend",
                   choices=["ascend", "cuda", "cpu"],
                   help="Hardware backend (default: ascend).")
    p.add_argument("--arch", default="ascend910b4",
                   help="Arch string (e.g. ascend910b3 / a100 / x86_64). "
                        "Default: ascend910b4.")
    p.add_argument("--devices", default="0",
                   help="Comma-separated device IDs, e.g. '5' or '0,1,2,3'. "
                        "Default: 0.")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("WORKER_PORT", 9001)),
                   help="TCP port (default: 9001 or $WORKER_PORT).")
    p.add_argument("--host", default=os.environ.get("WORKER_HOST", "0.0.0.0"),
                   help="Bind address. 0.0.0.0 (default) / 127.0.0.1 / :: .")
    p.add_argument("--bg", action="store_true",
                   help="Daemon mode. Detaches, writes log to "
                        "/tmp/ar_worker_<port>.log, prints PID, returns.")

    args = p.parse_args()
    return _run_daemon(args) if args.bg else _run_foreground(args)


if __name__ == "__main__":
    sys.exit(main())

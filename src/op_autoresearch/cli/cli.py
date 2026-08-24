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

"""Standalone worker lifecycle CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

from op_autoresearch.cli.service import remote_dispatch
from op_autoresearch.cli.service.worker_config import (
    WorkerConfig,
    probe_local_arch,
)
from op_autoresearch.cli.utils.paths import get_process_log_dir
from op_autoresearch.cli.utils.worker_state import (
    get_worker_entry,
    live_worker_pid,
    load_worker_state,
    pid_alive,
    remove_worker_entry,
    save_worker_state,
    set_worker_entry,
    terminate_pid,
)
from op_autoresearch.core.worker.eval_config import eval_defaults
from op_autoresearch.utils.process_utils import reap_orphaned_process_groups


def _status(port: int, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/status", timeout=timeout
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _registry_path(port: int) -> str:
    return str(Path(tempfile.gettempdir()) /
               f"op_autoresearch_worker_{port}_process_groups.json")


def _stop_local(port: int) -> int:
    state = load_worker_state()
    entry = get_worker_entry(state, port)
    pid = live_worker_pid(port)
    if (pid is None and entry and isinstance(entry.get("pid"), int)
            and pid_alive(entry["pid"])):
        pid = entry["pid"]
    if pid:
        terminate_pid(pid, timeout=eval_defaults().kill_grace_s)
    os.environ["OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY"] = _registry_path(port)
    reaped = reap_orphaned_process_groups()
    remove_worker_entry(state, port)
    save_worker_state(state)
    print(json.dumps({"stopped_pid": pid, "reaped_process_groups": reaped}))
    return 0


def _start_local(args, cfg: WorkerConfig) -> int:
    existing = _status(args.port, cfg.timing.status_timeout)
    if existing and existing.get("status") == "ready":
        print(json.dumps(existing, indent=2))
        return 0
    if live_worker_pid(args.port):
        _stop_local(args.port)

    devices = args.devices or cfg.devices
    device_ids = [int(item.strip()) for item in devices.split(",") if item.strip()]
    backend = (args.backend or cfg.backend).lower()
    arch = args.arch or probe_local_arch(backend, device_ids[0]) or cfg.arch

    env = os.environ.copy()
    env.update(cfg.timing.as_env())
    env.update(eval_defaults().as_env())
    env.update({
        "WORKER_HOST": args.host,
        "WORKER_PORT": str(args.port),
        "WORKER_BACKEND": backend,
        "WORKER_ARCH": arch,
        "WORKER_DEVICES": ",".join(str(item) for item in device_ids),
        "OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY": _registry_path(args.port),
    })
    log_path = get_process_log_dir() / f"worker_{args.port}.log"
    env["OP_AUTORESEARCH_WORKER_LOG_FILE"] = str(log_path)

    flags = 0
    kwargs = {}
    if os.name == "posix":
        kwargs["preexec_fn"] = os.setsid
    else:
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP |
                 getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
    log_handle = open(log_path, "ab", buffering=0)
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "op_autoresearch.worker.server"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=flags,
            **kwargs,
        )
    finally:
        log_handle.close()

    state = load_worker_state()
    set_worker_entry(state, args.port, {
        "pid": process.pid,
        "backend": backend,
        "arch": arch,
        "devices": device_ids,
        "log_file": str(log_path),
        "started_at": time.time(),
    })
    save_worker_state(state)

    deadline = time.time() + cfg.timing.ready_timeout
    while time.time() < deadline:
        status = _status(args.port, cfg.timing.ready_probe_timeout)
        if status and status.get("status") == "ready":
            print(json.dumps(status, indent=2))
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.5)

    _stop_local(args.port)
    tail = ""
    try:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
    except OSError:
        pass
    print(f"worker failed to become ready; log={log_path}\n{tail}", file=sys.stderr)
    return 1


def _worker(args) -> int:
    cfg = WorkerConfig.load(args.config)
    args.port = args.port or cfg.port
    if args.remote_host:
        host_cfg = remote_dispatch.load_remote_host_config(
            args.remote_host, args.config
        )
        if host_cfg is None:
            print(f"unknown remote worker host: {args.remote_host}", file=sys.stderr)
            return 2
        if args.start:
            return remote_dispatch.dispatch_start(
                args.remote_host, host_cfg, args.backend, args.arch,
                args.devices, args.port, args.dsl,
            )
        if args.stop:
            return remote_dispatch.dispatch_stop(args.remote_host, host_cfg,
                                                   args.port)
        return remote_dispatch.dispatch_status(
            args.remote_host, host_cfg, args.port,
            backend=args.backend, dsl=args.dsl,
        )
    if args.start:
        return _start_local(args, cfg)
    if args.stop:
        return _stop_local(args.port)
    status = _status(args.port, cfg.timing.status_timeout)
    if status is None:
        print(f"worker 127.0.0.1:{args.port} is unreachable", file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="op-autoresearch")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="manage a local or remote worker")
    action = worker.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true")
    action.add_argument("--stop", action="store_true")
    action.add_argument("--status", action="store_true")
    worker.add_argument("--backend")
    worker.add_argument("--arch")
    worker.add_argument("--devices")
    worker.add_argument("--dsl")
    worker.add_argument("--host", default="127.0.0.1")
    worker.add_argument("--port", type=int)
    worker.add_argument("--remote-host")
    worker.add_argument("--config")
    worker.set_defaults(func=_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

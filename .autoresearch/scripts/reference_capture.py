#!/usr/bin/env python3
"""Capture PyTorch reference outputs once, save for reuse across eval rounds.

Mirrors AKG's `kernel_verifier.generate_reference_data()` flow. Rather than
re-running the reference Model every verify round, we run it ONCE on CPU
(deterministic seed, no device dependency) and persist inputs + outputs to
`<task_dir>/.ar_state/reference.pt`. Every subsequent verify just loads the
.pt and compares against stored tensors — saves a PyTorch forward pass per
round and decouples ref capture from kernel correctness.

Usage:
    python .autoresearch/scripts/reference_capture.py <task_dir>
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from phase_machine import state_path


_REMOTE_CACHE_ROOT = "/tmp/ar_cache"  # worker-side cache dir; predictable


def _remote_cache_path(task_dir: str) -> str:
    """Absolute path on the worker where reference.pt is cached."""
    return f"{_REMOTE_CACHE_ROOT}/{os.path.basename(os.path.abspath(task_dir))}/reference.pt"


def _upload_to_worker(local_pt: str, task_dir: str, ssh_host: str) -> bool:
    """scp reference.pt to the worker under /tmp/ar_cache/<task_basename>/.

    Returns True on success. Writes a marker file so _build_package can
    detect the upload and skip bundling reference.pt in the tarball.
    """
    remote_path = _remote_cache_path(task_dir)
    remote_dir = os.path.dirname(remote_path)
    try:
        subprocess.run(
            ["ssh", ssh_host, f"mkdir -p {remote_dir}"],
            check=True, capture_output=True, text=True, timeout=120,
        )
        size_mb = os.path.getsize(local_pt) / 1e6
        print(f"[reference_capture] scp {size_mb:.1f}MB -> {ssh_host}:{remote_path}",
              file=sys.stderr)
        # 30 min ceiling: covers slow-handshake / low-bandwidth tunnels.
        subprocess.run(
            ["scp", "-q", local_pt, f"{ssh_host}:{remote_path}"],
            check=True, capture_output=True, text=True, timeout=1800,
        )
    except subprocess.CalledProcessError as e:
        print(f"[reference_capture] UPLOAD FAILED: {e.stderr or e}",
              file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("[reference_capture] UPLOAD TIMED OUT (>30 min)", file=sys.stderr)
        return False

    marker = state_path(task_dir, ".ref_on_worker")
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"ssh_host": ssh_host, "remote_path": remote_path}, f)
    print(f"[reference_capture] cached on worker; eval tarballs will skip "
          f"reference.pt", file=sys.stderr)
    return True


_CAPTURE_SCRIPT = r'''
import json, os, sys, traceback
sys.path.insert(0, {task_dir!r})

try:
    import torch
    from {ref_mod} import Model, get_inputs, get_init_inputs
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"ok": False, "error": f"import failed: {{e}}"}}))
    sys.exit(1)

try:
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).cpu().eval()

    inputs = get_inputs()
    inputs = [x.cpu() if hasattr(x, "cpu") else x for x in inputs]

    with torch.no_grad():
        outs = model(*inputs)
    if isinstance(outs, torch.Tensor):
        outs = [outs]
    elif not isinstance(outs, (list, tuple)):
        outs = [outs]

    # ONLY save outputs + input metadata. Input tensors are regenerable from
    # get_inputs() (deterministic seed) and storing them would bloat the
    # reference.pt that ships in every eval tarball.
    payload = {{
        "outputs": [o.detach().cpu() for o in outs],
        "input_shapes":  [tuple(x.shape) if hasattr(x, "shape") else None for x in inputs],
        "input_dtypes":  [str(x.dtype) if hasattr(x, "dtype") else None for x in inputs],
        "output_shapes": [tuple(o.shape) for o in outs],
        "output_dtypes": [str(o.dtype) for o in outs],
    }}
    torch.save(payload, {out_path!r})
    print(json.dumps({{
        "ok": True,
        "path": {out_path!r},
        "n_outputs": len(outs),
        "output_shapes": [list(o.shape) for o in outs],
    }}))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"ok": False, "error": str(e)}}))
    sys.exit(1)
'''


def main():
    if len(sys.argv) < 2:
        print("Usage: reference_capture.py <task_dir>", file=sys.stderr)
        sys.exit(1)

    task_dir = os.path.abspath(sys.argv[1])
    ref_mod = "reference"  # reference.py at task_dir root
    out_path = state_path(task_dir, "reference.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Run capture in a child process so PyTorch import failures, GPU attempts,
    # etc. don't pollute the caller. CPU-only to avoid device dependencies
    # (the .pt is device-agnostic; worker will .to(device) on load).
    code = _CAPTURE_SCRIPT.format(
        task_dir=task_dir, ref_mod=ref_mod, out_path=out_path,
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "ASCEND_RT_VISIBLE_DEVICES": "",
        # Windows libiomp5 double-load workaround (no-op on Linux).
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=task_dir,
    )
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0:
        print(f"[reference_capture] FAILED (rc={r.returncode})", file=sys.stderr)
        print(r.stdout, end="")
        sys.exit(r.returncode)

    # Forward the child's JSON
    print(r.stdout, end="")

    # Optional: scp to the worker's /tmp cache so future eval tarballs can
    # skip bundling reference.pt. Triggered by task.yaml worker.ssh_host.
    # (Stale marker from a prior run is dropped — upload failure should not
    #  leave us claiming the worker still has the file.)
    marker = state_path(task_dir, ".ref_on_worker")
    if os.path.exists(marker):
        os.remove(marker)
    try:
        from task_config import load_task_config
        cfg = load_task_config(task_dir)
    except Exception:
        cfg = None
    if cfg and cfg.worker_ssh_host:
        _upload_to_worker(out_path, task_dir, cfg.worker_ssh_host)


if __name__ == "__main__":
    main()

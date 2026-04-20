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
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from phase_machine import state_path


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
    # - CPU-only (avoid device dependencies; .pt is device-agnostic)
    # - KMP_DUPLICATE_LIB_OK: defuse Windows libiomp5 conflict when multiple
    #   MKL-linked libs coexist (anaconda + torch). Linux is unaffected.
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "ASCEND_RT_VISIBLE_DEVICES": "",
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


if __name__ == "__main__":
    main()

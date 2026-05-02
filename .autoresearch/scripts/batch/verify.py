"""Pre-flight verification for batch workspaces.

Two tiers:
  Tier 1 (default, no hardware needed):
    - Python syntax compiles
    - Module imports cleanly (catches missing deps / syntax / import errors)
    - Required symbols exist:
        ref.py    : Model class + get_inputs + get_init_inputs
        kernel.py : ModelNew class

  Tier 2 (--full, needs the same hardware /autoresearch eval would use):
    - Runs ref(*get_inputs()) and kernel(*get_inputs())
    - torch.allclose with atol/rtol = 1e-2
    - Only meaningful for --mode ref-kernel; --mode ref has no kernel to compare

Each op is run in its own subprocess so import errors / device state in one op
don't poison the others. Results land in <workspace>/verify_results.json.

Usage:
    python .autoresearch/scripts/batch/verify.py <workspace_dir>             # Tier 1
    python .autoresearch/scripts/batch/verify.py <workspace_dir> --full      # Tier 1 + Tier 2
    python .autoresearch/scripts/batch/verify.py <workspace_dir> --only op1,op2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
# Reach up one level so we can import the shared correctness module the
# eval-package verify script also uses. Single source of truth for the
# allclose comparison — verify.py and autoresearch's eval can't drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from correctness import compare_outputs  # noqa: E402

VERIFY_RESULTS = "verify_results.json"
TIER1_TIMEOUT = 30
TIER2_TIMEOUT = 300
# Defaults match autoresearch's loader fallback (loader.py:49-50). They are
# only used when neither the CLI flag nor the manifest provides a value.
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2

REF_REQUIRED = ("Model", "get_inputs", "get_init_inputs")
KERNEL_REQUIRED = ("ModelNew",)


# ---------------------------------------------------------------------------
# Subprocess workers (this same file is re-invoked with --tier-worker)
# ---------------------------------------------------------------------------
def _tier1_inspect(path: Path, required: tuple[str, ...]) -> dict:
    """Compile, import, check required attrs are present."""
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "skip", "missing": [], "msg": ""}
    try:
        # utf-8-sig: PowerShell / Notepad on Windows tends to write source
        # files with a UTF-8 BOM; plain utf-8 leaves U+FEFF in the string and
        # compile() then dies with "invalid non-printable character U+FEFF".
        src = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        out["compile"] = "FAIL"
        out["msg"] = f"read error: {e}"
        return out
    try:
        compile(src, str(path), "exec")
        out["compile"] = "PASS"
    except SyntaxError as e:
        out["compile"] = "FAIL"
        out["msg"] = f"syntax error line {e.lineno}: {e.msg}"
        return out

    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            f"_verify_{path.stem}", str(path)
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not build spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out["import"] = "PASS"
    except Exception as e:
        out["import"] = "FAIL"
        out["msg"] = f"{type(e).__name__}: {e}"
        return out

    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        out["exports"] = "FAIL"
        out["missing"] = missing
        out["msg"] = f"missing: {', '.join(missing)}"
    else:
        out["exports"] = "PASS"
    return out


def _tier2_run(ref_path: Path, kernel_path: Path,
               atol: float, rtol: float) -> dict:
    """Run ref + kernel, compare outputs via the shared correctness module
    (autoresearch's eval calls into the same `compare_outputs`)."""
    out: dict = {"status": "skip", "msg": "", "max_abs_diff": None,
                 "atol": atol, "rtol": rtol}

    try:
        import torch  # type: ignore
    except ImportError as e:
        out["status"] = "ERROR"
        out["msg"] = f"torch import failed: {e}"
        return out
    try:
        import torch_npu  # type: ignore  # noqa: F401
    except Exception:
        pass  # not on Ascend; fine — kernel will pick its own device

    import importlib.util
    try:
        ref_spec = importlib.util.spec_from_file_location("_v_ref", str(ref_path))
        ref_mod = importlib.util.module_from_spec(ref_spec)  # type: ignore[arg-type]
        ref_spec.loader.exec_module(ref_mod)  # type: ignore[union-attr]
        kernel_spec = importlib.util.spec_from_file_location("_v_kernel", str(kernel_path))
        kernel_mod = importlib.util.module_from_spec(kernel_spec)  # type: ignore[arg-type]
        kernel_spec.loader.exec_module(kernel_mod)  # type: ignore[union-attr]
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"import: {type(e).__name__}: {e}"
        return out

    try:
        init_args = ref_mod.get_init_inputs()
        inputs = ref_mod.get_inputs()
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"get_inputs/get_init_inputs: {type(e).__name__}: {e}"
        return out

    try:
        ref = ref_mod.Model(*init_args)
        new = kernel_mod.ModelNew(*init_args)
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"construct: {type(e).__name__}: {e}"
        return out

    try:
        with torch.no_grad():
            out_ref = ref(*inputs)
            out_new = new(*inputs)
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"forward: {type(e).__name__}: {e}"
        return out

    def _to_list(x):
        if isinstance(x, (tuple, list)):
            return list(x)
        return [x]

    ref_outs = _to_list(out_ref)
    new_outs = _to_list(out_new)

    cmp = compare_outputs(ref_outs, new_outs, atol, rtol)
    out["max_abs_diff"] = cmp["max_abs_diff"]
    if cmp["correctness"]:
        out["status"] = "PASS"
        out["msg"] = "OK"
    else:
        out["status"] = "FAIL"
        # Surface the first failing diagnostic — the full list goes into the
        # JSON output for verify_results.json. Fall back to a generic note
        # if compare_outputs returned no diagnostics (shouldn't happen).
        bad = next((d for d in cmp["diagnostics"] if "OK" not in d), None)
        out["msg"] = bad or "correctness mismatch (no diagnostics)"
        out["diagnostics"] = cmp["diagnostics"]
    return out


def _worker_main() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel", "2"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-worker sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = _tier1_inspect(ref_path, REF_REQUIRED)
    elif args.tier == "1kernel":
        if kernel_path is None:
            result = {"compile": "skip", "import": "skip", "exports": "skip",
                      "missing": [], "msg": "no kernel for this op"}
        else:
            result = _tier1_inspect(kernel_path, KERNEL_REQUIRED)
    else:  # tier == "2"
        if kernel_path is None:
            result = {"status": "skip", "msg": "ref-only mode; no kernel to compare"}
        else:
            result = _tier2_run(ref_path, kernel_path, args.atol, args.rtol)

    Path(args.sidecar).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_subprocess(*, tier: str, ref: Path, kernel: Path | None,
                    timeout: int, atol: float = DEFAULT_ATOL,
                    rtol: float = DEFAULT_RTOL) -> dict:
    sidecar = Path(os.environ.get("TMP", "/tmp")) / f"_verify_{os.getpid()}_{tier}_{ref.stem}.json"
    if sidecar.exists():
        sidecar.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--tier-worker",
           "--tier", tier,
           "--ref", str(ref),
           "--sidecar", str(sidecar)]
    if kernel is not None:
        cmd += ["--kernel", str(kernel)]
    if tier == "2":
        cmd += ["--atol", repr(atol), "--rtol", repr(rtol)]

    env = os.environ.copy()
    # Default the Windows libomp/libiomp5md double-init workaround so users
    # don't see a wall of OMP error #15 on first run. No-op on Linux. Anyone
    # who wants the strict behavior can pre-set KMP_DUPLICATE_LIB_OK=FALSE.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "ERROR", "msg": f"timeout after {timeout}s",
                "elapsed_s": round(time.time() - t0, 2)}

    elapsed = round(time.time() - t0, 2)
    if not sidecar.exists():
        return {"status": "ERROR",
                "msg": f"no result; rc={proc.returncode}",
                "stderr_tail": (proc.stderr or proc.stdout)[-400:],
                "elapsed_s": elapsed}
    try:
        result = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "ERROR", "msg": f"parse sidecar: {e}",
                "elapsed_s": elapsed}
    finally:
        try:
            sidecar.unlink()
        except OSError:
            pass
    result["elapsed_s"] = elapsed
    return result


def _verify_one(case: dict, mode: str, full: bool,
                atol: float, rtol: float) -> dict:
    op = case["op_name"]
    ref = Path(case["ref"])
    kernel = Path(case["kernel"]) if case.get("kernel") else None

    out: dict = {"op_name": op, "tier1_ref": None, "tier1_kernel": None,
                 "tier2": None}

    out["tier1_ref"] = _run_subprocess(tier="1ref", ref=ref, kernel=None,
                                       timeout=TIER1_TIMEOUT)

    if mode == "ref-kernel":
        out["tier1_kernel"] = _run_subprocess(tier="1kernel", ref=ref,
                                              kernel=kernel,
                                              timeout=TIER1_TIMEOUT)

    tier1_ok = (out["tier1_ref"].get("exports") == "PASS"
                and (mode != "ref-kernel" or
                     out["tier1_kernel"].get("exports") == "PASS"))

    if full and mode == "ref-kernel":
        if tier1_ok:
            out["tier2"] = _run_subprocess(tier="2", ref=ref, kernel=kernel,
                                           timeout=TIER2_TIMEOUT,
                                           atol=atol, rtol=rtol)
        else:
            out["tier2"] = {"status": "skip",
                            "msg": "tier1 failed; skipping tier2",
                            "elapsed_s": 0}

    return out


def _summary_status(record: dict, mode: str, full: bool) -> str:
    """Distill a single-letter overall status: P/F/E/S (Pass/Fail/Error/Skip)."""
    t1r = record["tier1_ref"]
    t1k = record["tier1_kernel"]
    t2 = record["tier2"]

    def _bad(t):
        return t and ("FAIL" in (t.get("compile"), t.get("import"), t.get("exports"))
                      or t.get("status") in ("FAIL", "ERROR"))

    if _bad(t1r):
        return "F" if t1r.get("compile") == "FAIL" or t1r.get("exports") == "FAIL" else "E"
    if mode == "ref-kernel" and _bad(t1k):
        return "F" if t1k and (t1k.get("compile") == "FAIL"
                               or t1k.get("exports") == "FAIL") else "E"
    if full and mode == "ref-kernel" and t2:
        if t2.get("status") == "PASS":
            return "P"
        if t2.get("status") == "FAIL":
            return "F"
        if t2.get("status") == "ERROR":
            return "E"
        return "S"
    return "P"


def _print_table(results: dict, mode: str, full: bool) -> None:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for op, rec in results.items():
        t1r = rec["tier1_ref"]
        t1k = rec["tier1_kernel"]
        t2 = rec["tier2"]

        col_t1r = "PASS" if t1r and t1r.get("exports") == "PASS" else (
            "FAIL" if t1r and t1r.get("exports") == "FAIL" else (
                "FAIL" if t1r and (t1r.get("compile") == "FAIL"
                                   or t1r.get("import") == "FAIL") else "ERROR"))

        if mode == "ref-kernel" and t1k is not None:
            col_t1k = "PASS" if t1k.get("exports") == "PASS" else (
                "FAIL" if t1k.get("exports") == "FAIL" else (
                    "FAIL" if t1k.get("compile") == "FAIL"
                              or t1k.get("import") == "FAIL" else "ERROR"))
        else:
            col_t1k = "-"

        if full and t2 is not None:
            col_t2 = t2.get("status", "?")
        else:
            col_t2 = "-"

        # Pick the most informative message
        msg = ""
        for src in (t2, t1k, t1r):
            if src and src.get("msg") and src.get("msg") != "OK":
                msg = src["msg"]
                if "FAIL" in (src.get("compile"), src.get("import"),
                              src.get("exports")) or src.get("status") in ("FAIL", "ERROR"):
                    break
        rows.append((op, col_t1r, col_t1k, col_t2,
                     _summary_status(rec, mode, full), msg[:70]))

    op_w = max(8, max(len(r[0]) for r in rows))
    headers = ("op", "t1_ref", "t1_kern", "t2", "ok", "note")
    print(f"  {headers[0]:<{op_w}}  {headers[1]:<6}  {headers[2]:<7}  "
          f"{headers[3]:<6}  {headers[4]:<3}  {headers[5]}")
    print(f"  {'-' * op_w}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 3}  {'-' * 60}")
    for op, t1r, t1k, t2, ok, msg in rows:
        print(f"  {op:<{op_w}}  {t1r:<6}  {t1k:<7}  {t2:<6}  {ok:<3}  {msg}")


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--tier-worker":
        return _worker_main()

    ap = argparse.ArgumentParser(description="Pre-flight verify for batch workspaces.")
    ap.add_argument("workspace_dir")
    ap.add_argument("--mode", choices=mf.VALID_MODES,
                    help="ref-kernel or ref (overrides manifest.mode)")
    ap.add_argument("--full", action="store_true",
                    help="also run Tier 2 (execute ref + kernel, compare outputs); "
                         "needs the same hardware /autoresearch eval would use")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    ap.add_argument("--correctness-atol", type=float, default=None,
                    help="Tier 2 absolute tolerance for torch.allclose. "
                         "Precedence: CLI > manifest.correctness_atol > "
                         f"default ({DEFAULT_ATOL}). Mirrors the value "
                         "scaffold writes into task.yaml.metric.correctness_atol.")
    ap.add_argument("--correctness-rtol", type=float, default=None,
                    help="Tier 2 relative tolerance for torch.allclose. "
                         "Precedence: CLI > manifest.correctness_rtol > "
                         f"default ({DEFAULT_RTOL}).")
    args = ap.parse_args()

    workspace_dir = Path(args.workspace_dir).resolve()
    if not workspace_dir.is_dir():
        sys.exit(f"workspace dir not found: {workspace_dir}")

    try:
        manifest_path = mf.find_manifest(workspace_dir)
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(str(e))

    mode = args.mode or manifest_data.get("mode")
    if not mode:
        sys.exit("--mode is required (also accepted as `mode:` in manifest)")
    if mode not in mf.VALID_MODES:
        sys.exit(f"--mode must be one of {mf.VALID_MODES}, got {mode!r}")

    # Resolve tolerances: CLI > manifest > default. Same defaults as
    # autoresearch's loader, so a no-flag run agrees with /autoresearch's
    # eval byte-for-byte.
    atol = (args.correctness_atol
            if args.correctness_atol is not None
            else float(manifest_data.get("correctness_atol", DEFAULT_ATOL)))
    rtol = (args.correctness_rtol
            if args.correctness_rtol is not None
            else float(manifest_data.get("correctness_rtol", DEFAULT_RTOL)))

    try:
        cases = mf.resolve_cases(workspace_dir, manifest_data, mode)
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    if only:
        cases = [c for c in cases if c["op_name"] in only]
        if not cases:
            sys.exit(f"--only filtered out all ops")

    if args.full and mode == "ref":
        print("note: --full has no effect in --mode ref (no kernel to compare)")

    print(f"verify  workspace={workspace_dir}  mode={mode}  "
          f"tier={'1+2' if args.full else '1'}  ops={len(cases)}  "
          f"tols: atol={atol:g}  rtol={rtol:g}")
    print()

    results: dict = {}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        op = case["op_name"]
        sys.stdout.write(f"  [{i:>3}/{len(cases)}] {op} ... ")
        sys.stdout.flush()
        rec = _verify_one(case, mode, full=args.full, atol=atol, rtol=rtol)
        results[op] = rec
        ok = _summary_status(rec, mode, full=args.full)
        sys.stdout.write(f"{ok}\n")
        sys.stdout.flush()

    out_path = workspace_dir / VERIFY_RESULTS
    out_path.write_text(json.dumps({
        "mode": mode,
        "full": args.full,
        "atol": atol,
        "rtol": rtol,
        "results": results,
    }, indent=2), encoding="utf-8")

    print()
    _print_table(results, mode, full=args.full)
    print()

    n_pass = sum(1 for op in results if _summary_status(results[op], mode, args.full) == "P")
    n_fail = sum(1 for op in results if _summary_status(results[op], mode, args.full) == "F")
    n_err = sum(1 for op in results if _summary_status(results[op], mode, args.full) == "E")
    print(f"  total={len(results)}  pass={n_pass}  fail={n_fail}  error={n_err}  "
          f"elapsed={time.time()-t0:.1f}s")
    print(f"  results: {out_path}")
    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

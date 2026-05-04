"""Batch driver for /autoresearch.

Loads a manifest from <workspace_dir>/manifest.{yaml,json}, resolves the op
list against the <op_name>_{ref,kernel}.py naming convention, then drives each
op end-to-end via headless `claude --print`. Streams stdout to console and
batch.log, updates batch_progress.json after every op.

Usage:
    python .autoresearch/scripts/batch/run.py <workspace_dir> \\
        --mode {ref-kernel,ref} [--dsl triton_ascend] \\
        [--devices N | --worker-url host:port] \\
        [--max-rounds 30] [--eval-timeout 120] [--timeout-min 180] \\
        [--only op1,op2] [--limit N] [--retry-errored] [--cooldown-sec 5]
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

# Force line-buffered stdout so logs flush in real time when run via nohup.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


PROMPT_TEMPLATE = """\
/autoresearch --ref {ref}{kernel_arg} --op-name {op} --dsl {dsl} {hw} --max-rounds {rounds} --eval-timeout {timeout}

CRITICAL rules — read carefully, this session is non-interactive:

1. After scaffold prints "Task directory created: <path>", your VERY FIRST
   subsequent action MUST be exactly:
       export AR_TASK_DIR="<that path>"
   The double quotes are required — paths with spaces or backslashes
   (e.g. C:\\Users\\Foo Bar\\...) get truncated otherwise. Without this
   step .autoresearch/.active_task is never written, the PostToolUse
   Edit hook is gated off, validate_kernel never runs, and phase stays
   stuck forever. THIS IS THE SINGLE MOST IMPORTANT STEP.

2. {mode_block}

3. In EDIT phase use the Edit tool (or Write for full rewrites).
   PostToolUse will validate kernel.py and auto-advance on pass.

4. Follow ALL hook guidance verbatim. Do not skip phases. Do not run
   baseline.py while phase != BASELINE — hooks will block it. Read the
   hook error messages — they tell you the next action.

5. Keep going through every phase until FINISH or max-rounds. Never stop
   to ask the user — there is no user. If a phase stalls, read the latest
   hook output and try what it says.

6. WHEN AND ONLY WHEN you have nothing more to do (phase=FINISH, exhausted
   max-rounds, or truly stuck), print exactly one line in this format and
   then stop:

       AUTORESEARCH_RESULT task_dir="<absolute path>" phase=<phase> status=<ok|stuck>

   status=ok if phase==FINISH, status=stuck otherwise. The task_dir
   value MUST be wrapped in double quotes so that paths containing
   spaces survive the orchestrator's parser.
"""

MODE_BLOCK = {
    "ref-kernel": (
        "The kernel.py we passed via --kernel is a seed implementation. "
        "Scaffold's --run-baseline will run it; baseline should PASS, and "
        ".ar_state/.phase will be set to PLAN immediately, skipping "
        "GENERATE_KERNEL. Your job is PERFORMANCE OPTIMIZATION via "
        "PLAN -> EDIT -> VERIFY for the configured max-rounds. Do NOT rewrite "
        "ModelNew from scratch — propose targeted edits and let pipeline.py "
        "measure the speedup. If baseline unexpectedly fails, the hook will "
        "demote to GENERATE_KERNEL — fix the regression with minimal diffs "
        "against the seed, do not start over."
    ),
    "ref": (
        "No seed kernel was supplied. Scaffold will run GENERATE_KERNEL "
        "first; produce a working kernel.py that passes baseline, then "
        "optimize via PLAN -> EDIT -> VERIFY for the remaining rounds."
    ),
}

# task_dir may be quoted ("...") so paths with spaces survive parsing;
# bare \S+ form is also accepted for backward compatibility with existing
# in-flight sessions. phase / status are simple identifiers and never
# contain whitespace or quotes.
MARKER_RE = re.compile(
    r'AUTORESEARCH_RESULT\s+task_dir=(?:"([^"]*)"|(\S+))'
    r'\s+phase=(\S+)\s+status=(\S+)'
)


def parse_marker(text: str) -> tuple[str, str, str] | None:
    # Take the LAST match: Claude may echo the marker format mid-stream while
    # explaining the contract; only the final line is authoritative.
    matches = MARKER_RE.findall(text)
    if not matches:
        return None
    quoted, bare, phase, status = matches[-1]
    return (quoted or bare, phase, status)


def health_check_worker(worker_url: str) -> None:
    """Probe http://<host>:<port>/api/v1/status. Raises SystemExit on failure."""
    if "://" not in worker_url:
        url = f"http://{worker_url}/api/v1/status"
    else:
        url = worker_url.rstrip("/") + "/api/v1/status"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                raise urllib.error.URLError(f"HTTP {resp.status}")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        sys.exit(
            f"\nworker daemon at {worker_url} is unreachable ({e}).\n"
            f"start it first:\n"
            f"    python .autoresearch/scripts/ar_cli.py worker --start --bg "
            f"--port {worker_url.split(':')[-1] or '9111'}\n"
            f"or pass --devices N to use in-process eval (slower for batch runs).\n"
        )


LOCK_FILENAME = ".batch.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            # Can't tell — err on the safe side and assume alive so the user
            # has to confirm by removing the lock manually.
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_lock(workspace_dir: Path) -> Path:
    """Prevent two run.py instances racing on the same batch_progress.json.
    Stale locks (PID gone) are auto-cleared; live locks abort with a hint."""
    lock = workspace_dir / LOCK_FILENAME
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            sys.exit(
                f"\nanother batch run is active on this workspace "
                f"(pid={pid}, lock={lock}).\n"
                f"if you're sure no run.py is running, remove {lock} and retry.\n"
            )
        # stale lock — overwrite below
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def recover_stale_running(progress: dict) -> int:
    """Demote any 'running' cases to 'error'. We hold the workspace lock by
    the time this is called, so anything still 'running' is an orphan from a
    previous run.py that died (SIGKILL, OOM, machine reboot)."""
    cases = progress.get("cases", {})
    n = 0
    now = mf.now_iso()
    for c in cases.values():
        if c.get("status") == "running":
            c["status"] = "error"
            c["finished_at"] = now
            existing = (c.get("note") or "").strip()
            tag = "stale running, demoted on batch restart"
            c["note"] = f"{existing}; {tag}" if existing else tag
            n += 1
    return n


def build_prompt(case: dict, mode: str, dsl: str, hw_arg: str,
                 max_rounds: int, eval_timeout: int) -> str:
    """Quote every value-bearing flag with shlex.quote so paths with
    spaces (e.g. workspace under `C:\\Users\\Foo Bar\\...`, or
    `--output-dir "my tasks"`) reach /autoresearch as one argv each.
    `hw_arg` is constructed by the caller from already-validated CLI
    flags — pass through unchanged."""
    kernel_arg = (f" --kernel {shlex.quote(case['kernel'])}"
                  if case.get("kernel") else "")
    return PROMPT_TEMPLATE.format(
        ref=shlex.quote(case["ref"]),
        kernel_arg=kernel_arg,
        op=shlex.quote(case["op_name"]),
        dsl=shlex.quote(dsl),
        hw=hw_arg,
        rounds=max_rounds,
        timeout=eval_timeout,
        mode_block=MODE_BLOCK[mode],
    )


def build_claude_cmd(args: argparse.Namespace, prompt: str) -> list[str]:
    cmd = [
        args.claude_bin,
        "--print",
        "--permission-mode", "acceptEdits",
        "--output-format", "text",
    ]
    if args.model:
        cmd += ["--model", args.model]
    cmd += args.extra_claude_arg
    cmd += [prompt]
    return cmd


def env_with_no_proxy() -> dict[str, str]:
    env = os.environ.copy()
    extras = "127.0.0.1,localhost"
    existing = env.get("NO_PROXY", "")
    env["NO_PROXY"] = f"{existing},{extras}".strip(",") if existing else extras
    env["no_proxy"] = env["NO_PROXY"]
    return env


def run_one(workspace_dir: Path, case: dict, args: argparse.Namespace,
            mode: str, dsl: str, hw_arg: str, log_fp) -> int:
    op = case["op_name"]
    repo_root = mf.repo_root()
    prompt = build_prompt(case, mode, dsl, hw_arg,
                          args.max_rounds, args.eval_timeout)
    cmd = build_claude_cmd(args, prompt)

    started = time.time()
    started_iso = mf.now_iso()
    mf.update_case(workspace_dir, op,
                   status="running",
                   started_at=started_iso,
                   finished_at=None,
                   task_dir=None,
                   final_phase=None,
                   rc=None,
                   note="")

    header = (f"\n{'=' * 72}\n"
              f"[run {datetime.now().isoformat(timespec='seconds')}] op={op} "
              f"{hw_arg} rounds={args.max_rounds}\n"
              f"[run] launching: {args.claude_bin} --print "
              f"(cwd={repo_root}, timeout={args.timeout_min}min)\n"
              f"{'─' * 72}\n")
    sys.stdout.write(header)
    sys.stdout.flush()
    log_fp.write(header)
    log_fp.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env_with_no_proxy(),
    )

    captured: list[str] = []
    timeout_s = args.timeout_min * 60
    interrupted = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()
            captured.append(line)
            if time.time() - started > timeout_s:
                msg = f"[run] WALL-CLOCK TIMEOUT after {args.timeout_min}min, killing claude\n"
                sys.stdout.write(msg)
                log_fp.write(msg)
                proc.kill()
                break
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        interrupted = True
        msg = "\n[run] Ctrl-C received, killing claude\n"
        sys.stdout.write(msg)
        log_fp.write(msg)
        try:
            proc.kill()
        except Exception:
            pass

    elapsed = time.time() - started
    full_out = "".join(captured)
    footer = (f"{'─' * 72}\n"
              f"[run] claude exited rc={proc.returncode} after {elapsed:.0f}s\n")
    sys.stdout.write(footer)
    log_fp.write(footer)
    log_fp.flush()

    parsed = parse_marker(full_out)
    if parsed:
        task_dir_str, phase, status_marker = parsed
        task_dir = Path(task_dir_str)
    else:
        td = mf.find_recent_task_dir(op, since_ts=started - 5)
        if td is None:
            mf.update_case(workspace_dir, op,
                           status="error",
                           finished_at=mf.now_iso(),
                           rc=proc.returncode,
                           note=f"no task_dir found; rc={proc.returncode}"
                                + ("; interrupted" if interrupted else ""))
            return 130 if interrupted else 2
        task_dir = td
        phase = mf.read_phase(td)
        status_marker = "ok" if phase == "FINISH" else "stuck"
        sys.stdout.write(
            f"[run] no marker line; inferred task_dir={task_dir} phase={phase}\n"
        )

    result = mf.read_task_state(task_dir)
    final_status = "done" if (phase == "FINISH" and status_marker == "ok"
                              and not interrupted) else "error"
    note = ""
    if final_status == "error":
        note = f"phase={phase} status={status_marker} rc={proc.returncode}"
        if interrupted:
            note += "; interrupted"

    mf.update_case(workspace_dir, op,
                   status=final_status,
                   task_dir=str(task_dir.resolve()),
                   finished_at=mf.now_iso(),
                   final_phase=phase,
                   rc=proc.returncode,
                   result=result,
                   note=note)

    sys.stdout.write(
        f"[run] result: op={op} task_dir={task_dir} phase={phase} "
        f"status={final_status}\n"
    )
    if interrupted:
        return 130
    return 0 if final_status == "done" else 1


def filter_queue(progress: dict, args: argparse.Namespace) -> list[dict]:
    statuses = {"pending"}
    if args.retry_errored:
        statuses.add("error")
    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    out: list[dict] = []
    for v in progress.get("cases", {}).values():
        if v.get("status") not in statuses:
            continue
        if only and v.get("op_name") not in only:
            continue
        out.append(v)
    return out


def print_summary(workspace_dir: Path, total_elapsed: float,
                  ok: int, fail: int, skipped: int) -> None:
    progress = mf.load_progress(workspace_dir)
    cases = progress.get("cases", {})
    counts = {"done": 0, "error": 0, "skip": 0, "pending": 0, "running": 0}
    speedups: list[float] = []
    for v in cases.values():
        s = v.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
        if s != "done":
            continue
        r = v.get("result") or {}
        bm, best = r.get("baseline_metric"), r.get("best_metric")
        if isinstance(bm, (int, float)) and isinstance(best, (int, float)) and best > 0:
            speedups.append(bm / best)

    print()
    print("=" * 72)
    print(f"[batch done] elapsed={total_elapsed/60:.1f}min  "
          f"ok={ok}  fail={fail}  skipped={skipped}")
    print(f"           total cases: done={counts['done']}  error={counts['error']}  "
          f"skip={counts['skip']}  pending={counts['pending']}  running={counts['running']}")
    if speedups:
        import statistics
        improved = sum(1 for s in speedups if s > 1.05)
        onpar = sum(1 for s in speedups if 0.95 <= s <= 1.05)
        regr = sum(1 for s in speedups if s < 0.95)
        print(f"           speedup: median={statistics.median(speedups):.2f}x  "
              f"best={max(speedups):.2f}x  worst={min(speedups):.2f}x  "
              f"(n={len(speedups)})")
        print(f"           improved={improved}  on-par={onpar}  regress={regr}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch driver for /autoresearch.")
    ap.add_argument("workspace_dir", help="dir containing manifest.yaml/json")
    ap.add_argument("--mode", choices=mf.VALID_MODES,
                    help="ref-kernel or ref (overrides manifest.mode)")
    ap.add_argument("--dsl", default="",
                    help="DSL passed to /autoresearch (overrides manifest.dsl)")
    ap.add_argument("--devices", default="",
                    help="NPU device ids, e.g. 0 or 0,1; mutually exclusive with --worker-url")
    ap.add_argument("--worker-url", default="",
                    help="autoresearch worker URL; default 127.0.0.1:9111 if "
                         "neither --devices nor --worker-url is given")
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--eval-timeout", type=int, default=120)
    ap.add_argument("--timeout-min", type=int, default=180,
                    help="hard wall-clock cap per op in minutes")
    ap.add_argument("--only", default="", help="comma-separated op names")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N ops (0 = no limit)")
    ap.add_argument("--retry-errored", action="store_true",
                    help="also queue ops with status=error")
    ap.add_argument("--cooldown-sec", type=int, default=5,
                    help="seconds to sleep between ops")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--model", default="")
    ap.add_argument("--extra-claude-arg", action="append", default=[],
                    help="extra arg to pass to claude (repeatable)")
    args = ap.parse_args()

    workspace_dir = Path(args.workspace_dir).resolve()
    if not workspace_dir.is_dir():
        sys.exit(f"workspace dir not found: {workspace_dir}")

    try:
        manifest_path = mf.find_manifest(workspace_dir)
    except mf.ManifestError as e:
        sys.exit(str(e))

    try:
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(f"failed to load {manifest_path}: {e}")

    mode = args.mode or manifest_data.get("mode")
    if not mode:
        sys.exit("--mode is required (also accepted as `mode:` in manifest)")
    if mode not in mf.VALID_MODES:
        sys.exit(f"--mode must be one of {mf.VALID_MODES}, got {mode!r}")

    dsl = args.dsl or manifest_data.get("dsl") or ""
    if not dsl:
        sys.exit("--dsl is required (also accepted as `dsl:` in manifest)")

    if args.devices and args.worker_url:
        sys.exit("--devices and --worker-url are mutually exclusive")
    if args.devices:
        hw_arg = f"--devices {args.devices}"
    elif args.worker_url:
        hw_arg = f"--worker-url {args.worker_url}"
        health_check_worker(args.worker_url)
    else:
        worker_url = "127.0.0.1:9111"
        hw_arg = f"--worker-url {worker_url}"
        health_check_worker(worker_url)

    try:
        cases = mf.resolve_cases(workspace_dir, manifest_data, mode)
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    lock_path = acquire_lock(workspace_dir)
    try:
        progress = mf.load_progress(workspace_dir)
        demoted = recover_stale_running(progress)
        progress, dropped = mf.merge_cases(progress, cases, mode, dsl)
        mf.save_progress(workspace_dir, progress)
        if demoted:
            print(f"[batch] demoted {demoted} stale 'running' op(s) "
                  f"from a previous run -> error")
        if dropped:
            preview = ", ".join(dropped[:5]) + (
                f", ... (+{len(dropped) - 5} more)" if len(dropped) > 5 else "")
            print(f"[batch] dropped {len(dropped)} op(s) no longer in manifest: "
                  f"{preview}")

        queue = filter_queue(progress, args)
        if not queue:
            print("nothing to run.")
            return 0
        if args.limit:
            queue = queue[: args.limit]

        print(f"[batch {datetime.now().isoformat(timespec='seconds')}] "
              f"workspace={workspace_dir}  mode={mode}  dsl={dsl}  {hw_arg}\n"
              f"[batch] queue size: {len(queue)}  rounds={args.max_rounds}")

        log_path = workspace_dir / mf.LOG_FILENAME
        log_fp = log_path.open("a", encoding="utf-8", buffering=1)

        succeeded = failed = skipped = 0
        total_started = time.time()
        rc_final = 0
        try:
            for i, case in enumerate(queue, 1):
                op = case["op_name"]
                current = filter_queue(mf.load_progress(workspace_dir), args)
                if not any(c["op_name"] == op for c in current):
                    print(f"[{i}/{len(queue)}] {op}: status changed underfoot, skipping")
                    skipped += 1
                    continue

                print(f"\n[{i}/{len(queue)}] starting op={op}  "
                      f"elapsed_total={(time.time()-total_started)/60:.1f}min")

                try:
                    rc = run_one(workspace_dir, case, args, mode, dsl, hw_arg, log_fp)
                except KeyboardInterrupt:
                    print("\n[batch] Ctrl-C — current op recorded, stopping.")
                    rc_final = 130
                    break

                if rc == 0:
                    succeeded += 1
                elif rc == 130:
                    failed += 1
                    print("\n[batch] op interrupted, stopping.")
                    rc_final = 130
                    break
                else:
                    failed += 1

                print(f"[{i}/{len(queue)}] {op} done rc={rc}  "
                      f"running totals: ok={succeeded} fail={failed} skipped={skipped}")

                if i < len(queue) and args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)
        finally:
            log_fp.close()

        print_summary(workspace_dir, time.time() - total_started,
                      succeeded, failed, skipped)
        if rc_final:
            return rc_final
        return 0 if failed == 0 else 1
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    sys.exit(main())

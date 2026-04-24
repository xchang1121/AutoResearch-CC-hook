"""Static code checker for autoresearch kernel edits.

Standalone port of `code_checker.CodeChecker`, with the
async wrapper and logger dependencies dropped. Pure stdlib: ast, tokenize,
py_compile, importlib, re, tempfile.

Pipeline:
  1. ast.parse                   — syntax
  2. py_compile                  — compile (catches SyntaxWarning upgrades)
  3. importlib.util.find_spec    — imports resolve in current env
  4. tokenize + unicode scan     — stray Chinese text in code (LLM hazard)
  5. DSL compliance (triton*)    — forbids torch compute APIs in forward(),
                                   requires @triton.jit kernel + launch
  6. @triton.autotune compliance — must declare restore_value

Entry point:
    checker = CodeChecker(backend="ascend", dsl="triton_ascend")
    passed, message, errors = checker.check(code)
"""
import ast
import importlib.util
import io
import os
import py_compile
import re
import tempfile
import tokenize
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Policy — loaded once from .autoresearch/code_checker.yaml via settings.py.
# Missing/malformed keys surface as KeyError at import time.
# `worker_only_modules` lives in config.yaml, not code_checker.yaml.
# ---------------------------------------------------------------------------
from settings import (
    code_checker_triton_decorators,
    code_checker_torch_call_prefixes,
    code_checker_hard_ops,
    code_checker_soft_ops,
    code_checker_kernel_class_name,
    code_checker_kernel_forward_method,
    code_checker_triton_module_name,
    code_checker_dsl_compliance_prefix,
    code_checker_stray_text_re,
    code_checker_autotune_re,
    code_checker_restore_value_re,
    worker_only_modules,
)

_TRITON_DECORATORS = code_checker_triton_decorators()
_TORCH_CALL_PREFIXES = code_checker_torch_call_prefixes()
_TORCH_OPS_HARD = code_checker_hard_ops()
_TORCH_OPS_SOFT = code_checker_soft_ops()
_KERNEL_CLASS = code_checker_kernel_class_name()
_KERNEL_FORWARD = code_checker_kernel_forward_method()
_TRITON_MODULE = code_checker_triton_module_name()
_DSL_PREFIX = code_checker_dsl_compliance_prefix()
_STRAY_TEXT_RE = code_checker_stray_text_re()
_AUTOTUNE_RE = code_checker_autotune_re()
_RESTORE_VALUE_RE = code_checker_restore_value_re()
_WORKER_ONLY = worker_only_modules()


def _is_triton_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == _TRITON_MODULE
            and node.attr in _TRITON_DECORATORS
        )
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func)
    return False


def _find_model_new_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == _KERNEL_CLASS:
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == _KERNEL_FORWARD:
            return item
    return None


@dataclass
class CheckError:
    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str


# ---------------------------------------------------------------------------
# CodeChecker
# ---------------------------------------------------------------------------

class CodeChecker:
    def __init__(self, backend: str, dsl: str):
        self.backend = (backend or "").lower()
        self.dsl = (dsl or "").lower()

    def check(self, code: str) -> Tuple[bool, str, List[Dict]]:
        """Run the pipeline. Returns (passed, formatted_message, errors_list)."""
        if not code or not code.strip():
            err = {
                "line": 0,
                "error_type": "empty_code",
                "detail": "Code is empty.",
                "suggestion": "Provide a non-empty source file.",
                "code_snippet": "",
            }
            return False, self._format_errors([err]), [err]

        errors: List[Dict] = []

        # Step 1: AST parse
        errors.extend(self._check_python_syntax(code))

        # Step 2: py_compile (only if syntax passed)
        if not errors:
            errors.extend(self._check_py_compile(code))

        # Step 3: imports (only if compile passed)
        if not errors:
            errors.extend(self._check_imports(code))

        # Step 4: stray Chinese text (always)
        errors.extend(self._check_stray_chinese(code))

        # Step 5/6: DSL + autotune (only when syntax is clean)
        has_syntax_err = any(
            e.get("error_type") in ("syntax_error", "compile_error") for e in errors
        )
        if not has_syntax_err:
            errors.extend(self._check_dsl_compliance(code))
            if self.dsl.startswith(_DSL_PREFIX):
                errors.extend(self._check_autotune_compliance(code))

        passed = len(errors) == 0
        code_lines = code.split("\n")
        message = self._format_errors(errors, code_lines) if errors else ""
        return passed, message, errors

    # ------------------------------------------------------------------
    # Step 1: ast.parse
    # ------------------------------------------------------------------

    def _check_python_syntax(self, code: str) -> List[Dict]:
        try:
            ast.parse(code)
        except SyntaxError as e:
            line_num = e.lineno or 0
            code_lines = code.split("\n")
            snippet = code_lines[line_num - 1].rstrip() if 0 < line_num <= len(code_lines) else ""
            msg = e.msg or "syntax error"
            if e.offset:
                msg += f" (col {e.offset})"
            return [{
                "line": line_num,
                "error_type": "syntax_error",
                "detail": f"Python syntax error: {msg}",
                "suggestion": "Check brackets/quotes/indentation/colons at the flagged line.",
                "code_snippet": snippet,
            }]
        return []

    # ------------------------------------------------------------------
    # Step 2: py_compile
    # ------------------------------------------------------------------

    def _check_py_compile(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        tmp_src = None
        tmp_pyc = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_src = f.name
            fd, tmp_pyc = tempfile.mkstemp(suffix=".pyc")
            os.close(fd)
            py_compile.compile(tmp_src, cfile=tmp_pyc, doraise=True)
        except py_compile.PyCompileError as e:
            line_num = 0
            m = re.search(r"line (\d+)", str(e))
            if m:
                line_num = int(m.group(1))
            code_lines = code.split("\n")
            snippet = code_lines[line_num - 1].rstrip() if 0 < line_num <= len(code_lines) else ""
            errors.append({
                "line": line_num,
                "error_type": "compile_error",
                "detail": f"Python compile error: {e}",
                "suggestion": "Check the flagged line for illegal expressions, invalid identifiers, or version-incompatible syntax.",
                "code_snippet": snippet,
            })
        except Exception:
            pass
        finally:
            for path in (tmp_src, tmp_pyc):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        return errors

    # ------------------------------------------------------------------
    # Step 3: imports
    # ------------------------------------------------------------------

    def _check_imports(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return errors

        checked = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in checked or top in _WORKER_ONLY:
                        continue
                    checked.add(top)
                    if not self._is_module_available(top):
                        errors.append({
                            "line": node.lineno,
                            "error_type": "import_error",
                            "detail": f"Module '{alias.name}' not importable in this env.",
                            "suggestion": f"Check the spelling of '{alias.name}' or install the package.",
                            "code_snippet": "",
                        })
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top = node.module.split(".")[0]
                    if top in checked or top in _WORKER_ONLY:
                        continue
                    checked.add(top)
                    if not self._is_module_available(top):
                        errors.append({
                            "line": node.lineno,
                            "error_type": "import_error",
                            "detail": f"Module '{node.module}' not importable in this env.",
                            "suggestion": f"Check the spelling of '{node.module}' or install the package.",
                            "code_snippet": "",
                        })
        return errors

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Step 4: stray Chinese text
    # ------------------------------------------------------------------

    def _check_stray_chinese(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except (tokenize.TokenError, IndentationError):
            return errors

        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type in (
                tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING,
            ):
                continue
            m = _STRAY_TEXT_RE.search(tok.string)
            if m:
                line_num = tok.start[0]
                errors.append({
                    "line": line_num,
                    "error_type": "stray_chinese_text",
                    "detail": f"Non-code Chinese text '{m.group()}' detected in source token.",
                    "suggestion": f"Remove or convert to a comment (prefix with '#') on line {line_num}. Ignore if it's an intentional Chinese identifier.",
                    "code_snippet": "",
                })
        return errors

    # ------------------------------------------------------------------
    # Step 5: DSL compliance (triton* only)
    # ------------------------------------------------------------------

    def _check_dsl_compliance(self, code: str) -> List[Dict]:
        if not self.dsl.startswith(_DSL_PREFIX):
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        errors: List[Dict] = []

        # A. Collect @triton.jit kernel names
        triton_kernels: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if _is_triton_decorator(dec):
                        triton_kernels.add(node.name)
                        break

        if not triton_kernels:
            errors.append({
                "line": 0,
                "error_type": "no_triton_kernel",
                "detail": (
                    f"DSL is {self.dsl} but no @triton.jit-decorated kernel function was found. "
                    f"Code likely uses torch high-level APIs in place of a triton kernel."
                ),
                "suggestion": (
                    "Define at least one @triton.jit kernel and launch it via "
                    "kernel[grid](...) inside ModelNew.forward()."
                ),
                "code_snippet": "",
            })
            return errors

        # B. Any kernel[grid](...) launch?
        launched_kernels: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                value = node.func.value
                if isinstance(value, ast.Name) and value.id in triton_kernels:
                    launched_kernels.add(value.id)

        kernels_not_launched = not launched_kernels
        if kernels_not_launched:
            errors.append({
                "line": 0,
                "error_type": "triton_kernel_not_called",
                "detail": (
                    f"Triton kernel(s) {sorted(triton_kernels)} are defined but never launched via "
                    f"`kernel_name[grid](...)` syntax. Decorator-only kernels perform no computation."
                ),
                "suggestion": (
                    "In ModelNew.forward() (or a helper), invoke kernel_name[grid_size](...) "
                    "to actually run the kernel."
                ),
                "code_snippet": "",
            })

        # C. torch compute APIs in forward()
        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return errors
        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return errors

        hard_calls: List[tuple] = []
        soft_calls: List[tuple] = []
        for node in ast.walk(forward_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                mod = node.func.value
                method = node.func.attr
                if isinstance(mod, ast.Name) and mod.id in _TORCH_CALL_PREFIXES:
                    label = f"{mod.id}.{method}"
                    if method in _TORCH_OPS_HARD:
                        hard_calls.append((node.lineno, label))
                    elif method in _TORCH_OPS_SOFT:
                        soft_calls.append((node.lineno, label))
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                hard_calls.append((node.lineno, "@ (matmul operator)"))

        def _fmt(calls: List[tuple], limit: int = 5) -> str:
            summary = ", ".join(f"{name}(line {line})" for line, name in calls[:limit])
            if len(calls) > limit:
                summary += f", ... ({len(calls)} total)"
            return summary

        if hard_calls:
            errors.append({
                "line": hard_calls[0][0],
                "error_type": "torch_api_instead_of_kernel",
                "detail": (
                    f"forward() uses {len(hard_calls)} forbidden torch high-level compute API(s): "
                    f"{_fmt(hard_calls)}. Matrix multiply, conv, normalization, pooling, etc. "
                    f"must be implemented inside the triton kernel."
                ),
                "suggestion": (
                    "Move the flagged operations into the @triton.jit kernel. "
                    "forward() should only prepare inputs, launch the kernel, and return outputs."
                ),
                "code_snippet": "",
            })

        if soft_calls and kernels_not_launched:
            errors.append({
                "line": soft_calls[0][0],
                "error_type": "torch_api_without_kernel",
                "detail": (
                    f"forward() uses {len(soft_calls)} torch compute API(s) without launching any "
                    f"triton kernel: {_fmt(soft_calls)}. The kernel is decorative only."
                ),
                "suggestion": (
                    "Ensure the triton kernel is launched. Simple ops (exp/relu/sum) may "
                    "remain as kernel pre/post-processing, but only if the kernel carries the "
                    "main compute."
                ),
                "code_snippet": "",
            })

        return errors

    # ------------------------------------------------------------------
    # Step 6: @triton.autotune compliance
    # ------------------------------------------------------------------

    def _check_autotune_compliance(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        m = _AUTOTUNE_RE.search(code)
        if not m:
            return errors
        autotune_line = code[: m.start()].count("\n") + 1

        # Find matching close paren
        paren_depth = 0
        start = m.end() - 1
        end = start
        for i in range(start, len(code)):
            if code[i] == "(":
                paren_depth += 1
            elif code[i] == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    end = i + 1
                    break
        block = code[start:end]

        if not _RESTORE_VALUE_RE.search(block):
            errors.append({
                "line": autotune_line,
                "error_type": "autotune_missing_restore_value",
                "detail": (
                    "@triton.autotune is missing `restore_value=`. Autotune re-runs the kernel "
                    "for each config and output buffers pollute across runs — verification will fail."
                ),
                "suggestion": (
                    "Add `restore_value=['<output_ptr_name>', ...]` listing every output pointer "
                    "argument. Example:\n"
                    "  @triton.autotune(\n"
                    "      configs=[...],\n"
                    "      key=[...],\n"
                    "      restore_value=['output_ptr'],\n"
                    "  )"
                ),
                "code_snippet": "",
            })

        return errors

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_errors(self, errors: List[Dict], code_lines: Optional[List[str]] = None) -> str:
        if not errors:
            return ""
        out = [
            "## CodeChecker static report",
            "",
            f"**Found {len(errors)} issue(s); fix before re-running eval:**",
            "",
        ]
        for i, err in enumerate(errors, 1):
            ln = err["line"]
            out.append(f"### Issue {i}: line {ln} [{err.get('error_type', 'unknown')}]")
            out.append(f"  {err['detail']}")
            if code_lines is not None and ln > 0:
                start = max(1, ln - 3)
                end = min(len(code_lines), ln + 3)
                out.append(f"  Context (lines {start}-{end}):")
                for n in range(start, end + 1):
                    pointer = ">>> " if n == ln else "    "
                    out.append(f"  {pointer}{n:4d} | {code_lines[n - 1]}")
            elif err.get("code_snippet"):
                out.append(f"  Source: {err['code_snippet']}")
            if err.get("suggestion"):
                out.append("  Suggestion:")
                for line in err["suggestion"].strip().split("\n"):
                    out.append(f"    {line}")
            out.append("")
        out.append("**Note: syntax checks stop at the first error; fix then re-check for additional issues.**")
        return "\n".join(out)

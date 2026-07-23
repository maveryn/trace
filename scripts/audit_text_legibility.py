"""Audit text-legibility invariants that can be checked without rendering."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SCAN_ROOTS = (
    "src/trace_tasks/resources/configs",
    "docs/domains",
    "docs/tasks",
    "src/trace_tasks/resources/prompts",
    "src/trace_tasks/tasks",
)

SEMANTIC_TEXT_COLOR_PATTERNS = (
    re.compile(r"\btext\s+color\s+indicates\b", re.IGNORECASE),
    re.compile(r"\bword\s+color\s+indicates\b", re.IGNORECASE),
    re.compile(r"\blabel\s+color\s+indicates\b", re.IGNORECASE),
    re.compile(r"\bcolor\s+of\s+(?:the\s+)?(?:text|word|label|letter)\b", re.IGNORECASE),
    re.compile(r"\b(?:text|word|label|letter)\s+is\s+colored\b", re.IGNORECASE),
)

TEXT_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".md", ".txt"}
PYTHON_EXTENSIONS = {".py"}
EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "benchmark",
    "external",
    "logs",
    "review",
    "runs",
    "samples",
}

RENDER_TEXT_CALL_PATTERNS = (
    re.compile(r"\bdraw\.text\("),
)
TRACED_TEXT_FUNCTION_NAMES = {
    "draw_text_traced",
    "draw_traced_text",
    "draw_centered_traced_text",
}

RENDER_TEXT_ALLOWLIST_PATHS = {
    "src/trace_tasks/tasks/shared/text_legibility.py",
    "src/trace_tasks/tasks/shared/text_rendering.py",
    "src/trace_tasks/tasks/shared/drawing.py",
}
DIRECT_FONT_ALLOWLIST_PATHS = {
    "src/trace_tasks/tasks/shared/text_rendering.py",
}

READOUT_CONTEXT_FONT_NAME_PATTERNS = (
    re.compile(r"(?:^|_)context(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)decorative(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)chrome(?:_|$)", re.IGNORECASE),
)
NON_READOUT_REQUIRED_FONT_NAME_PATTERNS = (
    re.compile(r"(?:^|_)(?:axis|tick|label|value|measurement|readout|option|rank|card|node)(?:_|$)", re.IGNORECASE),
)


def _iter_files(root: Path, scan_roots: Iterable[str]) -> Iterable[Path]:
    for relative in scan_roots:
        base = root / relative
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix.lower() in TEXT_EXTENSIONS:
                yield base
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
                continue
            yield path


def semantic_text_color_findings(root: Path, scan_roots: Iterable[str] = DEFAULT_SCAN_ROOTS) -> list[str]:
    """Return findings where glyph text color is described as semantic."""

    findings: list[str] = []
    for path in _iter_files(root, scan_roots):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in SEMANTIC_TEXT_COLOR_PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(root)
                    findings.append(f"{rel}:{line_no}: {line.strip()}")
                    break
    return findings


def renderer_text_migration_findings(
    root: Path,
    scan_roots: Iterable[str] = ("src/trace_tasks/tasks",),
) -> list[str]:
    """Return direct task-renderer text calls not yet routed through legibility helpers.

    This is an explicit migration audit. It is stricter than the default
    semantic-color audit because many legacy renderers still draw text directly;
    use --strict-renderer-migration when auditing a domain or scene migration.
    """

    findings: list[str] = []
    for path in _iter_files(root, scan_roots):
        if path.suffix.lower() not in PYTHON_EXTENSIONS:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in RENDER_TEXT_ALLOWLIST_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                continue
            if "draw_readable_text(" in stripped or "draw_centered_readable_text(" in stripped:
                continue
            for pattern in RENDER_TEXT_CALL_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{line_no}: {stripped}")
                    break
    return findings


def renderer_role_metadata_findings(
    root: Path,
    scan_roots: Iterable[str] = ("src/trace_tasks/tasks",),
) -> list[str]:
    """Return traced text calls that do not explicitly declare role/required."""

    findings: list[str] = []
    for path in _iter_files(root, scan_roots):
        if path.suffix.lower() not in PYTHON_EXTENSIONS:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in RENDER_TEXT_ALLOWLIST_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                func_name = str(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                func_name = str(node.func.attr)
            else:
                continue
            if func_name not in TRACED_TEXT_FUNCTION_NAMES:
                continue
            keyword_names = {str(keyword.arg) for keyword in node.keywords if keyword.arg}
            missing = sorted({"role", "required"}.difference(keyword_names))
            if missing:
                findings.append(
                    f"{rel}:{int(getattr(node, 'lineno', 0))}: "
                    f"{func_name} missing explicit {', '.join(missing)}"
                )
    return findings


def font_routing_findings(
    root: Path,
    scan_roots: Iterable[str] = ("src/trace_tasks/tasks",),
) -> list[str]:
    """Return font routing findings for role-aware task renderers."""

    findings: list[str] = []
    for path in _iter_files(root, scan_roots):
        if path.suffix.lower() not in PYTHON_EXTENSIONS:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                func_name = str(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                func_name = str(node.func.attr)
            else:
                continue
            if func_name == "sample_font_family":
                keywords = {str(keyword.arg): keyword.value for keyword in node.keywords if keyword.arg}
                role_value = keywords.get("role")
                if not isinstance(role_value, ast.Constant) or str(role_value.value) not in {
                    "readout",
                    "context",
                    "decorative",
                }:
                    findings.append(f"{rel}:{int(getattr(node, 'lineno', 0))}: sample_font_family missing explicit valid role")
                    continue
                role = str(role_value.value)
                sampled_names: list[str] = []
                for key in ("namespace", "explicit_key", "weights_key"):
                    value = keywords.get(key)
                    if isinstance(value, ast.Constant):
                        sampled_names.append(str(value.value))
                    elif isinstance(value, ast.JoinedStr):
                        sampled_names.extend(
                            str(part.value)
                            for part in value.values
                            if isinstance(part, ast.Constant)
                        )
                name_blob = " ".join(sampled_names)
                if role == "readout" and any(pattern.search(name_blob) for pattern in READOUT_CONTEXT_FONT_NAME_PATTERNS):
                    findings.append(
                        f"{rel}:{int(getattr(node, 'lineno', 0))}: readout font role looks like context/decorative text"
                    )
                if role in {"context", "decorative"} and any(
                    pattern.search(name_blob) for pattern in NON_READOUT_REQUIRED_FONT_NAME_PATTERNS
                ):
                    findings.append(
                        f"{rel}:{int(getattr(node, 'lineno', 0))}: {role} font role looks like required/readout text"
                    )
                continue
            if func_name in {"truetype", "load_default"} and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if isinstance(receiver, ast.Name) and receiver.id == "ImageFont" and rel not in DIRECT_FONT_ALLOWLIST_PATHS:
                    findings.append(
                        f"{rel}:{int(getattr(node, 'lineno', 0))}: direct ImageFont.{func_name} bypasses shared font routing"
                    )
    return findings


def _iter_text_legibility_blocks(value: Any, *, field_path: str = "render_spec") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        return
    block = value.get("text_legibility")
    if isinstance(block, Mapping):
        yield f"{field_path}.text_legibility", block
    for key, child in value.items():
        if key == "text_legibility":
            continue
        if isinstance(child, Mapping):
            yield from _iter_text_legibility_blocks(child, field_path=f"{field_path}.{key}")
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    yield from _iter_text_legibility_blocks(item, field_path=f"{field_path}.{key}[{index}]")


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def runtime_text_coverage_findings(
    *,
    sample_count: int,
    max_attempts: int,
    domain: str | None = None,
    require_required_roles: bool = False,
    fail_unvalidated_required_draws: bool = False,
    fail_generation_errors: bool = False,
) -> list[str]:
    """Generate small samples and return text-legibility coverage findings."""

    from trace_tasks.core.seed import hash64
    from trace_tasks.tasks import create_task
    from trace_tasks.tasks.registry import list_default_task_ids

    findings: list[str] = []
    task_ids = [
        task_id
        for task_id in list_default_task_ids()
        if domain is None or task_id.startswith(f"task_{domain}__")
    ]
    for task_id in task_ids:
        task = create_task(task_id)
        generated = False
        last_error: Exception | None = None
        for sample_index in range(max(1, int(sample_count))):
            instance_seed = int(hash64(0, f"{task_id}:text_legibility_runtime", sample_index))
            try:
                output = task.generate(instance_seed, params={}, max_attempts=max(1, int(max_attempts)))
            except Exception as exc:  # pragma: no cover - audit reporting path.
                last_error = exc
                continue
            generated = True
            render_spec = output.trace_payload.get("render_spec", {}) if isinstance(output.trace_payload, Mapping) else {}
            blocks = list(_iter_text_legibility_blocks(render_spec))
            failure_count = sum(_to_int(block.get("failure_count")) for _, block in blocks)
            drawn_count = sum(_to_int(block.get("drawn_text_record_count")) for _, block in blocks)
            required_count = sum(_to_int(block.get("required_role_count")) for _, block in blocks)
            unvalidated_required = 0
            for _, block in blocks:
                records = block.get("records")
                if not isinstance(records, list):
                    continue
                for record in records:
                    if isinstance(record, Mapping) and bool(record.get("declared_required_without_contrast_metadata", False)):
                        unvalidated_required += 1
            if failure_count > 0:
                findings.append(
                    f"{task_id}: sample {sample_index} has text_legibility failure_count={failure_count}"
                )
            if fail_unvalidated_required_draws and unvalidated_required > 0:
                findings.append(
                    f"{task_id}: sample {sample_index} has {unvalidated_required} drawn required text record(s) "
                    "without contrast metadata"
                )
            if require_required_roles and drawn_count > 0 and required_count <= 0:
                findings.append(
                    f"{task_id}: sample {sample_index} draws text but records no required/read-off text roles"
                )
        if not generated and fail_generation_errors:
            findings.append(f"{task_id}: failed to generate runtime text-legibility sample: {last_error}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="relative path to scan; may be passed more than once",
    )
    parser.add_argument(
        "--strict-renderer-migration",
        action="store_true",
        help="also fail on direct renderer text calls not routed through shared legibility helpers",
    )
    parser.add_argument(
        "--strict-role-metadata",
        action="store_true",
        help="also fail on traced text helper calls missing explicit role= and required= keywords",
    )
    parser.add_argument(
        "--strict-font-routing",
        action="store_true",
        help="also fail on font sampling/direct font-loading that bypasses role-aware routing",
    )
    parser.add_argument(
        "--runtime-coverage",
        action="store_true",
        help="generate active task samples and fail on malformed/failing text-legibility metadata",
    )
    parser.add_argument(
        "--runtime-domain",
        default="",
        help="optional domain name for --runtime-coverage, e.g. graph",
    )
    parser.add_argument(
        "--runtime-sample-count",
        type=int,
        default=1,
        help="number of seed attempts per task for --runtime-coverage",
    )
    parser.add_argument(
        "--runtime-max-attempts",
        type=int,
        default=120,
        help="max_attempts passed to task.generate for --runtime-coverage",
    )
    parser.add_argument(
        "--require-required-roles",
        action="store_true",
        help="with --runtime-coverage, fail when generated text has no required/read-off role metadata",
    )
    parser.add_argument(
        "--fail-unvalidated-required-draws",
        action="store_true",
        help="with --runtime-coverage, fail when a drawn record declared required=True without contrast metadata",
    )
    parser.add_argument(
        "--fail-generation-errors",
        action="store_true",
        help="with --runtime-coverage, fail when no sample can be generated for a task",
    )
    parser.add_argument(
        "--max-renderer-findings",
        type=int,
        default=80,
        help="maximum strict renderer findings to print",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    findings = semantic_text_color_findings(root, tuple(args.scan_roots or DEFAULT_SCAN_ROOTS))
    if findings:
        print("Semantic glyph text-color usage is not allowed for active Trace tasks.", file=sys.stderr)
        print("Use nonsemantic readable text ink and encode categories with marks/swatches instead.", file=sys.stderr)
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    if bool(args.strict_renderer_migration):
        renderer_findings = renderer_text_migration_findings(
            root,
            tuple(args.scan_roots or ("src/trace_tasks/tasks",)),
        )
        if renderer_findings:
            print("Direct renderer text calls still need migration to shared text-legibility helpers.", file=sys.stderr)
            for finding in renderer_findings[: max(0, int(args.max_renderer_findings))]:
                print(finding, file=sys.stderr)
            remaining = len(renderer_findings) - max(0, int(args.max_renderer_findings))
            if remaining > 0:
                print(f"... {remaining} more findings omitted", file=sys.stderr)
            return 1
        print("text-legibility strict renderer migration audit passed")
    if bool(args.strict_role_metadata):
        role_findings = renderer_role_metadata_findings(
            root,
            tuple(args.scan_roots or ("src/trace_tasks/tasks",)),
        )
        if role_findings:
            print("Traced text calls need explicit role= and required= metadata.", file=sys.stderr)
            for finding in role_findings[: max(0, int(args.max_renderer_findings))]:
                print(finding, file=sys.stderr)
            remaining = len(role_findings) - max(0, int(args.max_renderer_findings))
            if remaining > 0:
                print(f"... {remaining} more findings omitted", file=sys.stderr)
            return 1
        print("text-legibility strict role metadata audit passed")
    if bool(args.strict_font_routing):
        routing_findings = font_routing_findings(
            root,
            tuple(args.scan_roots or ("src/trace_tasks/tasks",)),
        )
        if routing_findings:
            print("Font routing still needs role-aware migration.", file=sys.stderr)
            for finding in routing_findings[: max(0, int(args.max_renderer_findings))]:
                print(finding, file=sys.stderr)
            remaining = len(routing_findings) - max(0, int(args.max_renderer_findings))
            if remaining > 0:
                print(f"... {remaining} more findings omitted", file=sys.stderr)
            return 1
        print("text-legibility strict font routing audit passed")
    if bool(args.runtime_coverage):
        runtime_findings = runtime_text_coverage_findings(
            sample_count=max(1, int(args.runtime_sample_count)),
            max_attempts=max(1, int(args.runtime_max_attempts)),
            domain=str(args.runtime_domain).strip() or None,
            require_required_roles=bool(args.require_required_roles),
            fail_unvalidated_required_draws=bool(args.fail_unvalidated_required_draws),
            fail_generation_errors=bool(args.fail_generation_errors),
        )
        if runtime_findings:
            print("Runtime text-legibility coverage audit found issues.", file=sys.stderr)
            for finding in runtime_findings[: max(0, int(args.max_renderer_findings))]:
                print(finding, file=sys.stderr)
            remaining = len(runtime_findings) - max(0, int(args.max_renderer_findings))
            if remaining > 0:
                print(f"... {remaining} more findings omitted", file=sys.stderr)
            return 1
        print("text-legibility runtime coverage audit passed")
    if (
        bool(args.strict_renderer_migration)
        or bool(args.strict_role_metadata)
        or bool(args.strict_font_routing)
        or bool(args.runtime_coverage)
    ):
        return 0
    print("text-legibility semantic color audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

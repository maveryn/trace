#!/usr/bin/env python3
"""Generate the public Trace task catalog from reviewed repository sources.

The catalog is derived from the active task registry, public taxonomy, task
documentation, and code-authoritative reasoning-operation declarations.  It
must not become a second hand-maintained list of task ids.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

# Make the source checkout importable without relying on an editable install or
# an ambient PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import trace_tasks.tasks  # noqa: F401 - make the complete registry discoverable.
from trace_tasks.core.reasoning_operations import (
    extract_program_contract_section,
    parse_reasoning_operations,
    program_contract_sha256,
)
from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.core.taxonomy import ACTIVE_DOMAINS, resolve_task_taxonomy
from trace_tasks.core.type_registry import load_type_registry
from trace_tasks.tasks.registry import (
    list_default_task_ids,
    list_task_ids,
    task_reasoning_operations,
)

INVENTORY_PATH = REPO_ROOT / "docs" / "ACTIVE_TASK_INVENTORY.md"
INDEX_PATH = REPO_ROOT / "docs" / "TASK_CATALOG.md"
CATALOG_DIR = REPO_ROOT / "docs" / "task_catalog"
CATALOG_JSON_PATH = CATALOG_DIR / "catalog.v1.json"
CATALOG_SCHEMA_VERSION = "trace_task_catalog_v1"
PUBLIC_SOURCE_BASE_URL = "https://github.com/maveryn/trace/blob/main"

_INVENTORY_TASK_RE = re.compile(
    r"^- `(task_[a-z0-9_]+__[a-z0-9_]+__[a-z0-9_]+)`$", re.MULTILINE
)
_PROGRAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Program:\s*`([^`]+)`", re.MULTILINE),
    re.compile(r"^-\s*`([^`]+)`", re.MULTILINE),
    re.compile(r"^\d+\.\s*Program schema:\s*`([^`]+)`", re.MULTILINE),
    re.compile(r"^Program schema:\s*`([^`]+)`", re.MULTILINE),
)

_ANSWER_ALIASES: dict[str, str] = {
    "count_integer": "integer",
    "integer_count": "integer",
    "integer_value": "integer",
    "label": "string",
    "label_string": "string",
    "numeric_integer": "integer",
    "numeric_value": "number",
    "one_based_line_number": "integer",
    "option_label": "option_letter",
    "probability_option_letter": "option_letter",
    "row_label": "string",
    "string_label": "string",
    "string_label_or_unanswerable": "string",
}


class CatalogError(RuntimeError):
    """Raised when registry, documentation, or generated catalog disagree."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _task_doc_path(task_id: str) -> Path:
    parts = parse_public_task_id(task_id)
    return (
        REPO_ROOT / "docs" / "tasks" / parts.domain / parts.scene_id / f"{task_id}.md"
    )


def _task_source_path(task_id: str) -> Path:
    parts = parse_public_task_id(task_id)
    return (
        REPO_ROOT
        / "src"
        / "trace_tasks"
        / "tasks"
        / parts.domain
        / parts.scene_id
        / f"{parts.objective_contract}.py"
    )


def _first_match(text: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            return match.group(1).strip()
    return None


def _first_registered_token(value: str, allowed: set[str]) -> str | None:
    for token in re.findall(r"[a-z][a-z0-9_]*", value.lower()):
        if token in allowed:
            return token
    return None


def _normalize_answer_type(schema: str, allowed: set[str]) -> str | None:
    normalized = schema.lower().strip().rstrip(".")
    if normalized in allowed:
        return normalized
    alias = _ANSWER_ALIASES.get(normalized)
    if alias in allowed:
        return alias
    return _first_registered_token(normalized, allowed)


def _extract_answer_metadata(
    doc_text: str,
    program_section: str,
    *,
    allowed: set[str],
) -> tuple[str, str]:
    schema = _first_match(
        doc_text,
        (
            r"Generator `answer_gt\.type`:\s*`([^`]+)`",
            r"`answer_gt\.type`\s*:\s*`([^`]+)`",
            r"`answer_gt\.type\s*=\s*([a-z][a-z0-9_]*)`",
            r"answer_gt\.type\s*:\s*`([^`]+)`",
            r"Answer (?:schema|type):\s*`([^`]+)`",
            r"answer`? uses the `([^`]+)` schema",
        ),
    )
    if schema is None:
        schema = _first_match(
            program_section,
            (
                r"output(?:_role)?\s*=\s*([a-z][a-z0-9_]*)",
                r"bound by `([^`]+)`",
            ),
        )
    if schema is None:
        raise CatalogError("task document does not expose an answer schema")
    answer_type = _normalize_answer_type(schema, allowed)
    if answer_type is None:
        raise CatalogError(f"unrecognized answer schema {schema!r}")
    return schema, answer_type


def _extract_annotation_metadata(
    doc_text: str,
    program_section: str,
    *,
    allowed: set[str],
) -> tuple[str, str]:
    raw = _first_match(
        doc_text,
        (
            r"Generator `annotation_gt\.type`:\s*`([^`]+)`",
            r"`annotation_gt\.type`\s*:\s*`([^`]+)`",
            r"`annotation_gt\.type\s*=\s*([a-z][a-z0-9_]*)`",
            r"annotation_gt\.type\s*:\s*`([^`]+)`",
            r"Annotation (?:schema|type):[^\n]*?`([^`]+)`",
            r"annotation`? uses the `([^`]+)` schema",
        ),
    )
    if raw is None:
        raw = _first_match(
            program_section,
            (
                r"annotation\s*=\s*([a-z][a-z0-9_]*)",
                r"Annotation schema:[^\n]*?`([^`]+)`",
                r"`([^`]+)` witnesses bound by",
            ),
        )
    if raw is None:
        raise CatalogError("task document does not expose an annotation schema")
    annotation_type = _first_registered_token(raw, allowed)
    if annotation_type is None:
        raise CatalogError(f"unrecognized annotation schema {raw!r}")
    return annotation_type, annotation_type


def _extract_program(program_section: str) -> str:
    for pattern in _PROGRAM_PATTERNS:
        match = pattern.search(program_section)
        if match is not None:
            return match.group(1).strip()
    raise CatalogError("Program Contract has no machine-readable program expression")


def _inventory_task_ids() -> list[str]:
    if not INVENTORY_PATH.is_file():
        raise CatalogError(f"missing {_repo_path(INVENTORY_PATH)}")
    return _INVENTORY_TASK_RE.findall(INVENTORY_PATH.read_text(encoding="utf-8"))


def collect_catalog() -> dict[str, Any]:
    """Collect and cross-check the complete public catalog."""

    default_task_ids = list_default_task_ids()
    registered_task_ids = list_task_ids()
    inventory_task_ids = _inventory_task_ids()
    if len(default_task_ids) != len(set(default_task_ids)):
        raise CatalogError("default task registry contains duplicate ids")
    if inventory_task_ids != default_task_ids:
        missing = sorted(set(default_task_ids) - set(inventory_task_ids))
        extra = sorted(set(inventory_task_ids) - set(default_task_ids))
        raise CatalogError(
            "active inventory disagrees with default registry "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    if registered_task_ids != default_task_ids:
        non_default = sorted(set(registered_task_ids) - set(default_task_ids))
        missing_registered = sorted(set(default_task_ids) - set(registered_task_ids))
        raise CatalogError(
            "registered and default task surfaces disagree "
            f"(non_default={non_default[:5]}, missing_registered={missing_registered[:5]})"
        )

    type_registry = load_type_registry()
    records: list[dict[str, Any]] = []
    by_domain_scene: dict[str, defaultdict[str, list[str]]] = {
        domain: defaultdict(list) for domain in ACTIVE_DOMAINS
    }

    for task_id in default_task_ids:
        parts = parse_public_task_id(task_id)
        taxonomy = resolve_task_taxonomy(task_id)
        if (parts.domain, parts.scene_id) != (taxonomy.domain, taxonomy.scene_id):
            raise CatalogError(f"task-id and public taxonomy disagree for {task_id}")

        doc_path = _task_doc_path(task_id)
        source_path = _task_source_path(task_id)
        if not doc_path.is_file():
            raise CatalogError(f"missing task documentation: {_repo_path(doc_path)}")
        if not source_path.is_file():
            raise CatalogError(
                f"missing task implementation: {_repo_path(source_path)}"
            )

        doc_bytes = doc_path.read_bytes()
        doc_text = doc_bytes.decode("utf-8")
        if task_id not in doc_text.splitlines()[0]:
            raise CatalogError(f"task document title disagrees for {task_id}")
        program_section = extract_program_contract_section(doc_text)
        code_operations = task_reasoning_operations(task_id)
        documented_operations = parse_reasoning_operations(doc_text)
        if documented_operations != code_operations:
            raise CatalogError(f"reasoning operations disagree for {task_id}")

        try:
            answer_schema, answer_type = _extract_answer_metadata(
                doc_text,
                program_section,
                allowed=set(type_registry.answer_types),
            )
            annotation_schema, annotation_type = _extract_annotation_metadata(
                doc_text,
                program_section,
                allowed=set(type_registry.annotation_types),
            )
            program = _extract_program(program_section)
        except CatalogError as exc:
            raise CatalogError(f"{task_id}: {exc}") from exc

        records.append(
            {
                "annotation_schema": annotation_schema,
                "annotation_type": annotation_type,
                "answer_schema": answer_schema,
                "answer_type": answer_type,
                "doc_path": _repo_path(doc_path),
                "doc_sha256": _sha256_bytes(doc_bytes),
                "domain": taxonomy.domain,
                "objective_contract": parts.objective_contract,
                "program_contract": program,
                "program_contract_sha256": program_contract_sha256(doc_text),
                "reasoning_operations": list(code_operations),
                "scene_id": taxonomy.scene_id,
                "source_path": _repo_path(source_path),
                "source_sha256": _sha256_bytes(source_path.read_bytes()),
                "task_id": task_id,
            }
        )
        by_domain_scene[taxonomy.domain][taxonomy.scene_id].append(task_id)

    answer_counts = Counter(record["answer_type"] for record in records)
    annotation_counts = Counter(record["annotation_type"] for record in records)
    reasoning_counts: Counter[str] = Counter()
    for record in records:
        reasoning_counts.update(record["reasoning_operations"])

    domain_rows: list[dict[str, Any]] = []
    for domain in ACTIVE_DOMAINS:
        scenes = by_domain_scene[domain]
        domain_rows.append(
            {
                "domain": domain,
                "scene_count": len(scenes),
                "task_count": sum(len(task_ids) for task_ids in scenes.values()),
            }
        )

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "sources": {
            "active_inventory": _repo_path(INVENTORY_PATH),
            "default_registry": "trace_tasks.tasks.registry.list_default_task_ids",
            "generator": {
                "path": "scripts/generate_task_catalog.py",
                "sha256": _sha256_bytes(Path(__file__).resolve().read_bytes()),
            },
            "public_taxonomy": "src/trace_tasks/core/taxonomy.py",
            "reasoning_operations": "task class reasoning_operations tuples",
            "task_docs": "docs/tasks/<domain>/<scene_id>/<task_id>.md",
        },
        "summary": {
            "annotation_type_counts": dict(sorted(annotation_counts.items())),
            "answer_type_counts": dict(sorted(answer_counts.items())),
            "default_task_count": len(default_task_ids),
            "domain_count": len(ACTIVE_DOMAINS),
            "registered_task_count": len(registered_task_ids),
            "reasoning_operation_counts": dict(sorted(reasoning_counts.items())),
            "scene_count": sum(row["scene_count"] for row in domain_rows),
        },
        "domains": domain_rows,
        "tasks": records,
    }


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_index(catalog: Mapping[str, Any]) -> str:
    """Render the generated human-facing catalog index."""

    summary = catalog["summary"]
    domains = catalog["domains"]
    lines = [
        "# Trace Task Catalog",
        "",
        "Generated by `scripts/generate_task_catalog.py` from the public task registry, taxonomy, implementation files, and reviewed task contracts.",
        "Do not edit this page or its domain pages by hand.",
        "",
        "## Summary",
        "",
        f"- Active tasks: `{summary['default_task_count']}`",
        f"- Public domains: `{summary['domain_count']}`",
        f"- Public scenes: `{summary['scene_count']}`",
        "",
        "Each domain page links every active task to its reviewed contract and implementation. The machine-readable [`catalog.v1.json`](task_catalog/catalog.v1.json) additionally records answer and annotation schemas, normalized public types, program contracts, reasoning-operation families, and SHA-256 provenance.",
        "",
        "## Domains",
        "",
    ]
    lines.extend(
        _table(
            ["Domain", "Scenes", "Tasks"],
            (
                [
                    f"[{row['domain']}](task_catalog/{row['domain']}.md)",
                    row["scene_count"],
                    row["task_count"],
                ]
                for row in domains
            ),
        )
    )
    lines.extend(["", "## Public Type Coverage", "", "### Answers", ""])
    lines.extend(
        _table(
            ["Type", "Tasks"],
            (
                [f"`{key}`", value]
                for key, value in summary["answer_type_counts"].items()
            ),
        )
    )
    lines.extend(["", "### Annotations", ""])
    lines.extend(
        _table(
            ["Type", "Tasks"],
            (
                [f"`{key}`", value]
                for key, value in summary["annotation_type_counts"].items()
            ),
        )
    )
    lines.extend(["", "## Reasoning Coverage", ""])
    lines.extend(
        _table(
            ["Operation family", "Tasks"],
            (
                [f"`{key}`", value]
                for key, value in summary["reasoning_operation_counts"].items()
            ),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def render_domain_page(catalog: Mapping[str, Any], domain: str) -> str:
    """Render one domain's generated scene and task indexes."""

    records = [record for record in catalog["tasks"] if record["domain"] == domain]
    by_scene: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[record["scene_id"]].append(record)

    lines = [
        f"# {domain} tasks",
        "",
        "[Back to the task catalog](../TASK_CATALOG.md)",
        "",
        "Generated by `scripts/generate_task_catalog.py`; do not edit by hand.",
        "",
        f"This domain contains `{len(records)}` active tasks across `{len(by_scene)}` scenes.",
        "",
        "## Scenes",
        "",
    ]
    lines.extend(
        _table(
            ["Scene", "Tasks"],
            (
                [f"[`{scene}`](#{scene})", len(rows)]
                for scene, rows in sorted(by_scene.items())
            ),
        )
    )
    lines.append("")

    for scene, scene_records in sorted(by_scene.items()):
        lines.extend([f"## {scene}", ""])
        task_rows: list[list[str]] = []
        for record in sorted(scene_records, key=lambda value: value["task_id"]):
            doc_link = "../" + record["doc_path"].removeprefix("docs/")
            source_link = f"{PUBLIC_SOURCE_BASE_URL}/{record['source_path']}"
            operations = ", ".join(f"`{op}`" for op in record["reasoning_operations"])
            task_rows.append(
                [
                    f"[`{record['task_id']}`]({doc_link})",
                    f"`{record['objective_contract']}`",
                    f"`{record['answer_type']}`",
                    f"`{record['annotation_type']}`",
                    operations,
                    f"[source]({source_link})",
                ]
            )
        lines.extend(
            _table(
                ["Task", "Objective", "Answer", "Annotation", "Reasoning", "Code"],
                task_rows,
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_outputs(catalog: Mapping[str, Any]) -> dict[Path, bytes]:
    """Return every generated catalog output as deterministic bytes."""

    outputs: dict[Path, bytes] = {
        INDEX_PATH: render_index(catalog).encode("utf-8"),
        CATALOG_JSON_PATH: (
            json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    }
    for domain in ACTIVE_DOMAINS:
        outputs[CATALOG_DIR / f"{domain}.md"] = render_domain_page(
            catalog, domain
        ).encode("utf-8")
    return outputs


def check_outputs(outputs: Mapping[Path, bytes]) -> list[str]:
    """Return descriptions of stale or unexpected generated outputs."""

    problems: list[str] = []
    for path, expected in outputs.items():
        actual = path.read_bytes() if path.is_file() else b""
        if actual != expected:
            problems.append(f"stale or missing: {_repo_path(path)}")

    expected_catalog_files = set(outputs) - {INDEX_PATH}
    if CATALOG_DIR.is_dir():
        unexpected = sorted(
            path for path in CATALOG_DIR.iterdir() if path not in expected_catalog_files
        )
        problems.extend(
            f"unexpected catalog path: {_repo_path(path)}" for path in unexpected
        )
    return problems


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the public Trace task catalog"
    )
    parser.add_argument(
        "--check", action="store_true", help="Fail when generated outputs are stale"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        outputs = render_outputs(collect_catalog())
    except (CatalogError, ValueError) as exc:
        print(f"task catalog generation failed: {exc}", file=sys.stderr)
        return 1

    if args.check:
        problems = check_outputs(outputs)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print(
                "catalog is stale; run `python scripts/generate_task_catalog.py`",
                file=sys.stderr,
            )
            return 1
        print("task catalog is up to date")
        return 0

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"wrote {_repo_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

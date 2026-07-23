"""Canonical task-level reasoning-operation metadata.

Reasoning operations are analysis metadata derived from reviewed task contracts.
They are not public taxonomy nodes and do not affect task sampling.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any


REASONING_OPERATION_SCHEMA_VERSION = "trace_reasoning_operations_v2"
REASONING_OPERATION_KEYS: tuple[str, ...] = (
    "direct_retrieval",
    "filtering",
    "counting",
    "comparison",
    "ranking",
    "aggregation",
    "logical_composition",
    "spatial_relations",
    "topology",
    "transformation",
    "state_update",
    "formula_evaluation",
    "matching",
)

REASONING_OPERATION_LABELS: dict[str, str] = {
    "direct_retrieval": "Direct retrieval",
    "filtering": "Filtering",
    "counting": "Counting",
    "comparison": "Comparison",
    "ranking": "Ranking",
    "aggregation": "Aggregation",
    "logical_composition": "Logical composition",
    "spatial_relations": "Spatial relations",
    "topology": "Topology",
    "transformation": "Transformation",
    "state_update": "State update",
    "formula_evaluation": "Formula evaluation",
    "matching": "Matching",
}

_REASONING_OPERATIONS_SECTION_RE = re.compile(
    r"^## Reasoning Operations\s*$\n+(.*?)(?=^## |\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_PROGRAM_CONTRACT_SECTION_RE = re.compile(
    r"^## Program Contract\s*$\n+(.*?)(?=^## |\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_FAMILIES_LINE_RE = re.compile(r"^Families:\s*(.*?)\s*$", flags=re.MULTILINE)


def canonicalize_reasoning_operations(operations: Iterable[str]) -> tuple[str, ...]:
    """Validate and return operation keys in canonical display order."""

    supplied = tuple(operations)
    unknown = sorted(set(supplied) - set(REASONING_OPERATION_KEYS))
    if unknown:
        raise ValueError(f"unknown reasoning operations: {unknown}")
    if len(set(supplied)) != len(supplied):
        raise ValueError("reasoning operations must not contain duplicates")
    if not supplied:
        raise ValueError("at least one reasoning operation is required")
    if "direct_retrieval" in supplied and len(supplied) != 1:
        raise ValueError("direct_retrieval is an exclusive fallback operation")
    selected = set(supplied)
    return tuple(key for key in REASONING_OPERATION_KEYS if key in selected)


def validate_task_reasoning_operations(
    raw: Any,
    *,
    task_id: str,
) -> tuple[str, ...]:
    """Validate one code-authoritative public-task operation declaration."""

    if not isinstance(raw, tuple):
        raise ValueError(
            f"task {task_id!r} must declare reasoning_operations as a literal tuple"
        )
    if not all(isinstance(value, str) for value in raw):
        raise ValueError(
            f"task {task_id!r} reasoning_operations must contain only strings"
        )
    try:
        canonical = canonicalize_reasoning_operations(raw)
    except ValueError as exc:
        raise ValueError(f"task {task_id!r} has invalid reasoning_operations: {exc}") from exc
    if raw != canonical:
        raise ValueError(
            f"task {task_id!r} reasoning_operations must use canonical order: {canonical!r}"
        )
    return canonical


def parse_reasoning_operations(task_doc_text: str) -> tuple[str, ...]:
    """Read and validate the machine-readable operation block in a task doc."""

    section_match = _REASONING_OPERATIONS_SECTION_RE.search(task_doc_text)
    if section_match is None:
        raise ValueError("missing '## Reasoning Operations' section")
    families_match = _FAMILIES_LINE_RE.search(section_match.group(1))
    if families_match is None:
        raise ValueError("reasoning-operations section must contain a Families line")
    raw = families_match.group(1).strip()
    tokens = re.findall(r"`([a-z_]+)`", raw)
    expected_raw = ", ".join(f"`{token}`" for token in tokens)
    if not tokens or raw != expected_raw:
        raise ValueError(
            "Families must be a comma-separated list of backticked operation keys"
        )
    canonical = canonicalize_reasoning_operations(tokens)
    if tuple(tokens) != canonical:
        raise ValueError("reasoning operations are not in canonical order")
    return canonical


def format_reasoning_operations_section(operations: Iterable[str]) -> str:
    """Render the canonical task-doc section for operation metadata."""

    canonical = canonicalize_reasoning_operations(operations)
    families = ", ".join(f"`{key}`" for key in canonical)
    return f"## Reasoning Operations\n\nFamilies: {families}\n"


def extract_program_contract_section(task_doc_text: str) -> str:
    """Return the normalized Program Contract body used for provenance hashing."""

    match = _PROGRAM_CONTRACT_SECTION_RE.search(task_doc_text)
    if match is None:
        raise ValueError("missing '## Program Contract' section")
    lines = [line.rstrip() for line in match.group(1).strip().splitlines()]
    return "\n".join(lines)


def program_contract_sha256(task_doc_text: str) -> str:
    """Hash the reviewed Program Contract body for generated-artifact provenance."""

    body = extract_program_contract_section(task_doc_text)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()

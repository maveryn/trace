from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from trace_tasks.core.reasoning_operations import (
    REASONING_OPERATION_KEYS,
    parse_reasoning_operations,
    program_contract_sha256,
)
from trace_tasks.tasks.registry import task_reasoning_operations


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "ACTIVE_TASK_INVENTORY.md"


def _active_task_ids() -> list[str]:
    text = INVENTORY.read_text(encoding="utf-8")
    return re.findall(r"^- `(task_[a-z0-9_]+__[^`]+)`$", text, re.MULTILINE)


def _task_doc(task_id: str) -> Path:
    task_prefix, scene_id, _ = task_id.split("__", maxsplit=2)
    domain = task_prefix.removeprefix("task_")
    return ROOT / "docs" / "tasks" / domain / scene_id / f"{task_id}.md"


def _task_source(task_id: str) -> Path:
    task_prefix, scene_id, objective = task_id.split("__", maxsplit=2)
    domain = task_prefix.removeprefix("task_")
    return ROOT / "trace" / "tasks" / domain / scene_id / f"{objective}.py"


def _literal_source_operations(task_id: str) -> tuple[str, ...]:
    path = _task_source(task_id)
    assert path.is_file(), task_id
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declarations: list[ast.expr] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "reasoning_operations"
                for target in statement.targets
            ):
                declarations.append(statement.value)
    assert len(declarations) == 1, (
        f"{task_id} must have exactly one class-level reasoning_operations declaration"
    )
    value = declarations[0]
    assert isinstance(value, ast.Tuple), (
        f"{task_id} reasoning_operations must be a literal tuple"
    )
    assert all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in value.elts
    ), f"{task_id} reasoning_operations must contain literal strings"
    return tuple(str(element.value) for element in value.elts)


def test_every_active_task_has_valid_code_authoritative_reasoning_operations() -> None:
    task_ids = _active_task_ids()
    assert len(task_ids) == 1000
    assert len(set(task_ids)) == 1000

    counts: Counter[str] = Counter()
    for task_id in task_ids:
        code_operations = task_reasoning_operations(task_id)
        assert _literal_source_operations(task_id) == code_operations
        assert code_operations
        assert set(code_operations) <= set(REASONING_OPERATION_KEYS)
        counts.update(code_operations)

    # Dropping a whole reviewed family requires an explicit schema review.
    assert set(counts) == set(REASONING_OPERATION_KEYS)


def test_task_docs_mirror_code_authoritative_reasoning_operations() -> None:
    for task_id in _active_task_ids():
        path = _task_doc(task_id)
        assert path.is_file(), task_id
        text = path.read_text(encoding="utf-8")
        assert parse_reasoning_operations(text) == task_reasoning_operations(task_id)
        assert len(program_contract_sha256(text)) == 64


def test_direct_retrieval_is_exclusive_in_every_task() -> None:
    for task_id in _active_task_ids():
        operations = task_reasoning_operations(task_id)
        if "direct_retrieval" in operations:
            assert operations == ("direct_retrieval",), task_id
